"""
Dynamic Batching for Enigma AI Engine

Efficient batching for inference server.

Features:
- Continuous batching
- Sequence padding optimization
- Request scheduling
- Timeout handling
- Priority queues

Usage:
    from enigma_engine.core.dynamic_batching import DynamicBatcher, BatchingConfig
    
    batcher = DynamicBatcher(model, config)
    
    # Submit requests
    future = batcher.submit("Generate some text")
    result = future.result()
"""

import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappush, heappop
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class BatchPolicy(Enum):
    """Batching policies."""
    STATIC = "static"  # Fixed batch size
    DYNAMIC = "dynamic"  # Adapt to queue
    CONTINUOUS = "continuous"  # Continuous batching


class Priority(Enum):
    """Request priorities."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class BatchingConfig:
    """Configuration for dynamic batching."""
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    max_total_tokens: int = 65536  # Max tokens in batch
    min_batch_interval: float = 0.01  # Minimum time between batches
    max_wait_time: float = 0.1  # Max wait before processing
    policy: BatchPolicy = BatchPolicy.DYNAMIC


@dataclass
class InferenceRequest:
    """A single inference request."""
    id: str
    input_ids: torch.Tensor
    max_new_tokens: int = 256
    temperature: float = 0.7
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


@dataclass
class InferenceResult:
    """Result of inference."""
    request_id: str
    output_ids: torch.Tensor
    text: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0
    success: bool = True
    error: Optional[str] = None


class RequestBatch:
    """A batch of requests for inference."""
    
    def __init__(self) -> None:
        self.requests: List[InferenceRequest] = []
        self.padded_input: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.position_ids: Optional[torch.Tensor] = None
        self.max_new_tokens: int = 0
    
    def add(self, request: InferenceRequest) -> bool:
        """Try to add request to batch."""
        self.requests.append(request)
        self.max_new_tokens = max(self.max_new_tokens, request.max_new_tokens)
        return True
    
    def size(self) -> int:
        """Get current batch size."""
        return len(self.requests)
    
    def total_tokens(self) -> int:
        """Get total tokens in batch."""
        return sum(req.input_ids.shape[-1] for req in self.requests)
    
    def max_length(self) -> int:
        """Get max sequence length."""
        if not self.requests:
            return 0
        return max(req.input_ids.shape[-1] for req in self.requests)
    
    def prepare(self, device: torch.device, pad_token_id: int = 0) -> None:
        """Prepare padded batch for inference."""
        if not self.requests:
            return
        
        batch_size = len(self.requests)
        max_len = self.max_length()
        
        # Pad sequences
        padded = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
            device=device
        )
        
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device
        )
        
        for i, req in enumerate(self.requests):
            seq_len = req.input_ids.shape[-1]
            padded[i, :seq_len] = req.input_ids.squeeze()
            attention_mask[i, :seq_len] = 1
        
        self.padded_input = padded
        self.attention_mask = attention_mask
        
        # Position IDs
        self.position_ids = attention_mask.cumsum(dim=-1) - 1
        self.position_ids.masked_fill_(attention_mask == 0, 0)


class DynamicBatcher:
    """Dynamic batcher for inference server."""
    
    def __init__(
        self,
        model,
        tokenizer=None,
        config: Optional[BatchingConfig] = None,
    ) -> None:
        """
        Initialize dynamic batcher.
        
        Args:
            model: Model for inference
            tokenizer: Tokenizer for encoding/decoding
            config: Batching configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BatchingConfig()
        
        # Request queue (priority queue)
        self._queue: List[Tuple[int, float, InferenceRequest]] = []
        self._queue_lock = Lock()
        
        # Results storage
        self._results: Dict[str, Future] = {}
        self._results_lock = Lock()
        
        # Worker thread
        self._running = False
        self._worker_thread: Optional[Thread] = None
        
        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_tokens = 0
        
        # Device
        self._device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        
        logger.info(
            f"DynamicBatcher initialized: "
            f"max_batch={self.config.max_batch_size}, "
            f"policy={self.config.policy.value}"
        )
    
    def start(self) -> None:
        """Start batch processing worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("Batch worker started")
    
    def stop(self) -> None:
        """Stop batch processing worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Batch worker stopped")
    
    def submit(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None
    ) -> Future:
        """
        Submit request for batched inference.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            priority: Request priority
            timeout: Request timeout
            
        Returns:
            Future that will contain InferenceResult
        """
        import uuid
        
        # Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt)
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids])
        else:
            input_ids = torch.tensor([[ord(c) for c in prompt]])
        
        request = InferenceRequest(
            id=str(uuid.uuid4()),
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            priority=priority,
            timeout=timeout
        )
        
        # Create future
        future = Future()
        
        with self._results_lock:
            self._results[request.id] = future
        
        # Add to queue
        with self._queue_lock:
            heappush(
                self._queue,
                (priority.value, request.created_at, request)
            )
        
        self._total_requests += 1
        
        # Start worker if not running
        if not self._running:
            self.start()
        
        return future
    
    def submit_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Future]:
        """Submit multiple requests."""
        return [self.submit(p, **kwargs) for p in prompts]
    
    def _worker(self) -> None:
        """Background worker for batch processing."""
        last_batch_time = 0
        
        while self._running:
            # Wait for minimum interval
            elapsed = time.time() - last_batch_time
            if elapsed < self.config.min_batch_interval:
                time.sleep(self.config.min_batch_interval - elapsed)
            
            # Collect batch
            batch = self._collect_batch()
            
            if batch.size() == 0:
                time.sleep(0.001)  # Avoid busy loop
                continue
            
            # Process batch
            try:
                results = self._process_batch(batch)
                
                # Distribute results
                for result in results:
                    with self._results_lock:
                        if result.request_id in self._results:
                            future = self._results.pop(result.request_id)
                            future.set_result(result)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                
                # Set error for all requests in batch
                for req in batch.requests:
                    with self._results_lock:
                        if req.id in self._results:
                            future = self._results.pop(req.id)
                            future.set_exception(e)
            
            last_batch_time = time.time()
            self._total_batches += 1
    
    def _collect_batch(self) -> RequestBatch:
        """Collect requests for a batch."""
        batch = RequestBatch()
        wait_start = time.time()
        
        while True:
            # Check queue
            with self._queue_lock:
                if not self._queue:
                    if batch.size() > 0:
                        break
                    return batch
                
                # Check batch limits
                if batch.size() >= self.config.max_batch_size:
                    break
                
                # Peek at next request
                _, _, request = self._queue[0]
                
                # Check timeout
                if request.timeout:
                    if time.time() - request.created_at > request.timeout:
                        heappop(self._queue)
                        self._timeout_request(request)
                        continue
                
                # Check token limit
                new_tokens = batch.total_tokens() + request.input_ids.shape[-1]
                if new_tokens > self.config.max_total_tokens:
                    break
                
                # Add to batch
                heappop(self._queue)
                batch.add(request)
            
            # Check wait time
            if time.time() - wait_start > self.config.max_wait_time:
                break
            
            # Policy-specific behavior
            if self.config.policy == BatchPolicy.STATIC:
                if batch.size() >= self.config.max_batch_size:
                    break
            elif self.config.policy == BatchPolicy.CONTINUOUS:
                # Don't wait for full batch
                if batch.size() > 0:
                    break
        
        return batch
    
    def _timeout_request(self, request: InferenceRequest) -> None:
        """Handle timed out request."""
        with self._results_lock:
            if request.id in self._results:
                future = self._results.pop(request.id)
                future.set_result(InferenceResult(
                    request_id=request.id,
                    output_ids=torch.tensor([]),
                    success=False,
                    error="Request timed out"
                ))
    
    def _process_batch(self, batch: RequestBatch) -> List[InferenceResult]:
        """Process a batch of requests."""
        start_time = time.time()
        
        # Prepare padded batch
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0) if self.tokenizer else 0
        batch.prepare(self._device, pad_id)
        
        results = []
        
        with torch.no_grad():
            try:
                # Run inference
                outputs = self._generate_batch(batch)
                
                # Split results
                for i, request in enumerate(batch.requests):
                    input_len = request.input_ids.shape[-1]
                    output_ids = outputs[i, input_len:]
                    
                    # Decode
                    if self.tokenizer:
                        text = self.tokenizer.decode(output_ids.tolist())
                    else:
                        text = ''.join(chr(t) for t in output_ids.tolist() if 0 < t < 128)
                    
                    results.append(InferenceResult(
                        request_id=request.id,
                        output_ids=output_ids,
                        text=text,
                        tokens_generated=len(output_ids),
                        latency_ms=(time.time() - request.created_at) * 1000
                    ))
                    
                    self._total_tokens += len(output_ids)
                    
            except Exception as e:
                logger.error(f"Generation error: {e}")
                for request in batch.requests:
                    results.append(InferenceResult(
                        request_id=request.id,
                        output_ids=torch.tensor([]),
                        success=False,
                        error=str(e),
                        latency_ms=(time.time() - request.created_at) * 1000
                    ))
        
        return results
    
    def _generate_batch(self, batch: RequestBatch) -> torch.Tensor:
        """Run generation on batch."""
        # Simple autoregressive generation
        input_ids = batch.padded_input.to(self._device)
        attention_mask = batch.attention_mask.to(self._device)
        
        generated = input_ids.clone()
        
        for _ in range(batch.max_new_tokens):
            # Forward pass
            outputs = self.model(
                generated,
                attention_mask=attention_mask if hasattr(self.model, 'forward') else None
            )
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Get next token logits
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            temps = torch.tensor(
                [req.temperature for req in batch.requests],
                device=self._device
            ).unsqueeze(-1)
            next_logits = next_logits / temps.clamp(min=0.01)
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=self._device)
            ], dim=-1)
            
            # Check for EOS
            if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id'):
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
        
        return generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        with self._queue_lock:
            queue_size = len(self._queue)
        
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "total_tokens": self._total_tokens,
            "queue_size": queue_size,
            "avg_batch_size": self._total_requests / max(1, self._total_batches),
            "is_running": self._running
        }


class ContinuousBatcher(DynamicBatcher):
    """Continuous batching with iteration-level scheduling."""
    
    def __init__(self, model, tokenizer=None, config=None) -> None:
        """Initialize with continuous batching policy."""
        config = config or BatchingConfig()
        config.policy = BatchPolicy.CONTINUOUS
        super().__init__(model, tokenizer, config)
        
        # Active generations
        self._active: Dict[str, Dict] = {}
        self._active_lock = Lock()
    
    def _worker(self) -> None:
        """Worker with continuous batching."""
        while self._running:
            # Add new requests to active set
            self._add_new_requests()
            
            if not self._active:
                time.sleep(0.001)
                continue
            
            # Run one iteration for all active requests
            try:
                self._iteration()
            except Exception as e:
                logger.error(f"Iteration error: {e}")
            
            # Check for completed requests
            self._check_completions()
    
    def _add_new_requests(self) -> None:
        """Add pending requests to active set."""
        with self._queue_lock:
            while self._queue:
                if len(self._active) >= self.config.max_batch_size:
                    break
                
                _, _, request = heappop(self._queue)
                
                with self._active_lock:
                    self._active[request.id] = {
                        "request": request,
                        "generated": request.input_ids.to(self._device),
                        "tokens_generated": 0,
                        "start_time": time.time()
                    }
    
    def _iteration(self) -> None:
        """Run one generation iteration."""
        with self._active_lock:
            if not self._active:
                return
            
            # Batch all active generations
            batch_ids = list(self._active.keys())
            sequences = [self._active[rid]["generated"] for rid in batch_ids]
            
            # Pad to same length
            max_len = max(s.shape[-1] for s in sequences)
            padded = torch.zeros((len(sequences), max_len), dtype=torch.long, device=self._device)
            
            for i, seq in enumerate(sequences):
                padded[i, :seq.shape[-1]] = seq.squeeze()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(padded)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Sample next tokens
                for i, rid in enumerate(batch_ids):
                    state = self._active[rid]
                    seq_len = state["generated"].shape[-1]
                    
                    temp = state["request"].temperature
                    next_logits = logits[i, seq_len - 1, :] / max(temp, 0.01)
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)
                    
                    state["generated"] = torch.cat([state["generated"], next_token], dim=-1)
                    state["tokens_generated"] += 1
    
    def _check_completions(self) -> None:
        """Check for completed generations."""
        completed = []
        
        with self._active_lock:
            for rid, state in list(self._active.items()):
                request = state["request"]
                
                # Check max tokens
                if state["tokens_generated"] >= request.max_new_tokens:
                    completed.append(rid)
                    continue
                
                # Check EOS
                if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id'):
                    last_token = state["generated"][0, -1].item()
                    if last_token == self.tokenizer.eos_token_id:
                        completed.append(rid)
        
        # Complete requests
        for rid in completed:
            with self._active_lock:
                state = self._active.pop(rid)
            
            request = state["request"]
            output_ids = state["generated"][0, request.input_ids.shape[-1]:]
            
            if self.tokenizer:
                text = self.tokenizer.decode(output_ids.tolist())
            else:
                text = ''.join(chr(t) for t in output_ids.tolist() if 0 < t < 128)
            
            result = InferenceResult(
                request_id=request.id,
                output_ids=output_ids,
                text=text,
                tokens_generated=state["tokens_generated"],
                latency_ms=(time.time() - state["start_time"]) * 1000
            )
            
            with self._results_lock:
                if request.id in self._results:
                    future = self._results.pop(request.id)
                    future.set_result(result)
