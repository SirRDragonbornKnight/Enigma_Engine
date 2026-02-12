"""
Batch Inference for Enigma AI Engine

Efficient batch processing for high throughput.

Features:
- Dynamic batching
- Priority queues
- Request coalescing
- Async processing
- Memory-efficient batching

Usage:
    from enigma_engine.core.batch_inference import BatchProcessor, get_batch_processor
    
    processor = get_batch_processor(model)
    
    # Submit requests
    future1 = processor.submit("What is 2+2?")
    future2 = processor.submit("Hello world")
    
    # Get results
    result1 = future1.result()
    result2 = future2.result()
    
    # Or batch process
    results = processor.process_batch(["prompt1", "prompt2", "prompt3"])
"""

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class Priority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class InferenceRequest:
    """A single inference request."""
    id: str
    prompt: str
    priority: Priority = Priority.NORMAL
    max_tokens: int = 256
    temperature: float = 1.0
    created_at: float = field(default_factory=time.time)
    
    # Internal
    future: Optional[Future] = field(default=None, repr=False)
    
    def __lt__(self, other: "InferenceRequest") -> bool:
        """Priority comparison for queue."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.created_at < other.created_at  # Earlier first


@dataclass
class InferenceResult:
    """Result of an inference request."""
    id: str
    text: str
    tokens_generated: int
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_requests: int = 0
    total_tokens: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0


class RequestBatcher:
    """Combines requests into batches."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,
        max_tokens_per_batch: int = 2048
    ) -> None:
        """
        Initialize batcher.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_wait_time: Maximum time to wait for batch
            max_tokens_per_batch: Maximum tokens per batch
        """
        self._max_batch_size = max_batch_size
        self._max_wait_time = max_wait_time
        self._max_tokens = max_tokens_per_batch
        
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._lock = threading.Lock()
    
    def add_request(self, request: InferenceRequest) -> None:
        """Add a request to the queue."""
        self._queue.put(request)
    
    def get_batch(self) -> List[InferenceRequest]:
        """
        Get a batch of requests.
        
        Waits up to max_wait_time or until max_batch_size reached.
        
        Returns:
            List of requests to process
        """
        batch = []
        total_tokens = 0
        start_time = time.time()
        
        while len(batch) < self._max_batch_size:
            remaining_time = self._max_wait_time - (time.time() - start_time)
            
            if remaining_time <= 0 and batch:
                break
            
            try:
                request = self._queue.get(timeout=max(0.001, remaining_time))
                
                # Check token budget
                if total_tokens + request.max_tokens > self._max_tokens and batch:
                    # Put request back
                    self._queue.put(request)
                    break
                
                batch.append(request)
                total_tokens += request.max_tokens
                
            except queue.Empty:
                break
        
        return batch
    
    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()


class TokenPadder:
    """Handles padding for batch processing."""
    
    def __init__(self, pad_token_id: int = 0) -> None:
        """Initialize padder."""
        self._pad_id = pad_token_id
    
    def pad_batch(
        self,
        sequences: List[List[int]],
        pad_to: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of token sequences
            pad_to: Target length (None = max in batch)
            
        Returns:
            Tuple of (padded_tensor, attention_mask)
        """
        if not sequences:
            return None, None
        
        max_len = pad_to or max(len(s) for s in sequences)
        
        padded = []
        masks = []
        
        for seq in sequences:
            padding_length = max_len - len(seq)
            padded_seq = seq + [self._pad_id] * padding_length
            mask = [1] * len(seq) + [0] * padding_length
            
            padded.append(padded_seq)
            masks.append(mask)
        
        if HAS_TORCH:
            return torch.tensor(padded), torch.tensor(masks)
        else:
            return padded, masks
    
    def unpad_batch(
        self,
        sequences: List[List[int]],
        original_lengths: List[int]
    ) -> List[List[int]]:
        """Remove padding from sequences."""
        return [seq[:length] for seq, length in zip(sequences, original_lengths)]


class BatchProcessor:
    """
    Efficient batch inference processor.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,
        num_workers: int = 1
    ) -> None:
        """
        Initialize batch processor.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            max_batch_size: Maximum batch size
            max_wait_time: Time to wait for batch
            num_workers: Number of processing threads
        """
        self._model = model
        self._tokenizer = tokenizer
        
        self._batcher = RequestBatcher(max_batch_size, max_wait_time)
        self._padder = TokenPadder(
            getattr(tokenizer, 'pad_token_id', 0)
        )
        
        # Statistics
        self._stats = BatchStats()
        self._stats_lock = threading.Lock()
        
        # Processing
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Request counter
        self._request_counter = 0
        self._counter_lock = threading.Lock()
    
    def start(self) -> None:
        """Start batch processing."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Batch processor started")
    
    def stop(self) -> None:
        """Stop batch processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        self._executor.shutdown(wait=True)
        logger.info("Batch processor stopped")
    
    def submit(
        self,
        prompt: str,
        priority: Priority = Priority.NORMAL,
        max_tokens: int = 256,
        temperature: float = 1.0
    ) -> Future:
        """
        Submit a request for processing.
        
        Args:
            prompt: Input prompt
            priority: Request priority
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Future for the result
        """
        with self._counter_lock:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}"
        
        future: Future = Future()
        
        request = InferenceRequest(
            id=request_id,
            prompt=prompt,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature,
            future=future
        )
        
        self._batcher.add_request(request)
        
        # Start processing if not running
        if not self._running:
            self.start()
        
        return future
    
    def process_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 1.0
    ) -> List[InferenceResult]:
        """
        Process a batch of prompts synchronously.
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            
        Returns:
            List of results
        """
        futures = [
            self.submit(prompt, max_tokens=max_tokens, temperature=temperature)
            for prompt in prompts
        ]
        
        return [f.result() for f in futures]
    
    async def submit_async(
        self,
        prompt: str,
        priority: Priority = Priority.NORMAL,
        max_tokens: int = 256
    ) -> InferenceResult:
        """
        Submit request asynchronously.
        
        Args:
            prompt: Input prompt
            priority: Request priority
            max_tokens: Maximum tokens
            
        Returns:
            Inference result
        """
        future = self.submit(prompt, priority, max_tokens)
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, future.result)
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                batch = self._batcher.get_batch()
                
                if batch:
                    self._process_batch_internal(batch)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _process_batch_internal(self, batch: List[InferenceRequest]) -> None:
        """Process a batch of requests."""
        start_time = time.time()
        
        # Tokenize inputs
        prompts = [r.prompt for r in batch]
        
        try:
            if hasattr(self._tokenizer, 'encode_batch'):
                input_ids = self._tokenizer.encode_batch(prompts)
            else:
                input_ids = [self._tokenizer.encode(p) for p in prompts]
        except Exception as e:
            # Return errors
            for req in batch:
                if req.future:
                    req.future.set_result(InferenceResult(
                        id=req.id,
                        text="",
                        tokens_generated=0,
                        latency_ms=0,
                        success=False,
                        error=f"Tokenization error: {e}"
                    ))
            return
        
        # Pad to same length
        original_lengths = [len(ids) for ids in input_ids]
        padded_input, attention_mask = self._padder.pad_batch(input_ids)
        
        # Run inference
        try:
            outputs = self._generate_batch(
                padded_input,
                attention_mask,
                [r.max_tokens for r in batch],
                [r.temperature for r in batch]
            )
        except Exception as e:
            # Return errors
            for req in batch:
                if req.future:
                    req.future.set_result(InferenceResult(
                        id=req.id,
                        text="",
                        tokens_generated=0,
                        latency_ms=0,
                        success=False,
                        error=f"Generation error: {e}"
                    ))
            return
        
        # Process outputs
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000 / len(batch)
        
        total_tokens = 0
        
        for i, (req, output) in enumerate(zip(batch, outputs)):
            # Decode output
            if hasattr(self._tokenizer, 'decode'):
                text = self._tokenizer.decode(output)
            else:
                text = str(output)
            
            tokens_generated = len(output) - original_lengths[i]
            total_tokens += tokens_generated
            
            result = InferenceResult(
                id=req.id,
                text=text,
                tokens_generated=tokens_generated,
                latency_ms=latency_ms,
                success=True
            )
            
            if req.future:
                req.future.set_result(result)
        
        # Update stats
        with self._stats_lock:
            self._stats.total_requests += len(batch)
            self._stats.total_tokens += total_tokens
            self._stats.total_batches += 1
            self._stats.avg_batch_size = (
                self._stats.total_requests / self._stats.total_batches
            )
            self._stats.avg_latency_ms = (
                (self._stats.avg_latency_ms * (self._stats.total_batches - 1) + latency_ms)
                / self._stats.total_batches
            )
            
            elapsed = end_time - start_time
            if elapsed > 0:
                self._stats.throughput_tokens_per_sec = total_tokens / elapsed
    
    def _generate_batch(
        self,
        input_ids: Any,
        attention_mask: Any,
        max_tokens: List[int],
        temperatures: List[float]
    ) -> List[List[int]]:
        """
        Generate outputs for a batch.
        
        Args:
            input_ids: Padded input tensor
            attention_mask: Attention mask
            max_tokens: Max tokens per request
            temperatures: Temperature per request
            
        Returns:
            List of output token sequences
        """
        max_new_tokens = max(max_tokens)
        
        # Try model generate
        if hasattr(self._model, 'generate'):
            if HAS_TORCH:
                with torch.no_grad():
                    device = next(self._model.parameters()).device
                    input_ids = input_ids.to(device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    outputs = self._model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=sum(temperatures) / len(temperatures),
                        pad_token_id=self._padder._pad_id
                    )
                    
                    return outputs.tolist()
            else:
                # Non-torch model
                outputs = []
                for i in range(len(input_ids)):
                    out = self._model.generate(input_ids[i], max_new_tokens=max_tokens[i])
                    outputs.append(out)
                return outputs
        
        # Fallback: forward pass
        if HAS_TORCH:
            outputs = []
            with torch.no_grad():
                for i in range(len(input_ids)):
                    current = input_ids[i:i+1]
                    
                    for _ in range(max_tokens[i]):
                        logits = self._model(current)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        
                        # Sample next token
                        next_logits = logits[:, -1, :] / temperatures[i]
                        probs = torch.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        
                        current = torch.cat([current, next_token], dim=1)
                    
                    outputs.append(current[0].tolist())
            
            return outputs
        
        return []
    
    def get_stats(self) -> BatchStats:
        """Get processing statistics."""
        with self._stats_lock:
            return BatchStats(
                total_requests=self._stats.total_requests,
                total_tokens=self._stats.total_tokens,
                total_batches=self._stats.total_batches,
                avg_batch_size=self._stats.avg_batch_size,
                avg_latency_ms=self._stats.avg_latency_ms,
                throughput_tokens_per_sec=self._stats.throughput_tokens_per_sec
            )
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._batcher.queue_size


# Global instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor(
    model: Any = None,
    tokenizer: Any = None,
    **kwargs
) -> Optional[BatchProcessor]:
    """Get or create global batch processor."""
    global _batch_processor
    
    if _batch_processor is None and model is not None:
        _batch_processor = BatchProcessor(model, tokenizer, **kwargs)
    
    return _batch_processor
