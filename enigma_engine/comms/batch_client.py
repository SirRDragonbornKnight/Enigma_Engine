"""
Batch API Client for Enigma AI Engine

Efficient batch processing for API calls.

Features:
- Batch multiple requests
- Rate limiting
- Cost tracking
- Async processing
- Result caching

Usage:
    from enigma_engine.comms.batch_client import BatchAPIClient, get_batch_client
    
    client = get_batch_client()
    client.set_api_key("sk-...")
    
    # Queue requests
    client.enqueue("What is Python?")
    client.enqueue("What is JavaScript?")
    
    # Process batch
    results = client.process_batch()
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class BatchRequest:
    """A single request in a batch."""
    id: str
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchResponse:
    """Response for a batch request."""
    request_id: str
    success: bool
    response: str = ""
    error: str = ""
    tokens_used: int = 0
    latency_ms: float = 0
    cached: bool = False


@dataclass
class BatchResult:
    """Result of a batch operation."""
    total_requests: int
    successful: int
    failed: int
    total_tokens: int
    total_cost: float
    responses: List[BatchResponse] = field(default_factory=list)
    processing_time: float = 0


@dataclass
class CostConfig:
    """API cost configuration."""
    input_cost_per_1k: float = 0.001  # $ per 1k input tokens
    output_cost_per_1k: float = 0.002  # $ per 1k output tokens
    batch_discount: float = 0.5  # 50% discount for batch API


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000
    ):
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        
        self._request_tokens = float(requests_per_minute)
        self._token_bucket = float(tokens_per_minute)
        
        self._last_update = time.time()
        self._lock = Lock()
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire rate limit permit.
        
        Returns:
            Wait time in seconds (0 if immediate)
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now
            
            # Refill buckets
            self._request_tokens = min(
                self._rpm,
                self._request_tokens + elapsed * (self._rpm / 60)
            )
            self._token_bucket = min(
                self._tpm,
                self._token_bucket + elapsed * (self._tpm / 60)
            )
            
            # Check limits
            wait_time = 0
            
            if self._request_tokens < 1:
                wait_time = max(wait_time, (1 - self._request_tokens) * 60 / self._rpm)
            
            if self._token_bucket < tokens:
                wait_time = max(wait_time, (tokens - self._token_bucket) * 60 / self._tpm)
            
            if wait_time == 0:
                self._request_tokens -= 1
                self._token_bucket -= tokens
            
            return wait_time


class ResponseCache:
    """Cache for API responses."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size: int = 10000):
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._max_size = max_size
        self._lock = Lock()
        
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_key(self, prompt: str, params: Dict) -> str:
        """Generate cache key."""
        content = f"{prompt}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get(self, prompt: str, params: Dict) -> Optional[str]:
        """Get cached response."""
        key = self._get_key(prompt, params)
        
        with self._lock:
            if key in self._cache:
                response, _ = self._cache[key]
                return response
        
        # Check disk cache
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    return data.get("response")
                except Exception:
                    pass
        
        return None
    
    def set(self, prompt: str, params: Dict, response: str):
        """Cache response."""
        key = self._get_key(prompt, params)
        
        with self._lock:
            # Evict oldest if full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[key] = (response, time.time())
        
        # Write to disk
        if self._cache_dir:
            cache_file = self._cache_dir / f"{key}.json"
            try:
                cache_file.write_text(json.dumps({
                    "prompt": prompt,
                    "params": params,
                    "response": response,
                    "cached_at": time.time()
                }))
            except Exception:
                pass
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
        
        if self._cache_dir:
            for f in self._cache_dir.glob("*.json"):
                f.unlink()


class BatchAPIClient:
    """Batch API client for efficient processing."""
    
    def __init__(
        self,
        provider: APIProvider = APIProvider.OPENAI,
        max_batch_size: int = 100,
        max_concurrent: int = 10
    ):
        self._provider = provider
        self._max_batch_size = max_batch_size
        self._max_concurrent = max_concurrent
        
        self._api_key: Optional[str] = None
        self._model = "gpt-3.5-turbo"
        
        self._queue: Queue = Queue()
        self._rate_limiter = RateLimiter()
        self._cache = ResponseCache()
        self._cost_config = CostConfig()
        
        self._total_tokens = 0
        self._total_cost = 0.0
        self._lock = Lock()
        
        logger.info(f"BatchAPIClient initialized for {provider.value}")
    
    def set_api_key(self, key: str):
        """Set API key."""
        self._api_key = key
    
    def set_model(self, model: str):
        """Set model to use."""
        self._model = model
    
    def set_provider(self, provider: APIProvider):
        """Set provider."""
        self._provider = provider
    
    def enable_cache(self, cache_dir: Optional[Path] = None):
        """Enable response caching."""
        self._cache = ResponseCache(cache_dir)
    
    def enqueue(
        self,
        prompt: str,
        parameters: Optional[Dict] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Add request to batch queue.
        
        Returns:
            Request ID
        """
        import uuid
        
        request = BatchRequest(
            id=str(uuid.uuid4()),
            prompt=prompt,
            parameters=parameters or {},
            callback=callback
        )
        
        self._queue.put(request)
        return request.id
    
    def process_batch(self, wait_for_all: bool = True) -> BatchResult:
        """
        Process all queued requests.
        
        Args:
            wait_for_all: Wait for all requests to complete
            
        Returns:
            BatchResult with all responses
        """
        # Gather requests
        requests = []
        while not self._queue.empty() and len(requests) < self._max_batch_size:
            try:
                requests.append(self._queue.get_nowait())
            except Exception:
                break
        
        if not requests:
            return BatchResult(0, 0, 0, 0, 0.0)
        
        start_time = time.time()
        responses = []
        
        # Process with concurrency
        with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
            futures = [
                executor.submit(self._process_single, req)
                for req in requests
            ]
            
            for future in futures:
                try:
                    response = future.result(timeout=60)
                    responses.append(response)
                except Exception as e:
                    responses.append(BatchResponse(
                        request_id="unknown",
                        success=False,
                        error=str(e)
                    ))
        
        # Calculate totals
        successful = sum(1 for r in responses if r.success)
        failed = len(responses) - successful
        total_tokens = sum(r.tokens_used for r in responses)
        
        # Calculate cost with batch discount
        cost = self._calculate_cost(total_tokens) * self._cost_config.batch_discount
        
        with self._lock:
            self._total_tokens += total_tokens
            self._total_cost += cost
        
        return BatchResult(
            total_requests=len(requests),
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_cost=cost,
            responses=responses,
            processing_time=time.time() - start_time
        )
    
    def _process_single(self, request: BatchRequest) -> BatchResponse:
        """Process a single request."""
        start = time.time()
        
        # Check cache
        params = {**request.parameters, "model": self._model}
        cached = self._cache.get(request.prompt, params)
        
        if cached:
            response = BatchResponse(
                request_id=request.id,
                success=True,
                response=cached,
                cached=True,
                latency_ms=(time.time() - start) * 1000
            )
            
            if request.callback:
                request.callback(response)
            
            return response
        
        # Rate limit
        wait_time = self._rate_limiter.acquire(len(request.prompt) // 4)
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Make API call
        try:
            result = self._call_api(request.prompt, params)
            
            response = BatchResponse(
                request_id=request.id,
                success=True,
                response=result.get("text", ""),
                tokens_used=result.get("tokens", 0),
                latency_ms=(time.time() - start) * 1000
            )
            
            # Cache result
            self._cache.set(request.prompt, params, response.response)
            
        except Exception as e:
            response = BatchResponse(
                request_id=request.id,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )
        
        if request.callback:
            request.callback(response)
        
        return response
    
    def _call_api(self, prompt: str, params: Dict) -> Dict:
        """Make actual API call."""
        if self._provider == APIProvider.OPENAI:
            return self._call_openai(prompt, params)
        elif self._provider == APIProvider.ANTHROPIC:
            return self._call_anthropic(prompt, params)
        elif self._provider == APIProvider.LOCAL:
            return self._call_local(prompt, params)
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")
    
    def _call_openai(self, prompt: str, params: Dict) -> Dict:
        """Call OpenAI API."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model=params.get("model", self._model),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7)
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            
            return {"text": text, "tokens": tokens}
            
        except ImportError:
            # Fallback for testing
            return {"text": f"(OpenAI not available) Echo: {prompt[:100]}", "tokens": len(prompt) // 4}
    
    def _call_anthropic(self, prompt: str, params: Dict) -> Dict:
        """Call Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self._api_key)
            response = client.messages.create(
                model=params.get("model", "claude-3-sonnet-20240229"),
                max_tokens=params.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            
            return {"text": text, "tokens": tokens}
            
        except ImportError:
            return {"text": f"(Anthropic not available) Echo: {prompt[:100]}", "tokens": len(prompt) // 4}
    
    def _call_local(self, prompt: str, params: Dict) -> Dict:
        """Call local model."""
        # Placeholder for local inference
        return {"text": f"Local response to: {prompt[:50]}...", "tokens": len(prompt) // 4}
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost for tokens."""
        # Assume 50% input, 50% output
        input_tokens = tokens // 2
        output_tokens = tokens - input_tokens
        
        input_cost = (input_tokens / 1000) * self._cost_config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self._cost_config.output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        with self._lock:
            return {
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
                "queue_size": self._queue.qsize()
            }
    
    async def process_batch_async(self) -> BatchResult:
        """Async version of process_batch."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process_batch
        )


class BatchProcessor:
    """Background batch processor."""
    
    def __init__(self, client: BatchAPIClient, interval: float = 10.0):
        self._client = client
        self._interval = interval
        self._running = False
        self._thread: Optional[Thread] = None
    
    def start(self):
        """Start background processing."""
        if self._running:
            return
        
        self._running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop background processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _run(self):
        """Background processing loop."""
        while self._running:
            try:
                result = self._client.process_batch()
                if result.total_requests > 0:
                    logger.info(
                        f"Batch processed: {result.successful}/{result.total_requests} "
                        f"in {result.processing_time:.2f}s"
                    )
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
            
            time.sleep(self._interval)


# Global instance
_batch_client: Optional[BatchAPIClient] = None


def get_batch_client() -> BatchAPIClient:
    """Get or create global batch client."""
    global _batch_client
    if _batch_client is None:
        _batch_client = BatchAPIClient()
    return _batch_client
