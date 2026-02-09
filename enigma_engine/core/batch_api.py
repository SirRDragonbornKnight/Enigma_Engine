"""
Batch API Calls for Enigma AI Engine

Use batch APIs (OpenAI, etc.) for lower-cost training data generation.

Features:
- OpenAI Batch API support
- Request batching and queueing
- Async batch processing
- Cost tracking
- Rate limit handling

Usage:
    from enigma_engine.core.batch_api import BatchAPIClient
    
    client = BatchAPIClient(api_key="sk-...")
    
    # Queue requests
    for prompt in prompts:
        client.queue_request("gpt-4", prompt)
    
    # Submit batch
    batch_id = client.submit_batch()
    
    # Check status / get results
    results = client.get_results(batch_id)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request in a batch."""
    id: str
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 1024
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """A batch job."""
    batch_id: str
    requests: List[BatchRequest]
    status: str = "pending"  # pending, submitted, processing, completed, failed
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    api_batch_id: Optional[str] = None  # External API batch ID


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 50000  # OpenAI allows 50k requests per batch
    max_file_size_mb: int = 100  # Max JSONL file size
    check_interval: int = 60  # Seconds between status checks
    output_dir: Path = Path("data/batch_outputs")
    save_requests: bool = True


class BatchAPIClient:
    """Client for batch API operations."""
    
    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        config: Optional[BatchConfig] = None
    ):
        """
        Initialize batch API client.
        
        Args:
            api_key: API key
            provider: API provider (openai, anthropic)
            config: Batch configuration
        """
        self.api_key = api_key
        self.provider = provider
        self.config = config or BatchConfig()
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Request queue
        self._queue: List[BatchRequest] = []
        
        # Active batches
        self._batches: Dict[str, BatchJob] = {}
        
        # Cost tracking
        self._total_cost = 0.0
        
        # Initialize provider client
        self._client = self._init_client()
    
    def _init_client(self):
        """Initialize the API client."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI library not installed")
                return None
        
        return None
    
    def queue_request(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Queue a request for batch processing.
        
        Args:
            model: Model name
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens
            metadata: Additional metadata
            
        Returns:
            Request ID
        """
        request = BatchRequest(
            id=str(uuid.uuid4()),
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {}
        )
        
        self._queue.append(request)
        return request.id
    
    def queue_messages(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        metadata: Optional[Dict] = None
    ) -> str:
        """Queue a request with full message history."""
        request = BatchRequest(
            id=str(uuid.uuid4()),
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {}
        )
        
        self._queue.append(request)
        return request.id
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def clear_queue(self):
        """Clear the request queue."""
        self._queue.clear()
    
    def submit_batch(self) -> Optional[str]:
        """
        Submit queued requests as a batch.
        
        Returns:
            Batch ID if successful
        """
        if not self._queue:
            logger.warning("No requests in queue")
            return None
        
        # Create batch job
        batch_id = str(uuid.uuid4())[:8]
        batch = BatchJob(
            batch_id=batch_id,
            requests=list(self._queue)
        )
        
        # Clear queue
        self._queue.clear()
        
        # Submit based on provider
        if self.provider == "openai":
            success = self._submit_openai_batch(batch)
        else:
            logger.error(f"Unsupported provider: {self.provider}")
            return None
        
        if success:
            batch.status = "submitted"
            batch.submitted_at = time.time()
            self._batches[batch_id] = batch
            logger.info(f"Submitted batch {batch_id} with {len(batch.requests)} requests")
            return batch_id
        
        return None
    
    def _submit_openai_batch(self, batch: BatchJob) -> bool:
        """Submit batch to OpenAI."""
        if not self._client:
            logger.error("OpenAI client not initialized")
            return False
        
        try:
            # Create JSONL file
            jsonl_path = self.config.output_dir / f"batch_{batch.batch_id}_input.jsonl"
            
            with open(jsonl_path, 'w') as f:
                for request in batch.requests:
                    line = {
                        "custom_id": request.id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": request.model,
                            "messages": request.messages,
                            "temperature": request.temperature,
                            "max_tokens": request.max_tokens
                        }
                    }
                    f.write(json.dumps(line) + '\n')
            
            # Upload file
            with open(jsonl_path, 'rb') as f:
                file_response = self._client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            # Create batch
            batch_response = self._client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            batch.api_batch_id = batch_response.id
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit OpenAI batch: {e}")
            return False
    
    def get_batch_status(self, batch_id: str) -> Optional[str]:
        """
        Get batch status.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            Status string
        """
        batch = self._batches.get(batch_id)
        if not batch:
            return None
        
        # Check external status
        if self.provider == "openai" and batch.api_batch_id:
            try:
                response = self._client.batches.retrieve(batch.api_batch_id)
                
                status_map = {
                    "validating": "processing",
                    "in_progress": "processing",
                    "finalizing": "processing",
                    "completed": "completed",
                    "failed": "failed",
                    "expired": "failed",
                    "cancelled": "failed"
                }
                
                batch.status = status_map.get(response.status, response.status)
                
                if batch.status == "completed" and not batch.completed_at:
                    batch.completed_at = time.time()
                    
            except Exception as e:
                logger.error(f"Failed to get batch status: {e}")
        
        return batch.status
    
    def get_results(self, batch_id: str) -> Optional[List[Dict]]:
        """
        Get batch results.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            List of results if complete, None otherwise
        """
        batch = self._batches.get(batch_id)
        if not batch:
            return None
        
        # Check status first
        status = self.get_batch_status(batch_id)
        
        if status != "completed":
            logger.info(f"Batch {batch_id} status: {status}")
            return None
        
        # Return cached results if available
        if batch.results:
            return batch.results
        
        # Fetch results
        if self.provider == "openai":
            batch.results = self._get_openai_results(batch)
        
        return batch.results
    
    def _get_openai_results(self, batch: BatchJob) -> List[Dict]:
        """Fetch results from OpenAI."""
        results = []
        
        try:
            # Get batch info
            batch_info = self._client.batches.retrieve(batch.api_batch_id)
            
            if not batch_info.output_file_id:
                return results
            
            # Download output file
            content = self._client.files.content(batch_info.output_file_id)
            
            # Save output
            output_path = self.config.output_dir / f"batch_{batch.batch_id}_output.jsonl"
            output_path.write_bytes(content.content)
            
            # Parse results
            for line in content.content.decode().strip().split('\n'):
                if line:
                    result = json.loads(line)
                    results.append({
                        "request_id": result.get("custom_id"),
                        "response": result.get("response", {}).get("body", {}),
                        "error": result.get("error")
                    })
            
        except Exception as e:
            logger.error(f"Failed to get OpenAI results: {e}")
        
        return results
    
    async def wait_for_completion(
        self,
        batch_id: str,
        timeout: int = 86400,  # 24 hours
        callback: Optional[Callable[[str, str], None]] = None
    ) -> bool:
        """
        Wait for batch to complete.
        
        Args:
            batch_id: Batch ID
            timeout: Max wait time in seconds
            callback: Progress callback(batch_id, status)
            
        Returns:
            True if completed successfully
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_batch_status(batch_id)
            
            if callback:
                callback(batch_id, status)
            
            if status == "completed":
                return True
            elif status == "failed":
                return False
            
            await asyncio.sleep(self.config.check_interval)
        
        logger.warning(f"Batch {batch_id} timed out")
        return False
    
    def estimate_cost(
        self,
        model: str = "gpt-4",
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 500
    ) -> Dict[str, float]:
        """
        Estimate cost for current queue.
        
        Args:
            model: Model name
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request
            
        Returns:
            Cost estimates
        """
        # OpenAI batch API pricing (50% discount)
        # Prices per 1M tokens
        prices = {
            "gpt-4": {"input": 15.0, "output": 30.0},
            "gpt-4-turbo": {"input": 5.0, "output": 15.0},
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-4o-mini": {"input": 0.075, "output": 0.30},
            "gpt-3.5-turbo": {"input": 0.25, "output": 0.75}
        }
        
        model_prices = prices.get(model, prices["gpt-4"])
        
        # Batch discount (50%)
        batch_discount = 0.5
        
        num_requests = len(self._queue)
        total_input = num_requests * avg_input_tokens
        total_output = num_requests * avg_output_tokens
        
        input_cost = (total_input / 1_000_000) * model_prices["input"] * batch_discount
        output_cost = (total_output / 1_000_000) * model_prices["output"] * batch_discount
        
        return {
            "num_requests": num_requests,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "discount_percentage": (1 - batch_discount) * 100
        }
    
    def list_batches(self) -> List[Dict]:
        """List all batches."""
        return [
            {
                "batch_id": b.batch_id,
                "status": b.status,
                "num_requests": len(b.requests),
                "created_at": datetime.fromtimestamp(b.created_at).isoformat(),
                "submitted_at": datetime.fromtimestamp(b.submitted_at).isoformat() if b.submitted_at else None,
                "completed_at": datetime.fromtimestamp(b.completed_at).isoformat() if b.completed_at else None
            }
            for b in self._batches.values()
        ]
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch."""
        batch = self._batches.get(batch_id)
        if not batch:
            return False
        
        if self.provider == "openai" and batch.api_batch_id:
            try:
                self._client.batches.cancel(batch.api_batch_id)
                batch.status = "cancelled"
                return True
            except Exception as e:
                logger.error(f"Failed to cancel batch: {e}")
        
        return False


class TrainingDataBatchGenerator:
    """Generate training data using batch API."""
    
    def __init__(self, api_key: str, provider: str = "openai"):
        self.client = BatchAPIClient(api_key, provider)
    
    def generate_training_examples(
        self,
        task_prompts: List[str],
        system_prompt: str,
        model: str = "gpt-4o-mini",
        examples_per_prompt: int = 5
    ) -> str:
        """
        Generate training examples using batch API.
        
        Args:
            task_prompts: List of task prompts
            system_prompt: System prompt for generation
            model: Model to use
            examples_per_prompt: Examples to generate per prompt
            
        Returns:
            Batch ID
        """
        generation_prompt = f"{system_prompt}\n\nGenerate {examples_per_prompt} diverse training examples for this task. Format as JSONL with 'input' and 'output' fields."
        
        for task in task_prompts:
            self.client.queue_request(
                model=model,
                prompt=task,
                system_prompt=generation_prompt,
                temperature=0.9,
                max_tokens=2048,
                metadata={"task": task}
            )
        
        return self.client.submit_batch()
    
    def process_batch_results(
        self,
        batch_id: str,
        output_path: Path
    ) -> int:
        """
        Process batch results into training data.
        
        Args:
            batch_id: Batch ID
            output_path: Output JSONL path
            
        Returns:
            Number of examples generated
        """
        results = self.client.get_results(batch_id)
        if not results:
            return 0
        
        count = 0
        
        with open(output_path, 'w') as f:
            for result in results:
                response = result.get("response", {})
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Parse JSONL from content
                for line in content.strip().split('\n'):
                    try:
                        example = json.loads(line)
                        if "input" in example and "output" in example:
                            f.write(json.dumps(example) + '\n')
                            count += 1
                    except json.JSONDecodeError:
                        continue
        
        return count


# Convenience function
def create_batch_client(api_key: str, provider: str = "openai") -> BatchAPIClient:
    """Create a batch API client."""
    return BatchAPIClient(api_key, provider)
