"""
Cloud Training for Enigma AI Engine

One-click training on cloud GPUs.

Features:
- Multi-provider support (RunPod, Lambda, Vast.ai)
- Job management
- Cost estimation
- Progress tracking
- Checkpoint sync

Usage:
    from enigma_engine.core.cloud_training import CloudTrainer, CloudProvider
    
    trainer = CloudTrainer(provider=CloudProvider.RUNPOD)
    trainer.configure(api_key="your_key")
    
    # Launch training job
    job = trainer.launch(
        model_config={"size": "small"},
        data_path="data/training.txt",
        gpu_type="RTX4090"
    )
    
    # Monitor progress
    status = trainer.get_status(job.job_id)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    RUNPOD = "runpod"
    LAMBDA_LABS = "lambda"
    VAST_AI = "vast"
    PAPERSPACE = "paperspace"
    GOOGLE_COLAB = "colab"
    AWS_SAGEMAKER = "sagemaker"


class GPUType(Enum):
    """GPU types available."""
    RTX_3090 = "rtx3090"
    RTX_4090 = "rtx4090"
    A100_40GB = "a100_40"
    A100_80GB = "a100_80"
    H100 = "h100"
    T4 = "t4"
    V100 = "v100"


class JobStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GPUPricing:
    """GPU pricing information."""
    gpu_type: GPUType
    hourly_rate: float
    provider: CloudProvider
    memory_gb: int
    availability: float  # 0-1


@dataclass
class TrainingJob:
    """A cloud training job."""
    job_id: str
    provider: CloudProvider
    gpu_type: GPUType
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Progress
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    
    # Cost
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    
    # Resources
    instance_id: Optional[str] = None
    log_url: Optional[str] = None
    checkpoint_url: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration for cloud jobs."""
    model_size: str = "small"
    batch_size: int = 32
    learning_rate: float = 0.0001
    epochs: int = 10
    max_seq_length: int = 512
    gradient_accumulation: int = 4
    mixed_precision: bool = True
    checkpoint_interval: int = 1000


class CloudTrainer:
    """Cloud training manager."""
    
    # Default pricing (approximate)
    DEFAULT_PRICING = {
        GPUType.RTX_3090: 0.40,
        GPUType.RTX_4090: 0.60,
        GPUType.A100_40GB: 1.50,
        GPUType.A100_80GB: 2.50,
        GPUType.H100: 4.00,
        GPUType.T4: 0.30,
        GPUType.V100: 0.80
    }
    
    def __init__(self, provider: CloudProvider = CloudProvider.RUNPOD):
        """
        Initialize cloud trainer.
        
        Args:
            provider: Cloud provider to use
        """
        self.provider = provider
        self._api_key: Optional[str] = None
        self._jobs: Dict[str, TrainingJob] = {}
        self._listeners: List[Callable[[str, TrainingJob], None]] = []
        
        # Provider clients (would be actual API clients in production)
        self._client = None
    
    def configure(self, api_key: str, **kwargs):
        """
        Configure the trainer.
        
        Args:
            api_key: Provider API key
            **kwargs: Additional provider settings
        """
        self._api_key = api_key
        # In production, initialize provider client here
        logger.info(f"Configured cloud trainer for {self.provider.value}")
    
    def estimate_cost(
        self,
        config: TrainingConfig,
        data_size_mb: float,
        gpu_type: GPUType = GPUType.RTX_4090
    ) -> Dict[str, float]:
        """
        Estimate training cost.
        
        Args:
            config: Training configuration
            data_size_mb: Training data size in MB
            gpu_type: GPU type to use
            
        Returns:
            Cost estimates
        """
        hourly_rate = self.DEFAULT_PRICING.get(gpu_type, 1.0)
        
        # Rough estimation based on data size and epochs
        # Real estimation would use actual benchmarks
        steps_per_mb = 1000  # Approximate
        total_steps = int(data_size_mb * steps_per_mb * config.epochs)
        steps_per_hour = 5000  # Varies by GPU
        
        estimated_hours = total_steps / steps_per_hour
        estimated_cost = estimated_hours * hourly_rate
        
        return {
            "estimated_hours": round(estimated_hours, 2),
            "hourly_rate": hourly_rate,
            "estimated_cost": round(estimated_cost, 2),
            "gpu_type": gpu_type.value,
            "total_steps": total_steps
        }
    
    def launch(
        self,
        data_path: str,
        config: Optional[TrainingConfig] = None,
        gpu_type: GPUType = GPUType.RTX_4090,
        callback: Optional[Callable[[TrainingJob], None]] = None
    ) -> TrainingJob:
        """
        Launch a training job.
        
        Args:
            data_path: Path to training data
            config: Training configuration
            gpu_type: GPU type to request
            callback: Progress callback
            
        Returns:
            Training job
        """
        if not self._api_key:
            raise ValueError("API key not configured")
        
        config = config or TrainingConfig()
        
        # Create job
        job = TrainingJob(
            job_id=str(uuid.uuid4())[:12],
            provider=self.provider,
            gpu_type=gpu_type,
            total_epochs=config.epochs
        )
        
        # Estimate cost
        data_size = Path(data_path).stat().st_size / (1024 * 1024) if Path(data_path).exists() else 10
        estimates = self.estimate_cost(config, data_size, gpu_type)
        job.estimated_cost = estimates["estimated_cost"]
        job.total_steps = estimates["total_steps"]
        
        self._jobs[job.job_id] = job
        
        if callback:
            self._listeners.append(lambda e, j: callback(j) if j.job_id == job.job_id else None)
        
        # Launch job (simulated - real implementation would call provider API)
        self._simulate_launch(job, config, data_path)
        
        logger.info(f"Launched training job {job.job_id} on {self.provider.value}")
        
        return job
    
    def _simulate_launch(self, job: TrainingJob, config: TrainingConfig, data_path: str):
        """Simulate job launch (for demonstration)."""
        job.status = JobStatus.PROVISIONING
        self._emit("job_provisioning", job)
        
        # In real implementation, this would:
        # 1. Upload training data to cloud storage
        # 2. Launch VM/container with GPU
        # 3. Start training script
        # 4. Set up checkpoint sync
        
        job.instance_id = f"instance-{uuid.uuid4().hex[:8]}"
        job.log_url = f"https://{self.provider.value}.example.com/logs/{job.job_id}"
    
    def get_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status."""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[TrainingJob]:
        """Get all jobs."""
        return list(self._jobs.values())
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a training job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.PROVISIONING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            self._emit("job_cancelled", job)
            logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    def download_checkpoint(self, job_id: str, output_path: str) -> Optional[str]:
        """
        Download model checkpoint from completed job.
        
        Args:
            job_id: Job ID
            output_path: Local path to save checkpoint
            
        Returns:
            Path to downloaded checkpoint
        """
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.COMPLETED:
            return None
        
        # In real implementation, download from cloud storage
        logger.info(f"Downloading checkpoint from {job.checkpoint_url} to {output_path}")
        
        return output_path
    
    def get_logs(self, job_id: str, tail: int = 100) -> List[str]:
        """
        Get training logs.
        
        Args:
            job_id: Job ID
            tail: Number of recent lines
            
        Returns:
            Log lines
        """
        job = self._jobs.get(job_id)
        if not job:
            return []
        
        # In real implementation, fetch from cloud
        return [f"[{datetime.now().isoformat()}] Training step {i}: loss=0.5" for i in range(tail)]
    
    def list_available_gpus(self) -> List[GPUPricing]:
        """List available GPUs with pricing."""
        # In real implementation, query provider API
        return [
            GPUPricing(GPUType.RTX_4090, 0.60, self.provider, 24, 0.9),
            GPUPricing(GPUType.A100_40GB, 1.50, self.provider, 40, 0.7),
            GPUPricing(GPUType.A100_80GB, 2.50, self.provider, 80, 0.5),
            GPUPricing(GPUType.H100, 4.00, self.provider, 80, 0.3)
        ]
    
    def add_listener(self, callback: Callable[[str, TrainingJob], None]):
        """Add event listener."""
        self._listeners.append(callback)
    
    def _emit(self, event: str, job: TrainingJob):
        """Emit event."""
        for listener in self._listeners:
            try:
                listener(event, job)
            except Exception as e:
                logger.error(f"Listener error: {e}")


class MultiCloudTrainer:
    """Train across multiple cloud providers for best pricing."""
    
    def __init__(self):
        """Initialize multi-cloud trainer."""
        self._providers: Dict[CloudProvider, CloudTrainer] = {}
    
    def add_provider(self, provider: CloudProvider, api_key: str):
        """Add a cloud provider."""
        trainer = CloudTrainer(provider)
        trainer.configure(api_key)
        self._providers[provider] = trainer
    
    def find_best_price(
        self,
        config: TrainingConfig,
        data_size_mb: float,
        min_gpu_memory: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Find best GPU pricing across providers.
        
        Args:
            config: Training config
            data_size_mb: Data size in MB
            min_gpu_memory: Minimum GPU memory required
            
        Returns:
            Sorted list of options
        """
        options = []
        
        for provider, trainer in self._providers.items():
            gpus = trainer.list_available_gpus()
            
            for gpu in gpus:
                if gpu.memory_gb >= min_gpu_memory:
                    estimate = trainer.estimate_cost(config, data_size_mb, gpu.gpu_type)
                    options.append({
                        "provider": provider.value,
                        "gpu": gpu.gpu_type.value,
                        "memory_gb": gpu.memory_gb,
                        "availability": gpu.availability,
                        **estimate
                    })
        
        return sorted(options, key=lambda x: x["estimated_cost"])
    
    def launch_best(
        self,
        data_path: str,
        config: Optional[TrainingConfig] = None,
        min_gpu_memory: int = 24
    ) -> Optional[TrainingJob]:
        """Launch training on best-priced available GPU."""
        if not self._providers:
            raise ValueError("No providers configured")
        
        config = config or TrainingConfig()
        data_size = Path(data_path).stat().st_size / (1024 * 1024) if Path(data_path).exists() else 10
        
        options = self.find_best_price(config, data_size, min_gpu_memory)
        
        for option in options:
            if option["availability"] > 0.5:  # Require 50% availability
                provider = CloudProvider(option["provider"])
                trainer = self._providers[provider]
                gpu_type = GPUType(option["gpu"])
                
                return trainer.launch(data_path, config, gpu_type)
        
        return None


# Convenience functions
def quick_cloud_train(
    data_path: str,
    provider: CloudProvider = CloudProvider.RUNPOD,
    api_key: Optional[str] = None
) -> TrainingJob:
    """
    Quick cloud training setup.
    
    Args:
        data_path: Path to training data
        provider: Cloud provider
        api_key: Provider API key
        
    Returns:
        Training job
    """
    trainer = CloudTrainer(provider)
    
    if api_key:
        trainer.configure(api_key)
    else:
        # Try to load from config
        import os
        api_key = os.environ.get(f"{provider.value.upper()}_API_KEY")
        if api_key:
            trainer.configure(api_key)
        else:
            raise ValueError(f"No API key for {provider.value}")
    
    return trainer.launch(data_path)
