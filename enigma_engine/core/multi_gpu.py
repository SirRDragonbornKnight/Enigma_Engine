"""
================================================================================
🖥️ MULTI-GPU TRAINING — MG-B: DataParallel + MG-C: DistributedDataParallel
================================================================================

Utilities for splitting model training across multiple GPUs.

- MG-B: ``DataParallelWrapper`` — wraps a model in ``nn.DataParallel``
  for single-machine multi-GPU.  Minimal code change, limited scaling.
- MG-C: ``DistributedTrainer`` — proper ``DistributedDataParallel`` with
  per-process gradient sync.  Better performance but needs process spawning.

📍 FILE: enigma_engine/core/multi_gpu.py
🏷️ TYPE: Training / Infrastructure
🎯 MAIN CLASSES: DataParallelWrapper, DistributedTrainer

┌─────────────────────────────────────────────────────────────────────────────┐
│  MG-B: DataParallel (simple)                                                │
│    model = wrap_data_parallel(model)   # splits batches across GPUs        │
│    trainer.train(data)                  # unchanged training code           │
│                                                                             │
│  MG-C: DistributedDataParallel (proper)                                    │
│    trainer = DistributedTrainer(model, tokenizer, config)                  │
│    trainer.train(data)                  # handles spawn, sync, cleanup     │
└─────────────────────────────────────────────────────────────────────────────┘
│  Multi-GPU support utilities.                                               │
│  Currently DORMANT — no callers in the codebase.                            │
│  Exported in __init__.py for future use.                                    │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    ← EXPORTED BY: enigma_engine/__init__.py
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_torch_available = False
try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:
    pass


# =============================================================================
# GPU DETECTION
# =============================================================================

def get_gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    if not _torch_available:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_gpu_info() -> list[dict[str, Any]]:
    """Return info about each available GPU."""
    if not _torch_available or not torch.cuda.is_available():
        return []
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(total / (1024**3), 1),
            "free_memory_gb": round(free / (1024**3), 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return info


def is_multi_gpu() -> bool:
    """Check if more than one GPU is available."""
    return get_gpu_count() > 1


# =============================================================================
# MG-B: DataParallel WRAPPER
# =============================================================================

def wrap_data_parallel(
    model: Any,
    device_ids: list[int] | None = None,
) -> Any:
    """Wrap a model in ``nn.DataParallel`` for multi-GPU training.

    If only one GPU (or no GPU) is available, returns the model unchanged.
    The wrapper splits each batch across GPUs, computes forward in parallel,
    and gathers results on the primary device.

    Args:
        model: A PyTorch ``nn.Module``.
        device_ids: GPU indices to use (default: all available).

    Returns:
        The model wrapped in ``DataParallel``, or unchanged if single-GPU.
    """
    if not _torch_available:
        return model

    n_gpus = get_gpu_count()
    if n_gpus <= 1:
        logger.debug("Single GPU or CPU — skipping DataParallel")
        return model

    if device_ids is None:
        device_ids = list(range(n_gpus))

    if isinstance(model, nn.DataParallel):
        logger.debug("Model already wrapped in DataParallel")
        return model

    logger.info(
        f"Wrapping model in DataParallel across {len(device_ids)} GPUs: "
        f"{device_ids}"
    )
    wrapped = nn.DataParallel(model, device_ids=device_ids)
    return wrapped


def unwrap_data_parallel(model: Any) -> Any:
    """Unwrap a ``DataParallel`` model to get the inner module.

    Safe to call on non-wrapped models — returns unchanged.
    """
    if not _torch_available:
        return model
    if isinstance(model, nn.DataParallel):
        return model.module
    if hasattr(model, "module"):
        return model.module
    return model


# =============================================================================
# MG-C: DistributedDataParallel TRAINER
# =============================================================================

@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        world_size: Total number of processes (default: GPU count).
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
        master_addr: Address of the rank-0 process.
        master_port: Port for process communication.
        device_ids: GPU indices to use (default: one per process).
    """
    world_size: int = 0  # 0 = auto-detect from GPU count
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    device_ids: list[int] | None = None


class DistributedTrainer:
    """Multi-GPU trainer using DistributedDataParallel (MG-C).

    Handles process group setup, model wrapping, gradient
    synchronisation, and cleanup.  Designed to work with the
    existing :class:`Trainer` from ``training.py``.

    Usage::

        from enigma_engine.core.multi_gpu import DistributedTrainer, DistributedConfig

        dt = DistributedTrainer(model, tokenizer, config=DistributedConfig())
        dt.setup()
        # ... use dt.model in training loop ...
        dt.cleanup()

    Or as a context manager::

        with DistributedTrainer(model, tokenizer) as dt:
            trainer = Trainer(dt.model, tokenizer, training_config)
            trainer.train(data)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: DistributedConfig | None = None,
    ):
        self.raw_model = model
        self.tokenizer = tokenizer
        self.config = config or DistributedConfig()
        self.model: Any = model  # becomes DDP-wrapped after setup()
        self._initialized = False
        self._rank = 0

    def setup(self, rank: int = 0) -> None:
        """Initialize the distributed process group and wrap the model.

        Args:
            rank: This process's rank in the group.
        """
        if not _torch_available:
            logger.warning("PyTorch not available — distributed setup skipped")
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA not available — distributed setup skipped")
            return

        cfg = self.config
        world_size = cfg.world_size or get_gpu_count()

        if world_size <= 1:
            logger.info("Single GPU — using plain model (no DDP)")
            return

        os.environ["MASTER_ADDR"] = cfg.master_addr
        os.environ["MASTER_PORT"] = cfg.master_port

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend=cfg.backend,
                rank=rank,
                world_size=world_size,
            )

        self._rank = rank
        torch.cuda.set_device(rank)
        self.raw_model = self.raw_model.to(rank)

        from torch.nn.parallel import DistributedDataParallel as DDP
        self.model = DDP(
            self.raw_model,
            device_ids=[rank],
            output_device=rank,
        )
        self._initialized = True
        logger.info(
            f"DDP initialized: rank={rank}, world_size={world_size}")

    def cleanup(self) -> None:
        """Destroy the process group."""
        if not self._initialized:
            return
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            os.environ.pop("MASTER_ADDR", None)
            os.environ.pop("MASTER_PORT", None)
            self._initialized = False
            logger.info("DDP process group destroyed")
        except Exception:
            logger.debug("DDP cleanup error", exc_info=True)

    def get_sampler(
        self, dataset: Any, shuffle: bool = True
    ) -> Any:
        """Create a DistributedSampler for a dataset.

        This ensures each rank gets a different shard of the data.
        """
        if not self._initialized:
            return None
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)

    @property
    def is_main_process(self) -> bool:
        """True if this is rank 0 (primary process)."""
        return self._rank == 0

    def unwrap(self) -> Any:
        """Get the unwrapped model (for saving checkpoints)."""
        return unwrap_data_parallel(self.model)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DistributedConfig",
    "DistributedTrainer",
    "get_gpu_count",
    "get_gpu_info",
    "is_multi_gpu",
    "unwrap_data_parallel",
    "wrap_data_parallel",
]
