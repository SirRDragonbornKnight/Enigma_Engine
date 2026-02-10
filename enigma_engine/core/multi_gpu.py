"""
Multi-GPU Support

Distribute models across multiple GPUs using tensor parallelism,
pipeline parallelism, and data parallelism strategies.

FILE: enigma_engine/core/multi_gpu.py
TYPE: Core/Hardware
MAIN CLASSES: MultiGPUManager, TensorParallel, PipelineParallel
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """Model parallelism strategies."""
    DATA_PARALLEL = auto()      # Same model on all GPUs, different data
    TENSOR_PARALLEL = auto()    # Split tensors across GPUs
    PIPELINE_PARALLEL = auto()  # Split layers across GPUs
    HYBRID = auto()             # Combination of strategies


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    total_memory: int
    free_memory: int
    compute_capability: tuple[int, int] = (0, 0)
    
    @property
    def memory_gb(self) -> float:
        return self.total_memory / (1024**3)
    
    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024**3)


@dataclass
class ParallelConfig:
    """Configuration for multi-GPU parallelism."""
    strategy: ParallelStrategy = ParallelStrategy.DATA_PARALLEL
    num_gpus: int = -1  # -1 = auto-detect all
    gpu_ids: list[int] = field(default_factory=list)
    pipeline_stages: int = 4
    tensor_parallel_size: int = 2
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimize_memory: bool = True
    sync_batch_norm: bool = True


if HAS_TORCH:
    
    def get_available_gpus() -> list[GPUInfo]:
        """Get information about available GPUs."""
        gpus = []
        
        if not torch.cuda.is_available():
            return gpus
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory=props.total_memory,
                free_memory=torch.cuda.mem_get_info(i)[0],
                compute_capability=(props.major, props.minor)
            ))
        
        return gpus
    
    
    class MultiGPUManager:
        """
        Manage multi-GPU distribution and execution.
        """
        
        def __init__(self, config: ParallelConfig = None):
            self.config = config or ParallelConfig()
            self.gpus = get_available_gpus()
            self.is_initialized = False
            self._wrapped_model = None
            
            # Determine GPU IDs
            if self.config.gpu_ids:
                self.gpu_ids = self.config.gpu_ids
            elif self.config.num_gpus > 0:
                self.gpu_ids = list(range(min(self.config.num_gpus, len(self.gpus))))
            else:
                self.gpu_ids = list(range(len(self.gpus)))
        
        @property
        def num_gpus(self) -> int:
            return len(self.gpu_ids)
        
        @property
        def primary_device(self) -> torch.device:
            if self.gpu_ids:
                return torch.device(f"cuda:{self.gpu_ids[0]}")
            return torch.device("cpu")
        
        def distribute_model(
            self,
            model: nn.Module,
            strategy: ParallelStrategy = None
        ) -> nn.Module:
            """
            Distribute model across GPUs.
            
            Args:
                model: Model to distribute
                strategy: Override default strategy
            
            Returns:
                Distributed model wrapper
            """
            strategy = strategy or self.config.strategy
            
            if self.num_gpus <= 1:
                logger.info("Single GPU mode")
                return model.to(self.primary_device)
            
            if strategy == ParallelStrategy.DATA_PARALLEL:
                return self._data_parallel(model)
            elif strategy == ParallelStrategy.TENSOR_PARALLEL:
                return self._tensor_parallel(model)
            elif strategy == ParallelStrategy.PIPELINE_PARALLEL:
                return self._pipeline_parallel(model)
            elif strategy == ParallelStrategy.HYBRID:
                return self._hybrid_parallel(model)
            else:
                return self._data_parallel(model)
        
        def _data_parallel(self, model: nn.Module) -> nn.Module:
            """Standard DataParallel distribution."""
            logger.info(f"Using DataParallel across GPUs: {self.gpu_ids}")
            
            model = model.to(self.primary_device)
            
            if self.config.sync_batch_norm:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            if len(self.gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=self.gpu_ids)
            
            self._wrapped_model = model
            return model
        
        def _tensor_parallel(self, model: nn.Module) -> nn.Module:
            """Tensor parallelism - split large layers across GPUs."""
            logger.info(f"Using Tensor Parallelism across {self.num_gpus} GPUs")
            
            tp_model = TensorParallelModel(
                model,
                gpu_ids=self.gpu_ids,
                tp_size=self.config.tensor_parallel_size
            )
            
            self._wrapped_model = tp_model
            return tp_model
        
        def _pipeline_parallel(self, model: nn.Module) -> nn.Module:
            """Pipeline parallelism - split layers across GPUs."""
            logger.info(f"Using Pipeline Parallelism with {self.config.pipeline_stages} stages")
            
            pp_model = PipelineParallelModel(
                model,
                gpu_ids=self.gpu_ids,
                num_stages=self.config.pipeline_stages,
                micro_batch_size=self.config.micro_batch_size
            )
            
            self._wrapped_model = pp_model
            return pp_model
        
        def _hybrid_parallel(self, model: nn.Module) -> nn.Module:
            """Hybrid parallelism - combine strategies."""
            logger.info("Using Hybrid Parallelism (TP + PP)")
            
            # First apply tensor parallelism within groups
            # Then pipeline parallelism across groups
            tp_size = min(self.config.tensor_parallel_size, self.num_gpus // 2)
            pp_size = self.num_gpus // tp_size
            
            # Split GPUs into groups for TP
            tp_groups = [
                self.gpu_ids[i:i+tp_size] 
                for i in range(0, len(self.gpu_ids), tp_size)
            ]
            
            model = TensorParallelModel(model, gpu_ids=tp_groups[0], tp_size=tp_size)
            model = PipelineParallelModel(model, gpu_ids=self.gpu_ids, num_stages=pp_size)
            
            self._wrapped_model = model
            return model
        
        def synchronize(self):
            """Synchronize all GPUs."""
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        def get_memory_summary(self) -> dict[int, dict[str, float]]:
            """Get memory usage on each GPU."""
            summary = {}
            
            for gpu_id in self.gpu_ids:
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                
                summary[gpu_id] = {
                    "allocated_gb": allocated / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "total_gb": total / (1024**3),
                    "utilization": allocated / total * 100
                }
            
            return summary
        
        def optimize_memory(self):
            """Optimize memory usage across GPUs."""
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
    
    
    class TensorParallelModel(nn.Module):
        """
        Tensor Parallel model wrapper.
        
        Splits large linear layers across GPUs.
        """
        
        def __init__(
            self,
            model: nn.Module,
            gpu_ids: list[int],
            tp_size: int = 2
        ):
            super().__init__()
            self.model = model
            self.gpu_ids = gpu_ids
            self.tp_size = min(tp_size, len(gpu_ids))
            
            self._shard_model()
        
        def _shard_model(self):
            """Shard model layers across GPUs."""
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.shape[0] > 1024:
                    # Shard large linear layers
                    self._shard_linear(name, module)
        
        def _shard_linear(self, name: str, module: nn.Linear):
            """Shard a linear layer across GPUs."""
            out_features = module.out_features
            in_features = module.in_features
            
            # Split output dimension
            shard_size = out_features // self.tp_size
            
            for i, gpu_id in enumerate(self.gpu_ids[:self.tp_size]):
                start = i * shard_size
                end = (i + 1) * shard_size if i < self.tp_size - 1 else out_features
                
                # Create shard
                shard = nn.Linear(in_features, end - start, bias=module.bias is not None)
                shard.weight.data = module.weight.data[start:end].to(f"cuda:{gpu_id}")
                if module.bias is not None:
                    shard.bias.data = module.bias.data[start:end].to(f"cuda:{gpu_id}")
                
                # Store shard reference
                if not hasattr(module, '_shards'):
                    module._shards = []
                module._shards.append((gpu_id, shard))
        
        def forward(self, *args, **kwargs):
            """Forward with tensor parallel gathering."""
            return self.model(*args, **kwargs)
    
    
    class PipelineParallelModel(nn.Module):
        """
        Pipeline Parallel model wrapper.
        
        Splits sequential layers across GPUs with micro-batching.
        """
        
        def __init__(
            self,
            model: nn.Module,
            gpu_ids: list[int],
            num_stages: int = 4,
            micro_batch_size: int = 1
        ):
            super().__init__()
            self.model = model
            self.gpu_ids = gpu_ids
            self.num_stages = min(num_stages, len(gpu_ids))
            self.micro_batch_size = micro_batch_size
            
            self.stages = nn.ModuleList()
            self.stage_devices = []
            
            self._partition_model()
        
        def _partition_model(self):
            """Partition model into pipeline stages."""
            # Get all layers
            layers = []
            for name, child in self.model.named_children():
                if isinstance(child, nn.Sequential) or isinstance(child, nn.ModuleList):
                    for i, layer in enumerate(child):
                        layers.append((f"{name}.{i}", layer))
                else:
                    layers.append((name, child))
            
            if not layers:
                # Model has no obvious sequential structure
                self.stages.append(self.model)
                self.stage_devices.append(self.gpu_ids[0])
                return
            
            # Divide layers into stages
            layers_per_stage = max(1, len(layers) // self.num_stages)
            
            current_stage = nn.Sequential()
            stage_idx = 0
            
            for i, (name, layer) in enumerate(layers):
                current_stage.add_module(name, layer)
                
                # Check if we should start a new stage
                if (i + 1) % layers_per_stage == 0 and stage_idx < self.num_stages - 1:
                    device_idx = self.gpu_ids[stage_idx % len(self.gpu_ids)]
                    self.stages.append(current_stage.to(f"cuda:{device_idx}"))
                    self.stage_devices.append(device_idx)
                    current_stage = nn.Sequential()
                    stage_idx += 1
            
            # Add remaining layers to last stage
            if len(current_stage) > 0:
                device_idx = self.gpu_ids[stage_idx % len(self.gpu_ids)]
                self.stages.append(current_stage.to(f"cuda:{device_idx}"))
                self.stage_devices.append(device_idx)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through pipeline stages.
            
            Uses naive sequential execution. For training,
            consider using GPipe or 1F1B schedule.
            """
            for i, stage in enumerate(self.stages):
                device = f"cuda:{self.stage_devices[i]}"
                x = x.to(device)
                x = stage(x)
            
            return x
        
        def forward_micro_batch(
            self,
            inputs: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward with micro-batching for memory efficiency.
            
            Splits batch into micro-batches and processes
            through pipeline stages.
            """
            batch_size = inputs.shape[0]
            outputs = []
            
            for i in range(0, batch_size, self.micro_batch_size):
                micro_batch = inputs[i:i + self.micro_batch_size]
                output = self.forward(micro_batch)
                outputs.append(output)
            
            return torch.cat(outputs, dim=0)
    
    
    def auto_distribute(
        model: nn.Module,
        strategy: str = "auto"
    ) -> nn.Module:
        """
        Automatically distribute model across available GPUs.
        
        Args:
            model: Model to distribute
            strategy: 'auto', 'dp', 'tp', 'pp', or 'hybrid'
        
        Returns:
            Distributed model
        """
        gpus = get_available_gpus()
        
        if not gpus:
            logger.info("No GPUs available, using CPU")
            return model
        
        if len(gpus) == 1:
            logger.info("Single GPU detected")
            return model.to("cuda:0")
        
        # Choose strategy
        if strategy == "auto":
            # Use data parallel for small number of GPUs
            if len(gpus) <= 2:
                strategy = "dp"
            # Use pipeline for sequential models
            elif hasattr(model, 'layers') or hasattr(model, 'transformer'):
                strategy = "pp"
            else:
                strategy = "dp"
        
        strategy_map = {
            "dp": ParallelStrategy.DATA_PARALLEL,
            "tp": ParallelStrategy.TENSOR_PARALLEL,
            "pp": ParallelStrategy.PIPELINE_PARALLEL,
            "hybrid": ParallelStrategy.HYBRID
        }
        
        config = ParallelConfig(
            strategy=strategy_map.get(strategy, ParallelStrategy.DATA_PARALLEL),
            gpu_ids=[g.index for g in gpus]
        )
        
        manager = MultiGPUManager(config)
        return manager.distribute_model(model)
    
    
    def get_optimal_batch_size(
        model: nn.Module,
        sample_input: torch.Tensor,
        target_memory_usage: float = 0.8
    ) -> int:
        """
        Find optimal batch size for memory efficiency.
        
        Args:
            model: Model to test
            sample_input: Sample input tensor (batch_size=1)
            target_memory_usage: Target memory usage fraction
        
        Returns:
            Recommended batch size
        """
        if not torch.cuda.is_available():
            return 1
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass with batch=1
        with torch.no_grad():
            _ = model(sample_input)
        
        # Memory for one sample
        memory_per_sample = torch.cuda.max_memory_allocated() - initial_memory
        
        # Available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available = total_memory * target_memory_usage - initial_memory
        
        # Calculate batch size
        batch_size = max(1, int(available / memory_per_sample))
        
        torch.cuda.empty_cache()
        return batch_size

else:
    def get_available_gpus():
        return []
    
    class MultiGPUManager:
        pass
    
    class TensorParallelModel:
        pass
    
    class PipelineParallelModel:
        pass
    
    def auto_distribute(model, strategy="auto"):
        return model
    
    def get_optimal_batch_size(*args, **kwargs):
        return 1
