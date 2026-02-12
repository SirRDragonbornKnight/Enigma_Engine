"""
Model Sharding - Split large models across multiple machines

Enables running models that don't fit on a single device by:
- Partitioning model layers across nodes
- Coordinating forward/backward passes
- Managing activation transfers
- Supporting Pipeline Parallelism (PP) and Tensor Parallelism (TP)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SHARD COORDINATOR                         │
    │         Manages shard assignment and communication          │
    ├─────────────────────────────────────────────────────────────┤
    │   Node 0         Node 1         Node 2         Node 3       │
    │   ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐      │
    │   │Layers │────>│Layers │────>│Layers │────>│Layers │      │
    │   │ 0-7   │     │ 8-15  │     │16-23  │     │24-31  │      │
    │   └───────┘     └───────┘     └───────┘     └───────┘      │
    │   (Embeddings)              (Attention)     (Output)        │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # On coordinator node
    coordinator = ShardCoordinator(model_config)
    coordinator.add_node("192.168.1.100", port=5000, device="cuda:0")
    coordinator.add_node("192.168.1.101", port=5000, device="cuda:0")
    coordinator.distribute_model(model_path)
    
    # Run inference
    output = coordinator.generate("Hello", max_tokens=50)
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """How to shard the model."""
    LAYER_PARALLEL = "layer"  # Split by layers (Pipeline Parallel)
    TENSOR_PARALLEL = "tensor"  # Split tensors (Tensor Parallel)
    EXPERT_PARALLEL = "expert"  # Split MoE experts
    HYBRID = "hybrid"  # Combination


@dataclass
class NodeInfo:
    """Information about a compute node."""
    node_id: str
    host: str
    port: int
    device: str = "cuda:0"
    memory_gb: float = 0.0
    is_online: bool = False
    shard_ids: list[int] = field(default_factory=list)
    load: float = 0.0
    last_heartbeat: float = 0.0


@dataclass
class ShardConfig:
    """Configuration for model sharding."""
    strategy: ShardingStrategy = ShardingStrategy.LAYER_PARALLEL
    num_shards: int = 2
    overlap_communication: bool = True
    micro_batch_size: int = 1
    pipeline_stages: int = 4
    tensor_parallel_size: int = 1
    activation_checkpointing: bool = True
    fp16_communication: bool = True
    
    @property
    def total_parallel_size(self) -> int:
        return self.pipeline_stages * self.tensor_parallel_size


@dataclass
class Shard:
    """A model shard assigned to a node."""
    shard_id: int
    node_id: str
    layer_start: int
    layer_end: int
    contains_embedding: bool = False
    contains_output: bool = False
    state_dict: Optional[dict] = None
    memory_usage_mb: float = 0.0


class ActivationCache:
    """Cache for intermediate activations during pipeline parallel."""
    
    def __init__(self, max_size_mb: float = 1024):
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.cache: dict[str, torch.Tensor] = {}
        self.current_size = 0
        self._lock = threading.Lock()
    
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Store activation in cache."""
        with self._lock:
            tensor_size = tensor.element_size() * tensor.nelement()
            
            # Evict if necessary
            while self.current_size + tensor_size > self.max_size and self.cache:
                oldest_key = next(iter(self.cache))
                self._evict(oldest_key)
            
            self.cache[key] = tensor
            self.current_size += tensor_size
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve activation from cache."""
        with self._lock:
            return self.cache.get(key)
    
    def remove(self, key: str) -> None:
        """Remove activation from cache."""
        with self._lock:
            if key in self.cache:
                self._evict(key)
    
    def _evict(self, key: str) -> None:
        tensor = self.cache.pop(key)
        self.current_size -= tensor.element_size() * tensor.nelement()
    
    def clear(self) -> None:
        """Clear all cached activations."""
        with self._lock:
            self.cache.clear()
            self.current_size = 0


class ShardWorker:
    """
    Worker that runs a model shard on a single node.
    
    Receives activations, runs forward pass, sends results.
    """
    
    def __init__(
        self,
        node_id: str,
        shard: Shard,
        device: str = "cuda:0"
    ):
        self.node_id = node_id
        self.shard = shard
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_layers: Optional[nn.ModuleList] = None
        self.embedding: Optional[nn.Module] = None
        self.output_head: Optional[nn.Module] = None
        self.activation_cache = ActivationCache()
        self._is_ready = False
    
    def load_shard(self, state_dict: dict) -> None:
        """Load model shard from state dict."""
        try:
            # This would be customized based on model architecture
            # For now, we'll store the state dict
            self.shard.state_dict = state_dict
            self._is_ready = True
            logger.info(f"Node {self.node_id}: Loaded shard {self.shard.shard_id}")
        except Exception as e:
            logger.error(f"Failed to load shard: {e}")
            raise
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        micro_batch_id: int = 0,
    ) -> torch.Tensor:
        """
        Run forward pass on this shard.
        
        Args:
            hidden_states: Input tensor (from previous shard or embedding)
            attention_mask: Attention mask
            micro_batch_id: ID for micro-batch (for pipelining)
            
        Returns:
            Output tensor to send to next shard
        """
        if not self._is_ready:
            raise RuntimeError("Shard not loaded")
        
        # Move to device
        hidden_states = hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Cache input for backward pass
        cache_key = f"mb{micro_batch_id}_layer{self.shard.layer_start}_input"
        self.activation_cache.put(cache_key, hidden_states)
        
        # Run layers in this shard
        # This is a simplified version - real implementation would
        # iterate through actual model layers
        output = hidden_states
        
        # If this shard contains the embedding layer
        if self.shard.contains_embedding and self.embedding is not None:
            output = self.embedding(hidden_states)
        
        # Run transformer layers
        if self.model_layers is not None:
            for layer in self.model_layers:
                output = layer(output, attention_mask=attention_mask)
        
        # If this shard contains the output head
        if self.shard.contains_output and self.output_head is not None:
            output = self.output_head(output)
        
        return output
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available() and "cuda" in str(self.device):
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        return 0.0


class ShardCoordinator:
    """
    Coordinates distributed model sharding across multiple nodes.
    
    Responsibilities:
    - Node discovery and management
    - Model partitioning and distribution
    - Forward pass orchestration
    - Activation transfer between nodes
    """
    
    def __init__(self, config: Optional[ShardConfig] = None):
        self.config = config or ShardConfig()
        self.nodes: dict[str, NodeInfo] = {}
        self.shards: list[Shard] = []
        self.workers: dict[str, ShardWorker] = {}
        self._is_distributed = False
        self._lock = threading.Lock()
        
        # Communication
        self._send_queues: dict[str, asyncio.Queue] = {}
        self._recv_queues: dict[str, asyncio.Queue] = {}
    
    def add_node(
        self,
        host: str,
        port: int = 5000,
        device: str = "cuda:0",
        memory_gb: float = 0.0,
    ) -> str:
        """
        Add a compute node to the cluster.
        
        Args:
            host: Node hostname or IP
            port: Port for communication
            device: Target device on node
            memory_gb: Available GPU memory
            
        Returns:
            Node ID
        """
        node_id = f"{host}:{port}"
        
        with self._lock:
            if node_id in self.nodes:
                logger.warning(f"Node {node_id} already registered")
                return node_id
            
            node = NodeInfo(
                node_id=node_id,
                host=host,
                port=port,
                device=device,
                memory_gb=memory_gb,
                is_online=False,
            )
            
            self.nodes[node_id] = node
            logger.info(f"Added node: {node_id} (device: {device})")
            
        return node_id
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                if node_id in self.workers:
                    del self.workers[node_id]
                logger.info(f"Removed node: {node_id}")
    
    def partition_model(
        self,
        model: nn.Module,
        num_layers: Optional[int] = None,
    ) -> list[Shard]:
        """
        Partition model into shards.
        
        Args:
            model: Model to partition
            num_layers: Total number of layers (if known)
            
        Returns:
            List of Shard objects
        """
        if not self.nodes:
            raise ValueError("No nodes registered")
        
        num_nodes = len(self.nodes)
        
        # Auto-detect layers if not specified
        if num_layers is None:
            if hasattr(model, 'layers'):
                num_layers = len(model.layers)
            elif hasattr(model, 'transformer'):
                num_layers = len(model.transformer.h)
            else:
                num_layers = 24  # Default assumption
        
        # Calculate layers per shard
        layers_per_shard = (num_layers + num_nodes - 1) // num_nodes
        
        shards = []
        node_ids = list(self.nodes.keys())
        
        for i, node_id in enumerate(node_ids):
            layer_start = i * layers_per_shard
            layer_end = min((i + 1) * layers_per_shard, num_layers)
            
            shard = Shard(
                shard_id=i,
                node_id=node_id,
                layer_start=layer_start,
                layer_end=layer_end,
                contains_embedding=(i == 0),
                contains_output=(i == num_nodes - 1),
            )
            
            shards.append(shard)
            self.nodes[node_id].shard_ids.append(i)
            
            logger.info(
                f"Shard {i}: layers {layer_start}-{layer_end} "
                f"(embed: {shard.contains_embedding}, output: {shard.contains_output})"
            )
        
        self.shards = shards
        return shards
    
    def distribute_model(
        self,
        model_path: Path,
        model: Optional[nn.Module] = None,
    ):
        """
        Distribute model to nodes.
        
        Args:
            model_path: Path to saved model
            model: Or provide model directly
        """
        logger.info(f"Distributing model from {model_path} to {len(self.nodes)} nodes")
        
        # Load model if not provided
        if model is None:
            state_dict = torch.load(model_path, map_location="cpu")
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
        else:
            state_dict = model.state_dict()
        
        # Partition if not already done
        if not self.shards:
            # Create a temporary model to get layer count
            self.partition_model(model, num_layers=None)
        
        # Extract and distribute shards
        for shard in self.shards:
            # Extract relevant layers for this shard
            shard_state = self._extract_shard_state(
                state_dict,
                shard.layer_start,
                shard.layer_end,
                shard.contains_embedding,
                shard.contains_output,
            )
            
            # Send to node
            self._send_shard_to_node(shard, shard_state)
        
        self._is_distributed = True
        logger.info("Model distribution complete")
    
    def _extract_shard_state(
        self,
        full_state: dict,
        layer_start: int,
        layer_end: int,
        include_embedding: bool,
        include_output: bool,
    ) -> dict:
        """Extract state dict for a specific shard."""
        shard_state = {}
        
        for key, value in full_state.items():
            # Check if this key belongs to this shard
            include = False
            
            # Embedding layer
            if include_embedding and any(x in key for x in ['embed', 'wte', 'wpe', 'token']):
                include = True
            
            # Output layer
            if include_output and any(x in key for x in ['lm_head', 'output', 'ln_f', 'norm']):
                include = True
            
            # Check layer number
            for layer_num in range(layer_start, layer_end):
                if f'.{layer_num}.' in key or f'[{layer_num}]' in key:
                    include = True
                    break
            
            if include:
                shard_state[key] = value
        
        return shard_state
    
    def _send_shard_to_node(self, shard: Shard, state_dict: dict) -> None:
        """Send shard state to a node."""
        node = self.nodes[shard.node_id]
        
        # For local execution, create worker directly
        worker = ShardWorker(
            node_id=node.node_id,
            shard=shard,
            device=node.device,
        )
        worker.load_shard(state_dict)
        self.workers[node.node_id] = worker
        
        # Calculate memory usage
        memory_mb = sum(
            v.element_size() * v.nelement() / (1024 ** 2)
            for v in state_dict.values()
            if isinstance(v, torch.Tensor)
        )
        shard.memory_usage_mb = memory_mb
        
        logger.info(f"Sent shard {shard.shard_id} to {node.node_id} ({memory_mb:.1f} MB)")
    
    async def forward_distributed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run distributed forward pass.
        
        Orchestrates activation flow through all shards.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Model output
        """
        if not self._is_distributed:
            raise RuntimeError("Model not distributed yet")
        
        hidden_states = input_ids
        
        # Process through each shard in order
        for shard in sorted(self.shards, key=lambda s: s.shard_id):
            worker = self.workers.get(shard.node_id)
            if worker is None:
                raise RuntimeError(f"Worker for shard {shard.shard_id} not found")
            
            # Forward through this shard
            hidden_states = worker.forward(
                hidden_states,
                attention_mask=attention_mask,
            )
            
            # Convert to fp16 for transfer if configured
            if self.config.fp16_communication and hidden_states.dtype == torch.float32:
                hidden_states = hidden_states.half()
        
        return hidden_states
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text using distributed model.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self._is_distributed:
            raise RuntimeError("Model not distributed")
        
        # This would need a tokenizer
        logger.warning("generate() requires tokenizer integration")
        return ""
    
    def get_cluster_status(self) -> dict:
        """Get status of all nodes in cluster."""
        status = {
            "num_nodes": len(self.nodes),
            "num_shards": len(self.shards),
            "is_distributed": self._is_distributed,
            "config": {
                "strategy": self.config.strategy.value,
                "pipeline_stages": self.config.pipeline_stages,
                "tensor_parallel_size": self.config.tensor_parallel_size,
            },
            "nodes": {},
        }
        
        for node_id, node in self.nodes.items():
            worker = self.workers.get(node_id)
            status["nodes"][node_id] = {
                "host": node.host,
                "port": node.port,
                "device": node.device,
                "is_online": node.is_online,
                "shard_ids": node.shard_ids,
                "memory_gb": node.memory_gb,
                "memory_used_mb": worker.get_memory_usage() if worker else 0,
            }
        
        return status
    
    def shutdown(self) -> None:
        """Shutdown all workers and cleanup."""
        logger.info("Shutting down shard coordinator...")
        
        for node_id, worker in self.workers.items():
            worker.activation_cache.clear()
        
        self.workers.clear()
        self._is_distributed = False


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_shard_memory(
    model_params: int,
    num_shards: int,
    dtype: torch.dtype = torch.float16,
) -> float:
    """
    Estimate memory needed per shard.
    
    Args:
        model_params: Total model parameters
        num_shards: Number of shards
        dtype: Data type
        
    Returns:
        Estimated memory in GB per shard
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)
    
    total_bytes = model_params * bytes_per_param
    per_shard_bytes = total_bytes / num_shards
    
    # Add ~20% overhead for activations, optimizer states
    per_shard_bytes *= 1.2
    
    return per_shard_bytes / (1024 ** 3)


def create_sharding_plan(
    model_params: int,
    available_nodes: list[dict[str, Any]],
    strategy: ShardingStrategy = ShardingStrategy.LAYER_PARALLEL,
) -> ShardConfig:
    """
    Create optimal sharding plan based on available resources.
    
    Args:
        model_params: Total model parameters
        available_nodes: List of nodes with memory info
        strategy: Sharding strategy
        
    Returns:
        ShardConfig optimized for available resources
    """
    total_memory = sum(n.get("memory_gb", 0) for n in available_nodes)
    model_size_gb = estimate_shard_memory(model_params, 1)
    
    # Calculate minimum shards needed
    min_shards = max(1, int(model_size_gb / min(n.get("memory_gb", 8) for n in available_nodes) * 1.2))
    
    # Use number of nodes if more than min_shards
    num_shards = max(min_shards, len(available_nodes))
    
    config = ShardConfig(
        strategy=strategy,
        num_shards=num_shards,
        pipeline_stages=num_shards if strategy == ShardingStrategy.LAYER_PARALLEL else 1,
        tensor_parallel_size=num_shards if strategy == ShardingStrategy.TENSOR_PARALLEL else 1,
    )
    
    logger.info(
        f"Created sharding plan: {num_shards} shards, "
        f"~{estimate_shard_memory(model_params, num_shards):.2f}GB per shard"
    )
    
    return config
