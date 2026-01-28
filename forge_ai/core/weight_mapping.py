"""
Weight Mapping Utility
======================

Unified weight mapping system for converting various model formats to Forge architecture.
Handles different naming conventions, tensor shapes, and quantized weights.

Usage:
    from forge_ai.core.weight_mapping import WeightMapper
    
    mapper = WeightMapper()
    forge_weights = mapper.map_huggingface_to_forge(hf_state_dict, config)
    forge_weights = mapper.map_gguf_to_forge(gguf_tensors, config)
    forge_weights = mapper.map_onnx_to_forge(onnx_weights, config)
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Optional torch import
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    logger.warning("torch not available - weight mapping will be limited")


class WeightMapper:
    """
    Maps weights from various model formats to Forge architecture.
    
    Handles:
    - Different naming conventions (model.layers.0.self_attn vs layers.0.attention)
    - Different tensor shapes (transpose if needed)
    - Missing/extra weights (skip or initialize)
    - Quantized weights (dequantize)
    """
    
    def __init__(self):
        """Initialize weight mapper."""
        self.mappings = {
            'huggingface': self._get_hf_mappings(),
            'gguf': self._get_gguf_mappings(),
            'onnx': self._get_onnx_mappings()
        }
    
    def map_name(self, source_name: str, source_format: str = 'huggingface') -> Optional[str]:
        """
        Map a single weight name from source format to Forge format.
        
        Args:
            source_name: Original weight name (e.g., "transformer.wte.weight")
            source_format: Source format ("huggingface", "gguf", "onnx")
            
        Returns:
            Mapped Forge weight name, or None if no mapping found
            
        Example:
            >>> mapper = WeightMapper()
            >>> mapper.map_name("transformer.wte.weight", "huggingface")
            'tok_embeddings.weight'
        """
        if source_format not in self.mappings:
            logger.warning(f"Unknown source format: {source_format}")
            return None
        
        mappings = self.mappings[source_format]
        
        # Try direct mappings first (non-layer weights)
        for forge_key, source_patterns in mappings.items():
            if '{}' not in forge_key:
                for pattern in source_patterns:
                    if pattern == source_name:
                        return forge_key
        
        # Try layer-based mappings
        for forge_key_template, source_patterns in mappings.items():
            if '{}' in forge_key_template:
                for pattern in source_patterns:
                    # Extract layer number from source name
                    pattern_regex = pattern.replace('.{}', r'\.(\d+)')
                    match = re.match(pattern_regex, source_name)
                    if match:
                        layer_num = match.group(1)
                        return forge_key_template.format(layer_num)
        
        return None
    
    def transform_tensor(
        self,
        tensor: Any,
        source_name: str,
        target_name: str,
        source_format: str = 'huggingface'
    ) -> Any:
        """
        Transform tensor if needed (transpose, reshape, split, etc.).
        
        Different model formats may require:
        - Transposition (Conv1D weights in GPT-2)
        - Splitting (combined QKV in GPT-2/GPT-NeoX)
        - Reshaping (different dimension orders)
        
        Args:
            tensor: Source tensor
            source_name: Original weight name
            target_name: Target weight name
            source_format: Source format
            
        Returns:
            Transformed tensor
            
        Example:
            >>> # GPT-2 has combined QKV that needs splitting
            >>> qkv_tensor = torch.randn(2304, 768)  # 3 * 768 = 2304
            >>> q_tensor = mapper.transform_tensor(
            ...     qkv_tensor, 
            ...     "transformer.h.0.attn.c_attn.weight",
            ...     "layers.0.attention.wq.weight",
            ...     "huggingface"
            ... )
            >>> q_tensor.shape
            torch.Size([768, 768])
        """
        if not HAVE_TORCH:
            return tensor
        
        import torch
        
        # GPT-2 style models: combined QKV needs splitting
        if 'c_attn' in source_name and source_format == 'huggingface':
            # Check if this is a Q, K, or V weight
            if any(x in target_name for x in ['.wq.', '.wk.', '.wv.']):
                return self._split_qkv(tensor, target_name)
        
        # GPT-NeoX style: combined query_key_value
        if 'query_key_value' in source_name and source_format == 'huggingface':
            return self._split_qkv(tensor, target_name)
        
        # GPT-2 uses Conv1D which needs transposition
        if source_format == 'huggingface' and 'gpt2' in source_name.lower():
            if tensor.dim() == 2 and 'weight' in source_name:
                # Conv1D stores weights transposed
                return tensor.transpose(0, 1).contiguous()
        
        # No transformation needed
        return tensor
    
    def _get_hf_mappings(self) -> Dict[str, List[str]]:
        """
        Get HuggingFace to Forge name mappings.
        
        Returns mapping from Forge name patterns to possible HF names.
        """
        return {
            # Embeddings
            'tok_embeddings.weight': [
                'transformer.wte.weight',
                'model.embed_tokens.weight',
                'embeddings.word_embeddings.weight',
                'gpt_neox.embed_in.weight',
                'wte.weight'
            ],
            
            # Output head
            'output.weight': [
                'lm_head.weight',
                'output.weight',
                'embed_out.weight'
            ],
            
            # Layer normalization
            'norm.weight': [
                'transformer.ln_f.weight',
                'model.norm.weight',
                'ln_f.weight',
                'final_layernorm.weight',
                'gpt_neox.final_layer_norm.weight'
            ],
            
            # Attention Q/K/V projections (per layer)
            'layers.{}.attention.wq.weight': [
                'transformer.h.{}.attn.c_attn.weight',  # GPT-2 style (combined QKV)
                'model.layers.{}.self_attn.q_proj.weight',  # LLaMA style
                'transformer.h.{}.attn.q_proj.weight',
                'gpt_neox.layers.{}.attention.query_key_value.weight'  # GPT-NeoX (combined)
            ],
            
            'layers.{}.attention.wk.weight': [
                'model.layers.{}.self_attn.k_proj.weight',
                'transformer.h.{}.attn.k_proj.weight'
            ],
            
            'layers.{}.attention.wv.weight': [
                'model.layers.{}.self_attn.v_proj.weight',
                'transformer.h.{}.attn.v_proj.weight'
            ],
            
            'layers.{}.attention.wo.weight': [
                'transformer.h.{}.attn.c_proj.weight',
                'model.layers.{}.self_attn.o_proj.weight',
                'transformer.h.{}.attn.out_proj.weight',
                'gpt_neox.layers.{}.attention.dense.weight'
            ],
            
            # Feed-forward
            'layers.{}.feed_forward.w1.weight': [
                'transformer.h.{}.mlp.c_fc.weight',
                'model.layers.{}.mlp.gate_proj.weight',  # LLaMA (SwiGLU gate)
                'gpt_neox.layers.{}.mlp.dense_h_to_4h.weight'
            ],
            
            'layers.{}.feed_forward.w2.weight': [
                'transformer.h.{}.mlp.c_proj.weight',
                'model.layers.{}.mlp.down_proj.weight',
                'gpt_neox.layers.{}.mlp.dense_4h_to_h.weight'
            ],
            
            'layers.{}.feed_forward.w3.weight': [
                'model.layers.{}.mlp.up_proj.weight'  # LLaMA SwiGLU up projection
            ],
            
            # Layer norms
            'layers.{}.attention_norm.weight': [
                'transformer.h.{}.ln_1.weight',
                'model.layers.{}.input_layernorm.weight',
                'gpt_neox.layers.{}.input_layernorm.weight'
            ],
            
            'layers.{}.ffn_norm.weight': [
                'transformer.h.{}.ln_2.weight',
                'model.layers.{}.post_attention_layernorm.weight',
                'gpt_neox.layers.{}.post_attention_layernorm.weight'
            ]
        }
    
    def _get_gguf_mappings(self) -> Dict[str, List[str]]:
        """
        Get GGUF to Forge name mappings.
        
        GGUF uses llama.cpp naming convention.
        """
        return {
            # Embeddings
            'tok_embeddings.weight': [
                'token_embd.weight',
                'tok_embeddings.weight'
            ],
            
            # Output
            'output.weight': [
                'output.weight',
                'output_norm.weight'
            ],
            
            # Final norm
            'norm.weight': [
                'output_norm.weight',
                'norm.weight'
            ],
            
            # Attention (per layer)
            'layers.{}.attention.wq.weight': [
                'blk.{}.attn_q.weight'
            ],
            
            'layers.{}.attention.wk.weight': [
                'blk.{}.attn_k.weight'
            ],
            
            'layers.{}.attention.wv.weight': [
                'blk.{}.attn_v.weight'
            ],
            
            'layers.{}.attention.wo.weight': [
                'blk.{}.attn_output.weight'
            ],
            
            # Feed-forward
            'layers.{}.feed_forward.w1.weight': [
                'blk.{}.ffn_gate.weight'
            ],
            
            'layers.{}.feed_forward.w2.weight': [
                'blk.{}.ffn_down.weight'
            ],
            
            'layers.{}.feed_forward.w3.weight': [
                'blk.{}.ffn_up.weight'
            ],
            
            # Norms
            'layers.{}.attention_norm.weight': [
                'blk.{}.attn_norm.weight'
            ],
            
            'layers.{}.ffn_norm.weight': [
                'blk.{}.ffn_norm.weight'
            ]
        }
    
    def _get_onnx_mappings(self) -> Dict[str, List[str]]:
        """
        Get ONNX to Forge name mappings.
        
        ONNX uses various naming depending on export source.
        """
        return {
            # Similar to HuggingFace but with potential prefixes
            'tok_embeddings.weight': [
                '/transformer/wte/weight',
                '/model/embed_tokens/weight',
                'input_ids'
            ],
            
            'output.weight': [
                '/lm_head/weight',
                '/output/weight'
            ]
            # Add more mappings as needed
        }
    
    def map_huggingface_to_forge(
        self, 
        hf_state_dict: Dict[str, Any],
        forge_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Map HuggingFace weight names to Forge names.
        
        Args:
            hf_state_dict: HuggingFace model state dict
            forge_config: Optional Forge config for validation
            
        Returns:
            Forge-compatible state dict
        """
        if not HAVE_TORCH:
            raise RuntimeError("torch required for weight mapping")
        
        forge_state_dict = {}
        mappings = self.mappings['huggingface']
        
        # Detect number of layers
        n_layers = self._detect_num_layers(hf_state_dict)
        logger.info(f"Detected {n_layers} layers in HuggingFace model")
        
        # Map non-layer weights
        for forge_key, hf_patterns in mappings.items():
            if '{}' not in forge_key:  # Non-layer weights
                for pattern in hf_patterns:
                    if pattern in hf_state_dict:
                        forge_state_dict[forge_key] = hf_state_dict[pattern]
                        logger.debug(f"Mapped {pattern} -> {forge_key}")
                        break
        
        # Map layer weights
        for layer_idx in range(n_layers):
            for forge_key_template, hf_patterns in mappings.items():
                if '{}' in forge_key_template:
                    forge_key = forge_key_template.format(layer_idx)
                    
                    for pattern in hf_patterns:
                        hf_key = pattern.format(layer_idx)
                        
                        if hf_key in hf_state_dict:
                            weight = hf_state_dict[hf_key]
                            
                            # Handle special cases
                            if 'c_attn' in hf_key:
                                # GPT-2 style: combined QKV needs splitting
                                weight = self._split_qkv(weight, forge_key)
                            
                            if 'query_key_value' in hf_key:
                                # GPT-NeoX style: combined QKV needs splitting
                                weight = self._split_qkv(weight, forge_key)
                            
                            forge_state_dict[forge_key] = weight
                            logger.debug(f"Mapped {hf_key} -> {forge_key}")
                            break
        
        logger.info(f"Mapped {len(forge_state_dict)} weights from HuggingFace to Forge")
        return forge_state_dict
    
    def map_gguf_to_forge(
        self,
        gguf_tensors: Dict[str, Any],
        forge_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Map GGUF tensor names to Forge names.
        
        Args:
            gguf_tensors: GGUF model tensors (already dequantized)
            forge_config: Optional Forge config for validation
            
        Returns:
            Forge-compatible state dict
        """
        if not HAVE_TORCH:
            raise RuntimeError("torch required for weight mapping")
        
        forge_state_dict = {}
        mappings = self.mappings['gguf']
        
        # Detect number of layers
        n_layers = max([
            int(name.split('.')[1]) 
            for name in gguf_tensors.keys() 
            if name.startswith('blk.')
        ], default=0) + 1
        logger.info(f"Detected {n_layers} layers in GGUF model")
        
        # Map weights
        for forge_key_template, gguf_patterns in mappings.items():
            if '{}' not in forge_key_template:
                # Non-layer weights
                for pattern in gguf_patterns:
                    if pattern in gguf_tensors:
                        forge_state_dict[forge_key_template] = gguf_tensors[pattern]
                        break
            else:
                # Layer weights
                for layer_idx in range(n_layers):
                    forge_key = forge_key_template.format(layer_idx)
                    for pattern in gguf_patterns:
                        gguf_key = pattern.format(layer_idx)
                        if gguf_key in gguf_tensors:
                            forge_state_dict[forge_key] = gguf_tensors[gguf_key]
                            break
        
        logger.info(f"Mapped {len(forge_state_dict)} weights from GGUF to Forge")
        return forge_state_dict
    
    def map_onnx_to_forge(
        self,
        onnx_weights: Dict[str, Any],
        forge_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Map ONNX weight names to Forge names.
        
        Args:
            onnx_weights: ONNX model weights
            forge_config: Optional Forge config for validation
            
        Returns:
            Forge-compatible state dict
        """
        if not HAVE_TORCH:
            raise RuntimeError("torch required for weight mapping")
        
        forge_state_dict = {}
        
        # ONNX naming is highly variable, use heuristics
        for onnx_key, weight in onnx_weights.items():
            forge_key = self._infer_forge_key(onnx_key)
            if forge_key:
                forge_state_dict[forge_key] = weight
        
        logger.info(f"Mapped {len(forge_state_dict)} weights from ONNX to Forge")
        return forge_state_dict
    
    def _detect_num_layers(self, state_dict: Dict[str, Any]) -> int:
        """Detect number of transformer layers in state dict."""
        layer_indices = set()
        
        for key in state_dict.keys():
            # Look for patterns like "layers.0", "h.0", "blocks.0"
            for pattern in ['model.layers.', 'transformer.h.', 'blocks.']:
                if pattern in key:
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            layer_indices.add(int(part))
                            break
        
        return max(layer_indices, default=0) + 1 if layer_indices else 0
    
    def _split_qkv(self, qkv_weight: Any, target_key: str) -> Any:
        """
        Split combined QKV weight into Q, K, or V based on target key.
        
        Args:
            qkv_weight: Combined QKV weight tensor
            target_key: Target key indicating which component (wq, wk, wv)
            
        Returns:
            Split weight for the requested component
        """
        if not HAVE_TORCH:
            return qkv_weight
        
        # Determine which component to extract
        if '.wq.' in target_key:
            idx = 0
        elif '.wk.' in target_key:
            idx = 1
        elif '.wv.' in target_key:
            idx = 2
        else:
            return qkv_weight
        
        # Split into 3 equal parts
        split_size = qkv_weight.shape[0] // 3
        start = idx * split_size
        end = (idx + 1) * split_size
        
        return qkv_weight[start:end, :]
    
    def _infer_forge_key(self, onnx_key: str) -> Optional[str]:
        """
        Infer Forge key from ONNX key using heuristics.
        
        Args:
            onnx_key: ONNX weight key
            
        Returns:
            Inferred Forge key or None
        """
        # Remove ONNX-specific prefixes
        key = onnx_key.lstrip('/')
        
        # Simple heuristic mapping
        if 'embed' in key.lower():
            return 'tok_embeddings.weight'
        elif 'lm_head' in key or 'output' in key:
            return 'output.weight'
        elif 'ln_f' in key or 'final' in key:
            return 'norm.weight'
        
        # Layer-specific weights
        # Extract layer number if present
        layer_num = None
        for part in key.split('.'):
            if part.isdigit():
                layer_num = part
                break
        
        if layer_num:
            if 'attn_q' in key or 'q_proj' in key:
                return f'layers.{layer_num}.attention.wq.weight'
            elif 'attn_k' in key or 'k_proj' in key:
                return f'layers.{layer_num}.attention.wk.weight'
            elif 'attn_v' in key or 'v_proj' in key:
                return f'layers.{layer_num}.attention.wv.weight'
            elif 'attn_output' in key or 'o_proj' in key:
                return f'layers.{layer_num}.attention.wo.weight'
            elif 'ffn_gate' in key or 'gate_proj' in key:
                return f'layers.{layer_num}.feed_forward.w1.weight'
            elif 'ffn_down' in key or 'down_proj' in key:
                return f'layers.{layer_num}.feed_forward.w2.weight'
            elif 'ffn_up' in key or 'up_proj' in key:
                return f'layers.{layer_num}.feed_forward.w3.weight'
        
        return None
    
    def dequantize_gguf_tensor(self, tensor_data: bytes, quant_type: str, shape: Tuple[int, ...]) -> Any:
        """
        Dequantize GGUF tensor data.
        
        Args:
            tensor_data: Raw tensor bytes
            quant_type: Quantization type (Q4_0, Q8_0, etc.)
            shape: Tensor shape
            
        Returns:
            Dequantized PyTorch tensor
            
        Raises:
            NotImplementedError: GGUF dequantization not fully implemented yet
        """
        if not HAVE_TORCH:
            raise RuntimeError("torch required for dequantization")
        
        # Full GGUF dequantization requires implementing various quantization formats
        # This is a complex task that depends on the gguf library's internal format
        raise NotImplementedError(
            f"GGUF dequantization for {quant_type} not yet implemented. "
            f"Use the gguf library's built-in dequantization instead."
        )
    
    def validate_mapping(
        self,
        forge_state_dict: Dict[str, Any],
        expected_keys: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that mapping produced expected keys.
        
        Args:
            forge_state_dict: Mapped state dict
            expected_keys: Optional list of expected keys
            
        Returns:
            Tuple of (is_valid, missing_keys)
        """
        if expected_keys is None:
            return True, []
        
        mapped_keys = set(forge_state_dict.keys())
        expected_keys_set = set(expected_keys)
        
        missing_keys = list(expected_keys_set - mapped_keys)
        
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys: {missing_keys[:5]}...")
            return False, missing_keys
        
        return True, []


def create_mapper() -> WeightMapper:
    """Create a WeightMapper instance."""
    return WeightMapper()


def create_weight_mapper(source_format: str = 'huggingface', target_format: str = 'forge') -> WeightMapper:
    """
    Factory function to create a weight mapper.
    
    This is the primary way to create weight mappers for model conversion.
    
    Args:
        source_format: Source model format ("huggingface", "gguf", "onnx")
        target_format: Target format (currently only "forge" is supported)
        
    Returns:
        Configured WeightMapper instance
        
    Raises:
        ValueError: If target_format is not "forge"
        
    Example:
        >>> mapper = create_weight_mapper(source_format="onnx", target_format="forge")
        >>> forge_weights = mapper.map_onnx_to_forge(onnx_weights, config)
    """
    if target_format != 'forge':
        raise ValueError(f"Only 'forge' target format is supported, got: {target_format}")
    
    if source_format not in ['huggingface', 'gguf', 'onnx']:
        logger.warning(
            f"Source format '{source_format}' may not be fully supported. "
            f"Supported formats: huggingface, gguf, onnx"
        )
    
    logger.info(f"Creating WeightMapper: {source_format} → {target_format}")
    return WeightMapper()
