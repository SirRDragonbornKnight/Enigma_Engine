"""
ONNX Model Loader
=================

Load and convert ONNX format models to Forge architecture.
Enables cross-platform deployment and inference optimization.

Usage:
    from forge_ai.core.onnx_loader import load_onnx_model
    
    model = load_onnx_model("model.onnx")
    # Returns a Forge model with loaded weights
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

# Check for required dependencies
HAVE_ONNX = False
HAVE_TORCH = False

try:
    import onnx
    from onnx import numpy_helper
    HAVE_ONNX = True
except ImportError:
    logger.warning("onnx not available - ONNX loading disabled")

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    logger.warning("torch not available - ONNX loading disabled")


def extract_onnx_weights(onnx_model_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Extract weights from ONNX model file.
    
    Args:
        onnx_model_path: Path to .onnx file
        
    Returns:
        Dictionary mapping parameter names to PyTorch tensors
    """
    if not HAVE_ONNX:
        raise RuntimeError(
            "ONNX loading requires onnx library. "
            "Install with: pip install onnx"
        )
    
    if not HAVE_TORCH:
        raise RuntimeError(
            "ONNX loading requires torch library. "
            "Install with: pip install torch"
        )
    
    logger.info(f"Loading ONNX model from: {onnx_model_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_model_path))
    
    # Check model is valid
    try:
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model structure is valid")
    except Exception as e:
        logger.warning(f"ONNX model validation warning: {e}")
    
    # Extract initializers (weights)
    weights = {}
    
    for initializer in onnx_model.graph.initializer:
        # Convert ONNX tensor to numpy array
        np_array = numpy_helper.to_array(initializer)
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(np_array)
        
        # Store with parameter name
        weights[initializer.name] = tensor
        logger.debug(f"Extracted weight: {initializer.name}, shape: {tensor.shape}")
    
    logger.info(f"Extracted {len(weights)} weight tensors from ONNX model")
    
    return weights


def infer_config_from_onnx(onnx_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer model configuration from ONNX weights.
    
    Args:
        onnx_weights: Dictionary of ONNX weight tensors
        
    Returns:
        Dictionary with inferred config parameters
    """
    config = {
        'vocab_size': None,
        'dim': None,
        'n_layers': 0,
        'n_heads': None,
    }
    
    # Try to infer vocabulary size from embedding or output layer
    for name, tensor in onnx_weights.items():
        if 'embed' in name.lower() and tensor.dim() == 2:
            config['vocab_size'] = tensor.shape[0]
            config['dim'] = tensor.shape[1]
            logger.info(f"Inferred vocab_size={config['vocab_size']}, dim={config['dim']}")
            break
        elif ('lm_head' in name.lower() or 'output' in name.lower()) and tensor.dim() == 2:
            config['vocab_size'] = tensor.shape[0]
            config['dim'] = tensor.shape[1]
            logger.info(f"Inferred vocab_size={config['vocab_size']}, dim={config['dim']}")
            break
    
    # Infer number of layers by counting layer-specific weights
    layer_indices = set()
    for name in onnx_weights.keys():
        # Look for patterns like "layer.0", "blocks.0", etc.
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:  # Avoid false positives
                layer_indices.add(int(part))
    
    if layer_indices:
        config['n_layers'] = max(layer_indices) + 1
        logger.info(f"Inferred n_layers={config['n_layers']}")
    
    # Try to infer number of heads from attention projection shapes
    for name, tensor in onnx_weights.items():
        if ('attn' in name.lower() or 'attention' in name.lower()) and tensor.dim() == 2:
            if config['dim'] and tensor.shape[0] == config['dim']:
                # Assume standard multi-head attention
                if tensor.shape[1] % config['dim'] == 0:
                    config['n_heads'] = tensor.shape[1] // config['dim']
                    if config['n_heads'] > 1:
                        logger.info(f"Inferred n_heads={config['n_heads']}")
                        break
    
    return config


def load_onnx_model(
    onnx_model_path: Union[str, Path],
    config: Optional[Any] = None,
    **kwargs
) -> 'Forge':
    """
    Load an ONNX model and convert it to Forge format.
    
    Steps:
    1. Load ONNX model file
    2. Extract weights from ONNX graph
    3. Infer or use provided Forge config
    4. Map ONNX weights to Forge layers
    5. Create Forge model and load weights
    
    Args:
        onnx_model_path: Path to .onnx file
        config: Optional ForgeConfig. If None, will try to infer from ONNX model
        **kwargs: Additional arguments
        
    Returns:
        Forge model with loaded weights
        
    Raises:
        RuntimeError: If required dependencies not installed
        FileNotFoundError: If model file not found
        ValueError: If model structure incompatible
    """
    if not HAVE_ONNX or not HAVE_TORCH:
        raise RuntimeError(
            "ONNX loading requires onnx and torch libraries. "
            "Install with: pip install onnx torch"
        )
    
    # Import here to avoid circular imports
    from .model import Forge, ForgeConfig
    from .weight_mapping import WeightMapper
    
    onnx_model_path = Path(onnx_model_path)
    
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    
    # Extract weights from ONNX
    logger.info("Extracting weights from ONNX model...")
    onnx_weights = extract_onnx_weights(onnx_model_path)
    
    if not onnx_weights:
        raise ValueError("No weights found in ONNX model")
    
    # Infer or validate config
    if config is None:
        logger.info("No config provided, inferring from ONNX model...")
        inferred_config = infer_config_from_onnx(onnx_weights)
        
        # Validate inferred config
        if inferred_config['vocab_size'] is None or inferred_config['dim'] is None:
            raise ValueError(
                "Could not infer model configuration from ONNX model. "
                "Please provide a ForgeConfig explicitly."
            )
        
        # Use inferred config with defaults for missing values
        config = ForgeConfig(
            vocab_size=inferred_config['vocab_size'],
            dim=inferred_config['dim'],
            n_layers=inferred_config['n_layers'] or 4,
            n_heads=inferred_config['n_heads'] or 8,
            **kwargs
        )
        logger.info(f"Created config: {config}")
    elif not isinstance(config, ForgeConfig):
        # Convert dict to ForgeConfig
        config = ForgeConfig(**config)
    
    # Create Forge model
    logger.info("Creating Forge model...")
    forge_model = Forge(config=config)
    
    # Map ONNX weights to Forge format
    logger.info("Mapping ONNX weights to Forge format...")
    mapper = WeightMapper()
    forge_weights = mapper.map_onnx_to_forge(onnx_weights, config)
    
    # Load weights into model (with strict=False to allow partial loading)
    logger.info("Loading weights into Forge model...")
    try:
        missing_keys, unexpected_keys = forge_model.load_state_dict(forge_weights, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            logger.info("Missing keys will be randomly initialized")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise ValueError(f"Weight loading failed: {e}")
    
    logger.info("ONNX model successfully loaded into Forge format")
    
    # Set to eval mode by default
    forge_model.eval()
    
    return forge_model


def validate_onnx_model(onnx_model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an ONNX model and return information about it.
    
    Args:
        onnx_model_path: Path to .onnx file
        
    Returns:
        Dictionary with model information:
        - valid: Whether model is valid
        - num_weights: Number of weight tensors
        - input_names: List of input names
        - output_names: List of output names
        - inferred_config: Inferred configuration
    """
    if not HAVE_ONNX:
        return {
            'valid': False,
            'error': 'onnx library not installed'
        }
    
    try:
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        
        # Extract info
        weights = extract_onnx_weights(onnx_model_path)
        inferred_config = infer_config_from_onnx(weights)
        
        input_names = [inp.name for inp in onnx_model.graph.input]
        output_names = [out.name for out in onnx_model.graph.output]
        
        return {
            'valid': True,
            'num_weights': len(weights),
            'input_names': input_names,
            'output_names': output_names,
            'inferred_config': inferred_config,
            'file_size_mb': Path(onnx_model_path).stat().st_size / (1024 * 1024)
        }
    
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test ONNX loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python onnx_loader.py <path_to_onnx_model>")
        print("\nThis will validate the ONNX model and show its structure.")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print(f"Validating ONNX model: {model_path}")
    print("=" * 60)
    
    info = validate_onnx_model(model_path)
    
    if info['valid']:
        print("[OK] Model is valid")
        print(f"\nModel Information:")
        print(f"  File size: {info['file_size_mb']:.2f} MB")
        print(f"  Number of weights: {info['num_weights']}")
        print(f"  Inputs: {info['input_names']}")
        print(f"  Outputs: {info['output_names']}")
        print(f"\nInferred Configuration:")
        for key, value in info['inferred_config'].items():
            print(f"  {key}: {value}")
        
        print("\n[INFO] You can load this model with:")
        print(f"  from forge_ai.core.onnx_loader import load_onnx_model")
        print(f"  model = load_onnx_model('{model_path}')")
    else:
        print(f"[ERROR] Model validation failed: {info['error']}")
