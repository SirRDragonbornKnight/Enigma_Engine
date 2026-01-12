"""
Model Quantization - Run larger models on smaller hardware.

INT8 quantization gives ~2x speedup with minimal quality loss.
INT4 quantization gives ~4x speedup with some quality loss.
"""

def quantize_model(model, dtype: str = "int8"):
    """
    Quantize model weights for faster inference.
    
    Args:
        model: Enigma model
        dtype: "int8" or "int4"
    
    Returns:
        Quantized model
    """
    import torch
    
    if dtype == "int8":
        return torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
    elif dtype == "int4":
        try:
            import bitsandbytes as bnb
            # Use bitsandbytes for INT4
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    setattr(model, name, bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    ))
            return model
        except ImportError:
            print("Install bitsandbytes for INT4: pip install bitsandbytes")
            return model
    
    return model


def load_quantized(path: str, dtype: str = "int8"):
    """Load a model and quantize it."""
    from .model import create_model
    from .model_registry import safe_load_weights
    
    state_dict = safe_load_weights(path, map_location="cpu")
    model = create_model("auto")
    model.load_state_dict(state_dict, strict=False)
    
    return quantize_model(model, dtype)
