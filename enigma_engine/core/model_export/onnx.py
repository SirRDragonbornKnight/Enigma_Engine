"""
ONNX export provider.

Export models to ONNX format for deployment on edge devices,
mobile, web browsers, and various inference engines.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .base import ExportProvider, ExportResult, ExportStatus

logger = logging.getLogger(__name__)

# Check for ONNX
HAVE_ONNX = False
try:
    HAVE_ONNX = True
except ImportError:
    pass  # Intentionally silent

HAVE_ONNXRUNTIME = False
try:
    HAVE_ONNXRUNTIME = True
except ImportError:
    pass  # Intentionally silent


class ONNXProvider(ExportProvider):
    """
    Export models to ONNX format.
    
    ONNX (Open Neural Network Exchange) enables:
    - Cross-platform deployment
    - Mobile/edge inference (ONNX Runtime, CoreML, TensorRT)
    - Web deployment (ONNX.js, WebNN)
    - Optimized inference on various hardware
    
    Requirements:
        pip install onnx onnxruntime
    
    Usage:
        provider = ONNXProvider()
        result = provider.export(
            "my_model",
            output_dir="./onnx_model",
            opset_version=14,
            optimize=True
        )
    """
    
    NAME = "onnx"
    DESCRIPTION = "Export to ONNX - cross-platform model deployment"
    REQUIRES_AUTH = False
    AUTH_ENV_VAR = None
    SUPPORTED_FORMATS = ["onnx"]
    WEBSITE = "https://onnx.ai"
    
    def _build_model(self, config: dict[str, Any]) -> torch.nn.Module:
        """Build a Enigma AI Engine model from config for export."""
        from ..model import Forge, ForgeConfig

        # Convert to ForgeConfig
        model_config = ForgeConfig.from_dict(config)
        model = Forge(config=model_config)
        return model
    
    def export(
        self,
        model_name: str,
        output_dir: Optional[str] = None,
        opset_version: int = 14,
        optimize: bool = True,
        dynamic_axes: bool = True,
        **kwargs
    ) -> ExportResult:
        """
        Export to ONNX format.
        
        Args:
            model_name: Enigma AI Engine model name
            output_dir: Output directory
            opset_version: ONNX opset version (14+ recommended)
            optimize: Apply ONNX optimizations
            dynamic_axes: Enable dynamic batch/sequence dimensions
        """
        if not HAVE_ONNX:
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message="onnx not installed. Run: pip install onnx onnxruntime"
            )
        
        try:
            model_path = self._get_model_path(model_name)
            config = self._load_config(model_path)
            metadata = self._load_metadata(model_path)
            
            if not output_dir:
                output_dir = str(self.models_dir / f"{model_name}_onnx")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Build and load model
            model = self._build_model(config)
            
            weights_path = model_path / "weights.pth"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
            
            model.eval()
            
            # Create dummy input
            batch_size = 1
            seq_len = min(config.get("max_seq_len", 512), 512)  # Limit for export
            vocab_size = config.get("vocab_size", 8000)
            
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Configure dynamic axes
            dynamic_axes_config = None
            if dynamic_axes:
                dynamic_axes_config = {
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                }
            
            # Export to ONNX
            onnx_path = output_path / "model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes_config,
            )
            
            logger.info(f"Exported ONNX model to {onnx_path}")
            
            # Optimize if requested
            if optimize and HAVE_ONNXRUNTIME:
                try:
                    import onnxruntime as ort
                    
                    optimized_path = output_path / "model_optimized.onnx"
                    
                    # Use ONNX Runtime optimization
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess_options.optimized_model_filepath = str(optimized_path)
                    
                    # Create session to trigger optimization
                    ort.InferenceSession(str(onnx_path), sess_options)
                    
                    if optimized_path.exists():
                        logger.info(f"Created optimized model: {optimized_path}")
                except Exception as e:
                    logger.warning(f"Optimization failed (non-critical): {e}")
            
            # Verify the model
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verified successfully")
            except Exception as e:
                logger.warning(f"ONNX verification warning: {e}")
            
            # Save config for reference
            with open(output_path / "config.json", "w") as f:
                json.dump({
                    "forge_config": config,
                    "onnx_config": {
                        "opset_version": opset_version,
                        "dynamic_axes": dynamic_axes,
                        "optimized": optimize,
                    }
                }, f, indent=2)
            
            # Create README
            readme = f"""# {model_name} (ONNX)

Enigma AI Engine model exported to ONNX format.

## Files

- `model.onnx` - ONNX model
- `model_optimized.onnx` - Optimized model (if available)
- `config.json` - Model configuration

## Usage

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Prepare input (token IDs)
input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

# Run inference
outputs = session.run(None, {{"input_ids": input_ids}})
logits = outputs[0]
```

### JavaScript (ONNX.js)

```javascript
const ort = require('onnxruntime-web');

const session = await ort.InferenceSession.create('model.onnx');
const inputIds = new ort.Tensor('int64', [1n, 2n, 3n, 4n, 5n], [1, 5]);
const results = await session.run({{ input_ids: inputIds }});
```

## Deployment Options

- **ONNX Runtime** - CPU/GPU inference
- **TensorRT** - NVIDIA GPU optimization
- **CoreML** - Apple devices
- **OpenVINO** - Intel hardware
- **ONNX.js / WebNN** - Web browsers
"""
            with open(output_path / "README.md", "w") as f:
                f.write(readme)
            
            return ExportResult(
                status=ExportStatus.SUCCESS,
                provider=self.NAME,
                model_name=model_name,
                local_path=str(output_path),
                message=f"ONNX model exported to {output_path}",
                details={
                    "opset_version": opset_version,
                    "optimized": optimize,
                    "dynamic_axes": dynamic_axes,
                }
            )
            
        except Exception as e:
            logger.exception("ONNX export failed")
            return ExportResult(
                status=ExportStatus.FAILED,
                provider=self.NAME,
                model_name=model_name,
                message=str(e)
            )
