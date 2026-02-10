"""
NPU/TPU Hardware Support

Support for neural processing units and tensor processing units.

FILE: enigma_engine/core/hardware_accelerators.py
TYPE: Core
MAIN CLASSES: AcceleratorManager, NPUDevice, TPUDevice, CoralDevice
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for accelerator libraries
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pycoral.adapters import common
    from pycoral.utils.edgetpu import make_interpreter
    HAS_CORAL = True
except ImportError:
    HAS_CORAL = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class AcceleratorType(Enum):
    """Hardware accelerator types."""
    CPU = auto()
    CUDA = auto()
    MPS = auto()  # Apple Metal
    NPU = auto()
    TPU = auto()
    CORAL = auto()  # Google Coral Edge TPU
    QUALCOMM_NPU = auto()
    INTEL_NPU = auto()
    AMD_XIPU = auto()


class DeviceCapability(Enum):
    """Device capabilities."""
    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    INT8 = auto()
    INT4 = auto()
    TENSOR_CORES = auto()
    SPARSE_COMPUTE = auto()


@dataclass
class DeviceInfo:
    """Information about a device."""
    device_type: AcceleratorType
    name: str
    memory_gb: float
    compute_capability: str = ""
    capabilities: list[DeviceCapability] = field(default_factory=list)
    driver_version: str = ""
    temperature: float = 0.0
    utilization: float = 0.0


class AcceleratorDevice(ABC):
    """Abstract base for accelerator devices."""
    
    @abstractmethod
    def get_info(self) -> DeviceInfo:
        """Get device information."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if device is available."""
    
    @abstractmethod
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        """Allocate memory on device."""
    
    @abstractmethod
    def free_memory(self, handle: Any):
        """Free device memory."""
    
    @abstractmethod
    def transfer_to_device(self, data: Any) -> Any:
        """Transfer data to device."""
    
    @abstractmethod
    def transfer_to_host(self, data: Any) -> Any:
        """Transfer data back to host."""
    
    @abstractmethod
    def compile_model(self, model: Any) -> Any:
        """Compile model for device."""
    
    @abstractmethod
    def run_inference(self, model: Any, inputs: Any) -> Any:
        """Run inference on device."""


class CUDADevice(AcceleratorDevice):
    """NVIDIA CUDA device."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = f"cuda:{device_id}" if HAS_TORCH else None
    
    def get_info(self) -> DeviceInfo:
        """Get CUDA device info."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return DeviceInfo(
                device_type=AcceleratorType.CUDA,
                name="unavailable",
                memory_gb=0
            )
        
        props = torch.cuda.get_device_properties(self.device_id)
        memory_gb = props.total_memory / (1024**3)
        
        capabilities = [DeviceCapability.FP32, DeviceCapability.FP16]
        if props.major >= 7:
            capabilities.append(DeviceCapability.TENSOR_CORES)
        if props.major >= 8:
            capabilities.append(DeviceCapability.BF16)
        
        return DeviceInfo(
            device_type=AcceleratorType.CUDA,
            name=props.name,
            memory_gb=memory_gb,
            compute_capability=f"{props.major}.{props.minor}",
            capabilities=capabilities,
            driver_version=torch.version.cuda or ""
        )
    
    def is_available(self) -> bool:
        """Check CUDA availability."""
        return HAS_TORCH and torch.cuda.is_available()
    
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        """Allocate CUDA memory."""
        if not self.is_available():
            return None
        return torch.empty(size_bytes, dtype=torch.uint8, device=self.device)
    
    def free_memory(self, handle: Any):
        """Free CUDA memory."""
        del handle
        if self.is_available():
            torch.cuda.empty_cache()
    
    def transfer_to_device(self, data: Any) -> Any:
        """Transfer to GPU."""
        if hasattr(data, "to"):
            return data.to(self.device)
        return data
    
    def transfer_to_host(self, data: Any) -> Any:
        """Transfer to CPU."""
        if hasattr(data, "cpu"):
            return data.cpu()
        return data
    
    def compile_model(self, model: Any) -> Any:
        """Compile for CUDA (torch.compile if available)."""
        if not self.is_available():
            return model
        
        model = model.to(self.device)
        
        # Use torch.compile for PyTorch 2.0+
        if hasattr(torch, "compile"):
            try:
                return torch.compile(model)
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        return model
    
    def run_inference(self, model: Any, inputs: Any) -> Any:
        """Run on CUDA."""
        inputs = self.transfer_to_device(inputs)
        
        with torch.no_grad():
            return model(inputs)


class MPSDevice(AcceleratorDevice):
    """Apple Metal Performance Shaders device."""
    
    def __init__(self):
        self.device = "mps" if HAS_TORCH else None
    
    def get_info(self) -> DeviceInfo:
        """Get MPS device info."""
        if not self.is_available():
            return DeviceInfo(
                device_type=AcceleratorType.MPS,
                name="unavailable",
                memory_gb=0
            )
        
        # MPS doesn't expose detailed info like CUDA
        return DeviceInfo(
            device_type=AcceleratorType.MPS,
            name="Apple Metal",
            memory_gb=0,  # Shared memory
            capabilities=[DeviceCapability.FP32, DeviceCapability.FP16]
        )
    
    def is_available(self) -> bool:
        """Check MPS availability."""
        return HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        """Allocate MPS memory."""
        if not self.is_available():
            return None
        return torch.empty(size_bytes, dtype=torch.uint8, device="mps")
    
    def free_memory(self, handle: Any):
        """Free MPS memory."""
        del handle
    
    def transfer_to_device(self, data: Any) -> Any:
        """Transfer to MPS."""
        if hasattr(data, "to"):
            return data.to("mps")
        return data
    
    def transfer_to_host(self, data: Any) -> Any:
        """Transfer to CPU."""
        if hasattr(data, "cpu"):
            return data.cpu()
        return data
    
    def compile_model(self, model: Any) -> Any:
        """Compile for MPS."""
        if not self.is_available():
            return model
        return model.to("mps")
    
    def run_inference(self, model: Any, inputs: Any) -> Any:
        """Run on MPS."""
        inputs = self.transfer_to_device(inputs)
        
        with torch.no_grad():
            return model(inputs)


class CoralDevice(AcceleratorDevice):
    """Google Coral Edge TPU device."""
    
    def __init__(self, device_path: str = None):
        self.device_path = device_path
        self._interpreter = None
    
    def get_info(self) -> DeviceInfo:
        """Get Coral device info."""
        return DeviceInfo(
            device_type=AcceleratorType.CORAL,
            name="Coral Edge TPU",
            memory_gb=0.008,  # ~8MB SRAM
            capabilities=[DeviceCapability.INT8]
        )
    
    def is_available(self) -> bool:
        """Check Coral availability."""
        if not HAS_CORAL:
            return False
        
        # Check for Edge TPU device
        try:
            import subprocess
            result = subprocess.run(
                ["ls", "/dev/bus/usb"],
                capture_output=True,
                timeout=5
            )
            # Look for Coral vendor ID
            return True  # Simplified check
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
    
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        """Coral manages memory internally."""
        return None
    
    def free_memory(self, handle: Any):
        """Coral manages memory internally."""
    
    def transfer_to_device(self, data: Any) -> Any:
        """Prepare input for Coral."""
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.astype(np.uint8)
        return data
    
    def transfer_to_host(self, data: Any) -> Any:
        """Get output from Coral."""
        return data
    
    def compile_model(self, model: Any) -> Any:
        """Load EdgeTPU model."""
        if not HAS_CORAL:
            raise ImportError("pycoral required for Coral support")
        
        # Model should be a path to EdgeTPU-compiled TFLite model
        if isinstance(model, str):
            self._interpreter = make_interpreter(model)
            self._interpreter.allocate_tensors()
            return self._interpreter
        
        return model
    
    def run_inference(self, model: Any, inputs: Any) -> Any:
        """Run on Coral."""
        if not HAS_CORAL:
            raise ImportError("pycoral required")
        
        interpreter = model if model else self._interpreter
        
        common.set_input(interpreter, inputs)
        interpreter.invoke()
        
        return common.output_tensor(interpreter, 0)


class TPUDevice(AcceleratorDevice):
    """Google Cloud TPU device."""
    
    def __init__(self, tpu_name: str = None):
        self.tpu_name = tpu_name
        self._resolver = None
        self._strategy = None
    
    def get_info(self) -> DeviceInfo:
        """Get TPU device info."""
        if not self.is_available():
            return DeviceInfo(
                device_type=AcceleratorType.TPU,
                name="unavailable",
                memory_gb=0
            )
        
        return DeviceInfo(
            device_type=AcceleratorType.TPU,
            name=self.tpu_name or "TPU",
            memory_gb=16,  # TPU v3 has 16GB HBM per core
            capabilities=[
                DeviceCapability.FP32,
                DeviceCapability.BF16,
                DeviceCapability.INT8
            ]
        )
    
    def is_available(self) -> bool:
        """Check TPU availability."""
        if not HAS_TF:
            return False
        
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                self.tpu_name
            )
            return True
        except (ValueError, RuntimeError, Exception):
            return False
    
    def _init_tpu(self):
        """Initialize TPU."""
        if self._resolver is None:
            self._resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                self.tpu_name
            )
            tf.config.experimental_connect_to_cluster(self._resolver)
            tf.tpu.experimental.initialize_tpu_system(self._resolver)
            self._strategy = tf.distribute.TPUStrategy(self._resolver)
    
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        """TPU manages memory internally."""
        return None
    
    def free_memory(self, handle: Any):
        """TPU manages memory internally."""
    
    def transfer_to_device(self, data: Any) -> Any:
        """Convert to TPU-compatible tensor."""
        return tf.constant(data)
    
    def transfer_to_host(self, data: Any) -> Any:
        """Convert back to numpy."""
        if hasattr(data, "numpy"):
            return data.numpy()
        return data
    
    def compile_model(self, model: Any) -> Any:
        """Compile for TPU."""
        if not self.is_available():
            return model
        
        self._init_tpu()
        
        with self._strategy.scope():
            # Recreate model under TPU strategy
            return model
    
    def run_inference(self, model: Any, inputs: Any) -> Any:
        """Run on TPU."""
        if not self.is_available():
            raise RuntimeError("TPU not available")
        
        @tf.function
        def inference_step(x):
            return model(x)
        
        inputs = self.transfer_to_device(inputs)
        return inference_step(inputs)


class QualcommNPUDevice(AcceleratorDevice):
    """Qualcomm Hexagon NPU device (mobile/edge)."""
    
    def __init__(self):
        self._ctx = None
    
    def get_info(self) -> DeviceInfo:
        """Get NPU device info."""
        return DeviceInfo(
            device_type=AcceleratorType.QUALCOMM_NPU,
            name="Qualcomm Hexagon DSP",
            memory_gb=0,  # Uses system memory
            capabilities=[DeviceCapability.INT8, DeviceCapability.INT4]
        )
    
    def is_available(self) -> bool:
        """Check Qualcomm NPU availability."""
        try:
            import qaic  # Qualcomm AI Engine Direct
            return True
        except ImportError:
            return False
    
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        return None
    
    def free_memory(self, handle: Any):
        pass
    
    def transfer_to_device(self, data: Any) -> Any:
        return data
    
    def transfer_to_host(self, data: Any) -> Any:
        return data
    
    def compile_model(self, model: Any) -> Any:
        """Compile using SNPE or QNN."""
        logger.info("Qualcomm NPU compilation requires SNPE/QNN toolkit")
        return model
    
    def run_inference(self, model: Any, inputs: Any) -> Any:
        return model(inputs)


class IntelNPUDevice(AcceleratorDevice):
    """Intel Neural Processing Unit (Meteor Lake+)."""
    
    def __init__(self):
        pass
    
    def get_info(self) -> DeviceInfo:
        """Get Intel NPU info."""
        return DeviceInfo(
            device_type=AcceleratorType.INTEL_NPU,
            name="Intel NPU",
            memory_gb=0,
            capabilities=[DeviceCapability.INT8]
        )
    
    def is_available(self) -> bool:
        """Check Intel NPU availability."""
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            return "NPU" in devices
        except (ImportError, RuntimeError, Exception):
            return False
    
    def allocate_memory(self, size_bytes: int) -> Optional[Any]:
        return None
    
    def free_memory(self, handle: Any):
        pass
    
    def transfer_to_device(self, data: Any) -> Any:
        return data
    
    def transfer_to_host(self, data: Any) -> Any:
        return data
    
    def compile_model(self, model: Any) -> Any:
        """Compile using OpenVINO."""
        try:
            from openvino.runtime import Core
            
            core = Core()
            compiled = core.compile_model(model, "NPU")
            return compiled
        except ImportError:
            logger.warning("OpenVINO required for Intel NPU")
            return model
    
    def run_inference(self, model: Any, inputs: Any) -> Any:
        """Run on Intel NPU via OpenVINO."""
        if hasattr(model, "infer"):
            return model.infer(inputs)
        return model(inputs)


class AcceleratorManager:
    """Manages multiple hardware accelerators."""
    
    def __init__(self):
        self.devices: dict[str, AcceleratorDevice] = {}
        self.primary_device: Optional[AcceleratorDevice] = None
        self._discover_devices()
    
    def _discover_devices(self):
        """Discover available accelerators."""
        # CUDA
        if HAS_TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = CUDADevice(i)
                self.devices[f"cuda:{i}"] = device
                if self.primary_device is None:
                    self.primary_device = device
        
        # MPS (Apple)
        mps = MPSDevice()
        if mps.is_available():
            self.devices["mps"] = mps
            if self.primary_device is None:
                self.primary_device = mps
        
        # Coral
        coral = CoralDevice()
        if coral.is_available():
            self.devices["coral"] = coral
        
        # Intel NPU
        intel_npu = IntelNPUDevice()
        if intel_npu.is_available():
            self.devices["intel_npu"] = intel_npu
        
        # TPU (Cloud)
        tpu = TPUDevice()
        if tpu.is_available():
            self.devices["tpu"] = tpu
        
        logger.info(f"Discovered {len(self.devices)} accelerator(s)")
    
    def get_device(self, name: str) -> Optional[AcceleratorDevice]:
        """Get device by name."""
        return self.devices.get(name)
    
    def get_best_device(self) -> Optional[AcceleratorDevice]:
        """Get best available device."""
        return self.primary_device
    
    def list_devices(self) -> list[DeviceInfo]:
        """List all available devices."""
        return [device.get_info() for device in self.devices.values()]
    
    def select_device_for_model(
        self,
        model_size_gb: float,
        requires_fp16: bool = False,
        requires_int8: bool = False
    ) -> Optional[AcceleratorDevice]:
        """Select appropriate device for model."""
        candidates = []
        
        for name, device in self.devices.items():
            info = device.get_info()
            
            # Check memory
            if info.memory_gb < model_size_gb:
                continue
            
            # Check precision support
            if requires_fp16 and DeviceCapability.FP16 not in info.capabilities:
                continue
            if requires_int8 and DeviceCapability.INT8 not in info.capabilities:
                continue
            
            candidates.append((name, device, info))
        
        if not candidates:
            return None
        
        # Prefer devices with tensor cores
        for name, device, info in candidates:
            if DeviceCapability.TENSOR_CORES in info.capabilities:
                return device
        
        # Fall back to first candidate
        return candidates[0][1]


# Convenience functions
def get_accelerator_manager() -> AcceleratorManager:
    """Get global accelerator manager."""
    return AcceleratorManager()


def get_best_device() -> Optional[AcceleratorDevice]:
    """Get best available accelerator."""
    manager = AcceleratorManager()
    return manager.get_best_device()


def auto_device(model: Any) -> Any:
    """Automatically move model to best device."""
    device = get_best_device()
    if device:
        return device.compile_model(model)
    return model


def list_accelerators() -> list[dict[str, Any]]:
    """List available accelerators."""
    manager = AcceleratorManager()
    
    return [
        {
            "type": info.device_type.name,
            "name": info.name,
            "memory_gb": info.memory_gb,
            "capabilities": [c.name for c in info.capabilities]
        }
        for info in manager.list_devices()
    ]
