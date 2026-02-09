"""
GPU Container Support

NVIDIA and ROCm container configuration for GPU acceleration.
Generates Dockerfiles and runtime configurations for GPU support.

FILE: enigma_engine/deploy/gpu_container.py
TYPE: Deployment
MAIN CLASSES: GPUConfig, NvidiaContainer, ROCmContainer
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"


class CUDAVersion(Enum):
    """CUDA version options."""
    CUDA_11_8 = "11.8"
    CUDA_12_0 = "12.0"
    CUDA_12_1 = "12.1"
    CUDA_12_2 = "12.2"
    CUDA_12_3 = "12.3"


class ROCmVersion(Enum):
    """ROCm version options."""
    ROCM_5_6 = "5.6"
    ROCM_5_7 = "5.7"
    ROCM_6_0 = "6.0"


@dataclass
class GPUConfig:
    """GPU container configuration."""
    vendor: GPUVendor = GPUVendor.NVIDIA
    cuda_version: CUDAVersion = CUDAVersion.CUDA_12_1
    rocm_version: ROCmVersion = ROCmVersion.ROCM_6_0
    python_version: str = "3.10"
    torch_version: str = "2.1.0"
    enable_cudnn: bool = True
    enable_tensorrt: bool = False
    memory_limit_gb: int = 0  # 0 = no limit
    visible_devices: str = "all"  # or "0,1" for specific GPUs


class GPUContainerGenerator:
    """Base class for GPU container generation."""
    
    def __init__(self, config: GPUConfig, output_dir: Path):
        self._config = config
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self) -> dict[str, Path]:
        """Generate all GPU-related files."""
        raise NotImplementedError


class NvidiaContainer(GPUContainerGenerator):
    """NVIDIA GPU container generator."""
    
    def generate(self) -> dict[str, Path]:
        """Generate NVIDIA container files."""
        files = {}
        
        files["Dockerfile.gpu"] = self._generate_dockerfile()
        files["docker-compose.gpu.yml"] = self._generate_compose()
        files[".devcontainer/devcontainer.json"] = self._generate_devcontainer()
        
        return files
    
    def _generate_dockerfile(self) -> Path:
        """Generate GPU-enabled Dockerfile."""
        cuda = self._config.cuda_version.value
        python = self._config.python_version
        
        # Select appropriate base image
        if self._config.enable_cudnn:
            base_tag = f"{cuda}-cudnn8-devel-ubuntu22.04"
        else:
            base_tag = f"{cuda}-base-ubuntu22.04"
        
        dockerfile = f'''# Enigma AI Engine GPU Container (NVIDIA CUDA)
# Auto-generated GPU configuration

FROM nvidia/cuda:{base_tag}

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \\
    python{python} \\
    python{python}-pip \\
    python{python}-venv \\
    python{python}-dev \\
    git \\
    curl \\
    wget \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set Python symlinks
RUN update-alternatives --install /usr/bin/python python /usr/bin/python{python} 1 \\
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${{CUDA_HOME}}/bin:${{PATH}}
ENV LD_LIBRARY_PATH=${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH}}

# cuDNN library path
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu
'''

        # TensorRT support
        if self._config.enable_tensorrt:
            dockerfile += '''
# Install TensorRT
RUN apt-get update && apt-get install -y \\
    libnvinfer8 \\
    libnvinfer-plugin8 \\
    python3-libnvinfer \\
    && rm -rf /var/lib/apt/lists/*
'''

        dockerfile += f'''
# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \\
    torch=={self._config.torch_version}+cu{cuda.replace(".", "")} \\
    torchvision \\
    torchaudio \\
    --index-url https://download.pytorch.org/whl/cu{cuda.replace(".", "")}

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment
ENV FORGE_GPU=nvidia
ENV NVIDIA_VISIBLE_DEVICES={self._config.visible_devices}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\
    CMD python -c "import torch; assert torch.cuda.is_available()"

# Entry point
CMD ["python", "run.py", "--serve"]
'''
        
        path = self._output_dir / "Dockerfile.gpu"
        path.write_text(dockerfile)
        return path
    
    def _generate_compose(self) -> Path:
        """Generate Docker Compose for GPU."""
        cuda = self._config.cuda_version.value
        
        compose = f'''version: '3.8'

services:
  forge-ai-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: forge-ai-gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES={self._config.visible_devices}
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VERSION={cuda}
      - FORGE_GPU=nvidia
'''
        
        if self._config.memory_limit_gb > 0:
            compose += f'''    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
        limits:
          memory: {self._config.memory_limit_gb}G
'''
        else:
            compose += '''    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
'''
        
        compose += '''    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
'''
        
        path = self._output_dir / "docker-compose.gpu.yml"
        path.write_text(compose)
        return path
    
    def _generate_devcontainer(self) -> Path:
        """Generate VS Code devcontainer config."""
        cuda = self._config.cuda_version.value
        
        devcontainer = {
            "name": "Enigma AI Engine GPU Dev",
            "build": {
                "dockerfile": "../Dockerfile.gpu",
                "context": ".."
            },
            "runArgs": [
                "--gpus", "all",
                "--shm-size", "8g"
            ],
            "containerEnv": {
                "NVIDIA_VISIBLE_DEVICES": self._config.visible_devices,
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"
            },
            "customizations": {
                "vscode": {
                    "extensions": [
                        "ms-python.python",
                        "ms-python.vscode-pylance",
                        "ms-toolsai.jupyter"
                    ]
                }
            },
            "postCreateCommand": "pip install -e ."
        }
        
        devcontainer_dir = self._output_dir / ".devcontainer"
        devcontainer_dir.mkdir(exist_ok=True)
        
        import json
        path = devcontainer_dir / "devcontainer.json"
        path.write_text(json.dumps(devcontainer, indent=2))
        return path


class ROCmContainer(GPUContainerGenerator):
    """AMD ROCm GPU container generator."""
    
    def generate(self) -> dict[str, Path]:
        """Generate ROCm container files."""
        files = {}
        
        files["Dockerfile.rocm"] = self._generate_dockerfile()
        files["docker-compose.rocm.yml"] = self._generate_compose()
        
        return files
    
    def _generate_dockerfile(self) -> Path:
        """Generate ROCm-enabled Dockerfile."""
        rocm = self._config.rocm_version.value
        python = self._config.python_version
        
        dockerfile = f'''# Enigma AI Engine GPU Container (AMD ROCm)
# Auto-generated GPU configuration

FROM rocm/pytorch:rocm{rocm}_ubuntu22.04_py{python}_pytorch_{self._config.torch_version}

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Install additional dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    wget \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# ROCm environment
ENV ROCM_HOME=/opt/rocm
ENV PATH=${{ROCM_HOME}}/bin:${{PATH}}
ENV LD_LIBRARY_PATH=${{ROCM_HOME}}/lib:${{LD_LIBRARY_PATH}}

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment
ENV FORGE_GPU=amd
ENV HIP_VISIBLE_DEVICES=all

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\
    CMD python -c "import torch; assert torch.cuda.is_available()"

CMD ["python", "run.py", "--serve"]
'''
        
        path = self._output_dir / "Dockerfile.rocm"
        path.write_text(dockerfile)
        return path
    
    def _generate_compose(self) -> Path:
        """Generate Docker Compose for ROCm."""
        compose = f'''version: '3.8'

services:
  forge-ai-rocm:
    build:
      context: .
      dockerfile: Dockerfile.rocm
    container_name: forge-ai-rocm
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    environment:
      - HIP_VISIBLE_DEVICES=all
      - HSA_OVERRIDE_GFX_VERSION=10.3.0
      - FORGE_GPU=amd
    security_opt:
      - seccomp:unconfined
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
'''
        
        path = self._output_dir / "docker-compose.rocm.yml"
        path.write_text(compose)
        return path


def generate_gpu_configs(output_dir: Path,
                        nvidia_config: GPUConfig = None,
                        rocm_config: GPUConfig = None) -> dict[str, dict[str, Path]]:
    """
    Generate GPU container configs for both vendors.
    
    Args:
        output_dir: Output directory
        nvidia_config: NVIDIA configuration
        rocm_config: ROCm configuration
        
    Returns:
        Dict mapping vendor to generated files
    """
    results = {}
    
    if nvidia_config is None:
        nvidia_config = GPUConfig(vendor=GPUVendor.NVIDIA)
    
    if rocm_config is None:
        rocm_config = GPUConfig(vendor=GPUVendor.AMD)
    
    try:
        nvidia_gen = NvidiaContainer(nvidia_config, output_dir)
        results["nvidia"] = nvidia_gen.generate()
    except Exception as e:
        logger.error(f"NVIDIA config generation failed: {e}")
        results["nvidia"] = {}
    
    try:
        rocm_gen = ROCmContainer(rocm_config, output_dir)
        results["rocm"] = rocm_gen.generate()
    except Exception as e:
        logger.error(f"ROCm config generation failed: {e}")
        results["rocm"] = {}
    
    return results


def detect_gpu() -> Optional[GPUVendor]:
    """
    Detect available GPU vendor.
    
    Returns:
        GPUVendor or None if no GPU found
    """
    try:
        import subprocess

        # Check NVIDIA
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return GPUVendor.NVIDIA
        
        # Check AMD ROCm
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return GPUVendor.AMD
        
    except FileNotFoundError:
        pass
    
    return None


__all__ = [
    'GPUConfig',
    'GPUVendor',
    'CUDAVersion',
    'ROCmVersion',
    'NvidiaContainer',
    'ROCmContainer',
    'generate_gpu_configs',
    'detect_gpu'
]
