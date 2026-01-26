"""
Hardware Profile System - Device-specific configurations.

Provides deployment profiles for different hardware scenarios:
- Raspberry Pi (robot/embedded)
- Phone (avatar only)
- PC Gaming (AI running while gaming)
- Workstation (full power with RTX/high-end GPU)
- Custom (user-defined)

Each profile controls:
- CPU/RAM/GPU limits
- Which modules/tools are enabled
- Model size limits
- Generation capabilities
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import json
import platform


class DeviceType(Enum):
    """Device type classification."""
    RASPBERRY_PI = "raspberry_pi"
    PHONE = "phone"
    PC_GAMING = "pc_gaming"
    WORKSTATION = "workstation"
    CUSTOM = "custom"
    AUTO_DETECT = "auto"


@dataclass
class HardwareProfile:
    """A hardware/deployment profile."""
    name: str
    device_type: str
    description: str
    
    # Resource limits
    max_cpu_percent: int = 50
    max_memory_percent: int = 40
    max_gpu_memory_percent: int = 50
    
    # Model constraints
    max_model_size: str = "small"  # nano, micro, tiny, small, medium, large, xl
    use_quantization: bool = False
    quantization_bits: int = 8
    
    # Enabled features (what this profile supports)
    enable_chat: bool = True
    enable_avatar: bool = True
    enable_voice_input: bool = False
    enable_voice_output: bool = False
    enable_vision: bool = False
    enable_image_gen: bool = False
    enable_code_gen: bool = False
    enable_video_gen: bool = False
    enable_audio_gen: bool = False
    enable_3d_gen: bool = False
    enable_training: bool = False
    enable_tools: bool = True
    
    # Tool restrictions (empty = all allowed, list = only these)
    allowed_tools: List[str] = None
    blocked_tools: List[str] = None
    
    # Networking
    enable_api_server: bool = False
    enable_remote_client: bool = False
    
    # Performance tuning
    batch_size: int = 1
    max_context_length: int = 512
    inference_threads: int = 1
    
    # Priority (for process scheduling)
    process_priority: str = "normal"  # low, normal, high
    
    def __post_init__(self):
        if self.allowed_tools is None:
            self.allowed_tools = []
        if self.blocked_tools is None:
            self.blocked_tools = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareProfile':
        """Create from dictionary."""
        return cls(**data)


# ===== Built-in Profiles =====

BUILTIN_PROFILES: Dict[str, HardwareProfile] = {
    "raspberry_pi": HardwareProfile(
        name="Raspberry Pi / Robot",
        device_type=DeviceType.RASPBERRY_PI.value,
        description="Minimal resources for Pi, embedded systems, or robots. Text-only AI with basic tools.",
        max_cpu_percent=80,  # Pi can use more CPU since it's dedicated
        max_memory_percent=60,
        max_gpu_memory_percent=0,  # No GPU typically
        max_model_size="micro",
        use_quantization=True,
        quantization_bits=4,
        enable_chat=True,
        enable_avatar=False,
        enable_voice_input=True,  # Voice commands for robot
        enable_voice_output=True,  # Speech for robot
        enable_vision=True,  # Camera for robot
        enable_image_gen=False,
        enable_code_gen=False,
        enable_video_gen=False,
        enable_audio_gen=False,
        enable_3d_gen=False,
        enable_training=False,
        enable_tools=True,
        allowed_tools=["robot_control", "camera_capture", "sensor_read", "motor_control"],
        enable_api_server=True,  # Can receive commands from PC
        enable_remote_client=False,
        batch_size=1,
        max_context_length=256,
        inference_threads=2,  # Pi has limited threads
        process_priority="normal",
    ),
    
    "phone": HardwareProfile(
        name="Phone / Tablet (Avatar Only)",
        device_type=DeviceType.PHONE.value,
        description="Avatar display only - connects to PC for AI processing. Minimal local compute.",
        max_cpu_percent=30,
        max_memory_percent=20,
        max_gpu_memory_percent=20,
        max_model_size="nano",  # Very small or no local model
        use_quantization=True,
        quantization_bits=4,
        enable_chat=False,  # Chat handled by remote
        enable_avatar=True,  # Main purpose
        enable_voice_input=True,  # Can send voice to PC
        enable_voice_output=True,  # Can play audio
        enable_vision=False,
        enable_image_gen=False,
        enable_code_gen=False,
        enable_video_gen=False,
        enable_audio_gen=False,
        enable_3d_gen=False,
        enable_training=False,
        enable_tools=False,
        enable_api_server=False,
        enable_remote_client=True,  # Connects to PC
        batch_size=1,
        max_context_length=128,
        inference_threads=1,
        process_priority="low",
    ),
    
    "pc_gaming": HardwareProfile(
        name="PC Gaming Mode",
        device_type=DeviceType.PC_GAMING.value,
        description="AI runs in background while gaming. Limited resources to not impact game performance.",
        max_cpu_percent=15,  # Very low to not affect game
        max_memory_percent=15,
        max_gpu_memory_percent=10,  # Minimal GPU usage
        max_model_size="small",
        use_quantization=True,
        quantization_bits=8,
        enable_chat=True,
        enable_avatar=True,  # Overlay avatar
        enable_voice_input=True,  # Voice commands while gaming
        enable_voice_output=True,
        enable_vision=True,  # Can watch game screen
        enable_image_gen=False,  # Too GPU heavy
        enable_code_gen=False,
        enable_video_gen=False,
        enable_audio_gen=False,
        enable_3d_gen=False,
        enable_training=False,
        enable_tools=True,
        blocked_tools=["image_gen", "video_gen", "3d_gen"],  # Block heavy tools
        enable_api_server=True,  # Serve phone/other devices
        enable_remote_client=False,
        batch_size=1,
        max_context_length=512,
        inference_threads=2,
        process_priority="low",  # Don't compete with game
    ),
    
    "workstation": HardwareProfile(
        name="Workstation / RTX Full Power",
        device_type=DeviceType.WORKSTATION.value,
        description="Maximum performance for high-end workstations with RTX GPUs. All features enabled.",
        max_cpu_percent=70,
        max_memory_percent=60,
        max_gpu_memory_percent=80,
        max_model_size="xl",  # Can run large models
        use_quantization=False,  # Full precision
        quantization_bits=16,
        enable_chat=True,
        enable_avatar=True,
        enable_voice_input=True,
        enable_voice_output=True,
        enable_vision=True,
        enable_image_gen=True,
        enable_code_gen=True,
        enable_video_gen=True,
        enable_audio_gen=True,
        enable_3d_gen=True,
        enable_training=True,
        enable_tools=True,
        enable_api_server=True,
        enable_remote_client=False,
        batch_size=8,
        max_context_length=4096,
        inference_threads=8,
        process_priority="normal",
    ),
    
    "balanced": HardwareProfile(
        name="Balanced (Default)",
        device_type=DeviceType.CUSTOM.value,
        description="Good balance for typical desktop use. Most features enabled with reasonable limits.",
        max_cpu_percent=50,
        max_memory_percent=40,
        max_gpu_memory_percent=50,
        max_model_size="medium",
        use_quantization=False,
        quantization_bits=16,
        enable_chat=True,
        enable_avatar=True,
        enable_voice_input=True,
        enable_voice_output=True,
        enable_vision=True,
        enable_image_gen=True,
        enable_code_gen=True,
        enable_video_gen=False,  # Heavy
        enable_audio_gen=True,
        enable_3d_gen=False,  # Heavy
        enable_training=True,
        enable_tools=True,
        enable_api_server=False,
        enable_remote_client=False,
        batch_size=2,
        max_context_length=2048,
        inference_threads=4,
        process_priority="normal",
    ),
}


class HardwareProfileManager:
    """
    Manages hardware profiles for different deployment scenarios.
    
    Usage:
        manager = HardwareProfileManager()
        manager.set_active_profile("pc_gaming")
        
        # Get current settings
        limits = manager.get_resource_limits()
        features = manager.get_enabled_features()
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize hardware profile manager.
        
        Args:
            storage_path: Path to store custom profiles and settings
        """
        if storage_path is None:
            try:
                from ..config import CONFIG
                storage_path = Path(CONFIG["data_dir"]) / "hardware_profiles.json"
            except ImportError:
                storage_path = Path("data/hardware_profiles.json")
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.custom_profiles: Dict[str, HardwareProfile] = {}
        self.active_profile_name: str = "balanced"
        
        self._load()
    
    def _load(self):
        """Load saved profiles and active profile."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                self.active_profile_name = data.get("active_profile", "balanced")
                
                # Load custom profiles
                for name, profile_data in data.get("custom_profiles", {}).items():
                    self.custom_profiles[name] = HardwareProfile.from_dict(profile_data)
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load hardware profiles: {e}")
    
    def _save(self):
        """Save profiles and active profile."""
        data = {
            "active_profile": self.active_profile_name,
            "custom_profiles": {
                name: profile.to_dict() 
                for name, profile in self.custom_profiles.items()
            }
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_all_profiles(self) -> Dict[str, HardwareProfile]:
        """Get all available profiles (builtin + custom)."""
        profiles = dict(BUILTIN_PROFILES)
        profiles.update(self.custom_profiles)
        return profiles
    
    def get_profile(self, name: str) -> Optional[HardwareProfile]:
        """Get a specific profile by name."""
        if name in BUILTIN_PROFILES:
            return BUILTIN_PROFILES[name]
        return self.custom_profiles.get(name)
    
    def get_active_profile(self) -> HardwareProfile:
        """Get the currently active profile."""
        profile = self.get_profile(self.active_profile_name)
        if profile is None:
            profile = BUILTIN_PROFILES["balanced"]
        return profile
    
    def set_active_profile(self, name: str) -> bool:
        """
        Set the active hardware profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if successful
        """
        if name not in self.get_all_profiles():
            return False
        
        self.active_profile_name = name
        self._save()
        self._apply_profile()
        return True
    
    def _apply_profile(self):
        """Apply the active profile to the system."""
        profile = self.get_active_profile()
        
        # Apply resource limits via ResourceAllocator
        try:
            from .resource_allocator import ResourceAllocator
            allocator = ResourceAllocator()
            
            # Map profile to resource mode
            if profile.max_cpu_percent <= 20:
                allocator.set_mode('minimal')
            elif profile.max_cpu_percent <= 40:
                allocator.set_mode('balanced')
            elif profile.max_cpu_percent <= 60:
                allocator.set_mode('performance')
            else:
                allocator.set_mode('maximum')
        except ImportError:
            pass
        
        # Apply tool restrictions via ToolManager
        try:
            from ..tools.tool_manager import get_tool_manager
            manager = get_tool_manager()
            
            if profile.allowed_tools:
                # Enable only allowed tools
                for tool in manager.get_all_tools():
                    if tool in profile.allowed_tools:
                        manager.enable_tool(tool)
                    else:
                        manager.disable_tool(tool)
            elif profile.blocked_tools:
                # Block specific tools
                for tool in profile.blocked_tools:
                    manager.disable_tool(tool)
        except ImportError:
            pass
    
    def create_custom_profile(
        self, 
        name: str,
        base_profile: str = "balanced",
        **overrides
    ) -> HardwareProfile:
        """
        Create a custom profile based on an existing one.
        
        Args:
            name: Name for the new profile
            base_profile: Profile to base it on
            **overrides: Settings to override
            
        Returns:
            The new profile
        """
        base = self.get_profile(base_profile)
        if base is None:
            base = BUILTIN_PROFILES["balanced"]
        
        # Create copy with overrides
        profile_data = base.to_dict()
        profile_data["name"] = name
        profile_data["device_type"] = DeviceType.CUSTOM.value
        profile_data.update(overrides)
        
        profile = HardwareProfile.from_dict(profile_data)
        self.custom_profiles[name] = profile
        self._save()
        
        return profile
    
    def delete_custom_profile(self, name: str) -> bool:
        """Delete a custom profile."""
        if name in self.custom_profiles:
            del self.custom_profiles[name]
            if self.active_profile_name == name:
                self.active_profile_name = "balanced"
            self._save()
            return True
        return False
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits from active profile."""
        profile = self.get_active_profile()
        return {
            "max_cpu_percent": profile.max_cpu_percent,
            "max_memory_percent": profile.max_memory_percent,
            "max_gpu_memory_percent": profile.max_gpu_memory_percent,
            "max_model_size": profile.max_model_size,
            "use_quantization": profile.use_quantization,
            "quantization_bits": profile.quantization_bits,
            "batch_size": profile.batch_size,
            "max_context_length": profile.max_context_length,
            "inference_threads": profile.inference_threads,
            "process_priority": profile.process_priority,
        }
    
    def get_enabled_features(self) -> Dict[str, bool]:
        """Get enabled features from active profile."""
        profile = self.get_active_profile()
        return {
            "chat": profile.enable_chat,
            "avatar": profile.enable_avatar,
            "voice_input": profile.enable_voice_input,
            "voice_output": profile.enable_voice_output,
            "vision": profile.enable_vision,
            "image_gen": profile.enable_image_gen,
            "code_gen": profile.enable_code_gen,
            "video_gen": profile.enable_video_gen,
            "audio_gen": profile.enable_audio_gen,
            "3d_gen": profile.enable_3d_gen,
            "training": profile.enable_training,
            "tools": profile.enable_tools,
            "api_server": profile.enable_api_server,
            "remote_client": profile.enable_remote_client,
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        features = self.get_enabled_features()
        return features.get(feature, False)
    
    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool restrictions from active profile."""
        profile = self.get_active_profile()
        return {
            "allowed_tools": profile.allowed_tools,
            "blocked_tools": profile.blocked_tools,
        }
    
    def auto_detect_profile(self) -> str:
        """
        Auto-detect the best profile for this hardware.
        
        Returns:
            Name of recommended profile
        """
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check platform
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Raspberry Pi detection
        if "arm" in machine or "aarch64" in machine:
            if memory_gb < 4:
                return "raspberry_pi"
        
        # Check for GPU
        has_gpu = False
        gpu_memory_gb = 0
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        # Decision tree
        if memory_gb < 2:
            return "raspberry_pi"
        elif memory_gb < 4:
            return "phone"
        elif has_gpu and gpu_memory_gb >= 8:
            return "workstation"
        elif has_gpu:
            return "balanced"
        else:
            return "balanced"
    
    def get_profile_summary(self, name: str = None) -> Dict[str, Any]:
        """Get a summary of a profile for display."""
        if name is None:
            name = self.active_profile_name
        
        profile = self.get_profile(name)
        if profile is None:
            return {"error": "Profile not found"}
        
        features = self.get_enabled_features()
        enabled_count = sum(1 for v in features.values() if v)
        
        return {
            "name": profile.name,
            "device_type": profile.device_type,
            "description": profile.description,
            "resource_limits": {
                "cpu": f"{profile.max_cpu_percent}%",
                "memory": f"{profile.max_memory_percent}%",
                "gpu": f"{profile.max_gpu_memory_percent}%",
            },
            "model": {
                "max_size": profile.max_model_size,
                "quantized": profile.use_quantization,
            },
            "features_enabled": f"{enabled_count}/14",
            "is_active": name == self.active_profile_name,
        }


# Singleton instance
_profile_manager: Optional[HardwareProfileManager] = None


def get_profile_manager() -> HardwareProfileManager:
    """Get the global hardware profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = HardwareProfileManager()
    return _profile_manager


if __name__ == "__main__":
    # Test hardware profiles
    print("Hardware Profile System Test")
    print("=" * 60)
    
    manager = HardwareProfileManager()
    
    # Show auto-detection
    recommended = manager.auto_detect_profile()
    print(f"\nAuto-detected profile: {recommended}")
    
    # Show all profiles
    print("\n\nAvailable Profiles:")
    print("-" * 60)
    
    for name, profile in manager.get_all_profiles().items():
        summary = manager.get_profile_summary(name)
        active = " [ACTIVE]" if summary["is_active"] else ""
        print(f"\n{profile.name}{active}")
        print(f"  Type: {profile.device_type}")
        print(f"  Description: {profile.description}")
        print(f"  Resources: CPU {profile.max_cpu_percent}%, RAM {profile.max_memory_percent}%, GPU {profile.max_gpu_memory_percent}%")
        print(f"  Max model: {profile.max_model_size}")
        print(f"  Features: {summary['features_enabled']}")
    
    # Test setting profile
    print("\n\nSetting profile to 'pc_gaming'...")
    manager.set_active_profile("pc_gaming")
    
    print("\nResource limits:")
    for k, v in manager.get_resource_limits().items():
        print(f"  {k}: {v}")
    
    print("\nEnabled features:")
    for k, v in manager.get_enabled_features().items():
        status = "Yes" if v else "No"
        print(f"  {k}: {status}")
