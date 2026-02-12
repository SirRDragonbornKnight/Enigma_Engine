"""
Resource Allocation System - Manage CPU, RAM, and GPU usage

Provides different performance modes:
- Minimal: Low resource usage, slower performance
- Balanced: Good balance between speed and resource usage
- Performance: Fast, uses more resources
- Maximum: All available resources

Also includes "Speed vs Quality" toggle for generation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


class ResourceAllocator:
    """
    Manages resource allocation for AI operations.
    
    Features:
    - CPU thread control
    - Memory limits
    - GPU memory fraction
    - Process priority
    - Performance modes
    """
    
    MODES = {
        'minimal': {
            'name': 'Minimal',
            'description': 'Low resource usage - runs in background',
            'cpu_threads': 1,
            'memory_limit_percent': 10,
            'gpu_memory_fraction': 0.3,
            'process_priority': 'low',
            'batch_size': 1,
            'model_size_limit': 'small'
        },
        'balanced': {
            'name': 'Balanced',
            'description': 'Good balance - recommended for most users',
            'cpu_threads': 0,  # Auto (half of available)
            'memory_limit_percent': 40,
            'gpu_memory_fraction': 0.5,
            'process_priority': 'normal',
            'batch_size': 2,
            'model_size_limit': 'medium'
        },
        'performance': {
            'name': 'Performance',
            'description': 'Fast processing - uses significant resources',
            'cpu_threads': 0,  # Auto (most available)
            'memory_limit_percent': 60,
            'gpu_memory_fraction': 0.7,
            'process_priority': 'normal',
            'batch_size': 4,
            'model_size_limit': 'large'
        },
        'maximum': {
            'name': 'Maximum',
            'description': 'All available resources - may affect other apps',
            'cpu_threads': 0,  # Auto (all available)
            'memory_limit_percent': 80,
            'gpu_memory_fraction': 0.9,
            'process_priority': 'high',
            'batch_size': 8,
            'model_size_limit': 'xl'
        }
    }
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize resource allocator.
        
        Args:
            storage_path: Path to store settings
        """
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "resource_settings.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.current_mode = self._load_mode()
        self.speed_vs_quality = self._load_preference()
    
    def _load_mode(self) -> str:
        """Load saved resource mode."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                return data.get('mode', 'balanced')
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted resource config, using balanced mode: {e}")
                return 'balanced'
            except OSError as e:
                logger.warning(f"Could not read resource config: {e}")
                return 'balanced'
        return 'balanced'
    
    def _load_preference(self) -> str:
        """Load speed vs quality preference."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                return data.get('preference', 'balanced')
            except json.JSONDecodeError as e:
                logger.debug(f"Corrupted preference file: {e}")
                return 'balanced'
            except OSError as e:
                logger.debug(f"Could not read preference file: {e}")
                return 'balanced'
        return 'balanced'
    
    def _save(self):
        """Save settings."""
        data = {
            'mode': self.current_mode,
            'preference': self.speed_vs_quality
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def set_mode(self, mode: str) -> bool:
        """
        Set resource allocation mode.
        
        Args:
            mode: Mode name ('minimal', 'balanced', 'performance', 'maximum')
            
        Returns:
            True if mode was set successfully
        """
        if mode not in self.MODES:
            return False
        
        self.current_mode = mode
        self._save()
        self._apply_mode()
        return True
    
    def _apply_mode(self):
        """Apply current mode settings to the process."""
        settings = self.MODES[self.current_mode]
        
        # Set process priority
        try:
            priority = settings['process_priority']
            if priority == 'low':
                psutil.Process().nice(10)  # Lower priority
            elif priority == 'high':
                psutil.Process().nice(-10)  # Higher priority
            else:
                psutil.Process().nice(0)  # Normal
        except (OSError, psutil.AccessDenied):
            pass  # May fail on some systems
        
        # Set CPU affinity if threads specified
        try:
            threads = settings['cpu_threads']
            if threads > 0:
                cpu_count = psutil.cpu_count()
                cpus = list(range(min(threads, cpu_count)))
                psutil.Process().cpu_affinity(cpus)
        except (OSError, psutil.AccessDenied, AttributeError):
            pass  # Intentionally silent
    
    def get_current_mode(self) -> dict[str, Any]:
        """Get current mode settings."""
        return {
            'mode': self.current_mode,
            'settings': self.MODES[self.current_mode]
        }
    
    def get_all_modes(self) -> dict[str, dict]:
        """Get all available modes."""
        return self.MODES.copy()
    
    def get_system_info(self) -> dict[str, Any]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            info = {
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
            }
            
            # Check for GPU
            try:
                import torch
                if torch.cuda.is_available():
                    info['gpu_available'] = True
                    info['gpu_name'] = torch.cuda.get_device_name(0)
                    info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    info['gpu_available'] = False
            except ImportError:
                info['gpu_available'] = False
            
            return info
        except Exception as e:
            return {'error': str(e)}
    
    def get_recommended_mode(self) -> str:
        """Get recommended mode based on system resources."""
        info = self.get_system_info()
        
        if 'error' in info:
            return 'balanced'
        
        available_gb = info.get('memory_available_gb', 2)
        cpu_count = info.get('cpu_count', 2)
        
        # Recommend based on available memory
        if available_gb < 2:
            return 'minimal'
        elif available_gb < 4:
            return 'balanced'
        elif available_gb < 8:
            return 'performance'
        else:
            return 'maximum'
    
    def set_speed_vs_quality(self, preference: str) -> bool:
        """
        Set speed vs quality preference.
        
        Args:
            preference: 'speed', 'balanced', or 'quality'
            
        Returns:
            True if preference was set
        """
        if preference not in ['speed', 'balanced', 'quality']:
            return False
        
        self.speed_vs_quality = preference
        self._save()
        return True
    
    def get_generation_params(self) -> dict[str, Any]:
        """
        Get generation parameters based on speed/quality preference.
        
        Returns:
            Dict with parameters for generation
        """
        if self.speed_vs_quality == 'speed':
            return {
                'max_tokens': 100,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50,
                'num_beams': 1,  # Greedy
                'do_sample': True
            }
        elif self.speed_vs_quality == 'quality':
            return {
                'max_tokens': 500,
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': 100,
                'num_beams': 4,  # Beam search
                'do_sample': True
            }
        else:  # balanced
            return {
                'max_tokens': 250,
                'temperature': 0.75,
                'top_p': 0.92,
                'top_k': 75,
                'num_beams': 2,
                'do_sample': True
            }
    
    def get_training_params(self) -> dict[str, Any]:
        """
        Get training parameters based on current mode.
        
        Returns:
            Dict with training parameters
        """
        settings = self.MODES[self.current_mode]
        
        return {
            'batch_size': settings['batch_size'],
            'num_workers': 0 if settings['cpu_threads'] == 1 else 2,
            'pin_memory': True if self.get_system_info().get('gpu_available') else False,
        }
    
    def get_resource_limits(self) -> dict[str, Any]:
        """
        Get resource limits for current mode.
        
        Returns:
            Dict with resource limits
        """
        settings = self.MODES[self.current_mode]
        info = self.get_system_info()
        
        total_memory_gb = info.get('memory_total_gb', 8)
        memory_limit_gb = total_memory_gb * (settings['memory_limit_percent'] / 100)
        
        return {
            'memory_limit_gb': memory_limit_gb,
            'gpu_memory_fraction': settings['gpu_memory_fraction'],
            'cpu_threads': settings['cpu_threads'],
            'model_size_limit': settings['model_size_limit']
        }
    
    def check_if_model_fits(self, model_size: str) -> bool:
        """
        Check if a model size fits within resource limits.
        
        Args:
            model_size: Model size name
            
        Returns:
            True if model fits
        """
        size_order = ['nano', 'micro', 'tiny', 'small', 'medium', 'large', 'xl', 'xxl']
        limits = self.get_resource_limits()
        max_size = limits['model_size_limit']
        
        try:
            model_idx = size_order.index(model_size)
            max_idx = size_order.index(max_size)
            return model_idx <= max_idx
        except ValueError:
            return False


class PerformanceMonitor:
    """Monitor performance metrics during operations."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = []
    
    def start(self):
        """Start monitoring."""
        import time
        self.start_time = time.time()
        self.metrics = []
    
    def record(self, operation: str):
        """Record a metric."""
        import time
        if self.start_time is None:
            self.start()
        
        elapsed = time.time() - self.start_time
        
        # Get current resource usage
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        self.metrics.append({
            'operation': operation,
            'elapsed': elapsed,
            'cpu_percent': cpu,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3)
        })
    
    def get_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        return {
            'total_operations': len(self.metrics),
            'total_time': self.metrics[-1]['elapsed'],
            'avg_cpu': sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics),
            'max_cpu': max(m['cpu_percent'] for m in self.metrics),
            'avg_memory': sum(m['memory_percent'] for m in self.metrics) / len(self.metrics),
            'max_memory': max(m['memory_percent'] for m in self.metrics),
            'operations': self.metrics
        }


if __name__ == "__main__":
    # Test resource allocator
    print("Resource Allocation System Test")
    print("=" * 60)
    
    allocator = ResourceAllocator()
    
    # Show system info
    print("\nSystem Information:")
    print("-" * 60)
    info = allocator.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Show available modes
    print("\n\nAvailable Modes:")
    print("-" * 60)
    for mode_name, mode_settings in allocator.get_all_modes().items():
        print(f"\n{mode_settings['name']} ({mode_name}):")
        print(f"  {mode_settings['description']}")
        print(f"  CPU threads: {mode_settings['cpu_threads']}")
        print(f"  Memory limit: {mode_settings['memory_limit_percent']}%")
        print(f"  GPU memory: {mode_settings['gpu_memory_fraction'] * 100}%")
    
    # Show recommended mode
    recommended = allocator.get_recommended_mode()
    print(f"\n\nRecommended mode for this system: {recommended}")
    
    # Test speed vs quality
    print("\n\nGeneration Parameters:")
    print("-" * 60)
    for pref in ['speed', 'balanced', 'quality']:
        allocator.set_speed_vs_quality(pref)
        params = allocator.get_generation_params()
        print(f"\n{pref.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
