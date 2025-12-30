"""
Addon Registry
==============

Registry of all available addons.
Provides easy access and discovery.
"""

from typing import Dict, List, Optional, Type
from .base import Addon, AddonType, AddonProvider
from .builtin import BUILTIN_ADDONS


# Global registry
ADDON_REGISTRY: Dict[str, Type[Addon]] = {}


def register_addon(name: str, addon_class: Type[Addon]):
    """Register an addon class."""
    ADDON_REGISTRY[name] = addon_class


def unregister_addon(name: str):
    """Unregister an addon class."""
    if name in ADDON_REGISTRY:
        del ADDON_REGISTRY[name]


def get_addon(name: str) -> Optional[Type[Addon]]:
    """Get an addon class by name."""
    return ADDON_REGISTRY.get(name) or BUILTIN_ADDONS.get(name)


def list_addons(addon_type: Optional[AddonType] = None) -> List[str]:
    """List all registered addon names."""
    all_addons = {**BUILTIN_ADDONS, **ADDON_REGISTRY}
    
    if addon_type is None:
        return list(all_addons.keys())
    
    # Filter by type - need to instantiate to check type
    result = []
    for name, cls in all_addons.items():
        try:
            # Check if the class is a subclass of the appropriate base
            from .base import ImageAddon, CodeAddon, VideoAddon, AudioAddon, EmbeddingAddon
            
            type_map = {
                AddonType.IMAGE: ImageAddon,
                AddonType.CODE: CodeAddon,
                AddonType.VIDEO: VideoAddon,
                AddonType.AUDIO: AudioAddon,
                AddonType.EMBEDDING: EmbeddingAddon,
            }
            
            if addon_type in type_map:
                if issubclass(cls, type_map[addon_type]):
                    result.append(name)
            else:
                result.append(name)
        except:
            pass
    
    return result


def list_builtin_addons() -> Dict[str, dict]:
    """List all built-in addons with info."""
    result = {}
    
    for name, cls in BUILTIN_ADDONS.items():
        try:
            instance = cls()
            result[name] = {
                'type': instance.addon_type.name,
                'provider': instance.provider.name,
                'version': instance.config.version,
            }
        except:
            result[name] = {'type': 'unknown', 'provider': 'unknown'}
    
    return result


def get_addons_by_type() -> Dict[str, List[str]]:
    """Get all addons grouped by type."""
    result = {t.name: [] for t in AddonType}
    
    for name in list_addons():
        cls = get_addon(name)
        if cls:
            try:
                instance = cls()
                result[instance.addon_type.name].append(name)
            except:
                pass
    
    # Remove empty types
    return {k: v for k, v in result.items() if v}


def get_addons_by_provider() -> Dict[str, List[str]]:
    """Get all addons grouped by provider."""
    result = {p.name: [] for p in AddonProvider}
    
    for name in list_addons():
        cls = get_addon(name)
        if cls:
            try:
                instance = cls()
                result[instance.provider.name].append(name)
            except:
                pass
    
    # Remove empty providers
    return {k: v for k, v in result.items() if v}


# Register built-in addons
for name, cls in BUILTIN_ADDONS.items():
    register_addon(name, cls)


# === Addon Discovery Info ===

ADDON_INFO = {
    'image': {
        'description': 'Image generation capabilities',
        'addons': {
            'stable_diffusion_local': {
                'name': 'Stable Diffusion (Local)',
                'description': 'Run Stable Diffusion locally on your GPU',
                'requirements': ['torch', 'diffusers', 'transformers', 'accelerate'],
                'provider': 'LOCAL',
            },
            'openai_dalle': {
                'name': 'OpenAI DALL-E',
                'description': 'Generate images using DALL-E 3',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
            'replicate_image': {
                'name': 'Replicate (SDXL, Flux)',
                'description': 'Cloud-based image generation',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
                'needs_api_key': True,
            },
        }
    },
    'code': {
        'description': 'Code generation and editing',
        'addons': {
            'enigma_code': {
                'name': 'Enigma Code',
                'description': 'Use Enigma\'s own model for code',
                'requirements': [],
                'provider': 'LOCAL',
            },
            'openai_code': {
                'name': 'OpenAI Code',
                'description': 'GPT-4 for code generation',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
        }
    },
    'video': {
        'description': 'Video generation capabilities',
        'addons': {
            'replicate_video': {
                'name': 'Replicate Video',
                'description': 'Cloud-based video generation',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
                'needs_api_key': True,
            },
            'local_video': {
                'name': 'AnimateDiff (Local)',
                'description': 'Local video generation with AnimateDiff',
                'requirements': ['torch', 'diffusers'],
                'provider': 'LOCAL',
            },
        }
    },
    'audio': {
        'description': 'Audio and speech generation',
        'addons': {
            'elevenlabs_tts': {
                'name': 'ElevenLabs TTS',
                'description': 'High quality text-to-speech',
                'requirements': ['elevenlabs'],
                'provider': 'ELEVENLABS',
                'needs_api_key': True,
            },
            'local_tts': {
                'name': 'Local TTS',
                'description': 'Offline text-to-speech',
                'requirements': ['pyttsx3'],
                'provider': 'LOCAL',
            },
            'replicate_audio': {
                'name': 'Replicate Audio',
                'description': 'Music generation with MusicGen',
                'requirements': ['replicate'],
                'provider': 'REPLICATE',
                'needs_api_key': True,
            },
        }
    },
    'embedding': {
        'description': 'Vector embeddings for search',
        'addons': {
            'local_embedding': {
                'name': 'Local Embedding',
                'description': 'sentence-transformers embeddings',
                'requirements': ['sentence-transformers'],
                'provider': 'LOCAL',
            },
            'openai_embedding': {
                'name': 'OpenAI Embedding',
                'description': 'OpenAI text embeddings',
                'requirements': ['openai'],
                'provider': 'OPENAI',
                'needs_api_key': True,
            },
        }
    },
}
