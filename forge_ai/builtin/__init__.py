"""
ForgeAI Built-in Fallbacks

Zero-dependency implementations that work out of the box.
These provide basic functionality when external libraries aren't installed.

Usage:
    from forge_ai.builtin import BuiltinTTS, BuiltinImageGen
    
    tts = BuiltinTTS()
    tts.load()
    tts.speak("Hello world!")
    
    img = BuiltinImageGen()
    img.load()
    result = img.generate("a sunset over mountains")
    with open("output.png", "wb") as f:
        f.write(result["image_data"])
"""

from .tts import BuiltinTTS
from .embeddings import BuiltinEmbeddings
from .code_gen import BuiltinCodeGen
from .image_gen import BuiltinImageGen
from .chat import BuiltinChat
from .video_gen import BuiltinVideoGen
from .threed_gen import Builtin3DGen
from .stt import BuiltinSTT

__all__ = [
    'BuiltinTTS',
    'BuiltinEmbeddings', 
    'BuiltinCodeGen',
    'BuiltinImageGen',
    'BuiltinChat',
    'BuiltinVideoGen',
    'Builtin3DGen',
    'BuiltinSTT',
]


def get_builtin_status() -> dict:
    """Check which built-in modules are available."""
    return {
        "tts": True,  # Always available (uses system speech)
        "stt": True,  # Available on Windows, limited elsewhere
        "embeddings": True,  # Always available (pure Python)
        "code_gen": True,  # Always available (templates)
        "image_gen": True,  # Always available (pure Python PNG)
        "video_gen": True,  # Always available (pure Python GIF)
        "threed_gen": True,  # Always available (pure Python OBJ)
        "chat": True,  # Always available (rule-based)
    }

