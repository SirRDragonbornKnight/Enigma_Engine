"""
================================================================================
STANDALONE TOOLS - USE AI CAPABILITIES WITHOUT LLM
================================================================================

Provides a unified interface to use Enigma AI Engine's tools and capabilities
without requiring the full chat/LLM system.

Want to just generate an image? Use vision to analyze a picture? Generate code?
You can do all of these WITHOUT loading a full chat model.

FILE: enigma_engine/core/standalone_tools.py
TYPE: Standalone Tool Interface
MAIN FUNCTION: use_tool

USAGE:
    from enigma_engine import use_tool
    
    # Generate an image without chat
    image = use_tool("image", prompt="A sunset over mountains", width=512, height=512)
    
    # Analyze an image without chat
    description = use_tool("vision", image_path="photo.jpg", question="What's in this?")
    
    # Generate code without chat
    code = use_tool("code", prompt="Python function to sort a list")
    
    # Text-to-speech without chat
    use_tool("tts", text="Hello world", output_file="hello.wav")
    
    # Speech-to-text without chat
    text = use_tool("stt", audio_file="recording.wav")
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL EXECUTION INTERFACE
# =============================================================================

def use_tool(
    tool_name: str,
    *args,
    **kwargs,
) -> Any:
    """
    Use a Enigma AI Engine tool standalone (without LLM/chat).
    
    Args:
        tool_name: Name of tool to use
        *args: Positional arguments for the tool
        **kwargs: Keyword arguments for the tool
        
    Returns:
        Tool result
        
    Available Tools:
        - image: Generate images
        - vision: Analyze images
        - code: Generate code
        - video: Generate videos
        - audio: Generate audio/music
        - tts: Text to speech
        - stt: Speech to text
        - embed: Create text embeddings
        - 3d: Generate 3D models
        - gif: Create animated GIFs
        - avatar: Control avatar
        - web: Web search and browsing
        - file: File operations
    
    Examples:
        >>> image = use_tool("image", prompt="a cat", width=512, height=512)
        >>> description = use_tool("vision", image_path="cat.jpg")
        >>> code = use_tool("code", prompt="sort function", language="python")
    """
    # Map tool names to execution functions
    tool_map = {
        "image": _use_image_generation,
        "image_gen": _use_image_generation,
        "vision": _use_vision,
        "see": _use_vision,
        "code": _use_code_generation,
        "code_gen": _use_code_generation,
        "video": _use_video_generation,
        "video_gen": _use_video_generation,
        "audio": _use_audio_generation,
        "audio_gen": _use_audio_generation,
        "tts": _use_text_to_speech,
        "text_to_speech": _use_text_to_speech,
        "stt": _use_speech_to_text,
        "speech_to_text": _use_speech_to_text,
        "embed": _use_embeddings,
        "embedding": _use_embeddings,
        "3d": _use_3d_generation,
        "threed": _use_3d_generation,
        "gif": _use_gif_generation,
        "avatar": _use_avatar,
        "web": _use_web_tools,
        "web_search": _use_web_tools,
        "file": _use_file_tools,
    }
    
    tool_name_lower = tool_name.lower()
    if tool_name_lower not in tool_map:
        available = ", ".join(sorted(set(tool_map.keys())))
        raise ValueError(
            f"Unknown tool: {tool_name}. Available tools: {available}"
        )
    
    # Execute the tool
    tool_func = tool_map[tool_name_lower]
    try:
        return tool_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        raise


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def _use_image_generation(
    prompt: str,
    width: int = 512,
    height: int = 512,
    provider: str = "auto",
    output_path: Optional[str] = None,
    **kwargs,
) -> Union[str, Any]:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt: Text description of image
        width: Image width
        height: Image height
        provider: "local", "api", or "auto"
        output_path: Optional path to save image
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Path to generated image or image data
    """
    from ..modules.manager import ModuleManager
    
    manager = ModuleManager()
    
    # Determine which image gen module to use
    if provider == "auto":
        # Prefer local if available
        if manager.is_available("image_gen_local"):
            provider = "local"
        elif manager.is_available("image_gen_api"):
            provider = "api"
        else:
            raise RuntimeError(
                "No image generation module available. "
                "Enable 'image_gen_local' or 'image_gen_api' module."
            )
    
    # Load the module
    module_name = f"image_gen_{provider}"
    if not manager.is_loaded(module_name):
        manager.load(module_name)
    
    # Get the module
    module = manager.get_module(module_name)
    
    # Generate image
    result = module.generate(
        prompt=prompt,
        width=width,
        height=height,
        **kwargs
    )
    
    # Save if output path provided
    if output_path and hasattr(result, "save"):
        result.save(output_path)
        return output_path
    
    return result


# =============================================================================
# VISION (IMAGE UNDERSTANDING)
# =============================================================================

def _use_vision(
    image_path: Optional[str] = None,
    image: Optional[Any] = None,
    question: str = "Describe this image",
    **kwargs,
) -> str:
    """
    Analyze an image and answer questions about it.
    
    Args:
        image_path: Path to image file
        image: Or image data directly
        question: Question to ask about the image
        **kwargs: Additional arguments
        
    Returns:
        Description or answer about the image
    """
    from ..tools.vision import analyze_image
    
    if image_path is None and image is None:
        raise ValueError("Must provide either image_path or image")
    
    # Use the vision tool
    result = analyze_image(
        image_path=image_path,
        image=image,
        question=question,
        **kwargs
    )
    
    return result


# =============================================================================
# CODE GENERATION
# =============================================================================

def _use_code_generation(
    prompt: str,
    language: str = "python",
    provider: str = "auto",
    **kwargs,
) -> str:
    """
    Generate code from a description.
    
    Args:
        prompt: Description of what code should do
        language: Programming language
        provider: "local", "api", or "auto"
        **kwargs: Additional arguments
        
    Returns:
        Generated code
    """
    from ..modules.manager import ModuleManager
    
    manager = ModuleManager()
    
    # Determine which code gen module to use
    if provider == "auto":
        if manager.is_available("code_gen_local"):
            provider = "local"
        elif manager.is_available("code_gen_api"):
            provider = "api"
        else:
            raise RuntimeError(
                "No code generation module available. "
                "Enable 'code_gen_local' or 'code_gen_api' module."
            )
    
    # Load the module
    module_name = f"code_gen_{provider}"
    if not manager.is_loaded(module_name):
        manager.load(module_name)
    
    # Get the module
    module = manager.get_module(module_name)
    
    # Generate code
    result = module.generate(
        prompt=prompt,
        language=language,
        **kwargs
    )
    
    return result


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def _use_video_generation(
    prompt: str,
    duration: int = 5,
    provider: str = "auto",
    output_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate a video from a text prompt.
    
    Args:
        prompt: Description of video content
        duration: Duration in seconds
        provider: "local", "api", or "auto"
        output_path: Optional path to save video
        **kwargs: Additional arguments
        
    Returns:
        Path to generated video
    """
    from ..modules.manager import ModuleManager
    
    manager = ModuleManager()
    
    # Determine which video gen module to use
    if provider == "auto":
        if manager.is_available("video_gen_local"):
            provider = "local"
        elif manager.is_available("video_gen_api"):
            provider = "api"
        else:
            raise RuntimeError(
                "No video generation module available. "
                "Enable 'video_gen_local' or 'video_gen_api' module."
            )
    
    # Load the module
    module_name = f"video_gen_{provider}"
    if not manager.is_loaded(module_name):
        manager.load(module_name)
    
    # Get the module
    module = manager.get_module(module_name)
    
    # Generate video
    result = module.generate(
        prompt=prompt,
        duration=duration,
        output_path=output_path,
        **kwargs
    )
    
    return result


# =============================================================================
# AUDIO GENERATION
# =============================================================================

def _use_audio_generation(
    prompt: str,
    provider: str = "auto",
    output_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate audio/music from a text prompt.
    
    Args:
        prompt: Description of audio to generate
        provider: "local", "api", or "auto"
        output_path: Optional path to save audio
        **kwargs: Additional arguments
        
    Returns:
        Path to generated audio
    """
    from ..modules.manager import ModuleManager
    
    manager = ModuleManager()
    
    # Determine which audio gen module to use
    if provider == "auto":
        if manager.is_available("audio_gen_local"):
            provider = "local"
        elif manager.is_available("audio_gen_api"):
            provider = "api"
        else:
            raise RuntimeError(
                "No audio generation module available. "
                "Enable 'audio_gen_local' or 'audio_gen_api' module."
            )
    
    # Load the module
    module_name = f"audio_gen_{provider}"
    if not manager.is_loaded(module_name):
        manager.load(module_name)
    
    # Get the module
    module = manager.get_module(module_name)
    
    # Generate audio
    result = module.generate(
        prompt=prompt,
        output_path=output_path,
        **kwargs
    )
    
    return result


# =============================================================================
# TEXT-TO-SPEECH
# =============================================================================

def _use_text_to_speech(
    text: str,
    output_file: Optional[str] = None,
    voice: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """
    Convert text to speech.
    
    Args:
        text: Text to speak
        output_file: Optional file to save audio
        voice: Optional voice selection
        **kwargs: Additional arguments
        
    Returns:
        Path to audio file if output_file provided, else None (plays audio)
    """
    try:
        from ..voice.voice_generator import AIVoiceGenerator
        
        generator = AIVoiceGenerator()
        
        if output_file:
            generator.speak(text, save_to_file=output_file)
            return output_file
        else:
            generator.speak(text)
            return None
    
    except ImportError:
        # Fallback to pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            if output_file:
                engine.save_to_file(text, output_file)
                engine.runAndWait()
                return output_file
            else:
                engine.say(text)
                engine.runAndWait()
                return None
        
        except ImportError:
            raise ImportError(
                "No TTS engine available. Install pyttsx3: pip install pyttsx3"
            )


# =============================================================================
# SPEECH-TO-TEXT
# =============================================================================

def _use_speech_to_text(
    audio_file: Optional[str] = None,
    duration: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Convert speech to text.
    
    Args:
        audio_file: Path to audio file, or None to record from microphone
        duration: Duration to record (if audio_file is None)
        **kwargs: Additional arguments
        
    Returns:
        Transcribed text
    """
    try:
        from ..voice.listener import VoiceListener
        
        listener = VoiceListener()
        
        if audio_file:
            text = listener.transcribe_file(audio_file)
        else:
            text = listener.listen(duration=duration)
        
        return text
    
    except ImportError:
        raise ImportError(
            "Speech recognition not available. "
            "Install: pip install speechrecognition pyaudio"
        )


# =============================================================================
# EMBEDDINGS
# =============================================================================

def _use_embeddings(
    text: Union[str, list[str]],
    provider: str = "auto",
    **kwargs,
) -> Union[list[float], list[list[float]]]:
    """
    Create text embeddings for semantic search.
    
    Args:
        text: Text or list of texts to embed
        provider: "local", "api", or "auto"
        **kwargs: Additional arguments
        
    Returns:
        Embedding vector(s)
    """
    from ..modules.manager import ModuleManager
    
    manager = ModuleManager()
    
    # Determine which embedding module to use
    if provider == "auto":
        if manager.is_available("embedding_local"):
            provider = "local"
        elif manager.is_available("embedding_api"):
            provider = "api"
        else:
            raise RuntimeError(
                "No embedding module available. "
                "Enable 'embedding_local' or 'embedding_api' module."
            )
    
    # Load the module
    module_name = f"embedding_{provider}"
    if not manager.is_loaded(module_name):
        manager.load(module_name)
    
    # Get the module
    module = manager.get_module(module_name)
    
    # Generate embeddings
    result = module.embed(text, **kwargs)
    
    return result


# =============================================================================
# 3D GENERATION
# =============================================================================

def _use_3d_generation(
    prompt: str,
    provider: str = "auto",
    output_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate 3D model from text prompt.
    
    Args:
        prompt: Description of 3D model
        provider: "local", "api", or "auto"
        output_path: Optional path to save model
        **kwargs: Additional arguments
        
    Returns:
        Path to generated 3D model
    """
    from ..modules.manager import ModuleManager
    
    manager = ModuleManager()
    
    # Determine which 3D gen module to use
    if provider == "auto":
        if manager.is_available("threed_gen_local"):
            provider = "local"
        elif manager.is_available("threed_gen_api"):
            provider = "api"
        else:
            raise RuntimeError(
                "No 3D generation module available. "
                "Enable 'threed_gen_local' or 'threed_gen_api' module."
            )
    
    # Load the module
    module_name = f"threed_gen_{provider}"
    if not manager.is_loaded(module_name):
        manager.load(module_name)
    
    # Get the module
    module = manager.get_module(module_name)
    
    # Generate 3D model
    result = module.generate(
        prompt=prompt,
        output_path=output_path,
        **kwargs
    )
    
    return result


# =============================================================================
# GIF GENERATION
# =============================================================================

def _use_gif_generation(
    prompt: str,
    frames: int = 10,
    output_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate animated GIF.
    
    Args:
        prompt: Description of animation
        frames: Number of frames
        output_path: Optional path to save GIF
        **kwargs: Additional arguments (width, height, fps, loop)
        
    Returns:
        Path to generated GIF
    """
    try:
        import time

        from PIL import Image

        # Get dimensions and other params
        width = kwargs.get("width", 256)
        height = kwargs.get("height", 256)
        fps = kwargs.get("fps", 5)
        loop = kwargs.get("loop", 0)  # 0 = infinite loop
        
        # Determine output path
        if output_path is None:
            outputs_dir = Path.cwd() / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            output_path = str(outputs_dir / f"animation_{timestamp}.gif")
        
        # Check for image generation capability
        try:
            from ..modules import get_module_manager
            manager = get_module_manager()
            
            # Try to get image generation module
            image_gen = None
            for name in ["image_gen_local", "image_gen_api", "image_gen"]:
                if manager.is_loaded(name):
                    image_gen = manager.get_module(name)
                    break
            
            if image_gen is not None:
                # Generate frames using image generation
                frame_images = []
                
                for i in range(frames):
                    # Modify prompt with frame number for animation effect
                    frame_prompt = f"{prompt}, frame {i+1} of {frames}"
                    
                    # Generate frame
                    if hasattr(image_gen, "generate"):
                        result = image_gen.generate(
                            prompt=frame_prompt,
                            width=width,
                            height=height
                        )
                    else:
                        result = image_gen(frame_prompt, width=width, height=height)
                    
                    # Handle result
                    if isinstance(result, (str, Path)):
                        img = Image.open(result)
                    elif hasattr(result, 'images') and len(result.images) > 0:
                        img = result.images[0]
                    elif isinstance(result, Image.Image):
                        img = result
                    else:
                        raise ValueError(f"Unknown image result type: {type(result)}")
                    
                    # Resize if needed
                    if img.size != (width, height):
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    frame_images.append(img)
                
                # Save as GIF
                duration_ms = int(1000 / fps)
                frame_images[0].save(
                    output_path,
                    save_all=True,
                    append_images=frame_images[1:],
                    duration=duration_ms,
                    loop=loop,
                    optimize=False
                )
                
                return output_path
        except ImportError:
            pass  # Intentionally silent
        
        # Fallback: Create simple procedural animation
        # This creates a gradient animation without AI image generation
        frame_images = []
        
        for i in range(frames):
            # Create frame with shifting colors
            img = Image.new("RGB", (width, height))
            pixels = img.load()
            
            # Simple gradient animation
            phase = i / frames
            for x in range(width):
                for y in range(height):
                    # Animated gradient pattern
                    import math
                    r = int(127 + 127 * math.sin(2 * math.pi * (x / width + phase)))
                    g = int(127 + 127 * math.sin(2 * math.pi * (y / height + phase)))
                    b = int(127 + 127 * math.sin(2 * math.pi * ((x + y) / (width + height) + phase)))
                    pixels[x, y] = (r, g, b)
            
            frame_images.append(img)
        
        # Save as GIF
        duration_ms = int(1000 / fps)
        frame_images[0].save(
            output_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=duration_ms,
            loop=loop,
            optimize=True
        )
        
        return output_path
        
    except Exception as e:
        logger.error(f"GIF generation error: {e}")
        raise RuntimeError(f"GIF generation failed: {e}")


# =============================================================================
# AVATAR CONTROL
# =============================================================================

def _use_avatar(
    action: str,
    **kwargs,
) -> dict[str, Any]:
    """
    Control avatar without LLM.
    
    Args:
        action: Action to perform (e.g., "move", "expression", "speak")
        **kwargs: Action-specific arguments
        
    Returns:
        Result dictionary
    """
    try:
        from ..avatar.controller import get_avatar_controller
        
        controller = get_avatar_controller()
        
        # Execute action
        if action == "move":
            return controller.move(**kwargs)
        elif action == "expression":
            return controller.set_expression(**kwargs)
        elif action == "speak":
            return controller.speak(**kwargs)
        elif action == "gesture":
            return controller.gesture(**kwargs)
        else:
            raise ValueError(f"Unknown avatar action: {action}")
    
    except ImportError:
        raise ImportError("Avatar system not available")


# =============================================================================
# WEB TOOLS
# =============================================================================

def _use_web_tools(
    action: str,
    url: Optional[str] = None,
    query: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Use web tools (search, browse, etc.).
    
    Args:
        action: "search", "browse", "download", etc.
        url: URL for browse/download
        query: Search query
        **kwargs: Additional arguments
        
    Returns:
        Web tool result
    """
    try:
        from ..tools.web import browse_url, download_file, search_web
        
        if action == "search":
            if not query:
                raise ValueError("query required for web search")
            return search_web(query, **kwargs)
        
        elif action == "browse":
            if not url:
                raise ValueError("url required for browsing")
            return browse_url(url, **kwargs)
        
        elif action == "download":
            if not url:
                raise ValueError("url required for download")
            return download_file(url, **kwargs)
        
        else:
            raise ValueError(f"Unknown web action: {action}")
    
    except ImportError:
        raise ImportError("Web tools not available")


# =============================================================================
# FILE TOOLS
# =============================================================================

def _use_file_tools(
    action: str,
    path: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Use file tools (read, write, list, etc.).
    
    Args:
        action: "read", "write", "list", "delete", etc.
        path: File/directory path
        **kwargs: Additional arguments
        
    Returns:
        File tool result
    """
    try:
        from ..tools.file_tools import (
            delete_file,
            list_directory,
            read_file,
            write_file,
        )
        
        if action == "read":
            if not path:
                raise ValueError("path required for read")
            return read_file(path, **kwargs)
        
        elif action == "write":
            if not path:
                raise ValueError("path required for write")
            content = kwargs.pop("content", "")
            return write_file(path, content, **kwargs)
        
        elif action == "list":
            if not path:
                raise ValueError("path required for list")
            return list_directory(path, **kwargs)
        
        elif action == "delete":
            if not path:
                raise ValueError("path required for delete")
            return delete_file(path, **kwargs)
        
        else:
            raise ValueError(f"Unknown file action: {action}")
    
    except ImportError:
        raise ImportError("File tools not available")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_available_tools() -> list[str]:
    """
    Get list of all available tools.
    
    Returns:
        List of tool names
    """
    return [
        "image",
        "vision",
        "code",
        "video",
        "audio",
        "tts",
        "stt",
        "embed",
        "3d",
        "gif",
        "avatar",
        "web",
        "file",
    ]


def get_tool_info(tool_name: str) -> dict[str, Any]:
    """
    Get information about a tool.
    
    Args:
        tool_name: Name of tool
        
    Returns:
        Dictionary with tool information
    """
    tool_info = {
        "image": {
            "description": "Generate images from text descriptions",
            "parameters": ["prompt", "width", "height", "provider"],
        },
        "vision": {
            "description": "Analyze images and answer questions about them",
            "parameters": ["image_path", "image", "question"],
        },
        "code": {
            "description": "Generate code from descriptions",
            "parameters": ["prompt", "language", "provider"],
        },
        "video": {
            "description": "Generate video clips from text",
            "parameters": ["prompt", "duration", "provider", "output_path"],
        },
        "audio": {
            "description": "Generate audio/music from text",
            "parameters": ["prompt", "provider", "output_path"],
        },
        "tts": {
            "description": "Convert text to speech",
            "parameters": ["text", "output_file", "voice"],
        },
        "stt": {
            "description": "Convert speech to text",
            "parameters": ["audio_file", "duration"],
        },
        "embed": {
            "description": "Create text embeddings for semantic search",
            "parameters": ["text", "provider"],
        },
        "3d": {
            "description": "Generate 3D models from text",
            "parameters": ["prompt", "provider", "output_path"],
        },
        "avatar": {
            "description": "Control avatar without LLM",
            "parameters": ["action", "..."],
        },
        "web": {
            "description": "Web search, browse, download",
            "parameters": ["action", "url", "query"],
        },
        "file": {
            "description": "File operations (read, write, list, delete)",
            "parameters": ["action", "path", "content"],
        },
    }
    
    return tool_info.get(tool_name.lower(), {})
