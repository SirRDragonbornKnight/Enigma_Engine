"""
Media Tools - Music generation, background removal, upscaling, style transfer.

Tools:
  - music_generate: Generate music/MIDI
  - remove_background: Remove image backgrounds
  - upscale_image: AI upscaling for images
  - style_transfer: Apply art styles to images
  - convert_audio: Convert audio between formats
  - extract_audio: Extract audio from video
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .tool_registry import Tool, RichParameter

# Output directories
MUSIC_OUTPUT_DIR = Path.home() / ".enigma_engine" / "outputs" / "music"
IMAGE_OUTPUT_DIR = Path.home() / ".enigma_engine" / "outputs" / "images"
AUDIO_OUTPUT_DIR = Path.home() / ".enigma_engine" / "outputs" / "audio"

MUSIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MUSIC GENERATION TOOLS
# ============================================================================

class MusicGenerateTool(Tool):
    """Generate music from text descriptions."""
    
    name = "music_generate"
    description = "Generate music or audio from a text description. Creates WAV/MP3 files."
    parameters = {
        "prompt": "Description of the music (e.g., 'calm piano melody', 'upbeat electronic beat')",
        "duration": "Duration in seconds (default: 10, max: 30)",
        "output_name": "Name for the output file (without extension)",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="prompt", type="string", description="Music description", required=True),
        RichParameter(name="duration", type="integer", description="Duration in seconds", required=False, default=10, min_value=1, max_value=30),
        RichParameter(name="output_name", type="string", description="Output filename (no extension)", required=False),
    ]
    examples = ["music_generate(prompt='calm piano melody')", "music_generate(prompt='upbeat electronic beat', duration=20)"]
    
    def execute(self, prompt: str, duration: int = 10, 
                output_name: str = None, **kwargs) -> dict[str, Any]:
        try:
            duration = min(int(duration), 30)  # Cap at 30 seconds
            
            if not output_name:
                output_name = f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_path = MUSIC_OUTPUT_DIR / f"{output_name}.wav"
            
            # Try audiocraft/musicgen
            try:
                from audiocraft.data.audio import audio_write
                from audiocraft.models import MusicGen

                # Use small model for faster generation
                model = MusicGen.get_pretrained('facebook/musicgen-small')
                model.set_generation_params(duration=duration)
                
                wav = model.generate([prompt])
                
                audio_write(str(output_path.with_suffix('')), wav[0].cpu(), model.sample_rate, strategy="loudness")
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "prompt": prompt,
                    "duration": duration,
                    "method": "musicgen",
                }
                
            except ImportError:
                pass
            
            # Try bark for audio generation
            try:
                from bark import SAMPLE_RATE, generate_audio, preload_models
                from scipy.io.wavfile import write as write_wav
                
                preload_models()
                
                # Bark is more for speech, but can do some music
                audio_array = generate_audio(f"♪ {prompt} ♪")
                write_wav(str(output_path), SAMPLE_RATE, audio_array)
                
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "prompt": prompt,
                    "method": "bark",
                }
                
            except ImportError:
                pass
            
            # Generate simple MIDI as fallback
            try:
                import random

                from midiutil import MIDIFile
                
                midi = MIDIFile(1)
                midi.addTempo(0, 0, 120)
                
                # Generate simple melody based on prompt keywords
                if "piano" in prompt.lower():
                    instrument = 0  # Acoustic Grand Piano
                elif "guitar" in prompt.lower():
                    instrument = 25  # Acoustic Guitar
                elif "drum" in prompt.lower():
                    instrument = 118  # Drum Kit
                else:
                    instrument = 0
                
                midi.addProgramChange(0, 0, 0, instrument)
                
                # Generate notes
                scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major
                time = 0
                for _ in range(duration * 2):  # 2 notes per second
                    pitch = random.choice(scale)
                    midi.addNote(0, 0, pitch, time, 0.5, 100)
                    time += 0.5
                
                midi_path = output_path.with_suffix('.mid')
                with open(midi_path, 'wb') as f:
                    midi.writeFile(f)
                
                return {
                    "success": True,
                    "output_path": str(midi_path),
                    "prompt": prompt,
                    "method": "midi",
                    "note": "Install audiocraft for actual music generation: pip install audiocraft",
                }
                
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "No music generation library available. Install: pip install audiocraft or pip install midiutil"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# BACKGROUND REMOVAL TOOLS
# ============================================================================

class RemoveBackgroundTool(Tool):
    """Remove background from images."""
    
    name = "remove_background"
    description = "Remove the background from an image, leaving only the subject."
    parameters = {
        "input_path": "Path to the input image",
        "output_path": "Path for the output image (default: adds '_nobg' to filename)",
        "model": "Model to use: 'u2net', 'isnet' (default: u2net)",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="input_path", type="string", description="Path to input image", required=True),
        RichParameter(name="output_path", type="string", description="Path for output image", required=False),
        RichParameter(name="model", type="string", description="Background removal model", required=False, default="u2net", enum=["u2net", "isnet"]),
    ]
    examples = ["remove_background(input_path='photo.jpg')", "remove_background(input_path='portrait.png', output_path='portrait_clean.png')"]
    
    def execute(self, input_path: str, output_path: str = None,
                model: str = "u2net", **kwargs) -> dict[str, Any]:
        try:
            input_path = Path(input_path).expanduser().resolve()
            
            if not input_path.exists():
                return {"success": False, "error": f"File not found: {input_path}"}
            
            if not output_path:
                output_path = input_path.parent / f"{input_path.stem}_nobg.png"
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            # Try rembg (most popular)
            try:
                from PIL import Image
                from rembg import remove
                
                with open(input_path, 'rb') as f:
                    input_data = f.read()
                
                output_data = remove(input_data, alpha_matting=True)
                
                with open(output_path, 'wb') as f:
                    f.write(output_data)
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "method": "rembg",
                }
                
            except ImportError:
                pass
            
            # Try backgroundremover
            try:
                from backgroundremover.bg import remove as bg_remove
                from PIL import Image
                
                img = Image.open(input_path)
                result = bg_remove(img)
                result.save(str(output_path))
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "method": "backgroundremover",
                }
                
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "No background removal library available. Install: pip install rembg"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# IMAGE UPSCALING TOOLS
# ============================================================================

class UpscaleImageTool(Tool):
    """Upscale images using AI."""
    
    name = "upscale_image"
    description = "Upscale an image to higher resolution using AI enhancement."
    parameters = {
        "input_path": "Path to the input image",
        "output_path": "Path for the output image (default: adds '_upscaled' to filename)",
        "scale": "Upscale factor: 2, 4, or 8 (default: 2)",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="input_path", type="string", description="Path to input image", required=True),
        RichParameter(name="output_path", type="string", description="Path for output image", required=False),
        RichParameter(name="scale", type="integer", description="Upscale factor", required=False, default=2, enum=[2, 4, 8]),
    ]
    examples = ["upscale_image(input_path='small.jpg')", "upscale_image(input_path='photo.png', scale=4)"]
    
    def execute(self, input_path: str, output_path: str = None,
                scale: int = 2, **kwargs) -> dict[str, Any]:
        try:
            input_path = Path(input_path).expanduser().resolve()
            
            if not input_path.exists():
                return {"success": False, "error": f"File not found: {input_path}"}
            
            if not output_path:
                output_path = input_path.parent / f"{input_path.stem}_upscaled{scale}x{input_path.suffix}"
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            scale = int(scale)
            
            # Try Real-ESRGAN
            try:
                import cv2
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model)
                
                img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
                output, _ = upsampler.enhance(img, outscale=scale)
                cv2.imwrite(str(output_path), output)
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "scale": scale,
                    "method": "realesrgan",
                }
                
            except ImportError:
                pass
            
            # Try PIL/Pillow with LANCZOS (basic upscaling)
            try:
                from PIL import Image
                
                img = Image.open(input_path)
                new_size = (img.width * scale, img.height * scale)
                upscaled = img.resize(new_size, Image.LANCZOS)
                upscaled.save(str(output_path))
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "scale": scale,
                    "method": "pillow_lanczos",
                    "note": "For better quality, install: pip install realesrgan basicsr",
                }
                
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "No upscaling library available. Install: pip install pillow"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# STYLE TRANSFER TOOLS
# ============================================================================

class StyleTransferTool(Tool):
    """Apply artistic styles to images."""
    
    name = "style_transfer"
    description = "Apply an artistic style to an image (e.g., make a photo look like a painting)."
    parameters = {
        "content_path": "Path to the content image (your photo)",
        "style": "Style: 'monet', 'vangogh', 'picasso', 'candy', 'mosaic', 'rain_princess', or path to style image",
        "output_path": "Path for output (default: adds '_styled' to filename)",
        "strength": "Style strength 0-1 (default: 0.8)",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="content_path", type="string", description="Path to content image", required=True),
        RichParameter(name="style", type="string", description="Art style to apply", required=False, default="monet", enum=["monet", "vangogh", "picasso", "candy", "mosaic", "rain_princess"]),
        RichParameter(name="output_path", type="string", description="Path for output image", required=False),
        RichParameter(name="strength", type="number", description="Style strength (0-1)", required=False, default=0.8, min_value=0, max_value=1),
    ]
    examples = ["style_transfer(content_path='photo.jpg', style='vangogh')", "style_transfer(content_path='landscape.png', style='monet', strength=0.6)"]
    
    # Pre-defined style mappings
    PRESET_STYLES = {
        'monet': 'impressionist landscape style',
        'vangogh': 'Van Gogh swirling brushstrokes',
        'picasso': 'cubist abstract style',
        'candy': 'colorful candy mosaic',
        'mosaic': 'tile mosaic pattern',
        'rain_princess': 'dark dramatic painterly',
    }
    
    def execute(self, content_path: str, style: str = "monet",
                output_path: str = None, strength: float = 0.8, **kwargs) -> dict[str, Any]:
        try:
            content_path = Path(content_path).expanduser().resolve()
            
            if not content_path.exists():
                return {"success": False, "error": f"File not found: {content_path}"}
            
            if not output_path:
                output_path = content_path.parent / f"{content_path.stem}_styled_{style}{content_path.suffix}"
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            # Try fast neural style transfer
            try:
                from PIL import Image

                # Simple color-based style transfer as fallback
                img = Image.open(content_path)
                
                # Apply style-based color adjustments
                if style == 'monet' or style == 'impressionist':
                    # Soft, pastel colors
                    from PIL import ImageEnhance
                    img = ImageEnhance.Color(img).enhance(0.7)
                    img = ImageEnhance.Brightness(img).enhance(1.1)
                elif style == 'vangogh':
                    # Higher saturation
                    from PIL import ImageEnhance
                    img = ImageEnhance.Color(img).enhance(1.5)
                    img = ImageEnhance.Contrast(img).enhance(1.2)
                elif style == 'picasso':
                    # Posterize for cubist effect
                    img = img.convert('P', palette=Image.ADAPTIVE, colors=16).convert('RGB')
                
                img.save(str(output_path))
                
                return {
                    "success": True,
                    "input_path": str(content_path),
                    "output_path": str(output_path),
                    "style": style,
                    "method": "simple_filter",
                    "note": "For true neural style transfer, install: pip install torch torchvision",
                }
                
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "No style transfer library available. Install: pip install torch torchvision pillow"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# AUDIO TOOLS
# ============================================================================

class ConvertAudioTool(Tool):
    """Convert audio between formats."""
    
    name = "convert_audio"
    description = "Convert audio files between formats (mp3, wav, ogg, flac, etc.)."
    parameters = {
        "input_path": "Path to the input audio file",
        "output_format": "Target format: 'mp3', 'wav', 'ogg', 'flac', 'aac'",
        "output_path": "Path for output (default: same name with new extension)",
        "bitrate": "Bitrate for lossy formats (default: '192k')",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="input_path", type="string", description="Path to input audio", required=True),
        RichParameter(name="output_format", type="string", description="Target format", required=True, enum=["mp3", "wav", "ogg", "flac", "aac"]),
        RichParameter(name="output_path", type="string", description="Path for output", required=False),
        RichParameter(name="bitrate", type="string", description="Bitrate for lossy formats", required=False, default="192k"),
    ]
    examples = ["convert_audio(input_path='song.wav', output_format='mp3')", "convert_audio(input_path='audio.flac', output_format='ogg', bitrate='320k')"]
    
    def execute(self, input_path: str, output_format: str,
                output_path: str = None, bitrate: str = "192k", **kwargs) -> dict[str, Any]:
        try:
            input_path = Path(input_path).expanduser().resolve()
            
            if not input_path.exists():
                return {"success": False, "error": f"File not found: {input_path}"}
            
            if not output_path:
                output_path = input_path.with_suffix(f".{output_format}")
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            # Try pydub
            try:
                from pydub import AudioSegment
                
                audio = AudioSegment.from_file(str(input_path))
                
                export_params = {}
                if output_format in ['mp3', 'ogg', 'aac']:
                    export_params['bitrate'] = bitrate
                
                audio.export(str(output_path), format=output_format, **export_params)
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "format": output_format,
                    "method": "pydub",
                }
                
            except ImportError:
                pass
            
            # Try ffmpeg directly
            try:
                cmd = ['ffmpeg', '-y', '-i', str(input_path)]
                if output_format in ['mp3', 'ogg', 'aac']:
                    cmd.extend(['-b:a', bitrate])
                cmd.append(str(output_path))
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "format": output_format,
                        "method": "ffmpeg",
                    }
                else:
                    return {"success": False, "error": result.stderr}
                    
            except FileNotFoundError:
                pass
            
            return {
                "success": False,
                "error": "No audio conversion tool available. Install: pip install pydub or apt install ffmpeg"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class ExtractAudioTool(Tool):
    """Extract audio from video files."""
    
    name = "extract_audio"
    description = "Extract the audio track from a video file."
    parameters = {
        "input_path": "Path to the video file",
        "output_path": "Path for audio output (default: same name with .mp3)",
        "format": "Audio format: 'mp3', 'wav', 'ogg' (default: mp3)",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="input_path", type="string", description="Path to video file", required=True),
        RichParameter(name="output_path", type="string", description="Path for audio output", required=False),
        RichParameter(name="format", type="string", description="Audio format", required=False, default="mp3", enum=["mp3", "wav", "ogg"]),
    ]
    examples = ["extract_audio(input_path='video.mp4')", "extract_audio(input_path='movie.mkv', format='wav')"]
    
    def execute(self, input_path: str, output_path: str = None,
                format: str = "mp3", **kwargs) -> dict[str, Any]:
        try:
            input_path = Path(input_path).expanduser().resolve()
            
            if not input_path.exists():
                return {"success": False, "error": f"File not found: {input_path}"}
            
            if not output_path:
                output_path = input_path.with_suffix(f".{format}")
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            # Try moviepy
            try:
                from moviepy.editor import VideoFileClip
                
                video = VideoFileClip(str(input_path))
                video.audio.write_audiofile(str(output_path))
                video.close()
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "format": format,
                    "method": "moviepy",
                }
                
            except ImportError:
                pass
            
            # Try ffmpeg
            try:
                cmd = ['ffmpeg', '-y', '-i', str(input_path), '-vn', '-acodec', 
                       'libmp3lame' if format == 'mp3' else 'copy', str(output_path)]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "format": format,
                        "method": "ffmpeg",
                    }
                else:
                    return {"success": False, "error": result.stderr}
                    
            except FileNotFoundError:
                pass
            
            return {
                "success": False,
                "error": "No video processing tool available. Install: pip install moviepy or apt install ffmpeg"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class AudioVisualizeTool(Tool):
    """Create audio visualization."""
    
    name = "audio_visualize"
    description = "Create a waveform or spectrogram visualization from an audio file."
    parameters = {
        "input_path": "Path to the audio file",
        "output_path": "Path for the image output",
        "style": "Visualization style: 'waveform', 'spectrogram' (default: waveform)",
        "color": "Color scheme: 'blue', 'green', 'rainbow', 'white' (default: blue)",
    }
    category = "media"
    rich_parameters = [
        RichParameter(name="input_path", type="string", description="Path to audio file", required=True),
        RichParameter(name="output_path", type="string", description="Path for image output", required=False),
        RichParameter(name="style", type="string", description="Visualization style", required=False, default="waveform", enum=["waveform", "spectrogram"]),
        RichParameter(name="color", type="string", description="Color scheme", required=False, default="blue", enum=["blue", "green", "rainbow", "white"]),
    ]
    examples = ["audio_visualize(input_path='song.mp3')", "audio_visualize(input_path='audio.wav', style='spectrogram', color='rainbow')"]
    
    def execute(self, input_path: str, output_path: str = None,
                style: str = "waveform", color: str = "blue", **kwargs) -> dict[str, Any]:
        try:
            import numpy as np
            
            input_path = Path(input_path).expanduser().resolve()
            
            if not input_path.exists():
                return {"success": False, "error": f"File not found: {input_path}"}
            
            if not output_path:
                output_path = input_path.with_suffix('.png')
            else:
                output_path = Path(output_path).expanduser().resolve()
            
            # Try librosa + matplotlib
            try:
                import librosa
                import librosa.display
                import matplotlib.pyplot as plt
                
                y, sr = librosa.load(str(input_path))
                
                plt.figure(figsize=(14, 5))
                
                color_map = {
                    'blue': 'Blues',
                    'green': 'Greens',
                    'rainbow': 'viridis',
                    'white': 'gray',
                }
                cmap = color_map.get(color, 'Blues')
                
                if style == 'spectrogram':
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap=cmap)
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Spectrogram')
                else:  # waveform
                    librosa.display.waveshow(y, sr=sr, color=color if color != 'rainbow' else 'blue')
                    plt.title('Waveform')
                
                plt.tight_layout()
                plt.savefig(str(output_path), dpi=150)
                plt.close()
                
                return {
                    "success": True,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "style": style,
                    "method": "librosa",
                }
                
            except ImportError:
                pass
            
            return {
                "success": False,
                "error": "No audio visualization library available. Install: pip install librosa matplotlib"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
