#!/usr/bin/env python3
"""
ForgeAI Video Generation Example
=================================

Complete example showing how to generate videos using ForgeAI including:
- Local video generation with AnimateDiff
- Cloud generation via Replicate
- Built-in GIF fallback for testing
- Video from image sequences

The video generation system supports multiple backends with automatic
fallback for systems without GPU.

Dependencies:
    pip install diffusers torch  # For local AnimateDiff
    pip install replicate        # For cloud generation
    pip install pillow imageio   # For GIF/video processing

Run: python examples/video_example.py
"""

import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VideoConfig:
    """Video generation configuration."""
    width: int = 512
    height: int = 512
    num_frames: int = 16
    fps: int = 8
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    output_dir: str = "outputs/videos"


# =============================================================================
# Local Video Generation (Simulated)
# =============================================================================

class LocalVideoGenerator:
    """
    Local video generation using AnimateDiff.
    
    AnimateDiff converts Stable Diffusion into a video generator
    by adding motion modules. Requires CUDA GPU for reasonable speed.
    """
    
    def __init__(self, model_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"):
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False
        self._use_builtin = True  # Use built-in fallback
    
    def load(self) -> bool:
        """Load the AnimateDiff pipeline."""
        print("Attempting to load AnimateDiff...")
        
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
            import torch
            
            print("Loading motion adapter...")
            adapter = MotionAdapter.from_pretrained(self.model_id)
            
            print("Loading base model...")
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            self.pipe = AnimateDiffPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
            )
            
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                beta_schedule="linear"
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("Using CUDA GPU")
            else:
                print("No CUDA - generation will be slow")
            
            self.is_loaded = True
            self._use_builtin = False
            return True
            
        except ImportError as e:
            print(f"AnimateDiff not available: {e}")
            print("Using built-in GIF fallback")
            self._use_builtin = True
            return True
        except Exception as e:
            print(f"Error loading AnimateDiff: {e}")
            self._use_builtin = True
            return True
    
    def generate(self, prompt: str, config: VideoConfig) -> Optional[str]:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description of video to generate
            config: Video generation config
            
        Returns:
            Path to generated video file, or None on failure
        """
        # Ensure output directory exists
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        if self._use_builtin:
            return self._generate_builtin(prompt, config, timestamp)
        else:
            return self._generate_animatediff(prompt, config, timestamp)
    
    def _generate_animatediff(self, prompt: str, config: VideoConfig, 
                              timestamp: int) -> Optional[str]:
        """Generate using actual AnimateDiff."""
        output_path = Path(config.output_dir) / f"video_{timestamp}.gif"
        
        print(f"Generating video: '{prompt}'")
        print(f"  Size: {config.width}x{config.height}, Frames: {config.num_frames}")
        
        result = self.pipe(
            prompt=prompt,
            num_frames=config.num_frames,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            height=config.height,
            width=config.width,
        )
        
        # Export frames
        from diffusers.utils import export_to_gif
        export_to_gif(result.frames[0], str(output_path))
        
        print(f"Video saved to: {output_path}")
        return str(output_path)
    
    def _generate_builtin(self, prompt: str, config: VideoConfig,
                          timestamp: int) -> Optional[str]:
        """Generate simple animated GIF as fallback."""
        output_path = Path(config.output_dir) / f"video_{timestamp}.gif"
        
        print(f"[Simulated] Generating video: '{prompt}'")
        print(f"  Using built-in GIF generator (no GPU needed)")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            frames = []
            for i in range(config.num_frames):
                # Create frame with text and animation
                img = Image.new('RGB', (config.width, config.height), 
                               color=(30, 30, 40))
                draw = ImageDraw.Draw(img)
                
                # Animated circle
                x = int(50 + (config.width - 100) * (i / config.num_frames))
                y = config.height // 2 + int(30 * (1 if i % 2 else -1))
                draw.ellipse([x-20, y-20, x+20, y+20], fill=(100, 150, 255))
                
                # Prompt text
                text = prompt[:40] + "..." if len(prompt) > 40 else prompt
                draw.text((10, 10), text, fill=(255, 255, 255))
                draw.text((10, config.height - 30), f"Frame {i+1}/{config.num_frames}", 
                         fill=(150, 150, 150))
                
                frames.append(img)
            
            # Save as GIF
            frames[0].save(
                str(output_path),
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / config.fps),
                loop=0
            )
            
            print(f"GIF saved to: {output_path}")
            return str(output_path)
            
        except ImportError:
            print("PIL not installed. Install with: pip install pillow")
            return None


# =============================================================================
# Cloud Video Generation (Replicate)
# =============================================================================

class ReplicateVideoGenerator:
    """
    Cloud video generation using Replicate.
    
    Uses Replicate's API to generate videos on cloud GPUs.
    No local GPU required, but needs API key and costs money.
    """
    
    def __init__(self, model: str = "stability-ai/stable-video-diffusion"):
        self.model = model
        self.api_key = os.environ.get("REPLICATE_API_TOKEN")
    
    def is_available(self) -> bool:
        """Check if Replicate is configured."""
        return bool(self.api_key)
    
    def generate(self, prompt: str, config: VideoConfig) -> Optional[str]:
        """
        Generate video via Replicate API.
        
        Args:
            prompt: Text description or image URL
            config: Video configuration
            
        Returns:
            Path to downloaded video, or None on failure
        """
        if not self.api_key:
            print("REPLICATE_API_TOKEN not set")
            print("Get key from: https://replicate.com/account/api-tokens")
            return None
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        
        try:
            import replicate
            
            print(f"Generating video via Replicate: '{prompt}'")
            print(f"  Model: {self.model}")
            
            # Run prediction
            output = replicate.run(
                self.model,
                input={
                    "prompt": prompt,
                    "video_length": config.num_frames,
                    "width": config.width,
                    "height": config.height,
                    "fps": config.fps,
                }
            )
            
            # Download result
            if output:
                import urllib.request
                output_path = output_dir / f"video_{timestamp}.mp4"
                
                video_url = output[0] if isinstance(output, list) else output
                urllib.request.urlretrieve(video_url, str(output_path))
                
                print(f"Video saved to: {output_path}")
                return str(output_path)
            
            return None
            
        except ImportError:
            print("replicate package not installed")
            print("Install with: pip install replicate")
            return None
        except Exception as e:
            print(f"Replicate error: {e}")
            return None


# =============================================================================
# Image Sequence to Video
# =============================================================================

class ImageSequenceConverter:
    """Convert a sequence of images to video."""
    
    @staticmethod
    def images_to_video(image_paths: List[str], output_path: str,
                        fps: int = 8) -> Optional[str]:
        """
        Convert image sequence to video/GIF.
        
        Args:
            image_paths: List of paths to images
            output_path: Output video path (.gif or .mp4)
            fps: Frames per second
            
        Returns:
            Output path on success, None on failure
        """
        if not image_paths:
            print("No images provided")
            return None
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from PIL import Image
            
            frames = []
            for path in image_paths:
                img = Image.open(path)
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                frames.append(img)
            
            if output.suffix.lower() == '.gif':
                frames[0].save(
                    str(output),
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / fps),
                    loop=0
                )
            else:
                # Use imageio for mp4
                try:
                    import imageio
                    import numpy as np
                    
                    writer = imageio.get_writer(str(output), fps=fps)
                    for frame in frames:
                        writer.append_data(np.array(frame))
                    writer.close()
                except ImportError:
                    print("imageio not installed, saving as GIF instead")
                    gif_path = output.with_suffix('.gif')
                    frames[0].save(
                        str(gif_path),
                        save_all=True,
                        append_images=frames[1:],
                        duration=int(1000 / fps),
                        loop=0
                    )
                    return str(gif_path)
            
            print(f"Video saved to: {output}")
            return str(output)
            
        except ImportError:
            print("PIL not installed. Install with: pip install pillow")
            return None
        except Exception as e:
            print(f"Error creating video: {e}")
            return None
    
    @staticmethod
    def video_to_frames(video_path: str, output_dir: str) -> List[str]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            
        Returns:
            List of paths to extracted frames
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        
        try:
            from PIL import Image
            
            # Try to open as GIF
            img = Image.open(video_path)
            
            frame_num = 0
            while True:
                try:
                    img.seek(frame_num)
                    frame_path = output / f"frame_{frame_num:04d}.png"
                    img.save(str(frame_path))
                    frame_paths.append(str(frame_path))
                    frame_num += 1
                except EOFError:
                    break
            
            print(f"Extracted {len(frame_paths)} frames to {output}")
            return frame_paths
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return frame_paths


# =============================================================================
# Unified Video Generator
# =============================================================================

class VideoGenerator:
    """
    Unified video generation interface.
    
    Automatically selects the best available backend:
    1. Local AnimateDiff (if GPU available)
    2. Replicate API (if key configured)
    3. Built-in GIF generator (fallback)
    """
    
    def __init__(self, prefer_local: bool = True):
        """
        Initialize video generator.
        
        Args:
            prefer_local: Prefer local generation over cloud
        """
        self.prefer_local = prefer_local
        self.local_gen = LocalVideoGenerator()
        self.cloud_gen = ReplicateVideoGenerator()
        self.converter = ImageSequenceConverter()
        
        self._backend = None
    
    def _select_backend(self) -> str:
        """Select best available backend."""
        if self.prefer_local:
            self.local_gen.load()
            if self.local_gen.is_loaded:
                return "local"
        
        if self.cloud_gen.is_available():
            return "cloud"
        
        # Fallback to built-in
        self.local_gen._use_builtin = True
        return "builtin"
    
    def generate(self, prompt: str, 
                 config: Optional[VideoConfig] = None) -> Optional[str]:
        """
        Generate video from prompt.
        
        Args:
            prompt: Text description of video
            config: Optional video configuration
            
        Returns:
            Path to generated video
        """
        config = config or VideoConfig()
        
        if self._backend is None:
            self._backend = self._select_backend()
            print(f"Using backend: {self._backend}")
        
        if self._backend == "local":
            return self.local_gen.generate(prompt, config)
        elif self._backend == "cloud":
            return self.cloud_gen.generate(prompt, config)
        else:
            return self.local_gen.generate(prompt, config)


# =============================================================================
# Example Usage
# =============================================================================

def example_local_video():
    """Generate video locally."""
    print("\n" + "="*60)
    print("Example 1: Local Video Generation")
    print("="*60)
    
    config = VideoConfig(
        width=256,
        height=256,
        num_frames=8,
        fps=4
    )
    
    generator = LocalVideoGenerator()
    generator.load()
    
    result = generator.generate(
        prompt="A cat sitting on a windowsill watching birds",
        config=config
    )
    
    if result:
        print(f"Generated: {result}")


def example_cloud_video():
    """Generate video via cloud API."""
    print("\n" + "="*60)
    print("Example 2: Cloud Video Generation (Replicate)")
    print("="*60)
    
    generator = ReplicateVideoGenerator()
    
    if not generator.is_available():
        print("Replicate not configured. Set REPLICATE_API_TOKEN environment variable.")
        print("Get your token from: https://replicate.com/account/api-tokens")
        return
    
    config = VideoConfig(
        width=512,
        height=512,
        num_frames=16,
        fps=8
    )
    
    result = generator.generate(
        prompt="A rocket launching into space with flames and smoke",
        config=config
    )
    
    if result:
        print(f"Generated: {result}")


def example_image_sequence():
    """Create video from image sequence."""
    print("\n" + "="*60)
    print("Example 3: Image Sequence to Video")
    print("="*60)
    
    # Create some test images
    try:
        from PIL import Image, ImageDraw
        
        test_dir = Path("outputs/test_frames")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for i in range(8):
            img = Image.new('RGB', (256, 256), color=(30, 30, 40))
            draw = ImageDraw.Draw(img)
            
            # Moving circle
            x = 30 + i * 25
            draw.ellipse([x, 100, x+50, 150], fill=(100, 200, 100))
            draw.text((10, 10), f"Frame {i+1}", fill=(255, 255, 255))
            
            path = test_dir / f"frame_{i:04d}.png"
            img.save(str(path))
            image_paths.append(str(path))
        
        print(f"Created {len(image_paths)} test frames")
        
        # Convert to video
        converter = ImageSequenceConverter()
        result = converter.images_to_video(
            image_paths,
            "outputs/videos/sequence_test.gif",
            fps=4
        )
        
        if result:
            print(f"Video created: {result}")
            
    except ImportError:
        print("PIL not installed. Install with: pip install pillow")


def example_unified():
    """Use unified generator with automatic backend selection."""
    print("\n" + "="*60)
    print("Example 4: Unified Video Generator")
    print("="*60)
    
    generator = VideoGenerator(prefer_local=True)
    
    config = VideoConfig(
        width=256,
        height=256,
        num_frames=8,
        fps=4
    )
    
    result = generator.generate(
        prompt="Ocean waves crashing on a beach at sunset",
        config=config
    )
    
    if result:
        print(f"Generated: {result}")


def example_forge_integration():
    """Using actual ForgeAI video generation."""
    print("\n" + "="*60)
    print("Example 5: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI video generation:")
    print("""
    from forge_ai.gui.tabs.video_tab import LocalVideo, ReplicateVideo
    
    # Local generation (needs GPU)
    local = LocalVideo()
    if local.load():
        result = local.generate(
            prompt="A butterfly flying through a garden",
            width=512,
            height=512,
            num_frames=16
        )
    
    # Cloud generation (needs API key)
    cloud = ReplicateVideo()
    result = cloud.generate(
        prompt="A butterfly flying through a garden",
        width=512,
        height=512,
        num_frames=16
    )
    
    # The GUI tab provides:
    # - Provider selection (local/cloud)
    # - Frame count and FPS controls
    # - Resolution settings
    # - Progress tracking
    # - Output preview
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Video Generation Examples")
    print("="*60)
    
    example_local_video()
    example_image_sequence()
    example_unified()
    example_cloud_video()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("Video Generation Summary:")
    print("="*60)
    print("""
Available Backends:

1. Local (AnimateDiff):
   - Requires: CUDA GPU, torch, diffusers
   - Quality: High
   - Speed: 1-5 minutes per video
   - Cost: Free (electricity)

2. Cloud (Replicate):
   - Requires: API key, internet
   - Quality: High  
   - Speed: 10-60 seconds
   - Cost: ~$0.05-0.50 per video

3. Built-in (GIF Fallback):
   - Requires: PIL only
   - Quality: Low (placeholder)
   - Speed: Instant
   - Cost: Free

Video Config Options:
   - width/height: Video resolution
   - num_frames: Number of frames (8-64)
   - fps: Playback speed
   - guidance_scale: How closely to follow prompt
   - num_inference_steps: Quality vs speed

For ForgeAI GUI:
    python run.py --gui
    # Then use the Video tab
""")
