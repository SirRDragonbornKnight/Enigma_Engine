#!/usr/bin/env python3
"""
ForgeAI Voice Cloning Example
==============================

Complete example showing how to use voice cloning including:
- Clone a voice from audio samples
- Generate speech with cloned voice
- Voice profiles and management
- Real-time voice conversion

Voice cloning allows you to create AI voices that sound like specific
people from just a few seconds of audio.

Dependencies:
    pip install TTS  # Coqui TTS for voice cloning
    pip install torch torchaudio  # For audio processing

Run: python examples/voice_clone_example.py
"""

import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# Voice Clone Configuration
# =============================================================================

@dataclass
class VoiceProfile:
    """A cloned voice profile."""
    name: str
    description: str
    sample_paths: List[str]
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    quality_rating: float = 0.0


@dataclass  
class CloneConfig:
    """Voice cloning configuration."""
    output_dir: str = "outputs/voice"
    sample_rate: int = 22050
    min_sample_duration: float = 3.0  # Minimum seconds of audio
    recommended_sample_duration: float = 10.0
    output_format: str = "wav"


# =============================================================================
# Voice Cloner (Simulated)
# =============================================================================

class VoiceCloner:
    """
    Voice cloning system.
    
    Uses Coqui TTS or similar for voice cloning from audio samples.
    Can clone a voice from as little as 3-10 seconds of audio.
    """
    
    def __init__(self, config: Optional[CloneConfig] = None):
        self.config = config or CloneConfig()
        self.model = None
        self.is_loaded = False
        
        self.profiles: Dict[str, VoiceProfile] = {}
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _log(self, message: str):
        """Log message."""
        print(f"[VoiceCloner] {message}")
    
    def load_model(self) -> bool:
        """Load voice cloning model."""
        self._log("Loading voice cloning model...")
        
        try:
            from TTS.api import TTS
            
            # Load a voice cloning model
            self.model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
            self.is_loaded = True
            self._log("Model loaded successfully")
            return True
            
        except ImportError:
            self._log("TTS not installed. Install with: pip install TTS")
            self._log("Using simulated mode")
            self.is_loaded = True  # Simulate for demo
            return True
        except Exception as e:
            self._log(f"Error loading model: {e}")
            return False
    
    def clone_voice(self, name: str, audio_paths: List[str], 
                    description: str = "") -> Optional[VoiceProfile]:
        """
        Clone a voice from audio samples.
        
        Args:
            name: Name for this voice profile
            audio_paths: List of paths to audio samples
            description: Description of the voice
            
        Returns:
            VoiceProfile on success, None on failure
        """
        if not self.is_loaded:
            self._log("Model not loaded")
            return None
        
        self._log(f"Cloning voice '{name}' from {len(audio_paths)} samples...")
        
        # Validate samples
        valid_samples = []
        for path in audio_paths:
            if Path(path).exists():
                valid_samples.append(path)
            else:
                self._log(f"  Warning: Sample not found: {path}")
        
        if not valid_samples:
            self._log("No valid samples found")
            return None
        
        # Extract voice embedding (simulated)
        self._log("Extracting voice embedding...")
        time.sleep(0.1)  # Simulate processing
        
        # Create voice profile
        profile = VoiceProfile(
            name=name,
            description=description,
            sample_paths=valid_samples,
            embedding=[0.1, 0.2, 0.3],  # Simulated embedding
            quality_rating=0.85
        )
        
        self.profiles[name] = profile
        self._log(f"Voice profile '{name}' created (quality: {profile.quality_rating:.2f})")
        
        return profile
    
    def generate_speech(self, text: str, voice_name: str,
                        output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate speech using a cloned voice.
        
        Args:
            text: Text to speak
            voice_name: Name of voice profile to use
            output_path: Optional output path
            
        Returns:
            Path to generated audio file
        """
        if voice_name not in self.profiles:
            self._log(f"Voice profile '{voice_name}' not found")
            return None
        
        profile = self.profiles[voice_name]
        
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"{self.config.output_dir}/speech_{voice_name}_{timestamp}.{self.config.output_format}"
        
        self._log(f"Generating speech with voice '{voice_name}'...")
        self._log(f"  Text: '{text[:50]}...'")
        
        # Generate speech (simulated)
        # In real implementation:
        # self.model.tts_to_file(
        #     text=text,
        #     speaker_wav=profile.sample_paths[0],
        #     file_path=output_path
        # )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()  # Create placeholder
        
        self._log(f"  Saved to: {output_path}")
        return output_path
    
    def list_voices(self) -> List[str]:
        """List available voice profiles."""
        return list(self.profiles.keys())
    
    def get_voice_info(self, name: str) -> Optional[Dict]:
        """Get information about a voice profile."""
        if name not in self.profiles:
            return None
        
        profile = self.profiles[name]
        return {
            "name": profile.name,
            "description": profile.description,
            "num_samples": len(profile.sample_paths),
            "quality": profile.quality_rating,
            "created": profile.created_at
        }
    
    def delete_voice(self, name: str) -> bool:
        """Delete a voice profile."""
        if name in self.profiles:
            del self.profiles[name]
            self._log(f"Deleted voice profile: {name}")
            return True
        return False


# =============================================================================
# Voice Converter (Real-time)
# =============================================================================

class VoiceConverter:
    """
    Real-time voice conversion.
    
    Converts live microphone input to a different voice in real-time.
    Useful for live streaming, gaming, etc.
    """
    
    def __init__(self, cloner: VoiceCloner):
        self.cloner = cloner
        self.is_converting = False
        self.current_voice: Optional[str] = None
    
    def _log(self, message: str):
        print(f"[VoiceConverter] {message}")
    
    def start_conversion(self, voice_name: str) -> bool:
        """
        Start real-time voice conversion.
        
        Args:
            voice_name: Voice profile to convert to
            
        Returns:
            True if conversion started
        """
        if voice_name not in self.cloner.profiles:
            self._log(f"Voice '{voice_name}' not found")
            return False
        
        self._log(f"Starting real-time conversion to '{voice_name}'...")
        self._log("Speak into your microphone - output will be in the target voice")
        
        self.is_converting = True
        self.current_voice = voice_name
        
        # In real implementation, this would:
        # 1. Capture microphone input
        # 2. Process audio in chunks
        # 3. Convert each chunk to target voice
        # 4. Output to speakers
        
        return True
    
    def stop_conversion(self):
        """Stop real-time voice conversion."""
        self.is_converting = False
        self.current_voice = None
        self._log("Stopped voice conversion")
    
    def get_status(self) -> Dict:
        """Get converter status."""
        return {
            "is_converting": self.is_converting,
            "current_voice": self.current_voice
        }


# =============================================================================
# Voice Manager (Profiles and Presets)
# =============================================================================

class VoiceManager:
    """
    Manage voice profiles and presets.
    
    Provides a higher-level interface for voice cloning with
    profile persistence, presets, and batch operations.
    """
    
    # Built-in voice presets (would be actual trained voices)
    PRESETS = {
        "narrator": "Deep, calm narrator voice",
        "assistant": "Friendly AI assistant voice",
        "character_deep": "Deep male character voice",
        "character_high": "High female character voice",
    }
    
    def __init__(self, data_dir: str = "data/voice_profiles"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cloner = VoiceCloner()
        self.converter = VoiceConverter(self.cloner)
    
    def _log(self, message: str):
        print(f"[VoiceManager] {message}")
    
    def initialize(self) -> bool:
        """Initialize the voice system."""
        self._log("Initializing voice system...")
        return self.cloner.load_model()
    
    def clone_from_youtube(self, name: str, youtube_url: str) -> Optional[VoiceProfile]:
        """
        Clone voice from YouTube video.
        
        Downloads audio from YouTube video and uses it for voice cloning.
        """
        self._log(f"Cloning voice from YouTube: {youtube_url}")
        
        # In real implementation:
        # 1. Download YouTube video audio
        # 2. Extract speech segments
        # 3. Pass to cloner
        
        # Simulated
        return self.cloner.clone_voice(
            name=name,
            audio_paths=["simulated_youtube_audio.wav"],
            description=f"Cloned from YouTube: {youtube_url}"
        )
    
    def clone_from_microphone(self, name: str, 
                              duration: float = 10.0) -> Optional[VoiceProfile]:
        """
        Clone voice from live microphone recording.
        
        Records audio from microphone for specified duration.
        """
        self._log(f"Recording {duration}s of audio from microphone...")
        self._log("Please speak clearly into your microphone")
        
        # Simulate recording
        time.sleep(0.5)  # Would actually record
        
        recording_path = f"{self.data_dir}/recording_{name}.wav"
        
        return self.cloner.clone_voice(
            name=name,
            audio_paths=[recording_path],
            description="Cloned from microphone"
        )
    
    def use_preset(self, preset_name: str) -> bool:
        """Load a built-in voice preset."""
        if preset_name not in self.PRESETS:
            self._log(f"Unknown preset: {preset_name}")
            self._log(f"Available: {list(self.PRESETS.keys())}")
            return False
        
        self._log(f"Loading preset: {preset_name}")
        self._log(f"  Description: {self.PRESETS[preset_name]}")
        
        # In real implementation, load pre-trained voice
        return True
    
    def quick_speak(self, text: str, voice: str = "assistant") -> Optional[str]:
        """
        Quickly generate speech using a voice.
        
        Args:
            text: Text to speak
            voice: Voice name or preset
            
        Returns:
            Path to generated audio
        """
        # Use preset if not a custom voice
        if voice in self.PRESETS and voice not in self.cloner.profiles:
            self.use_preset(voice)
        
        return self.cloner.generate_speech(text, voice)


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_cloning():
    """Basic voice cloning from audio file."""
    print("\n" + "="*60)
    print("Example 1: Basic Voice Cloning")
    print("="*60)
    
    cloner = VoiceCloner()
    cloner.load_model()
    
    # Clone voice from samples
    profile = cloner.clone_voice(
        name="my_voice",
        audio_paths=[
            "samples/voice_sample_1.wav",
            "samples/voice_sample_2.wav"
        ],
        description="My custom voice clone"
    )
    
    if profile:
        print(f"\nCreated voice profile: {profile.name}")
        print(f"Quality rating: {profile.quality_rating:.2f}")
        
        # Generate speech
        output = cloner.generate_speech(
            text="Hello! This is my cloned voice speaking.",
            voice_name="my_voice"
        )
        print(f"Generated audio: {output}")


def example_voice_profiles():
    """Managing multiple voice profiles."""
    print("\n" + "="*60)
    print("Example 2: Voice Profile Management")
    print("="*60)
    
    cloner = VoiceCloner()
    cloner.load_model()
    
    # Create multiple profiles
    cloner.clone_voice("voice_a", ["sample_a.wav"], "First voice")
    cloner.clone_voice("voice_b", ["sample_b.wav"], "Second voice")
    cloner.clone_voice("voice_c", ["sample_c.wav"], "Third voice")
    
    # List profiles
    print(f"\nAvailable voices: {cloner.list_voices()}")
    
    # Get info
    for name in cloner.list_voices():
        info = cloner.get_voice_info(name)
        print(f"  - {info['name']}: {info['description']}")
    
    # Delete a profile
    cloner.delete_voice("voice_b")
    print(f"\nAfter deletion: {cloner.list_voices()}")


def example_real_time():
    """Real-time voice conversion."""
    print("\n" + "="*60)
    print("Example 3: Real-time Voice Conversion")
    print("="*60)
    
    cloner = VoiceCloner()
    cloner.load_model()
    cloner.clone_voice("target_voice", ["target.wav"], "Voice to convert to")
    
    converter = VoiceConverter(cloner)
    
    print("\nStarting real-time conversion...")
    print("Your voice will be converted to the target voice in real-time")
    
    converter.start_conversion("target_voice")
    print(f"Status: {converter.get_status()}")
    
    time.sleep(0.5)  # Simulated conversion time
    
    converter.stop_conversion()
    print(f"Status: {converter.get_status()}")


def example_voice_manager():
    """High-level voice management."""
    print("\n" + "="*60)
    print("Example 4: Voice Manager")
    print("="*60)
    
    manager = VoiceManager()
    manager.initialize()
    
    print("\nAvailable presets:")
    for name, desc in VoiceManager.PRESETS.items():
        print(f"  - {name}: {desc}")
    
    # Use preset
    manager.use_preset("narrator")
    
    # Clone from YouTube (simulated)
    manager.clone_from_youtube(
        "celebrity_voice",
        "https://youtube.com/watch?v=example"
    )
    
    # Quick speak
    output = manager.quick_speak(
        "Welcome to ForgeAI voice cloning!",
        voice="assistant"
    )


def example_tips():
    """Tips for better voice cloning."""
    print("\n" + "="*60)
    print("Example 5: Tips for Better Voice Cloning")
    print("="*60)
    
    print("""
Tips for High-Quality Voice Cloning:

1. Audio Sample Quality:
   - Use clean audio with minimal background noise
   - Record in quiet environment
   - Use good microphone if possible
   - Sample rate: 22050Hz or higher

2. Sample Duration:
   - Minimum: 3 seconds
   - Recommended: 10-30 seconds
   - More samples = better quality

3. Speaking Style:
   - Speak naturally and clearly
   - Include variety (questions, statements)
   - Multiple emotional tones help

4. Technical:
   - WAV format preferred
   - Mono audio works best
   - Normalize audio levels

5. Legal/Ethical:
   - Only clone voices you have permission to use
   - Don't use for impersonation/fraud
   - Be transparent about AI-generated audio
""")


def example_forge_integration():
    """ForgeAI integration."""
    print("\n" + "="*60)
    print("Example 6: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI voice cloning:")
    print("""
    from forge_ai.voice.voice_clone import VoiceCloner, VoiceProfile
    from forge_ai.gui.tabs.voice_clone_tab import VoiceCloneTab
    
    # Programmatic usage
    cloner = VoiceCloner()
    cloner.load_model()
    
    # Clone from audio file
    profile = cloner.clone_voice(
        name="my_voice",
        audio_paths=["samples/my_recording.wav"]
    )
    
    # Generate speech
    audio_path = cloner.generate_speech(
        text="Hello from my cloned voice!",
        voice_name="my_voice"
    )
    
    # GUI usage - Voice Clone tab in ForgeAI
    python run.py --gui
    # Go to Voice Clone tab to:
    # - Record voice samples from microphone
    # - Upload audio files
    # - Manage voice profiles
    # - Generate speech with any voice
    # - Real-time voice conversion
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Voice Cloning Examples")
    print("="*60)
    
    example_basic_cloning()
    example_voice_profiles()
    example_real_time()
    example_voice_manager()
    example_tips()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("Voice Cloning Summary:")
    print("="*60)
    print("""
Voice Cloning Workflow:

1. Collect Samples:
   - Record 10-30 seconds of audio
   - Use clear, noise-free recordings
   - Multiple samples improve quality

2. Create Voice Profile:
   cloner.clone_voice("name", ["sample.wav"])

3. Generate Speech:
   cloner.generate_speech("text", "name")

4. Optional - Real-time Conversion:
   converter.start_conversion("voice_name")
   # Speak into microphone - output is converted

Key Components:

- VoiceCloner: Core cloning functionality
- VoiceProfile: Stores voice embeddings
- VoiceConverter: Real-time conversion
- VoiceManager: High-level profile management

Supported Backends:
- Coqui TTS (recommended)
- ElevenLabs API
- Custom voice models

Requirements:
   pip install TTS torch torchaudio
   
GUI:
   python run.py --gui
   # Use Voice Clone tab for graphical interface
""")
