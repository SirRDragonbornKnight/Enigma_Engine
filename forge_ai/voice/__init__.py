"""
Voice Package - Unified TTS and STT interface with voice customization.

Usage:
    from forge_ai.voice import speak, listen, set_voice
    
    # Basic usage
    speak("Hello!")
    text = listen()
    
    # Use a character voice
    set_voice("glados")
    speak("Hello, test subject.")
    
    # Create custom voice
    from forge_ai.voice import VoiceProfile, VoiceEngine
    
    my_voice = VoiceProfile(
        name="MyCharacter",
        pitch=0.8,      # Lower pitch
        speed=1.1,      # Slightly faster
        voice="female"
    )
    my_voice.save()
    set_voice("MyCharacter")
    
    # AI voice discovery
    from forge_ai.voice import AIVoiceIdentity, discover_voice
    from forge_ai.core.personality import load_personality
    
    personality = load_personality("my_model")
    voice_profile = discover_voice(personality)
    
    # Dynamic voice adaptation
    from forge_ai.voice import adapt_voice_for_emotion
    
    happy_voice = adapt_voice_for_emotion("happy", my_voice)

Available Preset Voices:
    - default   : Standard neutral voice
    - glados    : Cold, sarcastic AI (Portal)
    - jarvis    : Formal British assistant
    - wheatley  : Nervous, rambling AI
    - hal9000   : Calm, deliberate AI
    - robot     : Classic robotic voice
    - cheerful  : Happy, energetic
    - wise      : Slow, thoughtful mentor

Components:
    - tts_simple.py: Basic text-to-speech
    - stt_simple.py: Speech-to-text (SpeechRecognition)
    - voice_profile.py: Voice customization system
    - voice_generator.py: AI voice generation from personality
    - voice_identity.py: AI voice self-discovery
    - voice_effects.py: Enhanced voice effects system
    - dynamic_adapter.py: Context-aware voice adaptation
    - voice_customizer.py: User customization tools
    - audio_analyzer.py: Audio analysis for cloning
    - whisper_stt.py: High-quality Whisper STT (optional)
    - natural_tts.py: Natural TTS with Coqui/Bark (optional)
    - trigger_phrases.py: Wake word detection
"""

from .stt_simple import transcribe_from_mic as listen
from .voice_profile import (
    VoiceProfile,
    VoiceEngine,
    speak,
    set_voice,
    get_engine,
    list_presets,
    list_custom_profiles,
    list_system_voices,
    PRESET_PROFILES,
)

# Voice generation and cloning
from .voice_generator import (
    AIVoiceGenerator,
    generate_voice_for_personality,
    create_voice_from_samples,
)

# AI voice identity and self-discovery
from .voice_identity import (
    AIVoiceIdentity,
    discover_voice,
    describe_voice,
    adjust_voice_from_feedback,
)

# Voice effects system
from .voice_effects import (
    VoiceEffects,
    apply_effect,
    apply_effects,
    effect_for_emotion,
    effect_for_context,
)

# Dynamic voice adaptation
from .dynamic_adapter import (
    DynamicVoiceAdapter,
    adapt_voice_for_emotion,
    adapt_voice_for_context,
    adapt_voice_for_personality,
)

# User customization tools
from .voice_customizer import (
    VoiceCustomizer,
    interactive_tuning,
    import_voice_profile,
    export_voice_profile,
    compare_voices,
)

# Audio analysis
from .audio_analyzer import (
    AudioAnalyzer,
    analyze_audio,
    estimate_voice_profile,
    compare_voices as compare_voice_audio,
)

# Wake word detection
from .trigger_phrases import (
    TriggerPhraseDetector,
    SmartWakeWords,
    suggest_wake_phrases,
    train_custom_wake_phrase,
    start_wake_word_detection,
    stop_wake_word_detection,
    is_wake_word_active,
)

# New voice options (optional dependencies)
from .whisper_stt import WhisperSTT
from .natural_tts import NaturalTTS

__all__ = [
    # Basic TTS/STT
    'speak',
    'listen', 
    'set_voice',
    'VoiceProfile',
    'VoiceEngine',
    'get_engine',
    'list_presets',
    'list_custom_profiles',
    'list_system_voices',
    'PRESET_PROFILES',
    
    # Voice generation
    'AIVoiceGenerator',
    'generate_voice_for_personality',
    'create_voice_from_samples',
    
    # AI voice identity
    'AIVoiceIdentity',
    'discover_voice',
    'describe_voice',
    'adjust_voice_from_feedback',
    
    # Voice effects
    'VoiceEffects',
    'apply_effect',
    'apply_effects',
    'effect_for_emotion',
    'effect_for_context',
    
    # Dynamic adaptation
    'DynamicVoiceAdapter',
    'adapt_voice_for_emotion',
    'adapt_voice_for_context',
    'adapt_voice_for_personality',
    
    # User customization
    'VoiceCustomizer',
    'interactive_tuning',
    'import_voice_profile',
    'export_voice_profile',
    'compare_voices',
    
    # Audio analysis
    'AudioAnalyzer',
    'analyze_audio',
    'estimate_voice_profile',
    'compare_voice_audio',
    
    # Wake words
    'TriggerPhraseDetector',
    'SmartWakeWords',
    'suggest_wake_phrases',
    'train_custom_wake_phrase',
    'start_wake_word_detection',
    'stop_wake_word_detection',
    'is_wake_word_active',
    
    # Optional advanced features
    'WhisperSTT',
    'NaturalTTS',
    
    # Voice pipeline (unified voice I/O)
    'VoicePipeline',
    'VoiceConfig',
    'VoiceMode',
    'VoiceDevice',
    'SpeechSegment',
    'get_voice_pipeline',
]

# Voice pipeline for unified STT/TTS
try:
    from .voice_pipeline import (
        VoicePipeline, VoiceConfig, VoiceMode, VoiceDevice, SpeechSegment,
        get_voice_pipeline,
    )
except ImportError:
    VoicePipeline = None
    VoiceConfig = None
    VoiceMode = None
    VoiceDevice = None
    SpeechSegment = None
    get_voice_pipeline = None

