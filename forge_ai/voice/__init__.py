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

# Audio analysis
from .audio_analyzer import (
    AudioAnalyzer,
    analyze_audio,
)
from .audio_analyzer import compare_voices as compare_voice_audio
from .audio_analyzer import (
    estimate_voice_profile,
)

# Dynamic voice adaptation
from .dynamic_adapter import (
    DynamicVoiceAdapter,
    adapt_voice_for_context,
    adapt_voice_for_emotion,
    adapt_voice_for_personality,
)
from .natural_tts import NaturalTTS
from .stt_simple import transcribe_from_mic as listen

# Wake word detection
from .trigger_phrases import (
    SmartWakeWords,
    TriggerPhraseDetector,
    is_wake_word_active,
    start_wake_word_detection,
    stop_wake_word_detection,
    suggest_wake_phrases,
    train_custom_wake_phrase,
)

# User customization tools
from .voice_customizer import (
    VoiceCustomizer,
    compare_voices,
    export_voice_profile,
    import_voice_profile,
    interactive_tuning,
)

# Voice effects system
from .voice_effects import (
    VoiceEffects,
    apply_effect,
    apply_effects,
    effect_for_context,
    effect_for_emotion,
)

# Voice generation and cloning
from .voice_generator import (
    AIVoiceGenerator,
    create_voice_from_samples,
    generate_voice_for_personality,
)

# AI voice identity and self-discovery
from .voice_identity import (
    AIVoiceIdentity,
    adjust_voice_from_feedback,
    describe_voice,
    discover_voice,
)
from .voice_profile import (
    PRESET_PROFILES,
    VoiceEngine,
    VoiceProfile,
    get_engine,
    list_custom_profiles,
    list_presets,
    list_system_voices,
    set_voice,
    speak,
)

# New voice options (optional dependencies)
from .whisper_stt import WhisperSTT

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
    
    # Noise reduction
    'NoiseReducer',
    'NoiseReductionConfig',
    'reduce_noise',
    
    # Echo cancellation
    'EchoCanceller',
    'EchoCancellationConfig',
    'cancel_echo',
    
    # Audio ducking
    'AudioDucker',
    'AudioDuckingConfig',
    'duck_audio',
    'get_ducker',
    
    # SSML support
    'SSMLParser',
    'SSMLProcessor',
    'SSMLSegment',
    'SSMLDocument',
    'ssml_to_text',
    'strip_ssml',
    
    # Emotional TTS
    'EmotionalTTS',
    'Emotion',
    'EmotionProfile',
    'EmotionDetector',
    'detect_emotion',
    'emotional_ssml',
    
    # Multilingual TTS
    'MultilingualTTS',
    'Language',
    'LanguageDetector',
    'VoiceInfo',
    'detect_language',
    'get_language_name',
    
    # Speed control
    'SpeedController',
    'SpeedConfig',
    'SpeedPreset',
    'set_speed',
    'get_speed',
    
    # Interruption handling
    'InterruptionHandler',
    'InterruptionConfig',
    'InterruptionMode',
    'start_barge_in_detection',
    'stop_barge_in_detection',
    'was_interrupted',
    
    # Streaming TTS
    'StreamingTTS',
    'StreamingConfig',
    'AudioChunk',
    'stream_speak',
    'stream_chunks',
    
    # Audio file input
    'AudioFileTranscriber',
    'TranscriptionConfig',
    'TranscriptionResult',
    'transcribe_file',
    'transcribe_with_timestamps',
    
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
        SpeechSegment,
        VoiceConfig,
        VoiceDevice,
        VoiceMode,
        VoicePipeline,
        get_voice_pipeline,
    )
except ImportError:
    VoicePipeline = None
    VoiceConfig = None
    VoiceMode = None
    VoiceDevice = None
    SpeechSegment = None
    get_voice_pipeline = None

# Noise reduction
try:
    from .noise_reduction import (
        NoiseReducer,
        NoiseReductionConfig,
        reduce_noise,
    )
except ImportError:
    NoiseReducer = None
    NoiseReductionConfig = None
    reduce_noise = None

# Echo cancellation
try:
    from .echo_cancellation import (
        EchoCancellationConfig,
        EchoCanceller,
        cancel_echo,
    )
except ImportError:
    EchoCanceller = None
    EchoCancellationConfig = None
    cancel_echo = None

# Audio ducking
try:
    from .audio_ducking import (
        AudioDucker,
        AudioDuckingConfig,
        duck_audio,
        get_ducker,
    )
except ImportError:
    AudioDucker = None
    AudioDuckingConfig = None
    duck_audio = None
    get_ducker = None

# SSML support
try:
    from .ssml import (
        SSMLDocument,
        SSMLParser,
        SSMLProcessor,
        SSMLSegment,
        ssml_to_text,
        strip_ssml,
    )
except ImportError:
    SSMLParser = None
    SSMLProcessor = None
    SSMLSegment = None
    SSMLDocument = None
    ssml_to_text = None
    strip_ssml = None

# Emotional TTS
try:
    from .emotional_tts import (
        Emotion,
        EmotionalTTS,
        EmotionDetector,
        EmotionProfile,
        detect_emotion,
        emotional_ssml,
    )
except ImportError:
    EmotionalTTS = None
    Emotion = None
    EmotionProfile = None
    EmotionDetector = None
    detect_emotion = None
    emotional_ssml = None

# Multilingual TTS
try:
    from .multilingual_tts import (
        Language,
        LanguageDetector,
        MultilingualTTS,
        VoiceInfo,
        detect_language,
        get_language_name,
    )
except ImportError:
    MultilingualTTS = None
    Language = None
    LanguageDetector = None
    VoiceInfo = None
    detect_language = None
    get_language_name = None

# Speed control
try:
    from .speed_control import (
        SpeedConfig,
        SpeedController,
        SpeedPreset,
        get_speed,
        set_speed,
    )
except ImportError:
    SpeedController = None
    SpeedConfig = None
    SpeedPreset = None
    set_speed = None
    get_speed = None

# Interruption handling
try:
    from .interruption import (
        InterruptionConfig,
        InterruptionHandler,
        InterruptionMode,
        start_barge_in_detection,
        stop_barge_in_detection,
        was_interrupted,
    )
except ImportError:
    InterruptionHandler = None
    InterruptionConfig = None
    InterruptionMode = None
    start_barge_in_detection = None
    stop_barge_in_detection = None
    was_interrupted = None

# Streaming TTS
try:
    from .streaming_tts import (
        AudioChunk,
        StreamingConfig,
        StreamingTTS,
        stream_chunks,
        stream_speak,
    )
except ImportError:
    StreamingTTS = None
    StreamingConfig = None
    AudioChunk = None
    stream_speak = None
    stream_chunks = None

# Audio file input
try:
    from .audio_file_input import (
        AudioFileTranscriber,
        TranscriptionConfig,
        TranscriptionResult,
        transcribe_file,
        transcribe_with_timestamps,
    )
except ImportError:
    AudioFileTranscriber = None
    TranscriptionConfig = None
    TranscriptionResult = None
    transcribe_file = None
    transcribe_with_timestamps = None

