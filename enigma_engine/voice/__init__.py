"""Voice Package - Unified TTS and STT interface (trimmed version)."""

# Core voice from kept files
from .natural_tts import NaturalTTS
from .stt_simple import transcribe_from_mic as listen
from .voice_generator import AIVoiceGenerator, generate_voice_for_personality, create_voice_from_samples
from .whisper_stt import WhisperSTT
from .vad import VAD as VADProcessor, VADConfig, get_vad

# Voice pipeline
try:
    from .voice_pipeline import (
        SpeechSegment, VoiceConfig, VoiceDevice, VoiceMode, VoicePipeline, get_voice_pipeline,
    )
except ImportError:
    VoicePipeline = VoiceConfig = VoiceMode = VoiceDevice = SpeechSegment = get_voice_pipeline = None

# Stub classes for deleted features
class VoiceProfile: pass
class VoiceEngine: pass
class VoiceListener: pass
def get_engine(): return None
def list_presets(): return []
def list_custom_profiles(): return []
def list_system_voices(): return []
def set_voice(*a, **k): pass
def speak(*a, **k): pass
PRESET_PROFILES = {}

class AIVoiceIdentity: pass
def discover_voice(*a, **k): return None
def describe_voice(*a, **k): return ""
def adjust_voice_from_feedback(*a, **k): return None

class VoiceEffects: pass
def apply_effect(*a, **k): return None
def apply_effects(*a, **k): return None
def effect_for_emotion(*a, **k): return None
def effect_for_context(*a, **k): return None

class DynamicVoiceAdapter: pass
def adapt_voice_for_emotion(*a, **k): return None
def adapt_voice_for_context(*a, **k): return None
def adapt_voice_for_personality(*a, **k): return None

class VoiceCustomizer: pass
def interactive_tuning(*a, **k): pass
def import_voice_profile(*a, **k): return None
def export_voice_profile(*a, **k): pass
def compare_voices(*a, **k): return {}

class AudioAnalyzer: pass
def analyze_audio(*a, **k): return {}
def estimate_voice_profile(*a, **k): return None
def compare_voice_audio(*a, **k): return {}

class TriggerPhraseDetector: pass
class SmartWakeWords: pass
def suggest_wake_phrases(*a, **k): return []
def train_custom_wake_phrase(*a, **k): pass
def start_wake_word_detection(*a, **k): pass
def stop_wake_word_detection(*a, **k): pass
def is_wake_word_active(): return False

class WakeWordDetector: pass
class WakeWordConfig: pass
class WakeWordBackend: pass
def create_wake_word_detector(*a, **k): return None

class NoiseReducer: pass
class NoiseReductionConfig: pass
def reduce_noise(*a, **k): return None

class EchoCanceller: pass
class EchoCancellationConfig: pass
def cancel_echo(*a, **k): return None

class AudioDucker: pass
class AudioDuckingConfig: pass
def duck_audio(*a, **k): return None
def get_ducker(): return None

class SSMLParser: pass
class SSMLProcessor: pass
class SSMLSegment: pass
class SSMLDocument: pass
def ssml_to_text(*a, **k): return ""
def strip_ssml(*a, **k): return ""

class EmotionalTTS: pass
class Emotion: pass
class EmotionProfile: pass
class EmotionDetector: pass
def detect_emotion(*a, **k): return None
def emotional_ssml(*a, **k): return ""

class MultilingualTTS: pass
class Language: pass
class LanguageDetector: pass
class VoiceInfo: pass
def detect_language(*a, **k): return None
def get_language_name(*a, **k): return ""

class SpeedController: pass
class SpeedConfig: pass  
class SpeedPreset: pass
def set_speed(*a, **k): pass
def get_speed(): return 1.0

class InterruptionHandler: pass
class InterruptionConfig: pass
class InterruptionMode: pass
def start_barge_in_detection(*a, **k): pass
def stop_barge_in_detection(*a, **k): pass
def was_interrupted(): return False

class StreamingTTS: pass
class StreamingConfig: pass
class AudioChunk: pass
def stream_speak(*a, **k): return
def stream_chunks(*a, **k): return

class AudioFileTranscriber: pass
class TranscriptionConfig: pass
class TranscriptionResult: pass
def transcribe_file(*a, **k): return None
def transcribe_with_timestamps(*a, **k): return None

__all__ = [
    # Core (from kept files)
    'speak', 'listen', 'set_voice', 'VoiceProfile', 'VoiceEngine', 'get_engine',
    'list_presets', 'list_custom_profiles', 'list_system_voices', 'PRESET_PROFILES',
    'AIVoiceGenerator', 'generate_voice_for_personality', 'create_voice_from_samples',
    'AIVoiceIdentity', 'discover_voice', 'describe_voice', 'adjust_voice_from_feedback',
    'VoiceEffects', 'apply_effect', 'apply_effects', 'effect_for_emotion', 'effect_for_context',
    'DynamicVoiceAdapter', 'adapt_voice_for_emotion', 'adapt_voice_for_context', 'adapt_voice_for_personality',
    'VoiceCustomizer', 'interactive_tuning', 'import_voice_profile', 'export_voice_profile', 'compare_voices',
    'AudioAnalyzer', 'analyze_audio', 'estimate_voice_profile', 'compare_voice_audio',
    'TriggerPhraseDetector', 'SmartWakeWords', 'suggest_wake_phrases', 'train_custom_wake_phrase',
    'start_wake_word_detection', 'stop_wake_word_detection', 'is_wake_word_active',
    'WakeWordDetector', 'WakeWordConfig', 'WakeWordBackend', 'create_wake_word_detector',
    'NoiseReducer', 'NoiseReductionConfig', 'reduce_noise',
    'EchoCanceller', 'EchoCancellationConfig', 'cancel_echo',
    'AudioDucker', 'AudioDuckingConfig', 'duck_audio', 'get_ducker',
    'SSMLParser', 'SSMLProcessor', 'SSMLSegment', 'SSMLDocument', 'ssml_to_text', 'strip_ssml',
    'EmotionalTTS', 'Emotion', 'EmotionProfile', 'EmotionDetector', 'detect_emotion', 'emotional_ssml',
    'MultilingualTTS', 'Language', 'LanguageDetector', 'VoiceInfo', 'detect_language', 'get_language_name',
    'SpeedController', 'SpeedConfig', 'SpeedPreset', 'set_speed', 'get_speed',
    'InterruptionHandler', 'InterruptionConfig', 'InterruptionMode',
    'start_barge_in_detection', 'stop_barge_in_detection', 'was_interrupted',
    'StreamingTTS', 'StreamingConfig', 'AudioChunk', 'stream_speak', 'stream_chunks',
    'AudioFileTranscriber', 'TranscriptionConfig', 'TranscriptionResult', 'transcribe_file', 'transcribe_with_timestamps',
    'WhisperSTT', 'NaturalTTS', 'VADProcessor',
    'VoicePipeline', 'VoiceConfig', 'VoiceMode', 'VoiceDevice', 'SpeechSegment', 'get_voice_pipeline',
]

