"""Audio analyzer stub - feature disabled."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioAnalysis:
    """Audio analysis results."""
    pitch: float = 0.0
    speed: float = 1.0
    energy: float = 0.0
    duration: float = 0.0


class AudioAnalyzer:
    """Stub - audio analysis disabled."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def analyze(self, audio_path: str) -> AudioAnalysis:
        return AudioAnalysis()
    
    def analyze_samples(self, audio_files: list) -> AudioAnalysis:
        return AudioAnalysis()
    
    def get_pitch(self, audio_path: str) -> float:
        return 0.0
    
    def get_speed(self, audio_path: str) -> float:
        return 1.0
