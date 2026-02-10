"""
Lip Sync System

Basic lip sync for speaking animations.
Maps phonemes to mouth shapes (visemes).
"""

import re


class LipSync:
    """
    Basic lip sync for speaking animations.
    
    Maps phonemes/sounds to mouth shapes (visemes) for
    more realistic speaking animation.
    """
    
    # Viseme mapping (simplified)
    VISEMES = {
        "silence": "mouth_closed",
        "aa": "mouth_open_wide",    # "father", "ah"
        "ee": "mouth_smile",         # "see", "me"
        "oo": "mouth_round",         # "food", "you"
        "oh": "mouth_open",          # "go", "no"
        "consonant": "mouth_closed", # Most consonants
        "mm": "mouth_closed",        # "mom", "um"
        "ff": "mouth_smile",         # "from", "if"
        "th": "mouth_open",          # "the", "think"
    }
    
    # Vowel patterns for detection
    VOWEL_PATTERNS = {
        r'[aA][hH]': 'aa',        # ah sound
        r'[aA][rR]': 'aa',        # car, far
        r'[eE][eE]': 'ee',        # see, me
        r'[iI]': 'ee',            # bit, sit
        r'[oO][oO]': 'oo',        # food, mood
        r'[uU]': 'oo',            # put, you
        r'[oO]': 'oh',            # go, no
        r'[aA]': 'aa',            # cat, hat
        r'[eE]': 'ee',            # bet, set
    }
    
    def __init__(self):
        """Initialize lip sync system."""
        self.current_viseme = "silence"
        self._viseme_duration = 0.1  # Default 100ms per viseme
    
    def text_to_visemes(self, text: str) -> list[tuple[str, float]]:
        """
        Convert text to timed viseme sequence.
        
        This is a simplified implementation. For better results,
        use a proper phoneme library like phonemizer or g2p.
        
        Args:
            text: Text to convert
            
        Returns:
            List of (viseme_name, duration) tuples
        """
        visemes = []
        words = text.split()
        
        for word in words:
            # Estimate word duration based on length
            word_duration = len(word) * 0.08  # ~80ms per character
            
            # Simple vowel detection
            vowel_found = False
            for pattern, viseme in self.VOWEL_PATTERNS.items():
                if re.search(pattern, word):
                    visemes.append((viseme, word_duration * 0.6))
                    vowel_found = True
                    break
            
            if not vowel_found:
                # Default to neutral if no vowel detected
                visemes.append(("consonant", word_duration * 0.3))
            
            # Add brief silence between words
            visemes.append(("silence", 0.05))
        
        return visemes
    
    def get_viseme_for_phoneme(self, phoneme: str) -> str:
        """
        Map phoneme to viseme.
        
        Args:
            phoneme: Phoneme symbol (e.g., 'AA', 'IY', 'UW')
            
        Returns:
            Viseme name
        """
        # ARPAbet to viseme mapping (simplified)
        phoneme_map = {
            'AA': 'aa', 'AE': 'aa', 'AH': 'aa', 'AO': 'oh',
            'AW': 'oh', 'AY': 'aa', 'EH': 'ee', 'ER': 'ee',
            'EY': 'ee', 'IH': 'ee', 'IY': 'ee', 'OW': 'oh',
            'OY': 'oh', 'UH': 'oo', 'UW': 'oo',
            # Consonants generally close the mouth
            'B': 'mm', 'P': 'mm', 'M': 'mm',
            'F': 'ff', 'V': 'ff',
            'TH': 'th', 'DH': 'th',
        }
        
        return phoneme_map.get(phoneme.upper(), 'consonant')
    
    def sync_with_audio(self, audio_data, text: str) -> list[tuple[str, float, float]]:
        """
        Sync visemes with audio data.
        
        This would analyze actual audio waveform for better timing.
        For now, returns estimated timing based on text.
        
        Args:
            audio_data: Audio waveform data (not used in simple version)
            text: Text being spoken
            
        Returns:
            List of (viseme_name, start_time, duration) tuples
        """
        visemes_with_timing = []
        current_time = 0.0
        
        visemes = self.text_to_visemes(text)
        
        for viseme_name, duration in visemes:
            visemes_with_timing.append((viseme_name, current_time, duration))
            current_time += duration
        
        return visemes_with_timing
    
    def get_mouth_shape_for_text(self, text: str, position: float = 0.0) -> str:
        """
        Get appropriate mouth shape for text at a specific time position.
        
        Args:
            text: Text being spoken
            position: Time position in seconds
            
        Returns:
            Mouth shape/viseme name
        """
        visemes = self.sync_with_audio(None, text)
        
        # Find viseme at current position
        for viseme_name, start_time, duration in visemes:
            if start_time <= position < start_time + duration:
                return self.VISEMES.get(viseme_name, "mouth_closed")
        
        return "mouth_closed"
    
    def animate_speaking(self, text: str) -> list[str]:
        """
        Generate frame-by-frame mouth shapes for speaking animation.
        
        Args:
            text: Text being spoken
            
        Returns:
            List of sprite names for animation frames
        """
        # Simple alternating pattern for speaking
        # In a real implementation, this would use the viseme sequence
        
        # Estimate number of frames based on text length
        word_count = len(text.split())
        frame_count = max(2, word_count * 2)
        
        # Alternate between speaking frames
        frames = []
        for i in range(frame_count):
            if i % 2 == 0:
                frames.append("speaking_1")
            else:
                frames.append("speaking_2")
        
        # End with neutral
        frames.append("idle")
        
        return frames
    
    def set_viseme_duration(self, duration: float):
        """
        Set default viseme duration.
        
        Args:
            duration: Duration in seconds
        """
        self._viseme_duration = max(0.05, min(0.5, duration))
