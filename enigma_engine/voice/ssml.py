"""
================================================================================
SSML Support - Speech Synthesis Markup Language for TTS control.
================================================================================

Parse and apply SSML tags to control text-to-speech output including:
- Prosody (rate, pitch, volume)
- Breaks/pauses
- Emphasis
- Say-as (dates, numbers, spelling)
- Phonemes
- Audio insertion

Backends that support SSML natively:
- Amazon Polly
- Google Cloud TTS
- Microsoft Azure TTS
- ElevenLabs (partial)

For backends without native SSML support (pyttsx3, espeak), this module
converts SSML to backend-specific commands or simulates effects.

USAGE:
    from enigma_engine.voice.ssml import SSMLParser, ssml_to_text
    
    # Parse SSML
    parser = SSMLParser()
    segments = parser.parse('<speak>Hello <break time="500ms"/> world!</speak>')
    
    # Convert to plain text with timing hints
    text, hints = ssml_to_text('<speak>Say it <emphasis>loudly</emphasis>!</speak>')
    
    # Apply SSML to TTS
    from enigma_engine.voice.ssml import SSMLProcessor
    processor = SSMLProcessor(tts_engine)
    processor.speak('<speak>Testing <prosody rate="slow">slower speech</prosody></speak>')
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class SSMLTag(Enum):
    """SSML tag types."""
    SPEAK = auto()
    BREAK = auto()
    PROSODY = auto()
    EMPHASIS = auto()
    SAY_AS = auto()
    PHONEME = auto()
    SUB = auto()
    AUDIO = auto()
    P = auto()  # Paragraph
    S = auto()  # Sentence
    VOICE = auto()
    LANG = auto()
    MARK = auto()


@dataclass
class SSMLSegment:
    """A segment of parsed SSML content."""
    
    text: str = ""
    tag: SSMLTag | None = None
    
    # Prosody modifiers (relative to current)
    rate: float = 1.0        # Speech rate multiplier (0.5 = half speed)
    pitch: float = 1.0       # Pitch multiplier
    volume: float = 1.0      # Volume multiplier
    
    # Timing
    break_ms: int = 0        # Pause duration in milliseconds
    
    # Emphasis
    emphasis_level: str = "none"  # none, reduced, moderate, strong
    
    # Say-as interpretation
    interpret_as: str | None = None  # date, time, telephone, characters, etc.
    format_hint: str | None = None   # Format for interpretation
    
    # Phoneme
    phoneme: str | None = None
    phoneme_alphabet: str = "ipa"  # ipa or x-sampa
    
    # Substitution
    alias: str | None = None
    
    # Audio
    audio_src: str | None = None
    
    # Voice/language
    voice_name: str | None = None
    language: str | None = None
    
    # Mark for synchronization
    mark_name: str | None = None


@dataclass
class SSMLDocument:
    """Parsed SSML document."""
    
    segments: list[SSMLSegment] = field(default_factory=list)
    xml_lang: str = "en-US"
    
    def get_plain_text(self) -> str:
        """Get plain text without SSML formatting."""
        return "".join(seg.text for seg in self.segments if seg.text)
    
    def get_total_break_time_ms(self) -> int:
        """Get total pause time in milliseconds."""
        return sum(seg.break_ms for seg in self.segments)


class SSMLParser:
    """
    Parse SSML markup into structured segments.
    
    Supports W3C SSML 1.1 specification with common extensions.
    """
    
    # Rate keywords to multipliers
    RATE_MAP = {
        "x-slow": 0.5,
        "slow": 0.75,
        "medium": 1.0,
        "fast": 1.25,
        "x-fast": 1.5,
    }
    
    # Pitch keywords to multipliers
    PITCH_MAP = {
        "x-low": 0.7,
        "low": 0.85,
        "medium": 1.0,
        "high": 1.15,
        "x-high": 1.3,
    }
    
    # Volume keywords to multipliers
    VOLUME_MAP = {
        "silent": 0.0,
        "x-soft": 0.25,
        "soft": 0.5,
        "medium": 1.0,
        "loud": 1.5,
        "x-loud": 2.0,
    }
    
    # Break strength to milliseconds
    BREAK_STRENGTH_MAP = {
        "none": 0,
        "x-weak": 100,
        "weak": 200,
        "medium": 400,
        "strong": 600,
        "x-strong": 1000,
    }
    
    def __init__(self):
        self._current_prosody: dict[str, float] = {
            "rate": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
        }
    
    def parse(self, ssml: str) -> SSMLDocument:
        """
        Parse SSML string into document structure.
        
        Args:
            ssml: SSML markup string
        
        Returns:
            Parsed SSMLDocument
        """
        doc = SSMLDocument()
        
        # Ensure proper XML structure
        ssml = ssml.strip()
        if not ssml.startswith("<speak"):
            ssml = f"<speak>{ssml}</speak>"
        
        try:
            # Parse XML
            root = ET.fromstring(ssml)
            
            # Get language from root
            doc.xml_lang = root.get("{http://www.w3.org/XML/1998/namespace}lang", 
                                    root.get("lang", "en-US"))
            
            # Process elements recursively
            self._process_element(root, doc.segments, {})
            
        except ET.ParseError as e:
            logger.warning(f"SSML parse error: {e}, treating as plain text")
            # Fallback: strip tags and use as plain text
            plain = re.sub(r'<[^>]+>', '', ssml)
            doc.segments.append(SSMLSegment(text=plain))
        
        return doc
    
    def _process_element(
        self,
        element: ET.Element,
        segments: list[SSMLSegment],
        inherited: dict[str, Any]
    ):
        """Process an SSML element recursively."""
        tag_name = element.tag.lower()
        if "}" in tag_name:  # Remove namespace
            tag_name = tag_name.split("}")[1]
        
        # Handle different tag types
        if tag_name == "speak":
            # Root element - process children
            if element.text:
                segments.append(SSMLSegment(text=element.text, **inherited))
            for child in element:
                self._process_element(child, segments, inherited)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **inherited))
        
        elif tag_name == "break":
            # Pause
            break_ms = self._parse_break(element)
            segments.append(SSMLSegment(break_ms=break_ms))
        
        elif tag_name == "prosody":
            # Prosody changes
            new_props = inherited.copy()
            
            rate = element.get("rate", "")
            if rate:
                new_props["rate"] = inherited.get("rate", 1.0) * self._parse_prosody_value(rate, self.RATE_MAP)
            
            pitch = element.get("pitch", "")
            if pitch:
                new_props["pitch"] = inherited.get("pitch", 1.0) * self._parse_prosody_value(pitch, self.PITCH_MAP)
            
            volume = element.get("volume", "")
            if volume:
                new_props["volume"] = inherited.get("volume", 1.0) * self._parse_prosody_value(volume, self.VOLUME_MAP)
            
            # Process children with new prosody
            if element.text:
                segments.append(SSMLSegment(text=element.text, **new_props))
            for child in element:
                self._process_element(child, segments, new_props)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **new_props))
        
        elif tag_name == "emphasis":
            # Emphasis
            level = element.get("level", "moderate")
            new_props = inherited.copy()
            new_props["emphasis_level"] = level
            
            # Emphasis affects pitch and rate slightly
            if level == "strong":
                new_props["pitch"] = inherited.get("pitch", 1.0) * 1.1
                new_props["rate"] = inherited.get("rate", 1.0) * 0.9
            elif level == "reduced":
                new_props["volume"] = inherited.get("volume", 1.0) * 0.8
            
            if element.text:
                segments.append(SSMLSegment(text=element.text, **new_props))
            for child in element:
                self._process_element(child, segments, new_props)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **new_props))
        
        elif tag_name == "say-as":
            # Interpretation hints
            interpret_as = element.get("interpret-as", "")
            format_hint = element.get("format", "")
            
            text = self._get_element_text(element)
            text = self._apply_say_as(text, interpret_as, format_hint)
            
            segments.append(SSMLSegment(
                text=text,
                interpret_as=interpret_as,
                format_hint=format_hint,
                **inherited
            ))
        
        elif tag_name == "phoneme":
            # Phonetic pronunciation
            alphabet = element.get("alphabet", "ipa")
            ph = element.get("ph", "")
            
            segments.append(SSMLSegment(
                text=self._get_element_text(element),
                phoneme=ph,
                phoneme_alphabet=alphabet,
                **inherited
            ))
        
        elif tag_name == "sub":
            # Substitution
            alias = element.get("alias", "")
            segments.append(SSMLSegment(
                text=alias if alias else self._get_element_text(element),
                alias=alias,
                **inherited
            ))
        
        elif tag_name == "audio":
            # Audio insertion
            src = element.get("src", "")
            segments.append(SSMLSegment(audio_src=src))
            
            # Fallback content if audio fails
            if element.text:
                segments.append(SSMLSegment(text=element.text, **inherited))
        
        elif tag_name in ("p", "paragraph"):
            # Paragraph - add break before and after
            segments.append(SSMLSegment(break_ms=400))
            if element.text:
                segments.append(SSMLSegment(text=element.text, **inherited))
            for child in element:
                self._process_element(child, segments, inherited)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **inherited))
            segments.append(SSMLSegment(break_ms=400))
        
        elif tag_name in ("s", "sentence"):
            # Sentence - add small break after
            if element.text:
                segments.append(SSMLSegment(text=element.text, **inherited))
            for child in element:
                self._process_element(child, segments, inherited)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **inherited))
            segments.append(SSMLSegment(break_ms=200))
        
        elif tag_name == "voice":
            # Voice change
            voice_name = element.get("name", "")
            new_props = inherited.copy()
            new_props["voice_name"] = voice_name
            
            if element.text:
                segments.append(SSMLSegment(text=element.text, **new_props))
            for child in element:
                self._process_element(child, segments, new_props)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **new_props))
        
        elif tag_name == "lang":
            # Language change
            lang = element.get("{http://www.w3.org/XML/1998/namespace}lang",
                             element.get("lang", ""))
            new_props = inherited.copy()
            new_props["language"] = lang
            
            if element.text:
                segments.append(SSMLSegment(text=element.text, **new_props))
            for child in element:
                self._process_element(child, segments, new_props)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **new_props))
        
        elif tag_name == "mark":
            # Synchronization mark
            name = element.get("name", "")
            segments.append(SSMLSegment(mark_name=name))
        
        else:
            # Unknown tag - treat as container
            if element.text:
                segments.append(SSMLSegment(text=element.text, **inherited))
            for child in element:
                self._process_element(child, segments, inherited)
                if child.tail:
                    segments.append(SSMLSegment(text=child.tail, **inherited))
    
    def _parse_break(self, element: ET.Element) -> int:
        """Parse break element into milliseconds."""
        time_attr = element.get("time", "")
        if time_attr:
            return self._parse_time(time_attr)
        
        strength = element.get("strength", "medium")
        return self.BREAK_STRENGTH_MAP.get(strength, 400)
    
    def _parse_time(self, time_str: str) -> int:
        """Parse time string (e.g., '500ms', '1s') to milliseconds."""
        time_str = time_str.strip().lower()
        
        if time_str.endswith("ms"):
            return int(float(time_str[:-2]))
        elif time_str.endswith("s"):
            return int(float(time_str[:-1]) * 1000)
        else:
            # Assume milliseconds
            try:
                return int(float(time_str))
            except ValueError:
                return 0
    
    def _parse_prosody_value(self, value: str, keyword_map: dict) -> float:
        """Parse prosody value (keyword or percentage/semitones)."""
        value = value.strip().lower()
        
        # Keyword lookup
        if value in keyword_map:
            return keyword_map[value]
        
        # Percentage (e.g., "+20%", "-10%", "150%")
        if value.endswith("%"):
            try:
                pct = float(value[:-1])
                if value.startswith("+") or value.startswith("-"):
                    return 1.0 + (pct / 100)
                else:
                    return pct / 100
            except ValueError:
                pass  # Intentionally silent
        
        # Semitones for pitch (e.g., "+2st", "-3st")
        if value.endswith("st"):
            try:
                st = float(value[:-2])
                return 2 ** (st / 12)  # Semitone to frequency ratio
            except ValueError:
                pass  # Intentionally silent
        
        # Hertz for pitch
        if value.endswith("hz"):
            # Can't convert without knowing base frequency
            return 1.0
        
        return 1.0
    
    def _get_element_text(self, element: ET.Element) -> str:
        """Get all text content from element."""
        return "".join(element.itertext())
    
    def _apply_say_as(self, text: str, interpret_as: str, format_hint: str) -> str:
        """Apply say-as interpretation to text."""
        if interpret_as == "characters" or interpret_as == "spell-out":
            # Spell out characters
            return " ".join(text)
        
        elif interpret_as == "cardinal" or interpret_as == "number":
            # Already a number, keep as is
            return text
        
        elif interpret_as == "ordinal":
            # Convert to ordinal (basic)
            try:
                n = int(text)
                if 10 <= n % 100 <= 20:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"{n}{suffix}"
            except ValueError:
                return text
        
        elif interpret_as == "telephone":
            # Add spaces between digits
            return " ".join(text.replace("-", " ").replace(".", " "))
        
        elif interpret_as == "date":
            # Date formatting would need more context
            return text
        
        elif interpret_as == "time":
            return text
        
        elif interpret_as == "currency":
            return text
        
        elif interpret_as == "unit":
            return text
        
        return text


class SSMLProcessor:
    """
    Process SSML and synthesize speech with appropriate TTS backend.
    
    For TTS backends that support SSML natively, passes through.
    For others, converts to backend-specific commands.
    """
    
    def __init__(self, tts_engine: Any = None):
        """
        Initialize SSML processor.
        
        Args:
            tts_engine: TTS engine instance (pyttsx3, etc.)
        """
        self.tts_engine = tts_engine
        self.parser = SSMLParser()
        
        # Callbacks for marks and audio
        self._mark_callbacks: list[Callable[[str], None]] = []
        self._audio_callback: Callable[[str], None] | None = None
    
    def on_mark(self, callback: Callable[[str], None]):
        """Register callback for SSML mark elements."""
        self._mark_callbacks.append(callback)
    
    def on_audio(self, callback: Callable[[str], None]):
        """Register callback for audio element (to play external audio)."""
        self._audio_callback = callback
    
    def speak(self, ssml: str, blocking: bool = True):
        """
        Speak SSML content.
        
        Args:
            ssml: SSML markup string
            blocking: Wait for speech to complete
        """
        doc = self.parser.parse(ssml)
        
        for segment in doc.segments:
            # Handle marks
            if segment.mark_name:
                for callback in self._mark_callbacks:
                    try:
                        callback(segment.mark_name)
                    except Exception as e:
                        logger.error(f"Mark callback error: {e}")
                continue
            
            # Handle audio
            if segment.audio_src:
                if self._audio_callback:
                    try:
                        self._audio_callback(segment.audio_src)
                    except Exception as e:
                        logger.error(f"Audio callback error: {e}")
                continue
            
            # Handle breaks
            if segment.break_ms > 0:
                time.sleep(segment.break_ms / 1000)
                continue
            
            # Handle text with prosody
            if segment.text:
                self._speak_segment(segment, blocking)
    
    def _speak_segment(self, segment: SSMLSegment, blocking: bool):
        """Speak a single segment with prosody applied."""
        if not self.tts_engine:
            logger.warning("No TTS engine configured")
            return
        
        text = segment.text.strip()
        if not text:
            return
        
        # Try to apply prosody to engine
        engine_type = self._detect_engine_type()
        
        if engine_type == "pyttsx3":
            self._speak_pyttsx3(segment, blocking)
        elif engine_type == "espeak":
            self._speak_espeak(segment, blocking)
        else:
            # Generic fallback
            self._speak_generic(segment, blocking)
    
    def _detect_engine_type(self) -> str:
        """Detect the type of TTS engine."""
        if self.tts_engine is None:
            return "none"
        
        engine_str = str(type(self.tts_engine)).lower()
        
        if "pyttsx3" in engine_str:
            return "pyttsx3"
        elif "espeak" in engine_str:
            return "espeak"
        
        return "generic"
    
    def _speak_pyttsx3(self, segment: SSMLSegment, blocking: bool):
        """Speak using pyttsx3 with prosody."""
        try:
            engine = self.tts_engine
            
            # Store original properties
            original_rate = engine.getProperty("rate")
            original_volume = engine.getProperty("volume")
            
            # Apply prosody
            new_rate = int(original_rate * segment.rate)
            new_volume = min(1.0, original_volume * segment.volume)
            
            engine.setProperty("rate", new_rate)
            engine.setProperty("volume", new_volume)
            
            # Note: pyttsx3 doesn't support pitch modification easily
            
            # Speak
            engine.say(segment.text)
            if blocking:
                engine.runAndWait()
            
            # Restore
            engine.setProperty("rate", original_rate)
            engine.setProperty("volume", original_volume)
            
        except Exception as e:
            logger.error(f"pyttsx3 speak error: {e}")
    
    def _speak_espeak(self, segment: SSMLSegment, blocking: bool):
        """Speak using espeak with prosody."""
        import subprocess
        
        try:
            # Build espeak command with prosody
            cmd = ["espeak"]
            
            # Rate: espeak uses words per minute (default ~175)
            rate = int(175 * segment.rate)
            cmd.extend(["-s", str(rate)])
            
            # Pitch: espeak uses 0-99 (default 50)
            pitch = int(50 * segment.pitch)
            pitch = max(0, min(99, pitch))
            cmd.extend(["-p", str(pitch)])
            
            # Volume: espeak uses 0-200 (default 100)
            volume = int(100 * segment.volume)
            volume = max(0, min(200, volume))
            cmd.extend(["-a", str(volume)])
            
            cmd.append(segment.text)
            
            if blocking:
                subprocess.run(cmd, capture_output=True, timeout=60)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
        except Exception as e:
            logger.error(f"espeak error: {e}")
    
    def _speak_generic(self, segment: SSMLSegment, blocking: bool):
        """Generic speech with limited prosody support."""
        try:
            if hasattr(self.tts_engine, "say"):
                self.tts_engine.say(segment.text)
                if blocking and hasattr(self.tts_engine, "runAndWait"):
                    self.tts_engine.runAndWait()
            elif hasattr(self.tts_engine, "speak"):
                self.tts_engine.speak(segment.text)
        except Exception as e:
            logger.error(f"Generic TTS error: {e}")


def ssml_to_text(ssml: str) -> tuple[str, list[dict]]:
    """
    Convert SSML to plain text with timing hints.
    
    Args:
        ssml: SSML markup string
    
    Returns:
        Tuple of (plain_text, hints) where hints contain timing/prosody info
    """
    parser = SSMLParser()
    doc = parser.parse(ssml)
    
    text_parts = []
    hints = []
    char_offset = 0
    
    for segment in doc.segments:
        if segment.text:
            text_parts.append(segment.text)
            
            hint = {
                "start": char_offset,
                "end": char_offset + len(segment.text),
                "rate": segment.rate,
                "pitch": segment.pitch,
                "volume": segment.volume,
                "emphasis": segment.emphasis_level,
            }
            hints.append(hint)
            
            char_offset += len(segment.text)
        
        elif segment.break_ms > 0:
            hints.append({
                "type": "break",
                "offset": char_offset,
                "duration_ms": segment.break_ms,
            })
    
    return "".join(text_parts), hints


def strip_ssml(ssml: str) -> str:
    """
    Strip all SSML tags, returning plain text.
    
    Args:
        ssml: SSML markup string
    
    Returns:
        Plain text without any markup
    """
    parser = SSMLParser()
    doc = parser.parse(ssml)
    return doc.get_plain_text()
