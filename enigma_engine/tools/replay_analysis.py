"""
Replay Analysis for Enigma AI Engine

Analyze game replays for learning and improvement.

Features:
- Frame extraction
- Event detection
- Performance metrics
- Strategy analysis
- Highlight generation

Usage:
    from enigma_engine.tools.replay_analysis import ReplayAnalyzer, get_analyzer
    
    analyzer = get_analyzer()
    
    # Load replay
    analyzer.load_video("gameplay.mp4")
    
    # Analyze
    results = analyzer.analyze()
    
    # Get highlights
    highlights = analyzer.get_highlights()
"""

import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of game events."""
    KILL = "kill"
    DEATH = "death"
    ASSIST = "assist"
    OBJECTIVE = "objective"
    ITEM_PICKUP = "item_pickup"
    ABILITY_USE = "ability_use"
    MOVEMENT = "movement"
    DAMAGE = "damage"
    HEAL = "heal"
    CUSTOM = "custom"


class HighlightType(Enum):
    """Types of highlights."""
    MULTI_KILL = "multi_kill"
    CLUTCH = "clutch"
    BIG_PLAY = "big_play"
    FAIL = "fail"
    SKILL_SHOT = "skill_shot"
    TEAM_FIGHT = "team_fight"


@dataclass
class GameEvent:
    """A detected game event."""
    event_type: EventType
    timestamp: float  # In seconds
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Detection
    confidence: float = 1.0
    frame_index: int = 0
    
    # Context
    player: Optional[str] = None
    target: Optional[str] = None


@dataclass
class Highlight:
    """A highlight moment."""
    highlight_type: HighlightType
    start_time: float
    end_time: float
    
    # Events in highlight
    events: List[GameEvent] = field(default_factory=list)
    
    # Score
    importance: float = 0.0
    description: str = ""


@dataclass
class PerformanceMetrics:
    """Performance statistics."""
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    
    # Derived
    kda: float = 0.0
    accuracy: float = 0.0
    
    # Time-based
    avg_reaction_time: float = 0.0
    apm: float = 0.0  # Actions per minute
    
    # Custom
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of replay analysis."""
    duration: float
    fps: float
    frame_count: int
    
    # Events
    events: List[GameEvent] = field(default_factory=list)
    
    # Metrics
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Highlights
    highlights: List[Highlight] = field(default_factory=list)
    
    # Strategy
    movement_patterns: Dict[str, Any] = field(default_factory=dict)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)


class FrameExtractor:
    """Extract frames from video."""
    
    def __init__(self):
        self._cv2 = None
        self._load_opencv()
    
    def _load_opencv(self):
        """Load OpenCV."""
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.warning("OpenCV not available")
    
    def extract_frames(
        self,
        video_path: str,
        interval: float = 1.0,
        max_frames: Optional[int] = None
    ) -> List[Tuple[float, Any]]:
        """
        Extract frames from video at interval.
        
        Returns:
            List of (timestamp, frame) tuples
        """
        if not self._cv2:
            return []
        
        frames = []
        cap = self._cv2.VideoCapture(video_path)
        
        fps = cap.get(self._cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * interval)
        
        frame_idx = 0
        extracted = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frames.append((timestamp, frame))
                extracted += 1
                
                if max_frames and extracted >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def get_video_info(
        self,
        video_path: str
    ) -> Dict[str, Any]:
        """Get video information."""
        if not self._cv2:
            return {}
        
        cap = self._cv2.VideoCapture(video_path)
        
        info = {
            "fps": cap.get(self._cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(self._cv2.CAP_PROP_FRAME_COUNT) / cap.get(self._cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info


class EventDetector:
    """Detect game events from frames."""
    
    def __init__(
        self,
        model: Any = None
    ):
        self._model = model
        self._detectors: Dict[EventType, Callable] = {}
        
        # Register default detectors
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default event detectors."""
        self._detectors[EventType.KILL] = self._detect_kill
        self._detectors[EventType.DEATH] = self._detect_death
        self._detectors[EventType.MOVEMENT] = self._detect_movement
    
    def register_detector(
        self,
        event_type: EventType,
        detector: Callable[[Any, float], Optional[GameEvent]]
    ):
        """Register a custom event detector."""
        self._detectors[event_type] = detector
    
    def detect_events(
        self,
        frames: List[Tuple[float, Any]]
    ) -> List[GameEvent]:
        """
        Detect events in frames.
        
        Returns:
            List of detected events
        """
        events = []
        
        for i, (timestamp, frame) in enumerate(frames):
            for event_type, detector in self._detectors.items():
                try:
                    event = detector(frame, timestamp)
                    if event:
                        event.frame_index = i
                        events.append(event)
                except Exception as e:
                    logger.error(f"Detection error: {e}")
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return events
    
    def _detect_kill(
        self,
        frame: Any,
        timestamp: float
    ) -> Optional[GameEvent]:
        """Detect kill events."""
        # Would use OCR or model to detect kill feed
        # Placeholder implementation
        return None
    
    def _detect_death(
        self,
        frame: Any,
        timestamp: float
    ) -> Optional[GameEvent]:
        """Detect death events."""
        # Would use screen analysis for death indicators
        return None
    
    def _detect_movement(
        self,
        frame: Any,
        timestamp: float
    ) -> Optional[GameEvent]:
        """Detect movement patterns."""
        # Would use optical flow or position tracking
        return None


class HighlightGenerator:
    """Generate highlights from events."""
    
    def __init__(
        self,
        min_highlight_duration: float = 5.0,
        max_highlight_duration: float = 30.0
    ):
        self._min_duration = min_highlight_duration
        self._max_duration = max_highlight_duration
    
    def generate_highlights(
        self,
        events: List[GameEvent],
        total_duration: float
    ) -> List[Highlight]:
        """
        Generate highlights from events.
        
        Returns:
            List of highlights
        """
        highlights = []
        
        # Find multi-kill sequences
        highlights.extend(self._find_multi_kills(events))
        
        # Find team fights
        highlights.extend(self._find_team_fights(events))
        
        # Find big plays
        highlights.extend(self._find_big_plays(events))
        
        # Sort by importance
        highlights.sort(key=lambda h: -h.importance)
        
        return highlights
    
    def _find_multi_kills(
        self,
        events: List[GameEvent],
        time_window: float = 10.0
    ) -> List[Highlight]:
        """Find multi-kill sequences."""
        highlights = []
        
        kill_events = [e for e in events if e.event_type == EventType.KILL]
        
        i = 0
        while i < len(kill_events):
            sequence = [kill_events[i]]
            
            # Find consecutive kills
            j = i + 1
            while j < len(kill_events):
                if kill_events[j].timestamp - sequence[-1].timestamp <= time_window:
                    sequence.append(kill_events[j])
                    j += 1
                else:
                    break
            
            if len(sequence) >= 2:
                highlight = Highlight(
                    highlight_type=HighlightType.MULTI_KILL,
                    start_time=max(0, sequence[0].timestamp - 2),
                    end_time=sequence[-1].timestamp + 2,
                    events=sequence,
                    importance=len(sequence) * 10,
                    description=f"{len(sequence)}-kill streak"
                )
                highlights.append(highlight)
            
            i = j
        
        return highlights
    
    def _find_team_fights(
        self,
        events: List[GameEvent],
        time_window: float = 15.0,
        min_events: int = 4
    ) -> List[Highlight]:
        """Find team fight moments."""
        highlights = []
        
        # Group events by time windows
        windows: List[List[GameEvent]] = []
        current_window: List[GameEvent] = []
        
        for event in events:
            if not current_window:
                current_window.append(event)
            elif event.timestamp - current_window[0].timestamp <= time_window:
                current_window.append(event)
            else:
                if len(current_window) >= min_events:
                    windows.append(current_window)
                current_window = [event]
        
        if len(current_window) >= min_events:
            windows.append(current_window)
        
        for window in windows:
            highlight = Highlight(
                highlight_type=HighlightType.TEAM_FIGHT,
                start_time=max(0, window[0].timestamp - 3),
                end_time=window[-1].timestamp + 3,
                events=window,
                importance=len(window) * 5,
                description=f"Team fight with {len(window)} events"
            )
            highlights.append(highlight)
        
        return highlights
    
    def _find_big_plays(
        self,
        events: List[GameEvent]
    ) -> List[Highlight]:
        """Find big play moments."""
        highlights = []
        
        # Look for high-confidence interesting events
        for event in events:
            if event.confidence >= 0.9:
                if event.event_type in [EventType.KILL, EventType.OBJECTIVE]:
                    highlight = Highlight(
                        highlight_type=HighlightType.BIG_PLAY,
                        start_time=max(0, event.timestamp - 3),
                        end_time=event.timestamp + 3,
                        events=[event],
                        importance=event.confidence * 15,
                        description=f"Big {event.event_type.value}"
                    )
                    highlights.append(highlight)
        
        return highlights


class StrategyAnalyzer:
    """Analyze strategic patterns."""
    
    def __init__(self, model: Any = None):
        self._model = model
    
    def analyze_movement(
        self,
        events: List[GameEvent]
    ) -> Dict[str, Any]:
        """Analyze movement patterns."""
        movement_events = [e for e in events if e.event_type == EventType.MOVEMENT]
        
        # Calculate statistics
        total_distance = 0
        positions = []
        
        for event in movement_events:
            if 'position' in event.data:
                positions.append(event.data['position'])
        
        if len(positions) >= 2:
            for i in range(1, len(positions)):
                p1, p2 = positions[i-1], positions[i]
                dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                total_distance += dist
        
        return {
            "total_distance": total_distance,
            "position_count": len(positions),
            "heatmap_data": positions
        }
    
    def analyze_decisions(
        self,
        events: List[GameEvent]
    ) -> List[Dict[str, Any]]:
        """Find decision points."""
        decision_points = []
        
        # Decision events
        decision_types = [EventType.ABILITY_USE, EventType.ITEM_PICKUP, EventType.OBJECTIVE]
        
        for event in events:
            if event.event_type in decision_types:
                decision_points.append({
                    "timestamp": event.timestamp,
                    "type": event.event_type.value,
                    "data": event.data,
                    "confidence": event.confidence
                })
        
        return decision_points
    
    def generate_suggestions(
        self,
        metrics: PerformanceMetrics,
        events: List[GameEvent]
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Based on metrics
        if metrics.deaths > metrics.kills * 2:
            suggestions.append("Focus on positioning to reduce deaths")
        
        if metrics.accuracy < 0.3:
            suggestions.append("Practice aim to improve accuracy")
        
        if metrics.apm < 30:
            suggestions.append("Increase actions per minute for better efficiency")
        
        # Based on events
        death_events = [e for e in events if e.event_type == EventType.DEATH]
        if death_events:
            # Could analyze death locations
            suggestions.append("Review death locations for common patterns")
        
        return suggestions


class ReplayAnalyzer:
    """Main replay analysis interface."""
    
    def __init__(
        self,
        model: Any = None
    ):
        """
        Initialize analyzer.
        
        Args:
            model: Optional AI model for detection
        """
        self._extractor = FrameExtractor()
        self._detector = EventDetector(model)
        self._highlight_gen = HighlightGenerator()
        self._strategy = StrategyAnalyzer(model)
        
        # Current video
        self._video_path: Optional[str] = None
        self._video_info: Dict[str, Any] = {}
        self._frames: List[Tuple[float, Any]] = []
        
        # Results
        self._result: Optional[AnalysisResult] = None
    
    def load_video(
        self,
        video_path: str,
        extract_interval: float = 1.0
    ):
        """
        Load a video for analysis.
        
        Args:
            video_path: Path to video file
            extract_interval: Seconds between frame extraction
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self._video_path = video_path
        self._video_info = self._extractor.get_video_info(video_path)
        self._frames = self._extractor.extract_frames(video_path, extract_interval)
        
        logger.info(f"Loaded video: {video_path}")
        logger.info(f"Duration: {self._video_info.get('duration', 0):.1f}s, "
                   f"FPS: {self._video_info.get('fps', 0):.1f}")
    
    def analyze(
        self,
        detect_events: bool = True,
        generate_highlights: bool = True,
        analyze_strategy: bool = True
    ) -> AnalysisResult:
        """
        Analyze the loaded video.
        
        Returns:
            AnalysisResult with all analysis data
        """
        if not self._frames:
            raise ValueError("No video loaded")
        
        result = AnalysisResult(
            duration=self._video_info.get('duration', 0),
            fps=self._video_info.get('fps', 0),
            frame_count=len(self._frames)
        )
        
        # Detect events
        if detect_events:
            result.events = self._detector.detect_events(self._frames)
            result.metrics = self._calculate_metrics(result.events)
        
        # Generate highlights
        if generate_highlights and result.events:
            result.highlights = self._highlight_gen.generate_highlights(
                result.events, result.duration
            )
        
        # Analyze strategy
        if analyze_strategy and result.events:
            result.movement_patterns = self._strategy.analyze_movement(result.events)
            result.decision_points = self._strategy.analyze_decisions(result.events)
        
        self._result = result
        return result
    
    def _calculate_metrics(
        self,
        events: List[GameEvent]
    ) -> PerformanceMetrics:
        """Calculate performance metrics from events."""
        metrics = PerformanceMetrics()
        
        for event in events:
            if event.event_type == EventType.KILL:
                metrics.kills += 1
            elif event.event_type == EventType.DEATH:
                metrics.deaths += 1
            elif event.event_type == EventType.ASSIST:
                metrics.assists += 1
            elif event.event_type == EventType.DAMAGE:
                metrics.damage_dealt += event.data.get('amount', 0)
        
        # Calculate derived metrics
        if metrics.deaths > 0:
            metrics.kda = (metrics.kills + metrics.assists) / metrics.deaths
        else:
            metrics.kda = metrics.kills + metrics.assists
        
        return metrics
    
    def get_highlights(
        self,
        min_importance: float = 0.0,
        max_count: Optional[int] = None
    ) -> List[Highlight]:
        """Get highlights sorted by importance."""
        if not self._result:
            return []
        
        highlights = [
            h for h in self._result.highlights
            if h.importance >= min_importance
        ]
        
        highlights.sort(key=lambda h: -h.importance)
        
        if max_count:
            highlights = highlights[:max_count]
        
        return highlights
    
    def get_suggestions(self) -> List[str]:
        """Get improvement suggestions."""
        if not self._result:
            return []
        
        return self._strategy.generate_suggestions(
            self._result.metrics,
            self._result.events
        )
    
    def export_report(
        self,
        output_path: str
    ):
        """Export analysis report."""
        if not self._result:
            return
        
        report = {
            "video": self._video_path,
            "duration": self._result.duration,
            "fps": self._result.fps,
            "metrics": {
                "kills": self._result.metrics.kills,
                "deaths": self._result.metrics.deaths,
                "assists": self._result.metrics.assists,
                "kda": self._result.metrics.kda
            },
            "highlights": [
                {
                    "type": h.highlight_type.value,
                    "start": h.start_time,
                    "end": h.end_time,
                    "importance": h.importance,
                    "description": h.description
                }
                for h in self._result.highlights
            ],
            "suggestions": self.get_suggestions()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to: {output_path}")


# Global instance
_analyzer: Optional[ReplayAnalyzer] = None


def get_analyzer(model: Any = None) -> ReplayAnalyzer:
    """Get or create global analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ReplayAnalyzer(model)
    return _analyzer
