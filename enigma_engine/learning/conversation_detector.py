"""
================================================================================
CONVERSATION DETECTOR - Real-time Learning Opportunity Detection
================================================================================

Detects learning opportunities from user messages in real-time:
- Corrections ("No, I meant..." "That's wrong...")
- Teaching ("Remember that..." "My name is...")
- Positive feedback ("Good!" "That's helpful!")
- Negative feedback ("That's not right" "You don't understand")

📍 FILE: enigma_engine/learning/conversation_detector.py
🏷️ TYPE: Learning Detection System
🎯 MAIN CLASS: ConversationDetector, DetectedLearning

INTEGRATES WITH:
    → enigma_engine/core/self_improvement.py (LearningEngine, LearningExample)
    → enigma_engine/core/autonomous.py (AutonomousLearner background processing)

USAGE:
    detector = ConversationDetector()
    
    # Process user message before generating response
    detected = detector.on_user_message(user_msg)
    if detected:
        print(f"Learning: {detected.type} (confidence: {detected.confidence})")
    
    # Track AI response for future correction matching
    detector.on_ai_response(ai_response)
================================================================================
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Import from existing self_improvement module
from ..core.self_improvement import (
    LearningExample,
    LearningSource,
    Priority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DetectedLearning:
    """
    A detected learning opportunity from conversation.
    
    This captures what the AI should learn from user feedback,
    corrections, or explicit teaching.
    """
    type: str  # "correction", "teaching", "positive_feedback", "negative_feedback"
    input_text: str  # What should trigger this learning
    target_output: str  # What the AI should learn to produce
    confidence: float  # 0.0-1.0 how confident we are in the detection
    context: Optional[str] = None  # Additional context
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Extracted information (if applicable)
    corrected_from: Optional[str] = None  # What was wrong
    corrected_to: Optional[str] = None  # What is right
    user_preference: Optional[str] = None  # User preference/fact
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type,
            'input_text': self.input_text,
            'target_output': self.target_output,
            'confidence': self.confidence,
            'context': self.context,
            'timestamp': self.timestamp,
            'corrected_from': self.corrected_from,
            'corrected_to': self.corrected_to,
            'user_preference': self.user_preference,
        }


# =============================================================================
# CONVERSATION DETECTOR
# =============================================================================

class ConversationDetector:
    """
    Detects learning opportunities in real-time conversation.
    
    This class monitors the conversation flow and identifies when
    the user is:
    - Correcting the AI
    - Teaching the AI something new
    - Giving positive feedback (to reinforce behavior)
    - Giving negative feedback (to adjust behavior)
    
    Integrates with existing LearningEngine from self_improvement.py
    by converting detections to LearningExample objects.
    
    Example:
        detector = ConversationDetector()
        
        # In chat loop:
        user_msg = "No, my name is Alex, not Alice"
        detected = detector.on_user_message(user_msg)
        
        if detected:
            # detected.type == "correction"
            # detected.target_output == "Your name is Alex"
            learning_example = detector.to_learning_example(detected)
            learning_engine.add_learning_example(learning_example)
    """
    
    # =========================================================================
    # DETECTION PATTERNS
    # =========================================================================
    # Each pattern is (regex_pattern, confidence_score)
    # Higher confidence = more certain this is the pattern type
    
    CORRECTION_PATTERNS: list[tuple[str, float]] = [
        # Direct corrections
        (r"no,?\s*(i meant|it's|it is|actually)", 0.9),
        (r"that'?s (wrong|incorrect|not right|not correct)", 0.95),
        (r"not quite", 0.7),
        (r"let me (correct|clarify)", 0.85),
        (r"what i (meant|mean) (was|is)", 0.8),
        (r"i said (.+),? not (.+)", 0.95),
        (r"i didn'?t (say|mean|ask)", 0.85),
        (r"that'?s not what i (said|meant|asked)", 0.9),
        (r"you (misunderstood|got it wrong|misheard)", 0.9),
        (r"no,? (it'?s|i'?m|my|the)", 0.75),
        (r"correction:", 0.95),
        (r"to correct you", 0.95),
        (r"actually,?\s*it'?s", 0.85),
        (r"wrong[.!]", 0.8),
        (r"incorrect[.!]", 0.85),
        (r"^no[,.]?\s+", 0.6),  # Simple "no" at start (lower confidence)
    ]
    
    TEACHING_PATTERNS: list[tuple[str, float]] = [
        # Explicit teaching
        (r"(remember|note|learn) that (.+)", 0.9),
        (r"(fyi|for your information|just so you know),?\s*(.+)", 0.85),
        (r"you should know (that )?(.+)", 0.85),
        (r"i'?ll teach you", 0.95),
        (r"let me (teach|tell|show) you", 0.9),
        
        # Personal information sharing
        (r"my (name|favorite|favourite) is (.+)", 0.9),
        (r"i (like|love|hate|prefer|enjoy|dislike) (.+)", 0.8),
        (r"i'?m (called|named) (.+)", 0.9),
        (r"call me (.+)", 0.85),
        (r"i (am|'m) (?:a |an )?(.+)", 0.6),  # "I am a teacher" etc.
        (r"i (live|work|study) (in|at) (.+)", 0.8),
        (r"my (.+) is (.+)", 0.7),  # "my dog is Max"
        (r"i have (a |an )?(.+)", 0.6),  # "I have a cat"
        
        # Preferences
        (r"i (always|never|usually|often) (.+)", 0.7),
        (r"i (don'?t|do) (like|want|need) (.+)", 0.75),
        (r"i (prefer|would rather) (.+)", 0.8),
        
        # Facts to remember
        (r"(here'?s|here is) (a fact|something|what you need to know)", 0.85),
        (r"this is important:", 0.9),
        (r"make (a )?note", 0.85),
        (r"keep in mind", 0.8),
    ]
    
    POSITIVE_PATTERNS: list[tuple[str, float]] = [
        # Affirmations (strict - require end of string or punctuation)
        (r"^(good|great|perfect|excellent|nice|awesome|correct|right|yes|exactly|precisely)[\s!.]*$", 0.9),
        (r"^(yep|yup|yeah|uh-?huh)[\s!.]*$", 0.8),
        (r"^(ok|okay|sure|alright|fine|cool)[\s!.]*$", 0.6),  # Lower confidence - can be neutral
        
        # Gratitude
        (r"(thanks|thank you|thx|ty)(!|\.|\s|$)", 0.75),
        (r"(much |very )?appreciate[ds]?", 0.75),
        
        # Explicit positive feedback
        (r"that('?s| is| was) (helpful|useful|right|correct|perfect|great|exactly)", 0.9),
        (r"(you'?re|you are) (right|correct|helpful|smart)", 0.9),
        (r"(good|great|nice|excellent) (job|work|answer|response)", 0.9),
        (r"that (helps|helped|worked)", 0.85),
        (r"(this|that) is what i (needed|wanted|was looking for)", 0.9),
        (r"(exactly|precisely) what i (meant|wanted|needed)", 0.95),
        (r"you (got|nailed|understood) it", 0.9),
        (r"spot on", 0.9),
        (r"well (said|done|put)", 0.85),
    ]
    
    NEGATIVE_PATTERNS: list[tuple[str, float]] = [
        # Explicit negative feedback
        (r"that('?s| is| was)? ?not (helpful|useful|right|what i asked)", 0.85),
        (r"(that|this)('?s| is)n'?t (helpful|useful|right|what i asked)", 0.85),
        (r"(i don'?t understand|what\?|huh\?)", 0.6),
        (r"(try again|no,? that'?s not)", 0.8),
        (r"you (don'?t|didn'?t) (understand|get it|help)", 0.85),
        (r"(that|this) (doesn'?t|does not) (help|make sense|work)", 0.85),
        (r"(useless|unhelpful|pointless)", 0.9),
        (r"(that'?s|this is) (confusing|unclear|wrong)", 0.85),
        (r"you'?re (not|being) (helpful|useful)", 0.9),
        (r"not (helpful|useful)", 0.7),
        (r"can you (do better|try harder|actually help)", 0.8),
        (r"(terrible|awful|bad) (answer|response)", 0.9),
        (r"(didn'?t|doesn'?t) answer my (question|query)", 0.85),
        (r"off topic", 0.8),
        (r"that'?s not (it|right)", 0.8),
        (r"(what|huh)\?+$", 0.5),  # Confusion
    ]
    
    def __init__(self):
        """Initialize the conversation detector."""
        # Track conversation state
        self.last_ai_response: Optional[str] = None
        self.last_user_message: Optional[str] = None
        self.conversation_context: list[tuple[str, str]] = []  # [(role, text), ...]
        
        # Statistics
        self.detections_count = {
            'correction': 0,
            'teaching': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
        }
        
        # Compile patterns for efficiency
        self._compiled_patterns = {
            'correction': [(re.compile(p, re.IGNORECASE), c) for p, c in self.CORRECTION_PATTERNS],
            'teaching': [(re.compile(p, re.IGNORECASE), c) for p, c in self.TEACHING_PATTERNS],
            'positive_feedback': [(re.compile(p, re.IGNORECASE), c) for p, c in self.POSITIVE_PATTERNS],
            'negative_feedback': [(re.compile(p, re.IGNORECASE), c) for p, c in self.NEGATIVE_PATTERNS],
        }
        
        logger.debug("ConversationDetector initialized")
    
    def on_user_message(self, message: str) -> Optional[DetectedLearning]:
        """
        Process user message and detect learning opportunity.
        
        Call this BEFORE generating AI response.
        
        Args:
            message: The user's message text
        
        Returns:
            DetectedLearning if a learning opportunity is found, None otherwise
        """
        if not message or not message.strip():
            return None
        
        message = message.strip()
        
        # Store for context
        self.last_user_message = message
        self.conversation_context.append(('user', message))
        
        # Trim context to last 10 exchanges
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # Try to detect each type of learning opportunity
        # Order matters: correction > teaching > negative > positive
        # (correction is most actionable, positive is least urgent)
        
        detected = None
        
        # Check for corrections (highest priority)
        detected = self._detect_correction(message)
        if detected:
            self.detections_count['correction'] += 1
            logger.info(f"Detected correction: {detected.confidence:.2f} confidence")
            return detected
        
        # Check for teaching
        detected = self._detect_teaching(message)
        if detected:
            self.detections_count['teaching'] += 1
            logger.info(f"Detected teaching: {detected.confidence:.2f} confidence")
            return detected
        
        # Check for behavior rules (user teaching custom action sequences)
        detected = self._detect_behavior_rule(message)
        if detected:
            self.detections_count['teaching'] += 1  # Count as teaching
            logger.info(f"Detected behavior rule: {detected.confidence:.2f} confidence")
            return detected
        
        # Check for negative feedback (higher priority than positive)
        detected = self._detect_negative_feedback(message)
        if detected:
            self.detections_count['negative_feedback'] += 1
            logger.info(f"Detected negative feedback: {detected.confidence:.2f} confidence")
            return detected
        
        # Check for positive feedback
        detected = self._detect_positive_feedback(message)
        if detected:
            self.detections_count['positive_feedback'] += 1
            logger.info(f"Detected positive feedback: {detected.confidence:.2f} confidence")
            return detected
        
        return None
    
    def on_ai_response(self, response: str) -> None:
        """
        Track AI response for future correction matching.
        
        Call this AFTER generating AI response.
        
        Args:
            response: The AI's response text
        """
        if response and response.strip():
            self.last_ai_response = response.strip()
            self.conversation_context.append(('ai', self.last_ai_response))
    
    def to_learning_example(self, detected: DetectedLearning) -> LearningExample:
        """
        Convert detected learning to LearningExample for the queue.
        
        This bridges the gap between detection and the existing
        LearningEngine system.
        
        Args:
            detected: The detected learning opportunity
        
        Returns:
            LearningExample ready for the learning queue
        """
        # Map detection type to learning source and priority
        source_map = {
            'correction': (LearningSource.CORRECTION, Priority.CRITICAL),
            'teaching': (LearningSource.CONVERSATION, Priority.HIGH),
            'positive_feedback': (LearningSource.CONVERSATION, Priority.MEDIUM),
            'negative_feedback': (LearningSource.CORRECTION, Priority.HIGH),
        }
        
        source, priority = source_map.get(
            detected.type, 
            (LearningSource.CONVERSATION, Priority.MEDIUM)
        )
        
        # Build metadata
        metadata = {
            'detection_type': detected.type,
            'confidence': detected.confidence,
            'timestamp': detected.timestamp,
        }
        
        if detected.corrected_from:
            metadata['corrected_from'] = detected.corrected_from
        if detected.corrected_to:
            metadata['corrected_to'] = detected.corrected_to
        if detected.user_preference:
            metadata['user_preference'] = detected.user_preference
        if detected.context:
            metadata['context'] = detected.context
        
        return LearningExample(
            input_text=detected.input_text,
            output_text=detected.target_output,
            source=source,
            priority=priority,
            quality_score=detected.confidence,
            metadata=metadata,
        )
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        total = sum(self.detections_count.values())
        return {
            'total_detections': total,
            **self.detections_count,
            'context_length': len(self.conversation_context),
        }
    
    def reset_context(self) -> None:
        """Reset conversation context (e.g., for new conversation)."""
        self.last_ai_response = None
        self.last_user_message = None
        self.conversation_context = []
    
    # =========================================================================
    # PRIVATE DETECTION METHODS
    # =========================================================================
    
    def _detect_correction(self, message: str) -> Optional[DetectedLearning]:
        """Detect if the user is correcting the AI."""
        best_match = None
        best_confidence = 0.0
        matched_pattern = None
        
        for pattern, confidence in self._compiled_patterns['correction']:
            match = pattern.search(message)
            if match and confidence > best_confidence:
                best_confidence = confidence
                best_match = match
                matched_pattern = pattern
        
        if best_match and best_confidence >= 0.6:  # Minimum threshold
            # Extract what the correction is about
            corrected_from = self.last_ai_response if self.last_ai_response else ""
            corrected_to = message
            
            # Try to extract specific correction from patterns like "not X, Y"
            not_pattern = re.search(r"not (.+?)[,.]?\s+(but |it'?s |i meant )?(.+)", message, re.IGNORECASE)
            if not_pattern:
                corrected_from = not_pattern.group(1).strip()
                corrected_to = not_pattern.group(3).strip() if not_pattern.group(3) else message
            
            # Build target output (what AI should have said)
            target = self._build_correction_target(message, corrected_to)
            
            return DetectedLearning(
                type='correction',
                input_text=self.last_user_message or "",
                target_output=target,
                confidence=best_confidence,
                context=self.last_ai_response,
                corrected_from=corrected_from[:200] if corrected_from else None,
                corrected_to=corrected_to[:200] if corrected_to else None,
            )
        
        return None
    
    def _detect_teaching(self, message: str) -> Optional[DetectedLearning]:
        """Detect if the user is teaching the AI something."""
        best_match = None
        best_confidence = 0.0
        
        for pattern, confidence in self._compiled_patterns['teaching']:
            match = pattern.search(message)
            if match and confidence > best_confidence:
                best_confidence = confidence
                best_match = match
        
        if best_match and best_confidence >= 0.6:
            # Extract what's being taught
            user_preference = None
            target = self._build_teaching_target(message, best_match)
            
            # Try to extract specific information
            # Name patterns
            name_match = re.search(r"(my name is|i'?m called|call me) (.+?)(?:[.!,]|$)", message, re.IGNORECASE)
            if name_match:
                user_preference = f"name:{name_match.group(2).strip()}"
                target = f"I'll remember that your name is {name_match.group(2).strip()}."
            
            # Preference patterns  
            pref_match = re.search(r"i (like|love|hate|prefer|enjoy|dislike) (.+?)(?:[.!,]|$)", message, re.IGNORECASE)
            if pref_match:
                verb = pref_match.group(1).lower()
                thing = pref_match.group(2).strip()
                user_preference = f"{verb}:{thing}"
                target = f"I've noted that you {verb} {thing}."
            
            # "My X is Y" patterns
            my_match = re.search(r"my (\w+) is (.+?)(?:[.!,]|$)", message, re.IGNORECASE)
            if my_match:
                attr = my_match.group(1).lower()
                value = my_match.group(2).strip()
                user_preference = f"{attr}:{value}"
                target = f"I'll remember that your {attr} is {value}."
            
            return DetectedLearning(
                type='teaching',
                input_text=message,
                target_output=target,
                confidence=best_confidence,
                user_preference=user_preference,
            )
        
        return None
    
    def _detect_behavior_rule(self, message: str) -> Optional[DetectedLearning]:
        """
        Detect if the user is teaching a behavior rule.
        
        Examples:
        - "Whenever you teleport, spawn a portal gun first"
        - "Before you attack, always cast a shield spell"
        - "When you eat, hold the food first"
        """
        try:
            from .behavior_preferences import check_behavior_statement, get_behavior_manager
            
            # Quick check if this looks like a behavior statement
            if not check_behavior_statement(message):
                return None
            
            # Try to learn the behavior
            manager = get_behavior_manager()
            rule = manager.learn_from_statement(message)
            
            if rule:
                # Successfully learned a behavior rule
                action_desc = ", ".join(
                    f"{a.timing.value} '{rule.trigger_action}' -> '{a.tool_name}'"
                    for a in rule.actions
                )
                
                return DetectedLearning(
                    type='teaching',
                    input_text=message,
                    target_output=f"I've learned a new behavior: {action_desc}. I'll remember this for future actions.",
                    confidence=0.9,
                    user_preference=f"behavior_rule:{rule.id}",
                    context=f"trigger={rule.trigger_action}, actions={len(rule.actions)}",
                )
                
        except ImportError:
            pass  # Behavior preferences module not available
        except Exception as e:
            logger.debug(f"Error detecting behavior rule: {e}")
        
        return None
    
    def _detect_positive_feedback(self, message: str) -> Optional[DetectedLearning]:
        """Detect if the user is giving positive feedback."""
        best_confidence = 0.0
        
        for pattern, confidence in self._compiled_patterns['positive_feedback']:
            match = pattern.search(message)
            if match and confidence > best_confidence:
                best_confidence = confidence
        
        if best_confidence >= 0.7:  # Higher threshold for positive
            # Positive feedback reinforces the last AI response
            return DetectedLearning(
                type='positive_feedback',
                input_text=self.last_user_message or "",
                target_output=self.last_ai_response or "",
                confidence=best_confidence,
                context=f"User feedback: {message}",
            )
        
        return None
    
    def _detect_negative_feedback(self, message: str) -> Optional[DetectedLearning]:
        """Detect if the user is giving negative feedback."""
        best_confidence = 0.0
        
        for pattern, confidence in self._compiled_patterns['negative_feedback']:
            match = pattern.search(message)
            if match and confidence > best_confidence:
                best_confidence = confidence
        
        if best_confidence >= 0.6:
            # Negative feedback indicates the last response was problematic
            return DetectedLearning(
                type='negative_feedback',
                input_text=self.last_user_message or "",
                target_output="[NEEDS BETTER RESPONSE]",  # Placeholder for improvement
                confidence=best_confidence,
                context=self.last_ai_response,
                corrected_from=self.last_ai_response,
            )
        
        return None
    
    def _build_correction_target(self, message: str, corrected_to: str) -> str:
        """Build the target output for a correction."""
        # Try to formulate what the AI should have said
        if corrected_to and corrected_to != message:
            return f"I understand. {corrected_to}"
        return f"I apologize for the confusion. You're right: {message}"
    
    def _build_teaching_target(self, message: str, match: re.Match) -> str:
        """Build the target output for teaching."""
        # Extract the information being taught
        try:
            groups = match.groups()
            if groups and groups[-1]:
                info = groups[-1].strip()
                return f"I'll remember that {info}."
        except Exception as e:
            logger.debug(f"Could not extract teaching info: {e}")
        return f"Thank you for telling me. I'll remember this."


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_learning(
    user_message: str,
    last_ai_response: Optional[str] = None
) -> Optional[DetectedLearning]:
    """
    One-shot learning detection (stateless).
    
    For simple use cases where you don't need full conversation tracking.
    
    Args:
        user_message: The user's message
        last_ai_response: The previous AI response (for correction context)
    
    Returns:
        DetectedLearning if found, None otherwise
    
    Example:
        detected = detect_learning("No, my name is Bob", "Nice to meet you, Alice!")
    """
    detector = ConversationDetector()
    if last_ai_response:
        detector.last_ai_response = last_ai_response
    return detector.on_user_message(user_message)


def is_correction(message: str) -> bool:
    """Check if a message is likely a correction."""
    detector = ConversationDetector()
    detected = detector.on_user_message(message)
    return detected is not None and detected.type == 'correction'


def is_teaching(message: str) -> bool:
    """Check if a message is likely teaching something."""
    detector = ConversationDetector()
    detected = detector.on_user_message(message)
    return detected is not None and detected.type == 'teaching'


def is_feedback(message: str) -> tuple[bool, Optional[str]]:
    """
    Check if a message is feedback and what kind.
    
    Returns:
        (is_feedback, feedback_type) where feedback_type is 
        'positive', 'negative', or None
    """
    detector = ConversationDetector()
    detected = detector.on_user_message(message)
    
    if detected is None:
        return (False, None)
    
    if detected.type == 'positive_feedback':
        return (True, 'positive')
    elif detected.type == 'negative_feedback':
        return (True, 'negative')
    
    return (False, None)
