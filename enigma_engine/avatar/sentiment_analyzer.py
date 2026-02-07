"""
Sentiment Analyzer - Extract emotions from text for avatar reactions.

This module analyzes text to detect emotional tone for avatar expressions:
- Positive emotions (happy, excited, grateful)
- Negative emotions (sad, frustrated, worried)
- Neutral (informational, factual)
- Special states (confused, thinking, surprised)

The detected sentiment can be used to make the avatar react naturally
to conversation content.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Avatar-compatible emotion types."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    THINKING = "thinking"
    CONFUSED = "confused"
    EXCITED = "excited"
    WORRIED = "worried"
    GRATEFUL = "grateful"
    FRUSTRATED = "frustrated"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    
    emotion: EmotionType
    confidence: float  # 0.0 to 1.0
    keywords: List[str]  # Matched keywords that triggered this emotion
    raw_score: Dict[str, float]  # Scores for each emotion type
    
    @property
    def avatar_expression(self) -> str:
        """Convert to avatar-compatible expression string."""
        # Map complex emotions to basic avatar expressions
        mapping = {
            EmotionType.NEUTRAL: "neutral",
            EmotionType.HAPPY: "happy",
            EmotionType.SAD: "sad",
            EmotionType.SURPRISED: "surprised",
            EmotionType.THINKING: "thinking",
            EmotionType.CONFUSED: "confused",
            EmotionType.EXCITED: "happy",  # Map to happy
            EmotionType.WORRIED: "sad",    # Map to sad
            EmotionType.GRATEFUL: "happy", # Map to happy
            EmotionType.FRUSTRATED: "sad", # Map to sad
        }
        return mapping.get(self.emotion, "neutral")


class SentimentAnalyzer:
    """
    Analyze text sentiment for avatar emotion reactions.
    
    Uses keyword and pattern matching for fast, offline analysis.
    More sophisticated than basic positive/negative, tuned for
    conversational AI interactions.
    """
    
    # Emotion keywords with weights
    EMOTION_KEYWORDS = {
        EmotionType.HAPPY: {
            "high": ["love", "amazing", "wonderful", "fantastic", "excellent", "great", "awesome", "happy", "joy"],
            "medium": ["good", "nice", "glad", "pleased", "enjoy", "like", "fun", "cool", "perfect"],
            "low": ["okay", "fine", "alright", "thanks", "thank"],
        },
        EmotionType.SAD: {
            "high": ["terrible", "awful", "horrible", "devastated", "heartbroken", "miserable", "depressed"],
            "medium": ["sad", "unhappy", "disappointed", "upset", "sorry", "unfortunate", "regret"],
            "low": ["down", "blue", "meh", "not great"],
        },
        EmotionType.SURPRISED: {
            "high": ["unbelievable", "incredible", "shocking", "astonishing", "mind-blowing"],
            "medium": ["wow", "surprised", "unexpected", "really", "seriously", "omg", "whoa"],
            "low": ["oh", "huh", "interesting", "didn't expect"],
        },
        EmotionType.THINKING: {
            "high": ["analyzing", "considering", "evaluating", "processing"],
            "medium": ["think", "wonder", "pondering", "maybe", "perhaps", "probably", "might"],
            "low": ["hmm", "let me see", "well", "so"],
        },
        EmotionType.CONFUSED: {
            "high": ["completely lost", "makes no sense", "totally confused"],
            "medium": ["confused", "don't understand", "unclear", "what do you mean", "how does"],
            "low": ["huh", "wait", "sorry", "pardon", "?"],
        },
        EmotionType.EXCITED: {
            "high": ["can't wait", "so excited", "thrilled", "pumped"],
            "medium": ["excited", "eager", "looking forward", "yes!", "finally"],
            "low": ["nice", "cool", "sweet"],
        },
        EmotionType.WORRIED: {
            "high": ["terrified", "panic", "desperate", "very worried"],
            "medium": ["worried", "concerned", "anxious", "nervous", "afraid"],
            "low": ["unsure", "hesitant", "careful"],
        },
        EmotionType.GRATEFUL: {
            "high": ["so grateful", "deeply appreciate", "means so much"],
            "medium": ["thank you", "thanks", "appreciate", "grateful", "helpful"],
            "low": ["thx", "ty", "cheers"],
        },
        EmotionType.FRUSTRATED: {
            "high": ["furious", "infuriating", "unacceptable", "ridiculous"],
            "medium": ["frustrated", "annoying", "irritating", "ugh", "come on"],
            "low": ["sigh", "again", "still"],
        },
    }
    
    # Patterns that indicate specific emotions
    EMOTION_PATTERNS = {
        EmotionType.HAPPY: [
            r":\)",
            r"😊|😃|😄|😁|🙂|❤️|💕|🎉",
            r"\bha+h*a+\b",  # haha, hahaha
            r"\blo+l+\b",    # lol, lool
        ],
        EmotionType.SAD: [
            r":\(",
            r"😢|😭|😔|😞|💔",
            r"\b(?:unfortunately|sadly)\b",
        ],
        EmotionType.SURPRISED: [
            r"!{2,}",  # Multiple exclamation marks
            r"😮|😲|🤯|😱",
            r"\bwow+\b",
            r"\bwhoa+\b",
        ],
        EmotionType.CONFUSED: [
            r"\?{2,}",  # Multiple question marks
            r"🤔|😕|🤷",
            r"\bwhat\??$",
            r"\bhow\??$",
        ],
        EmotionType.EXCITED: [
            r"!+$",  # Exclamation at end
            r"🎉|🥳|✨|🔥",
            r"\byes+!*\b",
        ],
    }
    
    # Negation words that can flip sentiment
    NEGATION_WORDS = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", 
                      "can't", "couldn't", "wouldn't", "shouldn't", "isn't", "aren't"]
    
    # Intensifiers that boost confidence
    INTENSIFIERS = ["very", "really", "so", "extremely", "absolutely", "totally", 
                    "completely", "incredibly", "super", "quite"]
    
    def __init__(self):
        # Compile patterns for efficiency
        self._compiled_patterns = {}
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            self._compiled_patterns[emotion] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze text to detect emotional content.
        
        Args:
            text: The text to analyze
            
        Returns:
            SentimentResult with detected emotion and confidence
        """
        if not text or not text.strip():
            return SentimentResult(
                emotion=EmotionType.NEUTRAL,
                confidence=1.0,
                keywords=[],
                raw_score={e.value: 0.0 for e in EmotionType}
            )
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Calculate scores for each emotion
        scores: Dict[EmotionType, float] = {e: 0.0 for e in EmotionType}
        matched_keywords: Dict[EmotionType, List[str]] = {e: [] for e in EmotionType}
        
        # Check for negation in text
        has_negation = any(neg in words for neg in self.NEGATION_WORDS)
        
        # Check for intensifiers
        intensifier_count = sum(1 for w in words if w in self.INTENSIFIERS)
        intensity_boost = 1.0 + (intensifier_count * 0.2)  # Up to 2x boost
        
        # Score by keywords
        for emotion, keyword_sets in self.EMOTION_KEYWORDS.items():
            for priority, keywords in keyword_sets.items():
                weight = {"high": 3.0, "medium": 2.0, "low": 1.0}[priority]
                
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        scores[emotion] += weight
                        matched_keywords[emotion].append(keyword)
        
        # Score by patterns
        for emotion, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    scores[emotion] += 2.0
        
        # Apply intensity boost
        scores = {e: s * intensity_boost for e, s in scores.items()}
        
        # Handle negation (flip positive/negative)
        if has_negation:
            # Swap happy/sad scores partially
            scores[EmotionType.HAPPY], scores[EmotionType.SAD] = (
                scores[EmotionType.HAPPY] * 0.3 + scores[EmotionType.SAD] * 0.7,
                scores[EmotionType.SAD] * 0.3 + scores[EmotionType.HAPPY] * 0.7,
            )
        
        # Find dominant emotion
        if all(s == 0 for s in scores.values()):
            dominant = EmotionType.NEUTRAL
            confidence = 0.5
        else:
            dominant = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = scores[dominant] / total_score if total_score > 0 else 0.5
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, max(0.0, confidence))
        
        return SentimentResult(
            emotion=dominant,
            confidence=confidence,
            keywords=matched_keywords.get(dominant, []),
            raw_score={e.value: s for e, s in scores.items()}
        )
    
    def analyze_conversation(self, messages: List[str]) -> SentimentResult:
        """
        Analyze sentiment across multiple messages.
        
        Useful for getting overall conversation mood.
        Later messages are weighted more heavily.
        """
        if not messages:
            return self.analyze("")
        
        # Analyze each message with increasing weight for recent ones
        weighted_scores: Dict[EmotionType, float] = {e: 0.0 for e in EmotionType}
        all_keywords = []
        
        for i, msg in enumerate(messages):
            weight = (i + 1) / len(messages)  # More recent = higher weight
            result = self.analyze(msg)
            
            for emotion_str, score in result.raw_score.items():
                emotion = EmotionType(emotion_str)
                weighted_scores[emotion] += score * weight
            
            all_keywords.extend(result.keywords)
        
        # Find dominant
        dominant = max(weighted_scores, key=weighted_scores.get)
        total = sum(weighted_scores.values())
        confidence = weighted_scores[dominant] / total if total > 0 else 0.5
        
        return SentimentResult(
            emotion=dominant,
            confidence=min(1.0, confidence),
            keywords=all_keywords,
            raw_score={e.value: s for e, s in weighted_scores.items()}
        )


# Global analyzer instance
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def analyze_text(text: str) -> SentimentResult:
    """Convenience function to analyze text sentiment."""
    return get_sentiment_analyzer().analyze(text)


def get_avatar_expression(text: str) -> str:
    """Get the avatar expression for the given text."""
    result = analyze_text(text)
    return result.avatar_expression


def analyze_for_avatar(text: str) -> Tuple[str, float]:
    """
    Analyze text and return avatar expression with confidence.
    
    Returns:
        Tuple of (expression_name, confidence)
    """
    result = analyze_text(text)
    return result.avatar_expression, result.confidence
