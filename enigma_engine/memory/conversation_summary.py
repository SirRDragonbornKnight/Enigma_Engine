"""
================================================================================
📝 CONVERSATION SUMMARIZER - COMPRESS CONTEXT FOR CONTINUITY
================================================================================

This module provides conversation summarization capabilities:
1. Summarize long conversations to fit in context windows
2. Extract key facts and topics for handoff to other AIs
3. Generate "previously on..." context for continuation
4. Create exportable conversation digests

📍 FILE: enigma_engine/memory/conversation_summary.py
🏷️ TYPE: Memory Compression & Context Management
🎯 MAIN CLASS: ConversationSummarizer

┌─────────────────────────────────────────────────────────────────────────────┐
│  SUMMARIZATION FLOW:                                                        │
│                                                                             │
│  [100 messages] → ConversationSummarizer → [Compact Summary]               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ "User discussed: coding project, AI training, cats                  │   │
│  │  Key facts: User has 2 cats named Whiskers and Mittens              │   │
│  │  Last topic: How to train a custom model                            │   │
│  │  User preferences: Prefers Python, likes detailed explanations"     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    → USES:      enigma_engine/core/inference.py (for AI-powered summarization)
    → USES:      enigma_engine/memory/manager.py (conversation storage)
    ← USED BY:   enigma_engine/gui/tabs/chat_tab.py (summary button)
    ← USED BY:   enigma_engine/core/tool_router.py (context for routing)

📖 USAGE:
    from enigma_engine.memory.conversation_summary import ConversationSummarizer
    
    summarizer = ConversationSummarizer()
    
    # Summarize a conversation
    summary = summarizer.summarize(messages)
    
    # Get context for continuing conversation
    context = summarizer.get_continuation_context(messages)
    
    # Export for another AI
    export = summarizer.export_for_handoff(messages)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default limits (can be overridden)
DEFAULT_MAX_SUMMARY_TOKENS = 500
DEFAULT_MAX_CONTEXT_MESSAGES = 20
DEFAULT_MIN_MESSAGES_TO_SUMMARIZE = 6


@dataclass
class ConversationSummary:
    """
    Structured summary of a conversation.
    
    Contains all the key information needed to:
    - Continue a conversation later
    - Hand off to another AI
    - Provide context for routing
    """
    # Core summary
    summary_text: str = ""
    
    # Extracted information
    topics: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    user_preferences: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    
    # Metadata
    message_count: int = 0
    time_range: tuple[float, float] = (0.0, 0.0)
    last_topic: str = ""
    sentiment: str = "neutral"  # positive, negative, neutral
    
    # For continuation
    continuation_context: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary_text": self.summary_text,
            "topics": self.topics,
            "key_facts": self.key_facts,
            "user_preferences": self.user_preferences,
            "action_items": self.action_items,
            "message_count": self.message_count,
            "time_range": list(self.time_range),
            "last_topic": self.last_topic,
            "sentiment": self.sentiment,
            "continuation_context": self.continuation_context,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationSummary:
        """Create from dictionary."""
        return cls(
            summary_text=data.get("summary_text", ""),
            topics=data.get("topics", []),
            key_facts=data.get("key_facts", []),
            user_preferences=data.get("user_preferences", []),
            action_items=data.get("action_items", []),
            message_count=data.get("message_count", 0),
            time_range=tuple(data.get("time_range", [0.0, 0.0])),
            last_topic=data.get("last_topic", ""),
            sentiment=data.get("sentiment", "neutral"),
            continuation_context=data.get("continuation_context", ""),
        )
    
    def to_context_string(self) -> str:
        """
        Format as a context string for injection into prompts.
        
        This is what gets prepended to give the AI context about
        the previous conversation.
        """
        parts = []
        
        if self.summary_text:
            parts.append(f"Previous conversation summary:\n{self.summary_text}")
        
        if self.key_facts:
            parts.append("Key facts from conversation:\n" + "\n".join(f"- {f}" for f in self.key_facts[:5]))
        
        if self.user_preferences:
            parts.append("User preferences:\n" + "\n".join(f"- {p}" for p in self.user_preferences[:3]))
        
        if self.last_topic:
            parts.append(f"Last topic discussed: {self.last_topic}")
        
        if self.action_items:
            parts.append("Pending action items:\n" + "\n".join(f"- {a}" for a in self.action_items[:3]))
        
        return "\n\n".join(parts)


# =============================================================================
# CONVERSATION SUMMARIZER
# =============================================================================

class ConversationSummarizer:
    """
    Summarizes conversations for context management and handoff.
    
    📖 WHY THIS EXISTS:
    Long conversations overflow the context window and cause hallucinations.
    This summarizer compresses old messages while preserving key information.
    
    📐 TWO MODES:
    1. EXTRACTIVE: Uses rules and patterns (fast, no model needed)
    2. ABSTRACTIVE: Uses AI to generate natural summary (better quality)
    
    📐 USAGE:
        summarizer = ConversationSummarizer()
        
        # Quick extractive summary
        summary = summarizer.summarize_extractive(messages)
        
        # AI-powered summary (requires engine)
        summary = summarizer.summarize_with_ai(messages, engine)
        
        # Get context for prompt injection
        context = summary.to_context_string()
    """
    
    def __init__(
        self,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
        min_messages: int = DEFAULT_MIN_MESSAGES_TO_SUMMARIZE
    ):
        """
        Initialize summarizer.
        
        Args:
            max_summary_tokens: Maximum tokens in generated summary
            min_messages: Minimum messages before summarization kicks in
        """
        self.max_summary_tokens = max_summary_tokens
        self.min_messages = min_messages
        
        # Patterns for extraction
        self._preference_patterns = [
            r"i (?:prefer|like|want|love|enjoy)\s+(.+?)(?:\.|$)",
            r"(?:please|always|never)\s+(.+?)(?:\.|$)",
            r"i(?:'m| am) (?:a |an )?(.+?)(?:\.|$)",
            r"my (?:name|job|work|hobby|favorite)\s+(?:is|are)\s+(.+?)(?:\.|$)",
        ]
        
        self._action_patterns = [
            r"(?:todo|task|remind me|don't forget|need to|should|must)\s*:?\s*(.+?)(?:\.|$)",
            r"(?:i will|i'll|we should|let's)\s+(.+?)(?:\.|$)",
        ]
        
        self._topic_keywords = {
            "coding": ["code", "program", "function", "bug", "error", "python", "javascript"],
            "ai": ["model", "train", "ai", "neural", "machine learning", "forge"],
            "creative": ["image", "draw", "paint", "create", "design", "art"],
            "help": ["help", "how to", "explain", "what is", "why"],
            "casual": ["hello", "hi", "thanks", "bye", "chat"],
        }
    
    def summarize_extractive(
        self,
        messages: list[dict[str, Any]],
        include_recent: int = 3
    ) -> ConversationSummary:
        """
        Create summary using rule-based extraction (no AI needed).
        
        This is fast and works without a model loaded.
        Good for real-time context management.
        
        Args:
            messages: List of message dicts with 'role', 'text', 'ts'
            include_recent: Number of recent messages to keep verbatim
            
        Returns:
            ConversationSummary with extracted information
        """
        if not messages:
            return ConversationSummary()
        
        summary = ConversationSummary()
        summary.message_count = len(messages)
        
        # Time range
        timestamps = [m.get("ts", 0) for m in messages if m.get("ts")]
        if timestamps:
            summary.time_range = (min(timestamps), max(timestamps))
        
        # Extract from all messages
        all_user_text = []
        all_ai_text = []
        
        for msg in messages:
            text = msg.get("text", msg.get("content", "")).lower()
            role = msg.get("role", "user")
            
            if role in ("user", "human"):
                all_user_text.append(text)
                
                # Extract preferences
                for pattern in self._preference_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    summary.user_preferences.extend(matches[:2])
                
                # Extract action items
                for pattern in self._action_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    summary.action_items.extend(matches[:2])
            else:
                all_ai_text.append(text)
        
        # Deduplicate
        summary.user_preferences = list(dict.fromkeys(summary.user_preferences))[:5]
        summary.action_items = list(dict.fromkeys(summary.action_items))[:5]
        
        # Detect topics
        combined_text = " ".join(all_user_text + all_ai_text)
        for topic, keywords in self._topic_keywords.items():
            if any(kw in combined_text for kw in keywords):
                summary.topics.append(topic)
        
        # Last topic from recent messages
        if messages:
            last_user_msgs = [m for m in messages[-6:] if m.get("role") in ("user", "human")]
            if last_user_msgs:
                last_text = last_user_msgs[-1].get("text", "")[:100]
                summary.last_topic = self._extract_topic_phrase(last_text)
        
        # Extract key facts (simple heuristic: sentences with "is", "are", "has", "have")
        fact_patterns = [
            r"(?:my|the|i|we)\s+\w+\s+(?:is|are|has|have)\s+(.+?)(?:\.|$)",
            r"(\w+)\s+(?:is called|is named|equals)\s+(.+?)(?:\.|$)",
        ]
        for text in all_user_text:
            for pattern in fact_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:1]:
                    if isinstance(match, tuple):
                        match = " ".join(match)
                    if len(match) > 10 and len(match) < 100:
                        summary.key_facts.append(match.strip())
        
        summary.key_facts = list(dict.fromkeys(summary.key_facts))[:5]
        
        # Generate summary text
        summary.summary_text = self._generate_extractive_summary(
            messages, summary.topics, include_recent
        )
        
        # Generate continuation context
        summary.continuation_context = self._generate_continuation_context(
            messages, summary, include_recent
        )
        
        return summary
    
    def _extract_topic_phrase(self, text: str) -> str:
        """Extract a short topic phrase from text."""
        # Remove common question words
        text = re.sub(r"^(how|what|why|when|where|can you|please|could you)\s+", "", text, flags=re.I)
        # Take first clause
        text = text.split(",")[0].split("?")[0].split(".")[0]
        # Limit length
        words = text.split()[:6]
        return " ".join(words).strip()
    
    def _generate_extractive_summary(
        self,
        messages: list[dict[str, Any]],
        topics: list[str],
        recent_count: int
    ) -> str:
        """Generate a text summary from extracted information."""
        parts = []
        
        # Message count
        user_count = sum(1 for m in messages if m.get("role") in ("user", "human"))
        parts.append(f"Conversation with {len(messages)} messages ({user_count} from user).")
        
        # Topics
        if topics:
            parts.append(f"Topics discussed: {', '.join(topics)}.")
        
        # Recent exchange summary
        if len(messages) > recent_count:
            recent = messages[-recent_count * 2:]  # Last N exchanges
            recent_topics = set()
            for m in recent:
                if m.get("role") in ("user", "human"):
                    text = m.get("text", "")[:50].lower()
                    for topic, keywords in self._topic_keywords.items():
                        if any(kw in text for kw in keywords):
                            recent_topics.add(topic)
            if recent_topics:
                parts.append(f"Recent focus: {', '.join(recent_topics)}.")
        
        return " ".join(parts)
    
    def _generate_continuation_context(
        self,
        messages: list[dict[str, Any]],
        summary: ConversationSummary,
        recent_count: int
    ) -> str:
        """Generate context string for continuing the conversation."""
        parts = []
        
        # Brief history
        parts.append(f"[Previous conversation: {summary.message_count} messages]")
        
        if summary.topics:
            parts.append(f"[Topics: {', '.join(summary.topics[:3])}]")
        
        if summary.key_facts:
            parts.append("[Key facts: " + "; ".join(summary.key_facts[:3]) + "]")
        
        # Include recent messages verbatim
        if len(messages) > recent_count:
            parts.append("\n[Recent conversation:]")
            for msg in messages[-recent_count * 2:]:
                role = "User" if msg.get("role") in ("user", "human") else "AI"
                text = msg.get("text", msg.get("content", ""))[:200]
                parts.append(f"{role}: {text}")
        
        return "\n".join(parts)
    
    def summarize_with_ai(
        self,
        messages: list[dict[str, Any]],
        engine: Any = None,
        model_name: str | None = None
    ) -> ConversationSummary:
        """
        Create summary using AI generation (better quality).
        
        Args:
            messages: List of message dicts
            engine: EnigmaEngine instance (loads one if None)
            model_name: Optional model name to use
            
        Returns:
            ConversationSummary with AI-generated content
        """
        # Start with extractive summary as base
        summary = self.summarize_extractive(messages)
        
        if len(messages) < self.min_messages:
            return summary
        
        # Load engine if needed
        if engine is None:
            try:
                from ..core.inference import EnigmaEngine
                engine = EnigmaEngine(model_name=model_name)
            except Exception as e:
                logger.warning(f"Could not load engine for AI summary: {e}")
                return summary
        
        # Build prompt for summarization
        conversation_text = self._format_conversation_for_prompt(messages)
        
        prompt = f"""Summarize this conversation briefly. Include:
1. Main topics discussed
2. Any important facts mentioned by the user
3. User preferences or requests
4. Any pending tasks or action items

Conversation:
{conversation_text}

Summary:"""
        
        try:
            # Generate summary
            ai_summary = engine.generate(
                prompt,
                max_gen=self.max_summary_tokens,
                temperature=0.3  # Low temperature for factual summary
            )
            
            # Clean up response
            ai_summary = ai_summary.strip()
            if ai_summary.startswith("Summary:"):
                ai_summary = ai_summary[8:].strip()
            
            summary.summary_text = ai_summary
            
            # Try to extract structured info from AI summary
            self._parse_ai_summary(ai_summary, summary)
            
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}")
        
        return summary
    
    def _format_conversation_for_prompt(
        self,
        messages: list[dict[str, Any]],
        max_messages: int = 30
    ) -> str:
        """Format messages for inclusion in prompt."""
        # Take most recent messages if too many
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        
        lines = []
        for msg in messages:
            role = "User" if msg.get("role") in ("user", "human") else "AI"
            text = msg.get("text", msg.get("content", ""))[:300]
            lines.append(f"{role}: {text}")
        
        return "\n".join(lines)
    
    def _parse_ai_summary(self, ai_text: str, summary: ConversationSummary):
        """Try to extract structured info from AI-generated summary."""
        text_lower = ai_text.lower()
        
        # Look for topics
        topic_match = re.search(r"topics?[:\s]+([^.]+)", text_lower)
        if topic_match:
            topics = [t.strip() for t in topic_match.group(1).split(",")]
            summary.topics = topics[:5]
        
        # Look for facts
        fact_match = re.search(r"facts?[:\s]+([^.]+)", text_lower)
        if fact_match:
            facts = [f.strip() for f in fact_match.group(1).split(",")]
            summary.key_facts.extend(facts[:3])
        
        # Look for action items
        action_match = re.search(r"(?:tasks?|action items?|todo)[:\s]+([^.]+)", text_lower)
        if action_match:
            actions = [a.strip() for a in action_match.group(1).split(",")]
            summary.action_items.extend(actions[:3])
    
    def get_handoff_context(
        self,
        messages: list[dict[str, Any]],
        target_ai: str = "general"
    ) -> str:
        """
        Generate context for handing conversation to another AI.
        
        This creates a formatted context string that can be prepended
        to prompts when routing to a specialized AI or continuing
        a conversation in a different session.
        
        Args:
            messages: Conversation messages
            target_ai: Type of AI receiving handoff (for context hints)
            
        Returns:
            Formatted context string
        """
        summary = self.summarize_extractive(messages)
        
        parts = [
            "=== CONVERSATION CONTEXT ===",
            f"Previous messages: {summary.message_count}",
        ]
        
        if summary.topics:
            parts.append(f"Topics: {', '.join(summary.topics)}")
        
        if summary.key_facts:
            parts.append("Key facts:")
            for fact in summary.key_facts[:5]:
                parts.append(f"  - {fact}")
        
        if summary.user_preferences:
            parts.append("User preferences:")
            for pref in summary.user_preferences[:3]:
                parts.append(f"  - {pref}")
        
        if summary.last_topic:
            parts.append(f"Last topic: {summary.last_topic}")
        
        # Add recent messages
        parts.append("\nRecent exchange:")
        for msg in messages[-4:]:
            role = "User" if msg.get("role") in ("user", "human") else "AI"
            text = msg.get("text", msg.get("content", ""))[:150]
            parts.append(f"  {role}: {text}")
        
        parts.append("=== END CONTEXT ===\n")
        
        return "\n".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_summarizer: ConversationSummarizer | None = None


def get_summarizer() -> ConversationSummarizer:
    """Get or create the default summarizer."""
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = ConversationSummarizer()
    return _default_summarizer


def summarize_conversation(
    messages: list[dict[str, Any]],
    use_ai: bool = False,
    engine: Any = None
) -> ConversationSummary:
    """
    Summarize a conversation.
    
    Args:
        messages: List of message dicts with 'role', 'text', 'ts'
        use_ai: Whether to use AI for better summary (slower)
        engine: EnigmaEngine instance (optional, loaded if needed)
        
    Returns:
        ConversationSummary object
    """
    summarizer = get_summarizer()
    
    if use_ai:
        return summarizer.summarize_with_ai(messages, engine)
    else:
        return summarizer.summarize_extractive(messages)


def get_continuation_context(
    messages: list[dict[str, Any]],
    max_tokens: int = 500
) -> str:
    """
    Get context string for continuing a conversation.
    
    Args:
        messages: Conversation messages
        max_tokens: Approximate max tokens for context
        
    Returns:
        Context string to prepend to prompts
    """
    summarizer = get_summarizer()
    summary = summarizer.summarize_extractive(messages)
    
    context = summary.to_context_string()
    
    # Rough truncation if needed (4 chars per token approx)
    max_chars = max_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[Context truncated]"
    
    return context


def export_for_handoff(
    messages: list[dict[str, Any]],
    format: str = "text"
) -> str:
    """
    Export conversation for handoff to another AI.
    
    Args:
        messages: Conversation messages
        format: "text" or "json"
        
    Returns:
        Formatted export string
    """
    summarizer = get_summarizer()
    
    if format == "json":
        summary = summarizer.summarize_extractive(messages)
        return json.dumps(summary.to_dict(), indent=2)
    else:
        return summarizer.get_handoff_context(messages)
