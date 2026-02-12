"""
================================================================================
Context Window Display - Track and visualize context usage.
================================================================================

Monitor and display context window utilization:
- Track token usage across conversations
- Calculate remaining capacity
- Visualize usage with progress bars
- Warn when approaching limits
- Support for various tokenizers

USAGE:
    from enigma_engine.utils.context_window import ContextTracker, get_context_tracker
    
    tracker = get_context_tracker(max_tokens=4096)
    
    # Add messages and track
    tracker.add_message("user", "Hello, how are you?")
    tracker.add_message("assistant", "I'm doing well, thank you!")
    
    # Check usage
    usage = tracker.get_usage()
    print(f"Used: {usage.used_tokens} / {usage.max_tokens}")
    print(f"Remaining: {usage.remaining_tokens}")
    print(f"Usage: {usage.percentage:.1f}%")
    
    # Get visual bar
    print(tracker.get_progress_bar())  # [=========>          ] 45%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class UsageLevel(Enum):
    """Context usage level."""
    LOW = auto()       # < 50%
    MEDIUM = auto()    # 50-75%
    HIGH = auto()      # 75-90%
    CRITICAL = auto()  # > 90%


@dataclass
class ContextUsage:
    """Context window usage statistics."""
    used_tokens: int
    max_tokens: int
    remaining_tokens: int
    percentage: float
    level: UsageLevel
    message_count: int
    system_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0


@dataclass
class TokenEstimate:
    """Estimation of tokens for text."""
    text: str
    char_count: int
    word_count: int
    estimated_tokens: int
    method: str  # "tiktoken", "simple", "sentencepiece"


@dataclass
class ContextConfig:
    """Configuration for context tracking."""
    max_tokens: int = 4096
    reserve_tokens: int = 500  # Reserve for response
    warn_percentage: float = 80.0
    critical_percentage: float = 95.0
    tokenizer_name: str = "cl100k_base"  # For tiktoken
    fallback_chars_per_token: float = 4.0
    # Auto-continue settings
    auto_continue_enabled: bool = True
    auto_continue_threshold: float = 85.0  # Auto-continue at this percentage
    auto_continue_keep_messages: int = 3   # Keep last N messages when continuing
    auto_continue_include_summary: bool = True  # Generate summary for new chat


class TokenCounter:
    """
    Counts tokens using various methods.
    """
    
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        """
        Initialize token counter.
        
        Args:
            tokenizer_name: Tiktoken encoder name (cl100k_base, p50k_base, etc.)
        """
        self._tokenizer_name = tokenizer_name
        self._tokenizer = None
        self._fallback_ratio = 4.0  # Chars per token estimate
        
        self._init_tokenizer()
    
    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        # Try tiktoken first (most accurate for modern models)
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(self._tokenizer_name)
            self._method = "tiktoken"
            logger.info(f"Using tiktoken ({self._tokenizer_name}) for token counting")
            return
        except ImportError:
            logger.debug("tiktoken not available")
        except Exception as e:
            logger.debug(f"tiktoken error: {e}")
        
        # Try sentencepiece
        try:
            pass

            # Use a simple estimate if no model loaded
            self._method = "sentencepiece_estimate"
            logger.info("Using sentencepiece estimate for token counting")
            return
        except ImportError:
            pass  # Intentionally silent
        
        # Fallback to character-based estimate
        self._method = "simple"
        logger.info("Using simple character-based token estimation")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tokenizer error, using fallback: {e}")
        
        # Fallback estimation
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens using heuristics."""
        # Average ~4 chars per token for English
        # Adjust for whitespace and punctuation
        char_count = len(text)
        word_count = len(text.split())
        
        # Use weighted average of character and word estimates
        char_estimate = char_count / self._fallback_ratio
        word_estimate = word_count * 1.3  # ~1.3 tokens per word on average
        
        return int((char_estimate + word_estimate) / 2)
    
    def get_estimate(self, text: str) -> TokenEstimate:
        """Get detailed token estimate."""
        return TokenEstimate(
            text=text[:100] + "..." if len(text) > 100 else text,
            char_count=len(text),
            word_count=len(text.split()),
            estimated_tokens=self.count_tokens(text),
            method=self._method
        )


class ContextTracker:
    """
    Track and manage context window usage.
    """
    
    def __init__(self, config: ContextConfig | None = None):
        """
        Initialize context tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or ContextConfig()
        self._counter = TokenCounter(self.config.tokenizer_name)
        
        self._messages: list[dict[str, Any]] = []
        self._system_prompt: str = ""
        
        self._usage_callbacks: list[Callable[[ContextUsage], None]] = []
        self._warning_fired = False
    
    def set_system_prompt(self, prompt: str) -> int:
        """
        Set the system prompt.
        
        Args:
            prompt: System prompt text
            
        Returns:
            Token count for system prompt
        """
        self._system_prompt = prompt
        return self._counter.count_tokens(prompt)
    
    def add_message(self, role: str, content: str) -> tuple[int, ContextUsage]:
        """
        Add a message and get updated usage.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            
        Returns:
            Tuple of (tokens added, current usage)
        """
        tokens = self._counter.count_tokens(content)
        
        self._messages.append({
            "role": role,
            "content": content,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat()
        })
        
        usage = self.get_usage()
        self._check_warnings(usage)
        self._notify_callbacks(usage)
        
        return tokens, usage
    
    def get_usage(self) -> ContextUsage:
        """
        Get current context usage.
        
        Returns:
            ContextUsage with current statistics
        """
        system_tokens = self._counter.count_tokens(self._system_prompt)
        user_tokens = sum(
            m["tokens"] for m in self._messages if m["role"] == "user"
        )
        assistant_tokens = sum(
            m["tokens"] for m in self._messages if m["role"] == "assistant"
        )
        
        used_tokens = system_tokens + user_tokens + assistant_tokens
        # Account for message formatting overhead (~4 tokens per message)
        used_tokens += len(self._messages) * 4
        
        effective_max = self.config.max_tokens - self.config.reserve_tokens
        remaining = max(0, effective_max - used_tokens)
        percentage = (used_tokens / effective_max * 100) if effective_max > 0 else 100
        
        # Determine level
        if percentage >= self.config.critical_percentage:
            level = UsageLevel.CRITICAL
        elif percentage >= self.config.warn_percentage:
            level = UsageLevel.HIGH
        elif percentage >= 50:
            level = UsageLevel.MEDIUM
        else:
            level = UsageLevel.LOW
        
        return ContextUsage(
            used_tokens=used_tokens,
            max_tokens=self.config.max_tokens,
            remaining_tokens=remaining,
            percentage=min(100, percentage),
            level=level,
            message_count=len(self._messages),
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens
        )
    
    def _check_warnings(self, usage: ContextUsage) -> None:
        """Check and log warnings."""
        if usage.level == UsageLevel.CRITICAL:
            if not self._warning_fired:
                logger.warning(f"Context nearly full: {usage.percentage:.1f}%")
                self._warning_fired = True
        elif usage.level == UsageLevel.HIGH:
            logger.info(f"Context usage high: {usage.percentage:.1f}%")
    
    def _notify_callbacks(self, usage: ContextUsage) -> None:
        """Notify usage callbacks."""
        for callback in self._usage_callbacks:
            try:
                callback(usage)
            except Exception as e:
                logger.error(f"Usage callback error: {e}")
    
    def add_usage_callback(self, callback: Callable[[ContextUsage], None]) -> None:
        """Add a callback for usage updates."""
        self._usage_callbacks.append(callback)
    
    def get_progress_bar(
        self,
        width: int = 30,
        fill_char: str = "=",
        empty_char: str = " ",
        show_percentage: bool = True
    ) -> str:
        """
        Get a text-based progress bar.
        
        Args:
            width: Bar width in characters
            fill_char: Character for filled portion
            empty_char: Character for empty portion
            show_percentage: Include percentage
            
        Returns:
            Progress bar string like [======>     ] 45%
        """
        usage = self.get_usage()
        filled = int(width * usage.percentage / 100)
        
        # Color indicator based on level
        indicator = ">" if filled < width else ""
        bar = fill_char * filled + indicator + empty_char * (width - filled - len(indicator))
        
        result = f"[{bar}]"
        if show_percentage:
            result += f" {usage.percentage:.0f}%"
        
        return result
    
    def get_colored_bar(self, width: int = 30) -> tuple[str, str]:
        """
        Get progress bar with color suggestion.
        
        Returns:
            Tuple of (bar_string, color_name)
        """
        usage = self.get_usage()
        bar = self.get_progress_bar(width)
        
        color_map = {
            UsageLevel.LOW: "green",
            UsageLevel.MEDIUM: "yellow",
            UsageLevel.HIGH: "orange",
            UsageLevel.CRITICAL: "red"
        }
        
        return bar, color_map[usage.level]
    
    def estimate_response_capacity(self) -> int:
        """
        Estimate how many tokens are available for response.
        
        Returns:
            Available tokens for response
        """
        usage = self.get_usage()
        return usage.remaining_tokens
    
    def can_fit_message(self, content: str) -> tuple[bool, int]:
        """
        Check if a message can fit in remaining context.
        
        Args:
            content: Message content
            
        Returns:
            Tuple of (can_fit, tokens_needed)
        """
        tokens = self._counter.count_tokens(content) + 4  # +4 for formatting
        usage = self.get_usage()
        return tokens <= usage.remaining_tokens, tokens
    
    def clear(self) -> None:
        """Clear all messages (keep system prompt)."""
        self._messages.clear()
        self._warning_fired = False
    
    def trim_oldest(self, keep_recent: int = 10) -> int:
        """
        Trim oldest messages to free space.
        
        Args:
            keep_recent: Number of recent messages to keep
            
        Returns:
            Number of messages removed
        """
        if len(self._messages) <= keep_recent:
            return 0
        
        removed_count = len(self._messages) - keep_recent
        self._messages = self._messages[-keep_recent:]
        self._warning_fired = False
        
        logger.info(f"Trimmed {removed_count} oldest messages")
        return removed_count
    
    def auto_trim(self, target_percentage: float = 70.0) -> int:
        """
        Automatically trim to reach target usage.
        
        Args:
            target_percentage: Target usage percentage
            
        Returns:
            Number of messages removed
        """
        removed = 0
        
        while self.get_usage().percentage > target_percentage and len(self._messages) > 2:
            self._messages.pop(0)
            removed += 1
        
        self._warning_fired = False
        return removed
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of context usage."""
        usage = self.get_usage()
        
        return {
            "tokens": {
                "used": usage.used_tokens,
                "max": usage.max_tokens,
                "remaining": usage.remaining_tokens,
                "system": usage.system_tokens,
                "user": usage.user_tokens,
                "assistant": usage.assistant_tokens
            },
            "percentage": usage.percentage,
            "level": usage.level.name,
            "message_count": usage.message_count,
            "progress_bar": self.get_progress_bar()
        }
    
    def estimate_tokens(self, text: str) -> TokenEstimate:
        """Get a detailed token estimate for text."""
        return self._counter.get_estimate(text)
    
    # =========================================================================
    # AUTO-CONTINUE METHODS
    # =========================================================================
    
    def set_auto_continue_threshold(self, percentage: float) -> None:
        """
        Set threshold for auto-continue.
        
        Args:
            percentage: Percentage (0-100) at which to trigger auto-continue
        """
        self.config.auto_continue_threshold = max(50.0, min(99.0, percentage))
    
    def should_auto_continue(self) -> bool:
        """
        Check if auto-continue should trigger.
        
        Returns:
            True if context usage exceeds auto-continue threshold
        """
        if not self.config.auto_continue_enabled:
            return False
        
        usage = self.get_usage()
        return usage.percentage >= self.config.auto_continue_threshold
    
    def get_messages_for_continue(self) -> list[dict[str, Any]]:
        """
        Get the last N messages to keep when auto-continuing.
        
        Returns:
            List of messages to preserve in new context
        """
        keep_n = self.config.auto_continue_keep_messages
        if len(self._messages) <= keep_n:
            return self._messages.copy()
        return self._messages[-keep_n:]
    
    def generate_conversation_summary(self) -> str:
        """
        Generate a summary of the conversation for smart continue.
        
        Returns:
            Summary string suitable for injecting as context
        """
        if not self._messages:
            return ""
        
        # Extract key points from conversation
        user_messages = [m for m in self._messages if m["role"] == "user"]
        assistant_messages = [m for m in self._messages if m["role"] == "assistant"]
        
        summary_parts = ["CONVERSATION SUMMARY:"]
        
        # First user topic
        if user_messages:
            first_msg = user_messages[0]["content"][:100]
            summary_parts.append(f"- Conversation started about: {first_msg}...")
        
        # Key topics discussed (simple extraction from user messages)
        if len(user_messages) > 1:
            topics = []
            for msg in user_messages[1:]:
                # Extract first sentence as topic indicator
                content = msg["content"]
                first_sent = content.split('.')[0][:50] if '.' in content else content[:50]
                if first_sent and first_sent not in topics:
                    topics.append(first_sent)
            
            if topics:
                summary_parts.append("- Topics discussed:")
                for topic in topics[:5]:  # Limit to 5 topics
                    summary_parts.append(f"  * {topic}")
        
        # Last exchange context
        if user_messages and assistant_messages:
            last_user = user_messages[-1]["content"][:100]
            summary_parts.append(f"- Most recent question: {last_user}...")
        
        summary_parts.append(f"- Total messages: {len(self._messages)}")
        
        return "\n".join(summary_parts)
    
    def prepare_auto_continue(self) -> tuple[str, list[dict[str, Any]]]:
        """
        Prepare context for auto-continuing in a new chat.
        
        Returns:
            Tuple of (summary_text, messages_to_keep)
        """
        summary = ""
        if self.config.auto_continue_include_summary:
            summary = self.generate_conversation_summary()
        
        messages = self.get_messages_for_continue()
        
        return summary, messages


# Context window sizes for common models
MODEL_CONTEXT_SIZES = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    
    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-2": 100000,
    
    # Open source
    "llama-2-7b": 4096,
    "llama-2-13b": 4096,
    "llama-2-70b": 4096,
    "mistral-7b": 8192,
    "mixtral-8x7b": 32768,
    "phi-2": 2048,
    
    # Enigma AI Engine
    "forge-nano": 512,
    "forge-micro": 1024,
    "forge-tiny": 2048,
    "forge-small": 4096,
    "forge-medium": 8192,
    "forge-large": 16384,
    "forge-xl": 32768,
}


def get_context_size(model_name: str) -> int:
    """
    Get context size for a model.
    
    Args:
        model_name: Model name or identifier
        
    Returns:
        Context size in tokens (defaults to 4096)
    """
    model_lower = model_name.lower()
    
    for key, size in MODEL_CONTEXT_SIZES.items():
        if key in model_lower:
            return size
    
    return 4096  # Default


# Singleton instance
_context_tracker_instance: ContextTracker | None = None


def get_context_tracker(
    max_tokens: int | None = None,
    model_name: str | None = None
) -> ContextTracker:
    """
    Get or create the singleton context tracker.
    
    Args:
        max_tokens: Override max tokens
        model_name: Model name to auto-detect context size
    """
    global _context_tracker_instance
    
    if _context_tracker_instance is None:
        if model_name and not max_tokens:
            max_tokens = get_context_size(model_name)
        
        config = ContextConfig(max_tokens=max_tokens or 4096)
        _context_tracker_instance = ContextTracker(config)
    
    return _context_tracker_instance


def reset_context_tracker() -> None:
    """Reset the singleton instance."""
    global _context_tracker_instance
    _context_tracker_instance = None


# Convenience functions
def count_tokens(text: str) -> int:
    """Quick token count."""
    return TokenCounter().count_tokens(text)


def estimate_tokens(text: str) -> int:
    """Estimate tokens without full tokenizer."""
    char_count = len(text)
    word_count = len(text.split())
    return int((char_count / 4 + word_count * 1.3) / 2)


def get_usage_bar(
    used: int,
    total: int,
    width: int = 30
) -> str:
    """Get a quick usage bar."""
    percentage = used / total * 100 if total > 0 else 0
    filled = int(width * percentage / 100)
    bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {percentage:.0f}%"
