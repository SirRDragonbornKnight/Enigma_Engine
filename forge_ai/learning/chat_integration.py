"""
================================================================================
LEARNING CHAT INTEGRATION - Connect Learning to Chat System
================================================================================

Integrates learning detection with the chat system for seamless 
real-time learning from conversations.

📍 FILE: forge_ai/learning/chat_integration.py
🏷️ TYPE: Integration Layer
🎯 MAIN CLASS: LearningChatIntegration

HOOKS INTO:
    → LearningEngine (self_improvement.py) - For learning queue
    → AutonomousLearner (autonomous.py) - For background processing
    → ConversationDetector (conversation_detector.py) - For detection

USAGE:
    integration = LearningChatIntegration(model)
    
    # In chat loop:
    user_msg = get_user_input()
    learning = integration.before_response(user_msg)
    
    ai_response = model.generate(user_msg)
    integration.after_response(ai_response)
    
    # Learning happens automatically in background
================================================================================
"""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime

from .conversation_detector import ConversationDetector, DetectedLearning

# Import from existing systems
from ..core.self_improvement import (
    LearningEngine,
    LearningExample,
    LearningSource,
    Priority,
    get_learning_engine,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LEARNING CHAT INTEGRATION
# =============================================================================

class LearningChatIntegration:
    """
    Integrates learning detection with the chat system.
    
    This class bridges the gap between real-time conversation and
    the ForgeAI learning system. It:
    
    1. Detects learning opportunities from user messages
    2. Feeds them to the existing LearningEngine
    3. Optionally triggers AutonomousLearner for background processing
    4. Tracks learning statistics
    
    Thread-safe for GUI usage.
    
    Hooks into existing systems:
    - LearningEngine (self_improvement.py) for queue management
    - AutonomousLearner (autonomous.py) for background processing
    
    Example:
        integration = LearningChatIntegration(model)
        
        # In chat loop:
        user_msg = get_user_input()
        learning = integration.before_response(user_msg)
        if learning:
            show_learning_indicator(learning.type)
        
        ai_response = model.generate(user_msg)
        integration.after_response(ai_response)
        
        # Learning happens automatically in background
    """
    
    def __init__(
        self,
        model: 'Forge',  # Forward reference to avoid circular import
        model_name: Optional[str] = None,
        learning_engine: Optional[LearningEngine] = None,
        autonomous_learner: Optional['AutonomousMode'] = None,
        auto_learn: bool = True,
        on_learning_detected: Optional[Callable[[DetectedLearning], None]] = None,
    ):
        """
        Initialize the learning chat integration.
        
        Args:
            model: The Forge model (used to identify learning engine)
            model_name: Name of the model (for learning engine lookup)
            learning_engine: Optional existing LearningEngine to use
            autonomous_learner: Optional existing AutonomousLearner to use
            auto_learn: If True, automatically add detected learning to queue
            on_learning_detected: Callback when learning is detected
        """
        self.model = model
        self.model_name = model_name or getattr(model, 'name', 'forge')
        self.auto_learn = auto_learn
        self.on_learning_detected = on_learning_detected
        
        # Get or create learning engine
        if learning_engine:
            self.learning_engine = learning_engine
        else:
            self.learning_engine = get_learning_engine(self.model_name)
        
        # Autonomous learner (optional - for background processing)
        self.autonomous_learner = autonomous_learner
        self._init_autonomous_learner()
        
        # Conversation detector
        self.detector = ConversationDetector()
        
        # Statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            'total_messages': 0,
            'total_detections': 0,
            'corrections_detected': 0,
            'teachings_detected': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'examples_queued': 0,
            'session_start': datetime.now(),
        }
        
        # Conversation tracking
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history = 50  # Keep last 50 messages
        
        logger.info(f"LearningChatIntegration initialized for {self.model_name}")
        logger.info(f"  Auto-learn: {self.auto_learn}")
        logger.info(f"  Learning engine queue: {len(self.learning_engine.learning_queue)} items")
    
    def _init_autonomous_learner(self):
        """Initialize or connect to autonomous learner if available."""
        if self.autonomous_learner is not None:
            return
        
        # Try to import and create autonomous learner
        try:
            from ..core.autonomous import AutonomousMode
            self.autonomous_learner = AutonomousMode(self.model_name)
            logger.info("AutonomousLearner connected")
        except ImportError:
            logger.debug("AutonomousLearner not available")
        except Exception as e:
            logger.warning(f"Could not initialize AutonomousLearner: {e}")
    
    def before_response(self, user_message: str) -> Optional[DetectedLearning]:
        """
        Call before generating AI response.
        
        Detects corrections/teaching from user message and optionally
        adds them to the learning queue.
        
        Args:
            user_message: The user's message
        
        Returns:
            DetectedLearning if found, None otherwise
        """
        if not user_message or not user_message.strip():
            return None
        
        # Update stats
        with self._stats_lock:
            self._stats['total_messages'] += 1
        
        # Add to history
        self._add_to_history('user', user_message)
        
        # Detect learning opportunity
        detected = self.detector.on_user_message(user_message)
        
        if detected:
            with self._stats_lock:
                self._stats['total_detections'] += 1
                
                # Update type-specific stats
                if detected.type == 'correction':
                    self._stats['corrections_detected'] += 1
                elif detected.type == 'teaching':
                    self._stats['teachings_detected'] += 1
                elif detected.type == 'positive_feedback':
                    self._stats['positive_feedback'] += 1
                elif detected.type == 'negative_feedback':
                    self._stats['negative_feedback'] += 1
            
            # Notify callback
            if self.on_learning_detected:
                try:
                    self.on_learning_detected(detected)
                except Exception as e:
                    logger.warning(f"Learning callback error: {e}")
            
            # Auto-learn if enabled
            if self.auto_learn:
                self._queue_learning(detected)
            
            logger.debug(f"Detected {detected.type}: confidence={detected.confidence:.2f}")
        
        return detected
    
    def after_response(self, ai_response: str) -> None:
        """
        Call after generating AI response.
        
        Tracks response for future correction matching.
        
        Args:
            ai_response: The AI's generated response
        """
        if not ai_response or not ai_response.strip():
            return
        
        # Track in detector
        self.detector.on_ai_response(ai_response)
        
        # Add to history
        self._add_to_history('assistant', ai_response)
    
    def on_conversation_end(self) -> None:
        """
        Call when conversation ends.
        
        Triggers reflection on the conversation if autonomous learner
        is available.
        """
        # Reset detector context for next conversation
        self.detector.reset_context()
        
        # Trigger autonomous reflection if available
        if self.autonomous_learner and len(self._conversation_history) > 2:
            try:
                # Create a reflection task
                self._trigger_reflection()
            except Exception as e:
                logger.warning(f"Could not trigger reflection: {e}")
        
        # Clear conversation history
        self._conversation_history = []
        
        logger.debug("Conversation ended, context reset")
    
    def get_learning_stats(self) -> dict:
        """
        Get statistics about learning.
        
        Returns:
            Dictionary with learning statistics
        """
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Add queue info from learning engine
        queue_stats = self.learning_engine.get_queue_stats()
        stats['queue_size'] = queue_stats['total_examples']
        stats['queue_by_priority'] = dict(queue_stats.get('by_priority', {}))
        stats['queue_avg_quality'] = queue_stats.get('avg_quality', 0.0)
        
        # Calculate session duration
        session_duration = datetime.now() - stats['session_start']
        stats['session_duration_minutes'] = session_duration.total_seconds() / 60
        
        # Detection rate
        if stats['total_messages'] > 0:
            stats['detection_rate'] = stats['total_detections'] / stats['total_messages']
        else:
            stats['detection_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset learning statistics."""
        with self._stats_lock:
            self._stats = {
                'total_messages': 0,
                'total_detections': 0,
                'corrections_detected': 0,
                'teachings_detected': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'examples_queued': 0,
                'session_start': datetime.now(),
            }
    
    def set_auto_learn(self, enabled: bool) -> None:
        """Enable or disable automatic learning."""
        self.auto_learn = enabled
        logger.info(f"Auto-learn {'enabled' if enabled else 'disabled'}")
    
    def manually_add_learning(
        self,
        input_text: str,
        output_text: str,
        source: str = "conversation",
        priority: str = "medium",
        quality: float = 0.8
    ) -> None:
        """
        Manually add a learning example.
        
        Useful for explicit teaching or importing examples.
        
        Args:
            input_text: The input/prompt
            output_text: The desired output
            source: Source type ("conversation", "correction", "teaching")
            priority: Priority level ("critical", "high", "medium", "low")
            quality: Quality score 0.0-1.0
        """
        # Map string to enum
        source_map = {
            'conversation': LearningSource.CONVERSATION,
            'correction': LearningSource.CORRECTION,
            'teaching': LearningSource.CONVERSATION,
            'practice': LearningSource.PRACTICE,
            'reflection': LearningSource.REFLECTION,
            'research': LearningSource.RESEARCH,
        }
        
        priority_map = {
            'critical': Priority.CRITICAL,
            'high': Priority.HIGH,
            'medium': Priority.MEDIUM,
            'low': Priority.LOW,
            'background': Priority.BACKGROUND,
        }
        
        example = LearningExample(
            input_text=input_text,
            output_text=output_text,
            source=source_map.get(source.lower(), LearningSource.CONVERSATION),
            priority=priority_map.get(priority.lower(), Priority.MEDIUM),
            quality_score=quality,
            metadata={'manual': True},
        )
        
        self.learning_engine.add_learning_example(example)
        
        with self._stats_lock:
            self._stats['examples_queued'] += 1
        
        logger.info(f"Manually added learning example: {input_text[:50]}...")
    
    def get_recent_detections(self, limit: int = 10) -> List[dict]:
        """
        Get recent detection history.
        
        Args:
            limit: Maximum number of detections to return
        
        Returns:
            List of detection dictionaries
        """
        # This would require storing detection history
        # For now, return detector stats
        return [{'stats': self.detector.get_stats()}]
    
    def is_autonomous_running(self) -> bool:
        """Check if autonomous learning is running."""
        if self.autonomous_learner is None:
            return False
        return getattr(self.autonomous_learner.config, 'enabled', False)
    
    def start_autonomous_learning(self) -> bool:
        """
        Start autonomous background learning.
        
        Returns:
            True if started successfully
        """
        if self.autonomous_learner is None:
            logger.warning("AutonomousLearner not available")
            return False
        
        try:
            self.autonomous_learner.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start autonomous learning: {e}")
            return False
    
    def stop_autonomous_learning(self) -> None:
        """Stop autonomous background learning."""
        if self.autonomous_learner:
            try:
                self.autonomous_learner.stop()
            except Exception as e:
                logger.warning(f"Error stopping autonomous learning: {e}")
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _queue_learning(self, detected: DetectedLearning) -> None:
        """Add detected learning to the queue."""
        try:
            example = self.detector.to_learning_example(detected)
            self.learning_engine.add_learning_example(example)
            
            with self._stats_lock:
                self._stats['examples_queued'] += 1
            
            logger.debug(f"Queued learning: {detected.type}")
            
        except Exception as e:
            logger.error(f"Failed to queue learning: {e}")
    
    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self._conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': time.time(),
        })
        
        # Trim if too long
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]
    
    def _trigger_reflection(self) -> None:
        """Trigger autonomous reflection on the conversation."""
        if not self.autonomous_learner:
            return
        
        # Format conversation for reflection
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self._conversation_history
        ])
        
        # Add to learning engine for reflection
        # The autonomous learner will pick this up in its next cycle
        reflection_example = LearningExample(
            input_text=f"[CONVERSATION TO REFLECT ON]\n{conversation_text[:2000]}",
            output_text="[PENDING REFLECTION]",
            source=LearningSource.REFLECTION,
            priority=Priority.LOW,
            quality_score=0.5,
            metadata={
                'needs_reflection': True,
                'conversation_length': len(self._conversation_history),
            },
        )
        
        self.learning_engine.add_learning_example(reflection_example)
        logger.debug("Queued conversation for reflection")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_chat_integration(
    model: 'Forge',
    model_name: Optional[str] = None,
    auto_learn: bool = True,
    on_learning: Optional[Callable[[DetectedLearning], None]] = None,
) -> LearningChatIntegration:
    """
    Create a learning chat integration for a model.
    
    Args:
        model: The Forge model
        model_name: Optional model name
        auto_learn: Whether to automatically queue learning
        on_learning: Callback when learning is detected
    
    Returns:
        Configured LearningChatIntegration
    
    Example:
        integration = create_chat_integration(
            model,
            on_learning=lambda d: print(f"Learning: {d.type}")
        )
    """
    return LearningChatIntegration(
        model=model,
        model_name=model_name,
        auto_learn=auto_learn,
        on_learning_detected=on_learning,
    )


# =============================================================================
# CHAT WRAPPER - For easy drop-in integration
# =============================================================================

class LearningChatWrapper:
    """
    Wrapper that adds learning to any chat function.
    
    This provides a drop-in way to add learning to existing chat code.
    
    Example:
        # Original code:
        response = generate_response(user_input)
        
        # With learning:
        wrapper = LearningChatWrapper(model)
        response = wrapper.chat(user_input, generate_response)
    """
    
    def __init__(
        self,
        model: 'Forge',
        model_name: Optional[str] = None,
        auto_learn: bool = True,
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: The Forge model
            model_name: Optional model name
            auto_learn: Whether to auto-queue learning
        """
        self.integration = LearningChatIntegration(
            model=model,
            model_name=model_name,
            auto_learn=auto_learn,
        )
        self._last_learning: Optional[DetectedLearning] = None
    
    def chat(
        self,
        user_message: str,
        generate_fn: Callable[[str], str],
    ) -> str:
        """
        Process a chat turn with learning.
        
        Args:
            user_message: The user's message
            generate_fn: Function that generates AI response
        
        Returns:
            The AI response
        """
        # Before: detect learning
        self._last_learning = self.integration.before_response(user_message)
        
        # Generate response
        response = generate_fn(user_message)
        
        # After: track response
        self.integration.after_response(response)
        
        return response
    
    def get_last_learning(self) -> Optional[DetectedLearning]:
        """Get the learning detected in the last chat turn."""
        return self._last_learning
    
    def end_conversation(self) -> None:
        """Signal end of conversation."""
        self.integration.on_conversation_end()
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        return self.integration.get_learning_stats()
