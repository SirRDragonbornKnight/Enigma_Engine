"""
================================================================================
EVENT BUS - Pub/Sub Event System for Module Communication
================================================================================

A centralized event bus that enables loose coupling between Enigma AI Engine modules.
Modules can publish events and subscribe to events without knowing about each other.

Features:
- Pub/sub pattern for decoupled communication
- Typed events with dataclass support
- Async event handling option
- Priority-based subscription
- Event filtering by pattern
- Event history for debugging
- Thread-safe operations

FILE: enigma_engine/utils/events.py
TYPE: System Utility
MAIN CLASSES: EventBus, Event

USAGE:
    from enigma_engine.utils.events import get_event_bus, Event, on_event
    
    # Define custom events
    @dataclass
    class ModelLoadedEvent(Event):
        model_name: str
        load_time: float
    
    # Subscribe to events
    @on_event("model.loaded")
    def handle_model_loaded(event):
        print(f"Model {event.model_name} loaded in {event.load_time}s")
    
    # Publish events
    bus = get_event_bus()
    bus.publish("model.loaded", ModelLoadedEvent("gpt-2", 1.5))
    
    # Pattern matching
    @on_event("model.*")  # Matches model.loaded, model.unloaded, etc.
    def handle_any_model_event(event):
        print(f"Model event: {event}")
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Base class for all events.
    
    Custom events should subclass this:
    
        @dataclass
        class UserMessageEvent(Event):
            message: str
            user_id: str
    """
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    
    @property
    def event_type(self) -> str:
        """Get the event type name."""
        return self.__class__.__name__


@dataclass
class Subscription:
    """A subscription to an event pattern."""
    
    callback: Callable[[Event], None]
    pattern: str
    priority: int = 50  # Higher = called first
    once: bool = False  # If True, unsubscribe after first call
    async_handler: bool = False  # If True, run in thread pool
    
    def __lt__(self, other):
        return self.priority > other.priority


class EventBus:
    """
    Central event bus for publish/subscribe communication.
    
    Enables loose coupling between modules by allowing them to
    communicate through events without direct dependencies.
    """
    
    _instance: EventBus | None = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the event bus."""
        if EventBus._initialized:
            return
        EventBus._initialized = True
        
        # Subscriptions by exact topic
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        
        # Pattern subscriptions (wildcards)
        self._pattern_subscriptions: list[Subscription] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event history for debugging
        self._history: list[dict[str, Any]] = []
        self._history_size = 100
        self._record_history = False
        
        # Async event loop for async handlers
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        
        logger.debug("EventBus initialized")
    
    def subscribe(
        self,
        pattern: str,
        callback: Callable[[Event], None],
        priority: int = 50,
        once: bool = False,
        async_handler: bool = False
    ) -> Callable[[], None]:
        """
        Subscribe to events matching a pattern.
        
        Args:
            pattern: Event type to match (supports wildcards: "model.*")
            callback: Function to call when event is published
            priority: Higher priority handlers are called first
            once: If True, automatically unsubscribe after first call
            async_handler: If True, run callback in a thread pool
            
        Returns:
            Unsubscribe function
        """
        sub = Subscription(
            callback=callback,
            pattern=pattern,
            priority=priority,
            once=once,
            async_handler=async_handler
        )
        
        with self._lock:
            if '*' in pattern or '?' in pattern:
                # Pattern subscription
                self._pattern_subscriptions.append(sub)
                self._pattern_subscriptions.sort()
            else:
                # Exact subscription
                self._subscriptions[pattern].append(sub)
                self._subscriptions[pattern].sort()
        
        logger.debug(f"Subscribed to '{pattern}' (priority={priority})")
        
        # Return unsubscribe function
        def unsubscribe():
            self._unsubscribe(sub)
        
        return unsubscribe
    
    def _unsubscribe(self, sub: Subscription):
        """Remove a subscription."""
        with self._lock:
            if '*' in sub.pattern or '?' in sub.pattern:
                try:
                    self._pattern_subscriptions.remove(sub)
                except ValueError:
                    pass  # Intentionally silent
            else:
                try:
                    self._subscriptions[sub.pattern].remove(sub)
                except (KeyError, ValueError):
                    pass  # Intentionally silent
    
    def publish(
        self,
        event_type: str,
        event: Event | dict[str, Any] | None = None,
        **kwargs
    ) -> int:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: The event type string (e.g., "model.loaded")
            event: Event object, dict, or None
            **kwargs: Additional data if event is None
            
        Returns:
            Number of handlers that were called
        """
        # Create event if needed
        if event is None:
            event = Event(source=kwargs.pop('source', ''))
            for k, v in kwargs.items():
                setattr(event, k, v)
        elif isinstance(event, dict):
            e = Event()
            for k, v in event.items():
                setattr(e, k, v)
            event = e
        
        # Set event type attribute
        event._event_type = event_type
        
        # Record history
        if self._record_history:
            self._add_to_history(event_type, event)
        
        # Find matching subscriptions
        handlers_called = 0
        to_remove = []
        
        with self._lock:
            # Exact matches
            subs = list(self._subscriptions.get(event_type, []))
            
            # Pattern matches
            for sub in self._pattern_subscriptions:
                if fnmatch.fnmatch(event_type, sub.pattern):
                    subs.append(sub)
            
            # Sort by priority
            subs.sort()
        
        # Call handlers
        for sub in subs:
            try:
                if sub.async_handler:
                    self._call_async(sub.callback, event)
                else:
                    sub.callback(event)
                
                handlers_called += 1
                
                if sub.once:
                    to_remove.append(sub)
                    
            except Exception as e:
                logger.error(f"Event handler error for '{event_type}': {e}")
        
        # Remove once handlers
        for sub in to_remove:
            self._unsubscribe(sub)
        
        if handlers_called > 0:
            logger.debug(f"Published '{event_type}' to {handlers_called} handlers")
        
        return handlers_called
    
    def _call_async(self, callback: Callable[[Event], None], event: Event):
        """Call a handler asynchronously."""
        thread = threading.Thread(
            target=callback,
            args=(event,),
            daemon=True,
            name=f"EventHandler-{callback.__name__}"
        )
        thread.start()
    
    def _add_to_history(self, event_type: str, event: Event):
        """Add event to history."""
        with self._lock:
            self._history.append({
                'type': event_type,
                'timestamp': event.timestamp,
                'source': event.source,
                'event': event
            })
            
            # Trim history
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]
    
    def enable_history(self, enabled: bool = True, size: int = 100):
        """Enable/disable event history recording."""
        self._record_history = enabled
        self._history_size = size
    
    def get_history(
        self,
        event_type: str = None,
        since: float = None,
        limit: int = None
    ) -> list[dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (supports wildcards)
            since: Only events after this timestamp
            limit: Max events to return
            
        Returns:
            List of event history entries
        """
        with self._lock:
            history = self._history.copy()
        
        if event_type:
            history = [
                h for h in history
                if fnmatch.fnmatch(h['type'], event_type)
            ]
        
        if since:
            history = [h for h in history if h['timestamp'] >= since]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self):
        """Clear event history."""
        with self._lock:
            self._history.clear()
    
    def has_subscribers(self, event_type: str) -> bool:
        """Check if an event type has any subscribers."""
        with self._lock:
            if self._subscriptions.get(event_type):
                return True
            
            for sub in self._pattern_subscriptions:
                if fnmatch.fnmatch(event_type, sub.pattern):
                    return True
        
        return False
    
    def list_subscriptions(self) -> dict[str, int]:
        """List all subscribed event patterns and their handler counts."""
        with self._lock:
            result = {}
            
            for pattern, subs in self._subscriptions.items():
                result[pattern] = len(subs)
            
            for sub in self._pattern_subscriptions:
                result[sub.pattern] = result.get(sub.pattern, 0) + 1
            
            return result
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing only)."""
        cls._instance = None
        cls._initialized = False


# =============================================================================
# Global access and decorators
# =============================================================================

_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def publish(event_type: str, event: Event | dict | None = None, **kwargs) -> int:
    """Convenience function to publish an event."""
    return get_event_bus().publish(event_type, event, **kwargs)


def subscribe(
    pattern: str,
    callback: Callable[[Event], None],
    priority: int = 50,
    once: bool = False
) -> Callable[[], None]:
    """Convenience function to subscribe to events."""
    return get_event_bus().subscribe(pattern, callback, priority, once)


def on_event(
    pattern: str,
    priority: int = 50,
    once: bool = False,
    async_handler: bool = False
):
    """
    Decorator to subscribe a function to an event pattern.
    
    Usage:
        @on_event("model.loaded")
        def handle_model_loaded(event):
            print(f"Model loaded: {event}")
        
        @on_event("chat.*", priority=100)
        def handle_all_chat_events(event):
            log_chat_event(event)
    """
    def decorator(func: Callable[[Event], None]) -> Callable[[Event], None]:
        get_event_bus().subscribe(
            pattern=pattern,
            callback=func,
            priority=priority,
            once=once,
            async_handler=async_handler
        )
        return func
    return decorator


# =============================================================================
# Built-in event types
# =============================================================================

@dataclass
class ModuleEvent(Event):
    """Event related to module lifecycle."""
    module_id: str = ""
    action: str = ""  # "loaded", "unloaded", "error"


@dataclass
class ModelEvent(Event):
    """Event related to AI models."""
    model_name: str = ""
    action: str = ""  # "loaded", "unloaded", "generated"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatEvent(Event):
    """Event related to chat/conversation."""
    message: str = ""
    role: str = ""  # "user", "assistant", "system"
    conversation_id: str = ""


@dataclass
class VoiceEvent(Event):
    """Event related to voice input/output."""
    action: str = ""  # "speech_recognized", "speech_started", "wake_word"
    text: str = ""


@dataclass
class SystemEvent(Event):
    """System-level events."""
    action: str = ""  # "startup", "shutdown", "error"
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Integration with Enigma AI Engine components
# =============================================================================

def _register_builtin_publishers():
    """Hook into Enigma AI Engine components to auto-publish events."""
    bus = get_event_bus()
    
    # These will be called by the respective components when they import this module
    # The components should call publish() directly
    
    logger.debug("Event bus ready for Enigma AI Engine component integration")


_register_builtin_publishers()


__all__ = [
    'Event',
    'EventBus',
    'Subscription',
    'get_event_bus',
    'publish',
    'subscribe',
    'on_event',
    # Built-in event types
    'ModuleEvent',
    'ModelEvent',
    'ChatEvent',
    'VoiceEvent',
    'SystemEvent',
]
