# type: ignore
"""
Chat Synchronization Manager - Keeps main chat and quick chat in sync.

Both chat interfaces share the same conversation history and display.
Messages sent from either chat appear in both immediately.
"""

import time
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

try:
    from PyQt5.QtCore import QObject, pyqtSignal, QMetaObject, Qt, Q_ARG
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None

from ..config import CONFIG


# Maximum messages to keep in memory (prevents unbounded growth)
MAX_MESSAGES = 200
MAX_RESPONSE_HISTORY = 50


class ChatMessage:
    """A single chat message."""
    
    __slots__ = ('role', 'text', 'source', 'timestamp')  # Memory optimization
    
    def __init__(self, role: str, text: str, source: str = "main", timestamp: float = None):
        self.role = role  # "user" or "assistant"
        self.text = text
        self.source = source  # "main" or "quick"
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "source": self.source,
            "ts": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data.get("role", "user"),
            text=data.get("text", ""),
            source=data.get("source", "main"),
            timestamp=data.get("ts", time.time())
        )


class ChatSync(QObject if HAS_PYQT else object):
    """
    Singleton manager for synchronized chat between main window and quick chat.
    
    Usage:
        sync = ChatSync.instance()
        sync.add_user_message("Hello", source="main")
        sync.add_ai_message("Hi there!", source="main")
    """
    
    _instance = None
    
    # Signals for UI updates (only defined if PyQt available)
    if HAS_PYQT:
        message_added = pyqtSignal(dict)  # Emitted when any message is added
        chat_cleared = pyqtSignal()  # Emitted when chat is cleared
        model_changed = pyqtSignal(str)  # Emitted when model changes
        generation_started = pyqtSignal(str)  # Emitted when AI starts thinking (user text)
        generation_finished = pyqtSignal(str)  # Emitted when AI finishes (response text)
        generation_stopped = pyqtSignal()  # Emitted when generation is stopped
    
    @classmethod
    def instance(cls) -> "ChatSync":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton (useful for testing or re-initialization)."""
        cls._instance = None
    
    def __init__(self):
        if ChatSync._instance is not None:
            raise RuntimeError("Use ChatSync.instance() to get the singleton")
        
        # Only call QObject.__init__ if we have PyQt and QApplication exists
        if HAS_PYQT:
            from PyQt5.QtWidgets import QApplication
            if QApplication.instance() is not None:
                super().__init__()
            else:
                # QApplication doesn't exist yet - we're being imported too early
                # Just initialize as a regular object, signals won't work
                pass
        
        self._messages: List[ChatMessage] = []
        self._model_name = "No model"
        self._user_display_name = "You"
        self._engine = None
        self._main_window = None
        self._quick_chat = None
        self._is_generating = False
        self._stop_requested = False
        
        # Response tracking for feedback
        self._response_history: Dict[int, Dict] = {}
    
    def set_main_window(self, window):
        """Connect the main window."""
        self._main_window = window
        # Sync existing messages to main window
        self._sync_to_main()
    
    def set_quick_chat(self, quick_chat):
        """Connect the quick chat widget."""
        self._quick_chat = quick_chat
        # Sync existing messages to quick chat
        self._sync_to_quick()
    
    def set_engine(self, engine):
        """Set the AI engine for generation."""
        self._engine = engine
    
    def set_model_name(self, name: str):
        """Update the model name."""
        self._model_name = name
        self.model_changed.emit(name)
    
    def set_user_name(self, name: str):
        """Update the user display name."""
        self._user_display_name = name
    
    @property
    def messages(self) -> List[ChatMessage]:
        """Get all messages."""
        return self._messages.copy()
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def user_name(self) -> str:
        return self._user_display_name
    
    @property
    def is_generating(self) -> bool:
        return self._is_generating
    
    def add_user_message(self, text: str, source: str = "main"):
        """Add a user message from either chat interface."""
        msg = ChatMessage(role="user", text=text, source=source)
        self._messages.append(msg)
        
        # Trim old messages to prevent memory growth
        if len(self._messages) > MAX_MESSAGES:
            self._messages = self._messages[-MAX_MESSAGES:]
        
        msg_dict = msg.to_dict()
        msg_dict["user_name"] = self._user_display_name
        
        self.message_added.emit(msg_dict)
        
        # Update both UIs
        self._update_user_message_ui(msg, source)
    
    def add_ai_message(self, text: str, source: str = "main"):
        """Add an AI response from either chat interface."""
        msg = ChatMessage(role="assistant", text=text, source=source)
        self._messages.append(msg)
        
        msg_dict = msg.to_dict()
        msg_dict["model_name"] = self._model_name
        
        self.message_added.emit(msg_dict)
        
        # Update both UIs
        self._update_ai_message_ui(msg, source)
    
    def clear_chat(self):
        """Clear all messages."""
        self._messages.clear()
        self._response_history.clear()
        self.chat_cleared.emit()
        
        # Clear both UIs
        if self._main_window and hasattr(self._main_window, 'chat_display'):
            try:
                QMetaObject.invokeMethod(
                    self._main_window.chat_display, "clear",
                    Qt.QueuedConnection
                )
                self._main_window.chat_messages = []
            except:
                pass
        
        if self._quick_chat and hasattr(self._quick_chat, 'response_area'):
            try:
                QMetaObject.invokeMethod(
                    self._quick_chat.response_area, "clear",
                    Qt.QueuedConnection
                )
            except:
                pass
    
    def generate_response(self, user_text: str, source: str = "main"):
        """Generate an AI response in background, update both chats."""
        if self._is_generating:
            return
        
        self._is_generating = True
        self._stop_requested = False
        
        # Add user message
        self.add_user_message(user_text, source)
        
        # Signal generation started
        self.generation_started.emit(user_text)
        
        # Run generation in background thread
        import threading
        thread = threading.Thread(
            target=self._do_generate,
            args=(user_text, source),
            daemon=True
        )
        thread.start()
    
    def stop_generation(self):
        """Stop current generation."""
        self._stop_requested = True
        self._is_generating = False
        self.generation_stopped.emit()
    
    def _do_generate(self, user_text: str, source: str):
        """Background generation."""
        try:
            response = None
            
            if self._stop_requested:
                return
            
            # Try to use the engine
            if self._engine:
                response = self._engine.generate(
                    user_text,
                    max_gen=200,
                    temperature=0.8
                )
            elif self._main_window and hasattr(self._main_window, 'engine') and self._main_window.engine:
                response = self._main_window.engine.generate(
                    user_text,
                    max_gen=200,
                    temperature=0.8
                )
            else:
                response = "⚠️ No AI model loaded. Open the full GUI to load a model."
            
            if self._stop_requested:
                return
            
            # Add AI response
            self.add_ai_message(response, source)
            
            # Signal done
            self._is_generating = False
            self.generation_finished.emit(response)
            
        except Exception as e:
            self._is_generating = False
            error_msg = f"<span style='color: #e74c3c;'>Error: {e}</span>"
            self.add_ai_message(error_msg, source)
            self.generation_finished.emit(error_msg)
    
    def _update_user_message_ui(self, msg: ChatMessage, source: str):
        """Update both chat UIs with a user message."""
        user_name = self._user_display_name
        html = (
            f'<div style="background-color: #313244; padding: 8px; margin: 4px 0; '
            f'border-radius: 8px; border-left: 3px solid #89b4fa;">'
            f'<b style="color: #89b4fa;">{user_name}:</b> {msg.text}</div>'
        )
        
        # Update main window (always, even if message came from quick chat)
        if self._main_window and hasattr(self._main_window, 'chat_display'):
            try:
                QMetaObject.invokeMethod(
                    self._main_window.chat_display, "append",
                    Qt.QueuedConnection, Q_ARG(str, html)
                )
                # Also update chat_messages list
                if hasattr(self._main_window, 'chat_messages'):
                    self._main_window.chat_messages.append(msg.to_dict())
            except:
                pass
        
        # Update quick chat (always, even if message came from main)
        if self._quick_chat and hasattr(self._quick_chat, 'response_area'):
            quick_html = (
                f"<div style='color: #9b59b6; margin: 4px 0;'>"
                f"<b>{user_name}:</b> {msg.text}</div>"
            )
            try:
                QMetaObject.invokeMethod(
                    self._quick_chat.response_area, "append",
                    Qt.QueuedConnection, Q_ARG(str, quick_html)
                )
            except:
                pass
    
    def _update_ai_message_ui(self, msg: ChatMessage, source: str):
        """Update both chat UIs with an AI message."""
        model_name = self._model_name
        
        # Main window HTML (with feedback links for local models)
        response_id = int(time.time() * 1000)
        self._response_history[response_id] = {
            'user_input': self._messages[-2].text if len(self._messages) >= 2 else "",
            'ai_response': msg.text,
            'timestamp': time.time()
        }
        
        # Trim old response history to prevent memory growth
        if len(self._response_history) > MAX_RESPONSE_HISTORY:
            oldest_keys = sorted(self._response_history.keys())[:-MAX_RESPONSE_HISTORY]
            for key in oldest_keys:
                del self._response_history[key]
        
        main_html = (
            f'<div style="background-color: #1e1e2e; padding: 8px; margin: 4px 0; '
            f'border-radius: 8px; border-left: 3px solid #a6e3a1;">'
            f'<b style="color: #a6e3a1;">{model_name}:</b> {msg.text}</div>'
        )
        
        # Update main window
        if self._main_window and hasattr(self._main_window, 'chat_display'):
            try:
                QMetaObject.invokeMethod(
                    self._main_window.chat_display, "append",
                    Qt.QueuedConnection, Q_ARG(str, main_html)
                )
                # Also update chat_messages list
                if hasattr(self._main_window, 'chat_messages'):
                    self._main_window.chat_messages.append(msg.to_dict())
            except:
                pass
        
        # Update quick chat
        if self._quick_chat and hasattr(self._quick_chat, 'response_area'):
            quick_html = (
                f"<div style='color: #3498db; margin-bottom: 8px;'>"
                f"<b>{model_name}:</b> {msg.text}</div>"
            )
            try:
                QMetaObject.invokeMethod(
                    self._quick_chat.response_area, "append",
                    Qt.QueuedConnection, Q_ARG(str, quick_html)
                )
            except:
                pass
    
    def _sync_to_main(self):
        """Sync all messages to main window."""
        if not self._main_window or not hasattr(self._main_window, 'chat_display'):
            return
        
        try:
            self._main_window.chat_display.clear()
            self._main_window.chat_messages = []
            
            for msg in self._messages:
                if msg.role == "user":
                    self._update_user_message_ui(msg, "sync")
                else:
                    self._update_ai_message_ui(msg, "sync")
        except:
            pass
    
    def _sync_to_quick(self):
        """Sync all messages to quick chat."""
        if not self._quick_chat or not hasattr(self._quick_chat, 'response_area'):
            return
        
        try:
            self._quick_chat.response_area.clear()
            
            for msg in self._messages:
                if msg.role == "user":
                    self._update_user_message_ui(msg, "sync")
                else:
                    self._update_ai_message_ui(msg, "sync")
        except:
            pass
    
    def get_history_for_context(self, max_messages: int = 6) -> List[Dict]:
        """Get recent messages for AI context."""
        recent = self._messages[-max_messages:] if len(self._messages) > max_messages else self._messages
        return [
            {"role": m.role, "content": m.text}
            for m in recent
        ]


def get_chat_sync() -> ChatSync:
    """Get the global ChatSync instance."""
    return ChatSync.instance()
