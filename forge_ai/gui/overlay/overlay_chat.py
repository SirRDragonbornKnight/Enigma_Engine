"""
Overlay Chat Bridge - Integration between overlay and main chat system.

Handles message passing and synchronization between the overlay
and the main ForgeAI chat interface.
"""

import logging
from typing import Optional, Callable, List, Dict, Any
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class OverlayChatBridge(QObject):
    """
    Bridge between overlay and main chat system.
    
    Signals:
        message_sent: Emitted when user sends a message from overlay
        response_received: Emitted when AI responds
        history_updated: Emitted when chat history changes
    """
    
    message_sent = pyqtSignal(str)  # User message
    response_received = pyqtSignal(str)  # AI response
    history_updated = pyqtSignal(list)  # Full chat history
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._chat_history: List[Dict[str, Any]] = []
        self._engine = None
        self._on_response_callback: Optional[Callable[[str], None]] = None
        
    def set_engine(self, engine):
        """Set the AI engine for generating responses."""
        self._engine = engine
        logger.info("Chat bridge engine set")
        
    def send_message(self, text: str):
        """
        Send message from overlay to AI.
        
        Args:
            text: User message text
        """
        if not text or not text.strip():
            return
            
        # Add to history
        self._chat_history.append({
            "role": "user",
            "content": text.strip()
        })
        
        # Emit signal
        self.message_sent.emit(text.strip())
        
        # Generate response if engine available
        if self._engine:
            try:
                response = self._engine.generate(
                    text.strip(),
                    max_gen=150,
                    temperature=0.8
                )
                self.receive_response(response)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                self.receive_response(f"Error: {e}")
        else:
            logger.warning("No engine available for response generation")
            
    def receive_response(self, response: str):
        """
        Receive and display AI response in overlay.
        
        Args:
            response: AI response text
        """
        if not response:
            return
            
        # Add to history
        self._chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Emit signals
        self.response_received.emit(response)
        self.history_updated.emit(self._chat_history.copy())
        
        # Call callback if set
        if self._on_response_callback:
            self._on_response_callback(response)
            
    def sync_history(self, history: Optional[List[Dict[str, Any]]] = None):
        """
        Sync chat history between overlay and main window.
        
        Args:
            history: Chat history to sync (if None, uses current history)
        """
        if history is not None:
            self._chat_history = history.copy()
        self.history_updated.emit(self._chat_history.copy())
        
    def get_history(self) -> List[Dict[str, Any]]:
        """Get current chat history."""
        return self._chat_history.copy()
        
    def clear_history(self):
        """Clear chat history."""
        self._chat_history.clear()
        self.history_updated.emit([])
        
    def set_response_callback(self, callback: Callable[[str], None]):
        """
        Set callback for when response is received.
        
        Args:
            callback: Function to call with response text
        """
        self._on_response_callback = callback
        
    def get_last_response(self) -> Optional[str]:
        """Get the last AI response from history."""
        for msg in reversed(self._chat_history):
            if msg.get("role") == "assistant":
                return msg.get("content")
        return None
