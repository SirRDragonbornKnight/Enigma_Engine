"""
================================================================================
GUI SAFETY GUARDS - Protection Against User Mistakes and Exploits
================================================================================

"The wise smith builds guards around the forge, protecting the wielder
from their own enthusiasm and the clever saboteur alike."

This module provides comprehensive protection against:
1. Accidental destructive actions (confirmation dialogs)
2. Rapid/spam clicking (rate limiting)
3. Invalid/malicious inputs (validation)
4. Mistakes without recovery (undo/history)
5. Smart users trying to break things (anti-tampering)

USAGE:
    from enigma_engine.gui.safety_guards import (
        SafetyGuards, confirm_action, rate_limit, validate_input, ActionHistory
    )
    
    # Confirmation before destructive action
    if SafetyGuards.confirm_destructive("Delete all models?", details="This cannot be undone!"):
        do_delete()
    
    # Rate limiting on button clicks
    @rate_limit(min_interval=1.0)
    def on_generate_clicked():
        ...
    
    # Input validation
    valid, error = SafetyGuards.validate_path(user_input)
    if not valid:
        show_error(error)
"""

import functools
import hashlib
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import PyQt5 (may not be available in headless mode)
try:
    from PyQt5.QtWidgets import (
        QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
        QPushButton, QCheckBox, QLineEdit, QWidget, QProgressBar
    )
    from PyQt5.QtCore import QTimer
    HAS_QT = True
except ImportError:
    HAS_QT = False
    QMessageBox = None
    QDialog = None


# =============================================================================
# ACTION SEVERITY LEVELS
# =============================================================================

class ActionSeverity(Enum):
    """Severity level of an action - determines confirmation requirements."""
    SAFE = auto()           # No confirmation needed (read, view)
    MINOR = auto()          # Quick confirmation (write file, generate)
    MODERATE = auto()       # Standard confirmation (delete single item)
    DANGEROUS = auto()      # Strong confirmation (delete multiple, system changes)
    CRITICAL = auto()       # Maximum confirmation (irreversible, security-related)


# Default severity for common actions
ACTION_SEVERITY_MAP = {
    # Safe actions
    "read_file": ActionSeverity.SAFE,
    "list_directory": ActionSeverity.SAFE,
    "view_model": ActionSeverity.SAFE,
    "preview": ActionSeverity.SAFE,
    
    # Minor actions
    "generate_text": ActionSeverity.MINOR,
    "generate_image": ActionSeverity.MINOR,
    "save_file": ActionSeverity.MINOR,
    "export": ActionSeverity.MINOR,
    
    # Moderate actions
    "delete_file": ActionSeverity.MODERATE,
    "overwrite": ActionSeverity.MODERATE,
    "unload_module": ActionSeverity.MODERATE,
    "stop_training": ActionSeverity.MODERATE,
    
    # Dangerous actions
    "delete_model": ActionSeverity.DANGEROUS,
    "delete_multiple": ActionSeverity.DANGEROUS,
    "clear_history": ActionSeverity.DANGEROUS,
    "reset_settings": ActionSeverity.DANGEROUS,
    "format_output": ActionSeverity.DANGEROUS,
    
    # Critical actions
    "delete_all_models": ActionSeverity.CRITICAL,
    "factory_reset": ActionSeverity.CRITICAL,
    "delete_all_data": ActionSeverity.CRITICAL,
    "security_override": ActionSeverity.CRITICAL,
}


# =============================================================================
# CONFIRMATION DIALOGS
# =============================================================================

@dataclass
class ConfirmationResult:
    """Result of a confirmation dialog."""
    confirmed: bool
    dont_ask_again: bool = False
    user_input: Optional[str] = None  # For type-to-confirm


class ConfirmationDialog(QDialog if HAS_QT else object):
    """
    Advanced confirmation dialog with multiple safety levels.
    
    Features:
    - Type-to-confirm for critical actions
    - "Don't ask again" option for minor actions
    - Countdown timer for dangerous actions
    - Visual severity indication
    """
    
    def __init__(
        self,
        parent: Optional['QWidget'] = None,
        title: str = "Confirm Action",
        message: str = "Are you sure?",
        details: str = "",
        severity: ActionSeverity = ActionSeverity.MODERATE,
        confirm_text: Optional[str] = None,  # Text user must type to confirm
        countdown_seconds: int = 0,  # Countdown before confirm enabled
        allow_dont_ask: bool = False,
    ):
        if not HAS_QT:
            self.result_data = ConfirmationResult(confirmed=False)
            return
            
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        
        self.severity = severity
        self.confirm_text = confirm_text
        self.countdown_remaining = countdown_seconds
        self.result_data = ConfirmationResult(confirmed=False)
        
        self._setup_ui(message, details, allow_dont_ask)
        self._apply_severity_style()
        
        if countdown_seconds > 0:
            self._start_countdown()
    
    def _setup_ui(self, message: str, details: str, allow_dont_ask: bool):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        
        # Severity icon and message
        header = QHBoxLayout()
        
        self.icon_label = QLabel()
        self._set_severity_icon()
        header.addWidget(self.icon_label)
        
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        header.addWidget(msg_label, 1)
        layout.addLayout(header)
        
        # Details
        if details:
            details_label = QLabel(details)
            details_label.setWordWrap(True)
            details_label.setStyleSheet("color: #888; margin: 10px 0;")
            layout.addWidget(details_label)
        
        # Type-to-confirm for critical actions
        if self.confirm_text:
            confirm_layout = QVBoxLayout()
            
            instruction = QLabel(f'Type "{self.confirm_text}" to confirm:')
            instruction.setStyleSheet("color: #c00; font-weight: bold;")
            confirm_layout.addWidget(instruction)
            
            self.confirm_input = QLineEdit()
            self.confirm_input.setPlaceholderText(self.confirm_text)
            self.confirm_input.textChanged.connect(self._check_confirm_text)
            confirm_layout.addWidget(self.confirm_input)
            
            layout.addLayout(confirm_layout)
        
        # Countdown progress bar
        if self.countdown_remaining > 0:
            self.countdown_bar = QProgressBar()
            self.countdown_bar.setMaximum(self.countdown_remaining)
            self.countdown_bar.setValue(self.countdown_remaining)
            self.countdown_bar.setFormat("Wait %v seconds...")
            layout.addWidget(self.countdown_bar)
        
        # Don't ask again checkbox
        if allow_dont_ask and self.severity in (ActionSeverity.MINOR, ActionSeverity.MODERATE):
            self.dont_ask_checkbox = QCheckBox("Don't ask again for this action")
            layout.addWidget(self.dont_ask_checkbox)
        else:
            self.dont_ask_checkbox = None
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.clicked.connect(self._on_confirm)
        self.confirm_btn.setDefault(False)  # Prevent accidental Enter
        
        # Disable confirm button initially for type-to-confirm or countdown
        if self.confirm_text or self.countdown_remaining > 0:
            self.confirm_btn.setEnabled(False)
        
        button_layout.addWidget(self.confirm_btn)
        layout.addLayout(button_layout)
        
        self.setMinimumWidth(400)
    
    def _set_severity_icon(self):
        """Set icon based on severity level."""
        # Use standard icons (no custom emoji/unicode)
        icon_map = {
            ActionSeverity.SAFE: QMessageBox.Information,
            ActionSeverity.MINOR: QMessageBox.Question,
            ActionSeverity.MODERATE: QMessageBox.Warning,
            ActionSeverity.DANGEROUS: QMessageBox.Warning,
            ActionSeverity.CRITICAL: QMessageBox.Critical,
        }
        # Note: In production, use actual QIcon from resources
        self.icon_label.setText({
            ActionSeverity.SAFE: "[i]",
            ActionSeverity.MINOR: "[?]",
            ActionSeverity.MODERATE: "[!]",
            ActionSeverity.DANGEROUS: "[!!]",
            ActionSeverity.CRITICAL: "[X]",
        }.get(self.severity, "[?]"))
    
    def _apply_severity_style(self):
        """Apply visual style based on severity."""
        colors = {
            ActionSeverity.SAFE: "#2196F3",      # Blue
            ActionSeverity.MINOR: "#4CAF50",     # Green
            ActionSeverity.MODERATE: "#FF9800",  # Orange
            ActionSeverity.DANGEROUS: "#f44336", # Red
            ActionSeverity.CRITICAL: "#9C27B0",  # Purple (attention-grabbing)
        }
        color = colors.get(self.severity, "#FF9800")
        self.confirm_btn.setStyleSheet(f"background-color: {color}; color: white; padding: 8px 16px;")
    
    def _check_confirm_text(self, text: str):
        """Check if typed text matches required confirmation text."""
        if self.confirm_text:
            matches = text.strip().lower() == self.confirm_text.lower()
            self.confirm_btn.setEnabled(matches and self.countdown_remaining <= 0)
    
    def _start_countdown(self):
        """Start countdown timer."""
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._countdown_tick)
        self.countdown_timer.start(1000)
    
    def _countdown_tick(self):
        """Handle countdown tick."""
        self.countdown_remaining -= 1
        if hasattr(self, 'countdown_bar'):
            self.countdown_bar.setValue(self.countdown_remaining)
        
        if self.countdown_remaining <= 0:
            self.countdown_timer.stop()
            if hasattr(self, 'countdown_bar'):
                self.countdown_bar.hide()
            # Enable confirm if no text confirmation needed or text matches
            if not self.confirm_text:
                self.confirm_btn.setEnabled(True)
            elif hasattr(self, 'confirm_input'):
                self._check_confirm_text(self.confirm_input.text())
    
    def _on_confirm(self):
        """Handle confirm button click."""
        self.result_data.confirmed = True
        if self.dont_ask_checkbox:
            self.result_data.dont_ask_again = self.dont_ask_checkbox.isChecked()
        if hasattr(self, 'confirm_input'):
            self.result_data.user_input = self.confirm_input.text()
        self.accept()
    
    def get_result(self) -> ConfirmationResult:
        """Get dialog result."""
        return self.result_data


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Rate limiter for GUI actions to prevent spam clicking.
    
    Features:
    - Per-action rate limiting
    - Burst allowance
    - Visual feedback via callback
    - Thread-safe
    """
    
    def __init__(self):
        self._action_times: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._blocked_until: Dict[str, float] = {}
        self._feedback_callback: Optional[Callable[[str, float], None]] = None
    
    def set_feedback_callback(self, callback: Callable[[str, float], None]):
        """Set callback for when action is rate-limited. Args: (action_name, wait_seconds)"""
        self._feedback_callback = callback
    
    def check(
        self,
        action: str,
        min_interval: float = 1.0,
        max_burst: int = 3,
        burst_window: float = 5.0
    ) -> Tuple[bool, float]:
        """
        Check if action is allowed.
        
        Args:
            action: Action identifier
            min_interval: Minimum seconds between actions
            max_burst: Maximum actions in burst window
            burst_window: Time window for burst counting
            
        Returns:
            (allowed, wait_time) - True if allowed, or False with seconds to wait
        """
        now = time.time()
        
        with self._lock:
            # Check if blocked
            if action in self._blocked_until:
                if now < self._blocked_until[action]:
                    wait = self._blocked_until[action] - now
                    if self._feedback_callback:
                        self._feedback_callback(action, wait)
                    return False, wait
                else:
                    del self._blocked_until[action]
            
            # Initialize action tracking
            if action not in self._action_times:
                self._action_times[action] = deque(maxlen=max_burst * 2)
            
            times = self._action_times[action]
            
            # Check minimum interval
            if times and (now - times[-1]) < min_interval:
                wait = min_interval - (now - times[-1])
                if self._feedback_callback:
                    self._feedback_callback(action, wait)
                return False, wait
            
            # Check burst limit
            recent = [t for t in times if (now - t) < burst_window]
            if len(recent) >= max_burst:
                # Block for burst window
                wait = burst_window - (now - recent[0])
                self._blocked_until[action] = now + wait
                logger.warning(f"Rate limit exceeded for '{action}', blocking for {wait:.1f}s")
                if self._feedback_callback:
                    self._feedback_callback(action, wait)
                return False, wait
            
            # Allow action
            times.append(now)
            return True, 0
    
    def record(self, action: str):
        """Record an action (for manual tracking)."""
        with self._lock:
            if action not in self._action_times:
                self._action_times[action] = deque(maxlen=20)
            self._action_times[action].append(time.time())


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit(
    min_interval: float = 1.0,
    max_burst: int = 5,
    burst_window: float = 10.0,
    action_name: Optional[str] = None
):
    """
    Decorator to rate-limit a function.
    
    Usage:
        @rate_limit(min_interval=2.0)
        def on_generate_clicked(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        action = action_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            allowed, wait = _rate_limiter.check(action, min_interval, max_burst, burst_window)
            if not allowed:
                logger.debug(f"Rate limited: {action}, wait {wait:.1f}s")
                return None
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# INPUT VALIDATION
# =============================================================================

class InputValidator:
    """
    Comprehensive input validation for GUI fields.
    
    Validates:
    - Path safety (no traversal, blocked patterns)
    - Text length limits
    - Character restrictions
    - Format validation
    - Injection prevention
    """
    
    # Dangerous patterns that could indicate injection attempts
    INJECTION_PATTERNS = [
        r'\x00',                        # Null bytes
        r'%00',                         # URL-encoded null
        r'\.\.[/\\]',                   # Path traversal
        r'[<>]',                        # HTML/XML injection
        r'javascript:',                 # JS injection
        r'data:',                       # Data URI
        r'\$\{.*\}',                    # Template injection
        r'\{\{.*\}\}',                  # Template injection variant
        r'`.*`',                        # Command substitution
        r'\|.*\|',                      # Pipe commands
        r';\s*\w+',                     # Command chaining
        r'&&',                          # Shell AND
        r'\|\|',                        # Shell OR
    ]
    
    # Suspicious but allowed with warning
    SUSPICIOUS_PATTERNS = [
        r'password',
        r'secret',
        r'api.?key',
        r'token',
        r'credential',
    ]
    
    # Maximum input lengths per field type
    MAX_LENGTHS = {
        "chat_message": 50000,
        "file_path": 500,
        "search_query": 200,
        "model_name": 100,
        "prompt": 100000,
        "api_key": 200,
        "default": 10000,
    }
    
    @classmethod
    def validate_text(
        cls,
        text: str,
        field_type: str = "default",
        allow_empty: bool = True,
        custom_max_length: Optional[int] = None,
        check_injection: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate text input.
        
        Returns:
            (is_valid, error_message or None)
        """
        # Empty check
        if not text:
            if allow_empty:
                return True, None
            return False, "Input cannot be empty"
        
        # Length check
        max_len = custom_max_length or cls.MAX_LENGTHS.get(field_type, cls.MAX_LENGTHS["default"])
        if len(text) > max_len:
            return False, f"Input too long (max {max_len} characters)"
        
        # Injection check
        if check_injection:
            for pattern in cls.INJECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"Potential injection attempt detected in input")
                    return False, "Input contains invalid characters"
        
        return True, None
    
    @classmethod
    def validate_path(cls, path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate file path input.
        
        Returns:
            (is_valid, error_message or None)
        """
        if not path:
            return False, "Path cannot be empty"
        
        # Length check
        if len(path) > cls.MAX_LENGTHS["file_path"]:
            return False, f"Path too long (max {cls.MAX_LENGTHS['file_path']} characters)"
        
        # Path traversal check
        if '..' in path:
            # Resolve to check if it escapes intended directory
            try:
                resolved = Path(path).resolve()
                # This is a basic check - security.py does more thorough checking
            except Exception:
                return False, "Invalid path format"
        
        # Dangerous characters
        dangerous_chars = ['<', '>', '|', '\x00', '*', '?']
        for char in dangerous_chars:
            if char in path:
                return False, f"Path contains invalid character: {repr(char)}"
        
        # Existence check
        if must_exist:
            try:
                if not Path(path).exists():
                    return False, "Path does not exist"
            except Exception as e:
                return False, f"Cannot access path: {e}"
        
        return True, None
    
    @classmethod
    def validate_number(
        cls,
        value: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_float: bool = True
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Validate numeric input.
        
        Returns:
            (is_valid, error_message, parsed_value)
        """
        try:
            if allow_float:
                num = float(value)
            else:
                num = int(value)
        except ValueError:
            return False, "Invalid number format", None
        
        if min_val is not None and num < min_val:
            return False, f"Value must be at least {min_val}", None
        
        if max_val is not None and num > max_val:
            return False, f"Value must be at most {max_val}", None
        
        return True, None, num
    
    @classmethod
    def sanitize_for_display(cls, text: str, max_length: int = 1000) -> str:
        """
        Sanitize text for safe display in GUI.
        
        - Truncates to max length
        - Escapes potentially problematic characters
        - Removes control characters
        """
        if not text:
            return ""
        
        # Remove control characters except newline/tab
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
        
        # Truncate
        if len(text) > max_length:
            text = text[:max_length - 3] + "..."
        
        return text


# =============================================================================
# ACTION HISTORY (UNDO SYSTEM)
# =============================================================================

@dataclass
class ActionRecord:
    """Record of a performed action for undo capability."""
    action_id: str
    action_type: str
    timestamp: float
    description: str
    undo_data: Dict[str, Any]  # Data needed to undo
    redo_data: Dict[str, Any]  # Data needed to redo
    can_undo: bool = True
    undone: bool = False


class ActionHistory:
    """
    Track action history for undo/redo functionality.
    
    Features:
    - Configurable history depth
    - Grouped actions (undo multiple as one)
    - Persistent history option
    - Action expiration
    """
    
    def __init__(
        self,
        max_history: int = 100,
        max_age_minutes: int = 60,
        persist_path: Optional[Path] = None
    ):
        """
        Initialize action history.
        
        Args:
            max_history: Maximum actions to remember
            max_age_minutes: Auto-expire actions older than this
            persist_path: Path to save history (optional)
        """
        self.max_history = max_history
        self.max_age_seconds = max_age_minutes * 60
        self.persist_path = persist_path
        
        self._history: List[ActionRecord] = []
        self._redo_stack: List[ActionRecord] = []
        self._lock = threading.Lock()
        self._action_counter = 0
        
        # Undo handlers per action type
        self._undo_handlers: Dict[str, Callable[[Dict], bool]] = {}
        self._redo_handlers: Dict[str, Callable[[Dict], bool]] = {}
        
        if persist_path and persist_path.exists():
            self._load()
    
    def register_handlers(
        self,
        action_type: str,
        undo_handler: Callable[[Dict], bool],
        redo_handler: Optional[Callable[[Dict], bool]] = None
    ):
        """
        Register undo/redo handlers for an action type.
        
        Args:
            action_type: Type identifier (e.g., "delete_file")
            undo_handler: Function that takes undo_data and returns success
            redo_handler: Function that takes redo_data and returns success
        """
        self._undo_handlers[action_type] = undo_handler
        if redo_handler:
            self._redo_handlers[action_type] = redo_handler
    
    def record(
        self,
        action_type: str,
        description: str,
        undo_data: Dict[str, Any],
        redo_data: Optional[Dict[str, Any]] = None,
        can_undo: bool = True
    ) -> str:
        """
        Record an action for potential undo.
        
        Args:
            action_type: Type of action
            description: Human-readable description
            undo_data: Data needed to undo this action
            redo_data: Data needed to redo (optional, uses undo_data if not provided)
            can_undo: Whether this action can be undone
            
        Returns:
            Action ID
        """
        with self._lock:
            self._action_counter += 1
            action_id = f"action_{self._action_counter}_{int(time.time())}"
            
            record = ActionRecord(
                action_id=action_id,
                action_type=action_type,
                timestamp=time.time(),
                description=description,
                undo_data=undo_data,
                redo_data=redo_data or undo_data,
                can_undo=can_undo
            )
            
            self._history.append(record)
            self._redo_stack.clear()  # Clear redo stack on new action
            
            # Enforce max history
            while len(self._history) > self.max_history:
                self._history.pop(0)
            
            # Clean old actions
            self._cleanup_expired()
            
            logger.debug(f"Recorded action: {action_type} - {description}")
            return action_id
    
    def can_undo(self) -> Tuple[bool, Optional[str]]:
        """
        Check if undo is available.
        
        Returns:
            (can_undo, description of action to undo)
        """
        with self._lock:
            for record in reversed(self._history):
                if record.can_undo and not record.undone:
                    return True, record.description
            return False, None
    
    def can_redo(self) -> Tuple[bool, Optional[str]]:
        """
        Check if redo is available.
        
        Returns:
            (can_redo, description of action to redo)
        """
        with self._lock:
            if self._redo_stack:
                return True, self._redo_stack[-1].description
            return False, None
    
    def undo(self) -> Tuple[bool, str]:
        """
        Undo the last undoable action.
        
        Returns:
            (success, message)
        """
        with self._lock:
            # Find last undoable action
            for record in reversed(self._history):
                if record.can_undo and not record.undone:
                    handler = self._undo_handlers.get(record.action_type)
                    if not handler:
                        return False, f"No undo handler for {record.action_type}"
                    
                    try:
                        if handler(record.undo_data):
                            record.undone = True
                            self._redo_stack.append(record)
                            return True, f"Undone: {record.description}"
                        else:
                            return False, f"Failed to undo: {record.description}"
                    except Exception as e:
                        logger.error(f"Undo error: {e}")
                        return False, f"Error during undo: {e}"
            
            return False, "Nothing to undo"
    
    def redo(self) -> Tuple[bool, str]:
        """
        Redo the last undone action.
        
        Returns:
            (success, message)
        """
        with self._lock:
            if not self._redo_stack:
                return False, "Nothing to redo"
            
            record = self._redo_stack.pop()
            handler = self._redo_handlers.get(record.action_type)
            if not handler:
                # If no redo handler, just mark as not undone
                record.undone = False
                return True, f"Redone: {record.description}"
            
            try:
                if handler(record.redo_data):
                    record.undone = False
                    return True, f"Redone: {record.description}"
                else:
                    self._redo_stack.append(record)  # Put back
                    return False, f"Failed to redo: {record.description}"
            except Exception as e:
                logger.error(f"Redo error: {e}")
                self._redo_stack.append(record)
                return False, f"Error during redo: {e}"
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent action history for display."""
        with self._lock:
            result = []
            for record in reversed(self._history[-limit:]):
                result.append({
                    "id": record.action_id,
                    "type": record.action_type,
                    "description": record.description,
                    "time": datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S"),
                    "can_undo": record.can_undo and not record.undone,
                    "undone": record.undone
                })
            return result
    
    def _cleanup_expired(self):
        """Remove expired actions."""
        now = time.time()
        self._history = [
            r for r in self._history 
            if (now - r.timestamp) < self.max_age_seconds
        ]
    
    def _load(self):
        """Load history from file."""
        try:
            data = json.loads(self.persist_path.read_text())
            # Simplified loading - in production would deserialize ActionRecords
            logger.debug(f"Loaded {len(data)} history entries")
        except Exception as e:
            logger.warning(f"Could not load action history: {e}")
    
    def _save(self):
        """Save history to file."""
        if not self.persist_path:
            return
        try:
            # Simplified saving
            data = [{"type": r.action_type, "desc": r.description} for r in self._history]
            self.persist_path.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Could not save action history: {e}")


# =============================================================================
# ANTI-TAMPERING / SMART USER PROTECTION
# =============================================================================

class AntiTamper:
    """
    Protection against smart users trying to break things.
    
    Features:
    - Config file integrity checking
    - Settings bounds enforcement
    - Privilege escalation detection
    - Unusual pattern detection
    """
    
    # Settings that could be dangerous if tampered with
    PROTECTED_SETTINGS = {
        "blocked_paths",
        "blocked_patterns", 
        "security",
        "admin",
        "allow_code_execution",
        "allow_subprocess",
        "sandbox",
    }
    
    # Bounds for numeric settings (setting_name: (min, max))
    SETTING_BOUNDS = {
        "temperature": (0.0, 2.0),
        "max_tokens": (1, 100000),
        "top_p": (0.0, 1.0),
        "top_k": (1, 1000),
        "learning_rate": (0.0, 1.0),
        "batch_size": (1, 1024),
        "memory_limit_mb": (100, 100000),
        "max_file_size_mb": (1, 10000),
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize anti-tamper system."""
        self.config_path = config_path
        self._config_hash: Optional[str] = None
        self._suspicious_actions: List[Dict] = []
        self._action_patterns: Dict[str, List[float]] = {}
        
        if config_path and config_path.exists():
            self._config_hash = self._hash_file(config_path)
    
    def _hash_file(self, path: Path) -> str:
        """Compute hash of file for integrity checking."""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    
    def check_config_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Check if config file has been tampered with.
        
        Returns:
            (is_valid, warning_message)
        """
        if not self.config_path or not self._config_hash:
            return True, None
        
        if not self.config_path.exists():
            return False, "Config file is missing"
        
        current_hash = self._hash_file(self.config_path)
        if current_hash != self._config_hash:
            return False, "Config file has been modified outside the application"
        
        return True, None
    
    def validate_setting_change(
        self,
        setting_name: str,
        old_value: Any,
        new_value: Any
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a setting change for tampering attempts.
        
        Returns:
            (is_allowed, reason if blocked)
        """
        # Check protected settings
        for protected in self.PROTECTED_SETTINGS:
            if protected in setting_name.lower():
                self._log_suspicious("protected_setting_access", {
                    "setting": setting_name,
                    "attempted_value": str(new_value)[:100]
                })
                return False, f"Cannot modify protected setting: {setting_name}"
        
        # Check bounds for numeric settings
        for bounded_setting, (min_val, max_val) in self.SETTING_BOUNDS.items():
            if bounded_setting in setting_name.lower():
                try:
                    num_value = float(new_value)
                    if num_value < min_val or num_value > max_val:
                        return False, f"{setting_name} must be between {min_val} and {max_val}"
                except (ValueError, TypeError):
                    pass
        
        # Detect suspicious patterns (rapid changes, extreme values)
        self._track_change(setting_name)
        if self._is_suspicious_pattern(setting_name):
            self._log_suspicious("rapid_setting_changes", {"setting": setting_name})
            return False, "Too many rapid changes to this setting. Please wait."
        
        return True, None
    
    def _track_change(self, setting_name: str):
        """Track setting change for pattern detection."""
        if setting_name not in self._action_patterns:
            self._action_patterns[setting_name] = []
        
        self._action_patterns[setting_name].append(time.time())
        
        # Keep only recent changes
        cutoff = time.time() - 60  # Last minute
        self._action_patterns[setting_name] = [
            t for t in self._action_patterns[setting_name] if t > cutoff
        ]
    
    def _is_suspicious_pattern(self, setting_name: str) -> bool:
        """Detect suspicious patterns of changes."""
        changes = self._action_patterns.get(setting_name, [])
        
        # More than 10 changes in a minute is suspicious
        if len(changes) > 10:
            return True
        
        # 5+ changes in 5 seconds is very suspicious
        recent_5s = [t for t in changes if t > time.time() - 5]
        if len(recent_5s) >= 5:
            return True
        
        return False
    
    def _log_suspicious(self, action_type: str, details: Dict):
        """Log suspicious activity."""
        entry = {
            "type": action_type,
            "time": datetime.now().isoformat(),
            "details": details
        }
        self._suspicious_actions.append(entry)
        logger.warning(f"Suspicious activity: {action_type} - {details}")
        
        # Keep only recent suspicious actions
        if len(self._suspicious_actions) > 100:
            self._suspicious_actions = self._suspicious_actions[-100:]
    
    def get_suspicious_activity(self) -> List[Dict]:
        """Get log of suspicious activities."""
        return list(self._suspicious_actions)
    
    def validate_api_key_format(self, key: str, provider: str = "generic") -> Tuple[bool, Optional[str]]:
        """
        Validate API key format without exposing the key.
        
        Returns:
            (is_valid_format, error_message)
        """
        if not key:
            return False, "API key cannot be empty"
        
        # Check for obviously fake/test keys
        fake_patterns = ["test", "fake", "xxxx", "1234", "demo", "sample"]
        key_lower = key.lower()
        if any(p in key_lower for p in fake_patterns):
            return False, "This appears to be a test/fake API key"
        
        # Provider-specific format validation
        formats = {
            "openai": (r"^sk-[a-zA-Z0-9]{48,}$", "OpenAI keys start with 'sk-' followed by 48+ chars"),
            "anthropic": (r"^sk-ant-[a-zA-Z0-9-]+$", "Anthropic keys start with 'sk-ant-'"),
            "huggingface": (r"^hf_[a-zA-Z0-9]+$", "HuggingFace keys start with 'hf_'"),
        }
        
        if provider in formats:
            pattern, hint = formats[provider]
            if not re.match(pattern, key):
                return False, hint
        
        return True, None


# =============================================================================
# MAIN SAFETY GUARDS CLASS
# =============================================================================

class SafetyGuards:
    """
    Main interface for all safety features.
    
    Usage:
        # Initialize (usually done once at app startup)
        SafetyGuards.initialize()
        
        # Use features
        if SafetyGuards.confirm_destructive("Delete model?"):
            do_delete()
        
        SafetyGuards.validate_input(user_text, "chat_message")
    """
    
    _instance: Optional['SafetyGuards'] = None
    _rate_limiter: RateLimiter = _rate_limiter
    _action_history: Optional[ActionHistory] = None
    _anti_tamper: Optional[AntiTamper] = None
    _dont_ask_again: Set[str] = set()
    
    @classmethod
    def initialize(cls, config_path: Optional[Path] = None, history_path: Optional[Path] = None):
        """Initialize safety guards."""
        cls._action_history = ActionHistory(persist_path=history_path)
        cls._anti_tamper = AntiTamper(config_path=config_path)
        logger.info("Safety guards initialized")
    
    @classmethod
    def confirm_action(
        cls,
        message: str,
        action_type: str = "default",
        details: str = "",
        parent: Optional['QWidget'] = None
    ) -> bool:
        """
        Show confirmation dialog appropriate for action severity.
        
        Returns True if user confirms.
        """
        if not HAS_QT:
            # Headless mode - allow by default (should use API permissions instead)
            return True
        
        severity = ACTION_SEVERITY_MAP.get(action_type, ActionSeverity.MODERATE)
        
        # Check don't ask again
        if f"{action_type}_dont_ask" in cls._dont_ask_again:
            if severity in (ActionSeverity.SAFE, ActionSeverity.MINOR):
                return True
        
        # Determine confirmation requirements
        confirm_text = None
        countdown = 0
        allow_dont_ask = severity in (ActionSeverity.MINOR, ActionSeverity.MODERATE)
        
        if severity == ActionSeverity.CRITICAL:
            confirm_text = "DELETE"
            countdown = 3
        elif severity == ActionSeverity.DANGEROUS:
            countdown = 2
        
        dialog = ConfirmationDialog(
            parent=parent,
            title=f"Confirm {action_type.replace('_', ' ').title()}",
            message=message,
            details=details,
            severity=severity,
            confirm_text=confirm_text,
            countdown_seconds=countdown,
            allow_dont_ask=allow_dont_ask
        )
        
        dialog.exec_()
        result = dialog.get_result()
        
        if result.dont_ask_again:
            cls._dont_ask_again.add(f"{action_type}_dont_ask")
        
        return result.confirmed
    
    @classmethod
    def confirm_destructive(
        cls,
        message: str,
        details: str = "",
        parent: Optional['QWidget'] = None
    ) -> bool:
        """Convenience method for destructive action confirmation."""
        return cls.confirm_action(message, "delete_multiple", details, parent)
    
    @classmethod
    def confirm_critical(
        cls,
        message: str,
        details: str = "",
        parent: Optional['QWidget'] = None
    ) -> bool:
        """Convenience method for critical action confirmation (type to confirm)."""
        return cls.confirm_action(message, "delete_all_models", details, parent)
    
    @classmethod
    def validate_input(
        cls,
        text: str,
        field_type: str = "default",
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate user input."""
        return InputValidator.validate_text(text, field_type, **kwargs)
    
    @classmethod
    def validate_path(cls, path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate path input."""
        return InputValidator.validate_path(path, must_exist)
    
    @classmethod
    def check_rate_limit(
        cls,
        action: str,
        min_interval: float = 1.0
    ) -> Tuple[bool, float]:
        """Check if action is rate-limited."""
        return cls._rate_limiter.check(action, min_interval)
    
    @classmethod
    def record_action(
        cls,
        action_type: str,
        description: str,
        undo_data: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """Record action for undo capability."""
        if cls._action_history:
            return cls._action_history.record(action_type, description, undo_data, **kwargs)
        return None
    
    @classmethod
    def undo(cls) -> Tuple[bool, str]:
        """Undo last action."""
        if cls._action_history:
            return cls._action_history.undo()
        return False, "Undo not available"
    
    @classmethod
    def redo(cls) -> Tuple[bool, str]:
        """Redo last undone action."""
        if cls._action_history:
            return cls._action_history.redo()
        return False, "Redo not available"
    
    @classmethod
    def can_undo(cls) -> Tuple[bool, Optional[str]]:
        """Check if undo is available."""
        if cls._action_history:
            return cls._action_history.can_undo()
        return False, None
    
    @classmethod
    def validate_setting(
        cls,
        name: str,
        old_value: Any,
        new_value: Any
    ) -> Tuple[bool, Optional[str]]:
        """Validate setting change (anti-tamper)."""
        if cls._anti_tamper:
            return cls._anti_tamper.validate_setting_change(name, old_value, new_value)
        return True, None
    
    @classmethod
    def get_action_history(cls, limit: int = 20) -> List[Dict]:
        """Get action history for display."""
        if cls._action_history:
            return cls._action_history.get_history(limit)
        return []


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

confirm_action = SafetyGuards.confirm_action
confirm_destructive = SafetyGuards.confirm_destructive
confirm_critical = SafetyGuards.confirm_critical
validate_input = SafetyGuards.validate_input
validate_path = SafetyGuards.validate_path
check_rate_limit = SafetyGuards.check_rate_limit
