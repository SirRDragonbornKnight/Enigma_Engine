"""
Audit Logging for Security-Sensitive Operations

Logs important actions for security review and compliance.
Tracks who did what, when, and from where.

Usage:
    from enigma_engine.utils.audit_log import audit_log, AuditLogger, AuditEvent
    
    # Simple logging
    audit_log("file_access", user="admin", path="/etc/passwd", allowed=False)
    
    # Or use the logger directly
    logger = AuditLogger()
    logger.log(AuditEvent(
        action="model_load",
        user="system",
        details={"model": "forge-small"}
    ))
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"           # Normal operations
    WARNING = "warning"     # Unusual but allowed
    ALERT = "alert"         # Security-relevant
    CRITICAL = "critical"   # Security breach attempt


class AuditCategory(Enum):
    """Categories of auditable events."""
    AUTH = "authentication"
    ACCESS = "access"
    DATA = "data"
    CONFIG = "configuration"
    ADMIN = "administration"
    TOOL = "tool_execution"
    MODEL = "model"
    NETWORK = "network"


@dataclass
class AuditEvent:
    """An auditable event."""
    action: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user: str = "unknown"
    category: str = "general"
    severity: str = "info"
    success: bool = True
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Audit logger for security-sensitive operations.
    
    Supports multiple output destinations:
    - File (JSON lines format)
    - Python logging
    - Custom handlers
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
        log_to_console: bool = False
    ):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file (defaults to data/audit.log)
            max_file_size_mb: Max file size before rotation
            backup_count: Number of backup files to keep
            log_to_console: Also log to console
        """
        # Set default log path
        if log_file is None:
            from ..config import CONFIG
            data_dir = Path(CONFIG.get("data_dir", "data"))
            log_file = str(data_dir / "audit.log")
        
        self.log_file = Path(log_file)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.log_to_console = log_to_console
        
        # Ensure directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for file writes
        self._lock = threading.Lock()
        
        # Custom handlers
        self._handlers: list[Callable[[AuditEvent], None]] = []
        
        # Buffer for batching
        self._buffer: list[AuditEvent] = []
        self._buffer_size = 10
        self._last_flush = time.time()
        
        logger.info(f"Audit logger initialized: {self.log_file}")
    
    def add_handler(self, handler: Callable[[AuditEvent], None]):
        """Add a custom event handler."""
        self._handlers.append(handler)
    
    def log(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: The audit event to log
        """
        # Log to Python logger
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "alert": logging.WARNING,
            "critical": logging.CRITICAL
        }.get(event.severity, logging.INFO)
        
        logger.log(log_level, f"AUDIT: {event.action} by {event.user} - {event.details}")
        
        # Console output if enabled
        if self.log_to_console:
            status = "OK" if event.success else "FAIL"
            print(f"[AUDIT] [{event.severity.upper()}] {event.action}: {status}")
        
        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Audit handler error: {e}")
        
        # Add to buffer
        self._buffer.append(event)
        
        # Flush if buffer is full or critical event
        if len(self._buffer) >= self._buffer_size or event.severity == "critical":
            self.flush()
    
    def flush(self):
        """Flush buffered events to file."""
        if not self._buffer:
            return
        
        with self._lock:
            events = self._buffer.copy()
            self._buffer.clear()
        
        try:
            # Check for rotation
            self._rotate_if_needed()
            
            # Append to file
            with open(self.log_file, "a", encoding="utf-8") as f:
                for event in events:
                    f.write(event.to_json() + "\n")
            
            self._last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            # Re-add events to buffer on failure
            with self._lock:
                self._buffer = events + self._buffer
    
    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        if not self.log_file.exists():
            return
        
        if self.log_file.stat().st_size < self.max_file_size:
            return
        
        # Rotate files
        for i in range(self.backup_count - 1, 0, -1):
            old = self.log_file.with_suffix(f".log.{i}")
            new = self.log_file.with_suffix(f".log.{i+1}")
            if old.exists():
                old.rename(new)
        
        # Rename current to .1
        backup = self.log_file.with_suffix(".log.1")
        self.log_file.rename(backup)
        
        logger.info(f"Rotated audit log: {self.log_file}")
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user: Optional[str] = None,
        action: Optional[str] = None,
        category: Optional[str] = None,
        success_only: bool = False,
        limit: int = 100
    ) -> list[AuditEvent]:
        """
        Query audit log for events.
        
        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            user: Filter by user
            action: Filter by action
            category: Filter by category
            success_only: Only include successful events
            limit: Maximum events to return
            
        Returns:
            List of matching events
        """
        events = []
        
        if not self.log_file.exists():
            return events
        
        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if len(events) >= limit:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent(**data)
                        
                        # Apply filters
                        if user and event.user != user:
                            continue
                        if action and event.action != action:
                            continue
                        if category and event.category != category:
                            continue
                        if success_only and not event.success:
                            continue
                        
                        # Time filters
                        event_time = datetime.fromisoformat(event.timestamp)
                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue
                        
                        events.append(event)
                        
                    except (json.JSONDecodeError, TypeError):
                        continue
                        
        except Exception as e:
            logger.error(f"Error querying audit log: {e}")
        
        return events


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    action: str,
    user: str = "system",
    category: str = "general",
    severity: str = "info",
    success: bool = True,
    **details
):
    """
    Simple function to log an audit event.
    
    Usage:
        audit_log("file_access", user="admin", path="/etc/passwd", allowed=False)
    """
    event = AuditEvent(
        action=action,
        user=user,
        category=category,
        severity=severity,
        success=success,
        details=details
    )
    get_audit_logger().log(event)


# Decorator for auditing function calls
def audited(
    action: Optional[str] = None,
    category: str = "general",
    severity: str = "info"
):
    """
    Decorator to automatically audit function calls.
    
    Usage:
        @audited(action="load_model", category="model")
        def load_model(name: str):
            ...
    """
    def decorator(f: Callable) -> Callable:
        audit_action = action or f.__name__
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            result = None
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                audit_log(
                    audit_action,
                    category=category,
                    severity=severity if success else "alert",
                    success=success,
                    duration_ms=int(duration * 1000),
                    error=error,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys())
                )
        
        return wrapper
    return decorator
