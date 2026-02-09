"""
Audit Logging for Enigma AI Engine

Track all API calls, tool usage, and system events.

Features:
- Structured logging
- Timestamp tracking
- User/session tracking
- Query logging
- Export capabilities

Usage:
    from enigma_engine.utils.audit_logger import AuditLogger, get_audit_logger
    
    audit = get_audit_logger()
    
    # Log API call
    audit.log_api_call("inference", user_id="user1", input="Hello")
    
    # Log tool usage
    audit.log_tool_call("web_search", {"query": "weather"})
    
    # Export logs
    audit.export_logs("audit_2024.json")
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
from collections import deque

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of audit events."""
    API_CALL = "api_call"
    TOOL_CALL = "tool_call"
    MODEL_LOAD = "model_load"
    MODEL_INFERENCE = "model_inference"
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    FILE_ACCESS = "file_access"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    ERROR = "error"
    SYSTEM = "system"
    USER_ACTION = "user_action"
    MEMORY_ACCESS = "memory_access"


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Single audit event."""
    event_id: str
    event_type: EventType
    timestamp: float
    
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event details
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    
    # Metadata
    level: LogLevel = LogLevel.INFO
    source: str = ""
    duration_ms: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['event_type'] = self.event_type.value
        d['level'] = self.level.value
        d['timestamp_iso'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d


@dataclass
class AuditConfig:
    """Audit logger configuration."""
    # Storage
    log_dir: str = "logs/audit"
    max_file_size_mb: int = 100
    max_files: int = 10
    
    # What to log
    log_api_calls: bool = True
    log_tool_calls: bool = True
    log_inference: bool = True
    log_file_access: bool = True
    
    # Retention
    retention_days: int = 90
    
    # Memory buffer
    buffer_size: int = 1000
    flush_interval_seconds: int = 60
    
    # PII handling
    scrub_pii: bool = True
    redact_inputs: bool = False  # Redact all input text


class AuditLogger:
    """Log and track system events."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        """
        Initialize audit logger.
        
        Args:
            config: Logger configuration
        """
        self._config = config or AuditConfig()
        self._log_dir = Path(self._config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffer
        self._buffer: deque = deque(maxlen=self._config.buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Current log file
        self._current_file: Optional[Path] = None
        self._current_file_size = 0
        
        # Session tracking
        self._session_id = self._generate_id()
        
        # Auto-flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Event counter
        self._event_counter = 0
        self._counter_lock = threading.Lock()
        
        # Callbacks
        self._event_callbacks: List[Callable[[AuditEvent], None]] = []
        
        self._start_auto_flush()
        logger.info(f"AuditLogger initialized, session={self._session_id[:8]}")
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _start_auto_flush(self):
        """Start auto-flush thread."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def _flush_loop(self):
        """Background flush loop."""
        while self._running:
            time.sleep(self._config.flush_interval_seconds)
            try:
                self.flush()
            except Exception as e:
                logger.error(f"Flush error: {e}")
    
    def _get_log_file(self) -> Path:
        """Get current log file path."""
        # Rotate if needed
        if self._current_file:
            if self._current_file_size >= self._config.max_file_size_mb * 1024 * 1024:
                self._rotate_logs()
        
        if not self._current_file:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._current_file = self._log_dir / f"audit_{date_str}.jsonl"
            self._current_file_size = 0
        
        return self._current_file
    
    def _rotate_logs(self):
        """Rotate log files."""
        self._current_file = None
        
        # Clean old logs
        log_files = sorted(self._log_dir.glob("audit_*.jsonl"))
        while len(log_files) > self._config.max_files:
            old_file = log_files.pop(0)
            old_file.unlink()
            logger.info(f"Deleted old audit log: {old_file}")
    
    def log_event(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: Event to log
        """
        # Add to buffer
        with self._buffer_lock:
            self._buffer.append(event)
        
        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")
    
    def log(
        self,
        event_type: EventType,
        action: str,
        user_id: Optional[str] = None,
        parameters: Optional[Dict] = None,
        result: Optional[str] = None,
        level: LogLevel = LogLevel.INFO,
        source: str = "",
        duration_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Log an event with parameters.
        
        Args:
            event_type: Type of event
            action: Action performed
            user_id: User identifier
            parameters: Event parameters
            result: Result of action
            level: Log level
            source: Source module/component
            duration_ms: Duration in milliseconds
            error: Error message if any
        """
        with self._counter_lock:
            self._event_counter += 1
            event_id = f"{self._session_id[:8]}_{self._event_counter:08d}"
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            session_id=self._session_id,
            action=action,
            parameters=self._sanitize_params(parameters or {}),
            result=result,
            level=level,
            source=source,
            duration_ms=duration_ms,
            error=error
        )
        
        self.log_event(event)
    
    def _sanitize_params(self, params: Dict) -> Dict:
        """Sanitize parameters for logging."""
        if not self._config.scrub_pii:
            return params
        
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Check for sensitive keys
                if any(s in key.lower() for s in ['password', 'token', 'key', 'secret']):
                    sanitized[key] = "[REDACTED]"
                elif self._config.redact_inputs and key in ['input', 'prompt', 'text']:
                    sanitized[key] = f"[{len(value)} chars]"
                else:
                    sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    # Convenience methods
    def log_api_call(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log API call."""
        if not self._config.log_api_calls:
            return
        
        self.log(
            EventType.API_CALL,
            f"API: {endpoint}",
            user_id=user_id,
            parameters=kwargs,
            source="api"
        )
    
    def log_tool_call(
        self,
        tool_name: str,
        parameters: Optional[Dict] = None,
        result: Optional[str] = None,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None
    ):
        """Log tool usage."""
        if not self._config.log_tool_calls:
            return
        
        self.log(
            EventType.TOOL_CALL,
            f"Tool: {tool_name}",
            user_id=user_id,
            parameters=parameters,
            result=result,
            duration_ms=duration_ms,
            source="tools"
        )
    
    def log_inference(
        self,
        model_name: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
        user_id: Optional[str] = None
    ):
        """Log model inference."""
        if not self._config.log_inference:
            return
        
        self.log(
            EventType.MODEL_INFERENCE,
            f"Inference: {model_name}",
            user_id=user_id,
            parameters={
                "model": model_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out
            },
            duration_ms=duration_ms,
            source="inference"
        )
    
    def log_error(
        self,
        error: str,
        traceback: Optional[str] = None,
        source: str = "",
        user_id: Optional[str] = None
    ):
        """Log error."""
        with self._counter_lock:
            self._event_counter += 1
            event_id = f"{self._session_id[:8]}_{self._event_counter:08d}"
        
        event = AuditEvent(
            event_id=event_id,
            event_type=EventType.ERROR,
            timestamp=time.time(),
            user_id=user_id,
            session_id=self._session_id,
            action="Error",
            level=LogLevel.ERROR,
            source=source,
            error=error,
            error_traceback=traceback
        )
        
        self.log_event(event)
    
    def log_file_access(
        self,
        filepath: str,
        operation: str,  # "read", "write", "delete"
        user_id: Optional[str] = None
    ):
        """Log file access."""
        if not self._config.log_file_access:
            return
        
        self.log(
            EventType.FILE_ACCESS,
            f"File: {operation}",
            user_id=user_id,
            parameters={"path": filepath, "operation": operation},
            source="files"
        )
    
    def log_authentication(
        self,
        action: str,  # "login", "logout", "failed"
        user_id: Optional[str] = None,
        success: bool = True
    ):
        """Log authentication event."""
        self.log(
            EventType.AUTHENTICATION,
            f"Auth: {action}",
            user_id=user_id,
            parameters={"success": success},
            level=LogLevel.INFO if success else LogLevel.WARNING,
            source="auth"
        )
    
    # Buffer management
    def flush(self):
        """Flush buffer to disk."""
        with self._buffer_lock:
            if not self._buffer:
                return
            
            events = list(self._buffer)
            self._buffer.clear()
        
        log_file = self._get_log_file()
        
        with open(log_file, 'a') as f:
            for event in events:
                line = json.dumps(event.to_dict()) + '\n'
                f.write(line)
                self._current_file_size += len(line)
    
    # Query and export
    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[EventType] = None
    ) -> List[AuditEvent]:
        """Get recent events from buffer."""
        with self._buffer_lock:
            events = list(self._buffer)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-count:]
    
    def export_logs(
        self,
        output_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None
    ):
        """
        Export logs to file.
        
        Args:
            output_path: Output file path
            start_time: Start timestamp
            end_time: End timestamp
            event_types: Filter by event types
        """
        # Flush first
        self.flush()
        
        events = []
        
        # Read all log files
        for log_file in sorted(self._log_dir.glob("audit_*.jsonl")):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        
                        # Filter by time
                        if start_time and event['timestamp'] < start_time:
                            continue
                        if end_time and event['timestamp'] > end_time:
                            continue
                        
                        # Filter by type
                        if event_types:
                            if event['event_type'] not in [t.value for t in event_types]:
                                continue
                        
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        # Write export
        with open(output_path, 'w') as f:
            json.dump({
                'exported_at': datetime.now().isoformat(),
                'total_events': len(events),
                'events': events
            }, f, indent=2)
        
        logger.info(f"Exported {len(events)} events to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        self.flush()
        
        total_events = 0
        by_type = {}
        by_level = {}
        
        for log_file in self._log_dir.glob("audit_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        total_events += 1
                        
                        event_type = event.get('event_type', 'unknown')
                        by_type[event_type] = by_type.get(event_type, 0) + 1
                        
                        level = event.get('level', 'info')
                        by_level[level] = by_level.get(level, 0) + 1
                    except Exception:
                        continue
        
        return {
            'total_events': total_events,
            'by_type': by_type,
            'by_level': by_level,
            'session_id': self._session_id,
            'log_files': len(list(self._log_dir.glob("audit_*.jsonl")))
        }
    
    def on_event(self, callback: Callable[[AuditEvent], None]):
        """Register callback for events."""
        self._event_callbacks.append(callback)
    
    def shutdown(self):
        """Shutdown logger and flush remaining events."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=2)
        self.flush()


# Global instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(config: Optional[AuditConfig] = None) -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(config)
    return _audit_logger
