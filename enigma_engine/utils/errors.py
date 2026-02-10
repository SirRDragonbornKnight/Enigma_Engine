"""
Structured Error Handling - Proper error returns and exceptions.

Provides consistent error handling for:
- Custom exception hierarchy
- Result types (success/failure)
- Error codes and messages
- Error context and recovery hints
- Error aggregation

Part of the Enigma AI Engine core utilities.
"""

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

T = TypeVar('T')


class ErrorCode(Enum):
    """Standard error codes."""
    # General (1xxx)
    UNKNOWN = 1000
    INVALID_INPUT = 1001
    INVALID_STATE = 1002
    NOT_FOUND = 1003
    ALREADY_EXISTS = 1004
    PERMISSION_DENIED = 1005
    TIMEOUT = 1006
    CANCELLED = 1007
    
    # Configuration (2xxx)
    CONFIG_ERROR = 2000
    CONFIG_MISSING = 2001
    CONFIG_INVALID = 2002
    
    # Model (3xxx)
    MODEL_ERROR = 3000
    MODEL_NOT_LOADED = 3001
    MODEL_LOAD_FAILED = 3002
    MODEL_INFERENCE_FAILED = 3003
    MODEL_TRAINING_FAILED = 3004
    
    # Module (4xxx)
    MODULE_ERROR = 4000
    MODULE_NOT_FOUND = 4001
    MODULE_LOAD_FAILED = 4002
    MODULE_DEPENDENCY_MISSING = 4003
    MODULE_CONFLICT = 4004
    
    # Network (5xxx)
    NETWORK_ERROR = 5000
    CONNECTION_FAILED = 5001
    CONNECTION_TIMEOUT = 5002
    API_ERROR = 5003
    RATE_LIMITED = 5004
    
    # Storage (6xxx)
    STORAGE_ERROR = 6000
    FILE_NOT_FOUND = 6001
    FILE_READ_ERROR = 6002
    FILE_WRITE_ERROR = 6003
    DISK_FULL = 6004
    
    # Memory (7xxx)
    MEMORY_ERROR = 7000
    OUT_OF_MEMORY = 7001
    GPU_OUT_OF_MEMORY = 7002
    
    # Tool (8xxx)
    TOOL_ERROR = 8000
    TOOL_NOT_FOUND = 8001
    TOOL_EXECUTION_FAILED = 8002
    TOOL_INVALID_ARGS = 8003


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ErrorContext:
    """Additional context for an error."""
    operation: Optional[str] = None
    component: Optional[str] = None
    input_data: Optional[dict[str, Any]] = None
    recoverable: bool = True
    recovery_hint: Optional[str] = None
    related_errors: list['ForgeError'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "component": self.component,
            "input_data": self.input_data,
            "recoverable": self.recoverable,
            "recovery_hint": self.recovery_hint,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace
        }


class ForgeError(Exception):
    """
    Base exception for Enigma AI Engine.
    
    Usage:
        raise ForgeError(
            message="Model failed to load",
            code=ErrorCode.MODEL_LOAD_FAILED,
            context=ErrorContext(
                operation="load_model",
                component="model_registry",
                recoverable=True,
                recovery_hint="Try reducing model size or freeing memory"
            )
        )
    """
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize ForgeError.
        
        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            severity: Error severity level
            context: Additional error context
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        
        # Capture stack trace
        if self.context.stack_trace is None:
            self.context.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "message": self.message,
            "code": self.code.value,
            "code_name": self.code.name,
            "severity": self.severity.name,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"
    
    def __repr__(self) -> str:
        return f"ForgeError({self.code.name}, '{self.message}')"


# Specific exception types
class ConfigError(ForgeError):
    """Configuration-related errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', ErrorCode.CONFIG_ERROR)
        super().__init__(message, **kwargs)


class ModelError(ForgeError):
    """Model-related errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', ErrorCode.MODEL_ERROR)
        super().__init__(message, **kwargs)


class ModuleError(ForgeError):
    """Module-related errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', ErrorCode.MODULE_ERROR)
        super().__init__(message, **kwargs)


class NetworkError(ForgeError):
    """Network-related errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', ErrorCode.NETWORK_ERROR)
        super().__init__(message, **kwargs)


class StorageError(ForgeError):
    """Storage-related errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', ErrorCode.STORAGE_ERROR)
        super().__init__(message, **kwargs)


class ToolError(ForgeError):
    """Tool-related errors."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('code', ErrorCode.TOOL_ERROR)
        super().__init__(message, **kwargs)


@dataclass
class Result(Generic[T]):
    """
    Result type for functions that can fail.
    
    Usage:
        def divide(a: int, b: int) -> Result[float]:
            if b == 0:
                return Result.failure("Division by zero", ErrorCode.INVALID_INPUT)
            return Result.success(a / b)
        
        result = divide(10, 2)
        if result.is_success:
            print(f"Result: {result.value}")
        else:
            print(f"Error: {result.error}")
        
        # Or use unwrap
        try:
            value = result.unwrap()
        except ForgeError as e:
            print(f"Failed: {e}")
    """
    _value: Optional[T] = None
    _error: Optional[ForgeError] = None
    
    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._error is None
    
    @property
    def is_failure(self) -> bool:
        """Check if result is a failure."""
        return self._error is not None
    
    @property
    def value(self) -> Optional[T]:
        """Get the value (None if failure)."""
        return self._value
    
    @property
    def error(self) -> Optional[ForgeError]:
        """Get the error (None if success)."""
        return self._error
    
    def unwrap(self) -> T:
        """
        Get value or raise error.
        
        Returns:
            The value if successful
            
        Raises:
            ForgeError: If result is a failure
        """
        if self._error:
            raise self._error
        return self._value  # type: ignore
    
    def unwrap_or(self, default: T) -> T:
        """Get value or return default."""
        return self._value if self.is_success else default
    
    def unwrap_or_else(self, fn: Callable[[ForgeError], T]) -> T:
        """Get value or compute from error."""
        if self.is_success:
            return self._value  # type: ignore
        return fn(self._error)  # type: ignore
    
    def map(self, fn: Callable[[T], 'U']) -> 'Result[U]':
        """Transform the value if successful."""
        if self.is_success:
            try:
                return Result.success(fn(self._value))  # type: ignore
            except Exception as e:
                return Result.failure(str(e))
        return Result(_error=self._error)
    
    def flat_map(self, fn: Callable[[T], 'Result[U]']) -> 'Result[U]':
        """Chain result-returning functions."""
        if self.is_success:
            return fn(self._value)  # type: ignore
        return Result(_error=self._error)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        if self.is_success:
            return {"success": True, "value": self._value}
        return {"success": False, "error": self._error.to_dict() if self._error else None}
    
    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        """Create a successful result."""
        return cls(_value=value)
    
    @classmethod
    def failure(
        cls,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        **kwargs
    ) -> 'Result[T]':
        """Create a failed result."""
        error = ForgeError(message, code=code, **kwargs)
        return cls(_error=error)
    
    @classmethod
    def from_exception(cls, exc: Exception) -> 'Result[T]':
        """Create result from exception."""
        if isinstance(exc, ForgeError):
            return cls(_error=exc)
        return cls(_error=ForgeError(str(exc), cause=exc))


U = TypeVar('U')


class ErrorAggregator:
    """
    Collect and aggregate multiple errors.
    
    Usage:
        aggregator = ErrorAggregator()
        
        for item in items:
            try:
                process(item)
            except ForgeError as e:
                aggregator.add(e)
        
        if aggregator.has_errors:
            raise aggregator.to_error()
    """
    
    def __init__(self):
        """Initialize error aggregator."""
        self._errors: list[ForgeError] = []
    
    def add(self, error: Union[ForgeError, Exception, str]):
        """Add an error."""
        if isinstance(error, ForgeError):
            self._errors.append(error)
        elif isinstance(error, Exception):
            self._errors.append(ForgeError(str(error), cause=error))
        else:
            self._errors.append(ForgeError(str(error)))
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors collected."""
        return len(self._errors) > 0
    
    @property
    def count(self) -> int:
        """Get error count."""
        return len(self._errors)
    
    @property
    def errors(self) -> list[ForgeError]:
        """Get all errors."""
        return self._errors.copy()
    
    def clear(self):
        """Clear all errors."""
        self._errors.clear()
    
    def to_error(self, message: Optional[str] = None) -> ForgeError:
        """
        Convert to single aggregated error.
        
        Args:
            message: Optional custom message
            
        Returns:
            Aggregated ForgeError
        """
        if not self._errors:
            return ForgeError("No errors", severity=ErrorSeverity.INFO)
        
        if len(self._errors) == 1:
            return self._errors[0]
        
        msg = message or f"Multiple errors occurred ({len(self._errors)} total)"
        return ForgeError(
            message=msg,
            context=ErrorContext(related_errors=self._errors)
        )
    
    def to_result(self) -> Result[None]:
        """Convert to Result type."""
        if self.has_errors:
            return Result(_error=self.to_error())
        return Result.success(None)


def safe_call(
    fn: Callable[..., T],
    *args,
    default: Optional[T] = None,
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    **kwargs
) -> Result[T]:
    """
    Safely call a function and wrap result.
    
    Args:
        fn: Function to call
        *args: Positional arguments
        default: Default value on error (if set, returns success with default)
        error_code: Error code for failures
        **kwargs: Keyword arguments
        
    Returns:
        Result wrapping the function return or error
    """
    try:
        result = fn(*args, **kwargs)
        return Result.success(result)
    except ForgeError as e:
        if default is not None:
            return Result.success(default)
        return Result(_error=e)
    except Exception as e:
        if default is not None:
            return Result.success(default)
        return Result.failure(str(e), code=error_code, cause=e)


def handle_errors(
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    default: Optional[T] = None,
    reraise: bool = False
):
    """
    Decorator to handle errors in functions.
    
    Usage:
        @handle_errors(ErrorCode.MODEL_INFERENCE_FAILED, default="")
        def generate(prompt: str) -> str:
            # May raise exceptions
            return model.generate(prompt)
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., Union[T, Result[T]]]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Union[T, Result[T]]:
            try:
                return fn(*args, **kwargs)
            except ForgeError:
                if reraise:
                    raise
                if default is not None:
                    return default
                raise
            except Exception as e:
                error = ForgeError(str(e), code=error_code, cause=e)
                if reraise:
                    raise error from e
                if default is not None:
                    return default
                raise error from e
        
        return wrapper
    return decorator


def ensure(
    condition: bool,
    message: str,
    code: ErrorCode = ErrorCode.INVALID_STATE
):
    """
    Ensure a condition is true or raise error.
    
    Args:
        condition: Condition to check
        message: Error message if condition is false
        code: Error code
        
    Raises:
        ForgeError: If condition is false
    """
    if not condition:
        raise ForgeError(message, code=code)


def ensure_not_none(
    value: Optional[T],
    message: str = "Value cannot be None",
    code: ErrorCode = ErrorCode.INVALID_INPUT
) -> T:
    """
    Ensure a value is not None.
    
    Args:
        value: Value to check
        message: Error message
        code: Error code
        
    Returns:
        The value if not None
        
    Raises:
        ForgeError: If value is None
    """
    if value is None:
        raise ForgeError(message, code=code)
    return value


def from_tool_result(result: Dict[str, Any], value_key: str = "result") -> Result[Any]:
    """
    Convert a tool result dict to a Result type.
    
    Tool functions return {"success": bool, "error": str, ...} dicts.
    This converts them to proper Result types for chaining.
    
    Args:
        result: Tool result dictionary
        value_key: Key containing the value (default: "result")
        
    Returns:
        Result wrapping success value or error
        
    Example:
        result = execute_tool("read_file", path="test.txt")
        r = from_tool_result(result, "content")
        if r.is_success:
            print(r.value)
    """
    if result.get("success", False):
        # Extract value - try value_key first, then common alternatives
        value = result.get(value_key)
        if value is None:
            for key in ["content", "data", "output", "results", "value"]:
                if key in result:
                    value = result[key]
                    break
        if value is None:
            # Return the whole dict minus success/error keys
            value = {k: v for k, v in result.items() if k not in ("success", "error")}
        return Result.success(value)
    else:
        error_msg = result.get("error", "Unknown error")
        return Result.failure(error_msg, code=ErrorCode.TOOL_EXECUTION_FAILED)


def as_result(
    error_code: ErrorCode = ErrorCode.UNKNOWN,
    value_key: str = "result"
) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Result[Any]]]:
    """
    Decorator to convert dict-returning functions to Result-returning.
    
    Use this to wrap tool execute methods for functional-style chaining.
    
    Args:
        error_code: Error code for failures
        value_key: Key containing the value
        
    Returns:
        Decorator function
        
    Example:
        @as_result(ErrorCode.FILE_READ_ERROR, "content")
        def read_file(path: str) -> dict:
            return {"success": True, "content": "..."}
    """
    def decorator(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Result[Any]]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Result[Any]:
            try:
                result = fn(*args, **kwargs)
                return from_tool_result(result, value_key)
            except ForgeError as e:
                return Result(_error=e)
            except Exception as e:
                return Result.failure(str(e), code=error_code, cause=e)
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each attempt
        exceptions: Exception types to catch
        
    Returns:
        Decorator function
        
    Example:
        @retry(max_attempts=3, delay=1.0)
        def fetch_data(url: str) -> str:
            return requests.get(url).text
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            # All attempts failed
            if isinstance(last_error, ForgeError):
                raise last_error
            raise ForgeError(
                f"Failed after {max_attempts} attempts: {last_error}",
                code=ErrorCode.UNKNOWN,
                cause=last_error
            )
        return wrapper
    return decorator
