"""
Standardized Tool Result Classes for ForgeAI

Provides consistent result types for all tool operations,
enabling better error handling and type safety.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Any
from enum import Enum
import json


class ErrorCode(Enum):
    """Standardized error codes for tool operations."""
    # Success (no error)
    SUCCESS = "SUCCESS"
    
    # Input validation errors (400-level)
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    PARAMETER_OUT_OF_RANGE = "PARAMETER_OUT_OF_RANGE"
    
    # Permission/Security errors (403-level)
    PERMISSION_DENIED = "PERMISSION_DENIED"
    BLOCKED_PATH = "BLOCKED_PATH"
    BLOCKED_COMMAND = "BLOCKED_COMMAND"
    RATE_LIMITED = "RATE_LIMITED"
    
    # Resource errors (404/409-level)
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    RESOURCE_BUSY = "RESOURCE_BUSY"
    
    # Execution errors (500-level)
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    
    # Unknown
    UNKNOWN = "UNKNOWN"


T = TypeVar('T')


@dataclass
class ToolResult(Generic[T]):
    """
    Standardized result type for all tool operations.
    
    Provides consistent success/failure reporting with detailed
    error information and optional metadata.
    
    Usage:
        # Success case
        result = ToolResult.ok(data={"file": "test.txt"}, message="File created")
        
        # Failure case
        result = ToolResult.fail(
            error="File not found",
            error_code=ErrorCode.NOT_FOUND,
            details={"path": "/missing/file.txt"}
        )
        
        # Check result
        if result.success:
            print(result.data)
        else:
            print(f"Error [{result.error_code}]: {result.error}")
    """
    success: bool
    data: T | None = None
    error: str | None = None
    error_code: ErrorCode = ErrorCode.SUCCESS
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    
    @classmethod
    def ok(
        cls,
        data: T = None,
        message: str = None,
        **metadata
    ) -> 'ToolResult[T]':
        """
        Create a successful result.
        
        Args:
            data: The result data (any type)
            message: Optional success message
            **metadata: Additional metadata key-value pairs
        """
        return cls(
            success=True,
            data=data,
            message=message,
            metadata=metadata if metadata else {}
        )
    
    @classmethod
    def fail(
        cls,
        error: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN,
        details: dict[str, Any] | None = None,
        **metadata
    ) -> ToolResult[T]:
        """
        Create a failure result.
        
        Args:
            error: Human-readable error message
            error_code: Standardized error code
            details: Additional error details
            **metadata: Additional metadata
        """
        meta = metadata if metadata else {}
        if details:
            meta['error_details'] = details
        
        return cls(
            success=False,
            error=error,
            error_code=error_code,
            metadata=meta
        )
    
    def add_warning(self, warning: str) -> 'ToolResult[T]':
        """Add a warning to the result (returns self for chaining)."""
        self.warnings.append(warning)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This format is compatible with existing ForgeAI tool responses.
        """
        result = {'success': self.success}
        
        if self.success:
            if self.data is not None:
                # If data is a dict, merge it into result for backwards compatibility
                if isinstance(self.data, dict):
                    result.update(self.data)
                else:
                    result['data'] = self.data
            if self.message:
                result['message'] = self.message
        else:
            result['error'] = self.error
            result['error_code'] = self.error_code.value
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        if self.warnings:
            result['warnings'] = self.warnings
        
        return result
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success
    
    def unwrap(self) -> T:
        """
        Get the data, raising exception if failed.
        
        Raises:
            ValueError: If the result is a failure
        """
        if not self.success:
            raise ValueError(f"Cannot unwrap failed result: {self.error}")
        return self.data
    
    def unwrap_or(self, default: T) -> T:
        """Get the data or a default value if failed."""
        return self.data if self.success else default
    
    def map(self, func) -> 'ToolResult':
        """
        Apply a function to the data if successful.
        
        Args:
            func: Function to apply to the data
            
        Returns:
            New ToolResult with transformed data or the same error
        """
        if self.success:
            try:
                new_data = func(self.data)
                return ToolResult.ok(data=new_data, message=self.message, **self.metadata)
            except Exception as e:
                return ToolResult.fail(
                    error=str(e),
                    error_code=ErrorCode.EXECUTION_ERROR
                )
        return self


# Type aliases for common result types
FileResult = ToolResult[dict[str, Any]]
CommandResult = ToolResult[dict[str, Any]]
SearchResult = ToolResult[list[dict[str, Any]]]


def from_legacy_dict(result: dict[str, Any]) -> ToolResult:
    """
    Convert a legacy {"success": bool, ...} dict to ToolResult.
    
    For backwards compatibility with existing tool implementations.
    """
    if result.get('success', False):
        # Remove 'success' key and use rest as data
        data = {k: v for k, v in result.items() if k != 'success'}
        return ToolResult.ok(data=data)
    else:
        error = result.get('error', 'Unknown error')
        return ToolResult.fail(error=error)
