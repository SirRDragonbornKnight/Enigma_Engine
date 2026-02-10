"""
Result Types for Fallible Operations
=====================================

Rust-inspired Result type for functional error handling.
Avoids exceptions for expected failure cases, making code flow clearer.

Usage:
    from enigma_engine.utils.result import Result, Ok, Err
    
    def divide(a: float, b: float) -> Result[float, str]:
        if b == 0:
            return Err("Division by zero")
        return Ok(a / b)
    
    result = divide(10, 2)
    if result.is_ok():
        print(f"Result: {result.unwrap()}")
    else:
        print(f"Error: {result.unwrap_err()}")
    
    # Pattern matching style
    match result:
        case Ok(value):
            print(f"Got {value}")
        case Err(error):
            print(f"Failed: {error}")
    
    # Chaining operations
    result = (
        Ok(10)
        .map(lambda x: x * 2)
        .and_then(lambda x: Ok(x + 5) if x < 100 else Err("Too large"))
        .unwrap_or(0)
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    NoReturn,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")  # Success value type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Mapped value type
F = TypeVar("F")  # Mapped error type


class ResultError(Exception):
    """Raised when unwrapping a Result fails."""


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """
    Represents a successful result.
    
    Usage:
        result = Ok(42)
        value = result.unwrap()  # 42
    """
    _value: T
    
    def is_ok(self) -> bool:
        return True
    
    def is_err(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        """Get the success value."""
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        """Get the success value, or a default if Err."""
        return self._value
    
    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Get the success value, or compute a default if Err."""
        return self._value
    
    def unwrap_err(self) -> NoReturn:
        """Get the error value. Raises for Ok."""
        raise ResultError(f"Called unwrap_err on Ok({self._value!r})")
    
    def expect(self, msg: str) -> T:
        """Get the success value, with custom error message if Err."""
        return self._value
    
    def expect_err(self, msg: str) -> NoReturn:
        """Get the error value, with custom error message if Ok."""
        raise ResultError(msg)
    
    def map(self, f: Callable[[T], U]) -> Result[U, Any]:
        """Transform the success value."""
        return Ok(f(self._value))
    
    def map_err(self, f: Callable[[Any], F]) -> Result[T, F]:
        """Transform the error value (no-op for Ok)."""
        return self  # type: ignore
    
    def and_then(self, f: Callable[[T], Result[U, Any]]) -> Result[U, Any]:
        """Chain another operation that may fail."""
        return f(self._value)
    
    def or_else(self, f: Callable[[Any], Result[T, F]]) -> Result[T, F]:
        """Provide an alternative if Err (no-op for Ok)."""
        return self  # type: ignore
    
    def ok(self) -> Optional[T]:
        """Convert to Optional, discarding error."""
        return self._value
    
    def err(self) -> None:
        """Get the error as Optional."""
        return None
    
    def __repr__(self) -> str:
        return f"Ok({self._value!r})"
    
    def __bool__(self) -> bool:
        """Ok is truthy."""
        return True
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Ok):
            return self._value == other._value
        return False
    
    def __hash__(self) -> int:
        return hash(("Ok", self._value))


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """
    Represents a failed result.
    
    Usage:
        result = Err("File not found")
        error = result.unwrap_err()  # "File not found"
    """
    _error: E
    
    def is_ok(self) -> bool:
        return False
    
    def is_err(self) -> bool:
        return True
    
    def unwrap(self) -> NoReturn:
        """Get the success value. Raises for Err."""
        raise ResultError(f"Called unwrap on Err({self._error!r})")
    
    def unwrap_or(self, default: T) -> T:
        """Get the success value, or a default if Err."""
        return default
    
    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Get the success value, or compute a default if Err."""
        return f()
    
    def unwrap_err(self) -> E:
        """Get the error value."""
        return self._error
    
    def expect(self, msg: str) -> NoReturn:
        """Get the success value. Raises with custom message for Err."""
        raise ResultError(f"{msg}: {self._error!r}")
    
    def expect_err(self, msg: str) -> E:
        """Get the error value, with custom error message if Ok."""
        return self._error
    
    def map(self, f: Callable[[Any], U]) -> Result[U, E]:
        """Transform the success value (no-op for Err)."""
        return self  # type: ignore
    
    def map_err(self, f: Callable[[E], F]) -> Result[Any, F]:
        """Transform the error value."""
        return Err(f(self._error))
    
    def and_then(self, f: Callable[[Any], Result[U, E]]) -> Result[U, E]:
        """Chain another operation (no-op for Err)."""
        return self  # type: ignore
    
    def or_else(self, f: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Provide an alternative if Err."""
        return f(self._error)
    
    def ok(self) -> None:
        """Convert to Optional, discarding error."""
        return None
    
    def err(self) -> Optional[E]:
        """Get the error as Optional."""
        return self._error
    
    def __repr__(self) -> str:
        return f"Err({self._error!r})"
    
    def __bool__(self) -> bool:
        """Err is falsy."""
        return False
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Err):
            return self._error == other._error
        return False
    
    def __hash__(self) -> int:
        return hash(("Err", self._error))


# Type alias for Result
Result = Union[Ok[T], Err[E]]


# ===== Helper Functions =====

def try_call(f: Callable[[], T], error_type: type = Exception) -> Result[T, Exception]:
    """
    Try to call a function, catching exceptions.
    
    Usage:
        result = try_call(lambda: risky_operation())
        if result.is_ok():
            print(result.unwrap())
    """
    try:
        return Ok(f())
    except error_type as e:
        return Err(e)


def try_await(coro, error_type: type = Exception):
    """
    Async version of try_call.
    
    Usage:
        result = await try_await(async_risky_operation())
    """
    async def _wrapper():
        try:
            return Ok(await coro)
        except error_type as e:
            return Err(e)
    return _wrapper()


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """
    Collect a list of Results into a Result of list.
    
    Returns Ok(list of values) if all are Ok, or first Err encountered.
    
    Usage:
        results = [Ok(1), Ok(2), Ok(3)]
        combined = collect_results(results)  # Ok([1, 2, 3])
        
        results = [Ok(1), Err("fail"), Ok(3)]
        combined = collect_results(results)  # Err("fail")
    """
    values = []
    for result in results:
        if result.is_err():
            return result  # type: ignore
        values.append(result.unwrap())
    return Ok(values)


def partition_results(results: list[Result[T, E]]) -> tuple[list[T], list[E]]:
    """
    Partition a list of Results into successes and errors.
    
    Usage:
        results = [Ok(1), Err("a"), Ok(2), Err("b")]
        oks, errs = partition_results(results)
        # oks = [1, 2], errs = ["a", "b"]
    """
    oks = []
    errs = []
    for result in results:
        if result.is_ok():
            oks.append(result.unwrap())
        else:
            errs.append(result.unwrap_err())
    return oks, errs


def result_from_optional(value: Optional[T], error: E) -> Result[T, E]:
    """
    Convert an Optional to a Result.
    
    Usage:
        maybe_value = get_optional_value()
        result = result_from_optional(maybe_value, "Value was None")
    """
    if value is not None:
        return Ok(value)
    return Err(error)


__all__ = [
    "Result",
    "Ok",
    "Err",
    "ResultError",
    "try_call",
    "try_await",
    "collect_results",
    "partition_results",
    "result_from_optional",
]
