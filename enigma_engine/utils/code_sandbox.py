"""
Code Sandbox Execution for Enigma AI Engine

Safe execution of AI-generated code in isolated environments.

Features:
- Process isolation
- Resource limits
- Output capture
- Timeout handling
- Import restrictions

Usage:
    from enigma_engine.utils.code_sandbox import CodeSandbox, get_sandbox
    
    sandbox = get_sandbox()
    
    # Execute code safely
    result = sandbox.execute("print('Hello')")
    print(result.output)  # "Hello\n"
"""

import ast
import io
import logging
import multiprocessing
import time
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Dangerous modules that should be blocked
BLOCKED_MODULES = {
    "os", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "ftplib", "smtplib", "telnetlib",
    "pickle", "marshal", "shelve",
    "ctypes", "multiprocessing", "threading",
    "importlib", "builtins", "__builtins__",
    "sys", "inspect", "gc", "code", "codeop",
    "pty", "tty", "termios", "fcntl",
    "signal", "resource", "mmap",
    "webbrowser", "antigravity"
}

# Dangerous builtins
BLOCKED_BUILTINS = {
    "eval", "exec", "compile", "__import__",
    "open", "input", "breakpoint",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "hasattr",
    "memoryview", "type", "object"
}

# Safe builtins to allow
SAFE_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool",
    "bytearray", "bytes", "callable", "chr",
    "classmethod", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format",
    "frozenset", "hash", "hex", "id", "int",
    "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "oct",
    "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted",
    "staticmethod", "str", "sum", "tuple", "zip"
}


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str = ""
    error: Optional[str] = None
    return_value: Any = None
    
    # Execution info
    execution_time: float = 0.0
    memory_used: int = 0
    
    # Safety info
    blocked_imports: List[str] = field(default_factory=list)
    blocked_calls: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "blocked_imports": self.blocked_imports,
            "blocked_calls": self.blocked_calls
        }


@dataclass
class SandboxConfig:
    """Configuration for sandbox."""
    # Timeouts
    timeout_seconds: float = 5.0
    
    # Resource limits
    max_memory_mb: int = 100
    max_output_length: int = 10000
    
    # Security
    allow_imports: bool = False
    allowed_modules: Set[str] = field(default_factory=lambda: {"math", "random", "datetime", "json", "re", "collections", "itertools", "functools"})
    
    # Execution
    restrict_builtins: bool = True
    capture_return: bool = True


class ImportBlocker:
    """Blocks dangerous imports during execution."""
    
    def __init__(self, allowed: Set[str]):
        self._allowed = allowed
        self._blocked: List[str] = []
    
    def find_module(self, name: str, path=None):
        """Block non-allowed imports."""
        if name.split(".")[0] in self._allowed:
            return None  # Allow normal import
        
        self._blocked.append(name)
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    
    @property
    def blocked(self) -> List[str]:
        return self._blocked


class CodeAnalyzer:
    """Static analysis of code for safety."""
    
    def __init__(self, config: SandboxConfig):
        self._config = config
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Analyze code for potential issues.
        
        Returns dict with:
        - is_safe: bool
        - issues: List of issues found
        - imports: List of imports
        - calls: List of function calls
        """
        result = {
            "is_safe": True,
            "issues": [],
            "imports": [],
            "calls": []
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result["is_safe"] = False
            result["issues"].append(f"Syntax error: {e}")
            return result
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.names[0].name if isinstance(node, ast.Import) else node.module
                result["imports"].append(module)
                
                if module and module.split(".")[0] in BLOCKED_MODULES:
                    result["is_safe"] = False
                    result["issues"].append(f"Blocked import: {module}")
            
            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    result["calls"].append(name)
                    
                    if name in BLOCKED_BUILTINS:
                        result["is_safe"] = False
                        result["issues"].append(f"Blocked builtin: {name}")
                
                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    result["calls"].append(attr)
            
            # Check for exec/eval patterns
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    if node.value.func.id in ("exec", "eval"):
                        result["is_safe"] = False
                        result["issues"].append("Direct exec/eval call detected")
        
        return result


def _execute_in_process(
    code: str,
    config_dict: Dict[str, Any],
    result_queue: multiprocessing.Queue
):
    """Execute code in isolated process."""
    config = SandboxConfig(**config_dict)
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Build restricted globals
    restricted_globals = {
        "__name__": "__main__",
        "__doc__": None
    }
    
    if config.restrict_builtins:
        safe_builtins = {
            name: getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, name, None)
            for name in SAFE_BUILTINS
        }
        safe_builtins["__build_class__"] = __builtins__.__build_class__ if hasattr(__builtins__, "__build_class__") else __build_class__
        restricted_globals["__builtins__"] = safe_builtins
    else:
        restricted_globals["__builtins__"] = __builtins__
    
    # Add allowed imports
    if config.allow_imports:
        for module_name in config.allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass  # Intentionally silent
    
    result = ExecutionResult(success=False)
    
    start_time = time.time()
    
    try:
        # Redirect output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, "<sandbox>", "exec"), restricted_globals)
        
        result.success = True
        result.output = stdout_capture.getvalue()[:config.max_output_length]
        
        if stderr_capture.getvalue():
            result.output += "\n[stderr]\n" + stderr_capture.getvalue()[:1000]
            
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {str(e)}"
        result.output = stdout_capture.getvalue()
    
    result.execution_time = time.time() - start_time
    
    result_queue.put(result)


class CodeSandbox:
    """Sandbox for safe code execution."""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox.
        
        Args:
            config: Sandbox configuration
        """
        self._config = config or SandboxConfig()
        self._analyzer = CodeAnalyzer(self._config)
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code without executing."""
        return self._analyzer.analyze(code)
    
    def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        pre_check: bool = True
    ) -> ExecutionResult:
        """
        Execute code safely.
        
        Args:
            code: Python code to execute
            timeout: Override timeout (seconds)
            pre_check: Run static analysis first
            
        Returns:
            ExecutionResult with output and status
        """
        timeout = timeout or self._config.timeout_seconds
        
        # Pre-execution analysis
        if pre_check:
            analysis = self.analyze(code)
            if not analysis["is_safe"]:
                return ExecutionResult(
                    success=False,
                    error="Code failed safety check",
                    blocked_imports=analysis.get("imports", []),
                    blocked_calls=[c for c in analysis.get("calls", []) if c in BLOCKED_BUILTINS]
                )
        
        # Execute in isolated process
        result_queue = multiprocessing.Queue()
        
        config_dict = {
            "timeout_seconds": self._config.timeout_seconds,
            "max_memory_mb": self._config.max_memory_mb,
            "max_output_length": self._config.max_output_length,
            "allow_imports": self._config.allow_imports,
            "allowed_modules": set(self._config.allowed_modules),
            "restrict_builtins": self._config.restrict_builtins,
            "capture_return": self._config.capture_return
        }
        
        process = multiprocessing.Process(
            target=_execute_in_process,
            args=(code, config_dict, result_queue)
        )
        
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            
            if process.is_alive():
                process.kill()
            
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
                execution_time=timeout
            )
        
        try:
            result = result_queue.get_nowait()
            return result
        except Exception:
            return ExecutionResult(
                success=False,
                error="Failed to retrieve execution result"
            )
    
    def execute_simple(self, code: str) -> str:
        """Simple execution returning output string."""
        result = self.execute(code)
        
        if result.success:
            return result.output
        else:
            return f"Error: {result.error}"
    
    def test_code(self, code: str, expected: str) -> bool:
        """Test if code produces expected output."""
        result = self.execute(code)
        return result.success and result.output.strip() == expected.strip()


class REPLSandbox:
    """Interactive REPL sandbox."""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self._sandbox = CodeSandbox(config)
        self._history: List[str] = []
        self._namespace: Dict[str, Any] = {}
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute code in REPL context."""
        self._history.append(code)
        return self._sandbox.execute(code)
    
    def get_history(self) -> List[str]:
        """Get execution history."""
        return self._history.copy()
    
    def clear_history(self):
        """Clear history."""
        self._history.clear()
        self._namespace.clear()


# Global instance
_sandbox: Optional[CodeSandbox] = None


def get_sandbox() -> CodeSandbox:
    """Get or create global sandbox."""
    global _sandbox
    if _sandbox is None:
        _sandbox = CodeSandbox()
    return _sandbox
