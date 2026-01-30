"""
Error Handler - Graceful error handling with fallback responses

Provides friendly error messages and recovery suggestions for common errors.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Central error handling system with graceful fallback responses.
    
    Features:
    - User-friendly error messages
    - Recovery suggestions
    - Error categorization
    - Fallback responses
    """
    
    # Error categories with friendly messages
    ERROR_MESSAGES = {
        # File errors
        "FileNotFoundError": {
            "message": "I couldn't find that file.",
            "suggestions": [
                "Check if the file path is correct",
                "Make sure the file exists",
                "Try using the absolute path"
            ]
        },
        "PermissionError": {
            "message": "I don't have permission to access that file.",
            "suggestions": [
                "Check file permissions",
                "Try running with appropriate privileges",
                "Make sure the file isn't locked by another program"
            ]
        },
        "IsADirectoryError": {
            "message": "That's a directory, not a file.",
            "suggestions": [
                "Specify a file within the directory",
                "Use the list_directory tool to see files",
            ]
        },
        
        # Format errors
        "UnsupportedFileFormat": {
            "message": "I don't support that file format.",
            "suggestions": [
                "Supported formats: PDF, TXT, DOCX, EPUB, MD, HTML",
                "Try converting the file to a supported format",
                "Use extract_text for unknown formats"
            ]
        },
        "JSONDecodeError": {
            "message": "I couldn't parse that JSON data.",
            "suggestions": [
                "Check for syntax errors in the JSON",
                "Make sure quotes and brackets are balanced",
                "Use a JSON validator to check the format"
            ]
        },
        
        # Network errors
        "ConnectionError": {
            "message": "I couldn't connect to that service.",
            "suggestions": [
                "Check your internet connection",
                "The service might be down",
                "Try again in a moment"
            ]
        },
        "TimeoutError": {
            "message": "The request took too long.",
            "suggestions": [
                "The service might be slow right now",
                "Try again with a shorter request",
                "Check your internet connection"
            ]
        },
        
        # Module errors
        "ModuleNotFoundError": {
            "message": "A required module is missing.",
            "suggestions": [
                "Install the required package",
                "Check requirements.txt for dependencies",
                "Try: pip install -r requirements.txt"
            ]
        },
        "ImportError": {
            "message": "I couldn't load a required module.",
            "suggestions": [
                "Make sure all dependencies are installed",
                "Try reinstalling the package",
                "Check if the module is compatible with your Python version"
            ]
        },
        
        # Memory/Resource errors
        "MemoryError": {
            "message": "I ran out of memory.",
            "suggestions": [
                "Try processing a smaller amount of data",
                "Close other applications to free up memory",
                "Use a smaller model size"
            ]
        },
        "OSError": {
            "message": "A system error occurred.",
            "suggestions": [
                "Check available disk space",
                "Ensure you have necessary permissions",
                "Try restarting the application"
            ]
        },
        
        # Model errors
        "ModelNotFoundError": {
            "message": "The AI model couldn't be found.",
            "suggestions": [
                "Train a model first: python run.py --train",
                "Or use the GUI to train a model",
                "Check if the model file exists in models/"
            ]
        },
        "ModelLoadError": {
            "message": "I couldn't load the AI model.",
            "suggestions": [
                "The model file might be corrupted",
                "Try retraining: python run.py --train --force",
                "Check if you have enough memory"
            ]
        },
        
        # Input errors
        "ValueError": {
            "message": "The input value doesn't look right.",
            "suggestions": [
                "Check the format of your input",
                "Make sure numbers are valid",
                "Verify dates are in the correct format"
            ]
        },
        "TypeError": {
            "message": "The input type is incorrect.",
            "suggestions": [
                "Check if you're using the right type of input",
                "Numbers should be numbers, text should be text",
                "Check the tool documentation for expected types"
            ]
        },
        
        # Generic fallback
        "Unknown": {
            "message": "Something unexpected happened.",
            "suggestions": [
                "Try your request again",
                "Simplify your request",
                "Check the logs for more details"
            ]
        }
    }
    
    @classmethod
    def handle_error(cls, error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle an error and return a user-friendly response.
        
        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
            
        Returns:
            Dictionary with error info and recovery suggestions
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Get friendly message for this error type
        error_info = cls.ERROR_MESSAGES.get(
            error_type,
            cls.ERROR_MESSAGES["Unknown"]
        )
        
        # Log the full error
        logger.error(f"Error in {context or 'unknown context'}: {error_type}: {error_msg}")
        logger.debug(traceback.format_exc())
        
        return {
            "success": False,
            "error_type": error_type,
            "error_message": error_msg,
            "friendly_message": error_info["message"],
            "suggestions": error_info["suggestions"],
            "context": context
        }
    
    @classmethod
    def wrap_function(cls, func: Callable, context: Optional[str] = None) -> Callable:
        """
        Wrap a function with error handling.
        
        Args:
            func: Function to wrap
            context: Context description for error messages
            
        Returns:
            Wrapped function that returns error dict on exception
        """
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return cls.handle_error(e, context or func.__name__)
        
        return wrapped
    
    @classmethod
    def get_fallback_response(cls, error_type: str) -> str:
        """
        Get a fallback response for a given error type.
        
        Args:
            error_type: The type of error
            
        Returns:
            A friendly fallback message
        """
        error_info = cls.ERROR_MESSAGES.get(error_type, cls.ERROR_MESSAGES["Unknown"])
        message = error_info["message"]
        suggestions = error_info["suggestions"]
        
        response = f"{message}\n\nHere's what you can try:\n"
        for i, suggestion in enumerate(suggestions, 1):
            response += f"{i}. {suggestion}\n"
        
        return response


class GracefulFileHandler:
    """Handle file operations with graceful error recovery."""
    
    @staticmethod
    def read_file(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Read a file with error handling.
        
        Args:
            path: Path to the file
            encoding: File encoding
            
        Returns:
            Dict with success status and content or error info
        """
        try:
            path_obj = Path(path).expanduser().resolve()
            
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if path_obj.is_dir():
                raise IsADirectoryError(f"Expected a file, got directory: {path}")
            
            # Check file size
            size = path_obj.stat().st_size
            if size > 100 * 1024 * 1024:  # 100MB
                return {
                    "success": False,
                    "error": "File too large",
                    "friendly_message": "That file is too large to read.",
                    "suggestions": [
                        f"File size: {size / (1024*1024):.1f} MB (limit: 100 MB)",
                        "Try reading a smaller file",
                        "Use a streaming approach for large files"
                    ]
                }
            
            with open(path_obj, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "size": size,
                "path": str(path_obj)
            }
            
        except Exception as e:
            return ErrorHandler.handle_error(e, f"reading file: {path}")
    
    @staticmethod
    def write_file(path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """
        Write to a file with error handling.
        
        Args:
            path: Path to write to
            content: Content to write
            mode: Write mode ('w' or 'a')
            
        Returns:
            Dict with success status and info or error
        """
        try:
            path_obj = Path(path).expanduser().resolve()
            
            # Create parent directories
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path_obj, mode, encoding="utf-8") as f:
                f.write(content)
            
            return {
                "success": True,
                "path": str(path_obj),
                "bytes_written": len(content.encode('utf-8'))
            }
            
        except Exception as e:
            return ErrorHandler.handle_error(e, f"writing file: {path}")


class GracefulFormatHandler:
    """Handle format conversions with error recovery."""
    
    SUPPORTED_FORMATS = {
        'document': ['.txt', '.md', '.pdf', '.docx', '.epub', '.html', '.htm'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
        'data': ['.json', '.csv', '.xml', '.yaml', '.yml']
    }
    
    @staticmethod
    def check_format(path: str, category: str = 'document') -> Dict[str, Any]:
        """
        Check if a file format is supported.
        
        Args:
            path: Path to the file
            category: Format category ('document', 'image', 'data')
            
        Returns:
            Dict with support status and info
        """
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        
        supported = GracefulFormatHandler.SUPPORTED_FORMATS.get(category, [])
        
        if ext in supported:
            return {
                "success": True,
                "format": ext,
                "category": category,
                "supported": True
            }
        else:
            return {
                "success": False,
                "format": ext,
                "category": category,
                "supported": False,
                "friendly_message": f"I don't support {ext} files in the {category} category.",
                "suggestions": [
                    f"Supported {category} formats: {', '.join(supported)}",
                    "Try converting to a supported format",
                    "Or try extract_text for basic text extraction"
                ]
            }


# Decorator for automatic error handling
def graceful_errors(context: Optional[str] = None):
    """
    Decorator to add graceful error handling to functions.
    
    Usage:
        @graceful_errors("processing user input")
        def my_function(arg1, arg2):
            # function code
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return ErrorHandler.handle_error(e, context or func.__name__)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test error handling
    import json
    
    # Test file not found
    result = GracefulFileHandler.read_file("/nonexistent/file.txt")
    print("File not found error:")
    print(json.dumps(result, indent=2))
    
    # Test format checking
    result = GracefulFormatHandler.check_format("test.xyz", "document")
    print("\nUnsupported format:")
    print(json.dumps(result, indent=2))
    
    # Test decorator
    @graceful_errors("test function")
    def test_function():
        raise ValueError("Invalid input value")
    
    result = test_function()
    print("\nDecorator test:")
    print(json.dumps(result, indent=2))
    
    # Test fallback response
    print("\nFallback response:")
    print(ErrorHandler.get_fallback_response("FileNotFoundError"))
