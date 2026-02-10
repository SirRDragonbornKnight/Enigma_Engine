"""
Module Error Messages

User-friendly error messages for module loading failures.
Maps technical errors to helpful suggestions.

Usage:
    from enigma_engine.modules.error_messages import get_friendly_error
    
    # In module manager:
    can_load, reason = self.can_load(module_id)
    if not can_load:
        friendly_msg = get_friendly_error(module_id, reason)
        # Display to user: friendly_msg
"""

from dataclasses import dataclass


@dataclass
class ErrorInfo:
    """User-friendly error information."""
    title: str
    message: str
    suggestion: str
    severity: str  # "error", "warning", "info"


# Error message templates by pattern
ERROR_PATTERNS: dict[str, ErrorInfo] = {
    "not registered": ErrorInfo(
        title="Module Not Found",
        message="This module doesn't exist or hasn't been installed.",
        suggestion="Check the module name or try reinstalling Enigma AI Engine.",
        severity="error"
    ),
    "cloud services": ErrorInfo(
        title="Cloud Service Blocked",
        message="This module uses cloud APIs but local-only mode is enabled.",
        suggestion="Go to Settings > Privacy and disable 'Local Only Mode' to use cloud services.",
        severity="warning"
    ),
    "requires gpu": ErrorInfo(
        title="GPU Required",
        message="This module needs a GPU but none was detected.",
        suggestion="Try using the API version (e.g., image_gen_api instead of image_gen_local) which runs in the cloud.",
        severity="error"
    ),
    "no.*gpu": ErrorInfo(
        title="GPU Required",
        message="This module needs a GPU but none was detected.",
        suggestion="Try using the API version (e.g., image_gen_api instead of image_gen_local) which runs in the cloud.",
        severity="error"
    ),
    "VRAM": ErrorInfo(
        title="Not Enough GPU Memory",
        message="Your GPU doesn't have enough memory for this module.",
        suggestion="Try a smaller model size, close other GPU applications, or use the API version.",
        severity="error"
    ),
    "RAM": ErrorInfo(
        title="Not Enough System Memory",
        message="Your system doesn't have enough RAM for this module.",
        suggestion="Try a smaller model size or close other applications to free memory.",
        severity="error"
    ),
    "conflicts with": ErrorInfo(
        title="Module Conflict",
        message="This module can't run together with another loaded module.",
        suggestion="Unload the conflicting module first, then try again.",
        severity="warning"
    ),
    "already provided by": ErrorInfo(
        title="Feature Already Active",
        message="Another module is already providing this capability.",
        suggestion="You can either keep the current module or unload it to use a different one.",
        severity="info"
    ),
    "not loaded": ErrorInfo(
        title="Missing Dependency",
        message="This module requires another module to be loaded first.",
        suggestion="Load the required module first, then try again.",
        severity="error"
    ),
}

# Module-specific friendly names
MODULE_NAMES: dict[str, str] = {
    "model": "AI Model",
    "tokenizer": "Text Tokenizer",
    "inference": "Inference Engine",
    "image_gen_local": "Image Generator (Local)",
    "image_gen_api": "Image Generator (Cloud)",
    "code_gen_local": "Code Generator (Local)",
    "code_gen_api": "Code Generator (Cloud)",
    "video_gen_local": "Video Generator (Local)",
    "video_gen_api": "Video Generator (Cloud)",
    "audio_gen_local": "Audio Generator (Local)",
    "audio_gen_api": "Audio Generator (Cloud)",
    "threed_gen_local": "3D Generator (Local)",
    "threed_gen_api": "3D Generator (Cloud)",
    "embedding_local": "Embeddings (Local)",
    "embedding_api": "Embeddings (Cloud)",
    "memory": "Conversation Memory",
    "voice_input": "Voice Input",
    "voice_output": "Text-to-Speech",
    "avatar": "Avatar Display",
    "vision": "Vision Analysis",
    "camera": "Camera Capture",
}


def get_module_display_name(module_id: str) -> str:
    """Get user-friendly name for a module."""
    return MODULE_NAMES.get(module_id, module_id.replace("_", " ").title())


def get_friendly_error(module_id: str, technical_reason: str) -> ErrorInfo:
    """
    Convert technical error message to user-friendly explanation.
    
    Args:
        module_id: The module that failed to load
        technical_reason: The raw error string from can_load()
        
    Returns:
        ErrorInfo with user-friendly explanation and suggestions
    """
    module_name = get_module_display_name(module_id)
    reason_lower = technical_reason.lower()
    
    # Match against known patterns
    for pattern, template in ERROR_PATTERNS.items():
        if pattern in reason_lower:
            # Customize the message with module name
            return ErrorInfo(
                title=template.title,
                message=f"{module_name}: {template.message}",
                suggestion=template.suggestion,
                severity=template.severity
            )
    
    # Default: return technical message with generic help
    return ErrorInfo(
        title="Module Error",
        message=f"{module_name} couldn't be loaded: {technical_reason}",
        suggestion="Check the logs for more details or try restarting Enigma AI Engine.",
        severity="error"
    )


def format_error_for_gui(error_info: ErrorInfo) -> str:
    """Format error for display in GUI (rich text)."""
    severity_colors = {
        "error": "#f38ba8",
        "warning": "#f9e2af",
        "info": "#89b4fa"
    }
    color = severity_colors.get(error_info.severity, "#cdd6f4")
    
    return f"""
<div style="margin: 10px; padding: 10px; border-left: 3px solid {color}; background: #1e1e2e;">
    <h3 style="color: {color}; margin: 0 0 8px 0;">{error_info.title}</h3>
    <p style="color: #cdd6f4; margin: 0 0 8px 0;">{error_info.message}</p>
    <p style="color: #a6adc8; font-size: 0.9em; margin: 0;">
        <strong>Tip:</strong> {error_info.suggestion}
    </p>
</div>
"""


def format_error_for_terminal(error_info: ErrorInfo) -> str:
    """Format error for terminal/CLI display."""
    severity_symbols = {
        "error": "[X]",
        "warning": "[!]",
        "info": "[i]"
    }
    symbol = severity_symbols.get(error_info.severity, "[?]")
    
    return f"""
{symbol} {error_info.title}
    {error_info.message}
    
    Tip: {error_info.suggestion}
"""


def get_dependency_chain(
    module_id: str, 
    module_classes: dict
) -> tuple[list, list]:
    """
    Get the dependency chain for a module.
    
    Returns:
        (required, optional) - Lists of module IDs
    """
    if module_id not in module_classes:
        return [], []
    
    info = module_classes[module_id].get_info()
    return list(info.requires), list(info.optional)


def suggest_load_order(
    target_module: str,
    module_classes: dict,
    loaded_modules: set
) -> list:
    """
    Suggest the order to load modules to reach the target.
    
    Args:
        target_module: Module you want to load
        module_classes: All available module classes
        loaded_modules: Currently loaded module IDs
        
    Returns:
        List of module IDs to load in order
    """
    if target_module not in module_classes:
        return []
    
    to_load = []
    visited = set()
    
    def visit(mod_id: str):
        if mod_id in visited or mod_id in loaded_modules:
            return
        visited.add(mod_id)
        
        if mod_id not in module_classes:
            return
        
        # Visit dependencies first
        info = module_classes[mod_id].get_info()
        for dep in info.requires:
            visit(dep)
        
        # Then add this module
        to_load.append(mod_id)
    
    visit(target_module)
    return to_load
