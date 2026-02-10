"""
Tool Permission System
======================

Controls access to tools based on permission levels and user confirmation.
Prevents unauthorized or accidental execution of destructive operations.
"""

import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels for tool execution."""
    
    NONE = 0        # No access
    READ = 1        # Read-only operations
    WRITE = 2       # Write operations (create, modify files)
    EXECUTE = 3     # Execute commands, run processes
    ADMIN = 4       # Administrative operations (delete, system changes)


# Default permission requirements per tool
DEFAULT_TOOL_PERMISSIONS = {
    # Read-only tools
    "read_file": PermissionLevel.READ,
    "list_directory": PermissionLevel.READ,
    "get_system_info": PermissionLevel.READ,
    "web_search": PermissionLevel.READ,
    "fetch_webpage": PermissionLevel.READ,
    "read_document": PermissionLevel.READ,
    "extract_text": PermissionLevel.READ,
    "analyze_image": PermissionLevel.READ,
    "find_on_screen": PermissionLevel.READ,
    "list_modules": PermissionLevel.READ,
    "check_resources": PermissionLevel.READ,
    
    # Write operations
    "write_file": PermissionLevel.WRITE,
    "generate_image": PermissionLevel.WRITE,
    "generate_video": PermissionLevel.WRITE,
    "generate_audio": PermissionLevel.WRITE,
    "generate_code": PermissionLevel.WRITE,
    "generate_gif": PermissionLevel.WRITE,
    "edit_image": PermissionLevel.WRITE,
    "edit_gif": PermissionLevel.WRITE,
    "edit_video": PermissionLevel.WRITE,
    "speak": PermissionLevel.WRITE,
    "control_avatar": PermissionLevel.WRITE,
    
    # Execute operations
    "run_command": PermissionLevel.EXECUTE,
    "load_module": PermissionLevel.EXECUTE,
    "unload_module": PermissionLevel.EXECUTE,
    
    # Admin operations (destructive)
    "delete_file": PermissionLevel.ADMIN,
    "move_file": PermissionLevel.ADMIN,
}


# Tools that require user confirmation
CONFIRMATION_REQUIRED_TOOLS = {
    "delete_file",
    "move_file",
    "run_command",
}


class ToolPermissionManager:
    """
    Manage tool execution permissions.
    
    Features:
    - Permission level checking
    - User confirmation for destructive operations
    - Customizable confirmation callbacks
    - Permission auditing
    """
    
    def __init__(
        self,
        user_permission_level: PermissionLevel = PermissionLevel.WRITE,
        confirmation_callback: Optional[Callable[[str, dict], bool]] = None,
        auto_approve: bool = False
    ):
        """
        Initialize permission manager.
        
        Args:
            user_permission_level: Maximum permission level for user
            confirmation_callback: Function to request user confirmation
            auto_approve: Auto-approve all confirmations (dangerous!)
        """
        self.user_permission_level = user_permission_level
        self.confirmation_callback = confirmation_callback
        self.auto_approve = auto_approve
        
        # Custom permission overrides
        self.permission_overrides: dict[str, PermissionLevel] = {}
        
        # Tools that are explicitly blocked
        self.blocked_tools: set[str] = set()
        
        # Audit log
        self.denied_attempts: list = []
        
        logger.info(f"ToolPermissionManager initialized (level={user_permission_level.name})")
    
    def can_execute(self, tool_name: str, params: Optional[dict] = None) -> tuple[bool, Optional[str]]:
        """
        Check if user can execute a tool.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters (for logging)
            
        Returns:
            (allowed, reason) - True if allowed, False with reason if not
        """
        # Check if explicitly blocked
        if tool_name in self.blocked_tools:
            reason = f"Tool '{tool_name}' is blocked"
            self._log_denial(tool_name, params, reason)
            return False, reason
        
        # Get required permission level
        if tool_name in self.permission_overrides:
            required_level = self.permission_overrides[tool_name]
        elif tool_name in DEFAULT_TOOL_PERMISSIONS:
            required_level = DEFAULT_TOOL_PERMISSIONS[tool_name]
        else:
            # Unknown tool, default to WRITE
            required_level = PermissionLevel.WRITE
        
        # Check permission level
        if self.user_permission_level.value < required_level.value:
            reason = (
                f"Tool '{tool_name}' requires {required_level.name} permission, "
                f"but user has {self.user_permission_level.name}"
            )
            self._log_denial(tool_name, params, reason)
            return False, reason
        
        return True, None
    
    def requires_confirmation(self, tool_name: str) -> bool:
        """
        Check if tool requires user confirmation.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if confirmation required
        """
        return tool_name in CONFIRMATION_REQUIRED_TOOLS
    
    def request_confirmation(
        self,
        tool_name: str,
        params: dict
    ) -> bool:
        """
        Request user confirmation for tool execution.
        
        Args:
            tool_name: Name of the tool
            params: Tool parameters
            
        Returns:
            True if confirmed, False if denied
        """
        if self.auto_approve:
            logger.warning(f"Auto-approving {tool_name} (auto_approve=True)")
            return True
        
        if not self.requires_confirmation(tool_name):
            return True
        
        # Use callback if provided
        if self.confirmation_callback:
            try:
                return self.confirmation_callback(tool_name, params)
            except Exception as e:
                logger.error(f"Error in confirmation callback: {e}")
                return False
        
        # Default: deny if no callback
        logger.warning(f"Confirmation required for {tool_name} but no callback provided - denied")
        return False
    
    def _log_denial(self, tool_name: str, params: Optional[dict], reason: str):
        """Log a denied tool execution attempt."""
        from datetime import datetime
        
        self.denied_attempts.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "params": params,
            "reason": reason,
        })
        
        logger.warning(f"Tool execution denied: {reason}")
    
    def block_tool(self, tool_name: str):
        """
        Block a tool from execution.
        
        Args:
            tool_name: Name of the tool to block
        """
        self.blocked_tools.add(tool_name)
        logger.info(f"Blocked tool: {tool_name}")
    
    def unblock_tool(self, tool_name: str):
        """
        Unblock a tool.
        
        Args:
            tool_name: Name of the tool to unblock
        """
        self.blocked_tools.discard(tool_name)
        logger.info(f"Unblocked tool: {tool_name}")
    
    def set_tool_permission(self, tool_name: str, level: PermissionLevel):
        """
        Set custom permission level for a tool.
        
        Args:
            tool_name: Name of the tool
            level: Required permission level
        """
        self.permission_overrides[tool_name] = level
        logger.info(f"Set permission for {tool_name}: {level.name}")
    
    def get_available_tools(self) -> dict[str, PermissionLevel]:
        """
        Get all tools available at current permission level.
        
        Returns:
            Dictionary of tool_name -> required_permission
        """
        available = {}
        
        all_tools = {**DEFAULT_TOOL_PERMISSIONS, **self.permission_overrides}
        
        for tool_name, required_level in all_tools.items():
            if tool_name in self.blocked_tools:
                continue
            
            if self.user_permission_level.value >= required_level.value:
                available[tool_name] = required_level
        
        return available
    
    def get_statistics(self) -> dict:
        """Get permission system statistics."""
        all_tools = {**DEFAULT_TOOL_PERMISSIONS, **self.permission_overrides}
        
        available = self.get_available_tools()
        
        return {
            "user_permission_level": self.user_permission_level.name,
            "total_tools": len(all_tools),
            "available_tools": len(available),
            "blocked_tools": len(self.blocked_tools),
            "denied_attempts": len(self.denied_attempts),
            "auto_approve_enabled": self.auto_approve,
        }


# Default confirmation callback (console-based)
def default_confirmation_callback(tool_name: str, params: dict) -> bool:
    """
    Default confirmation callback using console input.
    
    Args:
        tool_name: Name of the tool
        params: Tool parameters
        
    Returns:
        True if confirmed, False otherwise
    """
    print(f"\n[CONFIRMATION REQUIRED]")
    print(f"Tool: {tool_name}")
    print(f"Parameters: {params}")
    
    response = input("Allow execution? (yes/no): ").strip().lower()
    
    return response in ("yes", "y")


__all__ = [
    "PermissionLevel",
    "ToolPermissionManager",
    "DEFAULT_TOOL_PERMISSIONS",
    "CONFIRMATION_REQUIRED_TOOLS",
    "default_confirmation_callback",
]
