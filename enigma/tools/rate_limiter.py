"""
Rate Limiting for Tool Execution
=================================

Prevents abuse and manages API quotas by limiting tool execution rates.
"""

import time
import logging
from typing import Dict, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


# Default rate limits per tool (requests per minute)
DEFAULT_RATE_LIMITS = {
    "web_search": 10,        # 10 searches per minute
    "fetch_webpage": 20,     # 20 page fetches per minute
    "run_command": 5,        # 5 commands per minute
    "write_file": 30,        # 30 file writes per minute
    "delete_file": 10,       # 10 file deletes per minute
    "generate_image": 5,     # 5 image generations per minute
    "generate_video": 2,     # 2 video generations per minute
    "generate_audio": 5,     # 5 audio generations per minute
    "analyze_image": 10,     # 10 image analyses per minute
}


class RateLimiter:
    """
    Rate limiter for tool execution.
    
    Uses a sliding window algorithm to track requests and enforce limits.
    """
    
    def __init__(
        self,
        custom_limits: Optional[Dict[str, int]] = None,
        window_seconds: int = 60
    ):
        """
        Initialize rate limiter.
        
        Args:
            custom_limits: Custom rate limits (requests per window)
            window_seconds: Time window in seconds (default: 60)
        """
        self.limits = DEFAULT_RATE_LIMITS.copy()
        if custom_limits:
            self.limits.update(custom_limits)
        
        self.window_seconds = window_seconds
        
        # Track requests per tool: {tool_name: deque of timestamps}
        self.requests: Dict[str, deque] = {}
        
        # Track blocked attempts
        self.blocked_count: Dict[str, int] = {}
        
        logger.info(f"RateLimiter initialized with {len(self.limits)} tool limits")
    
    def _cleanup_old_requests(self, tool_name: str):
        """Remove requests older than the window."""
        if tool_name not in self.requests:
            return
        
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Remove old requests
        while self.requests[tool_name] and self.requests[tool_name][0] < cutoff_time:
            self.requests[tool_name].popleft()
    
    def is_allowed(self, tool_name: str) -> bool:
        """
        Check if a tool execution is allowed under rate limits.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if allowed, False if rate limited
        """
        # No limit configured for this tool
        if tool_name not in self.limits:
            return True
        
        # Initialize request queue if needed
        if tool_name not in self.requests:
            self.requests[tool_name] = deque()
        
        # Clean up old requests
        self._cleanup_old_requests(tool_name)
        
        # Check if under limit
        limit = self.limits[tool_name]
        current_count = len(self.requests[tool_name])
        
        if current_count >= limit:
            # Rate limited
            if tool_name not in self.blocked_count:
                self.blocked_count[tool_name] = 0
            self.blocked_count[tool_name] += 1
            
            logger.warning(
                f"Rate limit exceeded for {tool_name}: "
                f"{current_count}/{limit} requests in {self.window_seconds}s"
            )
            return False
        
        return True
    
    def record_request(self, tool_name: str):
        """
        Record a tool execution request.
        
        Args:
            tool_name: Name of the tool
        """
        if tool_name not in self.requests:
            self.requests[tool_name] = deque()
        
        self.requests[tool_name].append(time.time())
    
    def wait_if_needed(self, tool_name: str, timeout: float = 60.0) -> bool:
        """
        Wait until the tool can be executed, up to a timeout.
        
        Args:
            tool_name: Name of the tool
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tool can now be executed, False if timeout reached
        """
        start_time = time.time()
        
        while not self.is_allowed(tool_name):
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for rate limit on {tool_name}")
                return False
            
            # Calculate wait time
            wait_time = self.get_wait_time(tool_name)
            if wait_time > 0:
                sleep_time = min(wait_time, timeout - (time.time() - start_time))
                if sleep_time > 0:
                    logger.debug(f"Waiting {sleep_time:.1f}s for rate limit on {tool_name}")
                    time.sleep(sleep_time)
        
        return True
    
    def get_remaining(self, tool_name: str) -> int:
        """
        Get remaining requests allowed in current window.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Number of remaining requests, or -1 if no limit
        """
        if tool_name not in self.limits:
            return -1  # No limit
        
        if tool_name not in self.requests:
            return self.limits[tool_name]
        
        self._cleanup_old_requests(tool_name)
        
        limit = self.limits[tool_name]
        current_count = len(self.requests[tool_name])
        
        return max(0, limit - current_count)
    
    def get_wait_time(self, tool_name: str) -> float:
        """
        Get time to wait before next request is allowed.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Wait time in seconds, or 0 if no wait needed
        """
        if tool_name not in self.limits:
            return 0.0
        
        if tool_name not in self.requests or not self.requests[tool_name]:
            return 0.0
        
        self._cleanup_old_requests(tool_name)
        
        limit = self.limits[tool_name]
        current_count = len(self.requests[tool_name])
        
        if current_count < limit:
            return 0.0
        
        # Need to wait until oldest request expires
        oldest_request = self.requests[tool_name][0]
        current_time = time.time()
        window_end = oldest_request + self.window_seconds
        
        return max(0.0, window_end - current_time)
    
    def reset(self, tool_name: Optional[str] = None):
        """
        Reset rate limiting state.
        
        Args:
            tool_name: Tool to reset, or None for all tools
        """
        if tool_name is None:
            self.requests.clear()
            self.blocked_count.clear()
            logger.info("Reset all rate limiting state")
        else:
            if tool_name in self.requests:
                del self.requests[tool_name]
            if tool_name in self.blocked_count:
                del self.blocked_count[tool_name]
            logger.info(f"Reset rate limiting state for {tool_name}")
    
    def set_limit(self, tool_name: str, limit: int):
        """
        Set or update rate limit for a tool.
        
        Args:
            tool_name: Name of the tool
            limit: Maximum requests per window
        """
        self.limits[tool_name] = limit
        logger.info(f"Set rate limit for {tool_name}: {limit} requests/{self.window_seconds}s")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get rate limiting statistics."""
        stats = {
            "tools_tracked": len(self.requests),
            "total_requests": sum(len(q) for q in self.requests.values()),
            "blocked_attempts": sum(self.blocked_count.values()),
            "per_tool": {},
        }
        
        for tool_name in self.limits:
            if tool_name in self.requests:
                self._cleanup_old_requests(tool_name)
                current_count = len(self.requests[tool_name])
            else:
                current_count = 0
            
            stats["per_tool"][tool_name] = {
                "limit": self.limits[tool_name],
                "current_count": current_count,
                "remaining": self.get_remaining(tool_name),
                "blocked_count": self.blocked_count.get(tool_name, 0),
                "wait_time": self.get_wait_time(tool_name),
            }
        
        return stats


__all__ = [
    "RateLimiter",
    "DEFAULT_RATE_LIMITS",
]
