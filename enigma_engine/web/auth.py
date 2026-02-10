"""
Authentication for Enigma AI Engine Web Interface

Simple but secure token-based authentication for local network access.
Tokens are generated on first connection and can be shared via QR code.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WebAuth:
    """
    Token-based authentication for web interface.
    
    Features:
    - Auto-generates token on first run
    - Tokens stored securely in memory and optionally on disk
    - Automatic token expiration
    - Support for multiple devices
    """
    
    def __init__(self, token_file: Optional[Path] = None, token_lifetime_hours: int = 720):
        """
        Initialize authentication system.
        
        Args:
            token_file: Path to save tokens (None = memory only)
            token_lifetime_hours: Hours before token expires (default: 30 days)
        """
        self.token_file = token_file
        self.token_lifetime = timedelta(hours=token_lifetime_hours)
        self.tokens: dict[str, dict] = {}
        self.master_token: Optional[str] = None
        
        # Load existing tokens from file
        if token_file and token_file.exists():
            self._load_tokens()
        
        # Ensure we have a master token
        if not self.master_token:
            self.master_token = self.generate_token()
            self._save_tokens()
    
    def generate_token(self, description: str = "Web Access") -> str:
        """
        Generate a new authentication token.
        
        Args:
            description: Human-readable description of this token
            
        Returns:
            The generated token string
        """
        token = secrets.token_urlsafe(32)
        self.tokens[token] = {
            "created": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "description": description,
            "uses": 0
        }
        self._save_tokens()
        return token
    
    def verify_token(self, token: str) -> bool:
        """
        Verify if a token is valid and not expired.
        
        Args:
            token: Token to verify
            
        Returns:
            True if token is valid, False otherwise
        """
        if not token or token not in self.tokens:
            return False
        
        token_info = self.tokens[token]
        
        # Check expiration
        created = datetime.fromisoformat(token_info["created"])
        if datetime.now() - created > self.token_lifetime:
            # Token expired
            del self.tokens[token]
            self._save_tokens()
            return False
        
        # Update last used time
        token_info["last_used"] = datetime.now().isoformat()
        token_info["uses"] = token_info.get("uses", 0) + 1
        self._save_tokens()
        
        return True
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (remove it).
        
        Args:
            token: Token to revoke
            
        Returns:
            True if token was found and revoked
        """
        if token in self.tokens:
            del self.tokens[token]
            self._save_tokens()
            return True
        return False
    
    def get_master_token(self) -> str:
        """
        Get the master token (for QR code, etc.).
        
        Returns:
            The master authentication token
        """
        return self.master_token
    
    def list_tokens(self) -> dict[str, dict]:
        """
        List all active tokens.
        
        Returns:
            Dictionary of tokens with their metadata
        """
        return self.tokens.copy()
    
    def _load_tokens(self):
        """Load tokens from file."""
        if not self.token_file:
            return
        
        try:
            with open(self.token_file) as f:
                data = json.load(f)
                self.tokens = data.get("tokens", {})
                self.master_token = data.get("master_token")
        except Exception as e:
            logger.warning(f"Could not load tokens from {self.token_file}: {e}")
    
    def _save_tokens(self):
        """Save tokens to file."""
        if not self.token_file:
            return
        
        try:
            # Ensure directory exists
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.token_file, 'w') as f:
                json.dump({
                    "master_token": self.master_token,
                    "tokens": self.tokens
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save tokens to {self.token_file}: {e}")


# Global auth instance (lazy initialized)
_auth_instance: Optional[WebAuth] = None


def get_auth(config: Optional[dict] = None) -> WebAuth:
    """
    Get or create the global auth instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        The global WebAuth instance
    """
    global _auth_instance
    
    if _auth_instance is None:
        token_file = None
        if config:
            from ..config import CONFIG
            memory_dir = Path(CONFIG.get("memory_dir", "memory"))
            token_file = memory_dir / "web_tokens.json"
        
        _auth_instance = WebAuth(token_file=token_file)
    
    return _auth_instance
