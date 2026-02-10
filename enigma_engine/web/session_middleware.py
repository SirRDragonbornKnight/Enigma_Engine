"""
Multi-User Session Middleware for Enigma AI Engine Web Interface

Integrates the full auth system (accounts.py) with the Flask web app
for proper multi-user support with:
- User registration/login
- Session management
- Role-based access control
- API authentication
"""

import functools
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from flask import Flask, g, jsonify, redirect, request, session, url_for
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Try to import the full auth system
try:
    from ..auth.accounts import (
        PasswordHasher,
        TokenManager,
        HAS_SQLITE,
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    HAS_SQLITE = False

from ..config import CONFIG


class UserSession:
    """Represents a user session in the web app."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        role: str = "user",
        display_name: Optional[str] = None,
        email: Optional[str] = None,
    ):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.display_name = display_name or username
        self.email = email
        self.created_at = datetime.now()
        self.last_active = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for session storage."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "display_name": self.display_name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserSession":
        """Create from dictionary."""
        sess = cls(
            user_id=data["user_id"],
            username=data["username"],
            role=data.get("role", "user"),
            display_name=data.get("display_name"),
            email=data.get("email"),
        )
        if "created_at" in data:
            sess.created_at = datetime.fromisoformat(data["created_at"])
        if "last_active" in data:
            sess.last_active = datetime.fromisoformat(data["last_active"])
        return sess
    
    @property
    def is_admin(self) -> bool:
        return self.role in ("admin", "owner")
    
    @property
    def is_moderator(self) -> bool:
        return self.role in ("admin", "owner", "moderator")


class SessionManager:
    """
    Manages user sessions for the web interface.
    
    Integrates with the full auth system for user verification
    while maintaining web-session-specific functionality.
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        session_timeout_hours: int = 24,
    ):
        self.db_path = db_path or CONFIG.paths.get("memory", Path("memory")) / "users.db"
        self.session_timeout_hours = session_timeout_hours
        self.token_manager = TokenManager() if AUTH_AVAILABLE else None
        self.password_hasher = PasswordHasher if AUTH_AVAILABLE else None
        
        # In-memory session cache
        self._sessions: dict[str, UserSession] = {}
        
        # User database (simple JSON fallback if no SQLite)
        self._users_file = self.db_path.parent / "users.json"
        self._users: dict[str, dict] = {}
        self._load_users()
    
    def _load_users(self):
        """Load users from file."""
        try:
            if self._users_file.exists():
                import json
                with open(self._users_file) as f:
                    self._users = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load users: {e}")
            self._users = {}
    
    def _save_users(self):
        """Save users to file."""
        try:
            import json
            self._users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._users_file, "w") as f:
                json.dump(self._users, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save users: {e}")
    
    def register_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        role: str = "user",
    ) -> tuple[bool, str, Optional[UserSession]]:
        """
        Register a new user.
        
        Returns:
            Tuple of (success, message, session or None)
        """
        # Validate username
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters", None
        
        if username.lower() in self._users:
            return False, "Username already exists", None
        
        # Validate password
        if not password or len(password) < 8:
            return False, "Password must be at least 8 characters", None
        
        # Hash password
        import secrets
        import hashlib
        
        if self.password_hasher:
            salt, password_hash = self.password_hasher.hash_password(password)
        else:
            # Fallback hashing
            salt = secrets.token_bytes(32)
            password_hash = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt, 100000
            )
        
        # Create user ID
        import uuid
        user_id = str(uuid.uuid4())
        
        # First user is admin
        if not self._users:
            role = "admin"
            logger.info(f"First user {username} registered as admin")
        
        # Store user
        self._users[username.lower()] = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "display_name": display_name or username,
            "role": role,
            "password_salt": salt.hex(),
            "password_hash": password_hash.hex(),
            "created_at": datetime.now().isoformat(),
            "conversation_count": 0,
            "message_count": 0,
            "settings": {},
        }
        self._save_users()
        
        # Create session
        user_session = UserSession(
            user_id=user_id,
            username=username,
            role=role,
            display_name=display_name or username,
            email=email,
        )
        
        return True, "Registration successful", user_session
    
    def login(
        self,
        username: str,
        password: str,
    ) -> tuple[bool, str, Optional[UserSession]]:
        """
        Authenticate user and create session.
        
        Returns:
            Tuple of (success, message, session or None)
        """
        user_data = self._users.get(username.lower())
        if not user_data:
            return False, "Invalid username or password", None
        
        # Verify password
        import hashlib
        
        try:
            salt = bytes.fromhex(user_data["password_salt"])
            stored_hash = bytes.fromhex(user_data["password_hash"])
            
            if self.password_hasher:
                valid = self.password_hasher.verify_password(password, salt, stored_hash)
            else:
                computed = hashlib.pbkdf2_hmac(
                    "sha256", password.encode(), salt, 100000
                )
                valid = computed == stored_hash
            
            if not valid:
                return False, "Invalid username or password", None
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False, "Authentication error", None
        
        # Update last login
        user_data["last_login"] = datetime.now().isoformat()
        self._save_users()
        
        # Create session
        user_session = UserSession(
            user_id=user_data["user_id"],
            username=user_data["username"],
            role=user_data.get("role", "user"),
            display_name=user_data.get("display_name"),
            email=user_data.get("email"),
        )
        
        return True, "Login successful", user_session
    
    def create_token(self, user_session: UserSession) -> str:
        """Create JWT token for API access."""
        if self.token_manager:
            return self.token_manager.create_token(
                user_id=user_session.user_id,
                expires_hours=self.session_timeout_hours,
                extra_claims={
                    "username": user_session.username,
                    "role": user_session.role,
                }
            )
        else:
            # Fallback token
            import secrets
            return f"{user_session.user_id}:{secrets.token_hex(16)}"
    
    def verify_token(self, token: str) -> Optional[UserSession]:
        """Verify token and return user session."""
        if self.token_manager:
            payload = self.token_manager.verify_token(token)
            if payload:
                user_id = payload.get("sub")
                # Find user by ID
                for username, data in self._users.items():
                    if data["user_id"] == user_id:
                        return UserSession(
                            user_id=user_id,
                            username=data["username"],
                            role=data.get("role", "user"),
                            display_name=data.get("display_name"),
                            email=data.get("email"),
                        )
        else:
            # Fallback verification
            try:
                parts = token.split(":")
                if len(parts) >= 2:
                    user_id = parts[0]
                    for username, data in self._users.items():
                        if data["user_id"] == user_id:
                            return UserSession(
                                user_id=user_id,
                                username=data["username"],
                                role=data.get("role", "user"),
                            )
            except Exception:
                pass
        return None
    
    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user data by ID."""
        for username, data in self._users.items():
            if data["user_id"] == user_id:
                # Return sanitized data (no password)
                return {
                    "user_id": data["user_id"],
                    "username": data["username"],
                    "email": data.get("email"),
                    "display_name": data.get("display_name"),
                    "role": data.get("role", "user"),
                    "created_at": data.get("created_at"),
                    "conversation_count": data.get("conversation_count", 0),
                    "message_count": data.get("message_count", 0),
                    "settings": data.get("settings", {}),
                }
        return None
    
    def update_user_stats(
        self,
        user_id: str,
        conversations: int = 0,
        messages: int = 0,
    ):
        """Update user statistics."""
        for username, data in self._users.items():
            if data["user_id"] == user_id:
                data["conversation_count"] = data.get("conversation_count", 0) + conversations
                data["message_count"] = data.get("message_count", 0) + messages
                self._save_users()
                return
    
    def update_user_settings(self, user_id: str, settings: dict):
        """Update user settings/preferences."""
        for username, data in self._users.items():
            if data["user_id"] == user_id:
                if "settings" not in data:
                    data["settings"] = {}
                data["settings"].update(settings)
                self._save_users()
                return
    
    def list_users(self, include_stats: bool = False) -> list[dict]:
        """List all users (admin only)."""
        users = []
        for username, data in self._users.items():
            user_info = {
                "user_id": data["user_id"],
                "username": data["username"],
                "display_name": data.get("display_name"),
                "role": data.get("role", "user"),
                "created_at": data.get("created_at"),
            }
            if include_stats:
                user_info["conversation_count"] = data.get("conversation_count", 0)
                user_info["message_count"] = data.get("message_count", 0)
                user_info["last_login"] = data.get("last_login")
            users.append(user_info)
        return users
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user (admin only)."""
        for username in list(self._users.keys()):
            if self._users[username]["user_id"] == user_id:
                del self._users[username]
                self._save_users()
                return True
        return False
    
    def change_role(self, user_id: str, new_role: str) -> bool:
        """Change user role (admin only)."""
        valid_roles = ("user", "moderator", "admin", "owner")
        if new_role not in valid_roles:
            return False
        
        for username, data in self._users.items():
            if data["user_id"] == user_id:
                data["role"] = new_role
                self._save_users()
                return True
        return False


# =============================================================================
# Flask Integration
# =============================================================================

# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def init_session_middleware(app: "Flask"):
    """
    Initialize session middleware for Flask app.
    
    Adds:
    - Before-request hook to load user session
    - Login/logout/register routes
    - Auth decorators
    """
    if not FLASK_AVAILABLE:
        logger.warning("Flask not available, skipping session middleware")
        return
    
    manager = get_session_manager()
    
    @app.before_request
    def load_user_session():
        """Load user session from cookie/token before each request."""
        g.user = None
        g.is_authenticated = False
        
        # Check session cookie first
        if "user_session" in session:
            try:
                g.user = UserSession.from_dict(session["user_session"])
                g.user.last_active = datetime.now()
                g.is_authenticated = True
            except Exception:
                session.pop("user_session", None)
        
        # Check Authorization header (for API requests)
        if not g.is_authenticated:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user_session = manager.verify_token(token)
                if user_session:
                    g.user = user_session
                    g.is_authenticated = True
    
    # =========================================================================
    # Auth Routes
    # =========================================================================
    
    @app.route("/auth/login", methods=["GET", "POST"])
    def login():
        """Login page and handler."""
        if request.method == "GET":
            return render_auth_page("login")
        
        data = request.get_json() if request.is_json else request.form
        username = data.get("username", "")
        password = data.get("password", "")
        
        success, message, user_session = manager.login(username, password)
        
        if success and user_session:
            session["user_session"] = user_session.to_dict()
            token = manager.create_token(user_session)
            
            if request.is_json:
                return jsonify({
                    "success": True,
                    "message": message,
                    "user": user_session.to_dict(),
                    "token": token,
                })
            return redirect(url_for("dashboard"))
        
        if request.is_json:
            return jsonify({"success": False, "error": message}), 401
        return render_auth_page("login", error=message)
    
    @app.route("/auth/register", methods=["GET", "POST"])
    def register():
        """Registration page and handler."""
        if request.method == "GET":
            return render_auth_page("register")
        
        data = request.get_json() if request.is_json else request.form
        username = data.get("username", "")
        password = data.get("password", "")
        email = data.get("email")
        display_name = data.get("display_name")
        
        success, message, user_session = manager.register_user(
            username=username,
            password=password,
            email=email,
            display_name=display_name,
        )
        
        if success and user_session:
            session["user_session"] = user_session.to_dict()
            token = manager.create_token(user_session)
            
            if request.is_json:
                return jsonify({
                    "success": True,
                    "message": message,
                    "user": user_session.to_dict(),
                    "token": token,
                })
            return redirect(url_for("dashboard"))
        
        if request.is_json:
            return jsonify({"success": False, "error": message}), 400
        return render_auth_page("register", error=message)
    
    @app.route("/auth/logout")
    def logout():
        """Logout handler."""
        session.pop("user_session", None)
        
        if request.is_json:
            return jsonify({"success": True, "message": "Logged out"})
        return redirect(url_for("index"))
    
    @app.route("/auth/me")
    def get_current_user():
        """Get current user info."""
        if not g.is_authenticated:
            return jsonify({"authenticated": False}), 401
        
        return jsonify({
            "authenticated": True,
            "user": g.user.to_dict(),
        })
    
    @app.route("/auth/refresh", methods=["POST"])
    def refresh_token():
        """Refresh authentication token."""
        if not g.is_authenticated:
            return jsonify({"error": "Not authenticated"}), 401
        
        token = manager.create_token(g.user)
        return jsonify({"token": token})


def render_auth_page(page_type: str, error: Optional[str] = None) -> str:
    """Render authentication page HTML."""
    from flask import render_template_string
    
    template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }} - Enigma AI</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .auth-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h1 {
            color: #fff;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            color: #ccc;
            margin-bottom: 8px;
            font-size: 14px;
        }
        input {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 16px;
        }
        input:focus {
            outline: none;
            border-color: #3498db;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #3498db, #8e44ad);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        .error {
            background: rgba(231,76,60,0.2);
            border: 1px solid #e74c3c;
            color: #e74c3c;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .switch-link {
            text-align: center;
            margin-top: 20px;
            color: #ccc;
        }
        .switch-link a {
            color: #3498db;
            text-decoration: none;
        }
        .logo {
            text-align: center;
            margin-bottom: 20px;
            font-size: 48px;
        }
    </style>
</head>
<body>
    <div class="auth-card">
        <div class="logo">E</div>
        <h1>{{ title }}</h1>
        
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        
        <form method="POST">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required minlength="3">
            </div>
            
            {% if page_type == 'register' %}
            <div class="form-group">
                <label for="email">Email (optional)</label>
                <input type="email" id="email" name="email">
            </div>
            <div class="form-group">
                <label for="display_name">Display Name (optional)</label>
                <input type="text" id="display_name" name="display_name">
            </div>
            {% endif %}
            
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required minlength="8">
            </div>
            
            <button type="submit">{{ button_text }}</button>
        </form>
        
        <div class="switch-link">
            {% if page_type == 'login' %}
            Don't have an account? <a href="/auth/register">Register</a>
            {% else %}
            Already have an account? <a href="/auth/login">Login</a>
            {% endif %}
        </div>
    </div>
</body>
</html>
    """
    
    return render_template_string(
        template,
        title="Login" if page_type == "login" else "Register",
        page_type=page_type,
        button_text="Login" if page_type == "login" else "Create Account",
        error=error,
    )


# =============================================================================
# Decorators for Route Protection
# =============================================================================

def login_required(f: Callable) -> Callable:
    """Decorator to require authentication."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not FLASK_AVAILABLE:
            return f(*args, **kwargs)
        
        if not g.get("is_authenticated"):
            if request.is_json:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f: Callable) -> Callable:
    """Decorator to require admin role."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not FLASK_AVAILABLE:
            return f(*args, **kwargs)
        
        if not g.get("is_authenticated"):
            if request.is_json:
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for("login"))
        
        if not g.user.is_admin:
            if request.is_json:
                return jsonify({"error": "Admin access required"}), 403
            return "Forbidden", 403
        
        return f(*args, **kwargs)
    return decorated


def role_required(*roles: str) -> Callable:
    """Decorator to require specific role(s)."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            if not FLASK_AVAILABLE:
                return f(*args, **kwargs)
            
            if not g.get("is_authenticated"):
                if request.is_json:
                    return jsonify({"error": "Authentication required"}), 401
                return redirect(url_for("login"))
            
            if g.user.role not in roles:
                if request.is_json:
                    return jsonify({"error": f"Requires role: {', '.join(roles)}"}), 403
                return "Forbidden", 403
            
            return f(*args, **kwargs)
        return decorated
    return decorator


# =============================================================================
# Admin API Routes
# =============================================================================

def add_admin_routes(app: "Flask"):
    """Add admin-only routes for user management."""
    if not FLASK_AVAILABLE:
        return
    
    manager = get_session_manager()
    
    @app.route("/api/admin/users", methods=["GET"])
    @admin_required
    def list_all_users():
        """List all users (admin only)."""
        include_stats = request.args.get("stats", "false").lower() == "true"
        users = manager.list_users(include_stats=include_stats)
        return jsonify({"users": users})
    
    @app.route("/api/admin/users/<user_id>", methods=["DELETE"])
    @admin_required
    def delete_user(user_id: str):
        """Delete a user (admin only)."""
        if user_id == g.user.user_id:
            return jsonify({"error": "Cannot delete yourself"}), 400
        
        if manager.delete_user(user_id):
            return jsonify({"success": True})
        return jsonify({"error": "User not found"}), 404
    
    @app.route("/api/admin/users/<user_id>/role", methods=["PUT"])
    @admin_required
    def change_user_role(user_id: str):
        """Change user role (admin only)."""
        data = request.get_json()
        new_role = data.get("role")
        
        if not new_role:
            return jsonify({"error": "Role required"}), 400
        
        if manager.change_role(user_id, new_role):
            return jsonify({"success": True})
        return jsonify({"error": "Invalid role or user not found"}), 400


# =============================================================================
# User Settings API
# =============================================================================

def add_user_settings_routes(app: "Flask"):
    """Add routes for user settings/preferences."""
    if not FLASK_AVAILABLE:
        return
    
    manager = get_session_manager()
    
    @app.route("/api/user/settings", methods=["GET"])
    @login_required
    def get_user_settings():
        """Get current user's settings."""
        user_data = manager.get_user(g.user.user_id)
        if user_data:
            return jsonify({"settings": user_data.get("settings", {})})
        return jsonify({"settings": {}})
    
    @app.route("/api/user/settings", methods=["PUT"])
    @login_required
    def update_user_settings():
        """Update current user's settings."""
        data = request.get_json()
        manager.update_user_settings(g.user.user_id, data)
        return jsonify({"success": True})
    
    @app.route("/api/user/profile", methods=["GET"])
    @login_required
    def get_user_profile():
        """Get current user's profile."""
        user_data = manager.get_user(g.user.user_id)
        if user_data:
            return jsonify({"profile": user_data})
        return jsonify({"error": "User not found"}), 404
