"""
Notification System - Push notifications, status bar, notification center, sounds.

User notification features:
- Push notifications with priorities
- Status bar with system status
- Notification center/history
- Sound settings for alerts

Part of the Enigma AI Engine GUI notification suite.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# NOTIFICATION TYPES & DATA
# =============================================================================

class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationType(Enum):
    """Types of notifications."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PROGRESS = "progress"
    CHAT = "chat"
    SYSTEM = "system"


@dataclass
class Notification:
    """A notification message."""
    id: str
    title: str
    message: str
    type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    read: bool = False
    dismissed: bool = False
    sound: Optional[str] = None
    icon: Optional[str] = None
    actions: list[dict[str, str]] = field(default_factory=list)
    progress: Optional[float] = None  # 0-1 for progress notifications
    source: str = ""  # Module/feature that sent it
    data: dict[str, Any] = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.timestamp
    
    def age_formatted(self) -> str:
        """Get human-readable age."""
        age = self.age_seconds()
        if age < 60:
            return "just now"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        elif age < 86400:
            return f"{int(age / 3600)}h ago"
        return f"{int(age / 86400)}d ago"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "type": self.type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "read": self.read,
            "dismissed": self.dismissed,
            "sound": self.sound,
            "icon": self.icon,
            "actions": self.actions,
            "progress": self.progress,
            "source": self.source
        }


# =============================================================================
# SOUND SETTINGS
# =============================================================================

class SoundType(Enum):
    """Sound event types."""
    NOTIFICATION = "notification"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    STARTUP = "startup"
    GENERATION_COMPLETE = "generation_complete"


@dataclass
class SoundProfile:
    """A collection of sound settings."""
    name: str
    enabled: bool = True
    volume: float = 0.7  # 0-1
    sounds: dict[str, Optional[str]] = field(default_factory=dict)  # SoundType -> path


class SoundSettings:
    """
    Manage notification and UI sounds.
    
    Features:
    - Enable/disable sounds per type
    - Volume control
    - Custom sound files
    - Sound profiles
    """
    
    # Default sound mappings (None = system default)
    DEFAULT_SOUNDS = {
        SoundType.NOTIFICATION: None,
        SoundType.MESSAGE_SENT: None,
        SoundType.MESSAGE_RECEIVED: None,
        SoundType.ERROR: None,
        SoundType.SUCCESS: None,
        SoundType.WARNING: None,
        SoundType.STARTUP: None,
        SoundType.GENERATION_COMPLETE: None,
    }
    
    BUILTIN_PROFILES = {
        "default": SoundProfile("Default", True, 0.7),
        "quiet": SoundProfile("Quiet", True, 0.3),
        "silent": SoundProfile("Silent", False, 0.0),
        "loud": SoundProfile("Loud", True, 1.0),
    }
    
    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize sound settings.
        
        Args:
            persist_path: Path to save settings
        """
        self.persist_path = persist_path
        
        self._enabled = True
        self._volume = 0.7
        self._muted = False
        self._sounds: dict[SoundType, Optional[str]] = self.DEFAULT_SOUNDS.copy()
        self._enabled_types: dict[SoundType, bool] = {t: True for t in SoundType}
        self._profiles = self.BUILTIN_PROFILES.copy()
        self._current_profile = "default"
        
        if persist_path and persist_path.exists():
            self._load()
    
    @property
    def enabled(self) -> bool:
        """Check if sounds are enabled."""
        return self._enabled and not self._muted
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable/disable sounds."""
        self._enabled = value
        self._save()
    
    @property
    def volume(self) -> float:
        """Get current volume (0-1)."""
        return self._volume
    
    @volume.setter
    def volume(self, value: float):
        """Set volume (0-1)."""
        self._volume = max(0.0, min(1.0, value))
        self._save()
    
    def mute(self, muted: bool = True):
        """Mute/unmute sounds."""
        self._muted = muted
    
    def is_muted(self) -> bool:
        """Check if muted."""
        return self._muted
    
    def set_sound_enabled(self, sound_type: SoundType, enabled: bool):
        """Enable/disable specific sound type."""
        self._enabled_types[sound_type] = enabled
        self._save()
    
    def is_sound_enabled(self, sound_type: SoundType) -> bool:
        """Check if sound type is enabled."""
        return self._enabled_types.get(sound_type, True)
    
    def set_custom_sound(self, sound_type: SoundType, path: Optional[str]):
        """Set custom sound file for a type."""
        self._sounds[sound_type] = path
        self._save()
    
    def get_sound_path(self, sound_type: SoundType) -> Optional[str]:
        """Get sound file path for a type."""
        return self._sounds.get(sound_type)
    
    def get_profiles(self) -> list[str]:
        """Get available profile names."""
        return list(self._profiles.keys())
    
    def apply_profile(self, name: str):
        """Apply a sound profile."""
        if name in self._profiles:
            profile = self._profiles[name]
            self._enabled = profile.enabled
            self._volume = profile.volume
            self._current_profile = name
            self._save()
    
    def play(self, sound_type: SoundType):
        """
        Play a sound (stub - actual implementation depends on platform).
        
        In real implementation, this would use:
        - Windows: winsound or playsound
        - macOS: afplay or AppKit
        - Linux: aplay or pygame
        """
        if not self.enabled:
            return
        
        if not self.is_sound_enabled(sound_type):
            return
        
        sound_path = self.get_sound_path(sound_type)
        
        # Log for debugging (actual play would go here)
        logger.debug(f"Would play sound: {sound_type.value} at volume {self._volume}")
        
        # Platform-specific play would go here
        # try:
        #     import winsound
        #     if sound_path:
        #         winsound.PlaySound(sound_path, winsound.SND_ASYNC)
        #     else:
        #         winsound.MessageBeep()
        # except ImportError:
        #     pass
    
    def _save(self):
        """Save to disk."""
        if not self.persist_path:
            return
        
        try:
            data = {
                "enabled": self._enabled,
                "volume": self._volume,
                "current_profile": self._current_profile,
                "sounds": {t.value: p for t, p in self._sounds.items()},
                "enabled_types": {t.value: e for t, e in self._enabled_types.items()}
            }
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save sound settings: {e}")
    
    def _load(self):
        """Load from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            self._enabled = data.get("enabled", True)
            self._volume = data.get("volume", 0.7)
            self._current_profile = data.get("current_profile", "default")
            
            for type_str, path in data.get("sounds", {}).items():
                try:
                    self._sounds[SoundType(type_str)] = path
                except ValueError:
                    pass
            
            for type_str, enabled in data.get("enabled_types", {}).items():
                try:
                    self._enabled_types[SoundType(type_str)] = enabled
                except ValueError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to load sound settings: {e}")


# =============================================================================
# STATUS BAR
# =============================================================================

class StatusLevel(Enum):
    """Status indicator levels."""
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class StatusItem:
    """A status bar item."""
    id: str
    text: str
    level: StatusLevel = StatusLevel.OK
    tooltip: str = ""
    icon: Optional[str] = None
    progress: Optional[float] = None
    clickable: bool = False
    visible: bool = True


class StatusBar:
    """
    System status bar management.
    
    Features:
    - Multiple status items
    - Progress indicators
    - Status levels with colors
    - Click handlers
    """
    
    def __init__(self):
        """Initialize status bar."""
        self._items: dict[str, StatusItem] = {}
        self._order: list[str] = []
        self._callbacks: list[Callable[[StatusItem], None]] = []
        self._click_handlers: dict[str, Callable[[], None]] = {}
        
        # Default items
        self._add_default_items()
    
    def _add_default_items(self):
        """Add default status items."""
        self.add_item(StatusItem(
            id="connection",
            text="Ready",
            level=StatusLevel.OK,
            tooltip="System ready"
        ))
        
        self.add_item(StatusItem(
            id="model",
            text="No model",
            level=StatusLevel.INFO,
            tooltip="No model loaded"
        ))
    
    def add_item(self, item: StatusItem, position: Optional[int] = None):
        """Add a status item."""
        self._items[item.id] = item
        
        if item.id not in self._order:
            if position is not None:
                self._order.insert(position, item.id)
            else:
                self._order.append(item.id)
        
        self._notify(item)
    
    def update_item(
        self,
        item_id: str,
        text: Optional[str] = None,
        level: Optional[StatusLevel] = None,
        tooltip: Optional[str] = None,
        progress: Optional[float] = None
    ):
        """Update a status item."""
        if item_id not in self._items:
            return
        
        item = self._items[item_id]
        
        if text is not None:
            item.text = text
        if level is not None:
            item.level = level
        if tooltip is not None:
            item.tooltip = tooltip
        if progress is not None:
            item.progress = progress
        
        self._notify(item)
    
    def remove_item(self, item_id: str):
        """Remove a status item."""
        if item_id in self._items:
            del self._items[item_id]
        if item_id in self._order:
            self._order.remove(item_id)
        if item_id in self._click_handlers:
            del self._click_handlers[item_id]
    
    def get_item(self, item_id: str) -> Optional[StatusItem]:
        """Get a status item."""
        return self._items.get(item_id)
    
    def get_all_items(self) -> list[StatusItem]:
        """Get all visible items in order."""
        return [
            self._items[item_id]
            for item_id in self._order
            if item_id in self._items and self._items[item_id].visible
        ]
    
    def set_click_handler(self, item_id: str, handler: Callable[[], None]):
        """Set click handler for an item."""
        self._click_handlers[item_id] = handler
        if item_id in self._items:
            self._items[item_id].clickable = True
    
    def handle_click(self, item_id: str):
        """Handle item click."""
        if item_id in self._click_handlers:
            try:
                self._click_handlers[item_id]()
            except Exception as e:
                logger.warning(f"Status click handler error: {e}")
    
    def show_progress(self, item_id: str, progress: float, text: Optional[str] = None):
        """Show progress on an item."""
        self.update_item(
            item_id,
            text=text,
            level=StatusLevel.LOADING,
            progress=max(0.0, min(1.0, progress))
        )
    
    def clear_progress(self, item_id: str, text: str = "Ready"):
        """Clear progress indicator."""
        self.update_item(
            item_id,
            text=text,
            level=StatusLevel.OK,
            progress=None
        )
    
    def on_change(self, callback: Callable[[StatusItem], None]):
        """Register callback for status changes."""
        self._callbacks.append(callback)
    
    def _notify(self, item: StatusItem):
        """Notify callbacks."""
        for callback in self._callbacks:
            try:
                callback(item)
            except Exception as e:
                logger.warning(f"Status callback error: {e}")


# =============================================================================
# NOTIFICATIONS CENTER
# =============================================================================

class NotificationCenter:
    """
    Central notification management and history.
    
    Features:
    - Queue and dispatch notifications
    - Notification history
    - Filter by type/priority
    - Bulk operations
    """
    
    def __init__(
        self,
        max_history: int = 500,
        persist_path: Optional[Path] = None
    ):
        """
        Initialize notification center.
        
        Args:
            max_history: Maximum notifications to keep
            persist_path: Path to save history
        """
        self.max_history = max_history
        self.persist_path = persist_path
        
        self._notifications: list[Notification] = []
        self._handlers: list[Callable[[Notification], None]] = []
        self._id_counter = 0
        self._lock = threading.Lock()
        
        # Unread badge count
        self._unread_count = 0
        self._badge_callbacks: list[Callable[[int], None]] = []
        
        if persist_path and persist_path.exists():
            self._load()
    
    def notify(
        self,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        sound: Optional[str] = None,
        actions: Optional[list[dict[str, str]]] = None,
        source: str = "",
        data: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Send a notification.
        
        Args:
            title: Notification title
            message: Notification message
            type: Notification type
            priority: Priority level
            sound: Sound to play (if any)
            actions: Action buttons
            source: Source module
            data: Additional data
            
        Returns:
            Notification ID
        """
        with self._lock:
            self._id_counter += 1
            notif_id = f"notif_{self._id_counter}_{int(time.time())}"
        
        notification = Notification(
            id=notif_id,
            title=title,
            message=message,
            type=type,
            priority=priority,
            sound=sound,
            actions=actions or [],
            source=source,
            data=data or {}
        )
        
        with self._lock:
            self._notifications.insert(0, notification)
            self._cleanup()
            self._unread_count += 1
        
        # Notify handlers
        for handler in self._handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.warning(f"Notification handler error: {e}")
        
        # Update badge
        self._notify_badge()
        
        self._save()
        
        return notif_id
    
    # Convenience methods
    def info(self, title: str, message: str, **kwargs) -> str:
        """Send info notification."""
        return self.notify(title, message, NotificationType.INFO, **kwargs)
    
    def success(self, title: str, message: str, **kwargs) -> str:
        """Send success notification."""
        return self.notify(title, message, NotificationType.SUCCESS, **kwargs)
    
    def warning(self, title: str, message: str, **kwargs) -> str:
        """Send warning notification."""
        return self.notify(
            title, message, NotificationType.WARNING,
            priority=NotificationPriority.HIGH, **kwargs
        )
    
    def error(self, title: str, message: str, **kwargs) -> str:
        """Send error notification."""
        return self.notify(
            title, message, NotificationType.ERROR,
            priority=NotificationPriority.URGENT, **kwargs
        )
    
    def progress(
        self,
        title: str,
        message: str,
        progress: float,
        **kwargs
    ) -> str:
        """Send progress notification."""
        notif_id = self.notify(
            title, message, NotificationType.PROGRESS, **kwargs
        )
        if notif_id:
            with self._lock:
                for n in self._notifications:
                    if n.id == notif_id:
                        n.progress = progress
                        break
        return notif_id
    
    def update_progress(self, notif_id: str, progress: float, message: Optional[str] = None):
        """Update progress notification."""
        with self._lock:
            for n in self._notifications:
                if n.id == notif_id:
                    n.progress = progress
                    if message:
                        n.message = message
                    break
    
    def get_all(self) -> list[Notification]:
        """Get all notifications."""
        return self._notifications.copy()
    
    def get_unread(self) -> list[Notification]:
        """Get unread notifications."""
        return [n for n in self._notifications if not n.read]
    
    def get_by_type(self, type: NotificationType) -> list[Notification]:
        """Get notifications by type."""
        return [n for n in self._notifications if n.type == type]
    
    def get_by_priority(self, priority: NotificationPriority) -> list[Notification]:
        """Get notifications by priority."""
        return [n for n in self._notifications if n.priority == priority]
    
    def get_recent(self, limit: int = 20) -> list[Notification]:
        """Get recent notifications."""
        return self._notifications[:limit]
    
    def mark_read(self, notif_id: str):
        """Mark notification as read."""
        with self._lock:
            for n in self._notifications:
                if n.id == notif_id and not n.read:
                    n.read = True
                    self._unread_count = max(0, self._unread_count - 1)
                    break
        self._notify_badge()
        self._save()
    
    def mark_all_read(self):
        """Mark all notifications as read."""
        with self._lock:
            for n in self._notifications:
                n.read = True
            self._unread_count = 0
        self._notify_badge()
        self._save()
    
    def dismiss(self, notif_id: str):
        """Dismiss a notification."""
        with self._lock:
            for n in self._notifications:
                if n.id == notif_id:
                    n.dismissed = True
                    if not n.read:
                        n.read = True
                        self._unread_count = max(0, self._unread_count - 1)
                    break
        self._notify_badge()
        self._save()
    
    def clear_all(self):
        """Clear all notifications."""
        with self._lock:
            self._notifications = []
            self._unread_count = 0
        self._notify_badge()
        self._save()
    
    def get_unread_count(self) -> int:
        """Get unread notification count."""
        return self._unread_count
    
    def on_notification(self, handler: Callable[[Notification], None]):
        """Register handler for new notifications."""
        self._handlers.append(handler)
    
    def on_badge_change(self, callback: Callable[[int], None]):
        """Register callback for badge count changes."""
        self._badge_callbacks.append(callback)
    
    def _notify_badge(self):
        """Notify badge callbacks."""
        for callback in self._badge_callbacks:
            try:
                callback(self._unread_count)
            except Exception as e:
                logger.warning(f"Badge callback error: {e}")
    
    def _cleanup(self):
        """Remove old notifications over limit."""
        # Keep only non-dismissed
        self._notifications = [
            n for n in self._notifications
            if not n.dismissed
        ][:self.max_history]
    
    def _save(self):
        """Save to disk."""
        if not self.persist_path:
            return
        
        try:
            data = [n.to_dict() for n in self._notifications[:100]]  # Save recent
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save notifications: {e}")
    
    def _load(self):
        """Load from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            for d in data:
                try:
                    notif = Notification(
                        id=d["id"],
                        title=d["title"],
                        message=d["message"],
                        type=NotificationType(d.get("type", "info")),
                        priority=NotificationPriority(d.get("priority", "normal")),
                        timestamp=d.get("timestamp", time.time()),
                        read=d.get("read", False),
                        dismissed=d.get("dismissed", False),
                        sound=d.get("sound"),
                        icon=d.get("icon"),
                        actions=d.get("actions", []),
                        progress=d.get("progress"),
                        source=d.get("source", "")
                    )
                    self._notifications.append(notif)
                    if not notif.read:
                        self._unread_count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to load notification: {e}")
        except Exception as e:
            logger.warning(f"Failed to load notifications: {e}")


# =============================================================================
# PUSH NOTIFICATION SERVICE
# =============================================================================

class PushNotificationService:
    """
    Push notification service for desktop notifications.
    
    Features:
    - System tray notifications
    - Platform-specific implementations
    - Action buttons
    - Click handling
    """
    
    def __init__(self, app_name: str = "Enigma AI Engine"):
        """
        Initialize push notification service.
        
        Args:
            app_name: Application name for notifications
        """
        self.app_name = app_name
        self._enabled = True
        self._click_handlers: dict[str, Callable[[], None]] = {}
    
    @property
    def enabled(self) -> bool:
        """Check if push notifications are enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable/disable push notifications."""
        self._enabled = value
    
    def show(
        self,
        title: str,
        message: str,
        icon: Optional[str] = None,
        timeout: int = 5000,
        on_click: Optional[Callable[[], None]] = None
    ):
        """
        Show a push notification.
        
        Args:
            title: Notification title
            message: Notification message
            icon: Icon path (optional)
            timeout: Display duration in ms
            on_click: Click callback
        """
        if not self._enabled:
            return
        
        notif_id = f"push_{int(time.time())}"
        if on_click:
            self._click_handlers[notif_id] = on_click
        
        # Platform-specific implementation would go here
        # For now, log the notification
        logger.info(f"Push notification: {title} - {message}")
        
        # In real implementation:
        # - Windows: win10toast or plyer
        # - macOS: pync or pyobjc
        # - Linux: notify2 or plyer
        
        try:
            # Try plyer (cross-platform)
            from plyer import notification as plyer_notif
            plyer_notif.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                timeout=timeout // 1000
            )
        except ImportError:
            # Fallback: just log
            logger.debug(f"Push notification (no plyer): {title}")


# =============================================================================
# NOTIFICATION BADGE MANAGER
# =============================================================================

class NotificationBadge:
    """
    Manage notification badges/counts.
    
    Features:
    - Per-source badge counts
    - Combined badge
    - Badge limits
    """
    
    def __init__(self, max_badge: int = 99):
        """
        Initialize badge manager.
        
        Args:
            max_badge: Maximum displayed count (shows "99+" above)
        """
        self.max_badge = max_badge
        self._counts: dict[str, int] = {}  # source -> count
        self._callbacks: list[Callable[[int, dict[str, int]], None]] = []
    
    def increment(self, source: str = "default", amount: int = 1):
        """Increment badge count."""
        self._counts[source] = self._counts.get(source, 0) + amount
        self._notify()
    
    def decrement(self, source: str = "default", amount: int = 1):
        """Decrement badge count."""
        if source in self._counts:
            self._counts[source] = max(0, self._counts[source] - amount)
            if self._counts[source] == 0:
                del self._counts[source]
        self._notify()
    
    def set(self, source: str, count: int):
        """Set badge count for source."""
        if count > 0:
            self._counts[source] = count
        elif source in self._counts:
            del self._counts[source]
        self._notify()
    
    def clear(self, source: Optional[str] = None):
        """Clear badge count."""
        if source:
            self._counts.pop(source, None)
        else:
            self._counts = {}
        self._notify()
    
    def get(self, source: Optional[str] = None) -> int:
        """Get badge count."""
        if source:
            return self._counts.get(source, 0)
        return sum(self._counts.values())
    
    def get_display(self) -> str:
        """Get badge display string."""
        total = self.get()
        if total == 0:
            return ""
        if total > self.max_badge:
            return f"{self.max_badge}+"
        return str(total)
    
    def get_by_source(self) -> dict[str, int]:
        """Get counts by source."""
        return self._counts.copy()
    
    def on_change(self, callback: Callable[[int, dict[str, int]], None]):
        """Register callback for badge changes."""
        self._callbacks.append(callback)
    
    def _notify(self):
        """Notify callbacks."""
        total = self.get()
        by_source = self.get_by_source()
        for callback in self._callbacks:
            try:
                callback(total, by_source)
            except Exception as e:
                logger.warning(f"Badge callback error: {e}")


# =============================================================================
# COMBINED NOTIFICATION SYSTEM
# =============================================================================

class NotificationSystem:
    """
    Combined notification system manager.
    
    Integrates:
    - Notification center
    - Push notifications
    - Status bar
    - Sound settings
    - Badge management
    """
    
    def __init__(
        self,
        app_name: str = "Enigma AI Engine",
        data_dir: Optional[Path] = None,
        max_history: int = 500
    ):
        """
        Initialize notification system.
        
        Args:
            app_name: Application name
            data_dir: Directory for persistence
            max_history: Max notifications to keep
        """
        notif_path = data_dir / "notifications.json" if data_dir else None
        sound_path = data_dir / "sound_settings.json" if data_dir else None
        
        self.center = NotificationCenter(max_history, notif_path)
        self.push = PushNotificationService(app_name)
        self.status = StatusBar()
        self.sounds = SoundSettings(sound_path)
        self.badges = NotificationBadge()
        
        # Wire up components
        self.center.on_notification(self._on_notification)
        self.center.on_badge_change(lambda count: self.badges.set("notifications", count))
    
    def notify(
        self,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        show_push: bool = True,
        play_sound: bool = True,
        **kwargs
    ) -> str:
        """
        Send a notification through the system.
        
        Args:
            title: Notification title
            message: Notification message
            type: Notification type
            priority: Priority level
            show_push: Show push notification
            play_sound: Play sound
            **kwargs: Additional notification args
            
        Returns:
            Notification ID
        """
        # Add to center
        notif_id = self.center.notify(title, message, type, priority, **kwargs)
        
        # Show push if enabled and high enough priority
        if show_push and priority in [NotificationPriority.HIGH, NotificationPriority.URGENT]:
            self.push.show(title, message)
        
        # Play sound
        if play_sound:
            sound_type = self._type_to_sound(type)
            self.sounds.play(sound_type)
        
        return notif_id
    
    def _on_notification(self, notification: Notification):
        """Handle new notification."""
        # Could update status bar, etc.
    
    def _type_to_sound(self, notif_type: NotificationType) -> SoundType:
        """Map notification type to sound type."""
        return {
            NotificationType.INFO: SoundType.NOTIFICATION,
            NotificationType.SUCCESS: SoundType.SUCCESS,
            NotificationType.WARNING: SoundType.WARNING,
            NotificationType.ERROR: SoundType.ERROR,
            NotificationType.PROGRESS: SoundType.NOTIFICATION,
            NotificationType.CHAT: SoundType.MESSAGE_RECEIVED,
            NotificationType.SYSTEM: SoundType.NOTIFICATION,
        }.get(notif_type, SoundType.NOTIFICATION)


# Singleton
_notification_system: Optional[NotificationSystem] = None


def get_notification_system(
    app_name: str = "Enigma AI Engine",
    data_dir: Optional[Path] = None
) -> NotificationSystem:
    """Get or create notification system."""
    global _notification_system
    if _notification_system is None:
        _notification_system = NotificationSystem(app_name, data_dir)
    return _notification_system
