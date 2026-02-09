"""
System Notifications for Enigma AI Engine

Cross-platform notification system:
- Desktop notifications
- Sound alerts
- System tray integration
- Notification history
- Do Not Disturb mode

Works on Windows, Linux, and macOS.

Usage:
    from enigma_engine.utils.notifications import notify, get_notification_manager
    
    notify("Task Complete", "Your model training finished!")
    
    # Or with more control
    manager = get_notification_manager()
    manager.send(
        title="Training Complete",
        message="Model accuracy: 95%",
        icon="success",
        sound=True
    )
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)
PLATFORM = platform.system().lower()


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationType(Enum):
    """Notification types."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PROGRESS = "progress"


@dataclass
class Notification:
    """A notification entry."""
    id: str
    title: str
    message: str
    type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    icon: str | None = None
    sound: bool = True
    actions: list[dict[str, Any]] = field(default_factory=list)
    timeout: int = 5000  # milliseconds
    read: bool = False
    dismissed: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "type": self.type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "icon": self.icon,
            "sound": self.sound,
            "actions": self.actions,
            "timeout": self.timeout,
            "read": self.read,
            "dismissed": self.dismissed
        }


class NotificationBackend:
    """Base class for notification backends."""
    
    def send(self, notification: Notification) -> bool:
        """Send a notification."""
        raise NotImplementedError
    
    def play_sound(self, sound_type: str):
        """Play a notification sound."""
        pass


class WindowsNotificationBackend(NotificationBackend):
    """Windows notification backend using win10toast or winrt."""
    
    def __init__(self):
        self._toaster = None
        self._winrt = None
        
        # Try win10toast first
        try:
            from win10toast import ToastNotifier
            self._toaster = ToastNotifier()
            logger.info("Using win10toast backend")
        except ImportError:
            pass
        
        # Try Windows Runtime (winrt) for modern notifications
        if self._toaster is None:
            try:
                from winrt.windows.data.xml.dom import XmlDocument
                from winrt.windows.ui.notifications import (
                    ToastNotification,
                    ToastNotificationManager,
                )
                self._winrt = True
                logger.info("Using Windows Runtime backend")
            except ImportError:
                pass
    
    def send(self, notification: Notification) -> bool:
        """Send a Windows notification."""
        if self._toaster:
            try:
                # Map icon to icon file
                icon_path = self._get_icon_path(notification.icon or notification.type.value)
                
                self._toaster.show_toast(
                    notification.title,
                    notification.message,
                    icon_path=icon_path,
                    duration=notification.timeout // 1000,
                    threaded=True
                )
                return True
            except Exception as e:
                logger.error(f"win10toast error: {e}")
        
        if self._winrt:
            try:
                return self._send_winrt(notification)
            except Exception as e:
                logger.error(f"winrt error: {e}")
        
        # Fallback to console
        logger.info(f"[NOTIFICATION] {notification.title}: {notification.message}")
        return True
    
    def _send_winrt(self, notification: Notification) -> bool:
        """Send using Windows Runtime."""
        from winrt.windows.data.xml.dom import XmlDocument
        from winrt.windows.ui.notifications import (
            ToastNotification,
            ToastNotificationManager,
        )

        # Create XML for toast
        xml = f"""
        <toast>
            <visual>
                <binding template="ToastGeneric">
                    <text>{notification.title}</text>
                    <text>{notification.message}</text>
                </binding>
            </visual>
            <audio silent="{str(not notification.sound).lower()}"/>
        </toast>
        """
        
        doc = XmlDocument()
        doc.load_xml(xml)
        
        toast = ToastNotification(doc)
        
        notifier = ToastNotificationManager.create_toast_notifier("Enigma AI Engine")
        notifier.show(toast)
        
        return True
    
    def _get_icon_path(self, icon_name: str) -> str | None:
        """Get icon file path."""
        icon_dir = Path(__file__).parent.parent / "data" / "icons"
        icon_file = icon_dir / f"{icon_name}.ico"
        
        if icon_file.exists():
            return str(icon_file)
        return None
    
    def play_sound(self, sound_type: str):
        """Play Windows notification sound."""
        try:
            import winsound
            sounds = {
                "info": winsound.MB_OK,
                "success": winsound.MB_OK,
                "warning": winsound.MB_ICONEXCLAMATION,
                "error": winsound.MB_ICONHAND,
            }
            winsound.MessageBeep(sounds.get(sound_type, winsound.MB_OK))
        except Exception:
            pass


class LinuxNotificationBackend(NotificationBackend):
    """Linux notification backend using notify2 or dbus."""
    
    def __init__(self):
        self._notify2 = None
        self._dbus = None
        
        # Try notify2
        try:
            import notify2
            notify2.init("Enigma AI Engine")
            self._notify2 = notify2
            logger.info("Using notify2 backend")
        except ImportError:
            pass
        
        # Try dbus directly
        if self._notify2 is None:
            try:
                import dbus
                self._dbus = dbus
                logger.info("Using dbus backend")
            except ImportError:
                pass
    
    def send(self, notification: Notification) -> bool:
        """Send a Linux notification."""
        if self._notify2:
            try:
                n = self._notify2.Notification(
                    notification.title,
                    notification.message
                )
                n.timeout = notification.timeout
                
                # Set urgency
                urgency_map = {
                    NotificationPriority.LOW: self._notify2.URGENCY_LOW,
                    NotificationPriority.NORMAL: self._notify2.URGENCY_NORMAL,
                    NotificationPriority.HIGH: self._notify2.URGENCY_CRITICAL,
                    NotificationPriority.URGENT: self._notify2.URGENCY_CRITICAL,
                }
                n.set_urgency(urgency_map.get(notification.priority, self._notify2.URGENCY_NORMAL))
                
                n.show()
                return True
            except Exception as e:
                logger.error(f"notify2 error: {e}")
        
        if self._dbus:
            try:
                return self._send_dbus(notification)
            except Exception as e:
                logger.error(f"dbus error: {e}")
        
        # Fallback
        logger.info(f"[NOTIFICATION] {notification.title}: {notification.message}")
        return True
    
    def _send_dbus(self, notification: Notification) -> bool:
        """Send using D-Bus directly."""
        bus = self._dbus.SessionBus()
        notify_obj = bus.get_object(
            "org.freedesktop.Notifications",
            "/org/freedesktop/Notifications"
        )
        notify_interface = self._dbus.Interface(
            notify_obj,
            "org.freedesktop.Notifications"
        )
        
        notify_interface.Notify(
            "Enigma AI Engine",  # app_name
            0,  # replaces_id
            "",  # app_icon
            notification.title,
            notification.message,
            [],  # actions
            {},  # hints
            notification.timeout
        )
        
        return True
    
    def play_sound(self, sound_type: str):
        """Play notification sound on Linux."""
        try:
            import subprocess

            # Try paplay (PulseAudio)
            sound_files = {
                "info": "/usr/share/sounds/freedesktop/stereo/message.oga",
                "success": "/usr/share/sounds/freedesktop/stereo/complete.oga",
                "warning": "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga",
                "error": "/usr/share/sounds/freedesktop/stereo/dialog-error.oga",
            }
            sound_file = sound_files.get(sound_type, sound_files["info"])
            subprocess.Popen(["paplay", sound_file], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        except Exception:
            pass


class MacOSNotificationBackend(NotificationBackend):
    """macOS notification backend using pync or osascript."""
    
    def __init__(self):
        self._pync = None
        
        try:
            import pync
            self._pync = pync
            logger.info("Using pync backend")
        except ImportError:
            logger.info("Using osascript backend")
    
    def send(self, notification: Notification) -> bool:
        """Send a macOS notification."""
        if self._pync:
            try:
                self._pync.notify(
                    notification.message,
                    title=notification.title,
                    sound="default" if notification.sound else None
                )
                return True
            except Exception as e:
                logger.error(f"pync error: {e}")
        
        # Fallback to osascript
        try:
            import subprocess
            script = f'''
            display notification "{notification.message}" with title "{notification.title}"
            '''
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
            return True
        except Exception as e:
            logger.error(f"osascript error: {e}")
        
        logger.info(f"[NOTIFICATION] {notification.title}: {notification.message}")
        return True
    
    def play_sound(self, sound_type: str):
        """Play macOS notification sound."""
        try:
            import subprocess
            sounds = {
                "info": "Ping",
                "success": "Glass",
                "warning": "Basso",
                "error": "Funk",
            }
            sound = sounds.get(sound_type, "Ping")
            subprocess.Popen(
                ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass


class PlyrNotificationBackend(NotificationBackend):
    """Cross-platform backend using plyer library."""
    
    def __init__(self):
        self._plyer = None
        
        try:
            from plyer import notification
            self._plyer = notification
            logger.info("Using plyer backend")
        except ImportError:
            pass
    
    def send(self, notification_obj: Notification) -> bool:
        """Send using plyer."""
        if self._plyer:
            try:
                self._plyer.notify(
                    title=notification_obj.title,
                    message=notification_obj.message,
                    timeout=notification_obj.timeout // 1000,
                    app_name="Enigma AI Engine"
                )
                return True
            except Exception as e:
                logger.error(f"plyer error: {e}")
        
        logger.info(f"[NOTIFICATION] {notification_obj.title}: {notification_obj.message}")
        return True


class NotificationSettings:
    """Per-type notification settings."""
    
    def __init__(self):
        self.enabled = True
        self.sound = True
        self.priority_override: NotificationPriority | None = None
        self.timeout_override: int | None = None
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "sound": self.sound,
            "priority_override": self.priority_override.value if self.priority_override else None,
            "timeout_override": self.timeout_override,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> NotificationSettings:
        settings = cls()
        settings.enabled = data.get("enabled", True)
        settings.sound = data.get("sound", True)
        if data.get("priority_override"):
            settings.priority_override = NotificationPriority(data["priority_override"])
        settings.timeout_override = data.get("timeout_override")
        return settings


class DNDSchedule:
    """Do Not Disturb schedule."""
    
    def __init__(self):
        self.enabled = False
        self.start_hour = 22  # 10 PM
        self.start_minute = 0
        self.end_hour = 8     # 8 AM
        self.end_minute = 0
        self.days = [0, 1, 2, 3, 4, 5, 6]  # All days (0=Monday)
    
    def is_active(self) -> bool:
        """Check if DND should be active right now."""
        if not self.enabled:
            return False
        
        now = datetime.now()
        current_day = now.weekday()
        
        if current_day not in self.days:
            return False
        
        current_minutes = now.hour * 60 + now.minute
        start_minutes = self.start_hour * 60 + self.start_minute
        end_minutes = self.end_hour * 60 + self.end_minute
        
        # Handle overnight schedules (e.g., 22:00 - 08:00)
        if start_minutes > end_minutes:
            return current_minutes >= start_minutes or current_minutes < end_minutes
        else:
            return start_minutes <= current_minutes < end_minutes
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "start_hour": self.start_hour,
            "start_minute": self.start_minute,
            "end_hour": self.end_hour,
            "end_minute": self.end_minute,
            "days": self.days,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> DNDSchedule:
        schedule = cls()
        schedule.enabled = data.get("enabled", False)
        schedule.start_hour = data.get("start_hour", 22)
        schedule.start_minute = data.get("start_minute", 0)
        schedule.end_hour = data.get("end_hour", 8)
        schedule.end_minute = data.get("end_minute", 0)
        schedule.days = data.get("days", [0, 1, 2, 3, 4, 5, 6])
        return schedule


class NotificationManager:
    """
    Cross-platform notification manager.
    
    Features:
    - Desktop notifications
    - Notification history
    - Do Not Disturb mode with scheduling
    - Sound alerts
    - Per-type notification settings
    - Callbacks for notification events
    """
    
    def __init__(self):
        self.history: list[Notification] = []
        self.max_history = 100
        self.do_not_disturb = False
        self.dnd_schedule = DNDSchedule()
        self.sound_enabled = True
        self.type_settings: dict[NotificationType, NotificationSettings] = {
            t: NotificationSettings() for t in NotificationType
        }
        self._backend: NotificationBackend | None = None
        self._callbacks: dict[str, list[Callable]] = {
            "on_send": [],
            "on_click": [],
            "on_dismiss": [],
        }
        self._lock = threading.Lock()
        self._next_id = 1
        self._settings_file = Path.home() / ".enigma_engine" / "notification_settings.json"
        
        # Load saved settings
        self._load_settings()
        
        # Initialize backend
        self._init_backend()
    
    def _load_settings(self):
        """Load notification settings from file."""
        try:
            if self._settings_file.exists():
                data = json.loads(self._settings_file.read_text())
                self.do_not_disturb = data.get("do_not_disturb", False)
                self.sound_enabled = data.get("sound_enabled", True)
                
                if "dnd_schedule" in data:
                    self.dnd_schedule = DNDSchedule.from_dict(data["dnd_schedule"])
                
                if "type_settings" in data:
                    for type_str, settings_data in data["type_settings"].items():
                        try:
                            ntype = NotificationType(type_str)
                            self.type_settings[ntype] = NotificationSettings.from_dict(settings_data)
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug(f"Could not load notification settings: {e}")
    
    def _save_settings(self):
        """Save notification settings to file."""
        try:
            self._settings_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "do_not_disturb": self.do_not_disturb,
                "sound_enabled": self.sound_enabled,
                "dnd_schedule": self.dnd_schedule.to_dict(),
                "type_settings": {
                    t.value: s.to_dict() for t, s in self.type_settings.items()
                },
            }
            self._settings_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Could not save notification settings: {e}")
    
    def set_type_settings(self, ntype: NotificationType, enabled: bool = None,
                          sound: bool = None, priority: NotificationPriority = None,
                          timeout: int = None):
        """Configure settings for a notification type."""
        settings = self.type_settings[ntype]
        if enabled is not None:
            settings.enabled = enabled
        if sound is not None:
            settings.sound = sound
        if priority is not None:
            settings.priority_override = priority
        if timeout is not None:
            settings.timeout_override = timeout
        self._save_settings()
    
    def set_dnd_schedule(self, enabled: bool = None, start_hour: int = None,
                         start_minute: int = None, end_hour: int = None,
                         end_minute: int = None, days: list = None):
        """Configure DND schedule."""
        if enabled is not None:
            self.dnd_schedule.enabled = enabled
        if start_hour is not None:
            self.dnd_schedule.start_hour = start_hour
        if start_minute is not None:
            self.dnd_schedule.start_minute = start_minute
        if end_hour is not None:
            self.dnd_schedule.end_hour = end_hour
        if end_minute is not None:
            self.dnd_schedule.end_minute = end_minute
        if days is not None:
            self.dnd_schedule.days = days
        self._save_settings()
    
    def is_dnd_active(self) -> bool:
        """Check if DND is currently active (manual or scheduled)."""
        return self.do_not_disturb or self.dnd_schedule.is_active()
    
    def _init_backend(self):
        """Initialize the appropriate backend."""
        # Try platform-specific backends first
        if PLATFORM == "windows":
            backend = WindowsNotificationBackend()
            if backend._toaster or backend._winrt:
                self._backend = backend
                return
        elif PLATFORM == "linux":
            backend = LinuxNotificationBackend()
            if backend._notify2 or backend._dbus:
                self._backend = backend
                return
        elif PLATFORM == "darwin":
            self._backend = MacOSNotificationBackend()
            return
        
        # Fallback to plyer
        backend = PlyrNotificationBackend()
        if backend._plyer:
            self._backend = backend
            return
        
        logger.warning("No notification backend available")
    
    def send(
        self,
        title: str,
        message: str,
        type: str | NotificationType = NotificationType.INFO,
        priority: str | NotificationPriority = NotificationPriority.NORMAL,
        icon: str | None = None,
        sound: bool | None = None,
        timeout: int = 5000,
        actions: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Send a notification.
        
        Args:
            title: Notification title
            message: Notification message
            type: Type (info, success, warning, error)
            priority: Priority level
            icon: Custom icon name
            sound: Play sound (None = use default)
            timeout: Display timeout in milliseconds
            actions: List of action buttons
            
        Returns:
            Notification ID
        """
        # Parse enums
        if isinstance(type, str):
            type = NotificationType(type)
        if isinstance(priority, str):
            priority = NotificationPriority(priority)
        
        # Check per-type settings
        type_settings = self.type_settings.get(type)
        if type_settings:
            # Check if this type is disabled
            if not type_settings.enabled:
                logger.debug(f"Notification suppressed (type disabled): {title}")
                return ""
            
            # Apply overrides
            if type_settings.priority_override:
                priority = type_settings.priority_override
            if type_settings.timeout_override:
                timeout = type_settings.timeout_override
            if sound is None:
                sound = type_settings.sound
        
        # Check DND (manual or scheduled)
        if self.is_dnd_active() and priority not in (NotificationPriority.HIGH, NotificationPriority.URGENT):
            logger.debug(f"Notification suppressed (DND): {title}")
            return ""
        
        with self._lock:
            # Create notification
            notification_id = f"notif_{self._next_id}"
            self._next_id += 1
            
            notification = Notification(
                id=notification_id,
                title=title,
                message=message,
                type=type,
                priority=priority,
                icon=icon,
                sound=sound if sound is not None else self.sound_enabled,
                timeout=timeout,
                actions=actions or []
            )
            
            # Add to history
            self.history.append(notification)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Send notification
            if self._backend:
                success = self._backend.send(notification)
                
                # Play sound
                if notification.sound:
                    self._backend.play_sound(type.value)
            else:
                # Console fallback
                print(f"[{type.value.upper()}] {title}: {message}")
                success = True
            
            # Trigger callbacks
            for callback in self._callbacks["on_send"]:
                try:
                    callback(notification)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            return notification_id
    
    def info(self, title: str, message: str, **kwargs) -> str:
        """Send info notification."""
        return self.send(title, message, type=NotificationType.INFO, **kwargs)
    
    def success(self, title: str, message: str, **kwargs) -> str:
        """Send success notification."""
        return self.send(title, message, type=NotificationType.SUCCESS, **kwargs)
    
    def warning(self, title: str, message: str, **kwargs) -> str:
        """Send warning notification."""
        return self.send(title, message, type=NotificationType.WARNING, 
                        priority=NotificationPriority.HIGH, **kwargs)
    
    def error(self, title: str, message: str, **kwargs) -> str:
        """Send error notification."""
        return self.send(title, message, type=NotificationType.ERROR,
                        priority=NotificationPriority.URGENT, **kwargs)
    
    def progress(
        self,
        title: str,
        message: str,
        progress: float,
        **kwargs
    ) -> str:
        """
        Send progress notification.
        
        Args:
            title: Notification title
            message: Message with {progress} placeholder
            progress: Progress value (0-1 or 0-100)
        """
        if progress > 1:
            progress = progress / 100
        
        message = message.format(progress=f"{progress*100:.0f}%")
        return self.send(title, message, type=NotificationType.PROGRESS, **kwargs)
    
    def get_history(
        self,
        limit: int | None = None,
        unread_only: bool = False
    ) -> list[Notification]:
        """Get notification history."""
        notifications = self.history
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        if limit:
            notifications = notifications[-limit:]
        
        return notifications
    
    def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        for n in self.history:
            if n.id == notification_id:
                n.read = True
                return True
        return False
    
    def mark_all_read(self):
        """Mark all notifications as read."""
        for n in self.history:
            n.read = True
    
    def dismiss(self, notification_id: str) -> bool:
        """Dismiss a notification."""
        for n in self.history:
            if n.id == notification_id:
                n.dismissed = True
                for callback in self._callbacks["on_dismiss"]:
                    try:
                        callback(n)
                    except Exception:
                        pass
                return True
        return False
    
    def clear_history(self):
        """Clear notification history."""
        self.history.clear()
    
    def set_dnd(self, enabled: bool):
        """Set Do Not Disturb mode."""
        self.do_not_disturb = enabled
        logger.info(f"Do Not Disturb: {enabled}")
    
    def on(self, event: str, callback: Callable):
        """
        Register event callback.
        
        Events:
        - on_send: Called when notification is sent
        - on_click: Called when notification is clicked
        - on_dismiss: Called when notification is dismissed
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)


# Global instance
_notification_manager: NotificationManager | None = None
_manager_lock = threading.Lock()


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager."""
    global _notification_manager
    
    with _manager_lock:
        if _notification_manager is None:
            _notification_manager = NotificationManager()
        return _notification_manager


# Convenience functions
def notify(title: str, message: str, **kwargs) -> str:
    """Send a notification (convenience function)."""
    return get_notification_manager().send(title, message, **kwargs)


def notify_info(title: str, message: str, **kwargs) -> str:
    """Send info notification."""
    return get_notification_manager().info(title, message, **kwargs)


def notify_success(title: str, message: str, **kwargs) -> str:
    """Send success notification."""
    return get_notification_manager().success(title, message, **kwargs)


def notify_warning(title: str, message: str, **kwargs) -> str:
    """Send warning notification."""
    return get_notification_manager().warning(title, message, **kwargs)


def notify_error(title: str, message: str, **kwargs) -> str:
    """Send error notification."""
    return get_notification_manager().error(title, message, **kwargs)
