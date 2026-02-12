"""
Webhook System - Send events to external URLs.

Provides webhook functionality for event notifications:
- Webhook registration and management
- Event delivery with retries
- Signature verification
- Rate limiting
- Async delivery

Part of the Enigma AI Engine networking utilities.
"""

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Optional


class WebhookEvent(Enum):
    """Standard webhook event types."""
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    
    # Model events
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETE = "training.complete"
    TRAINING_FAILED = "training.failed"
    
    # Chat events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_ENDED = "conversation.ended"
    MESSAGE_RECEIVED = "message.received"
    RESPONSE_GENERATED = "response.generated"
    
    # Tool events
    TOOL_EXECUTED = "tool.executed"
    TOOL_FAILED = "tool.failed"
    
    # Module events
    MODULE_LOADED = "module.loaded"
    MODULE_UNLOADED = "module.unloaded"
    
    # Custom
    CUSTOM = "custom"


class DeliveryStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    id: str
    url: str
    events: list[WebhookEvent]
    secret: Optional[str] = None
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay_seconds: int = 5
    timeout_seconds: int = 30
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "secret": "***" if self.secret else None,
            "enabled": self.enabled,
            "headers": {k: "***" if "auth" in k.lower() else v for k, v in self.headers.items()},
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    webhook_id: str
    event_type: WebhookEvent
    payload: dict[str, Any]
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    last_error: Optional[str] = None
    response_code: Optional[int] = None
    delivered_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "webhook_id": self.webhook_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "last_error": self.last_error,
            "response_code": self.response_code,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None
        }


class WebhookManager:
    """
    Manage webhooks and event delivery.
    
    Usage:
        manager = WebhookManager()
        
        # Register a webhook
        manager.register(
            id="slack-notifications",
            url="https://hooks.slack.com/services/...",
            events=[WebhookEvent.TRAINING_COMPLETE, WebhookEvent.SYSTEM_ERROR],
            secret="my-signing-secret"
        )
        
        # Send an event
        manager.emit(WebhookEvent.TRAINING_COMPLETE, {
            "model": "forge-base",
            "epochs": 10,
            "loss": 0.05
        })
        
        # Start async delivery
        manager.start()
        
        # Cleanup
        manager.stop()
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        async_delivery: bool = True
    ):
        """
        Initialize webhook manager.
        
        Args:
            config_file: Optional file to persist webhook configs
            async_delivery: Whether to deliver webhooks asynchronously
        """
        self._webhooks: dict[str, WebhookConfig] = {}
        self._delivery_history: list[WebhookDelivery] = []
        self._history_limit = 1000
        self._config_file = Path(config_file) if config_file else None
        self._async_delivery = async_delivery
        
        # Async delivery
        self._queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_delivery: Optional[Callable[[WebhookDelivery], None]] = None
        self._on_failure: Optional[Callable[[WebhookDelivery], None]] = None
        
        # Load persisted config
        if self._config_file and self._config_file.exists():
            self._load_config()
    
    def register(
        self,
        id: str,
        url: str,
        events: list[WebhookEvent],
        secret: Optional[str] = None,
        enabled: bool = True,
        headers: Optional[dict[str, str]] = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        timeout_seconds: int = 30
    ) -> WebhookConfig:
        """
        Register a new webhook.
        
        Args:
            id: Unique webhook identifier
            url: Webhook endpoint URL
            events: Events to subscribe to
            secret: Secret for signing payloads
            enabled: Whether webhook is active
            headers: Additional headers to send
            max_retries: Max delivery attempts
            retry_delay_seconds: Delay between retries
            timeout_seconds: Request timeout
            
        Returns:
            Created webhook config
        """
        config = WebhookConfig(
            id=id,
            url=url,
            events=events,
            secret=secret,
            enabled=enabled,
            headers=headers or {},
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds
        )
        
        with self._lock:
            self._webhooks[id] = config
        
        self._save_config()
        return config
    
    def unregister(self, id: str) -> bool:
        """
        Remove a webhook.
        
        Args:
            id: Webhook identifier
            
        Returns:
            True if removed
        """
        with self._lock:
            if id in self._webhooks:
                del self._webhooks[id]
                self._save_config()
                return True
        return False
    
    def get_webhook(self, id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID."""
        return self._webhooks.get(id)
    
    def list_webhooks(self) -> list[WebhookConfig]:
        """List all webhooks."""
        return list(self._webhooks.values())
    
    def enable(self, id: str) -> bool:
        """Enable a webhook."""
        webhook = self._webhooks.get(id)
        if webhook:
            webhook.enabled = True
            self._save_config()
            return True
        return False
    
    def disable(self, id: str) -> bool:
        """Disable a webhook."""
        webhook = self._webhooks.get(id)
        if webhook:
            webhook.enabled = False
            self._save_config()
            return True
        return False
    
    def emit(
        self,
        event: WebhookEvent,
        payload: dict[str, Any],
        webhook_ids: Optional[list[str]] = None
    ) -> list[str]:
        """
        Emit an event to subscribed webhooks.
        
        Args:
            event: Event type
            payload: Event data
            webhook_ids: Specific webhooks (all subscribers if None)
            
        Returns:
            List of webhook IDs that will receive the event
        """
        # Add standard event metadata
        full_payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": payload
        }
        
        # Find subscribed webhooks
        target_ids = []
        
        for webhook_id, config in self._webhooks.items():
            if not config.enabled:
                continue
            
            if webhook_ids and webhook_id not in webhook_ids:
                continue
            
            if event not in config.events and WebhookEvent.CUSTOM not in config.events:
                continue
            
            target_ids.append(webhook_id)
            
            # Create delivery record
            delivery = WebhookDelivery(
                webhook_id=webhook_id,
                event_type=event,
                payload=full_payload
            )
            
            if self._async_delivery and self._running:
                self._queue.put(delivery)
            else:
                self._deliver(delivery)
        
        return target_ids
    
    def _deliver(self, delivery: WebhookDelivery) -> bool:
        """
        Deliver a webhook event.
        
        Args:
            delivery: Delivery record
            
        Returns:
            True if successful
        """
        webhook = self._webhooks.get(delivery.webhook_id)
        if not webhook or not webhook.enabled:
            delivery.status = DeliveryStatus.FAILED
            delivery.last_error = "Webhook not found or disabled"
            return False
        
        # Prepare payload
        json_payload = json.dumps(delivery.payload).encode('utf-8')
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Enigma AI Engine-Webhook/1.0',
            'X-Webhook-Event': delivery.event_type.value,
            'X-Webhook-ID': webhook.id,
            'X-Delivery-Timestamp': str(int(time.time()))
        }
        headers.update(webhook.headers)
        
        # Sign payload if secret configured
        if webhook.secret:
            signature = hmac.new(
                webhook.secret.encode('utf-8'),
                json_payload,
                hashlib.sha256
            ).hexdigest()
            headers['X-Webhook-Signature'] = f"sha256={signature}"
        
        # Attempt delivery with retries
        while delivery.attempts < webhook.max_retries:
            delivery.attempts += 1
            delivery.last_attempt = datetime.now()
            
            try:
                req = urllib.request.Request(
                    webhook.url,
                    data=json_payload,
                    headers=headers,
                    method='POST'
                )
                
                with urllib.request.urlopen(req, timeout=webhook.timeout_seconds) as response:
                    delivery.response_code = response.status
                    
                    if 200 <= response.status < 300:
                        delivery.status = DeliveryStatus.DELIVERED
                        delivery.delivered_at = datetime.now()
                        self._record_delivery(delivery)
                        
                        if self._on_delivery:
                            self._on_delivery(delivery)
                        
                        return True
                    else:
                        delivery.last_error = f"HTTP {response.status}"
                        
            except urllib.error.HTTPError as e:
                delivery.response_code = e.code
                delivery.last_error = f"HTTP {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                delivery.last_error = f"URL Error: {e.reason}"
            except Exception as e:
                delivery.last_error = str(e)
            
            # Mark as retrying
            if delivery.attempts < webhook.max_retries:
                delivery.status = DeliveryStatus.RETRYING
                time.sleep(webhook.retry_delay_seconds)
            else:
                delivery.status = DeliveryStatus.FAILED
        
        self._record_delivery(delivery)
        
        if self._on_failure:
            self._on_failure(delivery)
        
        return False
    
    def _record_delivery(self, delivery: WebhookDelivery):
        """Record delivery in history."""
        with self._lock:
            self._delivery_history.append(delivery)
            
            # Trim history
            if len(self._delivery_history) > self._history_limit:
                self._delivery_history = self._delivery_history[-self._history_limit:]
    
    def get_delivery_history(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 100
    ) -> list[WebhookDelivery]:
        """Get delivery history."""
        history = self._delivery_history
        
        if webhook_id:
            history = [d for d in history if d.webhook_id == webhook_id]
        
        return history[-limit:]
    
    def start(self):
        """Start async delivery worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._delivery_worker,
            daemon=True
        )
        self._worker_thread.start()
    
    def stop(self):
        """Stop async delivery worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None
    
    def _delivery_worker(self):
        """Background worker for async delivery."""
        while self._running:
            try:
                delivery = self._queue.get(timeout=1)
                self._deliver(delivery)
            except Empty:
                continue
            except Exception:
                pass  # Intentionally silent
    
    def on_delivery(self, callback: Callable[[WebhookDelivery], None]):
        """Set callback for successful deliveries."""
        self._on_delivery = callback
    
    def on_failure(self, callback: Callable[[WebhookDelivery], None]):
        """Set callback for failed deliveries."""
        self._on_failure = callback
    
    def test_webhook(self, id: str) -> WebhookDelivery:
        """
        Send a test event to a webhook.
        
        Args:
            id: Webhook identifier
            
        Returns:
            Delivery result
        """
        delivery = WebhookDelivery(
            webhook_id=id,
            event_type=WebhookEvent.CUSTOM,
            payload={
                "event": "webhook.test",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "message": "This is a test webhook delivery",
                    "webhook_id": id
                }
            }
        )
        
        self._deliver(delivery)
        return delivery
    
    def _save_config(self):
        """Save webhook configs to file."""
        if not self._config_file:
            return
        
        configs = {}
        for id, config in self._webhooks.items():
            configs[id] = {
                "url": config.url,
                "events": [e.value for e in config.events],
                "secret": config.secret,
                "enabled": config.enabled,
                "headers": config.headers,
                "max_retries": config.max_retries,
                "retry_delay_seconds": config.retry_delay_seconds,
                "timeout_seconds": config.timeout_seconds
            }
        
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        self._config_file.write_text(json.dumps(configs, indent=2))
    
    def _load_config(self):
        """Load webhook configs from file."""
        if not self._config_file or not self._config_file.exists():
            return
        
        try:
            configs = json.loads(self._config_file.read_text())
            
            for id, data in configs.items():
                events = [WebhookEvent(e) for e in data.get("events", [])]
                
                self._webhooks[id] = WebhookConfig(
                    id=id,
                    url=data["url"],
                    events=events,
                    secret=data.get("secret"),
                    enabled=data.get("enabled", True),
                    headers=data.get("headers", {}),
                    max_retries=data.get("max_retries", 3),
                    retry_delay_seconds=data.get("retry_delay_seconds", 5),
                    timeout_seconds=data.get("timeout_seconds", 30)
                )
        except Exception:
            pass  # Intentionally silent


# Global webhook manager
_global_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = WebhookManager()
    return _global_manager


def register_webhook(
    id: str,
    url: str,
    events: list[WebhookEvent],
    **kwargs
) -> WebhookConfig:
    """Register webhook in global manager."""
    return get_webhook_manager().register(id, url, events, **kwargs)


def emit_event(event: WebhookEvent, payload: dict[str, Any]) -> list[str]:
    """Emit event via global manager."""
    return get_webhook_manager().emit(event, payload)


def start_webhook_delivery():
    """Start global webhook delivery."""
    get_webhook_manager().start()


def stop_webhook_delivery():
    """Stop global webhook delivery."""
    get_webhook_manager().stop()
