"""
Real-time Collaboration

Live editing, presence, and sync for multi-user sessions.

FILE: enigma_engine/collab/realtime.py
TYPE: Multi-User
MAIN CLASSES: CollaborationSession, PresenceManager, SyncEngine
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PresenceStatus(Enum):
    """User presence status."""
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"


class CollabEventType(Enum):
    """Collaboration event types."""
    JOIN = "join"
    LEAVE = "leave"
    MESSAGE = "message"
    TYPING = "typing"
    CURSOR = "cursor"
    SELECTION = "selection"
    EDIT = "edit"
    SYNC = "sync"
    PRESENCE = "presence"


@dataclass
class Cursor:
    """User cursor position."""
    user_id: str
    position: int  # Character position
    selection_start: int = None
    selection_end: int = None
    color: str = "#007ACC"


@dataclass
class UserPresence:
    """User presence information."""
    user_id: str
    username: str
    status: PresenceStatus = PresenceStatus.ONLINE
    last_active: float = field(default_factory=time.time)
    cursor: Cursor = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollabEvent:
    """Collaboration event."""
    event_type: CollabEventType
    session_id: str
    user_id: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Operation:
    """Operational transformation operation."""
    op_type: str  # insert, delete, retain
    position: int
    content: str = ""
    length: int = 0
    user_id: str = ""
    version: int = 0


class OperationalTransform:
    """
    Operational Transformation for concurrent editing.
    """
    
    @staticmethod
    def transform(op1: Operation, op2: Operation) -> Operation:
        """
        Transform op1 against op2 (concurrent operations).
        Returns transformed op1.
        """
        if op1.op_type == "insert":
            if op2.op_type == "insert":
                # Both inserts
                if op2.position <= op1.position:
                    return Operation(
                        op_type="insert",
                        position=op1.position + len(op2.content),
                        content=op1.content,
                        user_id=op1.user_id
                    )
            elif op2.op_type == "delete":
                # op1=insert, op2=delete
                if op2.position + op2.length <= op1.position:
                    return Operation(
                        op_type="insert",
                        position=op1.position - op2.length,
                        content=op1.content,
                        user_id=op1.user_id
                    )
                elif op2.position < op1.position:
                    return Operation(
                        op_type="insert",
                        position=op2.position,
                        content=op1.content,
                        user_id=op1.user_id
                    )
        
        elif op1.op_type == "delete":
            if op2.op_type == "insert":
                # op1=delete, op2=insert
                if op2.position <= op1.position:
                    return Operation(
                        op_type="delete",
                        position=op1.position + len(op2.content),
                        length=op1.length,
                        user_id=op1.user_id
                    )
            elif op2.op_type == "delete":
                # Both deletes
                if op2.position + op2.length <= op1.position:
                    return Operation(
                        op_type="delete",
                        position=op1.position - op2.length,
                        length=op1.length,
                        user_id=op1.user_id
                    )
                elif op2.position >= op1.position + op1.length:
                    return op1
                else:
                    # Overlapping deletes
                    if op2.position <= op1.position:
                        overlap = min(
                            op2.position + op2.length,
                            op1.position + op1.length
                        ) - op1.position
                        return Operation(
                            op_type="delete",
                            position=op2.position,
                            length=max(0, op1.length - overlap),
                            user_id=op1.user_id
                        )
        
        return op1
    
    @staticmethod
    def apply(document: str, op: Operation) -> str:
        """Apply operation to document."""
        if op.op_type == "insert":
            return document[:op.position] + op.content + document[op.position:]
        elif op.op_type == "delete":
            return document[:op.position] + document[op.position + op.length:]
        elif op.op_type == "retain":
            return document
        return document


class PresenceManager:
    """
    Manages user presence across sessions.
    """
    
    def __init__(self):
        self._presence: dict[str, dict[str, UserPresence]] = defaultdict(dict)
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._typing_timers: dict[str, float] = {}
    
    def join(
        self,
        session_id: str,
        user_id: str,
        username: str,
        metadata: dict = None
    ) -> UserPresence:
        """User joins session."""
        presence = UserPresence(
            user_id=user_id,
            username=username,
            metadata=metadata or {}
        )
        
        self._presence[session_id][user_id] = presence
        self._notify(session_id, "join", presence)
        
        return presence
    
    def leave(self, session_id: str, user_id: str):
        """User leaves session."""
        if user_id in self._presence.get(session_id, {}):
            presence = self._presence[session_id].pop(user_id)
            presence.status = PresenceStatus.OFFLINE
            self._notify(session_id, "leave", presence)
    
    def update_status(
        self,
        session_id: str,
        user_id: str,
        status: PresenceStatus
    ):
        """Update user status."""
        if user_id in self._presence.get(session_id, {}):
            self._presence[session_id][user_id].status = status
            self._presence[session_id][user_id].last_active = time.time()
            self._notify(session_id, "presence", self._presence[session_id][user_id])
    
    def update_cursor(
        self,
        session_id: str,
        user_id: str,
        position: int,
        selection_start: int = None,
        selection_end: int = None
    ):
        """Update user cursor position."""
        if user_id in self._presence.get(session_id, {}):
            cursor = Cursor(
                user_id=user_id,
                position=position,
                selection_start=selection_start,
                selection_end=selection_end
            )
            self._presence[session_id][user_id].cursor = cursor
            self._presence[session_id][user_id].last_active = time.time()
            self._notify(session_id, "cursor", cursor)
    
    def set_typing(self, session_id: str, user_id: str, is_typing: bool):
        """Set typing indicator."""
        key = f"{session_id}:{user_id}"
        
        if is_typing:
            self._typing_timers[key] = time.time() + 3.0  # 3 second timeout
            self._notify(session_id, "typing", {
                "user_id": user_id,
                "is_typing": True
            })
        else:
            self._typing_timers.pop(key, None)
            self._notify(session_id, "typing", {
                "user_id": user_id,
                "is_typing": False
            })
    
    def get_participants(self, session_id: str) -> list[UserPresence]:
        """Get all participants in session."""
        return list(self._presence.get(session_id, {}).values())
    
    def subscribe(self, session_id: str, callback: Callable):
        """Subscribe to presence updates."""
        self._callbacks[session_id].append(callback)
    
    def unsubscribe(self, session_id: str, callback: Callable):
        """Unsubscribe from updates."""
        if session_id in self._callbacks:
            self._callbacks[session_id] = [
                cb for cb in self._callbacks[session_id]
                if cb != callback
            ]
    
    def _notify(self, session_id: str, event: str, data: Any):
        """Notify subscribers."""
        for callback in self._callbacks.get(session_id, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event, data))
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Presence notification error: {e}")


class SyncEngine:
    """
    Synchronizes document state across clients.
    """
    
    def __init__(self):
        self._documents: dict[str, str] = {}
        self._versions: dict[str, int] = defaultdict(int)
        self._history: dict[str, list[Operation]] = defaultdict(list)
        self._pending: dict[str, list[Operation]] = defaultdict(list)
    
    def init_document(self, doc_id: str, content: str = ""):
        """Initialize document."""
        self._documents[doc_id] = content
        self._versions[doc_id] = 0
        self._history[doc_id] = []
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get document content."""
        return self._documents.get(doc_id)
    
    def get_version(self, doc_id: str) -> int:
        """Get document version."""
        return self._versions.get(doc_id, 0)
    
    def apply_operation(
        self,
        doc_id: str,
        operation: Operation,
        client_version: int
    ) -> tuple:
        """
        Apply operation with OT.
        
        Returns:
            (transformed_op, new_version, ack)
        """
        if doc_id not in self._documents:
            return None, 0, False
        
        server_version = self._versions[doc_id]
        
        # Transform against concurrent operations
        if client_version < server_version:
            # Get operations since client version
            concurrent_ops = self._history[doc_id][client_version:]
            
            transformed = operation
            for op in concurrent_ops:
                transformed = OperationalTransform.transform(transformed, op)
        else:
            transformed = operation
        
        # Apply to document
        self._documents[doc_id] = OperationalTransform.apply(
            self._documents[doc_id],
            transformed
        )
        
        # Update version and history
        self._versions[doc_id] += 1
        transformed.version = self._versions[doc_id]
        self._history[doc_id].append(transformed)
        
        # Trim history if too long
        if len(self._history[doc_id]) > 1000:
            self._history[doc_id] = self._history[doc_id][-500:]
        
        return transformed, self._versions[doc_id], True
    
    def get_diff(
        self,
        doc_id: str,
        from_version: int
    ) -> list[Operation]:
        """Get operations since version."""
        if doc_id not in self._history:
            return []
        
        return self._history[doc_id][from_version:]


class CollaborationSession:
    """
    Real-time collaboration session.
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        
        self.presence = PresenceManager()
        self.sync = SyncEngine()
        
        self._clients: dict[str, Any] = {}  # WebSocket connections
        self._event_handlers: dict[str, list[Callable]] = defaultdict(list)
    
    async def add_client(
        self,
        websocket,
        user_id: str,
        username: str
    ):
        """Add client connection."""
        self._clients[user_id] = websocket
        
        # Join presence
        presence = self.presence.join(
            self.session_id,
            user_id,
            username
        )
        
        # Send current state
        await self._send(user_id, {
            "type": "init",
            "session_id": self.session_id,
            "participants": [
                asdict(p) for p in self.presence.get_participants(self.session_id)
            ]
        })
        
        # Broadcast join
        await self._broadcast({
            "type": "join",
            "user_id": user_id,
            "username": username,
            "timestamp": time.time()
        }, exclude=user_id)
    
    async def remove_client(self, user_id: str):
        """Remove client connection."""
        self._clients.pop(user_id, None)
        self.presence.leave(self.session_id, user_id)
        
        await self._broadcast({
            "type": "leave",
            "user_id": user_id,
            "timestamp": time.time()
        })
    
    async def handle_message(self, user_id: str, message: dict):
        """Handle incoming message from client."""
        msg_type = message.get("type")
        
        if msg_type == "typing":
            self.presence.set_typing(
                self.session_id,
                user_id,
                message.get("is_typing", False)
            )
            await self._broadcast({
                "type": "typing",
                "user_id": user_id,
                "is_typing": message.get("is_typing", False)
            }, exclude=user_id)
        
        elif msg_type == "cursor":
            self.presence.update_cursor(
                self.session_id,
                user_id,
                message.get("position", 0),
                message.get("selection_start"),
                message.get("selection_end")
            )
            await self._broadcast({
                "type": "cursor",
                "user_id": user_id,
                "position": message.get("position", 0),
                "selection_start": message.get("selection_start"),
                "selection_end": message.get("selection_end")
            }, exclude=user_id)
        
        elif msg_type == "edit":
            doc_id = message.get("doc_id")
            operation = Operation(
                op_type=message.get("op_type", "insert"),
                position=message.get("position", 0),
                content=message.get("content", ""),
                length=message.get("length", 0),
                user_id=user_id
            )
            client_version = message.get("version", 0)
            
            transformed, new_version, ack = self.sync.apply_operation(
                doc_id,
                operation,
                client_version
            )
            
            if ack:
                # Send ack to sender
                await self._send(user_id, {
                    "type": "ack",
                    "doc_id": doc_id,
                    "version": new_version
                })
                
                # Broadcast to others
                await self._broadcast({
                    "type": "edit",
                    "doc_id": doc_id,
                    "op_type": transformed.op_type,
                    "position": transformed.position,
                    "content": transformed.content,
                    "length": transformed.length,
                    "user_id": user_id,
                    "version": new_version
                }, exclude=user_id)
        
        elif msg_type == "message":
            # Chat message
            await self._broadcast({
                "type": "message",
                "user_id": user_id,
                "content": message.get("content", ""),
                "timestamp": time.time()
            })
        
        elif msg_type == "sync":
            # Client requests sync
            doc_id = message.get("doc_id")
            from_version = message.get("version", 0)
            
            ops = self.sync.get_diff(doc_id, from_version)
            current = self.sync.get_document(doc_id)
            
            await self._send(user_id, {
                "type": "sync",
                "doc_id": doc_id,
                "content": current,
                "version": self.sync.get_version(doc_id),
                "operations": [asdict(op) for op in ops]
            })
        
        # Fire event handlers
        event = CollabEvent(
            event_type=CollabEventType(msg_type) if msg_type in CollabEventType.__members__.values() else CollabEventType.MESSAGE,
            session_id=self.session_id,
            user_id=user_id,
            data=message
        )
        
        for handler in self._event_handlers.get(msg_type, []):
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def on(self, event_type: str, handler: Callable):
        """Register event handler."""
        self._event_handlers[event_type].append(handler)
    
    async def _send(self, user_id: str, message: dict):
        """Send message to specific client."""
        if user_id in self._clients:
            try:
                await self._clients[user_id].send(json.dumps(message))
            except Exception as e:
                logger.error(f"Send error to {user_id}: {e}")
    
    async def _broadcast(
        self,
        message: dict,
        exclude: str = None
    ):
        """Broadcast message to all clients."""
        data = json.dumps(message)
        
        for user_id, ws in list(self._clients.items()):
            if user_id == exclude:
                continue
            
            try:
                await ws.send(data)
            except Exception as e:
                logger.error(f"Broadcast error to {user_id}: {e}")


if HAS_WEBSOCKETS:
    
    class CollaborationServer:
        """
        WebSocket server for real-time collaboration.
        """
        
        def __init__(self, host: str = "localhost", port: int = 8765):
            self.host = host
            self.port = port
            
            self._sessions: dict[str, CollaborationSession] = {}
            self._user_sessions: dict[str, str] = {}  # user_id -> session_id
        
        def get_or_create_session(
            self,
            session_id: str = None
        ) -> CollaborationSession:
            """Get or create collaboration session."""
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            
            session = CollaborationSession(session_id)
            self._sessions[session.session_id] = session
            return session
        
        async def handle_connection(
            self,
            websocket,
            path: str
        ):
            """Handle WebSocket connection."""
            user_id = None
            session = None
            
            try:
                # Wait for auth message
                auth_msg = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=10.0
                )
                
                auth = json.loads(auth_msg)
                user_id = auth.get("user_id")
                username = auth.get("username", "Anonymous")
                session_id = auth.get("session_id")
                
                if not user_id:
                    await websocket.close(1008, "Authentication required")
                    return
                
                # Get session
                session = self.get_or_create_session(session_id)
                self._user_sessions[user_id] = session.session_id
                
                # Add client
                await session.add_client(websocket, user_id, username)
                
                # Message loop
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await session.handle_message(user_id, data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from {user_id}")
            
            except asyncio.TimeoutError:
                logger.warning("Connection timeout")
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection closed for {user_id}")
            except Exception as e:
                logger.error(f"Connection error: {e}")
            finally:
                if user_id and session:
                    await session.remove_client(user_id)
                    self._user_sessions.pop(user_id, None)
        
        async def start(self):
            """Start collaboration server."""
            async with websockets.serve(
                self.handle_connection,
                self.host,
                self.port
            ):
                logger.info(f"Collaboration server running on ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever
        
        def run(self):
            """Run server (blocking)."""
            asyncio.run(self.start())

else:
    class CollaborationServer:
        pass


def create_collab_session(
    session_id: str = None
) -> CollaborationSession:
    """Create collaboration session."""
    return CollaborationSession(session_id)


def create_collab_server(
    host: str = "localhost",
    port: int = 8765
) -> 'CollaborationServer':
    """Create collaboration server."""
    if not HAS_WEBSOCKETS:
        raise ImportError("websockets library required")
    
    return CollaborationServer(host, port)
