"""
Shared Conversations

Multi-user conversation sharing and permissions.

FILE: enigma_engine/collab/shared_conversations.py
TYPE: Multi-User
MAIN CLASSES: SharedConversation, ConversationRoom, SharingManager
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharePermission(Enum):
    """Sharing permission levels."""
    VIEW = "view"  # Can read messages
    COMMENT = "comment"  # Can add comments
    PARTICIPATE = "participate"  # Can send messages
    MODERATE = "moderate"  # Can edit/delete
    ADMIN = "admin"  # Full control


class ConversationVisibility(Enum):
    """Conversation visibility."""
    PRIVATE = "private"  # Only owner
    SHARED = "shared"  # Specific users
    TEAM = "team"  # Team members
    PUBLIC = "public"  # Anyone


@dataclass
class Message:
    """Conversation message."""
    message_id: str
    conversation_id: str
    user_id: str
    role: str  # user, assistant, system
    content: str
    created_at: float = field(default_factory=time.time)
    edited_at: float = None
    reply_to: str = None  # Parent message ID
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Participant:
    """Conversation participant."""
    user_id: str
    permission: SharePermission
    joined_at: float = field(default_factory=time.time)
    last_read: float = None
    nickname: str = ""


@dataclass
class SharedConversation:
    """Shared conversation state."""
    conversation_id: str
    title: str
    owner_id: str
    visibility: ConversationVisibility = ConversationVisibility.PRIVATE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    participants: dict[str, Participant] = field(default_factory=dict)
    message_count: int = 0
    is_archived: bool = False
    settings: dict[str, Any] = field(default_factory=dict)


class MessageBroadcaster:
    """Broadcasts messages to participants."""
    
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}
    
    def subscribe(self, conversation_id: str, callback: Callable):
        """Subscribe to conversation updates."""
        if conversation_id not in self._listeners:
            self._listeners[conversation_id] = []
        self._listeners[conversation_id].append(callback)
    
    def unsubscribe(self, conversation_id: str, callback: Callable):
        """Unsubscribe from updates."""
        if conversation_id in self._listeners:
            self._listeners[conversation_id] = [
                cb for cb in self._listeners[conversation_id]
                if cb != callback
            ]
    
    def broadcast(self, conversation_id: str, event: str, data: dict):
        """Broadcast event to subscribers."""
        if conversation_id in self._listeners:
            for callback in self._listeners[conversation_id]:
                try:
                    callback(event, data)
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")


if HAS_SQLITE:
    
    class ConversationDatabase:
        """SQLite storage for shared conversations."""
        
        def __init__(self, db_path: str):
            self.db_path = db_path
            self._init_db()
        
        def _init_db(self):
            """Initialize database schema."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Conversations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        conversation_id TEXT PRIMARY KEY,
                        title TEXT,
                        owner_id TEXT,
                        visibility TEXT,
                        created_at REAL,
                        updated_at REAL,
                        message_count INTEGER,
                        is_archived INTEGER,
                        settings TEXT
                    )
                """)
                
                # Messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT,
                        user_id TEXT,
                        role TEXT,
                        content TEXT,
                        created_at REAL,
                        edited_at REAL,
                        reply_to TEXT,
                        metadata TEXT,
                        FOREIGN KEY (conversation_id) 
                            REFERENCES conversations(conversation_id)
                    )
                """)
                
                # Participants table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS participants (
                        conversation_id TEXT,
                        user_id TEXT,
                        permission TEXT,
                        joined_at REAL,
                        last_read REAL,
                        nickname TEXT,
                        PRIMARY KEY (conversation_id, user_id),
                        FOREIGN KEY (conversation_id) 
                            REFERENCES conversations(conversation_id)
                    )
                """)
                
                # Indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_conv 
                    ON messages(conversation_id, created_at)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_participants_user 
                    ON participants(user_id)
                """)
                
                conn.commit()
        
        def create_conversation(
            self,
            title: str,
            owner_id: str,
            visibility: ConversationVisibility = ConversationVisibility.PRIVATE
        ) -> SharedConversation:
            """Create new conversation."""
            conv = SharedConversation(
                conversation_id=str(uuid.uuid4()),
                title=title,
                owner_id=owner_id,
                visibility=visibility
            )
            
            # Add owner as admin participant
            conv.participants[owner_id] = Participant(
                user_id=owner_id,
                permission=SharePermission.ADMIN
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO conversations (
                        conversation_id, title, owner_id, visibility,
                        created_at, updated_at, message_count, is_archived, settings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conv.conversation_id,
                    conv.title,
                    conv.owner_id,
                    conv.visibility.value,
                    conv.created_at,
                    conv.updated_at,
                    0,
                    0,
                    json.dumps(conv.settings)
                ))
                
                # Add owner as participant
                cursor.execute("""
                    INSERT INTO participants (
                        conversation_id, user_id, permission, joined_at, nickname
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    conv.conversation_id,
                    owner_id,
                    SharePermission.ADMIN.value,
                    time.time(),
                    ""
                ))
                
                conn.commit()
            
            return conv
        
        def get_conversation(
            self,
            conversation_id: str
        ) -> Optional[SharedConversation]:
            """Get conversation by ID."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                conv = SharedConversation(
                    conversation_id=row[0],
                    title=row[1],
                    owner_id=row[2],
                    visibility=ConversationVisibility(row[3]),
                    created_at=row[4],
                    updated_at=row[5],
                    message_count=row[6],
                    is_archived=bool(row[7]),
                    settings=json.loads(row[8]) if row[8] else {}
                )
                
                # Load participants
                cursor.execute(
                    "SELECT * FROM participants WHERE conversation_id = ?",
                    (conversation_id,)
                )
                
                for p_row in cursor.fetchall():
                    conv.participants[p_row[1]] = Participant(
                        user_id=p_row[1],
                        permission=SharePermission(p_row[2]),
                        joined_at=p_row[3],
                        last_read=p_row[4],
                        nickname=p_row[5] or ""
                    )
                
                return conv
        
        def add_message(self, message: Message) -> Message:
            """Add message to conversation."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO messages (
                        message_id, conversation_id, user_id, role,
                        content, created_at, edited_at, reply_to, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.message_id,
                    message.conversation_id,
                    message.user_id,
                    message.role,
                    message.content,
                    message.created_at,
                    message.edited_at,
                    message.reply_to,
                    json.dumps(message.metadata)
                ))
                
                # Update conversation
                cursor.execute("""
                    UPDATE conversations 
                    SET message_count = message_count + 1, updated_at = ?
                    WHERE conversation_id = ?
                """, (time.time(), message.conversation_id))
                
                conn.commit()
            
            return message
        
        def get_messages(
            self,
            conversation_id: str,
            limit: int = 50,
            before: float = None,
            after: float = None
        ) -> list[Message]:
            """Get messages from conversation."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM messages WHERE conversation_id = ?"
                params = [conversation_id]
                
                if before:
                    query += " AND created_at < ?"
                    params.append(before)
                
                if after:
                    query += " AND created_at > ?"
                    params.append(after)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                messages = []
                for row in cursor.fetchall():
                    messages.append(Message(
                        message_id=row[0],
                        conversation_id=row[1],
                        user_id=row[2],
                        role=row[3],
                        content=row[4],
                        created_at=row[5],
                        edited_at=row[6],
                        reply_to=row[7],
                        metadata=json.loads(row[8]) if row[8] else {}
                    ))
                
                return list(reversed(messages))
        
        def add_participant(
            self,
            conversation_id: str,
            user_id: str,
            permission: SharePermission = SharePermission.PARTICIPATE
        ):
            """Add participant to conversation."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO participants (
                        conversation_id, user_id, permission, joined_at, nickname
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    user_id,
                    permission.value,
                    time.time(),
                    ""
                ))
                
                conn.commit()
        
        def remove_participant(self, conversation_id: str, user_id: str):
            """Remove participant from conversation."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM participants 
                    WHERE conversation_id = ? AND user_id = ?
                """, (conversation_id, user_id))
                conn.commit()
        
        def get_user_conversations(
            self,
            user_id: str,
            include_archived: bool = False
        ) -> list[SharedConversation]:
            """Get all conversations for a user."""
            conversations = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT c.* FROM conversations c
                    INNER JOIN participants p ON c.conversation_id = p.conversation_id
                    WHERE p.user_id = ?
                """
                
                if not include_archived:
                    query += " AND c.is_archived = 0"
                
                query += " ORDER BY c.updated_at DESC"
                
                cursor.execute(query, (user_id,))
                
                for row in cursor.fetchall():
                    conv = SharedConversation(
                        conversation_id=row[0],
                        title=row[1],
                        owner_id=row[2],
                        visibility=ConversationVisibility(row[3]),
                        created_at=row[4],
                        updated_at=row[5],
                        message_count=row[6],
                        is_archived=bool(row[7])
                    )
                    conversations.append(conv)
            
            return conversations
    
    
    class SharingManager:
        """
        Manages shared conversation access and permissions.
        """
        
        def __init__(self, db_path: str = "data/shared_conversations.db"):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.db = ConversationDatabase(db_path)
            self.broadcaster = MessageBroadcaster()
        
        def create_conversation(
            self,
            title: str,
            owner_id: str,
            visibility: ConversationVisibility = ConversationVisibility.PRIVATE
        ) -> SharedConversation:
            """Create shared conversation."""
            return self.db.create_conversation(title, owner_id, visibility)
        
        def get_conversation(
            self,
            conversation_id: str,
            user_id: str
        ) -> Optional[SharedConversation]:
            """Get conversation if user has access."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return None
            
            if not self._has_access(conv, user_id, SharePermission.VIEW):
                return None
            
            return conv
        
        def send_message(
            self,
            conversation_id: str,
            user_id: str,
            content: str,
            role: str = "user",
            reply_to: str = None,
            metadata: dict = None
        ) -> Optional[Message]:
            """Send message to conversation."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return None
            
            if not self._has_access(conv, user_id, SharePermission.PARTICIPATE):
                logger.warning(f"User {user_id} cannot send to {conversation_id}")
                return None
            
            message = Message(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                reply_to=reply_to,
                metadata=metadata or {}
            )
            
            self.db.add_message(message)
            
            # Broadcast to listeners
            self.broadcaster.broadcast(conversation_id, "new_message", {
                "message": asdict(message)
            })
            
            return message
        
        def get_messages(
            self,
            conversation_id: str,
            user_id: str,
            limit: int = 50,
            before: float = None
        ) -> list[Message]:
            """Get messages if user has access."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return []
            
            if not self._has_access(conv, user_id, SharePermission.VIEW):
                return []
            
            # Update last read
            if user_id in conv.participants:
                self._update_last_read(conversation_id, user_id)
            
            return self.db.get_messages(conversation_id, limit, before)
        
        def share_with_user(
            self,
            conversation_id: str,
            owner_id: str,
            target_user_id: str,
            permission: SharePermission = SharePermission.PARTICIPATE
        ) -> bool:
            """Share conversation with another user."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return False
            
            # Check if requester can share
            if not self._has_access(conv, owner_id, SharePermission.ADMIN):
                return False
            
            self.db.add_participant(conversation_id, target_user_id, permission)
            
            self.broadcaster.broadcast(conversation_id, "participant_added", {
                "user_id": target_user_id,
                "permission": permission.value
            })
            
            return True
        
        def remove_participant(
            self,
            conversation_id: str,
            requester_id: str,
            target_user_id: str
        ) -> bool:
            """Remove participant from conversation."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return False
            
            # Can remove self or admin can remove others
            if requester_id != target_user_id:
                if not self._has_access(conv, requester_id, SharePermission.ADMIN):
                    return False
            
            # Cannot remove owner
            if target_user_id == conv.owner_id:
                return False
            
            self.db.remove_participant(conversation_id, target_user_id)
            
            self.broadcaster.broadcast(conversation_id, "participant_removed", {
                "user_id": target_user_id
            })
            
            return True
        
        def update_permission(
            self,
            conversation_id: str,
            admin_id: str,
            target_user_id: str,
            new_permission: SharePermission
        ) -> bool:
            """Update participant permission."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return False
            
            if not self._has_access(conv, admin_id, SharePermission.ADMIN):
                return False
            
            self.db.add_participant(
                conversation_id,
                target_user_id,
                new_permission
            )
            
            return True
        
        def get_user_conversations(
            self,
            user_id: str,
            include_archived: bool = False
        ) -> list[SharedConversation]:
            """Get all conversations accessible by user."""
            return self.db.get_user_conversations(user_id, include_archived)
        
        def subscribe(
            self,
            conversation_id: str,
            user_id: str,
            callback: Callable
        ) -> bool:
            """Subscribe to conversation updates."""
            conv = self.db.get_conversation(conversation_id)
            
            if not conv:
                return False
            
            if not self._has_access(conv, user_id, SharePermission.VIEW):
                return False
            
            self.broadcaster.subscribe(conversation_id, callback)
            return True
        
        def unsubscribe(self, conversation_id: str, callback: Callable):
            """Unsubscribe from updates."""
            self.broadcaster.unsubscribe(conversation_id, callback)
        
        def _has_access(
            self,
            conv: SharedConversation,
            user_id: str,
            required: SharePermission
        ) -> bool:
            """Check if user has required permission."""
            # Owner has full access
            if user_id == conv.owner_id:
                return True
            
            # Public conversations
            if conv.visibility == ConversationVisibility.PUBLIC:
                if required in [SharePermission.VIEW, SharePermission.COMMENT]:
                    return True
            
            # Check explicit permission
            if user_id not in conv.participants:
                return False
            
            participant = conv.participants[user_id]
            
            permission_levels = {
                SharePermission.VIEW: 1,
                SharePermission.COMMENT: 2,
                SharePermission.PARTICIPATE: 3,
                SharePermission.MODERATE: 4,
                SharePermission.ADMIN: 5
            }
            
            return permission_levels.get(participant.permission, 0) >= \
                   permission_levels.get(required, 0)
        
        def _update_last_read(self, conversation_id: str, user_id: str):
            """Update user's last read timestamp."""
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE participants SET last_read = ?
                    WHERE conversation_id = ? AND user_id = ?
                """, (time.time(), conversation_id, user_id))
                conn.commit()

else:
    class ConversationDatabase:
        pass
    
    class SharingManager:
        pass


def create_sharing_manager(
    db_path: str = "data/shared_conversations.db"
) -> 'SharingManager':
    """Create sharing manager instance."""
    if not HAS_SQLITE:
        raise ImportError("SQLite required for shared conversations")
    
    return SharingManager(db_path)
