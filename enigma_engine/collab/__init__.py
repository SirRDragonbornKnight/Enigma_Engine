"""
Collaboration System

Real-time collaboration features for multi-user sessions.

Provides:
- CollaborationSession: Live editing session
- PresenceManager: User presence tracking
- SyncEngine: State synchronization
- SharedConversation: Shared conversation management
- Workspace: Collaborative workspace
"""

from .realtime import (
    CollabEvent,
    CollabEventType,
    CollaborationSession,
    Cursor,
    Operation,
    OperationalTransform,
    PresenceManager,
    PresenceStatus,
    SyncEngine,
    UserPresence,
)
from .shared_conversations import (
    ConversationVisibility,
    Message,
    MessageBroadcaster,
    Participant,
    SharedConversation,
    SharePermission,
)
from .workspaces import (
    ResourceType,
    TeamMember,
    TeamRole,
    Workspace,
    WorkspaceInvite,
    WorkspaceResource,
    WorkspaceVisibility,
)

__all__ = [
    # Real-time
    "CollaborationSession",
    "PresenceManager",
    "SyncEngine",
    "PresenceStatus",
    "CollabEventType",
    "CollabEvent",
    "Cursor",
    "UserPresence",
    "Operation",
    "OperationalTransform",
    # Shared conversations
    "SharedConversation",
    "SharePermission",
    "ConversationVisibility",
    "Message",
    "Participant",
    "MessageBroadcaster",
    # Workspaces
    "Workspace",
    "WorkspaceInvite",
    "WorkspaceResource",
    "WorkspaceVisibility",
    "TeamMember",
    "TeamRole",
    "ResourceType",
]
