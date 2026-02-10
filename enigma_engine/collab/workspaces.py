"""
Team Workspaces

Shared work areas with roles, resources, and project management.

FILE: enigma_engine/collab/workspaces.py
TYPE: Multi-User
MAIN CLASSES: Workspace, WorkspaceManager, TeamRole
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamRole(Enum):
    """Team member roles."""
    VIEWER = "viewer"  # Can view only
    MEMBER = "member"  # Can participate
    CONTRIBUTOR = "contributor"  # Can add/edit resources
    MODERATOR = "moderator"  # Can manage members
    ADMIN = "admin"  # Full access
    OWNER = "owner"  # Workspace owner


class ResourceType(Enum):
    """Workspace resource types."""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    FILE = "file"
    MODEL = "model"
    DATASET = "dataset"
    NOTEBOOK = "notebook"
    CUSTOM = "custom"


class WorkspaceVisibility(Enum):
    """Workspace visibility."""
    PRIVATE = "private"  # Invite only
    INTERNAL = "internal"  # Organization members
    PUBLIC = "public"  # Anyone can view


@dataclass
class TeamMember:
    """Team member in workspace."""
    user_id: str
    username: str
    role: TeamRole
    joined_at: float = field(default_factory=time.time)
    invited_by: str = None
    permissions: set[str] = field(default_factory=set)


@dataclass
class WorkspaceResource:
    """Resource in workspace."""
    resource_id: str
    resource_type: ResourceType
    name: str
    description: str = ""
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_pinned: bool = False


@dataclass
class Workspace:
    """Team workspace."""
    workspace_id: str
    name: str
    description: str = ""
    owner_id: str = ""
    visibility: WorkspaceVisibility = WorkspaceVisibility.PRIVATE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    members: dict[str, TeamMember] = field(default_factory=dict)
    resources: dict[str, WorkspaceResource] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    icon: str = ""
    color: str = "#007ACC"
    
    # Limits
    max_members: int = 100
    max_resources: int = 1000
    storage_limit_mb: int = 1000


@dataclass
class WorkspaceInvite:
    """Workspace invitation."""
    invite_id: str
    workspace_id: str
    inviter_id: str
    invitee_email: str
    role: TeamRole
    created_at: float = field(default_factory=time.time)
    expires_at: float = None
    accepted: bool = False


if HAS_SQLITE:
    
    class WorkspaceDatabase:
        """SQLite storage for workspaces."""
        
        def __init__(self, db_path: str):
            self.db_path = db_path
            self._init_db()
        
        def _init_db(self):
            """Initialize database schema."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Workspaces table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workspaces (
                        workspace_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        owner_id TEXT,
                        visibility TEXT,
                        created_at REAL,
                        updated_at REAL,
                        settings TEXT,
                        tags TEXT,
                        icon TEXT,
                        color TEXT,
                        max_members INTEGER,
                        max_resources INTEGER,
                        storage_limit_mb INTEGER
                    )
                """)
                
                # Members table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workspace_members (
                        workspace_id TEXT,
                        user_id TEXT,
                        username TEXT,
                        role TEXT,
                        joined_at REAL,
                        invited_by TEXT,
                        permissions TEXT,
                        PRIMARY KEY (workspace_id, user_id),
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
                    )
                """)
                
                # Resources table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workspace_resources (
                        resource_id TEXT PRIMARY KEY,
                        workspace_id TEXT,
                        resource_type TEXT,
                        name TEXT,
                        description TEXT,
                        created_by TEXT,
                        created_at REAL,
                        updated_at REAL,
                        metadata TEXT,
                        tags TEXT,
                        is_pinned INTEGER,
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
                    )
                """)
                
                # Invites table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workspace_invites (
                        invite_id TEXT PRIMARY KEY,
                        workspace_id TEXT,
                        inviter_id TEXT,
                        invitee_email TEXT,
                        role TEXT,
                        created_at REAL,
                        expires_at REAL,
                        accepted INTEGER,
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
                    )
                """)
                
                # Indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_members_user 
                    ON workspace_members(user_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_resources_workspace 
                    ON workspace_resources(workspace_id)
                """)
                
                conn.commit()
        
        def create_workspace(
            self,
            name: str,
            owner_id: str,
            description: str = "",
            visibility: WorkspaceVisibility = WorkspaceVisibility.PRIVATE
        ) -> Workspace:
            """Create new workspace."""
            workspace = Workspace(
                workspace_id=str(uuid.uuid4()),
                name=name,
                description=description,
                owner_id=owner_id,
                visibility=visibility
            )
            
            # Add owner as member
            workspace.members[owner_id] = TeamMember(
                user_id=owner_id,
                username="",  # To be filled
                role=TeamRole.OWNER
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO workspaces (
                        workspace_id, name, description, owner_id, visibility,
                        created_at, updated_at, settings, tags, icon, color,
                        max_members, max_resources, storage_limit_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workspace.workspace_id,
                    workspace.name,
                    workspace.description,
                    workspace.owner_id,
                    workspace.visibility.value,
                    workspace.created_at,
                    workspace.updated_at,
                    json.dumps(workspace.settings),
                    json.dumps(workspace.tags),
                    workspace.icon,
                    workspace.color,
                    workspace.max_members,
                    workspace.max_resources,
                    workspace.storage_limit_mb
                ))
                
                # Add owner as member
                cursor.execute("""
                    INSERT INTO workspace_members (
                        workspace_id, user_id, username, role, joined_at, permissions
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    workspace.workspace_id,
                    owner_id,
                    "",
                    TeamRole.OWNER.value,
                    time.time(),
                    json.dumps([])
                ))
                
                conn.commit()
            
            return workspace
        
        def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
            """Get workspace by ID."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM workspaces WHERE workspace_id = ?",
                    (workspace_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                workspace = Workspace(
                    workspace_id=row[0],
                    name=row[1],
                    description=row[2] or "",
                    owner_id=row[3],
                    visibility=WorkspaceVisibility(row[4]),
                    created_at=row[5],
                    updated_at=row[6],
                    settings=json.loads(row[7]) if row[7] else {},
                    tags=json.loads(row[8]) if row[8] else [],
                    icon=row[9] or "",
                    color=row[10] or "#007ACC",
                    max_members=row[11],
                    max_resources=row[12],
                    storage_limit_mb=row[13]
                )
                
                # Load members
                cursor.execute(
                    "SELECT * FROM workspace_members WHERE workspace_id = ?",
                    (workspace_id,)
                )
                
                for m_row in cursor.fetchall():
                    workspace.members[m_row[1]] = TeamMember(
                        user_id=m_row[1],
                        username=m_row[2] or "",
                        role=TeamRole(m_row[3]),
                        joined_at=m_row[4],
                        invited_by=m_row[5],
                        permissions=set(json.loads(m_row[6]) if m_row[6] else [])
                    )
                
                # Load resources
                cursor.execute(
                    "SELECT * FROM workspace_resources WHERE workspace_id = ?",
                    (workspace_id,)
                )
                
                for r_row in cursor.fetchall():
                    workspace.resources[r_row[0]] = WorkspaceResource(
                        resource_id=r_row[0],
                        resource_type=ResourceType(r_row[2]),
                        name=r_row[3],
                        description=r_row[4] or "",
                        created_by=r_row[5] or "",
                        created_at=r_row[6],
                        updated_at=r_row[7],
                        metadata=json.loads(r_row[8]) if r_row[8] else {},
                        tags=json.loads(r_row[9]) if r_row[9] else [],
                        is_pinned=bool(r_row[10])
                    )
                
                return workspace
        
        def update_workspace(self, workspace: Workspace):
            """Update workspace."""
            workspace.updated_at = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE workspaces SET
                        name = ?,
                        description = ?,
                        visibility = ?,
                        updated_at = ?,
                        settings = ?,
                        tags = ?,
                        icon = ?,
                        color = ?
                    WHERE workspace_id = ?
                """, (
                    workspace.name,
                    workspace.description,
                    workspace.visibility.value,
                    workspace.updated_at,
                    json.dumps(workspace.settings),
                    json.dumps(workspace.tags),
                    workspace.icon,
                    workspace.color,
                    workspace.workspace_id
                ))
                
                conn.commit()
        
        def delete_workspace(self, workspace_id: str):
            """Delete workspace."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM workspace_resources WHERE workspace_id = ?",
                    (workspace_id,)
                )
                cursor.execute(
                    "DELETE FROM workspace_members WHERE workspace_id = ?",
                    (workspace_id,)
                )
                cursor.execute(
                    "DELETE FROM workspace_invites WHERE workspace_id = ?",
                    (workspace_id,)
                )
                cursor.execute(
                    "DELETE FROM workspaces WHERE workspace_id = ?",
                    (workspace_id,)
                )
                conn.commit()
        
        def add_member(
            self,
            workspace_id: str,
            user_id: str,
            username: str,
            role: TeamRole,
            invited_by: str = None
        ):
            """Add member to workspace."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO workspace_members (
                        workspace_id, user_id, username, role, joined_at, invited_by, permissions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    workspace_id,
                    user_id,
                    username,
                    role.value,
                    time.time(),
                    invited_by,
                    json.dumps([])
                ))
                
                conn.commit()
        
        def remove_member(self, workspace_id: str, user_id: str):
            """Remove member from workspace."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM workspace_members 
                    WHERE workspace_id = ? AND user_id = ?
                """, (workspace_id, user_id))
                conn.commit()
        
        def add_resource(
            self,
            workspace_id: str,
            resource: WorkspaceResource
        ):
            """Add resource to workspace."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO workspace_resources (
                        resource_id, workspace_id, resource_type, name, description,
                        created_by, created_at, updated_at, metadata, tags, is_pinned
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resource.resource_id,
                    workspace_id,
                    resource.resource_type.value,
                    resource.name,
                    resource.description,
                    resource.created_by,
                    resource.created_at,
                    resource.updated_at,
                    json.dumps(resource.metadata),
                    json.dumps(resource.tags),
                    int(resource.is_pinned)
                ))
                
                conn.commit()
        
        def remove_resource(self, resource_id: str):
            """Remove resource from workspace."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM workspace_resources WHERE resource_id = ?",
                    (resource_id,)
                )
                conn.commit()
        
        def get_user_workspaces(self, user_id: str) -> list[Workspace]:
            """Get all workspaces for a user."""
            workspaces = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT w.workspace_id FROM workspaces w
                    INNER JOIN workspace_members m ON w.workspace_id = m.workspace_id
                    WHERE m.user_id = ?
                    ORDER BY w.updated_at DESC
                """, (user_id,))
                
                for row in cursor.fetchall():
                    workspace = self.get_workspace(row[0])
                    if workspace:
                        workspaces.append(workspace)
            
            return workspaces
        
        def create_invite(
            self,
            workspace_id: str,
            inviter_id: str,
            invitee_email: str,
            role: TeamRole
        ) -> WorkspaceInvite:
            """Create workspace invite."""
            invite = WorkspaceInvite(
                invite_id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                inviter_id=inviter_id,
                invitee_email=invitee_email,
                role=role,
                expires_at=time.time() + 86400 * 7  # 7 days
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO workspace_invites (
                        invite_id, workspace_id, inviter_id, invitee_email,
                        role, created_at, expires_at, accepted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    invite.invite_id,
                    invite.workspace_id,
                    invite.inviter_id,
                    invite.invitee_email,
                    invite.role.value,
                    invite.created_at,
                    invite.expires_at,
                    0
                ))
                
                conn.commit()
            
            return invite
        
        def get_invite(self, invite_id: str) -> Optional[WorkspaceInvite]:
            """Get invite by ID."""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM workspace_invites WHERE invite_id = ?",
                    (invite_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return WorkspaceInvite(
                        invite_id=row[0],
                        workspace_id=row[1],
                        inviter_id=row[2],
                        invitee_email=row[3],
                        role=TeamRole(row[4]),
                        created_at=row[5],
                        expires_at=row[6],
                        accepted=bool(row[7])
                    )
            return None
    
    
    class WorkspaceManager:
        """
        Manages team workspaces.
        """
        
        def __init__(self, db_path: str = "data/workspaces.db"):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = WorkspaceDatabase(db_path)
        
        def create_workspace(
            self,
            name: str,
            owner_id: str,
            description: str = "",
            visibility: WorkspaceVisibility = WorkspaceVisibility.PRIVATE
        ) -> Workspace:
            """Create new workspace."""
            return self.db.create_workspace(
                name,
                owner_id,
                description,
                visibility
            )
        
        def get_workspace(
            self,
            workspace_id: str,
            user_id: str
        ) -> Optional[Workspace]:
            """Get workspace if user has access."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return None
            
            if not self._has_access(workspace, user_id, TeamRole.VIEWER):
                return None
            
            return workspace
        
        def update_workspace(
            self,
            workspace_id: str,
            user_id: str,
            updates: dict[str, Any]
        ) -> bool:
            """Update workspace settings."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return False
            
            if not self._has_access(workspace, user_id, TeamRole.ADMIN):
                return False
            
            if "name" in updates:
                workspace.name = updates["name"]
            if "description" in updates:
                workspace.description = updates["description"]
            if "visibility" in updates:
                workspace.visibility = WorkspaceVisibility(updates["visibility"])
            if "settings" in updates:
                workspace.settings.update(updates["settings"])
            if "icon" in updates:
                workspace.icon = updates["icon"]
            if "color" in updates:
                workspace.color = updates["color"]
            
            self.db.update_workspace(workspace)
            return True
        
        def delete_workspace(
            self,
            workspace_id: str,
            user_id: str
        ) -> bool:
            """Delete workspace (owner only)."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return False
            
            if workspace.owner_id != user_id:
                return False
            
            self.db.delete_workspace(workspace_id)
            return True
        
        def invite_member(
            self,
            workspace_id: str,
            inviter_id: str,
            invitee_email: str,
            role: TeamRole = TeamRole.MEMBER
        ) -> Optional[WorkspaceInvite]:
            """Invite member to workspace."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return None
            
            if not self._has_access(workspace, inviter_id, TeamRole.MODERATOR):
                return None
            
            if len(workspace.members) >= workspace.max_members:
                logger.warning(f"Workspace {workspace_id} member limit reached")
                return None
            
            return self.db.create_invite(
                workspace_id,
                inviter_id,
                invitee_email,
                role
            )
        
        def accept_invite(
            self,
            invite_id: str,
            user_id: str,
            username: str
        ) -> bool:
            """Accept workspace invite."""
            invite = self.db.get_invite(invite_id)
            
            if not invite:
                return False
            
            if invite.accepted:
                return False
            
            if invite.expires_at and time.time() > invite.expires_at:
                return False
            
            self.db.add_member(
                invite.workspace_id,
                user_id,
                username,
                invite.role,
                invite.inviter_id
            )
            
            return True
        
        def remove_member(
            self,
            workspace_id: str,
            requester_id: str,
            target_user_id: str
        ) -> bool:
            """Remove member from workspace."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return False
            
            # Cannot remove owner
            if target_user_id == workspace.owner_id:
                return False
            
            # Can remove self or moderator+ can remove others
            if requester_id != target_user_id:
                if not self._has_access(workspace, requester_id, TeamRole.MODERATOR):
                    return False
            
            self.db.remove_member(workspace_id, target_user_id)
            return True
        
        def update_member_role(
            self,
            workspace_id: str,
            admin_id: str,
            target_user_id: str,
            new_role: TeamRole
        ) -> bool:
            """Update member role."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return False
            
            if not self._has_access(workspace, admin_id, TeamRole.ADMIN):
                return False
            
            # Cannot change owner role
            if target_user_id == workspace.owner_id:
                return False
            
            if target_user_id not in workspace.members:
                return False
            
            member = workspace.members[target_user_id]
            self.db.add_member(
                workspace_id,
                target_user_id,
                member.username,
                new_role,
                member.invited_by
            )
            
            return True
        
        def add_resource(
            self,
            workspace_id: str,
            user_id: str,
            resource_type: ResourceType,
            name: str,
            description: str = "",
            metadata: dict = None
        ) -> Optional[WorkspaceResource]:
            """Add resource to workspace."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return None
            
            if not self._has_access(workspace, user_id, TeamRole.CONTRIBUTOR):
                return None
            
            if len(workspace.resources) >= workspace.max_resources:
                logger.warning(f"Workspace {workspace_id} resource limit reached")
                return None
            
            resource = WorkspaceResource(
                resource_id=str(uuid.uuid4()),
                resource_type=resource_type,
                name=name,
                description=description,
                created_by=user_id,
                metadata=metadata or {}
            )
            
            self.db.add_resource(workspace_id, resource)
            return resource
        
        def remove_resource(
            self,
            workspace_id: str,
            user_id: str,
            resource_id: str
        ) -> bool:
            """Remove resource from workspace."""
            workspace = self.db.get_workspace(workspace_id)
            
            if not workspace:
                return False
            
            if resource_id not in workspace.resources:
                return False
            
            resource = workspace.resources[resource_id]
            
            # Creator or moderator+ can remove
            if resource.created_by != user_id:
                if not self._has_access(workspace, user_id, TeamRole.MODERATOR):
                    return False
            
            self.db.remove_resource(resource_id)
            return True
        
        def get_user_workspaces(self, user_id: str) -> list[Workspace]:
            """Get all workspaces for user."""
            return self.db.get_user_workspaces(user_id)
        
        def search_resources(
            self,
            workspace_id: str,
            user_id: str,
            query: str = "",
            resource_type: ResourceType = None,
            tags: list[str] = None
        ) -> list[WorkspaceResource]:
            """Search workspace resources."""
            workspace = self.get_workspace(workspace_id, user_id)
            
            if not workspace:
                return []
            
            results = []
            query_lower = query.lower()
            
            for resource in workspace.resources.values():
                # Filter by type
                if resource_type and resource.resource_type != resource_type:
                    continue
                
                # Filter by tags
                if tags and not any(t in resource.tags for t in tags):
                    continue
                
                # Filter by query
                if query:
                    if query_lower not in resource.name.lower() and \
                       query_lower not in resource.description.lower():
                        continue
                
                results.append(resource)
            
            return results
        
        def _has_access(
            self,
            workspace: Workspace,
            user_id: str,
            required_role: TeamRole
        ) -> bool:
            """Check if user has required role."""
            # Owner has full access
            if user_id == workspace.owner_id:
                return True
            
            # Public workspaces allow viewing
            if workspace.visibility == WorkspaceVisibility.PUBLIC:
                if required_role == TeamRole.VIEWER:
                    return True
            
            if user_id not in workspace.members:
                return False
            
            member = workspace.members[user_id]
            
            role_levels = {
                TeamRole.VIEWER: 1,
                TeamRole.MEMBER: 2,
                TeamRole.CONTRIBUTOR: 3,
                TeamRole.MODERATOR: 4,
                TeamRole.ADMIN: 5,
                TeamRole.OWNER: 6
            }
            
            return role_levels.get(member.role, 0) >= role_levels.get(required_role, 0)

else:
    class WorkspaceDatabase:
        pass
    
    class WorkspaceManager:
        pass


def create_workspace_manager(
    db_path: str = "data/workspaces.db"
) -> 'WorkspaceManager':
    """Create workspace manager."""
    if not HAS_SQLITE:
        raise ImportError("SQLite required for workspaces")
    
    return WorkspaceManager(db_path)
