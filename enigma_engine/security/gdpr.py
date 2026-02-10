"""
GDPR Compliance Tools

Data export, deletion, and privacy compliance utilities.
Implements right to access, right to erasure, and data portability.

FILE: enigma_engine/security/gdpr.py
TYPE: Privacy Compliance
MAIN CLASSES: GDPRManager, DataExport, DeletionManager
"""

import hashlib
import json
import logging
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Categories of personal data."""
    PROFILE = "profile"           # Username, email, preferences
    CONVERSATIONS = "conversations"  # Chat history
    MEMORIES = "memories"          # Stored memories
    VOICE = "voice"               # Voice recordings
    IMAGES = "images"             # Uploaded/generated images
    PREFERENCES = "preferences"    # Settings, customizations
    ANALYTICS = "analytics"        # Usage data
    LOGS = "logs"                 # Activity logs


class ConsentPurpose(Enum):
    """Purposes requiring consent."""
    ESSENTIAL = "essential"        # Core functionality
    ANALYTICS = "analytics"        # Usage analytics
    PERSONALIZATION = "personalization"  # AI personalization
    MARKETING = "marketing"        # Marketing communications
    THIRD_PARTY = "third_party"    # Third-party sharing
    TRAINING = "training"          # Model training


@dataclass
class ConsentRecord:
    """Record of user consent."""
    user_id: str
    purpose: ConsentPurpose
    granted: bool
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Where consent was given
    details: str = ""


@dataclass
class DataRequest:
    """GDPR data request."""
    id: str
    user_id: str
    request_type: str  # "export", "delete", "access"
    requested_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    status: str = "pending"  # pending, processing, completed, failed
    categories: list[DataCategory] = field(default_factory=list)
    result_path: str = ""
    error: str = ""


class DataExporter:
    """Exports user data in portable formats."""
    
    def __init__(self, data_dir: Path, export_dir: Path):
        """
        Initialize exporter.
        
        Args:
            data_dir: Base directory containing user data
            export_dir: Directory for exports
        """
        self._data_dir = data_dir
        self._export_dir = export_dir
        self._export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_all(self, 
                   user_id: str,
                   categories: list[DataCategory] = None) -> Path:
        """
        Export all user data.
        
        Args:
            user_id: User ID
            categories: Categories to export (all if None)
            
        Returns:
            Path to export file
        """
        if categories is None:
            categories = list(DataCategory)
        
        export_data = {
            "export_info": {
                "user_id": user_id,
                "exported_at": time.time(),
                "categories": [c.value for c in categories]
            },
            "data": {}
        }
        
        for category in categories:
            try:
                data = self._export_category(user_id, category)
                if data:
                    export_data["data"][category.value] = data
            except Exception as e:
                logger.error(f"Failed to export {category.value}: {e}")
                export_data["data"][category.value] = {"error": str(e)}
        
        # Create export file
        export_path = self._export_dir / f"export_{user_id}_{int(time.time())}.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Main data JSON
            zf.writestr("data.json", json.dumps(export_data, indent=2))
            
            # Include binary files separately
            self._add_binary_files(zf, user_id, categories)
        
        return export_path
    
    def _export_category(self, user_id: str, category: DataCategory) -> dict:
        """Export a specific category."""
        exporters = {
            DataCategory.PROFILE: self._export_profile,
            DataCategory.CONVERSATIONS: self._export_conversations,
            DataCategory.MEMORIES: self._export_memories,
            DataCategory.PREFERENCES: self._export_preferences,
            DataCategory.ANALYTICS: self._export_analytics,
            DataCategory.LOGS: self._export_logs,
        }
        
        exporter = exporters.get(category)
        if exporter:
            return exporter(user_id)
        return {}
    
    def _export_profile(self, user_id: str) -> dict:
        """Export profile data."""
        profile_path = self._data_dir / "users" / f"{user_id}.json"
        if profile_path.exists():
            with open(profile_path) as f:
                data = json.load(f)
            # Remove sensitive fields
            data.pop("password_hash", None)
            data.pop("salt", None)
            return data
        return {}
    
    def _export_conversations(self, user_id: str) -> list[dict]:
        """Export conversation history."""
        conv_dir = self._data_dir / "conversations" / user_id
        conversations = []
        
        if conv_dir.exists():
            for conv_file in conv_dir.glob("*.json"):
                with open(conv_file) as f:
                    conversations.append(json.load(f))
        
        return conversations
    
    def _export_memories(self, user_id: str) -> list[dict]:
        """Export stored memories."""
        mem_dir = self._data_dir / "memory" / user_id
        memories = []
        
        if mem_dir.exists():
            for mem_file in mem_dir.glob("*.json"):
                with open(mem_file) as f:
                    memories.append(json.load(f))
        
        return memories
    
    def _export_preferences(self, user_id: str) -> dict:
        """Export user preferences."""
        pref_path = self._data_dir / "preferences" / f"{user_id}.json"
        if pref_path.exists():
            with open(pref_path) as f:
                return json.load(f)
        return {}
    
    def _export_analytics(self, user_id: str) -> dict:
        """Export analytics data."""
        analytics_path = self._data_dir / "analytics" / f"{user_id}.json"
        if analytics_path.exists():
            with open(analytics_path) as f:
                return json.load(f)
        return {}
    
    def _export_logs(self, user_id: str) -> list[dict]:
        """Export activity logs."""
        logs_dir = self._data_dir / "logs"
        logs = []
        
        if logs_dir.exists():
            for log_file in logs_dir.glob(f"*{user_id}*.log"):
                with open(log_file) as f:
                    for line in f:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            logs.append({"raw": line.strip()})
        
        return logs
    
    def _add_binary_files(self, 
                          zf: zipfile.ZipFile,
                          user_id: str,
                          categories: list[DataCategory]):
        """Add binary files to export."""
        if DataCategory.VOICE in categories:
            voice_dir = self._data_dir / "voice" / user_id
            if voice_dir.exists():
                for audio_file in voice_dir.glob("*"):
                    zf.write(audio_file, f"voice/{audio_file.name}")
        
        if DataCategory.IMAGES in categories:
            images_dir = self._data_dir / "images" / user_id
            if images_dir.exists():
                for img_file in images_dir.glob("*"):
                    zf.write(img_file, f"images/{img_file.name}")


class DeletionManager:
    """Manages data deletion (right to erasure)."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize deletion manager.
        
        Args:
            data_dir: Base directory containing user data
        """
        self._data_dir = data_dir
        self._deletion_log: list[dict] = []
    
    def delete_all(self, 
                   user_id: str,
                   categories: list[DataCategory] = None) -> dict[str, Any]:
        """
        Delete all user data.
        
        Args:
            user_id: User ID
            categories: Categories to delete (all if None)
            
        Returns:
            Deletion report
        """
        if categories is None:
            categories = list(DataCategory)
        
        report = {
            "user_id": user_id,
            "deleted_at": time.time(),
            "categories": {}
        }
        
        for category in categories:
            try:
                result = self._delete_category(user_id, category)
                report["categories"][category.value] = result
            except Exception as e:
                logger.error(f"Failed to delete {category.value}: {e}")
                report["categories"][category.value] = {"error": str(e)}
        
        # Log deletion
        self._deletion_log.append(report)
        
        return report
    
    def _delete_category(self, user_id: str, category: DataCategory) -> dict:
        """Delete a specific category."""
        paths_deleted = []
        
        category_paths = {
            DataCategory.PROFILE: [self._data_dir / "users" / f"{user_id}.json"],
            DataCategory.CONVERSATIONS: [self._data_dir / "conversations" / user_id],
            DataCategory.MEMORIES: [self._data_dir / "memory" / user_id],
            DataCategory.VOICE: [self._data_dir / "voice" / user_id],
            DataCategory.IMAGES: [self._data_dir / "images" / user_id],
            DataCategory.PREFERENCES: [self._data_dir / "preferences" / f"{user_id}.json"],
            DataCategory.ANALYTICS: [self._data_dir / "analytics" / f"{user_id}.json"],
        }
        
        for path in category_paths.get(category, []):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                paths_deleted.append(str(path))
        
        return {
            "deleted": True,
            "paths": paths_deleted
        }
    
    def anonymize(self, user_id: str) -> dict:
        """
        Anonymize user data instead of deleting.
        Replaces identifiable info with hashed values.
        
        Args:
            user_id: User ID
            
        Returns:
            Anonymization report
        """
        anon_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        report = {
            "original_id": user_id,
            "anonymized_id": anon_id,
            "timestamp": time.time(),
            "actions": []
        }
        
        # Anonymize profile
        profile_path = self._data_dir / "users" / f"{user_id}.json"
        if profile_path.exists():
            self._anonymize_profile(profile_path, anon_id)
            report["actions"].append("profile_anonymized")
        
        # Anonymize conversations
        conv_dir = self._data_dir / "conversations" / user_id
        if conv_dir.exists():
            self._anonymize_conversations(conv_dir, anon_id)
            conv_dir.rename(self._data_dir / "conversations" / anon_id)
            report["actions"].append("conversations_anonymized")
        
        return report
    
    def _anonymize_profile(self, path: Path, anon_id: str):
        """Anonymize profile data."""
        with open(path) as f:
            data = json.load(f)
        
        data["id"] = anon_id
        data["username"] = f"user_{anon_id[:8]}"
        data["email"] = f"{anon_id[:8]}@anonymized.local"
        data.pop("password_hash", None)
        data.pop("salt", None)
        
        # Rename file
        new_path = path.parent / f"{anon_id}.json"
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=2)
        path.unlink()
    
    def _anonymize_conversations(self, conv_dir: Path, anon_id: str):
        """Anonymize conversation data."""
        for conv_file in conv_dir.glob("*.json"):
            with open(conv_file) as f:
                data = json.load(f)
            
            data["user_id"] = anon_id
            
            # Remove PII from messages
            for msg in data.get("messages", []):
                # Keep content but anonymize metadata
                msg.pop("ip_address", None)
                msg.pop("user_agent", None)
            
            with open(conv_file, 'w') as f:
                json.dump(data, f, indent=2)


class ConsentManager:
    """Manages user consent records."""
    
    def __init__(self, storage_path: Path):
        """
        Initialize consent manager.
        
        Args:
            storage_path: Path for storing consent records
        """
        self._storage_path = storage_path
        self._consents: dict[str, dict[ConsentPurpose, ConsentRecord]] = {}
        self._load()
    
    def record_consent(self,
                       user_id: str,
                       purpose: ConsentPurpose,
                       granted: bool,
                       source: str = "") -> ConsentRecord:
        """
        Record user consent.
        
        Args:
            user_id: User ID
            purpose: Consent purpose
            granted: Whether consent was granted
            source: Source of consent (e.g., "signup", "settings")
            
        Returns:
            Consent record
        """
        record = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            granted=granted,
            source=source
        )
        
        if user_id not in self._consents:
            self._consents[user_id] = {}
        
        self._consents[user_id][purpose] = record
        self._save()
        
        return record
    
    def has_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """Check if user has granted consent for a purpose."""
        user_consents = self._consents.get(user_id, {})
        record = user_consents.get(purpose)
        
        # Essential is always allowed
        if purpose == ConsentPurpose.ESSENTIAL:
            return True
        
        return record.granted if record else False
    
    def get_all_consents(self, user_id: str) -> dict[str, bool]:
        """Get all consent statuses for a user."""
        user_consents = self._consents.get(user_id, {})
        return {
            p.value: (user_consents.get(p).granted if user_consents.get(p) else False)
            for p in ConsentPurpose
        }
    
    def withdraw_all_consents(self, user_id: str):
        """Withdraw all consents for a user."""
        for purpose in ConsentPurpose:
            if purpose != ConsentPurpose.ESSENTIAL:
                self.record_consent(user_id, purpose, False, "withdrawal")
    
    def _save(self):
        """Save consent records."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for user_id, consents in self._consents.items():
            data[user_id] = {
                p.value: {
                    "granted": r.granted,
                    "timestamp": r.timestamp,
                    "source": r.source
                }
                for p, r in consents.items()
            }
        
        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load consent records."""
        if not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
            
            for user_id, consents in data.items():
                self._consents[user_id] = {}
                for purpose_str, record_data in consents.items():
                    purpose = ConsentPurpose(purpose_str)
                    self._consents[user_id][purpose] = ConsentRecord(
                        user_id=user_id,
                        purpose=purpose,
                        granted=record_data["granted"],
                        timestamp=record_data.get("timestamp", 0),
                        source=record_data.get("source", "")
                    )
        except Exception as e:
            logger.error(f"Failed to load consents: {e}")


class GDPRManager:
    """Main GDPR compliance manager."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize GDPR manager.
        
        Args:
            data_dir: Base data directory
        """
        self._data_dir = Path(data_dir)
        self._export_dir = self._data_dir / "exports"
        self._requests: dict[str, DataRequest] = {}
        
        self.exporter = DataExporter(data_dir, self._export_dir)
        self.deletion = DeletionManager(data_dir)
        self.consent = ConsentManager(data_dir / "consents.json")
    
    def request_data_export(self, user_id: str) -> DataRequest:
        """
        Request data export (right to access/portability).
        
        Args:
            user_id: User ID
            
        Returns:
            Data request object
        """
        request = DataRequest(
            id=f"exp_{int(time.time())}_{user_id[:8]}",
            user_id=user_id,
            request_type="export",
            categories=list(DataCategory)
        )
        
        self._requests[request.id] = request
        
        # Process immediately (could be async in production)
        try:
            export_path = self.exporter.export_all(user_id)
            request.result_path = str(export_path)
            request.status = "completed"
            request.completed_at = time.time()
        except Exception as e:
            request.status = "failed"
            request.error = str(e)
        
        return request
    
    def request_data_deletion(self, 
                              user_id: str,
                              anonymize: bool = False) -> DataRequest:
        """
        Request data deletion (right to erasure).
        
        Args:
            user_id: User ID
            anonymize: Anonymize instead of delete
            
        Returns:
            Data request object
        """
        request = DataRequest(
            id=f"del_{int(time.time())}_{user_id[:8]}",
            user_id=user_id,
            request_type="delete" if not anonymize else "anonymize",
            categories=list(DataCategory)
        )
        
        self._requests[request.id] = request
        
        try:
            if anonymize:
                self.deletion.anonymize(user_id)
            else:
                self.deletion.delete_all(user_id)
            
            # Also withdraw all consents
            self.consent.withdraw_all_consents(user_id)
            
            request.status = "completed"
            request.completed_at = time.time()
        except Exception as e:
            request.status = "failed"
            request.error = str(e)
        
        return request
    
    def get_request_status(self, request_id: str) -> Optional[DataRequest]:
        """Get status of a data request."""
        return self._requests.get(request_id)
    
    def get_user_requests(self, user_id: str) -> list[DataRequest]:
        """Get all requests for a user."""
        return [r for r in self._requests.values() if r.user_id == user_id]


# Singleton
_gdpr_manager: Optional[GDPRManager] = None


def get_gdpr_manager(data_dir: Path = None) -> GDPRManager:
    """Get the GDPR manager singleton."""
    global _gdpr_manager
    if _gdpr_manager is None:
        _gdpr_manager = GDPRManager(data_dir or Path("data"))
    return _gdpr_manager


__all__ = [
    'GDPRManager',
    'DataExporter',
    'DeletionManager',
    'ConsentManager',
    'ConsentRecord',
    'ConsentPurpose',
    'DataCategory',
    'DataRequest',
    'get_gdpr_manager'
]
