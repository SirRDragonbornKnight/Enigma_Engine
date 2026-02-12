"""
Memory Synchronization for Multi-Device Forge

Syncs conversations and memories between devices so:
  1. Both devices have the same context
  2. Conversations on one device are available on others
  3. Models can share learned context

SYNC STRATEGIES:
  - FULL: Copy all memories to peer (for initial setup)
  - DELTA: Only send new items since last sync
  - MERGE: Combine memories from both sides
"""

import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from ..config import CONFIG

logger = logging.getLogger(__name__)


class MemorySync:
    """
    Synchronize memories and conversations between Forge nodes.
    """
    
    def __init__(self, local_memory_dir: str = None):
        """
        Args:
            local_memory_dir: Path to local memory storage
        """
        self.memory_dir = Path(local_memory_dir or CONFIG.get("conversations_dir", "data/conversations"))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Track sync state
        self.sync_state_file = self.memory_dir / "_sync_state.json"
        self.sync_state = self._load_sync_state()
    
    def _load_sync_state(self) -> dict:
        """Load sync state from file."""
        if self.sync_state_file.exists():
            with open(self.sync_state_file) as f:
                return json.load(f)
        return {"peers": {}}
    
    def _save_sync_state(self):
        """Save sync state to file."""
        with open(self.sync_state_file, "w") as f:
            json.dump(self.sync_state, f, indent=2)
    
    def get_all_memories(self) -> list[dict]:
        """Get all local memories."""
        memories = []
        for file in self.memory_dir.glob("*.json"):
            if file.name.startswith("_"):  # Skip metadata files
                continue
            try:
                with open(file) as f:
                    data = json.load(f)
                    data["_file"] = file.name
                    memories.append(data)
            except (json.JSONDecodeError, OSError):
                pass  # Intentionally silent
        return memories
    
    def get_memories_since(self, timestamp: str) -> list[dict]:
        """Get memories created/modified after timestamp."""
        cutoff = datetime.fromisoformat(timestamp)
        memories = []
        
        for file in self.memory_dir.glob("*.json"):
            if file.name.startswith("_"):
                continue
            
            # Check file modification time
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime > cutoff:
                try:
                    with open(file) as f:
                        data = json.load(f)
                        data["_file"] = file.name
                        memories.append(data)
                except (json.JSONDecodeError, OSError):
                    pass  # Intentionally silent
        
        return memories
    
    def save_memory(self, memory: dict, filename: str = None):
        """Save a memory to local storage."""
        if filename is None:
            filename = f"memory_{int(time.time())}.json"
        
        filepath = self.memory_dir / filename
        memory["_synced"] = datetime.now().isoformat()
        
        with open(filepath, "w") as f:
            json.dump(memory, f, indent=2)
    
    def export_for_sync(self, peer_name: str = None) -> dict:
        """
        Export memories for syncing to another device.
        
        Args:
            peer_name: If provided, only export new items since last sync
            
        Returns:
            Dict with memories to sync
        """
        if peer_name and peer_name in self.sync_state.get("peers", {}):
            last_sync = self.sync_state["peers"][peer_name].get("last_sync")
            if last_sync:
                memories = self.get_memories_since(last_sync)
            else:
                memories = self.get_all_memories()
        else:
            memories = self.get_all_memories()
        
        return {
            "source": CONFIG.get("node_name", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "count": len(memories),
            "memories": memories,
        }
    
    def import_from_sync(self, sync_data: dict, peer_name: str = None):
        """
        Import memories from another device.
        
        Args:
            sync_data: Data from export_for_sync()
            peer_name: Name of the peer for tracking
        """
        memories = sync_data.get("memories", [])
        source = sync_data.get("source", "unknown")
        
        imported = 0
        for memory in memories:
            filename = memory.pop("_file", None)
            if filename:
                # Prefix with source to avoid conflicts
                filename = f"{source}_{filename}"
            self.save_memory(memory, filename)
            imported += 1
        
        # Update sync state
        if peer_name:
            if "peers" not in self.sync_state:
                self.sync_state["peers"] = {}
            self.sync_state["peers"][peer_name] = {
                "last_sync": datetime.now().isoformat(),
                "items_received": imported,
            }
            self._save_sync_state()
        
        logger.info("Imported %d memories from %s", imported, source)
        return imported
    
    def sync_with_peer(self, peer_url: str, peer_name: str = None) -> dict:
        """
        Full two-way sync with a peer.
        
        Args:
            peer_url: URL of the peer node
            peer_name: Name of the peer
            
        Returns:
            Sync statistics
        """
        if not peer_url.startswith("http"):
            peer_url = f"http://{peer_url}"
        
        stats = {
            "sent": 0,
            "received": 0,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Export our memories
        export_data = self.export_for_sync(peer_name)
        
        # Send to peer
        try:
            req = urllib.request.Request(
                f"{peer_url}/sync/receive",
                data=json.dumps(export_data).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())
                stats["sent"] = export_data["count"]
        except Exception as e:
            logger.error("Failed to send memories: %s", e)
        
        # Request peer's memories
        try:
            last_sync = None
            if peer_name and peer_name in self.sync_state.get("peers", {}):
                last_sync = self.sync_state["peers"][peer_name].get("last_sync")
            
            params = {}
            if last_sync:
                params["since"] = last_sync
            
            url = f"{peer_url}/sync/export"
            if params:
                url += "?" + urllib.parse.urlencode(params)
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as response:
                peer_data = json.loads(response.read().decode())
                received = self.import_from_sync(peer_data, peer_name)
                stats["received"] = received
        except Exception as e:
            logger.error("Failed to receive memories: %s", e)
        
        return stats


def add_sync_routes(app, memory_sync: MemorySync):
    """
    Add memory sync routes to a Flask app.
    
    Call this in your server setup to enable sync endpoints.
    """
    from flask import jsonify, request
    
    @app.route("/sync/receive", methods=["POST"])
    def sync_receive():
        data = request.json or {}
        source = data.get("source", "unknown")
        imported = memory_sync.import_from_sync(data, source)
        return jsonify({"received": imported})
    
    @app.route("/sync/export")
    def sync_export():
        since = request.args.get("since")
        if since:
            memories = memory_sync.get_memories_since(since)
        else:
            memories = memory_sync.get_all_memories()
        
        return jsonify({
            "source": CONFIG.get("node_name", "local"),
            "timestamp": datetime.now().isoformat(),
            "count": len(memories),
            "memories": memories,
        })
    
    @app.route("/sync/status")
    def sync_status():
        return jsonify({
            "memory_count": len(memory_sync.get_all_memories()),
            "sync_state": memory_sync.sync_state,
        })
    
    logger.info("Added sync routes: /sync/receive, /sync/export, /sync/status")


# File-based sync for disconnected devices

class OfflineSync:
    """
    Sync memories using files (USB drive, shared folder, etc.)
    for devices that aren't on the same network.
    """
    
    @staticmethod
    def export_to_file(output_path: str, memory_dir: str = None) -> str:
        """
        Export all memories to a file for transfer.
        
        Args:
            output_path: Where to save the export file
            memory_dir: Path to memory directory
            
        Returns:
            Path to created file
        """
        sync = MemorySync(memory_dir)
        export_data = sync.export_for_sync()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info("Exported %d memories to %s", export_data['count'], output_path)
        return str(output_path)
    
    @staticmethod
    def import_from_file(input_path: str, memory_dir: str = None) -> int:
        """
        Import memories from a file.
        
        Args:
            input_path: Path to the export file
            memory_dir: Path to memory directory
            
        Returns:
            Number of imported items
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise ValueError(f"File not found: {input_path}")
        
        with open(input_path) as f:
            sync_data = json.load(f)
        
        sync = MemorySync(memory_dir)
        imported = sync.import_from_sync(sync_data, sync_data.get("source"))
        
        return imported
