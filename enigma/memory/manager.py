"""
Conversation manager and long-term memory bridge.
Stores conversations to disk (json), pushes embeddings into SimpleVectorDB (if provided),
and uses memory_db for short message storage.
"""
import json
from pathlib import Path
from .vector_utils import SimpleVectorDB  # relative import - but vector_utils in enigma/memory
from ..config import CONFIG
from ..memory.memory_db import add_memory, recent
import time

CONV_DIR = Path(CONFIG["data_dir"]) / "conversations"
CONV_DIR.mkdir(parents=True, exist_ok=True)

class ConversationManager:
    def __init__(self, vector_db: SimpleVectorDB = None):
        self.conv_dir = CONV_DIR
        self.vector_db = vector_db or SimpleVectorDB(dim=CONFIG.get("embed_dim", 128))

    def save_conversation(self, name: str, messages: list):
        """messages: list of {"role": "user|assistant|system", "text": "...", "ts": 1234567.0}"""
        fname = self.conv_dir / f"{name}.json"
        data = {"name": name, "saved_at": time.time(), "messages": messages}
        fname.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Optionally push to memory DB
        for m in messages:
            add_memory(m.get("text",""), source=m.get("role","user"), meta={"conv": name})
        return str(fname)

    def load_conversation(self, name: str):
        fname = self.conv_dir / f"{name}.json"
        if not fname.exists():
            raise FileNotFoundError(fname)
        return json.loads(fname.read_text(encoding="utf-8"))

    def list_conversations(self):
        return [p.stem for p in sorted(self.conv_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)]

    def add_to_vector_db(self, id_, vector):
        self.vector_db.add(vector, id_)

    def search_vectors(self, query_vec, topk=5):
        return self.vector_db.search(query_vec, topk=topk)
