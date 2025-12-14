from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "memory" / "memory.db"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(Path(DB_PATH).parent, exist_ok=True)

CONFIG = {
    "root": str(ROOT),
    "data_dir": str(DATA_DIR),
    "models_dir": str(MODELS_DIR),
    "db_path": str(DB_PATH),
    "max_len": 512,
    "embed_dim": 128,
}
