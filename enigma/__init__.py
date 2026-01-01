# Re-export configuration from central location
from .config import CONFIG, get_config, update_config

# For backwards compatibility, export path constants
from pathlib import Path
ROOT = Path(CONFIG["root"])
DATA_DIR = Path(CONFIG["data_dir"])
MODELS_DIR = Path(CONFIG["models_dir"])
DB_PATH = Path(CONFIG["db_path"])
