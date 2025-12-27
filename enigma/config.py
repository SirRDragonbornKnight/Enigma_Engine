"""
Enigma Engine Configuration

Central configuration for all Enigma components.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "memory" / "memory.db"
VOCAB_DIR = ROOT / "enigma" / "vocab_model"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(Path(DB_PATH).parent, exist_ok=True)

CONFIG = {
    # Paths
    "root": str(ROOT),
    "data_dir": str(DATA_DIR),
    "models_dir": str(MODELS_DIR),
    "db_path": str(DB_PATH),
    "vocab_dir": str(VOCAB_DIR),
    
    # Model architecture defaults (used when not loading from registry)
    "embed_dim": 256,         # Hidden dimension
    "depth": 6,               # Number of transformer layers
    "heads": 8,               # Number of attention heads
    "max_len": 2048,          # Maximum sequence length
    "ff_mult": 4.0,           # FFN hidden dimension multiplier
    "dropout": 0.0,           # Dropout rate (0 for inference)
    
    # Training defaults
    "learning_rate": 1e-4,
    "batch_size": 4,
    "gradient_accumulation": 1,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Inference defaults
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_gen_tokens": 100,
    
    # Tokenizer
    "vocab_size": 32000,      # Default vocabulary size
    
    # Server
    "api_host": "127.0.0.1",
    "api_port": 5000,
}
