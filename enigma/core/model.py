"""
Enigma Model - Scalable Transformer Architecture

A real transformer that can scale from tiny to extra-large:
  - tiny:   64 dim, 2 layers  -> Fast, basic responses (Pi Zero)
  - small:  128 dim, 4 layers -> Good for Pi 4/5
  - medium: 256 dim, 6 layers -> Better quality (Pi 5 8GB)
  - large:  512 dim, 8 layers -> High quality (GPU required)
  - xl:     768 dim, 12 layers -> Very high quality (8GB+ GPU)
  - xxl:    1024 dim, 16 layers -> Near-GPT quality (16GB+ GPU)

The architecture is the same as GPT-style models, just configurable size.
With enough training data, this WILL learn and improve.

To make it smarter:
  1. Add more training data (conversations, Q&A, text)
  2. Train for more epochs
  3. Increase model size (if you have the hardware)
"""
import torch
import torch.nn as nn
from ..config import CONFIG

MAX_LEN = CONFIG.get("max_len", 512)

# Global registry of running model instances (for multi-model support)
_RUNNING_MODELS: dict = {}


def get_running_models() -> dict:
    """Get all currently loaded model instances."""
    return _RUNNING_MODELS.copy()


def is_model_loaded(name: str) -> bool:
    """Check if a model is already loaded."""
    return name in _RUNNING_MODELS


def register_model(name: str, model: 'Enigma'):
    """Register a model as running."""
    _RUNNING_MODELS[name] = model


def unregister_model(name: str):
    """Unregister a model."""
    if name in _RUNNING_MODELS:
        del _RUNNING_MODELS[name]


class Enigma(nn.Module):
    """
    A real transformer model that scales from tiny to extra-large.
    
    This is architecturally identical to GPT - just configurable size.
    It learns patterns from training data and generates based on those patterns.
    
    The model IS expandable:
      - Change dim to increase capacity (smarter but slower)
      - Change depth to add reasoning layers
      - Change heads for better attention patterns
    """
    
    def __init__(self, vocab_size, dim=None, depth=None, heads=None, max_len=MAX_LEN):
        super().__init__()
        
        # Use CONFIG values if not specified (allows easy scaling)
        self.dim = dim or CONFIG.get("embed_dim", 128)
        self.depth = depth or CONFIG.get("num_layers", 4)
        self.heads = heads or CONFIG.get("num_heads", 4)
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Token embeddings - converts words to vectors
        self.token_embed = nn.Embedding(vocab_size, self.dim)
        
        # Position embeddings - tells model where each word is
        self.pos = nn.Parameter(torch.randn(1, max_len, self.dim))
        
        # Transformer layers - the "thinking" part
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.dim, 
                nhead=self.heads,
                dim_feedforward=self.dim * 4,  # Standard 4x expansion
                dropout=0.1,
                activation='gelu'
            )
            for _ in range(self.depth)
        ])
        
        # Normalization for stable training
        self.norm = nn.LayerNorm(self.dim)
        
        # Output head - converts vectors back to word probabilities
        self.head = nn.Linear(self.dim, vocab_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Forward pass - process input and generate output.
        
        input_ids: (batch, seq_len) LongTensor of token IDs
        returns: logits (batch, seq_len, vocab_size) - probability of each word
        """
        seq_len = input_ids.size(1)
        
        # Convert tokens to embeddings and add position info
        x = self.token_embed(input_ids.long()) + self.pos[:, :seq_len]
        
        # Pass through transformer layers (the "thinking")
        for layer in self.layers:
            x = layer(x)
        
        # Normalize and project to vocabulary
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
    
    def get_config(self) -> dict:
        """Return model configuration for saving/loading."""
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "max_len": self.max_len
        }
    
    @classmethod
    def from_config(cls, config: dict):
        """Create model from configuration."""
        return cls(
            vocab_size=config["vocab_size"],
            dim=config.get("dim", 128),
            depth=config.get("depth", 4),
            heads=config.get("heads", 4),
            max_len=config.get("max_len", 512)
        )


# Aliases for compatibility
TinyEnigma = Enigma
EnigmaModel = Enigma
