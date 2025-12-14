"""
Toy transformer-like encoder model for educational purposes.
Not production-grade. Designed to be small and easy to read.
"""
import torch
import torch.nn as nn
from ..config import CONFIG

MAX_LEN = CONFIG.get("max_len", 512)

class TinyEnigma(nn.Module):
    def __init__(self, vocab_size, dim=128, depth=4, heads=4, max_len=MAX_LEN):
        super().__init__()
        self.dim = dim
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len) LongTensor
        returns logits (batch, seq_len, vocab_size)
        """
        x = self.token_embed(input_ids.long()) + self.pos[:, :input_ids.size(1)]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits
