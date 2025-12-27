"""
Simple training utilities for the TinyEnigma toy model.
"""
import torch
import torch.nn as nn
import os
from pathlib import Path
from .model import TinyEnigma
from .tokenizer import load_tokenizer
from ..config import CONFIG

MODEL_PATH = Path(CONFIG["models_dir"]) / "tiny_enigma.pth"

def train_model(force=False, num_epochs=10, lr=1e-4):
    tokenizer = load_tokenizer()
    # Very small toy dataset - expects a data.txt in project root or enigma/data
    data_file = Path(CONFIG["data_dir"]) / "data.txt"
    if not data_file.exists():
        data_file.write_text("hello world\nthis is a tiny dataset for enigma\n")
    raw = data_file.read_text(encoding="utf-8")

    # encode text
    if hasattr(tokenizer, "encode") or hasattr(tokenizer, "__call__"):
        enc = tokenizer(raw, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG.get("max_len",512))
        if isinstance(enc["input_ids"], list):
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0)
        else:
            input_ids = enc["input_ids"].long()
    else:
        raise RuntimeError("Tokenizer not usable")

    vocab_size_est = getattr(tokenizer, "vocab_size", 10000)
    model = TinyEnigma(vocab_size=vocab_size_est, dim=CONFIG.get("embed_dim",128)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    labels = input_ids.clone()

    if MODEL_PATH.exists() and not force:
        print("[SYSTEM] Model already exists. Skipping training (use force=True).")
        return

    print(f"[SYSTEM] Starting training for {num_epochs} epochs")
    for e in range(num_epochs):
        model.train()
        out = model(input_ids)
        loss = criterion(out.view(-1, out.size(-1)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[SYSTEM] Epoch {e+1}/{num_epochs} loss={loss.item():.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[SYSTEM] Saved model to {MODEL_PATH}")
