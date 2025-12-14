"""
Minimal inference wrapper using the toy TinyEnigma model.
"""
import torch
from .model import TinyEnigma
from .tokenizer import load_tokenizer
from ..config import CONFIG
from pathlib import Path

MODEL_PATH = Path(CONFIG["models_dir"]) / "tiny_enigma.pth"

class EnigmaEngine:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer()
        vocab_size = getattr(self.tokenizer, "vocab_size", 5000)
        self.model = TinyEnigma(vocab_size=vocab_size, dim=CONFIG.get("embed_dim",128))
        if MODEL_PATH.exists():
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_gen: int = 30, temperature: float = 1.0):
        # encode
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device).long()
        with torch.no_grad():
            for _ in range(max_gen):
                logits = self.model(input_ids)
                last = logits[:, -1, :] / max(1e-8, temperature)
                probs = torch.softmax(last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        # decode
        try:
            text = self.tokenizer.decode(input_ids[0].cpu().numpy(), skip_special_tokens=True)
        except Exception:
            # fallback
            text = " ".join(map(str, input_ids[0].cpu().numpy().tolist()))
        return text
