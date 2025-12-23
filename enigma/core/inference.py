"""
Minimal inference wrapper using the TinyEnigma model.
Works with the model registry system.
"""
import torch
from .model import TinyEnigma
from .tokenizer import load_tokenizer
from ..config import CONFIG
from pathlib import Path


class EnigmaEngine:
    """
    Inference engine for generating text with TinyEnigma models.
    Can accept a pre-loaded model or load the default model.
    """
    
    def __init__(self, model=None, device=None):
        """
        Initialize the inference engine.
        
        Args:
            model: Pre-loaded TinyEnigma model (optional)
            device: Device to run on (auto-detected if not specified)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer()
        
        if model is not None:
            # Use provided model
            self.model = model
        else:
            # Load default model from registry
            from .model_registry import ModelRegistry
            registry = ModelRegistry()
            default_name = CONFIG.get("default_model", "sacrifice")
            self.model, _ = registry.load_model(default_name)
        
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_gen: int = 30, temperature: float = 1.0):
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: Input text to continue
            max_gen: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text including the prompt
        """
        # Encode prompt
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device).long()
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_gen):
                logits = self.model(input_ids)
                last = logits[:, -1, :] / max(1e-8, temperature)
                probs = torch.softmax(last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode output
        try:
            text = self.tokenizer.decode(input_ids[0].cpu().numpy(), skip_special_tokens=True)
        except Exception:
            # Fallback for edge cases
            text = " ".join(map(str, input_ids[0].cpu().numpy().tolist()))
        
        return text
