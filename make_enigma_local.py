#!/usr/bin/env python
"""Make Enigma local: merge the personality LoRA into Qwen3-8B so Enigma is a
single self-contained local model (no adapter juggling) you can run and serve
like any other model. Output: models/enigma-8b/ (HF safetensors)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = Path(__file__).resolve().parent
BASE = ROOT / "models" / "qwen3-8b"
ADAPTER = ROOT / "models" / "enigma_lora_v1"
OUT = ROOT / "models" / "enigma-8b"

def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading base + adapter ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    print("Merging LoRA into the base weights ...", flush=True)
    model = model.merge_and_unload()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Saving standalone Enigma -> {OUT}", flush=True)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    n = sum(p.numel() for p in model.parameters())
    print(f"Done — Enigma is now a standalone local model ({n/1e9:.1f}B params) at {OUT}")

if __name__ == "__main__":
    main()
