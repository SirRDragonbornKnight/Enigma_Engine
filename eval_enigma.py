#!/usr/bin/env python
"""Side-by-side eval: base Qwen3-8B vs Enigma (LoRA) on HELD-OUT prompts.

Loads the base model once, attaches the trained adapter, and for each prompt
generates twice on the same weights — adapter disabled (= stock Qwen3) and
enabled (= Enigma) — so the only variable is the personality LoRA. The prompts
are deliberately NOT in data/personality_corpus.jsonl, so this measures whether
the voice generalizes rather than memorizes.

Run:  python eval_enigma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Windows consoles default to cp1252 and mangle em-dashes etc.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "models" / "qwen3-8b"
ADAPTER = ROOT / "models" / "enigma_lora_v1"

# None of these appear in the training corpus.
HELD_OUT = [
    "Explain recursion to a beginner.",
    "Should I learn Rust or Go?",
    "Write a haiku about debugging.",
    "Is it okay to lie to protect someone's feelings?",
    "How do I center a div in CSS?",
    "What happens to us after we die?",
]


def gen(model, tokenizer, q: str) -> str:
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=160,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.input_ids.shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base {BASE} + adapter {ADAPTER} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    model = AutoModelForCausalLM.from_pretrained(str(BASE), dtype=torch.bfloat16)
    model.to("cuda")
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()

    for q in HELD_OUT:
        with model.disable_adapter():
            base = gen(model, tokenizer, q)  # stock Qwen3
        enigma = gen(model, tokenizer, q)  # adapter active
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        print(f"\n[ base Qwen3 ]\n{base}")
        print(f"\n[ Enigma    ]\n{enigma}")


if __name__ == "__main__":
    main()
