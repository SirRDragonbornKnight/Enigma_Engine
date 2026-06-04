#!/usr/bin/env python
"""Generate text from a pretrained Enigma checkpoint (the from-scratch model).

  python sample_enigma.py                                   # uses latest base checkpoint
  python sample_enigma.py --ckpt models/enigma_pretrain_base/latest.pth --prompts "The" "In 1850"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "enigma_pretrain_base" / "latest.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--prompts", nargs="+", default=None)
    args = ap.parse_args()

    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig
    from enigma_engine.core.model_utils import sample_next_token
    from enigma_engine.core.tokenizer import get_tokenizer

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    tok = get_tokenizer("bpe")
    eos = getattr(tok, "eos_token_id", 2)
    bos = getattr(tok, "bos_token_id", 1)

    ck = torch.load(args.ckpt, map_location=device)
    config = ForgeConfig.from_dict(ck["config"])
    model = Enigma(config)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"loaded {args.ckpt}\n  step {ck.get('step','?')} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params | device {device}", flush=True)

    prompts = args.prompts or [
        "The history of",
        "In 1066, ",
        "Water is made of",
        "The capital of France is",
        "She opened the door and ",
    ]
    for p in prompts:
        ids_list = tok.encode(p)
        if ids_list and ids_list[-1] == eos:   # drop trailing EOS so it keeps going
            ids_list = ids_list[:-1]
        if not ids_list or ids_list[0] != bos:
            ids_list = [bos] + ids_list
        gen = torch.tensor([ids_list], device=device)
        # No-cache loop: full recompute each step. The KV-cache module was pruned
        # in the refocus; for short samples plain recompute is plenty fast and
        # avoids that dependency. (Restore kv_cache.py for efficient serving.)
        with torch.no_grad():
            for _ in range(args.max_new):
                logits = model(gen)  # use_cache=False (default) -> returns logits
                nxt = sample_next_token(logits[:, -1, :], gen, args.temperature,
                                        args.top_k, args.top_p, 1.1)
                gen = torch.cat([gen, nxt], dim=1)
                if int(nxt.item()) == eos or gen.shape[1] >= config.max_seq_len:
                    break
        text = tok.decode(gen[0].tolist())
        print(f"\n>>> {p!r}\n    {text!r}", flush=True)


if __name__ == "__main__":
    main()
