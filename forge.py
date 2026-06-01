#!/usr/bin/env python
"""The Forge — train our OWN Enigma model (our tokenizer + our architecture +
our weights), either from scratch or by distilling a teacher.

Two modes, one harness — because scratch vs distill differ ONLY in the data:

  scratch   train a fresh Enigma on a raw text corpus (pure next-token LM).
            The model learns language from nothing but the data we give it.

  distill   train a fresh Enigma on TEACHER-generated data so the student
            bootstraps from a strong brain (e.g. Qwen3) instead of from zero.
            Collect the teacher data first, e.g.:
              python collect_distill_data.py --endpoint http://127.0.0.1:8085/v1 \
                  --model local --magpie 2000 --tag qwen3 --template chatml
            then:  python forge.py --mode distill --data data/finetune/distill_qwen3.txt

Either way the stack is ours: we train a BPE tokenizer on the corpus, build an
Enigma from a ForgeConfig preset, and train it with the in-house Trainer.

Run (proof-of-forge, tiny + fast):
    python forge.py --mode scratch --size tiny --epochs 20
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import replace
from pathlib import Path

import torch

# Windows consoles default to cp1252 and crash printing unicode (e.g. �);
# force UTF-8 so generation samples never take down the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent


def load_corpus(patterns: list[str]) -> tuple[str, list[str]]:
    """Return (full_text, list_of_lines) from files/globs."""
    files: list[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(pat)] or ([Path(pat)] if Path(pat).exists() else [])
        files.extend(hits)
    if not files:
        raise SystemExit(f"No corpus files matched: {patterns}")
    chunks: list[str] = []
    for f in files:
        chunks.append(f.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(chunks)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    print(f"Corpus: {len(files)} file(s), {len(text):,} chars, {len(lines):,} non-empty lines")
    return text, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "distill"], default="scratch")
    ap.add_argument("--data", nargs="+", default=None,
                    help="corpus files/globs (default: data/*.txt for scratch)")
    ap.add_argument("--size", default="tiny", help="ForgeConfig preset")
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.data is None:
        args.data = ["data/*.txt"] if args.mode == "scratch" else ["data/finetune/distill_*.txt"]
    out_dir = Path(args.out) if args.out else ROOT / "models" / f"enigma_forge_{args.size}"

    # 1. Corpus -------------------------------------------------------------
    text, lines = load_corpus(args.data)

    # 2. Our tokenizer ------------------------------------------------------
    from enigma_engine.core.bpe_tokenizer import BPETokenizer
    print(f"Training BPE tokenizer (vocab={args.vocab}) on the corpus ...")
    tok = BPETokenizer()
    tok.use_utf8_bytes = True
    tok.train(lines, vocab_size=args.vocab, verbose=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(out_dir / "tokenizer.json")
    print(f"Tokenizer ready: vocab_size={tok.vocab_size}")

    # 3. Our model ----------------------------------------------------------
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import MODEL_PRESETS
    if args.size not in MODEL_PRESETS:
        raise SystemExit(f"Unknown preset {args.size}; choose from {list(MODEL_PRESETS)}")
    config = replace(MODEL_PRESETS[args.size], vocab_size=tok.vocab_size)
    model = Enigma(config)
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model: Enigma '{args.size}'  dim={config.dim} layers={config.n_layers} "
          f"heads={config.n_heads}  ->  {n_params/1e6:.1f}M params on {device}")

    # 4. Train --------------------------------------------------------------
    from enigma_engine.training.training import Trainer, TrainingConfig
    tc = TrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        use_amp=(device == "cuda"), warmup_steps=10,
        checkpoint_dir=str(out_dir),
    )
    print(f"Training [{args.mode}]: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    trainer = Trainer(model, tok, tc)
    losses: list[float] = []
    trainer.on_loss = lambda loss: losses.append(loss)
    state = trainer.train(text)

    if losses:
        print(f"Loss: {losses[0]:.4f} (start)  ->  {losses[-1]:.4f} (end)   "
              f"[{len(losses)} steps]")

    # 5. Save ---------------------------------------------------------------
    from enigma_engine.core.safe_save import atomic_torch_save
    ckpt = out_dir / "model.pth"
    atomic_torch_save({"model_state_dict": model.state_dict(),
                       "config": config.to_dict()}, str(ckpt))
    print(f"Saved -> {ckpt}")

    # 6. Generate a sample --------------------------------------------------
    print("\n=== sample generations (tiny model on a tiny corpus — expect rough) ===")
    model.eval()
    for prompt in ["The", "Enigma is", "How do I"]:
        ids = torch.tensor([tok.encode(prompt)], device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, temperature=0.8,
                                 top_k=40, top_p=0.9)
        seq = out[0].tolist() if hasattr(out, "tolist") else list(out[0])
        print(f"  {prompt!r} -> {tok.decode(seq)!r}")


if __name__ == "__main__":
    main()
