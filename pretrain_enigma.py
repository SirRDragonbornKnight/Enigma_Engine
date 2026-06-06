#!/usr/bin/env python
"""Pretrain the REAL Enigma — our own architecture, our own weights — on the
pre-tokenized ``data/pretrain/tokens.bin`` corpus (56.6B tokens, vocab 4718).

This is the genuine own-brained model: a from-scratch transformer that learns
language from the data, NOT a wrapper around someone else's model. The toy
``forge.py`` reads the whole corpus into RAM and trains a fresh tokenizer — it
cannot scale to a 211 GB binary. This loads the SAME AdvancedBPETokenizer that
produced ``tokens.bin`` and streams the tokens via memmap (nanoGPT-style).

  python pretrain_enigma.py --sanity                  # 1-step smoke test, then exit
  python pretrain_enigma.py --size base --tokens 2e9  # the real run (~GPT-2-small)
  python pretrain_enigma.py --resume models/enigma_pretrain_base/latest.pth

Checkpoints (model_state_dict + config + step + optimizer) land in
``models/enigma_pretrain_<size>/latest.pth`` every --save-every steps and are
fully resumable. The final model is written as ``model.pth`` in the standard
{model_state_dict, config} format the rest of the stack already loads.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:  # Windows consoles default to cp1252 and crash on unicode sample text.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
TOKENS_BIN = ROOT / "data" / "pretrain" / "tokens.bin"
TOKENS_META = ROOT / "data" / "pretrain" / "tokens.json"
HEADER_BYTES = 256  # ETOK reserved header (see pretokenize_data.py)


def get_lr(step: int, warmup: int, total: int, peak: float, min_ratio: float = 0.1) -> float:
    """Linear warmup → cosine decay to ``min_ratio * peak``."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return peak * min_ratio
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", help="ForgeConfig preset (tiny..xl)")
    ap.add_argument("--block", type=int, default=1024, help="sequence length")
    ap.add_argument("--micro-batch", type=int, default=12, help="sequences per forward")
    ap.add_argument("--grad-accum", type=int, default=16, help="micro-batches per optimizer step")
    ap.add_argument("--tokens", type=float, default=2e9, help="target training tokens")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout (0.0 for single-epoch pretraining; presets default to 0.1)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=250, help="steps between checkpoints")
    ap.add_argument("--eval-every", type=int, default=250, help="steps between val-loss checks")
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--val-tokens", type=int, default=10_000_000, help="tail tokens held out for val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--no-diff-attn", action="store_true",
                    help="disable differential attention -> use fused SDPA kernel (2-4x faster)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile the model (~1.5-2x; --no-compile for eager / if Triton is absent)")
    ap.add_argument("--sanity", action="store_true", help="one fwd/bwd step then exit")
    args = ap.parse_args()

    if not TOKENS_BIN.exists():
        raise SystemExit(f"missing corpus: {TOKENS_BIN}")
    meta = json.loads(TOKENS_META.read_text(encoding="utf-8"))
    if meta.get("dtype") != "uint32":
        raise SystemExit(f"expected uint32 tokens, got {meta.get('dtype')}")
    vocab_meta = meta["vocab_size"]
    print(f"corpus: {meta['total_tokens']:,} tokens, vocab {vocab_meta}, "
          f"{meta['file_size_gb']} GB ({meta['tokenizer']})", flush=True)

    # Vocab is authoritative from the corpus metadata: the model trains on the
    # raw token IDs, so it doesn't need the tokenizer at all. We still try to
    # load the exact one that produced tokens.bin (AdvancedBPETokenizer, via
    # 'bpe' — never 'auto', which would grab tiktoken) for readable samples,
    # but training proceeds fine without it.
    vocab_size = vocab_meta
    try:
        from enigma_engine.core.tokenizer import get_tokenizer
        tok = get_tokenizer("bpe")
        if getattr(tok, "vocab_size", None) != vocab_size:
            print(f"  WARN: tokenizer vocab {getattr(tok, 'vocab_size', '?')} != "
                  f"corpus vocab {vocab_size}; using corpus vocab", flush=True)
    except Exception as exc:
        tok = None
        print(f"  (tokenizer unavailable — training on raw IDs: {exc})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # memmap the uint32 token stream after the 256-byte header.
    data = np.memmap(TOKENS_BIN, dtype=np.uint32, mode="r", offset=HEADER_BYTES)
    n = len(data)
    val_n = min(args.val_tokens, n // 100)
    train_end = n - val_n
    print(f"memmapped {n:,} tokens  (train {train_end:,} / val {val_n:,})", flush=True)

    block = args.block

    def get_batch(split: str):
        lo, hi = (0, train_end) if split == "train" else (train_end, n)
        ix = np.random.randint(lo, hi - block - 1, size=args.micro_batch, dtype=np.int64)
        x = np.stack([np.asarray(data[i:i + block], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(data[i + 1:i + 1 + block], dtype=np.int64) for i in ix])
        X = torch.from_numpy(x).to(device, non_blocking=True)
        Y = torch.from_numpy(y).to(device, non_blocking=True)
        return X, Y

    # Build the model from a preset, sized to the corpus vocab.
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig, get_preset
    # On resume, rebuild config from the CHECKPOINT (exact architecture) rather than the
    # preset. Otherwise a flag mismatch (e.g. forgetting --no-diff-attn) builds a
    # differently-shaped model and load_state_dict(strict=False) silently leaves the
    # mismatched tensors at random init — a silent corruption. Trust the checkpoint.
    ck = None
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        config = ForgeConfig.from_dict(ck["config"])
        print(f"resume: config rebuilt from checkpoint ({args.resume})", flush=True)
    else:
        config = get_preset(args.size, vocab_size=vocab_size)
        config.neftune_alpha = 0.0  # NEFTune is a finetuning trick; off for pretraining
        if args.no_diff_attn:
            config.use_differential_attn = False  # fused SDPA; must stay consistent across resumes
    config.dropout = args.dropout  # 0.0 for single-epoch pretraining (preset default 0.1 undertrains)
    if block > config.max_seq_len:
        config.max_seq_len = block
    model = Enigma(config)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)
    raw_model = model  # the real nn.Module; checkpoints/optimizer bind here even when compiled
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Enigma '{args.size}': {n_params/1e6:.1f}M params, dim={config.dim} "
          f"layers={config.n_layers} heads={config.n_heads} block={block} "
          f"diff_attn={config.use_differential_attn}", flush=True)

    # Weight-decay only tensors with dim>=2 (matmuls, embeddings); never 1-D params
    # (RMSNorm gains) — decaying norm scales mildly hurts. Standard GPT-2/nanoGPT split.
    _decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    _no_decay = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": _decay, "weight_decay": args.weight_decay},
         {"params": _no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    print(f"optim: weight-decay on {len(_decay)} tensors, none on {len(_no_decay)}", flush=True)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tokens_per_step = args.micro_batch * args.grad_accum * block
    total_steps = max(1, int(args.tokens / tokens_per_step))

    out = Path(args.out) if args.out else ROOT / "models" / f"enigma_pretrain_{args.size}"
    out.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if ck is not None:
        missing, unexpected = raw_model.load_state_dict(ck["model_state_dict"], strict=False)
        # config came from the checkpoint, so any real mismatch signals corruption —
        # hard-fail rather than silently train half-random weights. (freqs_cis / causal
        # mask are non-persistent buffers recomputed at build time; ignore those.)
        real_missing = [k for k in missing if "freqs_cis" not in k and "causal_mask" not in k]
        if unexpected or real_missing:
            raise SystemExit(f"resume arch mismatch — refusing to corrupt: "
                             f"missing={real_missing[:5]} unexpected={unexpected[:5]}")
        if "optimizer" in ck:
            optim.load_state_dict(ck["optimizer"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    # torch.compile after any resume-load so weights land in raw_model first. The
    # compiled wrapper is used only for fwd/bwd; save/load/optimizer stay on
    # raw_model so the `_orig_mod.` prefix never leaks into checkpoints. Eager
    # fallback keeps the run alive where inductor/Triton is unavailable (Windows).
    if args.compile and device == "cuda":
        try:
            compiled = torch.compile(raw_model)
            # torch.compile traces lazily on first call, so a missing-Triton /
            # inductor failure would otherwise crash mid-run. Force the compile
            # NOW on the real training shape (same graph the loop uses -> no later
            # recompile); on any failure fall back to eager. Throwaway grads cleared.
            _x = torch.zeros((args.micro_batch, block), dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, _loss = compiled(_x, targets=_x)
            _loss.backward()
            optim.zero_grad(set_to_none=True)
            model = compiled
            print("torch.compile: enabled", flush=True)
        except Exception as exc:
            optim.zero_grad(set_to_none=True)
            model = raw_model
            print(f"torch.compile: unavailable -> eager ({str(exc).splitlines()[0][:140]})", flush=True)

    def save(tag: str, step: int):
        from enigma_engine.core.safe_save import atomic_torch_save
        atomic_torch_save({
            "model_state_dict": raw_model.state_dict(),
            "config": config.to_dict(),
            "step": step,
            "optimizer": optim.state_dict(),
        }, str(out / tag))
        (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    @torch.no_grad()
    def estimate_val() -> float:
        raw_model.eval()
        losses = []
        for _ in range(args.eval_iters):
            X, Y = get_batch("val")
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(X, targets=Y)
            losses.append(loss.item())
        raw_model.train()
        return sum(losses) / max(1, len(losses))

    # --- sanity: one fwd/bwd, report loss vs random baseline, exit ----------
    if args.sanity:
        raw_model.train()
        X, Y = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(X, targets=Y)
        loss.backward()
        base = math.log(vocab_meta)
        print(f"[sanity] batch={tuple(X.shape)} loss={loss.item():.4f} "
              f"(random baseline ln(V)={base:.3f}) — pipeline OK", flush=True)
        return

    print(f"training: target {args.tokens/1e9:.2f}B tokens over {total_steps:,} steps | "
          f"{tokens_per_step:,} tok/step (mb {args.micro_batch} x ga {args.grad_accum} x {block}) | "
          f"amp={'bf16' if use_bf16 else 'fp16'} ckpt={not args.no_grad_ckpt}", flush=True)

    raw_model.train()
    t0 = time.time()
    base_tokens = start_step * tokens_per_step
    seen = base_tokens
    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup, total_steps, args.lr)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            X, Y = get_batch("train")
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(X, targets=Y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        if use_scaler:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()
        seen += tokens_per_step

        if step % 10 == 0:
            dt = max(1e-9, time.time() - t0)
            tps = (seen - base_tokens) / dt
            print(f"step {step}/{total_steps} loss {loss_acc:.4f} lr {lr:.2e} "
                  f"{tps:,.0f} tok/s {seen/1e9:.3f}B", flush=True)

        if step > start_step and step % args.eval_every == 0:
            vl = estimate_val()
            print(f"  [val] step {step} loss {vl:.4f} ppl {math.exp(min(20, vl)):.1f}", flush=True)

        if step > start_step and step % args.save_every == 0:
            save("latest.pth", step)
            print(f"  [ckpt] step {step} -> {out/'latest.pth'}", flush=True)

    save("model.pth", total_steps)
    print(f"done -> {out/'model.pth'}  ({total_steps:,} steps, {seen/1e9:.2f}B tokens)", flush=True)


if __name__ == "__main__":
    main()
