#!/usr/bin/env python
"""Train Enigma's personality LoRA adapter on Qwen3-8B (Block 4.3).

Standard transformers + PEFT path: load Qwen3-8B (bf16), attach a LoRA adapter
on the attention projections, and SFT on data/personality_corpus.jsonl with the
prompt tokens masked out of the loss (the model only learns to *produce* the
assistant turn, not to predict the user's). Saves a portable PEFT adapter to
models/enigma_lora_v1/ and then demos the voice on a few held-out prompts.

Why this and not the in-house dispatcher LoRA path: LoraTrainer.train()
(lora_utils.py ~L983) reads `logits = outputs[0] if isinstance(outputs, tuple)
else outputs`, which mishandles an HF CausalLMOutput (it's neither a tuple nor a
raw tensor) -> .reshape() crashes. That pipeline was built for the custom Enigma
model; fixing it for HF models is tracked separately. This script is the
reliable route to an actual adapter today.

Run:  python train_enigma_lora.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "qwen3-8b"
DEFAULT_CORPUS = ROOT / "data" / "personality_corpus.jsonl"
DEFAULT_OUT = ROOT / "models" / "enigma_lora_v1"

DEMO_PROMPTS = [
    "Who are you?",
    "What's the best programming language?",
    "2 + 2 = 5, right?",
    "I'm stressed about a deadline.",
]


def load_corpus(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_examples(rows, tokenizer, max_length):
    """Tokenize prompt+completion; mask prompt tokens in the labels (-100)."""
    examples = []
    for row in rows:
        prompt, completion = row["prompt"], row["completion"]
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(prompt + completion,
                             add_special_tokens=False).input_ids[:max_length]
        labels = list(full_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100  # don't train on the user's turn
        if all(t == -100 for t in labels):
            continue  # completion got truncated away — skip
        examples.append((full_ids, labels))
    return examples


@torch.no_grad()
def demo(model, tokenizer, prompts):
    model.eval()
    model.config.use_cache = True  # re-enable KV cache for fast generation
    for q in prompts:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(text, add_special_tokens=False,
                        return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, max_new_tokens=160, do_sample=True, temperature=0.7,
            top_p=0.9, repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                 skip_special_tokens=True).strip()
        print(f"\n  You: {q}\n  Enigma: {reply}")
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer + model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16)
    model.to("cuda")
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # grads must flow through frozen embeds
    model.print_trainable_parameters()

    rows = load_corpus(Path(args.corpus))
    examples = build_examples(rows, tokenizer, args.max_length)
    print(f"Corpus: {len(rows)} rows -> {len(examples)} trainable examples")

    steps_per_epoch = max(1, math.ceil(len(examples) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):  # 5% warmup then cosine decay to 10%
        warmup = max(1, int(0.05 * total_steps))
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)

    print(f"Training: {args.epochs} epochs, {total_steps} optimizer steps, "
          f"lr={args.lr}, rank={args.rank}/alpha={args.alpha}")
    model.train()
    gstep = 0
    for epoch in range(args.epochs):
        # Vary order per epoch without RNG (Date/random are fine here):
        order = list(range(len(examples)))
        order = order[epoch % max(1, len(order)):] + order[:epoch % max(1, len(order))]
        optimizer.zero_grad()
        running = 0.0
        for i, idx in enumerate(order):
            input_ids, labels = examples[idx]
            input_ids = torch.tensor([input_ids], device="cuda")
            labels = torch.tensor([labels], device="cuda")
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                gstep += 1
        avg = running / max(1, len(order))
        print(f"  epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nSaved adapter -> {out_dir}")

    print("\n=== Enigma (fine-tuned) on held-out prompts ===")
    demo(model, tokenizer, DEMO_PROMPTS)


if __name__ == "__main__":
    main()
