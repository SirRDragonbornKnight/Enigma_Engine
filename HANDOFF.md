# Modkit — Hand-off / Start Here

> **⚠️ STATUS (2026-06-04) — read first; the sections below are partly stale.**
> There are now **two** "Enigmas" and the direction has moved:
> - **Qwen3-8B + LoRA** → `models/enigma-8b/`, served by `serve_enigma.py` (capable, tool-calling). §1–2 below describe this path and are still valid for *using* a capable Enigma today.
> - **From-scratch transformer** → `models/enigma_pretrain_base/` — the "own-brained" Enigma (~350M-param / GPT-2-class, vocab 4718, trained from `data/pretrain/tokens.bin`, 56.6B tokens) via `pretrain_enigma.py`. This is the current research direction. It needs its **own** serving runtime (`sample_enigma.py`, `kv_cache.py`) — a custom architecture can't be run by Odysseus/llama.cpp/Ollama.
> **Open decision (see the 2026-06-04 whole-project audit):** *capability* (Qwen+LoRA is far stronger today) vs *craft* (a genuinely own brain, but GPT-2-class for now). Resolve this before investing further in either path.

Everything you need to use what we built. **The model runs locally; the front-end is Odysseus.**

## 1. Chat with Enigma in Odysseus

Enigma = Qwen3-8B fine-tuned with its own voice, merged into `models/enigma-8b/`.

1. Start the Enigma server (Modkit's serving glue):
   ```
   python serve_enigma.py                # 4-bit NF4, ~9 GB VRAM (default)
   python serve_enigma.py --quant bf16   # full precision, ~16 GB (max quality)
   ```
   Serves an OpenAI-compatible API at `http://127.0.0.1:8000/v1` (Ctrl-C to stop).
   *(It's running right now in 4-bit from our session — you can skip straight to step 2.)*
2. In Odysseus chat:
   ```
   /setup local http://127.0.0.1:8000/v1
   ```
3. Open the model picker, choose **`enigma`**, and talk to it.

Your own model, on your machine, in your own workstation. Nothing leaves the box.

## 2. Improve Enigma (the Forge)

| Do this | Command |
|---|---|
| Edit the voice/identity | edit `data/enigma_voice.md` |
| Expand the training corpus | edit + run `python make_enigma_corpus.py` |
| Re-train the adapter | `python train_enigma_lora.py` → `models/enigma_lora_v1/` |
| Merge adapter → standalone model | `python make_enigma_local.py` → `models/enigma-8b/` |
| Compare base vs Enigma | `python eval_enigma.py` |
| Train from scratch / distill | `python forge.py` (see `MODKIT_PLAN.md`) |

## 3. Repo state — and how to finalize it

This branch (`refocus/modkit`, pushed) is **Modkit**: the old ~100K-line monolith (GUI, chat app,
inference runtime) was removed; what's left is the **Forge + the mods + Enigma**. 324 tests pass, ruff clean.

**`main` is still the full old monolith** (safe at `dde5c99`, plus the `legacy` branch). When you're
happy with the diff, make Modkit the live `main`:
```
git diff main refocus/modkit          # review
git checkout main && git merge --ff-only refocus/modkit && git push
```
Don't want it? Just stay on `main` — nothing is lost.

Deps to run the scripts: `pip install -e ".[full,server]"` (already installed in this env).

## 4. What's next (from the audit)

- **MCP server** — expose Modkit's mods to Odysseus so Enigma can *control* them (train, avatar,
  devices), not just chat. The leap from "a model in a chat box" to "an AI that runs its own backend."
- **Restore Forge test coverage** — deep model/training/tokenizer tests are in git history (`dde5c99`); only smoke-tested now. *(Partly restored: the from-scratch Enigma's KV-cache serving path is now locked by `tests/test_model_kv_cache.py` — cached decode is verified logit-for-logit equal to a full no-cache recompute. Training/tokenizer depth still pending.)*
- **The 3D rigged avatar** — last.
- Enigma **v2** (2026-06-02): corpus expanded 50→**130 examples** (38% identity) and retrained at **rank 32** → identity now **locks** (answers "Are you Qwen?" with "No, I'm Enigma"; owns the Qwen3 base honestly), and native tool-calling still fires. Remaining wrinkle: occasional clipped endings at temp 0.7 (mild overfit at 0.13 train loss) — fewer epochs or more corpus would smooth it.

## Models on disk
- `models/qwen3-8b/` — base model.
- `models/enigma_lora_v1/` — the personality LoRA adapter.
- `models/enigma-8b/` — Enigma merged (what `serve_enigma.py` serves).
- `models/enigma_forge_tiny/` — the from-scratch Forge proof-of-pipeline.
