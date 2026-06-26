# Modkit

**The AI's capability backend.** Modkit forges your own local models and exposes
*mods* — capabilities an AI controls — to a front-end like
[Odysseus](https://github.com/pewdiepie-archdaemon/odysseus).

## The shape of it

- **Odysseus** = the head + app (chat, agent, memory, UI). It runs the models and talks to the user.
- **Modkit** (this repo) = the AI's *hands*. The **Forge** (train models), the **avatar**, device/thing control — exposed to Odysseus's agent over **MCP**, so the AI can call them as tools.
- **Enigma** = a local model, forged here, served to Odysseus.

Models are *run* by Odysseus + an external runner (Ollama / llama.cpp / vLLM). Modkit *makes* and *manages* them — it's the factory, not the engine.

## What's here

### The Forge — make your own models
- `forge.py` — train a custom Enigma model from scratch, or by distilling a teacher.
- `train_enigma_lora.py` — LoRA-fine-tune a base model (e.g. Qwen3-8B) into Enigma.
- `make_enigma_corpus.py` — build a personality corpus (`data/enigma_voice.md` is the voice spec).
- `make_enigma_local.py` — merge a LoRA adapter into a standalone local model.
- `eval_enigma.py` — base-vs-fine-tune side-by-side eval.
- `enigma_engine/core/` (model, presets, tokenizer) + `enigma_engine/training/` — the training stack.
- `collect_*.py` — data collection (pretraining / fine-tuning / distillation).

### The mods — capabilities the AI controls
- `mods/` — self-contained mods (each a folder with a `mod.json`), discovered by `enigma_engine/core/mod_tools.py`.
- `mods/avatar/` — the planned 3D, rigged, swappable avatar.

## Status

Refactored from the former **"Enigma AI Engine"** monolith: its chat / inference
/ serving / GUI layers were removed in favour of Odysseus (see `MODKIT_PLAN.md`).
The import package is still named `enigma_engine` — a follow-up renames it to
`modkit`.

**Next:** an **MCP server** exposing the mods to Odysseus, a `ModBase` SDK for
authoring mods, validation on mod load, and the avatar.

## License

MIT.
