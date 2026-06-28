# Enigma Engine

This repository holds **two related but distinct projects**. They share a folder (and,
optionally, a local message bus) but they are NOT the same thing — keep them straight:

| | What it is | Lives in |
|---|---|---|
| **Enigma** — the engine | A **from-scratch decoder-only LLM**: its own architecture, BPE tokenizer (base vocab 4718), and weights. NOT a wrapper around another model. Pipeline: **pretrain -> SFT -> serve**. | `enigma_engine/` + the root pipeline scripts (`pretrain_enigma.py`, `make_sft_data.py`, `finetune_enigma.py`, `serve_enigma.py`) |
| **The Avatar** — the overlay | An **Electron desktop overlay** that floats a rigged 3D model (any `.glb`/`.gltf`/`.vrm`/`.fbx`) on your screen and animates it with **pure procedural motion** + spring physics + lip-sync. An optional *body* an LLM can drive. | `enigma-avatar/` |

A **third location lives OUTSIDE this repo** and belongs to the avatar:

- `C:\Users\SirKn\3d Avatar\` — the avatar's **GLB models** (`Avatars/`) and the original
  **design spec** (`The project is to make a 3d model t.txt`). The in-repo `enigma-avatar/models/`
  is gitignored (large + non-redistributable); models are sourced from here.

## How the two relate

Enigma is the **brain**; the avatar is an optional **body**. They meet only at a local
WebSocket bus (`ws://127.0.0.1:8765`, see `enigma-avatar/bus.py`): a served LLM — Enigma, or
Odysseus, or any OpenAI-compatible model — sends speech + motion commands and the overlay
renders them. **Either runs without the other**: you can pretrain/serve Enigma with no
avatar, and run the avatar against any LLM.

## Where to look

- **The LLM** — [`CLAUDE.md`](CLAUDE.md) is the authoritative guide: setup, the
  `pretrain -> SFT -> serve` pipeline, checkpointing, and guardrails.
- **The avatar** — [`enigma-avatar/STATUS.md`](enigma-avatar/STATUS.md) (what works + how to
  launch it) and [`enigma-avatar/TODO.md`](enigma-avatar/TODO.md) (backlog / audit log).
- **Project state** — `CLEANUP_TRACKER.md`, `CODE_REVIEW.md`, `KNOWN_ISSUES.md`, `SUGGESTIONS.md`.

## A note on older docs ("Modkit")

This repo is mid-transition. It began as **"Enigma AI Engine"**, briefly became **"Modkit"**
(a *Forge* that LoRA-fine-tuned Qwen, plus an MCP layer for Odysseus), and is now the
**from-scratch Enigma LLM** described in `CLAUDE.md`. The Modkit-era entry points
(`forge.py`, `train_enigma_lora.py`, `make_enigma_local.py`, `modkit_mcp.py`) have been
removed. **If a doc still says "Modkit" or references those files, it predates the pivot —
`CLAUDE.md` is the current source of truth.**

## License

MIT (see [`LICENSE`](LICENSE)).
