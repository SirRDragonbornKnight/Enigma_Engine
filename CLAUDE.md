# CLAUDE.md — Enigma Engine

Guidance for Claude Code working in this repo. Keep this file short (target <200 lines).
Harness-enforced rules (permissions, hooks, model) belong in `.claude/settings.json`, NOT here.

## What this is
Two subsystems live here:
1. **Enigma** — a **from-scratch** decoder-only LLM (its own architecture, BPE tokenizer base vocab
   4718, and weights; NOT a wrapper). Python. Pipeline is **pretrain → SFT → serve**; train + serve
   share one chat renderer (`enigma_engine/core/chat_format.py`) so the prompt format can't drift.
2. **Avatar overlay** (`mods/avatar/`) — an Electron + Three.js transparent desktop pet that drives
   any `.glb`/`.vrm` with **pure procedural** motion (no canned animation), composited from masked,
   weighted pose/flex layers fed over a local WebSocket bus. See the Avatar section below.

## Setup / build / test — run these first
- Python 3.12 (`C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe`).
- Enigma tests: `python -m pytest tests/ -q`   ·   Lint: `ruff check`
- Avatar tests: `cd mods/avatar && node --test` (Node built-in runner; ~197 tests, some skip
  without the real model library). `node --check <file>.js` for a quick syntax pass.
- If a fresh session can't run the tests from this section, fix THIS section first.

## The pipeline
- **Pretrain** — `python pretrain_enigma.py` (trains from scratch on the memmapped token corpus).
  Resume the live run with `resume_training.ps1`; watch it with `tail_training_log.ps1`.
- **SFT data** — `python make_sft_data.py` → writes `data/sft/{tool_calls,identity,mix}.jsonl`.
- **Finetune (SFT)** — `python finetune_enigma.py` (base checkpoint → instruct/tool model;
  imports the optimizer/LR "arsenal" from `pretrain_enigma.py`).
- **Serve** — `python serve_enigma.py` (OpenAI-compatible FastAPI server; loads the `.pth`
  checkpoint directly). Run with `--help` for flags.

## Conventions / guardrails
- **Console output must be ASCII** — the Windows cp1252 console can't print `→`, `—`, etc.
  Use ASCII in any script that prints.
- **Do not change the live pretrain defaults** (`--optimizer adamw --schedule cosine`) — they are
  asserted bit-identical to the live training lineage. Muon / WSD are future-run-only, behind flags.
- Checkpoints rotate `latest.pth` → `prev.pth` atomically with a finite-loss guard; resume rebuilds
  config from the checkpoint and hard-fails on any arch/optimizer mismatch.
- Training is **paused** at ~step 183.75k/287.9k (64%, val ppl 3.6), GPU idle; resume with the script above.
- From-scratch ethos: prefer fresh, correct code; engines should fail honestly ("feature absent")
  rather than guess.

## Avatar mod (`mods/avatar/`)
- **Authoritative spec lives OUTSIDE the repo:** `C:\Users\SirKn\3d Avatar\The project is to make a
  3d model t.txt` (REV 6). Models live in `C:\Users\SirKn\3d Avatar\Avatars\`. Judge/recode against
  the SPEC's intent, NOT against what the code currently does — passing tests often just enshrine
  wrong behavior. Loading from that external dir MUST keep working (no path-restricting the loader).
- **Control plane = a local WebSocket bus** (`bus.py`, `ws://127.0.0.1:8765`), driven by `say.py`
  (fire-and-forget) and `tools/avbus.py` (request/reply). The bus is the right way for an AI to drive
  her — pose/look/conjure/say/capabilities, plus inline perform tags in speech (`[pose:role=p/y/r]`,
  `[look:dir]`, `[conjure:x]`). It is **Origin-gated** (blocks browser drive-by / CSWSH); keep it so.
- **SAFETY — fail-safe click-through is load-bearing.** The overlay is transparent and must pass
  clicks THROUGH to the desktop whenever it's unsure the cursor is over her mesh — it once **locked
  the user out of their own desktop**. Never weaken: the arbiter's cursor-display gate, the
  `_forceThrough` panic latch, or the panic key (`Ctrl+Shift+Alt+C` = force-through, `…+Q` = quit,
  tray = independent reclaim). Every hit-test failure mode must default to THROUGH, never CAPTURE.
- **Guard at the engine boundary, not the caller.** The "fail honestly" ethos applies per-engine:
  validate inputs (finite numbers, well-formed shapes) where they ENTER an engine (setLayer, setMouth,
  throwProp, the loader), so a bad bus message degrades honestly instead of permanently bricking a
  bone/mouth/sim. Don't rely on the live caller happening to pass clean data.
- **Generic-only:** no per-model rig overrides, no canned gestures/emotes. Fit rigs via the 19-role
  cascade (VRM → name → geometry → between); author motion via pose/flex/setFingers. Don't re-add the
  removed override mechanism.

## Working style
- "Make a plan first" means present the plan and **stop for approval** — don't build it in the same pass.
- Scope to exactly what's asked; deliver small, verify, then continue.

## Project state docs
`CLEANUP_TRACKER.md`, `CODE_REVIEW.md`, `KNOWN_ISSUES.md`, `SUGGESTIONS.md`.
