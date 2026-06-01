# Known Issues — Enigma AI Engine

_Snapshot compiled 2026-06-01. This is a navigation layer over the detailed trackers, not a replacement._
_Authoritative trackers: `SUGGESTIONS.md` (active strategy + execution blocks), `CODE_REVIEW.md` (bugs/audits), `CLEANUP_TRACKER.md` (file-by-file cleanup)._

## 1. UI direction — RESOLVED June 1 2026 ✅
- **Ruling (2026-06-01):** the **interim main-chat UI is the terminal client** — `python run.py --client-chat --model <path>` — verified working end-to-end against the 30B GGUF (streaming, autospawn; see SUGGESTIONS.md June 1 closures TC-1..TC-3). No new UI framework; the fastest path to actually using the brain, and the user is fine chatting in PowerShell.
- **Gradio stays the planned richer GUI** (Block 2) for when sliders / image upload / a browser window are wanted — still pure Python, still reuses `client.py`. Not started (`gradio` not in deps); now **optional/deferred**, no longer a blocker.
- **tkinter (`enigma_engine/gui/`) and Svelte (`enigma_engine/web/`) stay slated for deletion** (CLEANUP_TRACKER) — neither is the path forward. The 2026-06-01 Svelte+pywebview exploration was dropped.

## 2. Three parallel UIs in the tree
- `enigma_engine/gui/` — tkinter, ~30K LOC / 26 files, several 1.5–3.5K-line god-modules. Works today. Marked for deletion.
- `enigma_engine/web/` — Svelte, ~782 LOC. **Does not build** (see #3). Marked for deletion.
- `enigma_engine/ui.py` — Gradio. Planned (Block 2). Does not exist yet.

## 3. Svelte web frontend is non-buildable _(relevant only if Svelte is chosen over Gradio)_
- `src/App.svelte` imports 5 pages that don't exist as files: `Training`, `Files`, `Models`, `Config`, `Terminal` → `vite build` fails.
- Only `pages/Chat.svelte` exists; `components/` is empty.
- `lib/api.ts` calls `/api/files/list|read|upload` endpoints that don't exist server-side.
- Chat renders plain text — no markdown/highlighting despite `marked` + `highlight.js` already in `package.json`.

## 4. Backend API surface is narrow _(matters for any HTTP frontend; largely moot for Gradio, which calls the engine in-process via `client.py`)_
`api/server.py` exposes ~25 endpoints (chat, models load/unload/list, profiles, config, history, style-preferences, one generic `/api/train`). No endpoints for: ROUTER assignment, model create/merge/grow/delete/download, tokenizer training, the ~19 FORGE training modes + queue + GGUF export + dataset review, DOCS file CRUD, CMD/shell, mods.

## 5. Code-health debt _(direction-independent)_
- **God-modules:** `training/training.py` (6038 L), `engine_generation.py` (3053 L), `model.py` / `inference.py` (~1800 L each), plus the tkinter GUI files. Already tracked in `CLEANUP_TRACKER.md`.
- **541 `except Exception`** across 66 files — some intentional resilience, but at this volume some likely mask real errors. Audit candidate. (Zero bare `except:`, zero TODO/FIXME/HACK — otherwise strong discipline.)
- **15 ruff issues** — all minor; mostly unused optional-import probes in `mods/vision/vision.py` (use `importlib.util.find_spec`). 10 auto-fixable.

## 6. Docs / config / env drift _(low severity)_
- **Port mismatch:** `run.py` and `api/server.py` use **8080**; `information/getting_started.md` says **5000**.
- **Missing README:** `pyproject.toml` sets `readme = "README.md"` but there is no root `README.md`.
- **Env sprawl:** three virtualenvs in tree (`venv/`, `.venv/`, `.venv314/`) — gitignored, but confusing.

## Blocked-on-UI items
`SUGGESTIONS.md` defers work until the UI (Block 2) ships — e.g. PERSONA-2 Slice 5 (FIX-button correction categorization). These unblock once the UI direction is resolved and built.
