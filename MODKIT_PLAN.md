# Modkit — Refocus Plan

"Enigma AI Engine" becomes **Modkit**: the AI's capability backend.

- **Odysseus** = head + backbone (chat, UI, agent, memory, research).
- **Modkit** (this repo) = the *mods* the AI controls — the **Forge** (train models incl. Enigma), the **avatar**, device/thing control, + whatever's next. Exposed to Odysseus's agent via an **MCP server** (Odysseus auto-registers MCP servers, so Enigma can call our mods as tools).
- **Enigma** = the model (the brain), forged by Modkit, served to Odysseus.
- **Modkit's own UI** = settings/service only (manage mods, trigger jobs). Not chat.

**Safety:** branch `refocus/modkit`; everything pushed (`dde5c99`). Reversible at every step; `main` untouched until you merge.

---

## ✅ KEEP — Modkit
- **The Forge:** `core/model*`, `model_presets`, `training/`, `bpe_tokenizer`, `tokenizer`, `dataset`, `curated_dataset`, `safe_save`, `weight_mapping`, `lora_utils`, `rl_training`, `adaptive_trainer`, `reward_functions`, `progressive_growing`, `model_merging`, `personality_data`, `hardware_detection`, `monologue`, `nf4_linear`; `forge.py`, `pretokenize_data.py`, `run_training_diagnostic.py`.
- **Fine-tune toolkit:** `train_enigma_lora.py`, `make_enigma_corpus.py`, `make_enigma_local.py`, `eval_enigma.py`, `collect_*.py`.
- **The mod system:** `mods/` (+ `_template/`, `avatar/`), `core/mod_tools.py`, `core/plugin_loader.py`, the mod-discovery + mod-routing core.
- **Enigma + identity:** `models/enigma-8b/`, `data/enigma_voice.md`, `data/personality_corpus.jsonl`.

## ⬆️ UPGRADE — the new work
- An **MCP server** exposing Modkit's mods to Odysseus (Enigma calls them as tools).
- A **`ModBase`** class (kill the per-mod TCP boilerplate).
- **Validation on mod load** (no more silent avatar deaths).
- A minimal **settings/service UI**.
- Rebuild the **avatar** as a mod (3D, rigged, swappable) — later.

## ❌ DELETE — Odysseus owns these
- `enigma_engine/gui/` (~31K lines, tkinter UI) + its tests.
- `enigma_engine/web/` (Svelte).
- `enigma_engine/api/` chat app (`server.py` chat/history/conversations) + `client.py`.
- `enigma_engine/core/` **inference-runtime**: `engine_chat`, `engine_generation`, `inference`, `kv_cache`, `streaming`, the loaders (`gguf*`, `huggingface_loader`, `gptq_awq_loader`, `ollama_loader`, `onnx_loader`), `rag*`, `memory`, `reasoning`, `sentiment`, `auto_research`, `ai_profile`, `model_context`, `builtin_commands`, `commands`, `json_schema_mask`, `vision_encoder`, `audio_encoder`, `document_readers`, `chat_export`, `web_utils`, `download_progress`, `model_registry`, `multi_gpu`, `style_preferences`, `personality_consistency`, `probe_history`, `advanced_tokenizer`, `char_tokenizer`, `tokenizer_metrics`, `model_utils`. *(Odysseus + an external runner run the model.)*
- `run.py` app modes, `run_model_output.py`, `migrate_legacy_lora.py`.
- Tests for deleted modules.
- `router.py`: strip to the mod-routing core (drop the chat/background-train glue), or fold into the MCP server.

## 🔤 RENAME
- Repo/project "Enigma AI Engine" → **Modkit** (`pyproject.toml` name, README, GitHub repo). **Enigma stays the model's name.** Defer the `enigma_engine`→`enigma` package-import rewrite.

## 🔧 EXECUTION ORDER (branch `refocus/modkit`)
1. Delete the UIs (`gui/`, `web/`) + their tests; fix `__init__.py`.
2. Delete the inference-runtime + chat app (verify no kept mod imports them).
3. Slim `router.py`; rename project → Modkit.
4. `pytest` the survivors green; commit per step.
5. **UPGRADE pass** (separate): MCP server, `ModBase`, validation, settings UI.
6. Review the branch diff → merge to `main`.
