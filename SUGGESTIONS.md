# ForgeAI Suggestions

Remaining improvements for the ForgeAI codebase.

**Last Updated:** February 5, 2026

---

## Performance

- [ ] **Lazy imports** for `torch` and `numpy` in core/voice modules (startup speed)
- [x] **Tokenizer caching** in `core/tokenizer.py` - cache instances by type *(Added `_tokenizer_cache` with thread-safe locking)*
- [x] **TTS engine pooling** in `voice/voice_pipeline.py` - reuse engines *(Added `TTSEnginePool` singleton class)*
- [x] **Thread cleanup** - daemon threads in voice modules lack proper `__del__` *(Added `__del__` methods to voice_pipeline.py and voice_chat.py)*
  - `voice/voice_pipeline.py` - `_listen_thread`, `_speak_thread`
  - `voice/voice_chat.py` - `_listen_thread`, `_playback_thread`

---

## Architecture

- [x] **Consolidate progress tracking** - duplicated in `utils/progress.py`, `core/download_progress.py`, GUI dialogs *(Added `to_progress_state()` method to bridge systems)*
- [ ] **Standardize JSON I/O** - use `utils/io_utils.py` consistently across voice/memory modules
- [ ] **Standardize configs** - use `@dataclass` for all configuration objects
- [ ] **Thread safety audit** - review all `threading.Thread` usage for proper locks

---

## Documentation

- [x] **API docs** - Add OpenAPI/Swagger annotations to `web/server.py` *(Added full OpenAPI metadata, tags, and endpoint descriptions)*
- [x] **GraphQL docs** - Document schema in `comms/graphql_api.py` *(Added comprehensive schema documentation in docstring)*
- [x] **REST docs** - Document endpoints in `comms/api_server.py` *(Added detailed endpoint docstrings with examples)*

---

## Quick Wins

- [x] **voice/voice_customizer.py** - 20+ print statements (keep for CLI, add `logger.debug` for state changes) *(Added logger.debug calls for profile operations)*

---

## Testing Gaps

Directories with minimal or no test coverage:

**Critical:**
- [x] `agents/` - No agent tests *(Created tests/test_agents.py)*
- [x] `avatar/` - Limited tests *(Created tests/test_avatar.py)*
- [x] `game/` - Only `test_game_mode.py` *(Created tests/test_game.py)*
- [x] `web/` - Only `test_web_server.py` *(Created tests/test_web.py)*

**High:**
- [x] `gui/tabs/` - No tab-specific tests *(Created tests/test_gui_tabs.py)*
- [x] `gui/dialogs/` - No dialog tests *(Created tests/test_gui_dialogs.py)*
- [x] `plugins/` - Plugin system untested *(Created tests/test_plugins.py)*
- [x] `marketplace/` - Marketplace logic untested *(Created tests/test_marketplace.py)*

**Medium:**
- [x] `cli/`, `collab/`, `companion/`, `deploy/`, `edge/`, `hub/`, `i18n/`, `integrations/`, `monitoring/`, `network/`, `personality/`, `prompts/`, `robotics/`, `mobile/` *(Created tests/test_remaining_modules.py)*
