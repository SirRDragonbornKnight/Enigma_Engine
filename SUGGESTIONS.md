# ForgeAI Suggestions

Remaining improvements for the ForgeAI codebase.

**Last Updated:** February 5, 2026

---

## Performance

- [ ] **Lazy imports** for `torch` and `numpy` in core/voice modules (startup speed)
- [ ] **Tokenizer caching** in `core/tokenizer.py` - cache instances by type
- [ ] **TTS engine pooling** in `voice/voice_pipeline.py` - reuse engines
- [ ] **Thread cleanup** - daemon threads in voice modules lack proper `__del__`
  - `voice/voice_pipeline.py` - `_listen_thread`, `_speak_thread`
  - `voice/voice_chat.py` - `_listen_thread`, `_playback_thread`

---

## Architecture

- [ ] **Consolidate progress tracking** - duplicated in `utils/progress.py`, `core/download_progress.py`, GUI dialogs
- [ ] **Standardize JSON I/O** - use `utils/io_utils.py` consistently across voice/memory modules
- [ ] **Standardize configs** - use `@dataclass` for all configuration objects
- [ ] **Thread safety audit** - review all `threading.Thread` usage for proper locks

---

## Documentation

- [ ] **API docs** - Add OpenAPI/Swagger annotations to `web/server.py`
- [ ] **GraphQL docs** - Document schema in `comms/graphql_api.py`
- [ ] **REST docs** - Document endpoints in `comms/api_server.py`

---

## Quick Wins

- [ ] **voice/voice_customizer.py** - 20+ print statements (keep for CLI, add `logger.debug` for state changes)

---

## Testing Gaps

Directories with minimal or no test coverage:

**Critical:**
- [ ] `agents/` - No agent tests
- [ ] `avatar/` - Limited tests
- [ ] `game/` - Only `test_game_mode.py`
- [ ] `web/` - Only `test_web_server.py`

**High:**
- [ ] `gui/tabs/` - No tab-specific tests
- [ ] `gui/dialogs/` - No dialog tests
- [ ] `plugins/` - Plugin system untested
- [ ] `marketplace/` - Marketplace logic untested

**Medium:**
- [ ] `cli/`, `collab/`, `companion/`, `deploy/`, `edge/`, `hub/`, `i18n/`, `integrations/`, `monitoring/`, `network/`, `personality/`, `prompts/`, `robotics/`, `mobile/`
