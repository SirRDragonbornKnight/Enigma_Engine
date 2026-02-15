# Enigma AI Engine - Suggestions


**Last Updated:** February 15, 2026

---

## Dead Code Cleanup Status - COMPLETED

### Summary

| Category | Files Removed | Status |
|----------|---------------|--------|
| **Phase 1 (core/)** | 15 | ✅ Done |
| **Phase 2 - Files** | | |
| - tools/ | 6 | ✅ Done |
| - voice/ | 15 | ✅ Done |
| - memory/ | 20 | ✅ Done |
| - avatar/ | 20 | ✅ Done |
| - comms/ | 15 | ✅ Done |
| **Phase 2 - Packages** | | |
| - docs/ | 4 | ✅ Removed |
| - monitoring/ | 2 | ✅ Removed |
| - robotics/ | 4 | ✅ Removed |
| - hub/ | 2 | ✅ Removed |
| - deploy/ | 4 | ✅ Removed |
| - collab/ | 4 | ✅ Removed |
| - testing/ | 3 | ✅ Removed |
| - scripts/ | 2 | ✅ Removed |
| - training/ | 2 | ✅ Removed |
| - sync/ | 2 | ✅ Removed |
| - prompts/ | 3 | ✅ Removed |
| - data/ | 4 | ✅ Removed |
| - edge/ | 4 | ✅ Removed |
| - personality/ | 2 | ✅ Removed |
| - federated/ | 4 | ✅ Removed |
| - integrations/ | 7 | ✅ Removed |
| **TOTAL** | **~134** | ✅ Verified |

**Original:** 816 Python files  
**Current:** 682 Python files  
**Reduction:** 16.4% of codebase removed  

All tests pass after cleanup (33 passed, 3 skipped).

---

## Documentation Cleanup Needed

Many docs reference removed packages or have outdated info. Review/update:

| File | Issue |
|------|-------|
| `docs/WEB_MOBILE.md` | References removed `mobile/` package imports |
| `docs/MULTI_INSTANCE.md` | References removed `mobile/` package |
| `information/` folder | 66 markdown files - many may be outdated |
| `temp_readme.md` | Temporary file - should be removed |
| `enigma_engine/learning/README.md` | References removed `federated/` |
| `mobile/README.md` | May need update after cleanup |

**Action:** Review these files and either update or remove outdated content.

---

## Dead Code Cleanup - Phase 1 COMPLETED

### Removed Files (15 files) - Feb 15, 2026
- `core/gguf_export.py` - Deprecated re-export
- `core/gguf_exporter.py` - Deprecated re-export
- `tools/battery_manager.py` - Duplicate of utils version
- `core/moe_router.py` - Duplicate of moe.py
- `core/moe_routing.py` - Duplicate of moe.py
- `core/dpo_training.py` - Unused duplicate
- `core/rlhf_training.py` - Unused duplicate
- `core/speculative_decoding.py` - Unused duplicate
- `core/curriculum_learning.py` - Unused duplicate
- `core/kv_compression.py` - Unused duplicate
- `core/kv_cache_compression.py` - Unused duplicate
- `core/kv_cache_quantization.py` - Unused duplicate
- `core/prompts.py` - Unused duplicate
- `core/prompt_manager.py` - Unused duplicate
- `core/prompt_templates.py` - Unused duplicate

---

## Verification Results - PASSED

All imports verified working:
- `enigma_engine.core` - OK
- `enigma_engine.tools` - OK
- `enigma_engine.modules` - OK
- `enigma_engine.utils` - OK

All 23 model tests passed.

---

## Future Features (Never Imported But Complete)

These files are complete implementations but currently not integrated. 
They may be future features or optional modules - **DO NOT DELETE** without review.

### Core AI Features
| File | Description | Status |
|------|-------------|--------|
| `core/gesture_manager.py` | Gesture detection system | Future |
| `core/ssm.py` | Mamba/S4 state space model | Future |
| `core/tree_attention.py` | Tree-based attention | Future |
| `core/infinite_context.py` | Streaming context extension | Future |
| `core/context_extender.py` | RoPE scaling extension | Future |
| `core/context_extension.py` | Context window extension | Future |
| `core/paged_attention.py` | Paged KV cache attention | Used by continuous_batching |
| `core/dpo.py` | Direct Preference Optimization | Future |
| `core/rlhf.py` | RLHF training | Future |
| `core/speculative.py` | Speculative decoding | Future |
| `core/curriculum.py` | Curriculum learning | Future |
| `core/api_key_manager.py` | API key management | Future |

### Batching Systems (Interconnected)
| File | Description | Status |
|------|-------------|--------|
| `core/batch_inference.py` | Batch processing | Future |
| `core/continuous_batching.py` | Continuous batching server | Uses paged_attention |
| `core/dynamic_batching.py` | Dynamic batch sizing | Future |

### Robotics Package (Complete)
| File | Description | Status |
|------|-------------|--------|
| `robotics/__init__.py` | Package exports | Standalone |
| `robotics/ros_integration.py` | ROS bridge | Future |
| `robotics/slam.py` | SLAM implementation | Future |
| `robotics/manipulation.py` | Robot arm kinematics | Future |

### Tools (Game/Sensor)
| File | Description | Status |
|------|-------------|--------|
| `tools/sensor_fusion.py` | Multi-sensor fusion | Future |
| `tools/achievement_tracker.py` | Game achievements | Future |
| `tools/multiplayer_awareness.py` | Multiplayer game AI | Future |
| `tools/replay_analysis.py` | Game replay analysis | Future |
| `tools/goal_tracker.py` | Goal tracking system | Future |

### Utils
| File | Description | Status |
|------|-------------|--------|
| `utils/result.py` | Rust-style Result type | Future |
| `utils/api_keys.py` | API key utilities | Future |
| `learning/ab_testing.py` | A/B testing for AI | Future |

---

## Active & Used Files

These files ARE used and should NOT be removed:

| File | Used By |
|------|---------|
| `core/meta_learning.py` | trainer_ai.py (5 imports) |
| `core/prompt_builder.py` | game_router.py, tests |
| `core/moe.py` | test_moe.py |
| `utils/battery_manager.py` | __init__.py, integration.py |
| `utils/api_key_encryption.py` | build_ai_tab.py, trainer_ai.py |
| `utils/starter_kits.py` | quick_create.py dialog |
| `federated/*` | enhanced_window.py |

---

## Statistics

| Metric | Count |
|--------|-------|
| Files Removed | 15 |
| Lines Removed | ~8,000 |
| Tests Passing | 23/23 |
| Import Errors | 0 |
| Future Feature Files | ~25 |

---

## Pending Items

*No pending suggestions at this time.*

---

## Full Codebase Review - Completed Feb 15, 2026

### Package Summary

| Package | Files | Status | Notes |
|---------|-------|--------|-------|
| `core/` | ~170 | Core | Main AI engine |
| `gui/tabs/` | 44 | Core | GUI tabs |
| `tools/` | ~70 | Core | AI tools |
| `utils/` | ~80 | Core | Helpers |
| `memory/` | 39 | Core | Memory storage |
| `voice/` | 44 | Core | TTS/STT |
| `avatar/` | ~55 | Core | Avatar control |
| `comms/` | 32 | Core | API/networking |
| `modules/` | 7 | Core | Module system |
| `federated/` | 8 | Used | Federated learning (GUI) |
| `learning/` | 17 | Used | Learning system (tests) |
| `agents/` | 12 | Tests only | Multi-agent system |
| `plugins/` | 5 | Core | Plugin system |
| `marketplace/` | 4 | Tests | Plugin marketplace |
| `integrations/` | 7 | Future | External integrations |
| `network/` | 6 | Minimal use | Network offloading |
| `web/` | ~12 | Core | Web interface |
| `auth/` | 1 | Minimal | Authentication |
| `builtin/` | 12 | Core | Fallback generators |
| `cli/` | 4 | Core | Command line |
| `collab/` | 4 | Future | Collaboration |
| `companion/` | 2 | Future | Companion mode |
| `config/` | 4 | Core | Configuration |
| `data/` | 4 | Core | Data handling |
| `deploy/` | 4 | Future | Deployment tools |
| `docs/` | 4 | Unused | Doc generators |
| `edge/` | 4 | Tests only | Edge device support |
| `game/` | 6 | Core | Game overlay |
| `hub/` | 2 | Future | Model hub |
| `i18n/` | 2 | Core | Translations |
| `mobile/` | 2 | Future | Mobile API |
| `monitoring/` | 2 | Unused | Prometheus metrics |
| `personality/` | 2 | Tests only | Curiosity system |
| `prompts/` | 3 | Future | Prompt library |
| `robotics/` | 4 | Future | ROS/SLAM |
| `scripts/` | 2 | Internal | Analysis scripts |
| `security/` | 6 | Tests only | Security features |
| `self_improvement/` | 7 | Used | Self-training |
| `sync/` | 2 | Future | Cloud sync |
| `testing/` | 3 | Internal | Test utilities |
| `training/` | 2 | Future | Training generators |
| `vocab_model/` | N/A | Data | Vocabulary files |

---

## Architecture Observations

### Duplicate Implementation: federated/ vs learning/

**Issue:** Two parallel federated learning implementations exist:

| Feature | `federated/` | `learning/` |
|---------|-------------|-------------|
| FederatedLearning | ✓ | ✓ |
| DifferentialPrivacy | ✓ | ✓ |
| FederatedCoordinator | ✓ | ✓ |
| Aggregation | ✓ | ✓ |
| **Used by GUI** | ✓ | ✗ |
| **Used by tests** | ✗ | ✓ |

**Recommendation:** Consolidate to one package. `learning/` re-exports are used by tests. Consider making `learning/__init__.py` import from `federated/` instead.

### Unused Packages (Safe to Remove Later)

| Package | Files | Notes |
|---------|-------|-------|
| `docs/` | 4 | Doc generators - no imports |
| `monitoring/` | 2 | Prometheus - no imports |

### Packages Used Only in Tests

| Package | Notes |
|---------|-------|
| `edge/` | Edge device GPIO/camera |
| `personality/` | Curiosity system |
| `security/` | Auth/GDPR features |
| `agents/` | Multi-agent system |

---

## Statistics (Final)

| Category | Count |
|----------|-------|
| Total packages | ~45 |
| Core packages | ~20 |
| Future/unused packages | ~10 |
| Test-only packages | ~5 |
| **Python files** | **816** |
| **Total lines** | **458,255** |
| Files removed | 15 |
| Lines removed | ~8,000 |