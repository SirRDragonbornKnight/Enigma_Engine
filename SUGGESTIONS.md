# ForgeAI Suggestions

Remaining improvements for the ForgeAI codebase.

**Last Updated:** February 6, 2026

---

## AI Training & Routing Enhancements

### AI Data Trainer (Lightweight)
- [x] **Data Trainer** - Lightweight `forge_ai/tools/data_trainer.py` with:
  - **CharacterTrainer**: Scan for character dialogue, extract personality traits, vocabulary, speech patterns
  - **TaskTrainer**: Generate training data for images, avatar, tools, code (loads examples from JSON)
  - No hardcoded examples - all training data loaded from external JSON files on demand
  - Generate character-specific training datasets automatically
  - Example: "Train AI as Sherlock Holmes" -> find all Sherlock lines -> create fine-tuning dataset

- [x] **Character Trainer Tab** - Add `forge_ai/gui/tabs/character_trainer_tab.py`:
  - UI to input character name/prompt and data source
  - Preview extracted character data before training
  - Train specialized models for specific characters
  - Save character profiles with associated model weights

- [x] **Router AI Trainer** - Extend `forge_ai/core/tool_router.py`:
  - Add ability to train the router itself from within the router
  - Self-improvement: router learns from successful/failed routes
  - Train sub-routers for specific domains (games, code, creative)
  - Recursive training: use trained AIs to generate training data for other AIs

### Unified Prompt System
- [x] **Global prompt manager** - Create `forge_ai/core/prompt_manager.py`:
  - Single source of truth for all AI system prompts
  - Every generation module should use this (image_tab, code_tab, audio_tab, etc.)
  - Currently: persona.py, prompt_templates.py, tool_prompts.py are separate
  - Needed: unified `PromptManager.get_system_prompt(module_name, persona_id)` API

- [x] **Prompt inheritance** - Extend `forge_ai/core/persona.py`:
  - Base prompts that all personas inherit from
  - Module-specific prompt overrides (image AI vs code AI vs chat AI)
  - Safety prompts that are always appended
  - Dynamic prompt injection based on context

- [x] **Prompt validation** - Add prompt validation before model calls:
  - Check prompt length limits
  - Validate variable substitution
  - Warn about conflicting instructions
  - Test prompt effectiveness with sample inputs

### Training Data Tools
- [x] **Data curator** - Add `forge_ai/tools/data_curator.py`:
  - Scan and index all training data files
  - Search by topic, character, style, or sentiment
  - Tag and categorize data automatically
  - Detect duplicate or low-quality entries
  - Merge/split datasets with smart deduplication

- [x] **Character extractor** - Extract characters from datasets:
  - Parse dialogue format: "CHARACTER: dialogue text"
  - Build character vocabulary profiles
  - Track character relationships and interactions
  - Generate character summary cards

- [ ] **Auto-trainer pipeline** - Automated training workflow:
  - Input: character name + data sources
  - Output: fine-tuned model + persona config
  - Steps: extract → filter → validate → train → test → deploy

---

## Recent Updates - Additional Suggestions

### Module System
- [ ] **Module dependency visualization** - Show module dependency graph in GUI
- [ ] **Module conflict detection** - Warn before loading conflicting modules
- [ ] **Module profiles** - Save/load sets of modules as profiles (e.g., "lightweight", "full")

### Networking & Multi-Device
- [ ] **Remote model training** - Train on one device, use on another
- [ ] **Model sync** - Auto-sync model weights across devices
- [ ] **Distributed inference** - Split model across multiple devices

### User Experience
- [ ] **Quick persona switch** - Hotkey to switch between personas
- [ ] **Prompt history** - Save and reuse successful prompts
- [ ] **Training data preview** - Preview what data will be used before training

### Code Generation
- [ ] **Language-specific code AIs** - Specialized models for Python, JS, Rust, etc.
- [ ] **Code style learning** - Learn from user's codebase style
- [ ] **Project context injection** - Include project structure in code prompts

### Avatar System
- [ ] **Avatar emotion recognition** - Avatar reacts to conversation sentiment
- [ ] **Multi-avatar support** - Multiple avatars for different personas
- [ ] **Avatar voice sync** - Lip sync avatar with TTS output

---

## GUI Organization

### Sidebar Reorganization
Current problem: Related features are scattered across different sections (30+ items).

- [ ] **Reorganize sidebar by TASK not type** - Group related features together:
  ```
  CHAT:     Chat, History
  MY AI:    Persona, Training, Learning, Scale  (all "building your AI")
  CREATE:   Image, Code, Video, Audio, 3D, GIF
  CONTROL:  Avatar, Game, Robot, Screen, Camera
  TOOLS:    Modules, Router, Tools, Compare
  SYSTEM:   Terminal, Logs, Files, Settings, Network
  ```

- [ ] **Remove duplicate navigation** - Avatar tab has sub-tabs for Game/Robot, but those are also separate sidebar items. Pick one approach.

- [ ] **Make Training visible** - Currently hidden as sub-tab inside AI tab, should be in main sidebar under "MY AI"

### Training Tab Enhancements
- [x] **Add system prompt editor to Training tab** - Edit the AI's system prompt alongside training:
  - Text area to edit system prompt
  - Preview how prompt affects responses
  - Save prompt with model/persona
  - Currently prompt editing is only in Persona tab - should also be accessible from Training

- [ ] **Unified "Build Your AI" workflow** - Single tab/wizard that combines:
  - System prompt (persona/personality)
  - Training data selection
  - Training controls
  - Testing the result
