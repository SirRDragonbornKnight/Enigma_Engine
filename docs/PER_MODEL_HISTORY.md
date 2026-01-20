# Per-Model Conversation History

## ✅ Feature Implemented!

Each AI model now has **its own separate conversation history**! This means conversations are isolated per AI, making each model truly independent.

## 📁 New Folder Structure

### Before (Shared History):
```
ForgeAI/
├── data/
│   └── conversations/          ← ALL models shared this
│       ├── chat_today.json
│       └── project_talk.json
└── models/
    ├── artemis/
    └── apollo/
```

### After (Per-Model History):
```
ForgeAI/
└── models/
    ├── artemis/
    │   ├── weights.pth
    │   ├── data/training.txt
    │   └── conversations/      ← artemis's own conversations!
    │       ├── chat_morning.json
    │       └── tech_talk.json
    │
    └── apollo/
        ├── weights.pth
        ├── data/training.txt
        └── conversations/      ← apollo's own conversations!
            ├── creative_writing.json
            └── debug_session.json
```

## 🎯 How It Works

### ConversationManager Now Accepts `model_name`

```python
from forge_ai.memory.manager import ConversationManager

# Per-model storage (recommended)
manager = ConversationManager(model_name="artemis")
manager.save_conversation("my_chat", messages)
# Saves to: models/artemis/conversations/my_chat.json

# Global storage (backward compatible)
manager = ConversationManager(model_name=None)
manager.save_conversation("my_chat", messages)
# Saves to: data/conversations/my_chat.json
```

### Automatic in GUI

When you use the GUI, it **automatically** stores conversations in the current model's folder:

1. **Load a model** (e.g., "artemis")
2. **Chat** with the AI
3. **Save conversation** → Goes to `models/artemis/conversations/`
4. **Switch to another model** (e.g., "apollo")
5. **Chat** with the new AI → Completely separate history!

## 🔄 Migration Guide

### Existing Conversations

Old conversations in `data/conversations/` are still accessible! They're treated as "global" conversations.

To move them to a specific model:

```bash
# Move conversations to artemis
mv data/conversations/*.json models/artemis/conversations/

# Or move to apollo
mv data/conversations/*.json models/apollo/conversations/
```

### Code Updates

If you have custom code using ConversationManager:

**Old code:**
```python
manager = ConversationManager()
```

**New code (model-specific):**
```python
manager = ConversationManager(model_name="your_model_name")
```

**New code (global - backward compatible):**
```python
manager = ConversationManager(model_name=None)  # or just ConversationManager()
```

## 📊 Benefits

### 1. **True Model Isolation**
Each AI has its own memory - no cross-contamination!

```
artemis knows about:
- Your coding projects
- Technical discussions

apollo knows about:
- Creative writing
- Story ideas

They don't share memories!
```

### 2. **Better Organization**
All AI data in one place:

```
models/artemis/
├── config.json          # Architecture
├── weights.pth          # Brain
├── data/
│   └── training.txt     # Training data
└── conversations/       # Memories
    └── *.json
```

### 3. **Easy Backup/Share**
Want to share "artemis" with a friend? Just zip the folder:

```bash
cd models
tar -czf artemis_ai.tar.gz artemis/
# Share artemis_ai.tar.gz - includes conversations!
```

### 4. **Model-Specific Context**
The AI can remember previous conversations with context:

```
User: "Remember what we talked about yesterday?"
artemis: *checks artemis/conversations/* → Finds relevant chat
apollo: *checks apollo/conversations/* → Finds different chat
```

## 🛠️ GUI Features

### Sessions Tab (History Viewer)

The **Sessions** tab now has an **AI selector**:

1. **"All AIs"** - View conversations from all models
2. **"artemis"** - View only artemis's conversations
3. **"apollo"** - View only apollo's conversations

Conversations are labeled: `[artemis] my_chat` or `[apollo] story_ideas`

### Chat Tab

Save/Load buttons automatically use the current model's folder:

- **Current Model**: artemis
- **Save Conversation** → `models/artemis/conversations/`
- **Load Conversation** → Lists from `models/artemis/conversations/`

Switch model → Switch conversation history!

## 🔧 Technical Details

### Storage Path Logic

```python
if model_name:
    # Per-model: models/{model_name}/conversations/
    conv_dir = Path("models") / model_name / "conversations"
else:
    # Global: data/conversations/
    conv_dir = Path("data") / "conversations"
```

### Backward Compatibility

- Old code without `model_name` parameter still works
- Global `data/conversations/` still accessible
- No breaking changes to existing APIs

### Model Creation

When you create a new AI model, the folder structure is automatically created:

```python
registry.create_model("jarvis", size="small")
```

Creates:
```
models/jarvis/
├── config.json
├── metadata.json
├── data/
│   ├── training.txt
│   └── instructions.txt
├── checkpoints/
└── conversations/        ← Ready for history!
```

## 📝 Examples

### Save Conversation to Specific AI

```python
from forge_ai.memory.manager import ConversationManager

messages = [
    {"role": "user", "text": "Hello!", "ts": 1234567890},
    {"role": "ai", "text": "Hi! How can I help?", "ts": 1234567891}
]

# Save to artemis's history
manager = ConversationManager(model_name="artemis")
manager.save_conversation("morning_chat", messages)
# File: models/artemis/conversations/morning_chat.json
```

### Load Conversation from Specific AI

```python
# Load from artemis
manager = ConversationManager(model_name="artemis")
data = manager.load_conversation("morning_chat")
print(data["messages"])
```

### List All Conversations for an AI

```python
manager = ConversationManager(model_name="artemis")
chats = manager.list_conversations()
print(f"artemis has {len(chats)} saved conversations")
# ['morning_chat', 'tech_discussion', 'project_planning']
```

### Compare Two AI Histories

```python
artemis_mgr = ConversationManager(model_name="artemis")
apollo_mgr = ConversationManager(model_name="apollo")

print(f"Artemis chats: {artemis_mgr.list_conversations()}")
print(f"Apollo chats: {apollo_mgr.list_conversations()}")
# Completely different lists!
```

## 🎓 Best Practices

### 1. Name Conversations Descriptively

```python
# Good
manager.save_conversation("python_debugging_2026-01-20", messages)

# Less good
manager.save_conversation("chat1", messages)
```

### 2. Use Different AIs for Different Domains

```python
# Coding AI
coding_ai = ConversationManager(model_name="coder_ai")
coding_ai.save_conversation("fix_bug_123", coding_messages)

# Creative AI
creative_ai = ConversationManager(model_name="writer_ai")
creative_ai.save_conversation("story_chapter_5", creative_messages)
```

### 3. Backup Important Conversations

```bash
# Backup all of artemis's conversations
cp -r models/artemis/conversations/ ~/backups/artemis_conversations_$(date +%Y%m%d)
```

### 4. Clean Up Old Conversations

```python
import os
from pathlib import Path

conv_dir = Path("models/artemis/conversations")
for conv_file in conv_dir.glob("*.json"):
    # Delete conversations older than 30 days
    age_days = (time.time() - conv_file.stat().st_mtime) / 86400
    if age_days > 30:
        print(f"Deleting old conversation: {conv_file.name}")
        conv_file.unlink()
```

## 🐛 Troubleshooting

### "No conversations found"

Make sure you're using the correct model name:

```python
# Check available models
from forge_ai.core.model_registry import ModelRegistry
registry = ModelRegistry()
models = registry.list_models()
print(f"Available models: {models}")

# Use exact model name
manager = ConversationManager(model_name=models[0])
```

### "Permission denied" when saving

Make sure the conversations folder exists and is writable:

```bash
chmod -R u+w models/artemis/conversations/
```

### Conversations not showing in GUI

1. Make sure you're on the correct model in the Sessions tab
2. Click "Refresh" button
3. Check the AI selector dropdown

## 📚 See Also

- [Memory System Documentation](../forge_ai/memory/README.md)
- [Model Registry Guide](MODEL_REGISTRY.md)
- [GUI Guide](GUI_GUIDE.md)
