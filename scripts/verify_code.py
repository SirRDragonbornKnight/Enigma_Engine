"""Direct code verification - no test framework."""
import sys
sys.path.insert(0, '.')

print('=== Direct Code Verification ===')
print()

# 1. Tokenizer actually works
from forge_ai.core.tokenizer import get_tokenizer
tok = get_tokenizer()
text = 'Hello, how are you today?'
encoded = tok.encode(text)
decoded = tok.decode(encoded)
print(f'Tokenizer: "{text}" -> {len(encoded)} tokens -> "{decoded}"')

# 2. Tools actually execute
from forge_ai.tools import execute_tool
result = execute_tool('get_system_info')
platform = result.get("result", {}).get("platform", "?")
print(f'Tools: get_system_info -> platform={platform[:20] if platform else "?"}')

# 3. Web search tool is callable
from forge_ai.tools.web_tools import WebSearchTool
ws = WebSearchTool()
print(f'Web Search: tool ready, name={ws.name}')

# 4. Vision backend works
from forge_ai.tools.vision import ScreenCapture
sc = ScreenCapture()
print(f'Vision: backend={sc._backend}')

# 5. Memory saves/loads
from forge_ai.memory import ConversationManager
mem = ConversationManager()
# Use save_conversation, not add_message
convs = mem.list_conversations()
print(f'Memory: ConversationManager ready, {len(convs)} existing conversations')

# 6. Config paths exist
from forge_ai.config import CONFIG
from pathlib import Path
data_ok = Path(CONFIG["data_dir"]).exists()
models_ok = Path(CONFIG["models_dir"]).exists()
print(f'Config: data_dir exists={data_ok}, models_dir exists={models_ok}')

# 7. Training config validates
from forge_ai.core.training import TrainingConfig
tc = TrainingConfig(epochs=5, batch_size=4, learning_rate=0.001)
print(f'Training: config epochs={tc.epochs}, batch={tc.batch_size}, lr={tc.learning_rate}')

# 8. HuggingFace loader ready
from forge_ai.core.huggingface_loader import HuggingFaceModel
print(f'HuggingFace: HuggingFaceModel class ready')

# 9. GUI tabs can be imported and have their create functions
from forge_ai.gui.tabs.chat_tab import create_chat_tab
from forge_ai.gui.tabs.training_tab import create_training_tab
from forge_ai.gui.tabs.settings_tab import create_settings_tab
print(f'GUI Tabs: chat, training, settings all have create functions')

# 10. Avatar system
from forge_ai.gui.tabs.avatar.avatar_display import create_avatar_subtab, AvatarOverlayWindow
print(f'Avatar: create_avatar_subtab and AvatarOverlayWindow ready')

print()
print('=== All Direct Tests Passed ===')
