# Avatar Background Processing System

## How It Works

The AI processes **two layers simultaneously** when controlling the avatar:

### 1. Foreground (User Sees) 🔵
- Natural conversation text: "Hello! 👋"
- Emojis and expressions
- Clean, readable responses
- **TTS reads ONLY this** - no technical jargon

### 2. Background (Invisible) 🔴
- Tool function calls execute silently
- Parameter passing to avatar controller
- Bone manipulation happens automatically
- Error handling and logging
- Result validation

---

## Comparison: Old vs New

### ❌ OLD APPROACH (Tag-Based)

**User says:** "wave hello"

**AI responds:**
```
Hello! <bone_control>right_arm|pitch=45,yaw=30,roll=0</bone_control>
```

**Problems:**
- User sees ugly tags in chat window
- TTS reads: "Hello bone control right arm pitch equals forty five yaw equals thirty..."
- Hard to read and understand
- Cluttered, unprofessional interface
- Tags might confuse users

### ✅ NEW APPROACH (Tool-Based)

**User says:** "wave hello"

**AI responds (visible):**
```
Hello! 👋
```

**Background processing (invisible):**
```python
control_avatar_bones(action="gesture", gesture_name="wave")
# Avatar waves silently while AI speaks naturally
```

**Benefits:**
- User sees clean text: "Hello! 👋"
- TTS reads naturally: "Hello!"
- Avatar waves simultaneously
- Professional user experience
- Technical details hidden from user

---

## Training Data Format

### Tool-Based Training (Recommended)

```
Q: wave hello
A: Hello! 👋
<tool_call>{"tool": "control_avatar_bones", "params": {"action": "gesture", "gesture_name": "wave"}}</tool_call>
<tool_result>{"tool": "control_avatar_bones", "success": true, "result": "Waved hand"}</tool_result>
```

**What happens:**
1. AI learns the pattern: "wave hello" → call tool + say "Hello!"
2. During inference, AI outputs both text and tool call
3. Enigma AI Engine separates them:
   - Text → Chat window + TTS
   - Tool call → Background execution
4. User experiences natural conversation with avatar gestures

### Why This Is Better

| Aspect | Tag-Based | Tool-Based |
|--------|-----------|------------|
| **Chat Display** | Tags visible ❌ | Clean text ✅ |
| **TTS Output** | Reads tags ❌ | Natural speech ✅ |
| **User Experience** | Technical ❌ | Professional ✅ |
| **Readability** | Cluttered ❌ | Clear ✅ |
| **Error Handling** | Text-based ❌ | Function-based ✅ |
| **Debugging** | Hard ❌ | Easy (logs) ✅ |

---

## How Enigma AI Engine Processes It

### Inference Flow

```python
# User: "wave hello"
# AI generates:
response = "Hello! 👋\n<tool_call>{...}</tool_call>"

# Enigma AI Engine splits processing:

# 1. Extract visible text
visible_text = "Hello! 👋"  # → Show in chat
speak(visible_text)          # → TTS reads this

# 2. Extract tool calls (background)
tool_calls = parse_tool_calls(response)  # Hidden from user
for tool in tool_calls:
    execute_silently(tool)  # Avatar waves
    log_result(tool)        # For debugging
```

### What User Experiences

1. **Types:** "wave hello"
2. **Sees:** AI response: "Hello! 👋" (clean text)
3. **Hears:** TTS says "Hello!" (natural voice)
4. **Watches:** Avatar waves hand smoothly
5. **Thinks:** "Wow, that's natural!"

**User never sees:**
- `<tool_call>` tags
- JSON parameters
- Function names
- Technical execution details
- Error logs (unless critical)

---

## Testing the System

### Run the Visualizer

```bash
python test_avatar_tool_system.py
```

**Shows:**
- What user sees vs what happens in background
- Side-by-side comparison of old vs new
- Live tool execution test
- Training data visualization

### Test in Live Chat

1. Train model with tool-based data:
   ```bash
   python scripts/train_avatar_control.py --data data/specialized/avatar_control_training_tool_based.txt
   ```

2. Run GUI:
   ```bash
   python run.py --gui
   ```

3. Say: "wave hello"

4. Observe:
   - Chat shows: "Hello! 👋"
   - TTS says: "Hello!"
   - Avatar waves
   - **No tags visible anywhere**

---

## Files & Locations

### Training Data
- **Tool-based**: `data/specialized/avatar_control_training_tool_based.txt` ✅ **Use this**
- **Tag-based** (old): `data/specialized/avatar_control_training.txt` (kept for reference)

### Implementation
- **Tool Executor**: `enigma_engine/tools/tool_executor.py` - Executes tools silently
- **Tool Definitions**: `enigma_engine/tools/tool_definitions.py` - Defines `control_avatar_bones` tool
- **Avatar Control**: `enigma_engine/avatar/bone_control.py` - Actual bone manipulation
- **AI Integration**: `enigma_engine/avatar/ai_control.py` - Connects AI to avatar

### Testing
- **Visualizer**: `test_avatar_tool_system.py` - Shows background processing
- **Training Script**: `scripts/train_avatar_control.py` - Train specialized model

---

## Background Processing Benefits

### For Users
- **Clean interface** - No technical clutter
- **Natural conversation** - TTS sounds human
- **Smooth experience** - Avatar syncs with speech
- **Professional** - Looks like commercial software

### For Developers
- **Easy debugging** - Tool calls are structured JSON
- **Error handling** - Function-based try/catch
- **Logging** - All tool calls logged automatically
- **Extensible** - Easy to add new gestures/actions
- **Testable** - Can test tools independently

### For AI Training
- **Clear format** - JSON is unambiguous
- **Consistent** - All tools use same structure
- **Validated** - Parameters type-checked automatically
- **Scalable** - Can add 100+ tools without confusion

---

## Advanced: What AI "Sees" vs What Users See

### AI's Internal Processing

```
Input: "wave hello"

AI thinks:
  1. User wants greeting gesture
  2. Need to: say hello + wave
  3. Output TWO things:
     - Text: "Hello! 👋"
     - Tool: control_avatar_bones(gesture="wave")

Output format (internal):
  A: Hello! 👋
  <tool_call>{"tool": "control_avatar_bones", "params": {"action": "gesture", "gesture_name": "wave"}}</tool_call>
  <tool_result>{"tool": "control_avatar_bones", "success": true}</tool_result>
```

### User's Experience

```
Input: "wave hello"

User sees:
  💬 Hello! 👋

User hears:
  🔊 "Hello!"

User watches:
  👋 Avatar waves hand

User thinks:
  "That felt natural and smooth!"
```

**The magic:** Background processing makes it seamless!

---

## Summary

The tool-based approach **solves the UX problem** you identified:

✅ **No visible tags** in chat  
✅ **TTS reads clean text** only  
✅ **Avatar syncs naturally** with conversation  
✅ **Professional appearance**  
✅ **Easy to maintain** and debug  

Train your model with `avatar_control_training_tool_based.txt` for the best results!
