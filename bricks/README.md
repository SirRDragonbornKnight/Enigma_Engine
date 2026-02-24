# Enigma Engine Bricks

Bricks are modular extensions that add capabilities to the Enigma AI Engine. Each brick is a separate process that connects to the router and handles specific commands.

## File Structure

Each brick folder contains:

```
bricks/mybrick/
├── brick.json      # Config: id, name, commands (EDIT THIS)
├── brick_base.py   # Connection plumbing (DO NOT EDIT)
└── main.py         # Your brick code (EDIT THIS)
```

### What to Edit vs What Not to Edit

| File | Edit? | Contains |
|------|-------|----------|
| `brick.json` | ✅ YES | Your brick's id, name, commands list |
| `main.py` | ✅ YES | Your `create_ui()`, `cmd_generate()`, etc. |
| `brick_base.py` | ❌ NO | TCP connection, protocol, message routing |

### In main.py - What You Override

| Method | Required? | Purpose |
|--------|-----------|---------|
| `create_ui()` | ✅ Yes | Return your QWidget (or None for headless) |
| `cmd_generate()` | ✅ Yes | Your main command logic |
| `cmd_*()` | Optional | Additional commands (match brick.json) |
| `on_theme_changed()` | Optional | Update your UI colors |

### In main.py - What You Use (Don't Override)

| Method | Purpose |
|--------|---------|
| `self.send_update()` | Send progress/results to GUI |
| `self.theme` | Current theme colors (set by GUI) |
| `self.brick_id` | Your brick's ID from brick.json |
| `self.name` | Your brick's name from brick.json |

## How Bricks Work

```
┌─────────────────┐        ┌─────────────────────────────────┐
│   Router        │◄──────►│  Brick (connects TO router)     │
│   (port 9900)   │        │  - Registers capabilities       │
│                 │        │  - Receives commands            │
│   Manages:      │        │  - Sends responses              │
│   - Messages    │        └─────────────────────────────────┘
│   - Routing     │
│   - Training    │        ┌─────────────────────────────────┐
│                 │◄──────►│  Another Brick                  │
└─────────────────┘        └─────────────────────────────────┘
```

### Protocol

Messages use a **4-byte length prefix** (big-endian) + JSON payload:

```
[4 bytes: length][N bytes: JSON data]
```

### Message Types

1. **Registration** (brick → router):
```json
{
    "type": "register",
    "brick_id": "mybrick",
    "name": "My Brick",
    "capabilities": ["command1", "command2"],
    "ui": {
        "layout": "vertical",
        "widgets": [...]
    },
    "prompt": "How to use this brick..."
}
```

**Bricks are self-sufficient** - they send their own UI definition, commands, and prompt on connect. The GUI just renders what the brick sends.

2. **Command** (router → brick):
```json
{
    "id": "cmd-123",
    "type": "command",
    "data": {
        "command": "generate",
        "args": {"prompt": "Hello"}
    }
}
```

3. **Response** (brick → router):
```json
{
    "id": "cmd-123",
    "type": "response",
    "success": true,
    "data": {"result": "..."}
}
```

---

## Creating a New Brick

### Quick Start

1. **Copy the template**:
   ```powershell
   Copy-Item -Recurse "bricks/_template" "bricks/mybrick"
   ```

2. **Edit `brick.json`**:
   ```json
   {
       "name": "My Brick",
       "id": "mybrick",
       "version": "1.0.0",
       "description": "What this brick does",
       "commands": [
           {"name": "mycommand", "description": "Does something"}
       ],
       "ui": {
           "layout": "vertical",
           "widgets": [
               {"type": "text_input", "id": "input", "label": "Input"},
               {"type": "button", "id": "run", "label": "Run", "command": "mycommand"}
           ]
       },
       "prompt": "For mybrick tasks, use: brick.send mybrick mycommand <input>"
   }
   ```
   
   **Required fields:** `name`, `id`, `commands`
   **Optional fields:** `ui` (brick's tab layout), `prompt` (added to AI's knowledge)

3. **Implement commands in `main.py`**:
   ```python
   async def cmd_mycommand(self, args: Dict[str, Any]) -> Dict[str, Any]:
       """Handle mycommand."""
       input_data = args.get("input", "")
       result = do_something(input_data)
       return {"result": result}
   ```

4. **Run your brick**:
   ```powershell
   python -m bricks.mybrick.main
   ```

---

## Reference: Echo Brick

The `bricks/echo/` folder contains a fully documented example brick. Use it as your reference.

### Key Files

| File | Purpose |
|------|---------|
| `brick.json` | Configuration (id, name, commands, settings) |
| `main.py` | Implementation (BrickClient subclass) |

### Echo Brick Commands

- `echo` - Returns the same message back
- `reverse` - Reverses a string
- `count` - Counts characters in text
- `status` - Returns brick status
- `stop` - Stops the brick

### Example: Echo Brick Code Structure

```python
class EchoBrick(BrickClient):
    """Example brick that echoes messages."""
    
    async def cmd_echo(self, args):
        """Echo command - returns message unchanged."""
        message = args.get("message", "")
        return {"message": message}
    
    async def cmd_reverse(self, args):
        """Reverse command - reverses the text."""
        text = args.get("text", "")
        return {"result": text[::-1], "original": text}
```

---

## Available Bricks

| Brick | ID | Description |
|-------|-----|-------------|
| Template | `_template` | Base template for new bricks |
| Echo | `echo` | Example brick with documentation |
| ImageGen | `imagegen` | Image generation (SD WebUI, ComfyUI, diffusers) |

---

## Brick Development Tips

### 1. Add Commands

Commands are methods named `cmd_<name>`:

```python
async def cmd_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = args.get("prompt", "")
    # Do work...
    return {"result": "Generated content"}
```

### 2. Error Handling

The base class handles errors, but you can raise exceptions:

```python
async def cmd_process(self, args):
    if not args.get("input"):
        raise ValueError("Missing required 'input' argument")
    return {"result": process(args["input"])}
```

### 3. Async vs Sync

Commands can be sync or async:

```python
# Async (recommended for I/O operations)
async def cmd_fetch(self, args):
    await some_async_operation()
    return {"done": True}

# Sync (fine for quick operations)
def cmd_calculate(self, args):
    return {"result": args.get("a", 0) + args.get("b", 0)}
```

### 4. Settings from brick.json

Access settings from config:

```python
class MyBrick(BrickClient):
    def __init__(self):
        super().__init__()
        self.model_path = self.config.get("settings", {}).get("model_path")
```

### 5. Logging

Use the logger for debug info:

```python
from logging import getLogger
logger = getLogger(__name__)

async def cmd_process(self, args):
    logger.info(f"Processing: {args}")
    logger.debug("Detailed debug info")
    return {"done": True}
```

---

## Testing Your Brick

### Manual Test

1. Start the router:
   ```python
   from enigma_engine.router import BrickRouter
   router = BrickRouter()
   router.start()
   ```

2. Run your brick:
   ```powershell
   python -m bricks.mybrick.main
   ```

3. Send a command through the router.

### Automated Test

```python
import asyncio
from enigma_engine.router import BrickRouter

async def test_brick():
    # Start router
    router = BrickRouter()
    router.start()
    await asyncio.sleep(0.5)
    
    # Start brick
    # (run in subprocess or separate task)
    
    # Send command
    result = await router.send_command("mybrick", "mycommand", {"arg": "value"})
    assert result["success"]
    
    # Cleanup
    router.stop()
```

---

## Files Structure

```
bricks/
├── README.md           # This file
├── _template/          # Template for new bricks
│   ├── brick.json
│   └── main.py
├── echo/               # Example brick (reference)
│   ├── brick.json
│   └── main.py
└── imagegen/           # Image generation brick
    ├── brick.json
    └── main.py
```

---

## Common Issues

### "Connection refused"
Router isn't running. Start it first:
```python
router = BrickRouter()
router.start()
```

### "Registration failed"
Check that your brick.json has valid `id`, `name`, and `commands`.

### "Unknown command"
Make sure your method is named `cmd_<command>` exactly.
