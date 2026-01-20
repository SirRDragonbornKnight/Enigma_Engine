# ForgeAI API Key Authentication

## 🔑 Using ForgeAI Like OpenAI's API

ForgeAI now has API key authentication so you can call it from external applications just like you would with OpenAI's API!

## Quick Start

### 1. Generate an API Key

```bash
# Start the ForgeAI API server
python run.py --serve

# In another terminal, generate a key:
curl -X POST http://localhost:5000/generate_key
```

Response:
```json
{
  "api_key": "sk-forge-randomgeneratedstring123abc",
  "message": "Save this key! Set it as FORGEAI_API_KEY environment variable.",
  "example": "export FORGEAI_API_KEY=sk-forge-randomgeneratedstring123abc"
}
```

### 2. Set the API Key

**Linux/macOS:**
```bash
export FORGEAI_API_KEY="sk-forge-yourkey"
```

**Windows (PowerShell):**
```powershell
$env:FORGEAI_API_KEY="sk-forge-yourkey"
```

**Or in `forge_config.json`:**
```json
{
  "forgeai_api_key": "sk-forge-yourkey",
  "require_api_key": true
}
```

### 3. Use the API

**Python:**
```python
import requests

API_URL = "http://localhost:5000/generate"
API_KEY = "sk-forge-yourkey"

response = requests.post(
    API_URL,
    headers={"X-API-Key": API_KEY},
    json={
        "prompt": "Explain quantum computing",
        "max_gen": 100,
        "temperature": 0.8
    }
)

print(response.json()["text"])
```

**JavaScript/TypeScript:**
```javascript
const response = await fetch('http://localhost:5000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'sk-forge-yourkey'
  },
  body: JSON.stringify({
    prompt: 'Write a poem',
    max_gen: 100
  })
});

const data = await response.json();
console.log(data.text);
```

**cURL:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-forge-yourkey" \
  -d '{"prompt": "Hello AI!", "max_gen": 50}'
```

## Authentication Methods

ForgeAI supports **three ways** to send your API key:

1. **`X-API-Key` header** (recommended):
   ```
   X-API-Key: sk-forge-yourkey
   ```

2. **`Authorization` header** (OpenAI-compatible):
   ```
   Authorization: Bearer sk-forge-yourkey
   ```

3. **Query parameter** (less secure):
   ```
   http://localhost:5000/generate?api_key=sk-forge-yourkey
   ```

## Configuration Options

In `forge_config.json` or environment variables:

```json
{
  "require_api_key": true,        // Require authentication (default: true)
  "forgeai_api_key": "sk-forge-...",  // Your API key
  "api_host": "0.0.0.0",          // Bind to all interfaces
  "api_port": 5000                 // API port
}
```

**Environment variables:**
```bash
export FORGEAI_API_KEY="sk-forge-yourkey"
export FORGE_API_HOST="0.0.0.0"
export FORGE_API_PORT="5000"
```

## Disable Authentication (Development)

For local testing without authentication:

```json
{
  "require_api_key": false
}
```

Or unset the API key:
```bash
unset FORGEAI_API_KEY
```

## Remote Access

To allow remote connections:

```bash
export FORGE_API_HOST="0.0.0.0"  # Listen on all interfaces
export FORGEAI_API_KEY="sk-forge-yourkey"  # Required for security!
```

**⚠️ Security Warning:** Always use an API key when exposing the server to the internet!

## Using in Code Editors (Like Doug Doug)

### VS Code Extension Example

Create a VS Code extension that calls ForgeAI:

```typescript
// extension.ts
import * as vscode from 'vscode';
import axios from 'axios';

const FORGEAI_URL = 'http://localhost:5000/generate';
const API_KEY = process.env.FORGEAI_API_KEY || 'sk-forge-yourkey';

export async function getAICompletion(prompt: string): Promise<string> {
    const response = await axios.post(FORGEAI_URL, {
        prompt: prompt,
        max_gen: 100,
        temperature: 0.7
    }, {
        headers: {
            'X-API-Key': API_KEY
        }
    });
    
    return response.data.text;
}

// Use it in your extension
vscode.commands.registerCommand('extension.askAI', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;
    
    const selection = editor.document.getText(editor.selection);
    const prompt = `Explain this code:\n${selection}`;
    
    const explanation = await getAICompletion(prompt);
    vscode.window.showInformationMessage(explanation);
});
```

### Game Integration Example

Inject ForgeAI into a game:

```python
# game_ai_helper.py
import requests
import pyautogui
import time

FORGEAI_API = "http://localhost:5000/generate"
API_KEY = "sk-forge-yourkey"

def ask_ai(question: str) -> str:
    """Ask ForgeAI a question."""
    response = requests.post(
        FORGEAI_API,
        headers={"X-API-Key": API_KEY},
        json={"prompt": question, "max_gen": 50}
    )
    return response.json()["text"]

def ai_controlled_game():
    """Let AI control the game based on what it sees."""
    while True:
        # AI sees the screen and decides what to do
        decision = ask_ai("I'm playing a game. What should I do next?")
        
        # Parse AI response and execute action
        if "jump" in decision.lower():
            pyautogui.press('space')
        elif "move left" in decision.lower():
            pyautogui.press('left')
        
        time.sleep(0.5)

if __name__ == "__main__":
    ai_controlled_game()
```

## API Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/health` | GET | ❌ No | Health check |
| `/info` | GET | ❌ No | Server info |
| `/generate_key` | POST | ❌ No | Generate new key (only if none set) |
| `/generate` | POST | ✅ Yes | Generate AI response |

## Error Responses

**401 Unauthorized:**
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API key. Set FORGEAI_API_KEY environment variable."
}
```

**403 Forbidden:**
```json
{
  "error": "API key already configured",
  "message": "Unset FORGEAI_API_KEY to generate a new one"
}
```

## Security Best Practices

1. **Never commit API keys to git** - Use environment variables
2. **Use HTTPS in production** - Set up reverse proxy (nginx, caddy)
3. **Rotate keys periodically** - Generate new keys regularly
4. **Limit network access** - Only bind to `127.0.0.1` if local-only
5. **Monitor usage** - Check logs for unauthorized access

## Comparison with OpenAI API

| Feature | OpenAI | ForgeAI |
|---------|--------|---------|
| Authentication | ✅ `Authorization: Bearer` | ✅ `X-API-Key` or `Bearer` |
| Local Execution | ❌ Cloud only | ✅ Runs on your hardware |
| Rate Limits | ✅ Enforced | ❌ None (your hardware is the limit) |
| Cost | 💰 Pay per token | ✅ Free (you own the model) |
| Privacy | ⚠️ Data sent to OpenAI | ✅ Everything stays local |
| API Format | REST JSON | REST JSON |

## Troubleshooting

**"Invalid or missing API key"**
- Make sure `FORGEAI_API_KEY` is set in your environment
- Check that the key matches exactly (no extra spaces)
- Try using a different authentication method

**"Connection refused"**
- Ensure the API server is running (`python run.py --serve`)
- Check the port is correct (default: 5000)
- Verify firewall settings

**"API key already configured"**
- You can only generate a key once for security
- To reset: `unset FORGEAI_API_KEY` then restart the server

## Next Steps

- [API Server Documentation](api_server.md)
- [Remote Access Guide](multi_device_guide.md)
- [Security Guidelines](../SECURITY.md)
