# Web Dashboard & Mobile Apps

## Web Dashboard

### Quick Start

```bash
python run.py --web
```

Access at: `http://localhost:8080`

### Features

- **Dashboard**: System status, model info, quick actions
- **Chat**: Real-time chat interface with WebSocket support
- **Train**: Model training interface (starts CLI training)
- **Settings**: Personality traits, voice profiles, instance management

### Pages

#### Dashboard (`/`)
- System status
- Current model
- Running instances
- Quick action buttons

#### Chat (`/chat`)
- Real-time chat with AI
- Streaming responses (if WebSocket available)
- Adjustable parameters (max tokens, temperature)
- Clear chat history

#### Train (`/train`)
- Select model size
- Set training parameters
- Provides CLI command for actual training

#### Settings (`/settings`)
- Select active model
- Adjust personality traits
- Configure voice profile
- View running instances

### API Endpoints

- `GET /api/status` - System status
- `GET /api/models` - List models
- `POST /api/generate` - Generate text
- WebSocket `/` - Real-time chat

### Customization

Templates: `forge_ai/web/templates/`
Styles: `forge_ai/web/static/css/style.css`
Scripts: `forge_ai/web/static/js/app.js`

## Mobile API

### Quick Start

```python
from forge_ai.mobile.api import run_mobile_api

run_mobile_api(host='0.0.0.0', port=5001)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Chat with AI |
| `/api/v1/models` | GET | List models |
| `/api/v1/status` | GET | System status |
| `/api/v1/personality` | GET/PUT | Get/update personality |
| `/api/v1/voice/speak` | POST | Text-to-speech |
| `/api/v1/voice/listen` | POST | Speech-to-text |

### Example Request

```javascript
// Chat request
const response = await fetch('http://YOUR_SERVER:5001/api/v1/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Hello AI!',
    max_length: 100,
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.response);
```

### Response Format

```json
{
  "response": "Hi! How can I help you?",
  "model": "enigma",
  "tokens_used": 10,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Building Mobile Apps

See `mobile/README.md` for detailed guides on:
- React Native (iOS/Android)
- Flutter (iOS/Android)
- Native iOS (Swift)
- Native Android (Kotlin)

### Quick Example (React Native)

```javascript
// ChatScreen.js
import React, { useState } from 'react';
import { View, TextInput, Button, Text } from 'react-native';
import axios from 'axios';

const API_URL = 'http://YOUR_SERVER_IP:5001/api/v1';

export default function ChatScreen() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');

  const sendMessage = async () => {
    const res = await axios.post(`${API_URL}/chat`, {
      message,
      max_length: 100
    });
    setResponse(res.data.response);
  };

  return (
    <View>
      <TextInput 
        value={message}
        onChangeText={setMessage}
        placeholder="Type your message"
      />
      <Button title="Send" onPress={sendMessage} />
      <Text>{response}</Text>
    </View>
  );
}
```

## Network Configuration

### Find Your Server IP

**Windows:**
```cmd
ipconfig
```

**Mac/Linux:**
```bash
ifconfig
# or
ip addr
```

### Allow Firewall Access

**Windows:**
```powershell
netsh advfirewall firewall add rule name="Enigma Web" dir=in action=allow protocol=TCP localport=8080
netsh advfirewall firewall add rule name="Enigma Mobile API" dir=in action=allow protocol=TCP localport=5001
```

**Linux:**
```bash
sudo ufw allow 8080/tcp
sudo ufw allow 5001/tcp
```

## Security Notes

⚠️ **Default setup has no authentication!**

For production:
1. Add API key authentication
2. Use HTTPS (SSL/TLS)
3. Implement rate limiting
4. Validate all inputs
5. Use authentication tokens

### Add Basic API Key

```python
# In forge_ai/web/app.py or forge_ai/mobile/api.py

API_KEY = "your-secret-key"

@app.before_request
def check_api_key():
    api_key = request.headers.get('X-API-Key')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
```

## Advanced: Running Both

```bash
# Terminal 1: Web dashboard
python run.py --web

# Terminal 2: Mobile API
python -c "from forge_ai.mobile.api import run_mobile_api; run_mobile_api()"
```

Access:
- Web: `http://localhost:8080`
- Mobile API: `http://localhost:5001`

## Troubleshooting

### "Connection Refused"
- Check server is running
- Verify IP address and port
- Check firewall settings

### "Model Not Loaded"
- Ensure model exists in `models/` directory
- Train a model first with `python run.py --train`

### "WebSocket Not Available"
- Install flask-socketio: `pip install flask-socketio`
- Restart server

## See Also

- [Multi-Instance Support](MULTI_INSTANCE.md)
- [Personality System](PERSONALITY.md)
- Mobile App Guide: `mobile/README.md`
