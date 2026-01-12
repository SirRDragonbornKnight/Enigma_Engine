# 📱 Enigma Mobile Apps

This directory contains starter templates and guides for building mobile apps that connect to your AI Tester Engine.

## Quick Start

### 1. Start the Mobile API Server

```bash
# From the AI Tester root directory
python -c "from ai_tester.mobile.api import run_mobile_api; run_mobile_api()"
```

This will start the API server on `http://0.0.0.0:5001`

### 2. Connect Your Mobile App

The mobile API provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Chat with AI |
| `/api/v1/models` | GET | List available models |
| `/api/v1/status` | GET | Get system status |
| `/api/v1/personality` | GET/PUT | Get/update personality |
| `/api/v1/voice/speak` | POST | Text-to-speech |
| `/api/v1/voice/listen` | POST | Speech-to-text |

## Example: Chat Request

```javascript
// JavaScript/React Native example
const response = await fetch('http://your-server:5001/api/v1/chat', {
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
console.log(data.response);  // AI's response
```

## Platform-Specific Guides

### React Native (Cross-Platform)

1. **Create New App:**
   ```bash
   npx react-native init EnigmaApp
   cd EnigmaApp
   ```

2. **Install Dependencies:**
   ```bash
   npm install axios
   ```

3. **Configure API URL:**
   ```javascript
   // config.js
   export const API_URL = 'http://YOUR_SERVER_IP:5001/api/v1';
   ```

4. **Basic Chat Component:**
   ```javascript
   import React, { useState } from 'react';
   import { View, TextInput, Button, Text } from 'react-native';
   import axios from 'axios';
   import { API_URL } from './config';

   export default function ChatScreen() {
     const [message, setMessage] = useState('');
     const [response, setResponse] = useState('');

     const sendMessage = async () => {
       try {
         const res = await axios.post(`${API_URL}/chat`, {
           message,
           max_length: 100
         });
         setResponse(res.data.response);
       } catch (error) {
         console.error(error);
       }
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

### Flutter

1. **Create New App:**
   ```bash
   flutter create enigma_app
   cd enigma_app
   ```

2. **Add HTTP Package:**
   ```yaml
   # pubspec.yaml
   dependencies:
     flutter:
       sdk: flutter
     http: ^1.1.0
   ```

3. **Basic Chat Service:**
   ```dart
   import 'dart:convert';
   import 'package:http/http.dart' as http;

   class EnigmaService {
     static const String apiUrl = 'http://YOUR_SERVER_IP:5001/api/v1';

     static Future<String> chat(String message) async {
       final response = await http.post(
         Uri.parse('$apiUrl/chat'),
         headers: {'Content-Type': 'application/json'},
         body: jsonEncode({
           'message': message,
           'max_length': 100,
         }),
       );

       if (response.statusCode == 200) {
         final data = jsonDecode(response.body);
         return data['response'];
       } else {
         throw Exception('Failed to send message');
       }
     }
   }
   ```

### Native iOS (Swift)

1. **Create Network Service:**
   ```swift
   import Foundation

   class EnigmaService {
       static let baseURL = "http://YOUR_SERVER_IP:5001/api/v1"
       
       static func chat(message: String, completion: @escaping (Result<String, Error>) -> Void) {
           guard let url = URL(string: "\(baseURL)/chat") else { return }
           
           var request = URLRequest(url: url)
           request.httpMethod = "POST"
           request.setValue("application/json", forHTTPHeaderField: "Content-Type")
           
           let body: [String: Any] = [
               "message": message,
               "max_length": 100
           ]
           request.httpBody = try? JSONSerialization.data(withJSONObject: body)
           
           URLSession.shared.dataTask(with: request) { data, response, error in
               if let error = error {
                   completion(.failure(error))
                   return
               }
               
               guard let data = data,
                     let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                     let response = json["response"] as? String else {
                   return
               }
               
               completion(.success(response))
           }.resume()
       }
   }
   ```

### Native Android (Kotlin)

1. **Add Dependencies:**
   ```gradle
   // build.gradle
   dependencies {
       implementation 'com.squareup.retrofit2:retrofit:2.9.0'
       implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
   }
   ```

2. **Create API Interface:**
   ```kotlin
   import retrofit2.Call
   import retrofit2.http.Body
   import retrofit2.http.POST

   data class ChatRequest(
       val message: String,
       val max_length: Int = 100
   )

   data class ChatResponse(
       val response: String,
       val model: String,
       val tokens_used: Int
   )

   interface EnigmaApi {
       @POST("chat")
       fun chat(@Body request: ChatRequest): Call<ChatResponse>
   }
   ```

## Network Configuration

### Finding Your Server IP

**On your PC (where Enigma is running):**

- **Windows:** `ipconfig` (look for IPv4 Address)
- **Mac/Linux:** `ifconfig` or `ip addr`

### Firewall Rules

Make sure port 5001 is open:

**Windows:**
```powershell
netsh advfirewall firewall add rule name="Enigma Mobile API" dir=in action=allow protocol=TCP localport=5001
```

**Linux:**
```bash
sudo ufw allow 5001/tcp
```

## Security Considerations

⚠️ **Important:** The mobile API currently has no authentication. For production use, you should:

1. Add API key authentication
2. Use HTTPS (SSL/TLS)
3. Implement rate limiting
4. Add user authentication
5. Validate all inputs

### Adding Basic API Key Authentication

```python
# In enigma/mobile/api.py, add before each route:

API_KEY = "your-secret-key"

@mobile_app.before_request
def check_api_key():
    if request.endpoint == 'mobile_status':
        return  # Allow status check without auth
    
    api_key = request.headers.get('X-API-Key')
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
```

Then in your mobile app:
```javascript
headers: {
  'Content-Type': 'application/json',
  'X-API-Key': 'your-secret-key'
}
```

## Features

### Voice Input/Output

The API supports voice features:

```javascript
// Text-to-Speech
const audio = await fetch(`${API_URL}/voice/speak`, {
  method: 'POST',
  body: JSON.stringify({ text: 'Hello world', voice: 'default' })
});

// Speech-to-Text
const formData = new FormData();
formData.append('audio', audioFile);

const text = await fetch(`${API_URL}/voice/listen`, {
  method: 'POST',
  body: formData
});
```

### Personality Control

```javascript
// Get personality
const personality = await fetch(`${API_URL}/personality`);

// Update personality
await fetch(`${API_URL}/personality`, {
  method: 'PUT',
  body: JSON.stringify({
    traits: {
      humor_level: 0.8,
      formality: 0.3
    }
  })
});
```

## Troubleshooting

### "Connection Refused"
- Check if the API server is running
- Verify the IP address and port
- Check firewall settings

### "Network Error"
- Make sure your phone and PC are on the same network
- Try using the PC's IP address instead of `localhost`

### "Model Not Loaded"
- Make sure you have a trained model
- Check that the model is in the `models/` directory

## Next Steps

1. Build a basic chat interface
2. Add voice input/output
3. Implement personality customization
4. Add offline caching
5. Create a beautiful UI

## Resources

- [React Native Docs](https://reactnative.dev/)
- [Flutter Docs](https://flutter.dev/)
- [iOS Development](https://developer.apple.com/)
- [Android Development](https://developer.android.com/)

## Need Help?

Check the main Enigma documentation or create an issue on GitHub.
