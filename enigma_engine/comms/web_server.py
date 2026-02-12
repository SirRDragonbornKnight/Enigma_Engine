"""
Web Server for Phone/Browser Connections

Provides:
  - WebSocket for real-time chat
  - REST API endpoints
  - Simple web interface
  - QR code for easy phone connection

Run with: python -m enigma_engine.comms.web_server
"""

import logging
import socket
from datetime import datetime
from typing import Callable, Optional

try:
    from flask import Flask, jsonify, render_template_string, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

from ..config import CONFIG

logger = logging.getLogger(__name__)

# HTML template for web interface
WEB_INTERFACE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enigma AI Engine</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            text-align: center;
            padding: 20px 0;
        }
        header h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
            font-size: 0.9rem;
            color: #888;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #888;
        }
        .status-dot.connected { background: #22c55e; }
        .status-dot.disconnected { background: #ef4444; }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            background: #7c3aed;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .message.ai {
            background: #2d2d44;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .message.system {
            background: transparent;
            color: #888;
            font-size: 0.85rem;
            align-self: center;
            text-align: center;
        }
        .message-time {
            font-size: 0.7rem;
            color: rgba(255,255,255,0.5);
            margin-top: 4px;
        }
        
        .input-container {
            display: flex;
            gap: 12px;
            padding: 20px 0;
        }
        .input-container input {
            flex: 1;
            padding: 14px 20px;
            border: none;
            border-radius: 25px;
            background: #2d2d44;
            color: #fff;
            font-size: 1rem;
            outline: none;
        }
        .input-container input:focus {
            box-shadow: 0 0 0 2px #7c3aed;
        }
        .input-container button {
            padding: 14px 24px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        .input-container button:hover {
            transform: scale(1.05);
        }
        .input-container button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .typing-indicator {
            display: none;
            align-self: flex-start;
            padding: 12px 16px;
            background: #2d2d44;
            border-radius: 16px;
        }
        .typing-indicator.active { display: flex; }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #888;
            border-radius: 50%;
            margin: 0 2px;
            animation: bounce 1.4s infinite;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        
        /* Mobile optimizations */
        @media (max-width: 600px) {
            .container { padding: 10px; }
            header h1 { font-size: 1.5rem; }
            .message { max-width: 90%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>[*] Enigma AI Engine</h1>
            <div class="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Connecting...</span>
            </div>
        </header>
        
        <div class="chat-container" id="chatContainer">
            <div class="message system">Welcome to Enigma AI Engine. Type a message to begin.</div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <span></span><span></span><span></span>
        </div>
        
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message..." autocomplete="off">
            <button id="sendBtn">Send</button>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const typingIndicator = document.getElementById('typingIndicator');
        
        let socket = null;
        let useWebSocket = true;
        
        // Try WebSocket first, fall back to REST API
        function initConnection() {
            try {
                socket = io();
                
                socket.on('connect', () => {
                    statusDot.classList.add('connected');
                    statusDot.classList.remove('disconnected');
                    statusText.textContent = 'Connected (WebSocket)';
                });
                
                socket.on('disconnect', () => {
                    statusDot.classList.remove('connected');
                    statusDot.classList.add('disconnected');
                    statusText.textContent = 'Disconnected';
                });
                
                socket.on('response', (data) => {
                    typingIndicator.classList.remove('active');
                    addMessage(data.text, 'ai');
                });
                
                socket.on('error', (data) => {
                    typingIndicator.classList.remove('active');
                    addMessage('Error: ' + data.message, 'system');
                });
                
            } catch (e) {
                useWebSocket = false;
                statusDot.classList.add('connected');
                statusText.textContent = 'Connected (REST API)';
            }
        }
        
        function addMessage(text, type) {
            const msg = document.createElement('div');
            msg.className = 'message ' + type;
            msg.textContent = text;
            
            const time = document.createElement('div');
            time.className = 'message-time';
            time.textContent = new Date().toLocaleTimeString();
            msg.appendChild(time);
            
            chatContainer.appendChild(msg);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const text = messageInput.value.trim();
            if (!text) return;
            
            addMessage(text, 'user');
            messageInput.value = '';
            sendBtn.disabled = true;
            typingIndicator.classList.add('active');
            
            if (useWebSocket && socket && socket.connected) {
                socket.emit('message', { text: text });
            } else {
                // Fall back to REST API
                try {
                    const response = await fetch('/api/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: text })
                    });
                    const data = await response.json();
                    typingIndicator.classList.remove('active');
                    addMessage(data.response || data.error || 'No response', 'ai');
                } catch (e) {
                    typingIndicator.classList.remove('active');
                    addMessage('Connection error: ' + e.message, 'system');
                }
            }
            
            sendBtn.disabled = false;
        }
        
        sendBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        initConnection();
    </script>
</body>
</html>
'''


class WebServer:
    """
    Web server with WebSocket support for phone/browser connections.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask not available. Install with: pip install flask flask-cors")
        
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        
        # WebSocket support (optional)
        if SOCKETIO_AVAILABLE:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        else:
            self.socketio = None
        
        # AI generate function (set via set_generate_func)
        self._generate_func: Optional[Callable] = None
        
        # Message history
        self.messages: list[dict] = []
        
        # Set up routes
        self._setup_routes()
        if self.socketio:
            self._setup_websocket()
    
    def set_generate_func(self, func: Callable):
        """Set the function to generate AI responses."""
        self._generate_func = func
    
    def _setup_routes(self):
        """Set up HTTP routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(WEB_INTERFACE_HTML)
        
        @self.app.route('/api/generate', methods=['POST'])
        def api_generate():
            data = request.get_json() or {}
            prompt = data.get('prompt', '')
            
            if not prompt:
                return jsonify({'error': 'No prompt provided'}), 400
            
            # Store user message
            self.messages.append({
                'role': 'user',
                'text': prompt,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate response
            response = self._generate(prompt)
            
            # Store AI response
            self.messages.append({
                'role': 'assistant',
                'text': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({'response': response})
        
        @self.app.route('/api/health')
        def api_health():
            return jsonify({
                'status': 'ok',
                'websocket': SOCKETIO_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/history')
        def api_history():
            return jsonify({'messages': self.messages[-50:]})  # Last 50 messages
        
        @self.app.route('/api/info')
        def api_info():
            return jsonify({
                'name': 'Enigma AI Engine',
                'version': CONFIG.get('version', '1.0.0'),
                'capabilities': ['chat', 'websocket', 'api'],
            })
        
        @self.app.route('/qr')
        def qr_code():
            """Generate QR code for easy phone connection."""
            url = f"http://{self._get_local_ip()}:{self.port}"
            
            try:
                import base64
                import io

                import qrcode
                
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(url)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                b64 = base64.b64encode(buffer.getvalue()).decode()
                
                return f'''
                <!DOCTYPE html>
                <html>
                <head><title>Connect to Enigma AI Engine</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 40px; background: #1a1a2e; color: white;">
                    <h1>[!] Connect Your Phone</h1>
                    <p>Scan this QR code with your phone camera:</p>
                    <img src="data:image/png;base64,{b64}" style="margin: 20px; border: 5px solid white;">
                    <p style="font-size: 1.2rem;">Or visit: <a href="{url}" style="color: #00d4ff;">{url}</a></p>
                </body>
                </html>
                '''
            except ImportError:
                return f'''
                <!DOCTYPE html>
                <html>
                <head><title>Connect to Enigma AI Engine</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 40px; background: #1a1a2e; color: white;">
                    <h1>[!] Connect Your Phone</h1>
                    <p>Install qrcode for QR support: pip install qrcode[pil]</p>
                    <p style="font-size: 1.5rem;">Visit: <a href="{url}" style="color: #00d4ff;">{url}</a></p>
                </body>
                </html>
                '''
    
    def _setup_websocket(self):
        """Set up WebSocket handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('connected', {'status': 'ok'})
        
        @self.socketio.on('message')
        def handle_message(data):
            text = data.get('text', '')
            if not text:
                emit('error', {'message': 'Empty message'})
                return
            
            # Store user message
            self.messages.append({
                'role': 'user',
                'text': text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate response
            response = self._generate(text)
            
            # Store AI response
            self.messages.append({
                'role': 'assistant',
                'text': response,
                'timestamp': datetime.now().isoformat()
            })
            
            emit('response', {'text': response})
    
    def _generate(self, prompt: str) -> str:
        """Generate AI response."""
        if self._generate_func:
            try:
                return self._generate_func(prompt)
            except Exception as e:
                return f"Error generating response: {e}"
        else:
            return "AI not configured. Set generate function with set_generate_func()"
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "localhost"
    
    def run(self, debug: bool = False):
        """Start the server."""
        ip = self._get_local_ip()
        logger.info("Enigma AI Engine Web Server starting")
        logger.info("Local:    http://localhost:%d", self.port)
        logger.info("Network:  http://%s:%d", ip, self.port)
        logger.info("QR Code:  http://%s:%d/qr", ip, self.port)
        logger.info("API:      http://%s:%d/api/generate", ip, self.port)
        logger.info("WebSocket: %s", "Enabled" if self.socketio else "Disabled")
        
        if self.socketio:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)
        else:
            self.app.run(host=self.host, port=self.port, debug=debug)


def create_web_server(host: str = '0.0.0.0', port: int = 5000) -> WebServer:
    """Create and configure web server with AI."""
    server = WebServer(host, port)
    
    # Try to connect AI
    try:
        from ..core.inference import generate
        server.set_generate_func(lambda prompt: generate(prompt))
    except Exception as e:
        logger.warning("Could not connect AI: %s", e)
        server.set_generate_func(lambda prompt: f"Echo: {prompt}")
    
    return server


# CLI entry point
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Enigma AI Engine Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    server = create_web_server(args.host, args.port)
    server.run(debug=args.debug)
