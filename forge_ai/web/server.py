"""
ForgeAI Web Server - FastAPI Implementation

Modern async web server with WebSocket support for real-time chat.
Provides full REST API and mobile-responsive interface.
"""

import logging
from pathlib import Path
from typing import Optional, List
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import get_auth, WebAuth
from .discovery import LocalDiscovery, get_local_ip

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """Chat message request."""
    content: str
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.8


class GenerateImageRequest(BaseModel):
    """Image generation request."""
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = 20


class GenerateCodeRequest(BaseModel):
    """Code generation request."""
    prompt: str
    language: Optional[str] = "python"


class GenerateAudioRequest(BaseModel):
    """Audio generation request."""
    text: str
    voice: Optional[str] = "default"


class SettingsUpdate(BaseModel):
    """Settings update request."""
    settings: dict


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and store a new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client."""
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")


manager = ConnectionManager()


# =============================================================================
# ForgeAI Web Server
# =============================================================================

class ForgeWebServer:
    """
    Self-hosted web server for ForgeAI.
    
    Built with FastAPI for modern async web framework.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, require_auth: bool = True):
        """
        Initialize web server.
        
        Args:
            host: Host to bind to (0.0.0.0 = all interfaces)
            port: Port to listen on
            require_auth: Whether to require authentication
        """
        self.app = FastAPI(title="ForgeAI Web", version="1.0.0")
        self.host = host
        self.port = port
        self.require_auth = require_auth
        self.discovery = LocalDiscovery()
        
        # Get paths
        self.web_dir = Path(__file__).parent
        self.static_dir = self.web_dir / "static"
        self.static_dir.mkdir(exist_ok=True)
        
        # Setup CORS
        self._setup_cors()
        
        # Register routes
        self._register_routes()
        
        # Mount static files
        if self.static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
    
    def _setup_cors(self):
        """Configure CORS with configurable origins."""
        # Get CORS origins from config
        try:
            from ..config import CONFIG
            
            # Check for CORS configuration in config
            if hasattr(CONFIG, 'CORS_ORIGINS') and CONFIG.CORS_ORIGINS:
                origins = CONFIG.CORS_ORIGINS
            elif hasattr(CONFIG, 'WEB_CORS_ORIGINS') and CONFIG.WEB_CORS_ORIGINS:
                origins = CONFIG.WEB_CORS_ORIGINS
            else:
                # Try to load from environment variable
                import os
                env_origins = os.environ.get('FORGE_CORS_ORIGINS', '')
                if env_origins:
                    origins = [o.strip() for o in env_origins.split(',') if o.strip()]
                else:
                    # Try to load from settings file
                    settings_file = CONFIG.DATA_DIR / "web_settings.json"
                    if settings_file.exists():
                        import json
                        with open(settings_file) as f:
                            settings = json.load(f)
                            origins = settings.get('cors_origins', ["*"])
                    else:
                        # Default: allow all in development
                        origins = ["*"]
        except Exception:
            # Fallback to allow all
            origins = ["*"]
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _get_auth(self) -> WebAuth:
        """Get auth instance."""
        return get_auth(config=True)
    
    async def _verify_token(self, token: Optional[str] = Query(None)) -> bool:
        """Verify authentication token."""
        if not self.require_auth:
            return True
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required"
            )
        
        auth = self._get_auth()
        if not auth.verify_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return True
    
    def _register_routes(self):
        """Register all web endpoints."""
        
        @self.app.get("/")
        async def index():
            """Serve main web interface."""
            html_file = self.static_dir / "index.html"
            if html_file.exists():
                return FileResponse(html_file)
            return HTMLResponse(content=self._get_default_html())
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok", "service": "ForgeAI Web"}
        
        @self.app.get("/api/info")
        async def info():
            """Get server information."""
            return {
                "name": "ForgeAI Web Server",
                "version": "1.0.0",
                "host": get_local_ip(),
                "port": self.port,
                "auth_required": self.require_auth
            }
        
        @self.app.post("/api/chat")
        async def chat(message: ChatMessage, authenticated: bool = Depends(self._verify_token)):
            """Chat endpoint - send message, get response."""
            try:
                # Get inference engine
                response_text = await self._generate_response(message.content)
                
                return {
                    "success": True,
                    "response": response_text,
                    "conversation_id": message.conversation_id or "default"
                }
            except Exception as e:
                logger.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/conversations")
        async def list_conversations(authenticated: bool = Depends(self._verify_token)):
            """List all conversations."""
            try:
                from ..memory.manager import ConversationManager
                conv_manager = ConversationManager()
                
                conversations = []
                for conv_file in conv_manager.conv_dir.glob('*.json'):
                    import json
                    try:
                        data = json.loads(conv_file.read_text())
                        conversations.append({
                            'id': conv_file.stem,
                            'name': data.get('name', conv_file.stem),
                            'message_count': len(data.get('messages', []))
                        })
                    except Exception:
                        pass
                
                return {"conversations": conversations}
            except Exception as e:
                logger.error(f"Error listing conversations: {e}")
                return {"conversations": []}
        
        @self.app.get("/api/conversations/{conv_id}")
        async def get_conversation(conv_id: str, authenticated: bool = Depends(self._verify_token)):
            """Get specific conversation."""
            try:
                from ..memory.manager import ConversationManager
                conv_manager = ConversationManager()
                messages = conv_manager.load_conversation(conv_id)
                return {"id": conv_id, "messages": messages}
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Conversation not found: {e}")
        
        @self.app.delete("/api/conversations/{conv_id}")
        async def delete_conversation(conv_id: str, authenticated: bool = Depends(self._verify_token)):
            """Delete conversation."""
            try:
                from ..memory.manager import ConversationManager
                conv_manager = ConversationManager()
                conv_file = conv_manager.conv_dir / f"{conv_id}.json"
                if conv_file.exists():
                    conv_file.unlink()
                return {"success": True}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/generate/image")
        async def generate_image(request: GenerateImageRequest, authenticated: bool = Depends(self._verify_token)):
            """Generate image."""
            return {
                "success": True,
                "message": "Image generation requires image_gen module to be loaded",
                "prompt": request.prompt
            }
        
        @self.app.post("/api/generate/code")
        async def generate_code(request: GenerateCodeRequest, authenticated: bool = Depends(self._verify_token)):
            """Generate code."""
            return {
                "success": True,
                "message": "Code generation requires code_gen module to be loaded",
                "prompt": request.prompt
            }
        
        @self.app.post("/api/generate/audio")
        async def generate_audio(request: GenerateAudioRequest, authenticated: bool = Depends(self._verify_token)):
            """Generate audio."""
            return {
                "success": True,
                "message": "Audio generation requires audio_gen module to be loaded",
                "text": request.text
            }
        
        @self.app.get("/api/settings")
        async def get_settings(authenticated: bool = Depends(self._verify_token)):
            """Get current settings."""
            from ..config import CONFIG
            return {
                "settings": {
                    "temperature": CONFIG.get("temperature", 0.8),
                    "max_gen": CONFIG.get("max_gen", 100),
                    "model": CONFIG.get("default_model", "forge_ai")
                }
            }
        
        @self.app.put("/api/settings")
        async def update_settings(update: SettingsUpdate, authenticated: bool = Depends(self._verify_token)):
            """Update settings."""
            from ..config import update_config
            update_config(update.settings)
            return {"success": True, "settings": update.settings}
        
        @self.app.get("/api/models")
        async def list_models(authenticated: bool = Depends(self._verify_token)):
            """List available models."""
            from ..config import CONFIG
            models_dir = Path(CONFIG["models_dir"])
            
            models = []
            if models_dir.exists():
                for model_path in models_dir.glob("*"):
                    if model_path.is_dir():
                        models.append({
                            "name": model_path.name,
                            "path": str(model_path)
                        })
            
            return {"models": models}
        
        @self.app.post("/api/models/switch")
        async def switch_model(model_name: str, authenticated: bool = Depends(self._verify_token)):
            """Switch active model."""
            from ..config import update_config
            update_config({"default_model": model_name})
            return {"success": True, "model": model_name}
        
        @self.app.get("/api/stats")
        async def get_stats(authenticated: bool = Depends(self._verify_token)):
            """Get system statistics."""
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        
        @self.app.get("/api/modules")
        async def list_modules(authenticated: bool = Depends(self._verify_token)):
            """List available modules."""
            try:
                from ..modules.manager import ModuleManager
                manager = ModuleManager()
                
                modules = []
                for mod_id, mod_info in manager.registry.modules.items():
                    state = manager.get_state(mod_id)
                    modules.append({
                        "id": mod_id,
                        "name": mod_info.name,
                        "loaded": state.loaded if state else False
                    })
                
                return {"modules": modules}
            except Exception as e:
                logger.error(f"Error listing modules: {e}")
                return {"modules": []}
        
        @self.app.post("/api/modules/{module_id}/toggle")
        async def toggle_module(module_id: str, authenticated: bool = Depends(self._verify_token)):
            """Enable/disable module."""
            try:
                from ..modules.manager import ModuleManager
                manager = ModuleManager()
                
                state = manager.get_state(module_id)
                if state and state.loaded:
                    manager.unload(module_id)
                    return {"success": True, "loaded": False}
                else:
                    manager.load(module_id)
                    return {"success": True, "loaded": True}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/qr")
        async def qr_code():
            """Generate QR code for easy connection."""
            auth = self._get_auth()
            token = auth.get_master_token()
            url = f"http://{get_local_ip()}:{self.port}?token={token}"
            
            try:
                import qrcode
                import io
                import base64
                
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(url)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                b64 = base64.b64encode(buffer.getvalue()).decode()
                
                html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Connect to ForgeAI</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                            text-align: center;
                            padding: 40px;
                            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            color: white;
                            min-height: 100vh;
                            margin: 0;
                        }}
                        h1 {{ color: #00d4ff; margin-bottom: 20px; }}
                        img {{ margin: 20px; border: 5px solid white; border-radius: 10px; }}
                        .url {{ font-size: 1.2rem; margin: 20px; word-break: break-all; }}
                        a {{ color: #00d4ff; text-decoration: none; }}
                    </style>
                </head>
                <body>
                    <h1>Connect Your Device</h1>
                    <p>Scan this QR code with your phone camera:</p>
                    <img src="data:image/png;base64,{b64}" alt="QR Code">
                    <div class="url">Or visit: <a href="{url}">{url}</a></div>
                </body>
                </html>
                '''
                return HTMLResponse(content=html)
            except ImportError:
                return HTMLResponse(content=f'''
                <!DOCTYPE html>
                <html>
                <head><title>Connect to ForgeAI</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 40px;">
                    <h1>Connect Your Device</h1>
                    <p>Install qrcode for QR support: pip install qrcode[pil]</p>
                    <p style="font-size: 1.5rem;">Visit: <a href="{url}">{url}</a></p>
                </body>
                </html>
                ''')
        
        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket, token: Optional[str] = Query(None)):
            """WebSocket endpoint for real-time chat."""
            # Verify authentication for WebSocket
            if self.require_auth:
                auth = self._get_auth()
                if not token or not auth.verify_token(token):
                    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                    return
            
            await manager.connect(websocket)
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    message_type = data.get("type", "message")
                    
                    if message_type == "message":
                        content = data.get("content", "")
                        
                        # Send typing indicator
                        await manager.send_personal_message(
                            {"type": "typing", "typing": True},
                            websocket
                        )
                        
                        # Generate response
                        response = await self._generate_response(content)
                        
                        # Send response
                        await manager.send_personal_message(
                            {
                                "type": "response",
                                "content": response,
                                "timestamp": str(asyncio.get_event_loop().time())
                            },
                            websocket
                        )
                    
                    elif message_type == "ping":
                        await manager.send_personal_message({"type": "pong"}, websocket)
            
            except WebSocketDisconnect:
                manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                manager.disconnect(websocket)
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate AI response."""
        try:
            from ..core.inference import ForgeEngine
            engine = ForgeEngine()
            response = engine.generate(prompt, max_gen=200)
            return response
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _get_default_html(self) -> str:
        """Get default HTML if index.html doesn't exist."""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>ForgeAI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>ForgeAI Web Interface</h1>
            <p>The static files are not yet set up. Please create index.html in the static directory.</p>
        </body>
        </html>
        '''
    
    def start(self, debug: bool = False):
        """Start the web server."""
        import uvicorn
        
        # Start local network discovery
        self.discovery.advertise(self.port)
        
        # Print connection info
        local_ip = get_local_ip()
        auth = self._get_auth()
        token = auth.get_master_token()
        
        print("\n" + "="*60)
        print("FORGEAI WEB SERVER")
        print("="*60)
        print(f"\nServer running on:")
        print(f"  Local:   http://localhost:{self.port}")
        print(f"  Network: http://{local_ip}:{self.port}")
        print(f"\nQR Code for mobile: http://{local_ip}:{self.port}/qr")
        
        if self.require_auth:
            print(f"\nAuthentication Token: {token}")
            print(f"Connection URL: http://{local_ip}:{self.port}?token={token}")
        
        print("\nAPI Documentation: http://localhost:{}/docs".format(self.port))
        print("="*60 + "\n")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info" if debug else "warning"
            )
        finally:
            self.discovery.stop()


def create_web_server(host: str = "0.0.0.0", port: int = 8080, require_auth: bool = True) -> ForgeWebServer:
    """
    Create and return a ForgeWebServer instance.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        require_auth: Whether to require authentication
        
    Returns:
        Configured ForgeWebServer instance
    """
    return ForgeWebServer(host=host, port=port, require_auth=require_auth)
