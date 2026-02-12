"""
Unity/Unreal Plugin Interface for Enigma AI Engine

Integration with game engines.

Features:
- WebSocket bridge
- JSON-RPC protocol
- Avatar control
- Voice streaming
- State synchronization

Usage:
    from enigma_engine.integrations.game_engine_bridge import GameEngineBridge
    
    bridge = GameEngineBridge(port=8765)
    
    # Start server
    bridge.start()
    
    # In Unity/Unreal, connect to ws://localhost:8765
"""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for game engine communication."""
    # Text/Chat
    CHAT_REQUEST = "chat_request"
    CHAT_RESPONSE = "chat_response"
    
    # Avatar
    AVATAR_STATE = "avatar_state"
    AVATAR_ANIMATION = "avatar_animation"
    AVATAR_EXPRESSION = "avatar_expression"
    AVATAR_LOOK_AT = "avatar_look_at"
    
    # Voice
    VOICE_START = "voice_start"
    VOICE_DATA = "voice_data"
    VOICE_END = "voice_end"
    
    # Control
    COMMAND = "command"
    STATUS = "status"
    ERROR = "error"
    
    # Events
    EVENT = "event"
    CALLBACK = "callback"


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    """Rotation quaternion."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Transform:
    """Transform data."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))


@dataclass
class AvatarState:
    """Avatar state for synchronization."""
    transform: Transform = field(default_factory=Transform)
    expression: str = "neutral"
    animation: str = "idle"
    look_at: Optional[Vector3] = None
    blend_shapes: Dict[str, float] = field(default_factory=dict)
    bone_transforms: Dict[str, Transform] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "transform": {
                "position": asdict(self.transform.position),
                "rotation": asdict(self.transform.rotation),
                "scale": asdict(self.transform.scale)
            },
            "expression": self.expression,
            "animation": self.animation,
            "lookAt": asdict(self.look_at) if self.look_at else None,
            "blendShapes": self.blend_shapes,
            "boneTransforms": {
                name: {
                    "position": asdict(t.position),
                    "rotation": asdict(t.rotation),
                    "scale": asdict(t.scale)
                }
                for name, t in self.bone_transforms.items()
            }
        }


@dataclass
class EngineMessage:
    """Message for game engine communication."""
    type: str
    id: str = ""
    data: Any = None
    timestamp: float = 0.0
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type,
            "id": self.id,
            "data": self.data,
            "timestamp": self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EngineMessage':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=data.get("type", ""),
            id=data.get("id", ""),
            data=data.get("data"),
            timestamp=data.get("timestamp", 0.0)
        )


class GameEngineBridge:
    """WebSocket bridge for game engine communication."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        on_chat: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize game engine bridge.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            on_chat: Chat callback
        """
        self.host = host
        self.port = port
        self.on_chat = on_chat
        
        # State
        self._running = False
        self._server = None
        self._clients: List[Any] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
        # Avatar state
        self._avatar_state = AvatarState()
        
        # Message handlers
        self._handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # Callbacks
        self._event_callbacks: List[Callable] = []
        
        logger.info(f"GameEngineBridge initialized on {host}:{port}")
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self._handlers[MessageType.CHAT_REQUEST.value] = self._handle_chat
        self._handlers[MessageType.COMMAND.value] = self._handle_command
        self._handlers[MessageType.AVATAR_STATE.value] = self._handle_avatar_state
    
    def start(self):
        """Start the bridge server."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        
        logger.info("GameEngineBridge started")
    
    def stop(self):
        """Stop the bridge server."""
        self._running = False
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        logger.info("GameEngineBridge stopped")
    
    def _run_server(self):
        """Run the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed")
            return
        
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        async def handler(websocket, path):
            self._clients.append(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                # Send initial state
                await self._send_avatar_state(websocket)
                
                async for message in websocket:
                    await self._handle_message(websocket, message)
                    
            except Exception as e:
                logger.error(f"Connection error: {e}")
            finally:
                self._clients.remove(websocket)
                logger.info(f"Client disconnected")
        
        async def main():
            async with websockets.serve(handler, self.host, self.port):
                while self._running:
                    await asyncio.sleep(0.1)
        
        try:
            self._loop.run_until_complete(main())
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    async def _handle_message(self, websocket, raw_message: str):
        """Handle incoming message."""
        try:
            message = EngineMessage.from_json(raw_message)
            
            handler = self._handlers.get(message.type)
            if handler:
                response = await asyncio.get_running_loop().run_in_executor(
                    None, handler, message
                )
                
                if response:
                    await websocket.send(response.to_json())
            else:
                logger.warning(f"Unknown message type: {message.type}")
                
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            error_msg = EngineMessage(
                type=MessageType.ERROR.value,
                data={"error": str(e)}
            )
            await websocket.send(error_msg.to_json())
    
    def _handle_chat(self, message: EngineMessage) -> Optional[EngineMessage]:
        """Handle chat request."""
        text = message.data.get("text", "") if isinstance(message.data, dict) else str(message.data)
        
        response_text = ""
        if self.on_chat:
            try:
                response_text = self.on_chat(text)
            except Exception as e:
                response_text = f"Error: {e}"
        else:
            response_text = f"Echo: {text}"
        
        return EngineMessage(
            type=MessageType.CHAT_RESPONSE.value,
            id=message.id,
            data={"text": response_text}
        )
    
    def _handle_command(self, message: EngineMessage) -> Optional[EngineMessage]:
        """Handle command."""
        command = message.data.get("command", "") if isinstance(message.data, dict) else ""
        
        # Handle built-in commands
        if command == "get_state":
            return EngineMessage(
                type=MessageType.AVATAR_STATE.value,
                id=message.id,
                data=self._avatar_state.to_dict()
            )
        elif command == "ping":
            return EngineMessage(
                type=MessageType.STATUS.value,
                id=message.id,
                data={"status": "ok", "message": "pong"}
            )
        
        return None
    
    def _handle_avatar_state(self, message: EngineMessage) -> Optional[EngineMessage]:
        """Handle avatar state update from engine."""
        data = message.data
        if not isinstance(data, dict):
            return None
        
        # Update local state from engine
        if "transform" in data:
            t = data["transform"]
            if "position" in t:
                self._avatar_state.transform.position = Vector3(**t["position"])
            if "rotation" in t:
                self._avatar_state.transform.rotation = Quaternion(**t["rotation"])
        
        if "expression" in data:
            self._avatar_state.expression = data["expression"]
        
        if "animation" in data:
            self._avatar_state.animation = data["animation"]
        
        return None
    
    async def _send_avatar_state(self, websocket):
        """Send current avatar state."""
        message = EngineMessage(
            type=MessageType.AVATAR_STATE.value,
            data=self._avatar_state.to_dict()
        )
        await websocket.send(message.to_json())
    
    def broadcast(self, message: EngineMessage):
        """Broadcast message to all clients."""
        if not self._loop or not self._clients:
            return
        
        async def _broadcast():
            for client in self._clients:
                try:
                    await client.send(message.to_json())
                except Exception:
                    pass  # Intentionally silent
        
        asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)
    
    def set_avatar_state(self, state: AvatarState):
        """Update and broadcast avatar state."""
        self._avatar_state = state
        
        message = EngineMessage(
            type=MessageType.AVATAR_STATE.value,
            data=state.to_dict()
        )
        self.broadcast(message)
    
    def play_animation(self, animation: str, **params):
        """Trigger animation on connected clients."""
        message = EngineMessage(
            type=MessageType.AVATAR_ANIMATION.value,
            data={"animation": animation, **params}
        )
        self.broadcast(message)
    
    def set_expression(self, expression: str, intensity: float = 1.0):
        """Set avatar expression."""
        self._avatar_state.expression = expression
        
        message = EngineMessage(
            type=MessageType.AVATAR_EXPRESSION.value,
            data={"expression": expression, "intensity": intensity}
        )
        self.broadcast(message)
    
    def look_at(self, target: Vector3):
        """Set avatar look-at target."""
        self._avatar_state.look_at = target
        
        message = EngineMessage(
            type=MessageType.AVATAR_LOOK_AT.value,
            data=asdict(target)
        )
        self.broadcast(message)
    
    def send_voice_data(self, audio_data: bytes):
        """Stream voice data to clients."""
        import base64
        
        message = EngineMessage(
            type=MessageType.VOICE_DATA.value,
            data={"audio": base64.b64encode(audio_data).decode()}
        )
        self.broadcast(message)
    
    def register_handler(
        self,
        message_type: str,
        handler: Callable[[EngineMessage], Optional[EngineMessage]]
    ):
        """Register custom message handler."""
        self._handlers[message_type] = handler
    
    def on_event(self, callback: Callable[[str, Any], None]):
        """Register event callback."""
        self._event_callbacks.append(callback)


# Unity SDK helper code
UNITY_SDK_CODE = '''
// Enigma AI Unity SDK
// Add to your Unity project

using System;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;

[Serializable]
public class EnigmaMessage {
    public string type;
    public string id;
    public object data;
    public float timestamp;
}

[Serializable]
public class Vector3Data {
    public float x, y, z;
}

[Serializable]
public class QuaternionData {
    public float x, y, z, w;
}

[Serializable]
public class AvatarStateData {
    public TransformData transform;
    public string expression;
    public string animation;
    public Vector3Data lookAt;
    public Dictionary<string, float> blendShapes;
}

[Serializable]
public class TransformData {
    public Vector3Data position;
    public QuaternionData rotation;
    public Vector3Data scale;
}

public class EnigmaAIClient : MonoBehaviour {
    [Header("Connection")]
    public string serverUrl = "ws://localhost:8765";
    public bool autoConnect = true;
    
    [Header("Avatar")]
    public Animator avatarAnimator;
    public SkinnedMeshRenderer faceMesh;
    
    private WebSocket ws;
    private Queue<Action> mainThreadActions = new Queue<Action>();
    
    public event Action<string> OnChatResponse;
    public event Action<AvatarStateData> OnAvatarState;
    
    void Start() {
        if (autoConnect) Connect();
    }
    
    void Update() {
        while (mainThreadActions.Count > 0) {
            mainThreadActions.Dequeue()?.Invoke();
        }
    }
    
    public void Connect() {
        ws = new WebSocket(serverUrl);
        
        ws.OnMessage += (sender, e) => {
            mainThreadActions.Enqueue(() => HandleMessage(e.Data));
        };
        
        ws.OnOpen += (sender, e) => Debug.Log("Connected to Enigma AI");
        ws.OnClose += (sender, e) => Debug.Log("Disconnected from Enigma AI");
        ws.OnError += (sender, e) => Debug.LogError($"Error: {e.Message}");
        
        ws.Connect();
    }
    
    public void Disconnect() {
        ws?.Close();
    }
    
    public void SendChat(string message) {
        var msg = new EnigmaMessage {
            type = "chat_request",
            id = Guid.NewGuid().ToString(),
            data = new { text = message }
        };
        ws.Send(JsonUtility.ToJson(msg));
    }
    
    void HandleMessage(string json) {
        var msg = JsonUtility.FromJson<EnigmaMessage>(json);
        
        switch (msg.type) {
            case "chat_response":
                var textData = msg.data as Dictionary<string, object>;
                OnChatResponse?.Invoke(textData["text"].ToString());
                break;
                
            case "avatar_state":
                var state = JsonUtility.FromJson<AvatarStateData>(JsonUtility.ToJson(msg.data));
                OnAvatarState?.Invoke(state);
                ApplyAvatarState(state);
                break;
                
            case "avatar_animation":
                var animData = msg.data as Dictionary<string, object>;
                if (avatarAnimator) avatarAnimator.Play(animData["animation"].ToString());
                break;
        }
    }
    
    void ApplyAvatarState(AvatarStateData state) {
        if (state == null) return;
        
        // Apply blend shapes
        if (faceMesh && state.blendShapes != null) {
            foreach (var kvp in state.blendShapes) {
                int idx = faceMesh.sharedMesh.GetBlendShapeIndex(kvp.Key);
                if (idx >= 0) faceMesh.SetBlendShapeWeight(idx, kvp.Value * 100);
            }
        }
        
        // Apply animation
        if (avatarAnimator && !string.IsNullOrEmpty(state.animation)) {
            avatarAnimator.Play(state.animation);
        }
    }
    
    void OnDestroy() {
        Disconnect();
    }
}
'''

# Unreal SDK helper code  
UNREAL_SDK_CODE = '''
// Enigma AI Unreal SDK
// Add to your Unreal project (requires WebSockets plugin)

// EnigmaAIClient.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "IWebSocket.h"
#include "EnigmaAIClient.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnChatResponse, const FString&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAvatarAnimation, const FString&, Animation);

UCLASS()
class YOURPROJECT_API AEnigmaAIClient : public AActor
{
    GENERATED_BODY()

public:
    AEnigmaAIClient();

    UPROPERTY(EditAnywhere, Category = "Enigma AI")
    FString ServerUrl = TEXT("ws://localhost:8765");

    UPROPERTY(BlueprintAssignable, Category = "Enigma AI")
    FOnChatResponse OnChatResponse;

    UPROPERTY(BlueprintAssignable, Category = "Enigma AI")
    FOnAvatarAnimation OnAvatarAnimation;

    UFUNCTION(BlueprintCallable, Category = "Enigma AI")
    void Connect();

    UFUNCTION(BlueprintCallable, Category = "Enigma AI")
    void Disconnect();

    UFUNCTION(BlueprintCallable, Category = "Enigma AI")
    void SendChat(const FString& Message);

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

private:
    TSharedPtr<IWebSocket> WebSocket;
    void HandleMessage(const FString& Message);
};
'''


def generate_unity_sdk(output_dir: str = "sdk/unity"):
    """Generate Unity SDK files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / "EnigmaAIClient.cs").write_text(UNITY_SDK_CODE)
    
    logger.info(f"Generated Unity SDK at {output_path}")
    return str(output_path)


def generate_unreal_sdk(output_dir: str = "sdk/unreal"):
    """Generate Unreal SDK files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    (output_path / "EnigmaAIClient.h").write_text(UNREAL_SDK_CODE)
    
    logger.info(f"Generated Unreal SDK at {output_path}")
    return str(output_path)


# Global instance
_bridge: Optional[GameEngineBridge] = None


def get_game_engine_bridge(port: int = 8765) -> GameEngineBridge:
    """Get or create global GameEngineBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = GameEngineBridge(port=port)
    return _bridge
