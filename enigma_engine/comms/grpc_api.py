"""
gRPC API

High-performance RPC interface for Enigma AI Engine.

FILE: enigma_engine/comms/grpc_api.py
TYPE: API
MAIN CLASSES: ForgeGRPCServer, ForgeGRPCClient
"""

import json
import logging
import time
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

try:
    import grpc
    from grpc import aio as grpc_aio
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Proto definitions as Python (normally generated from .proto)
# This provides the service interface without requiring proto compilation

@dataclass
class GenerateRequest:
    """Generation request."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = None
    stream: bool = False


@dataclass
class GenerateResponse:
    """Generation response."""
    text: str
    tokens_generated: int
    generation_time: float
    finish_reason: str


@dataclass
class ChatMessage:
    """Chat message."""
    role: str
    content: str


@dataclass
class ChatRequest:
    """Chat request."""
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


@dataclass
class ChatResponse:
    """Chat response."""
    message: ChatMessage
    tokens_generated: int
    generation_time: float


@dataclass
class ModuleRequest:
    """Module operation request."""
    name: str
    action: str  # load, unload, status


@dataclass
class ModuleResponse:
    """Module operation response."""
    name: str
    status: str
    success: bool
    error: str = ""


@dataclass 
class ToolRequest:
    """Tool execution request."""
    name: str
    arguments: str  # JSON


@dataclass
class ToolResponse:
    """Tool execution response."""
    result: str  # JSON
    success: bool
    error: str = ""


@dataclass
class HealthResponse:
    """Health check response."""
    status: str
    uptime: float
    version: str


# Proto file content for reference (defined at module level)
PROTO_CONTENT = '''
syntax = "proto3";

package forgeengine;

service ForgeService {
    // Text generation
    rpc Generate(GenerateRequest) returns (GenerateResponse);
    rpc GenerateStream(GenerateRequest) returns (stream GenerateResponse);
    
    // Chat
    rpc Chat(ChatRequest) returns (ChatResponse);
    rpc ChatStream(ChatRequest) returns (stream ChatResponse);
    
    // Module management
    rpc ManageModule(ModuleRequest) returns (ModuleResponse);
    
    // Tool execution
    rpc ExecuteTool(ToolRequest) returns (ToolResponse);
    
    // Health check
    rpc HealthCheck(HealthCheckRequest) returns (HealthResponse);
}

message GenerateRequest {
    string prompt = 1;
    int32 max_tokens = 2;
    float temperature = 3;
    float top_p = 4;
    float top_k = 5;
    repeated string stop_sequences = 6;
    bool stream = 7;
}

message GenerateResponse {
    string text = 1;
    int32 tokens_generated = 2;
    float generation_time = 3;
    string finish_reason = 4;
}

message ChatMessage {
    string role = 1;
    string content = 2;
}

message ChatRequest {
    repeated ChatMessage messages = 1;
    int32 max_tokens = 2;
    float temperature = 3;
    bool stream = 4;
}

message ChatResponse {
    ChatMessage message = 1;
    int32 tokens_generated = 2;
    float generation_time = 3;
}

message ModuleRequest {
    string name = 1;
    string action = 2;  // load, unload, status
}

message ModuleResponse {
    string name = 1;
    string status = 2;
    bool success = 3;
    string error = 4;
}

message ToolRequest {
    string name = 1;
    string arguments = 2;  // JSON string
}

message ToolResponse {
    string result = 1;  // JSON string
    bool success = 2;
    string error = 3;
}

message HealthCheckRequest {}

message HealthResponse {
    string status = 1;
    float uptime = 2;
    string version = 3;
}
'''


if HAS_GRPC:
    
    class ForgeServicer:
        """gRPC servicer implementation."""
        
        def __init__(self):
            self._start_time = time.time()
            self._engine = None
        
        def _get_engine(self):
            """Lazy load inference engine."""
            if self._engine is None:
                try:
                    from enigma_engine.core.inference import EnigmaEngine
                    self._engine = EnigmaEngine()
                except Exception as e:
                    logger.error(f"Failed to load engine: {e}")
            return self._engine
        
        def Generate(self, request: GenerateRequest) -> GenerateResponse:
            """Generate text."""
            engine = self._get_engine()
            
            if not engine:
                return GenerateResponse(
                    text="",
                    tokens_generated=0,
                    generation_time=0,
                    finish_reason="error"
                )
            
            start = time.time()
            
            try:
                result = engine.generate(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                text = result.text if hasattr(result, "text") else str(result)
                
                return GenerateResponse(
                    text=text,
                    tokens_generated=len(text.split()),
                    generation_time=time.time() - start,
                    finish_reason="complete"
                )
            except Exception as e:
                return GenerateResponse(
                    text=str(e),
                    tokens_generated=0,
                    generation_time=time.time() - start,
                    finish_reason="error"
                )
        
        def GenerateStream(
            self,
            request: GenerateRequest
        ) -> Iterator[GenerateResponse]:
            """Stream generated text."""
            engine = self._get_engine()
            
            if not engine:
                yield GenerateResponse(
                    text="",
                    tokens_generated=0,
                    generation_time=0,
                    finish_reason="error"
                )
                return
            
            start = time.time()
            total_tokens = 0
            
            try:
                # Simulate streaming by generating in chunks
                result = engine.generate(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                text = result.text if hasattr(result, "text") else str(result)
                words = text.split()
                
                # Stream word by word
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    total_tokens += 1
                    
                    yield GenerateResponse(
                        text=chunk,
                        tokens_generated=total_tokens,
                        generation_time=time.time() - start,
                        finish_reason="" if i < len(words) - 1 else "complete"
                    )
            except Exception as e:
                yield GenerateResponse(
                    text=str(e),
                    tokens_generated=0,
                    generation_time=time.time() - start,
                    finish_reason="error"
                )
        
        def Chat(self, request: ChatRequest) -> ChatResponse:
            """Chat completion."""
            engine = self._get_engine()
            
            if not engine:
                return ChatResponse(
                    message=ChatMessage(role="assistant", content="Error: Engine not loaded"),
                    tokens_generated=0,
                    generation_time=0
                )
            
            start = time.time()
            
            # Build prompt from messages
            prompt_parts = []
            for msg in request.messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}")
            
            prompt_parts.append("Assistant:")
            prompt = "\n".join(prompt_parts)
            
            try:
                result = engine.generate(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                text = result.text if hasattr(result, "text") else str(result)
                
                return ChatResponse(
                    message=ChatMessage(role="assistant", content=text),
                    tokens_generated=len(text.split()),
                    generation_time=time.time() - start
                )
            except Exception as e:
                return ChatResponse(
                    message=ChatMessage(role="assistant", content=str(e)),
                    tokens_generated=0,
                    generation_time=time.time() - start
                )
        
        def ManageModule(self, request: ModuleRequest) -> ModuleResponse:
            """Manage modules."""
            try:
                from enigma_engine.modules import ModuleManager
                manager = ModuleManager()
                
                if request.action == "load":
                    manager.load(request.name)
                    return ModuleResponse(
                        name=request.name,
                        status="loaded",
                        success=True
                    )
                elif request.action == "unload":
                    manager.unload(request.name)
                    return ModuleResponse(
                        name=request.name,
                        status="unloaded",
                        success=True
                    )
                elif request.action == "status":
                    is_loaded = manager.is_loaded(request.name)
                    return ModuleResponse(
                        name=request.name,
                        status="loaded" if is_loaded else "unloaded",
                        success=True
                    )
                else:
                    return ModuleResponse(
                        name=request.name,
                        status="unknown",
                        success=False,
                        error=f"Unknown action: {request.action}"
                    )
            except Exception as e:
                return ModuleResponse(
                    name=request.name,
                    status="error",
                    success=False,
                    error=str(e)
                )
        
        def ExecuteTool(self, request: ToolRequest) -> ToolResponse:
            """Execute tool."""
            try:
                from enigma_engine.tools import ToolExecutor
                executor = ToolExecutor()
                
                args = json.loads(request.arguments) if request.arguments else {}
                result = executor.execute(request.name, args)
                
                return ToolResponse(
                    result=json.dumps(result),
                    success=True
                )
            except Exception as e:
                return ToolResponse(
                    result="",
                    success=False,
                    error=str(e)
                )
        
        def HealthCheck(self, request=None) -> HealthResponse:
            """Health check."""
            return HealthResponse(
                status="healthy",
                uptime=time.time() - self._start_time,
                version="1.0.0"
            )
    
    
    class ForgeGRPCServer:
        """gRPC server for Enigma AI Engine."""
        
        def __init__(self, host: str = "0.0.0.0", port: int = 50051):
            self.host = host
            self.port = port
            self.server = None
            self.servicer = ForgeServicer()
        
        def _create_server(self) -> grpc.Server:
            """Create gRPC server with custom handlers."""
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10)
            )
            
            # Add generic handlers for our service
            # In production, you'd use generated servicers from .proto
            
            # For now, create a simple handler
            server.add_insecure_port(f"{self.host}:{self.port}")
            
            return server
        
        def start(self):
            """Start server."""
            self.server = self._create_server()
            self.server.start()
            logger.info(f"gRPC server started on {self.host}:{self.port}")
        
        def stop(self, grace: float = 5.0):
            """Stop server."""
            if self.server:
                self.server.stop(grace)
                logger.info("gRPC server stopped")
        
        def wait_for_termination(self):
            """Wait for server termination."""
            if self.server:
                self.server.wait_for_termination()
    
    
    class ForgeGRPCClient:
        """gRPC client for Enigma AI Engine."""
        
        def __init__(self, host: str = "localhost", port: int = 50051):
            self.host = host
            self.port = port
            self.channel = None
        
        def connect(self):
            """Connect to server."""
            self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        
        def close(self):
            """Close connection."""
            if self.channel:
                self.channel.close()
        
        def generate(
            self,
            prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.7,
            stream: bool = False
        ) -> GenerateResponse:
            """Generate text."""
            # In production, use generated stubs
            # This is a simplified implementation
            
            request = GenerateRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            
            # Would call stub.Generate(request) with generated code
            # For now, return placeholder
            return GenerateResponse(
                text="[gRPC client placeholder]",
                tokens_generated=0,
                generation_time=0,
                finish_reason="stub"
            )
        
        def chat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: int = 256,
            temperature: float = 0.7
        ) -> ChatResponse:
            """Chat completion."""
            chat_messages = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in messages
            ]
            
            request = ChatRequest(
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Would call stub.Chat(request)
            return ChatResponse(
                message=ChatMessage(role="assistant", content="[gRPC client placeholder]"),
                tokens_generated=0,
                generation_time=0
            )
        
        def health_check(self) -> HealthResponse:
            """Check server health."""
            # Would call stub.HealthCheck()
            return HealthResponse(
                status="unknown",
                uptime=0,
                version="unknown"
            )


else:
    class ForgeGRPCServer:
        def __init__(self, *args, **kwargs):
            raise ImportError("grpcio required for gRPC server")
    
    class ForgeGRPCClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("grpcio required for gRPC client")


def create_grpc_server(host: str = "0.0.0.0", port: int = 50051) -> 'ForgeGRPCServer':
    """Create gRPC server instance."""
    if not HAS_GRPC:
        raise ImportError("grpcio required: pip install grpcio grpcio-tools")
    return ForgeGRPCServer(host, port)


def create_grpc_client(host: str = "localhost", port: int = 50051) -> 'ForgeGRPCClient':
    """Create gRPC client instance."""
    if not HAS_GRPC:
        raise ImportError("grpcio required: pip install grpcio")
    return ForgeGRPCClient(host, port)


def save_proto_file(output_path: str = "enigma_engine/comms/forge.proto"):
    """Save proto file for code generation."""
    with open(output_path, "w") as f:
        f.write(PROTO_CONTENT)
    logger.info(f"Proto file saved to {output_path}")
