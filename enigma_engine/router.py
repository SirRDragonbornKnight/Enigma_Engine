"""
Brick Router - Central hub for brick connections with background training.

The router:
1. Accepts brick connections on port 9900
2. Routes messages between bricks and the engine
3. Runs background training while bricks operate
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BrickConnection:
    """Represents a connected brick."""
    brick_id: str
    name: str
    socket: socket.socket
    address: tuple
    capabilities: list[str] = field(default_factory=list)
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class TrainingExample:
    """A single training example collected from conversations."""
    prompt: str
    response: str
    score: float = 1.0
    source: str = "chat"
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# TRAINING THREAD
# =============================================================================

class BackgroundTrainer(threading.Thread):
    """
    Background training thread that learns from collected examples.
    
    Runs continuously while the router is active, processing training
    examples in a queue without blocking the main application.
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        learning_rate: float = 1e-5,
        batch_size: int = 2,
        save_interval: int = 100,
        checkpoint_dir: str = "models/checkpoints/router_training",
    ):
        super().__init__(daemon=True)
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # System prompt for context (set from PromptTab)
        self.system_prompt: str = ""
        
        # Training state
        self.example_queue: queue.Queue[TrainingExample] = queue.Queue()
        self.examples_processed = 0
        self.total_loss = 0.0
        self.running = False
        self.paused = False
        
        # Optimizer (created when model is set)
        self.optimizer: torch.optim.Optimizer | None = None
        
        # Callbacks
        self.on_progress: Callable[[int, float], None] | None = None
        self.on_checkpoint: Callable[[str], None] | None = None
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def set_model(self, model, tokenizer):
        """Set or update the model to train."""
        self.model = model
        self.tokenizer = tokenizer
        if model is not None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            model.train()  # Set to training mode
            logger.info("BackgroundTrainer: Model set for training")
        
    def add_example(self, prompt: str, response: str, score: float = 1.0, source: str = "chat"):
        """Add a training example to the queue."""
        example = TrainingExample(
            prompt=prompt,
            response=response,
            score=score,
            source=source
        )
        self.example_queue.put(example)
        logger.debug(f"Added training example (queue size: {self.example_queue.qsize()})")
        
    def run(self):
        """Main training loop."""
        self.running = True
        batch: list[TrainingExample] = []
        
        logger.info("BackgroundTrainer started")
        
        while self.running:
            # Check if paused
            if self.paused:
                time.sleep(0.5)
                continue
                
            # Check if model is available
            if self.model is None or self.tokenizer is None:
                time.sleep(1.0)
                continue
            
            # Collect batch
            try:
                example = self.example_queue.get(timeout=1.0)
                batch.append(example)
            except queue.Empty:
                # No examples available
                if batch:
                    # Process partial batch after timeout
                    self._train_batch(batch)
                    batch = []
                continue
            
            # Train when batch is full
            if len(batch) >= self.batch_size:
                self._train_batch(batch)
                batch = []
                
        logger.info("BackgroundTrainer stopped")
        
    def _train_batch(self, batch: list[TrainingExample]):
        """Train on a batch of examples."""
        if not batch or self.model is None:
            return
            
        try:
            self.model.train()
            total_batch_loss = 0.0
            
            for example in batch:
                # Prepare input with optional system prompt context
                if self.system_prompt:
                    text = f"System: {self.system_prompt}\n\nUser: {example.prompt}\n\nAssistant: {example.response}"
                else:
                    text = f"User: {example.prompt}\n\nAssistant: {example.response}"
                
                # Tokenize
                if hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = self.tokenizer(text)
                    
                if not tokens or len(tokens) < 2:
                    continue
                    
                # Convert to tensor
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
                
                # Move to device
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(input_ids)
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                # Weight by score
                loss = loss * example.score
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                self.optimizer.step()
                
                total_batch_loss += loss.item()
                
            # Restore eval mode so inference isn't affected by dropout
            self.model.eval()
            
            # Update stats
            self.examples_processed += len(batch)
            avg_loss = total_batch_loss / len(batch)
            self.total_loss = 0.9 * self.total_loss + 0.1 * avg_loss  # EMA
            
            # Callback
            if self.on_progress:
                self.on_progress(self.examples_processed, self.total_loss)
                
            # Periodic checkpoint
            if self.examples_processed % self.save_interval == 0:
                self._save_checkpoint()
                
            logger.debug(
                f"Trained batch: {len(batch)} examples, "
                f"loss={avg_loss:.4f}, total={self.examples_processed}"
            )
            
        except Exception as e:
            logger.error(f"Training batch error: {e}")
            
    def _save_checkpoint(self):
        """Save a training checkpoint."""
        if self.model is None:
            return
            
        checkpoint_path = self.checkpoint_dir / f"router_ckpt_{self.examples_processed}.pth"
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'examples_processed': self.examples_processed,
                'total_loss': self.total_loss,
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            if self.on_checkpoint:
                self.on_checkpoint(str(checkpoint_path))
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def pause(self):
        """Pause training."""
        self.paused = True
        logger.info("BackgroundTrainer paused")
        
    def resume(self):
        """Resume training."""
        self.paused = False
        logger.info("BackgroundTrainer resumed")
        
    def stop(self):
        """Stop the training thread."""
        self.running = False
        
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'running': self.running,
            'paused': self.paused,
            'examples_processed': self.examples_processed,
            'queue_size': self.example_queue.qsize(),
            'average_loss': self.total_loss,
            'has_model': self.model is not None,
        }


# =============================================================================
# BRICK ROUTER
# =============================================================================

class BrickRouter:
    """
    Central router for brick connections.
    
    Handles:
    - TCP server on port 9900
    - Brick connection management
    - Message routing between bricks and engine
    - Background training from conversations
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9900,
        enable_training: bool = True,
        heartbeat_interval: float = 30.0,
        max_connections: int = 50,
    ):
        self.host = host
        self.port = port
        self.enable_training = enable_training
        self.heartbeat_interval = heartbeat_interval
        self.max_connections = max_connections
        
        # Server state
        self.server_socket: socket.socket | None = None
        self.running = False
        self.accept_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        
        # Connected bricks
        self.bricks: dict[str, BrickConnection] = {}
        self.brick_lock = threading.Lock()
        
        # Multi-purpose prompts for different contexts
        self.prompts: dict[str, str] = {
            "chat": "You are a helpful AI assistant.",
            "gui_usage": "You can control the application using [CMD]command[/CMD] blocks.",
            "training_scorer": "Score this response from 1-100 based on helpfulness, accuracy, and clarity.",
            "brick_router": "Route tasks to the appropriate brick based on the request type.",
            "safety": "Be helpful, harmless, and honest. Refuse harmful requests politely.",
        }
        
        # Training
        self.trainer = BackgroundTrainer() if enable_training else None
        
        # Callbacks
        self.on_brick_connected: Callable[[BrickConnection], None] | None = None
        self.on_brick_disconnected: Callable[[str], None] | None = None
        self.on_message: Callable[[str, dict], None] | None = None
        
        # Message handlers
        self.message_handlers: dict[str, Callable] = {}
        
    def start(self) -> bool:
        """Start the router server."""
        if self.running:
            logger.warning("Router already running")
            return False
            
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            self.server_socket.settimeout(1.0)  # For clean shutdown
            
            self.running = True
            
            # Start accept thread
            self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.accept_thread.start()
            
            # Start training thread
            if self.trainer:
                self.trainer.start()

            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()
                
            logger.info(f"Router started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start router: {e}")
            self.running = False
            return False
            
    def stop(self):
        """Stop the router server."""
        self.running = False
        
        # Stop trainer
        if self.trainer:
            self.trainer.stop()
            
        # Close all brick connections
        with self.brick_lock:
            for brick in list(self.bricks.values()):
                try:
                    brick.socket.close()
                except Exception:
                    pass
            self.bricks.clear()
            
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None
            
        logger.info("Router stopped")
        
    def _heartbeat_loop(self):
        """Periodically ping bricks and remove dead connections."""
        while self.running:
            time.sleep(self.heartbeat_interval)
            if not self.running:
                break
            dead: list[str] = []
            with self.brick_lock:
                now = time.time()
                for brick_id, brick in self.bricks.items():
                    if now - brick.last_seen > self.heartbeat_interval * 3:
                        dead.append(brick_id)
                    else:
                        try:
                            self._send_message(brick.socket, {"type": "ping"})
                        except Exception:
                            dead.append(brick_id)
                for brick_id in dead:
                    brick = self.bricks.pop(brick_id, None)
                    if brick:
                        try:
                            brick.socket.close()
                        except Exception:
                            pass
            for brick_id in dead:
                logger.info(f"Heartbeat: removed dead brick {brick_id}")
                if self.on_brick_disconnected:
                    self.on_brick_disconnected(brick_id)

    def _accept_loop(self):
        """Accept incoming brick connections."""
        while self.running:
            try:
                # Reject when at capacity
                if len(self.bricks) >= self.max_connections:
                    time.sleep(0.5)
                    continue

                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address}")
                
                # Start handler thread
                handler = threading.Thread(
                    target=self._handle_brick,
                    args=(client_socket, address),
                    daemon=True
                )
                handler.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
                    
    def _handle_brick(self, client_socket: socket.socket, address: tuple):
        """Handle a connected brick."""
        brick_id = None
        
        try:
            client_socket.settimeout(30.0)  # 30s timeout for registration
            
            # Wait for registration message
            data = self._receive_message(client_socket)
            if not data:
                logger.warning(f"No registration from {address}")
                client_socket.close()
                return
                
            if data.get('type') != 'register':
                logger.warning(f"Expected register, got {data.get('type')}")
                client_socket.close()
                return
                
            # Create brick connection
            brick_id = data.get('brick_id', f"brick_{time.time()}")
            
            brick = BrickConnection(
                brick_id=brick_id,
                name=data.get('name', 'Unknown Brick'),
                socket=client_socket,
                address=address,
                capabilities=data.get('capabilities', [])
            )
            
            # Store brick
            with self.brick_lock:
                self.bricks[brick_id] = brick
                
            # Send acknowledgment
            self._send_message(client_socket, {
                'type': 'registered',
                'brick_id': brick_id,
                'status': 'ok'
            })
            
            logger.info(f"Brick registered: {brick.name} ({brick_id})")
            
            if self.on_brick_connected:
                self.on_brick_connected(brick)
                
            # Set normal timeout
            client_socket.settimeout(60.0)
            
            # Message loop
            while self.running:
                data = self._receive_message(client_socket)
                if data is None:
                    break
                    
                self._handle_message(brick_id, data)
                
        except socket.timeout:
            logger.debug(f"Brick timeout: {brick_id or address}")
        except ConnectionResetError:
            logger.debug(f"Brick disconnected: {brick_id or address}")
        except Exception as e:
            logger.error(f"Brick handler error: {e}")
        finally:
            # Cleanup
            if brick_id:
                with self.brick_lock:
                    if brick_id in self.bricks:
                        del self.bricks[brick_id]
                        
                if self.on_brick_disconnected:
                    self.on_brick_disconnected(brick_id)
                    
                logger.info(f"Brick disconnected: {brick_id}")
                
            try:
                client_socket.close()
            except Exception:
                pass
                
    def _handle_message(self, brick_id: str, data: dict):
        """Handle a message from a brick."""
        msg_type = data.get('type', 'unknown')
        
        # Check for registered handler
        if msg_type in self.message_handlers:
            try:
                self.message_handlers[msg_type](brick_id, data)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")
            return
            
        # Default handling
        if msg_type == 'response':
            # Brick completed a task
            prompt = data.get('prompt', '')
            response = data.get('response', '')
            score = data.get('score', 1.0)
            
            # Add to training queue
            if self.trainer and prompt and response:
                self.trainer.add_example(prompt, response, score, source=f"brick:{brick_id}")
                
        elif msg_type == 'ping':
            # Respond to ping
            brick = self.bricks.get(brick_id)
            if brick:
                brick.last_seen = time.time()
                self._send_message(brick.socket, {'type': 'pong'})

        elif msg_type == 'pong':
            # Heartbeat reply — update last_seen
            brick = self.bricks.get(brick_id)
            if brick:
                brick.last_seen = time.time()
                
        # Callback
        if self.on_message:
            self.on_message(brick_id, data)
            
    def _receive_message(self, sock: socket.socket) -> dict | None:
        """Receive a JSON message."""
        try:
            # Read length prefix (4 bytes)
            length_data = b''
            while len(length_data) < 4:
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return None
                length_data += chunk
                
            length = int.from_bytes(length_data, 'big')
            
            if length > 1_000_000:  # 1MB max
                logger.warning(f"Message too large: {length}")
                return None
                
            # Read message
            data = b''
            while len(data) < length:
                chunk = sock.recv(min(4096, length - len(data)))
                if not chunk:
                    return None
                data += chunk
                
            return json.loads(data.decode('utf-8'))
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"Receive error: {e}")
            return None
            
    def _send_message(self, sock: socket.socket, data: dict) -> bool:
        """Send a JSON message."""
        try:
            msg = json.dumps(data).encode('utf-8')
            length = len(msg).to_bytes(4, 'big')
            sock.sendall(length + msg)
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
            
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def send_to_brick(self, brick_id: str, message: dict) -> bool:
        """Send a message to a specific brick."""
        with self.brick_lock:
            brick = self.bricks.get(brick_id)
            if brick:
                return self._send_message(brick.socket, message)
        return False
        
    def broadcast(self, message: dict, exclude: list[str] | None = None):
        """Broadcast a message to all connected bricks."""
        exclude_set = set(exclude or [])
        with self.brick_lock:
            for brick_id, brick in self.bricks.items():
                if brick_id not in exclude_set:
                    self._send_message(brick.socket, message)
                    
    def get_connected_bricks(self) -> list[dict]:
        """Get list of connected bricks."""
        with self.brick_lock:
            return [
                {
                    'brick_id': b.brick_id,
                    'name': b.name,
                    'capabilities': b.capabilities,
                    'connected_at': b.connected_at,
                    'uptime': time.time() - b.connected_at,
                }
                for b in self.bricks.values()
            ]
            
    def add_training_example(self, prompt: str, response: str, score: float = 1.0):
        """Add a training example from chat."""
        if self.trainer:
            self.trainer.add_example(prompt, response, score, source="chat")
            
    def set_training_model(self, model, tokenizer, system_prompt: str = ""):
        """Set the model for background training."""
        if self.trainer:
            self.trainer.set_model(model, tokenizer)
            if system_prompt:
                self.trainer.system_prompt = system_prompt
                
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for training context."""
        if self.trainer:
            self.trainer.system_prompt = prompt
            
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        if self.trainer:
            return self.trainer.get_stats()
        return {'enabled': False}
        
    def pause_training(self):
        """Pause background training."""
        if self.trainer:
            self.trainer.pause()
            
    def resume_training(self):
        """Resume background training."""
        if self.trainer:
            self.trainer.resume()
            
    def get_status(self) -> dict:
        """Get full router status."""
        return {
            'running': self.running,
            'host': self.host,
            'port': self.port,
            'connected_bricks': len(self.bricks),
            'bricks': self.get_connected_bricks(),
            'training': self.get_training_stats(),
        }
    
    def get_prompt(self, purpose: str) -> str:
        """
        Get a prompt by purpose.
        
        Args:
            purpose: One of 'chat', 'gui_usage', 'training_scorer', 
                     'brick_router', 'safety', or a brick name
        
        Returns:
            The prompt text, or empty string if not found
        """
        return self.prompts.get(purpose, "")
    
    def set_prompt(self, purpose: str, prompt: str):
        """
        Set or update a prompt for a specific purpose.
        
        Args:
            purpose: The prompt identifier (e.g., 'chat', 'safety', brick name)
            prompt: The prompt text
        """
        self.prompts[purpose] = prompt
        
        # If setting the main system prompt, also update trainer
        if purpose == "chat" and self.trainer:
            self.trainer.system_prompt = prompt
    
    def add_brick_prompt(self, brick_name: str, prompt: str):
        """Add a prompt snippet from a brick (called during registration)."""
        if prompt:
            self.prompts[f"brick_{brick_name}"] = prompt
    
    def get_combined_prompt(self, *purposes: str) -> str:
        """
        Combine multiple prompts into one.
        
        Args:
            *purposes: Prompt purposes to combine (e.g., 'chat', 'safety')
        
        Returns:
            Combined prompt text with each section on new lines
        """
        parts = []
        for purpose in purposes:
            prompt = self.prompts.get(purpose, "")
            if prompt:
                parts.append(prompt)
        return "\n\n".join(parts)
        
    def register_handler(self, msg_type: str, handler: Callable):
        """Register a custom message handler."""
        self.message_handlers[msg_type] = handler


# =============================================================================
# SINGLETON ROUTER INSTANCE
# =============================================================================

_router_instance: BrickRouter | None = None


def get_router() -> BrickRouter:
    """Get or create the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = BrickRouter()
    return _router_instance


def start_router(host: str = "127.0.0.1", port: int = 9900) -> bool:
    """Start the global router."""
    router = get_router()
    router.host = host
    router.port = port
    return router.start()


def stop_router():
    """Stop the global router."""
    global _router_instance
    if _router_instance:
        _router_instance.stop()
        _router_instance = None
