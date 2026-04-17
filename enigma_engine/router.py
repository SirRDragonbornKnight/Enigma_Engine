"""
Mod Router - Central hub for mod connections with background training.

The router:
1. Accepts mod connections on port 9900
2. Routes messages between mods and the engine
3. Runs background training while mods operate
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Deferred torch import — loaded on first use to avoid 540 MB idle RAM
_torch = None
_torch_lock = threading.Lock()

def _ensure_torch() -> Any:
    """Import torch on first use."""
    global _torch
    if _torch is None:
        with _torch_lock:
            if _torch is None:
                import torch
                _torch = torch
    return _torch


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModConnection:
    """Represents a connected mod."""
    mod_id: str
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

    Smart features (BT-B + BT-D):
    - Replay buffer: rolling collection of recent examples.
      Periodic retraining on the full buffer prevents catastrophic
      forgetting.
    - DPO pairs: can be populated externally for preference training.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        learning_rate: float = 1e-5,
        batch_size: int = 2,
        save_interval: int = 100,
        checkpoint_dir: str = "models/checkpoints/router_training",
        replay_buffer_size: int = 1000,
        retrain_interval: int = 200,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps

        # System prompt for context (set from PromptTab)
        self.system_prompt: str = ""

        # Training state
        self.example_queue: queue.Queue[TrainingExample] = queue.Queue()
        self.examples_processed = 0
        self.total_loss = 0.0
        self.running = False
        self.paused = False

        # Lock to prevent train/eval mode conflicts with inference
        self._train_lock = threading.Lock()

        # Optimizer (created when model is set)
        self.optimizer: Any | None = None

        # Callbacks
        self.on_progress: Callable[[int, float], None] | None = None
        self.on_checkpoint: Callable[[str], None] | None = None

        # --- Smart features (BT-B + BT-D) ---

        # Replay buffer: capped at max size, keeps recent examples
        # for periodic retraining to prevent catastrophic forgetting.
        self.replay_buffer_size: int = replay_buffer_size
        self.replay_buffer: deque[TrainingExample] = deque(maxlen=replay_buffer_size)
        self._replay_lock = threading.Lock()

        # DPO preference pairs: (rejected_example, chosen_response)
        # Can be populated externally for preference training.
        self.dpo_pairs: list[dict[str, str]] = []

        # Retrain on replay buffer every N examples processed
        self.retrain_interval: int = retrain_interval

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def set_model(self, model, tokenizer) -> None:
        """Set or update the model to train."""
        with self._train_lock:
            self.model = model
            self.tokenizer = tokenizer
            if model is not None:
                torch = _ensure_torch()
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=0.01,
                    betas=self.adam_betas,
                    eps=self.adam_eps,
                )
                # Keep model in eval mode — _train_batch switches to
                # train mode only inside the lock and restores eval after.
                model.eval()
                logger.info("BackgroundTrainer: Model set for training")

    def add_example(self, prompt: str, response: str, score: float = 1.0, source: str = "chat") -> None:
        """Add a training example to the queue and replay buffer.

        All examples are queued for training and stored in the
        replay buffer (capped by recency).
        """
        example = TrainingExample(
            prompt=prompt,
            response=response,
            score=score,
            source=source
        )

        # Add to training queue
        self.example_queue.put(example)

        # Add to replay buffer (deque auto-trims to maxlen)
        with self._replay_lock:
            self.replay_buffer.append(example)

        logger.debug(
            "Added training example (queue=%d, replay=%d)",
            self.example_queue.qsize(),
            len(self.replay_buffer))

    def run(self) -> None:
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

    def _train_batch(self, batch: list[TrainingExample]) -> None:
        """Train on a batch of examples.

        Trains on all examples equally.
        Triggers replay retrain periodically (BT-D).
        """
        if not batch or self.model is None:
            return

        try:
            with self._train_lock:
                self.model.train()
                try:
                    total_batch_loss = 0.0

                    # Forward/backward all examples, single optimizer step
                    if self.optimizer is None:
                        logger.warning("No optimizer configured — skipping training batch")
                        return
                    self.optimizer.zero_grad()
                    valid_count = 0
                    batch_len = len(batch)

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
                        torch = _ensure_torch()
                        device = next(self.model.parameters()).device
                        input_ids = torch.tensor(
                            [tokens[:-1]], dtype=torch.long, device=device)
                        target_ids = torch.tensor(
                            [tokens[1:]], dtype=torch.long, device=device)

                        # Forward pass
                        output = self.model(input_ids)
                        # Unpack tuple if model returns (logits, loss)
                        logits = output[0] if isinstance(output, tuple) else output

                        # Calculate loss
                        loss = _ensure_torch().nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1)
                        )

                        # Backward — scale for gradient accumulation
                        (loss / batch_len).backward()

                        total_batch_loss += loss.item()
                        valid_count += 1

                    # Single optimizer step after all examples
                    if valid_count > 0:
                        _ensure_torch().nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        self.optimizer.step()
                finally:
                    # Always restore eval mode — even on exception — so
                    # dropout / batchnorm don't corrupt inference output.
                    self.model.eval()

            # Update stats
            self.examples_processed += len(batch)
            avg_loss = total_batch_loss / max(1, len(batch))
            self.total_loss = 0.9 * self.total_loss + 0.1 * avg_loss  # EMA

            # Callback
            if self.on_progress:
                self.on_progress(self.examples_processed, self.total_loss)

            # Periodic checkpoint
            if self.examples_processed % self.save_interval == 0:
                self._save_checkpoint()

            # Periodic retrain on replay buffer (BT-D)
            if (self.retrain_interval > 0
                    and self.examples_processed % self.retrain_interval == 0
                    and len(self.replay_buffer) >= self.batch_size):
                self._retrain_on_replay()

            logger.debug(
                "Trained batch: %d examples, "
                "loss=%.4f, total=%d",
                len(batch),
                avg_loss, self.examples_processed)

        except Exception as e:
            logger.error("Training batch error: %s\n%s", e,
                         traceback.format_exc())

    def _retrain_on_replay(self) -> None:
        """Retrain on the best examples in the replay buffer (BT-D).

        Takes a snapshot of the top examples from the replay buffer
        and runs a mini training pass.  This prevents catastrophic
        forgetting by periodically reinforcing the best exchanges.
        """
        if self.model is None or not self.replay_buffer:
            return
        if self.optimizer is None:
            return

        with self._replay_lock:
            # Sort by score (descending) so we retrain on the best examples
            sorted_buf = sorted(
                self.replay_buffer, key=lambda x: x.score, reverse=True)
            top_k = min(len(sorted_buf), self.replay_buffer_size // 2)
            replay_batch = list(sorted_buf[:top_k])

        if not replay_batch:
            return

        logger.info(
            "BackgroundTrainer: retraining on %d replay examples",
            len(replay_batch))

        try:
            with self._train_lock:
                self.model.train()
                # Halve LR for replay to avoid over-fitting on
                # the same examples, then restore after the loop.
                orig_lrs = []
                for pg in self.optimizer.param_groups:
                    orig_lrs.append(pg["lr"])
                    pg["lr"] = pg["lr"] * 0.5
                try:
                    self.optimizer.zero_grad()
                    valid_count = 0
                    replay_len = len(replay_batch)

                    for example in replay_batch:
                        if self.system_prompt:
                            text = (
                                f"System: {self.system_prompt}\n\n"
                                f"User: {example.prompt}\n\n"
                                f"Assistant: {example.response}")
                        else:
                            text = (
                                f"User: {example.prompt}\n\n"
                                f"Assistant: {example.response}")

                        if hasattr(self.tokenizer, "encode"):
                            tokens = self.tokenizer.encode(text)
                        else:
                            tokens = self.tokenizer(text)

                        if not tokens or len(tokens) < 2:
                            continue

                        torch = _ensure_torch()
                        device = next(self.model.parameters()).device
                        input_ids = torch.tensor(
                            [tokens[:-1]], dtype=torch.long, device=device)
                        target_ids = torch.tensor(
                            [tokens[1:]], dtype=torch.long, device=device)

                        output = self.model(input_ids)
                        logits = (output[0] if isinstance(output, tuple)
                                  else output)
                        loss = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1))
                        (loss / replay_len).backward()
                        valid_count += 1

                    if valid_count > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 1.0)
                        self.optimizer.step()
                finally:
                    # Restore original LR and eval mode
                    if len(orig_lrs) != len(self.optimizer.param_groups):
                        logger.warning(
                            "Param group count changed during replay "
                            "(%d -> %d), skipping LR restore",
                            len(orig_lrs),
                            len(self.optimizer.param_groups))
                    else:
                        for pg, lr in zip(self.optimizer.param_groups,
                                          orig_lrs):
                            pg["lr"] = lr
                    self.model.eval()

            logger.info("BackgroundTrainer: replay retrain complete")

        except Exception as e:
            logger.error("Replay retrain error: %s\n%s", e,
                         traceback.format_exc())

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        if self.model is None:
            return

        checkpoint_path = self.checkpoint_dir / f"router_ckpt_{self.examples_processed}.pth"

        try:
            from enigma_engine.core.safe_save import atomic_torch_save
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'examples_processed': self.examples_processed,
                'total_loss': self.total_loss,
                'replay_buffer_size': len(self.replay_buffer),
            }
            if hasattr(self.model, 'config'):
                c = self.model.config
                cfg_dict = {
                    "vocab_size": c.vocab_size, "dim": c.dim,
                    "n_layers": c.n_layers, "n_heads": c.n_heads,
                    "n_kv_heads": c.n_kv_heads,
                    "hidden_dim": c.hidden_dim,
                    "max_seq_len": c.max_seq_len,
                    "dropout": c.dropout,
                    "use_rope": c.use_rope,
                    "use_moe": c.use_moe,
                }
                save_data['model_config'] = cfg_dict
                save_data['config'] = cfg_dict
            atomic_torch_save(save_data, checkpoint_path)

            logger.info(f"Saved checkpoint: {checkpoint_path}")

            if self.on_checkpoint:
                self.on_checkpoint(str(checkpoint_path))

        except Exception as e:
            logger.error("Failed to save checkpoint: %s\n%s", e,
                         traceback.format_exc())

    def pause(self) -> None:
        """Pause training."""
        self.paused = True
        logger.info("BackgroundTrainer paused")

    def resume(self) -> None:
        """Resume training."""
        self.paused = False
        logger.info("BackgroundTrainer resumed")

    def stop(self) -> None:
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
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_max': self.replay_buffer_size,
            'dpo_pairs': len(self.dpo_pairs),
        }

    @property
    def train_lock(self) -> threading.Lock:
        """Expose the training lock so inference code can coordinate.

        Inference callers should ``with trainer.train_lock:`` around
        their forward pass to prevent train/eval mode interleaving
        and gradient corruption when the same model is shared.
        """
        return self._train_lock


# =============================================================================
# MOD ROUTER
# =============================================================================

class ModRouter:
    """
    Central router for mod connections.

    Handles:
    - TCP server on port 9900
    - Mod connection management
    - Message routing between mods and engine
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

        # Connected mods
        self.mods: dict[str, ModConnection] = {}
        self.mod_lock = threading.Lock()

        # Multi-purpose prompts for different contexts
        self.prompts: dict[str, str] = {
            "chat": "You are a helpful AI assistant.",
            "gui_usage": "You can control the application using [CMD]command[/CMD] blocks.",
            "training_scorer": "Score this response from 1-100 based on helpfulness, accuracy, and clarity.",
            "mod_router": "Route tasks to the appropriate mod based on the request type.",
        }

        # Training
        self._train_lock = threading.Lock()
        self.trainer = BackgroundTrainer() if enable_training else None

        # Callbacks
        self.on_mod_connected: Callable[[ModConnection], None] | None = None
        self.on_mod_disconnected: Callable[[str], None] | None = None
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

    def stop(self) -> None:
        """Stop the router server."""
        self.running = False

        # Stop trainer
        if self.trainer:
            self.trainer.stop()

        # Close all mod connections
        with self.mod_lock:
            for mod in list(self.mods.values()):
                try:
                    mod.socket.close()
                except Exception as e:
                    logger.debug("Error closing mod socket: %s", e)
            self.mods.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logger.debug("Error closing server socket: %s", e)
            self.server_socket = None

        logger.info("Router stopped")

    def _heartbeat_loop(self) -> None:
        """Periodically ping mods and remove dead connections."""
        while self.running:
            time.sleep(self.heartbeat_interval)
            if not self.running:
                break
            # Snapshot mods to ping outside the lock — socket I/O can block
            to_ping: list[tuple[str, "ModConnection"]] = []
            dead: list[str] = []
            with self.mod_lock:
                now = time.time()
                for mod_id, mod in self.mods.items():
                    if now - mod.last_seen > self.heartbeat_interval * 3:
                        dead.append(mod_id)
                    else:
                        to_ping.append((mod_id, mod))
            # Ping outside the lock to avoid blocking other mod ops
            for mod_id, mod in to_ping:
                try:
                    self._send_message(mod.socket, {"type": "ping"})
                except Exception:
                    dead.append(mod_id)
            # Remove dead mods under lock
            with self.mod_lock:
                for mod_id in dead:
                    mod = self.mods.pop(mod_id, None)
                    if mod:
                        try:
                            mod.socket.close()
                        except Exception:
                            pass
            for mod_id in dead:
                logger.info(f"Heartbeat: removed dead mod {mod_id}")
                if self.on_mod_disconnected:
                    self.on_mod_disconnected(mod_id)

    def _accept_loop(self) -> None:
        """Accept incoming mod connections."""
        while self.running:
            try:
                # Reject when at capacity
                with self.mod_lock:
                    at_capacity = len(self.mods) >= self.max_connections
                if at_capacity:
                    time.sleep(0.5)
                    continue

                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address}")

                # Start handler thread
                handler = threading.Thread(
                    target=self._handle_mod,
                    args=(client_socket, address),
                    daemon=True
                )
                handler.start()

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")

    def _handle_mod(self, client_socket: socket.socket, address: tuple) -> None:
        """Handle a connected mod."""
        mod_id = None

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

            # Create mod connection
            mod_id = data.get('mod_id', f"mod_{time.time()}")

            mod = ModConnection(
                mod_id=mod_id,
                name=data.get('name', 'Unknown Mod'),
                socket=client_socket,
                address=address,
                capabilities=data.get('capabilities', [])
            )

            # Store mod
            with self.mod_lock:
                self.mods[mod_id] = mod

            # Send acknowledgment
            self._send_message(client_socket, {
                'type': 'registered',
                'mod_id': mod_id,
                'status': 'ok'
            })

            logger.info(f"Mod registered: {mod.name} ({mod_id})")

            if self.on_mod_connected:
                self.on_mod_connected(mod)

            # Set normal timeout
            client_socket.settimeout(60.0)

            # Message loop
            while self.running:
                data = self._receive_message(client_socket)
                if data is None:
                    break

                self._handle_message(mod_id, data)

        except socket.timeout:
            logger.debug(f"Mod timeout: {mod_id or address}")
        except ConnectionResetError:
            logger.debug(f"Mod disconnected: {mod_id or address}")
        except Exception as e:
            logger.error(f"Mod handler error: {e}")
        finally:
            # Cleanup
            if mod_id:
                with self.mod_lock:
                    self.mods.pop(mod_id, None)

                if self.on_mod_disconnected:
                    self.on_mod_disconnected(mod_id)

                logger.info(f"Mod disconnected: {mod_id}")

            try:
                client_socket.close()
            except Exception:
                pass

    def _handle_message(self, mod_id: str, data: dict) -> None:
        """Handle a message from a mod."""
        msg_type = data.get('type', 'unknown')

        # Check for registered handler
        if msg_type in self.message_handlers:
            try:
                self.message_handlers[msg_type](mod_id, data)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")
            return

        # Default handling
        if msg_type == 'response':
            # Mod completed a task
            prompt = data.get('prompt', '')
            response = data.get('response', '')
            score = data.get('score', 1.0)

            # Add to training queue
            if self.trainer and prompt and response:
                self.trainer.add_example(prompt, response, score, source=f"mod:{mod_id}")

        elif msg_type == 'ping':
            # Respond to ping
            with self.mod_lock:
                mod = self.mods.get(mod_id)
            if mod:
                mod.last_seen = time.time()
                self._send_message(mod.socket, {'type': 'pong'})

        elif msg_type == 'pong':
            # Heartbeat reply — update last_seen
            with self.mod_lock:
                mod = self.mods.get(mod_id)
            if mod:
                mod.last_seen = time.time()

        # Callback
        if self.on_message:
            self.on_message(mod_id, data)

    def _receive_message(self, sock: socket.socket) -> dict | None:
        """Receive a JSON message."""
        try:
            deadline = time.monotonic() + 30.0  # aggregate timeout

            # Read length prefix (4 bytes)
            length_data = b''
            while len(length_data) < 4:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("Message receive timed out (header)")
                    return None
                sock.settimeout(min(remaining, 60.0))
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return None
                length_data += chunk

            length = int.from_bytes(length_data, 'big')

            if length > 1_000_000:  # 1MB max
                try:
                    client_ip = sock.getpeername()[0]
                except Exception:
                    client_ip = "unknown"
                logger.warning(
                    "Message too large from %s: %d bytes",
                    client_ip, length)
                return None

            # Read message
            data = b''
            while len(data) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("Message receive timed out (body)")
                    return None
                sock.settimeout(min(remaining, 60.0))
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

    def send_to_mod(self, mod_id: str, message: dict) -> bool:
        """Send a message to a specific mod."""
        with self.mod_lock:
            mod = self.mods.get(mod_id)
            if mod:
                return self._send_message(mod.socket, message)
        return False

    def broadcast(self, message: dict, exclude: list[str] | None = None) -> None:
        """Broadcast a message to all connected mods."""
        exclude_set = set(exclude or [])
        with self.mod_lock:
            targets = [
                (mod_id, mod) for mod_id, mod in self.mods.items()
                if mod_id not in exclude_set
            ]
        for _mod_id, mod in targets:
            self._send_message(mod.socket, message)

    def get_connected_mods(self) -> list[dict]:
        """Get list of connected mods."""
        with self.mod_lock:
            return [
                {
                    'mod_id': mod.mod_id,
                    'name': mod.name,
                    'capabilities': mod.capabilities,
                    'connected_at': mod.connected_at,
                    'uptime': time.time() - mod.connected_at,
                }
                for mod in self.mods.values()
            ]

    def add_training_example(self, prompt: str, response: str, score: float = 1.0) -> None:
        """Add a training example from chat."""
        if self.trainer:
            self.trainer.add_example(prompt, response, score, source="chat")

    def set_training_enabled(self, enabled: bool) -> None:
        """Enable or disable the background trainer at runtime."""
        with self._train_lock:
            if enabled:
                if self.trainer is not None:
                    return
                self.trainer = BackgroundTrainer()
                if self.running:
                    self.trainer.start()
                return

            trainer = self.trainer
            if trainer is None:
                return
            self.trainer = None
        trainer.stop()

    def set_training_model(self, model, tokenizer, system_prompt: str = "") -> None:
        """Set the model for background training."""
        if self.trainer:
            self.trainer.set_model(model, tokenizer)
            if system_prompt:
                self.trainer.system_prompt = system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for training context."""
        if self.trainer:
            self.trainer.system_prompt = prompt

    def get_training_stats(self) -> dict:
        """Get training statistics."""
        if self.trainer:
            return self.trainer.get_stats()
        return {'enabled': False}

    def get_train_lock(self) -> threading.Lock | None:
        """Return the training lock for inference coordination.

        When background training and inference share the same model,
        wrap the inference forward pass with::

            lock = router.get_train_lock()
            if lock:
                with lock:
                    output = model(input_ids)
            else:
                output = model(input_ids)

        Returns None if no trainer is active.
        """
        if self.trainer:
            return self.trainer.train_lock
        return None

    def pause_training(self) -> None:
        """Pause background training."""
        if self.trainer:
            self.trainer.pause()

    def resume_training(self) -> None:
        """Resume background training."""
        if self.trainer:
            self.trainer.resume()

    def get_status(self) -> dict:
        """Get full router status."""
        return {
            'running': self.running,
            'host': self.host,
            'port': self.port,
            'connected_mods': len(self.mods),
            'mods': self.get_connected_mods(),
            'training': self.get_training_stats(),
        }

    def get_prompt(self, purpose: str) -> str:
        """
        Get a prompt by purpose.

        Args:
            purpose: One of 'chat', 'gui_usage', 'training_scorer',
                     'mod_router', 'safety', or a mod name

        Returns:
            The prompt text, or empty string if not found
        """
        return self.prompts.get(purpose, "")

    def set_prompt(self, purpose: str, prompt: str) -> None:
        """
        Set or update a prompt for a specific purpose.

        Args:
            purpose: The prompt identifier (e.g., 'chat', 'safety', mod name)
            prompt: The prompt text
        """
        self.prompts[purpose] = prompt

        # If setting the main system prompt, also update trainer
        if purpose == "chat" and self.trainer:
            self.trainer.system_prompt = prompt

    def add_mod_prompt(self, mod_name: str, prompt: str) -> None:
        """Add a prompt snippet from a mod (called during registration)."""
        if prompt:
            self.prompts[f"mod_{mod_name}"] = prompt

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

    def register_handler(self, msg_type: str, handler: Callable) -> None:
        """Register a custom message handler."""
        self.message_handlers[msg_type] = handler


# =============================================================================
# SINGLETON ROUTER INSTANCE
# =============================================================================

_router_instance: ModRouter | None = None
_router_lock = threading.Lock()


def get_router() -> ModRouter:
    """Get or create the global router instance."""
    global _router_instance
    if _router_instance is None:
        with _router_lock:
            if _router_instance is None:
                _router_instance = ModRouter()
    return _router_instance


def start_router(host: str = "127.0.0.1", port: int = 9900) -> bool:
    """Start the global router."""
    router = get_router()
    router.host = host
    router.port = port
    return router.start()


def stop_router() -> None:
    """Stop the global router."""
    global _router_instance
    if _router_instance:
        _router_instance.stop()
        _router_instance = None
