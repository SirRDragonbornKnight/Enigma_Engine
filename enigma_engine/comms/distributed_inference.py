"""
Distributed Inference System

Split model inference across multiple devices for faster processing
or to run models too large for a single device.

Architecture:
- Coordinator: Orchestrates inference, splits prompts, merges results
- Worker: Processes a portion of the inference workload

Use cases:
- Run a 70B model split across 4 devices
- Parallel inference for batch processing
- Load balancing across multiple GPUs/machines

Usage:
    from enigma_engine.comms.distributed_inference import (
        DistributedCoordinator, 
        InferenceWorker
    )
    
    # On devices with model portions
    worker = InferenceWorker(port=5070, model_path="models/my_model")
    worker.start()
    
    # On coordinator device
    coordinator = DistributedCoordinator()
    coordinator.add_worker("192.168.1.100:5070")
    coordinator.add_worker("192.168.1.101:5070")
    
    # Run distributed inference
    response = coordinator.generate("Tell me a story", max_tokens=500)
"""

import json
import logging
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PyQt5.QtCore import QObject, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None


logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of an inference worker."""
    OFFLINE = "offline"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class WorkerInfo:
    """Information about an inference worker."""
    address: str
    status: WorkerStatus = WorkerStatus.OFFLINE
    model_name: str = ""
    model_size: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    current_task: str = ""
    tasks_completed: int = 0
    average_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "status": self.status.value,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "capabilities": self.capabilities,
            "last_heartbeat": self.last_heartbeat,
            "tasks_completed": self.tasks_completed,
            "average_latency_ms": self.average_latency_ms
        }


@dataclass 
class InferenceTask:
    """A distributed inference task."""
    task_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    status: str = "pending"
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0


class InferenceWorker:
    """
    Worker that processes inference requests from a coordinator.
    
    Run this on each machine that will participate in distributed inference.
    """
    
    def __init__(
        self, 
        port: int = 5070, 
        host: str = "0.0.0.0",
        model_path: str = None
    ):
        self.port = port
        self.host = host
        self.model_path = model_path
        
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._engine = None
        self._model_info: Dict[str, Any] = {}
        
        # Task queue
        self._task_queue: queue.Queue = queue.Queue()
        self._current_task: Optional[InferenceTask] = None
    
    def start(self):
        """Start the inference worker."""
        if self._running:
            return
        
        # Load model if path provided
        if self.model_path:
            self._load_model()
        
        self._running = True
        
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        # Start threads
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._process_tasks, daemon=True).start()
        
        logger.info(f"Inference worker started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the worker."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
    
    def _load_model(self):
        """Load the inference model."""
        try:
            from ..core.inference import EnigmaEngine
            
            self._engine = EnigmaEngine()
            self._engine.load(self.model_path)
            
            self._model_info = {
                "name": Path(self.model_path).stem,
                "size": "unknown",
                "loaded": True
            }
            
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_info = {"loaded": False, "error": str(e)}
    
    def _accept_loop(self):
        """Accept incoming connections."""
        while self._running:
            try:
                client_socket, address = self._server_socket.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address):
        """Handle a client connection."""
        try:
            # Set timeout for client socket to prevent indefinite blocking
            client_socket.settimeout(60.0)
            
            # Receive message
            length_data = client_socket.recv(4)
            if not length_data:
                return
            msg_length = struct.unpack(">I", length_data)[0]
            
            data = b""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(4096, msg_length - len(data)))
                if not chunk:
                    break
                data += chunk
            
            message = json.loads(data.decode())
            cmd = message.get("command")
            
            response = {"success": False, "error": "Unknown command"}
            
            if cmd == "status":
                response = self._handle_status()
            elif cmd == "inference":
                response = self._handle_inference(message)
            elif cmd == "heartbeat":
                response = {"success": True, "status": "alive"}
            elif cmd == "load_model":
                response = self._handle_load_model(message)
            
            self._send_message(client_socket, response)
            
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
    
    def _send_message(self, sock: socket.socket, data: dict):
        """Send a message with length prefix."""
        msg = json.dumps(data).encode()
        sock.sendall(struct.pack(">I", len(msg)) + msg)
    
    def _handle_status(self) -> dict:
        """Return worker status."""
        return {
            "success": True,
            "status": "busy" if self._current_task else "idle",
            "model_info": self._model_info,
            "queue_size": self._task_queue.qsize(),
            "has_model": self._engine is not None
        }
    
    def _handle_inference(self, message: dict) -> dict:
        """Handle an inference request."""
        if not self._engine:
            return {"success": False, "error": "No model loaded"}
        
        prompt = message.get("prompt", "")
        max_tokens = message.get("max_tokens", 100)
        temperature = message.get("temperature", 0.7)
        
        try:
            # Run inference
            start_time = time.time()
            
            result = self._engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "result": result,
                "latency_ms": latency_ms,
                "tokens_generated": len(result.split())  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_load_model(self, message: dict) -> dict:
        """Load a model on demand."""
        model_path = message.get("model_path")
        if not model_path:
            return {"success": False, "error": "No model path provided"}
        
        self.model_path = model_path
        self._load_model()
        
        return {
            "success": self._model_info.get("loaded", False),
            "model_info": self._model_info
        }
    
    def _process_tasks(self):
        """Process tasks from the queue."""
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                self._current_task = task
                
                # Process task
                if self._engine:
                    try:
                        result = self._engine.generate(
                            task.prompt,
                            max_new_tokens=task.max_tokens,
                            temperature=task.temperature
                        )
                        task.result = result
                        task.status = "completed"
                    except Exception as e:
                        task.error = str(e)
                        task.status = "failed"
                else:
                    task.error = "No model loaded"
                    task.status = "failed"
                
                task.completed_at = time.time()
                self._current_task = None
                
            except queue.Empty:
                continue


class DistributedCoordinator(QObject if HAS_PYQT else object):
    """
    Coordinates distributed inference across multiple workers.
    
    Manages worker connections, distributes tasks, and merges results.
    """
    
    # Signals
    if HAS_PYQT:
        worker_connected = pyqtSignal(str)  # address
        worker_disconnected = pyqtSignal(str)  # address
        inference_started = pyqtSignal(str)  # task_id
        inference_completed = pyqtSignal(str, str)  # task_id, result
        inference_failed = pyqtSignal(str, str)  # task_id, error
    
    def __init__(self):
        if HAS_PYQT:
            super().__init__()
        
        self._workers: Dict[str, WorkerInfo] = {}
        self._running = False
        
        # Task tracking
        self._tasks: Dict[str, InferenceTask] = {}
        self._task_counter = 0
        
        # Heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None
    
    def add_worker(self, address: str) -> bool:
        """
        Add an inference worker.
        
        Args:
            address: Worker address (host:port)
            
        Returns:
            True if worker connected successfully
        """
        if address in self._workers:
            return True
        
        # Test connection and get info
        try:
            response = self._send_to_worker(address, {"command": "status"})
            
            if response.get("success"):
                model_info = response.get("model_info", {})
                worker = WorkerInfo(
                    address=address,
                    status=WorkerStatus.IDLE if response.get("status") == "idle" else WorkerStatus.BUSY,
                    model_name=model_info.get("name", "unknown"),
                    last_heartbeat=time.time()
                )
                self._workers[address] = worker
                
                if HAS_PYQT:
                    self.worker_connected.emit(address)
                
                logger.info(f"Added worker: {address}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add worker {address}: {e}")
        
        return False
    
    def remove_worker(self, address: str):
        """Remove a worker."""
        if address in self._workers:
            del self._workers[address]
            if HAS_PYQT:
                self.worker_disconnected.emit(address)
            logger.info(f"Removed worker: {address}")
    
    def list_workers(self) -> List[WorkerInfo]:
        """List all workers."""
        return list(self._workers.values())
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        strategy: str = "round_robin"
    ) -> str:
        """
        Generate text using distributed workers.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            strategy: Distribution strategy ("round_robin", "fastest", "random")
            
        Returns:
            Generated text
        """
        if not self._workers:
            raise RuntimeError("No workers available")
        
        # Select worker based on strategy
        worker = self._select_worker(strategy)
        if not worker:
            raise RuntimeError("No available workers")
        
        # Create task
        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{int(time.time())}"
        
        task = InferenceTask(
            task_id=task_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        self._tasks[task_id] = task
        
        if HAS_PYQT:
            self.inference_started.emit(task_id)
        
        # Send to worker
        try:
            worker.status = WorkerStatus.BUSY
            worker.current_task = task_id
            
            response = self._send_to_worker(worker.address, {
                "command": "inference",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            })
            
            worker.status = WorkerStatus.IDLE
            worker.current_task = ""
            worker.tasks_completed += 1
            
            if response.get("success"):
                result = response.get("result", "")
                task.result = result
                task.status = "completed"
                task.completed_at = time.time()
                
                # Update latency stats
                latency = response.get("latency_ms", 0)
                if worker.average_latency_ms == 0:
                    worker.average_latency_ms = latency
                else:
                    worker.average_latency_ms = (worker.average_latency_ms + latency) / 2
                
                if HAS_PYQT:
                    self.inference_completed.emit(task_id, result)
                
                return result
            else:
                error = response.get("error", "Unknown error")
                task.error = error
                task.status = "failed"
                
                if HAS_PYQT:
                    self.inference_failed.emit(task_id, error)
                
                raise RuntimeError(error)
                
        except Exception as e:
            worker.status = WorkerStatus.ERROR
            task.error = str(e)
            task.status = "failed"
            
            if HAS_PYQT:
                self.inference_failed.emit(task_id, str(e))
            
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate text for multiple prompts in parallel.
        
        Distributes prompts across available workers.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            
        Returns:
            List of generated texts
        """
        if not self._workers:
            raise RuntimeError("No workers available")
        
        results = [""] * len(prompts)
        threads = []
        
        def process_prompt(idx: int, prompt: str, worker_addr: str):
            try:
                response = self._send_to_worker(worker_addr, {
                    "command": "inference",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                })
                
                if response.get("success"):
                    results[idx] = response.get("result", "")
                else:
                    results[idx] = f"[Error: {response.get('error')}]"
            except Exception as e:
                results[idx] = f"[Error: {e}]"
        
        # Distribute prompts across workers
        worker_list = list(self._workers.keys())
        
        for idx, prompt in enumerate(prompts):
            worker_addr = worker_list[idx % len(worker_list)]
            
            thread = threading.Thread(
                target=process_prompt,
                args=(idx, prompt, worker_addr)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        return results
    
    def _select_worker(self, strategy: str) -> Optional[WorkerInfo]:
        """Select a worker based on strategy."""
        available = [
            w for w in self._workers.values() 
            if w.status in (WorkerStatus.IDLE, WorkerStatus.BUSY)
        ]
        
        if not available:
            return None
        
        if strategy == "fastest":
            # Select worker with lowest average latency
            return min(available, key=lambda w: w.average_latency_ms or float('inf'))
        
        elif strategy == "random":
            import random
            return random.choice(available)
        
        else:  # round_robin
            # Select least loaded worker
            idle = [w for w in available if w.status == WorkerStatus.IDLE]
            return idle[0] if idle else available[0]
    
    def _send_to_worker(self, address: str, message: dict) -> dict:
        """Send a message to a worker and get response."""
        if ":" in address:
            host, port_str = address.rsplit(":", 1)
            port = int(port_str)
        else:
            host = address
            port = 5070
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(60.0)  # 60 second timeout for inference
        sock.connect((host, port))
        
        # Send message
        msg = json.dumps(message).encode()
        sock.sendall(struct.pack(">I", len(msg)) + msg)
        
        # Receive response
        length_data = sock.recv(4)
        if not length_data:
            sock.close()
            return {"success": False, "error": "No response"}
        
        msg_length = struct.unpack(">I", length_data)[0]
        
        data = b""
        while len(data) < msg_length:
            chunk = sock.recv(min(4096, msg_length - len(data)))
            if not chunk:
                break
            data += chunk
        
        sock.close()
        return json.loads(data.decode())
    
    def start_heartbeat(self, interval: float = 30.0):
        """Start heartbeat monitoring."""
        if self._running:
            return
        
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval,),
            daemon=True
        )
        self._heartbeat_thread.start()
    
    def stop_heartbeat(self):
        """Stop heartbeat monitoring."""
        self._running = False
    
    def _heartbeat_loop(self, interval: float):
        """Check worker health periodically."""
        while self._running:
            for address, worker in list(self._workers.items()):
                try:
                    response = self._send_to_worker(address, {"command": "heartbeat"})
                    if response.get("success"):
                        worker.last_heartbeat = time.time()
                        if worker.status == WorkerStatus.ERROR:
                            worker.status = WorkerStatus.IDLE
                    else:
                        worker.status = WorkerStatus.ERROR
                except Exception:
                    worker.status = WorkerStatus.OFFLINE
                    # Don't remove - might come back
            
            time.sleep(interval)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_coordinator: Optional[DistributedCoordinator] = None
_worker: Optional[InferenceWorker] = None


def get_coordinator() -> DistributedCoordinator:
    """Get or create global coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = DistributedCoordinator()
    return _coordinator


def start_inference_worker(port: int = 5070, model_path: str = None) -> InferenceWorker:
    """Start an inference worker."""
    global _worker
    if _worker is None:
        _worker = InferenceWorker(port=port, model_path=model_path)
    _worker.start()
    return _worker


def stop_inference_worker():
    """Stop the inference worker."""
    global _worker
    if _worker:
        _worker.stop()
        _worker = None


def distributed_generate(
    prompt: str,
    workers: List[str],
    max_tokens: int = 100
) -> str:
    """
    Convenience function for distributed generation.
    
    Args:
        prompt: Input prompt
        workers: List of worker addresses
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    coordinator = get_coordinator()
    
    for addr in workers:
        coordinator.add_worker(addr)
    
    return coordinator.generate(prompt, max_tokens=max_tokens)
