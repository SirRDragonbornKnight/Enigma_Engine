"""
Remote Training System

Train models on a remote device with more compute power, then download
the trained model back to the local device.

Use cases:
- Train a large model on a PC with GPU from a Raspberry Pi
- Offload training to a server while continuing to use local device
- Share training work between multiple devices

Usage:
    from enigma_engine.comms.remote_training import RemoteTrainer, TrainingServer
    
    # On the training server (PC with GPU)
    server = TrainingServer(port=5050)
    server.start()
    
    # On the client (Pi or laptop)
    trainer = RemoteTrainer()
    trainer.connect("192.168.1.100:5050")
    
    # Send training job
    job_id = trainer.submit_training(
        model_name="my_model",
        training_data="path/to/training.txt",
        epochs=10
    )
    
    # Check progress
    progress = trainer.get_progress(job_id)
    print(f"Training: {progress['percent']}% complete")
    
    # Download trained model when done
    trainer.download_model(job_id, "models/my_trained_model")
"""

import hashlib
import json
import logging
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from PyQt5.QtCore import QObject, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None

from ..config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """Represents a remote training job."""
    job_id: str
    model_name: str
    status: str = "pending"  # pending, training, completed, failed
    progress: float = 0.0
    epochs_total: int = 0
    epochs_done: int = 0
    submitted_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    loss_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "status": self.status,
            "progress": self.progress,
            "epochs_total": self.epochs_total,
            "epochs_done": self.epochs_done,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "loss_history": self.loss_history
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingJob":
        return cls(**data)


class TrainingServer:
    """
    Server that accepts and processes remote training requests.
    
    Run this on the machine with good compute resources (GPU).
    """
    
    def __init__(self, port: int = 5050, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.jobs: Dict[str, TrainingJob] = {}
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._job_queue: List[str] = []
        self._processing_thread: Optional[threading.Thread] = None
        
        # Working directory for training artifacts
        self._work_dir = Path(CONFIG.get("data_dir", "data")) / "remote_training"
        self._work_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start the training server."""
        if self._running:
            return
        
        self._running = True
        
        # Start server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        # Start accept thread
        threading.Thread(target=self._accept_loop, daemon=True).start()
        
        # Start job processing thread
        self._processing_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self._processing_thread.start()
        
        logger.info(f"Training server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the training server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
    
    def _accept_loop(self):
        """Accept incoming connections."""
        while self._running:
            try:
                client_socket, address = self._server_socket.accept()
                logger.info(f"Connection from {address}")
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
            
            # Receive message length
            length_data = client_socket.recv(4)
            if not length_data:
                return
            msg_length = struct.unpack(">I", length_data)[0]
            
            # Receive message
            data = b""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(4096, msg_length - len(data)))
                if not chunk:
                    break
                data += chunk
            
            message = json.loads(data.decode())
            
            # Handle command
            cmd = message.get("command")
            response = {"success": False, "error": "Unknown command"}
            
            if cmd == "submit_training":
                response = self._handle_submit_training(message, client_socket)
            elif cmd == "get_progress":
                response = self._handle_get_progress(message)
            elif cmd == "download_model":
                response = self._handle_download_model(message, client_socket)
            elif cmd == "list_jobs":
                response = self._handle_list_jobs()
            elif cmd == "cancel_job":
                response = self._handle_cancel_job(message)
            
            # Send response (for non-file-transfer commands)
            if response:
                self._send_message(client_socket, response)
                
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
    
    def _send_message(self, sock: socket.socket, data: dict):
        """Send a message with length prefix."""
        msg = json.dumps(data).encode()
        sock.sendall(struct.pack(">I", len(msg)) + msg)
    
    def _handle_submit_training(self, message: dict, client_socket: socket.socket) -> dict:
        """Handle a training submission request."""
        try:
            model_name = message.get("model_name", "remote_model")
            epochs = message.get("epochs", 3)
            learning_rate = message.get("learning_rate", 0.0001)
            batch_size = message.get("batch_size", 4)
            training_data_size = message.get("training_data_size", 0)
            
            # Generate job ID
            job_id = f"job_{int(time.time())}_{hashlib.md5(model_name.encode()).hexdigest()[:8]}"
            
            # Acknowledge and request data upload
            self._send_message(client_socket, {
                "success": True,
                "job_id": job_id,
                "ready_for_data": True
            })
            
            # Receive training data file
            job_dir = self._work_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            training_file = job_dir / "training_data.txt"
            
            received = 0
            with open(training_file, "wb") as f:
                while received < training_data_size:
                    chunk = client_socket.recv(min(65536, training_data_size - received))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
            
            # Create job
            job = TrainingJob(
                job_id=job_id,
                model_name=model_name,
                status="pending",
                epochs_total=epochs,
                submitted_at=datetime.now().isoformat()
            )
            
            # Store config
            config = {
                "model_name": model_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "training_file": str(training_file)
            }
            with open(job_dir / "config.json", "w") as f:
                json.dump(config, f)
            
            self.jobs[job_id] = job
            self._job_queue.append(job_id)
            
            logger.info(f"Training job {job_id} submitted: {model_name} for {epochs} epochs")
            
            return {
                "success": True,
                "job_id": job_id,
                "position_in_queue": len(self._job_queue)
            }
            
        except Exception as e:
            logger.error(f"Submit training error: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_get_progress(self, message: dict) -> dict:
        """Get training progress."""
        job_id = message.get("job_id")
        if job_id not in self.jobs:
            return {"success": False, "error": "Job not found"}
        
        job = self.jobs[job_id]
        return {
            "success": True,
            "job": job.to_dict()
        }
    
    def _handle_list_jobs(self) -> dict:
        """List all jobs."""
        return {
            "success": True,
            "jobs": [job.to_dict() for job in self.jobs.values()]
        }
    
    def _handle_cancel_job(self, message: dict) -> dict:
        """Cancel a job."""
        job_id = message.get("job_id")
        if job_id not in self.jobs:
            return {"success": False, "error": "Job not found"}
        
        job = self.jobs[job_id]
        if job.status in ("pending", "training"):
            job.status = "cancelled"
            if job_id in self._job_queue:
                self._job_queue.remove(job_id)
            return {"success": True}
        
        return {"success": False, "error": "Job cannot be cancelled"}
    
    def _handle_download_model(self, message: dict, client_socket: socket.socket) -> Optional[dict]:
        """Handle model download request."""
        job_id = message.get("job_id")
        if job_id not in self.jobs:
            return {"success": False, "error": "Job not found"}
        
        job = self.jobs[job_id]
        if job.status != "completed":
            return {"success": False, "error": "Training not complete"}
        
        # Find the model file
        job_dir = self._work_dir / job_id
        model_file = job_dir / "trained_model.pt"
        
        if not model_file.exists():
            return {"success": False, "error": "Model file not found"}
        
        # Send file size first
        file_size = model_file.stat().st_size
        self._send_message(client_socket, {
            "success": True,
            "file_size": file_size
        })
        
        # Send file data
        with open(model_file, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                client_socket.sendall(chunk)
        
        logger.info(f"Model for job {job_id} downloaded ({file_size} bytes)")
        return None  # Response already sent
    
    def _process_jobs(self):
        """Process training jobs from queue."""
        while self._running:
            if not self._job_queue:
                time.sleep(1)
                continue
            
            job_id = self._job_queue[0]
            job = self.jobs.get(job_id)
            
            if not job or job.status == "cancelled":
                self._job_queue.pop(0)
                continue
            
            # Start training
            job.status = "training"
            job.started_at = datetime.now().isoformat()
            
            try:
                self._run_training(job_id)
                job.status = "completed"
                job.completed_at = datetime.now().isoformat()
                job.progress = 100.0
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                logger.error(f"Training failed for {job_id}: {e}")
            
            self._job_queue.pop(0)
    
    def _run_training(self, job_id: str):
        """Run the actual training."""
        job = self.jobs[job_id]
        job_dir = self._work_dir / job_id
        
        # Load config
        with open(job_dir / "config.json") as f:
            config = json.load(f)
        
        # Import training components
        from ..core.training import TrainingConfig, train_model
        from ..core.model import create_model
        from ..core import get_tokenizer
        
        # Create or load model
        model = create_model(size="small")
        tokenizer = get_tokenizer()
        
        # Configure training
        training_config = TrainingConfig(
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            save_steps=100
        )
        
        # Training callback for progress updates
        def on_step(step, total_steps, loss):
            job.progress = (step / total_steps) * 100
            job.loss_history.append(loss)
        
        def on_epoch(epoch, loss):
            job.epochs_done = epoch
        
        # Run training
        train_model(
            model=model,
            tokenizer=tokenizer,
            data_path=config["training_file"],
            config=training_config,
            on_step=on_step,
            on_epoch=on_epoch
        )
        
        # Save trained model
        import torch
        model_path = job_dir / "trained_model.pt"
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Training completed for job {job_id}")


class RemoteTrainer(QObject if HAS_PYQT else object):
    """
    Client for submitting training jobs to a remote server.
    
    Use this on devices with limited compute to offload training
    to more powerful machines.
    """
    
    # Signals
    if HAS_PYQT:
        progress_updated = pyqtSignal(str, float)  # job_id, progress
        training_completed = pyqtSignal(str)  # job_id
        training_failed = pyqtSignal(str, str)  # job_id, error
    
    def __init__(self):
        if HAS_PYQT:
            super().__init__()
        
        self._server_host: str = ""
        self._server_port: int = 5050
        self._connected = False
        self._jobs: Dict[str, TrainingJob] = {}
        
        # Progress polling thread
        self._poll_thread: Optional[threading.Thread] = None
        self._polling = False
    
    def connect(self, address: str) -> bool:
        """
        Connect to a training server.
        
        Args:
            address: Server address (host:port or just host)
            
        Returns:
            True if connection successful
        """
        if ":" in address:
            host, port_str = address.rsplit(":", 1)
            port = int(port_str)
        else:
            host = address
            port = 5050
        
        self._server_host = host
        self._server_port = port
        
        # Test connection
        try:
            response = self._send_command({"command": "list_jobs"})
            self._connected = response.get("success", False)
            return self._connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def submit_training(
        self,
        model_name: str,
        training_data: str,
        epochs: int = 3,
        learning_rate: float = 0.0001,
        batch_size: int = 4
    ) -> Optional[str]:
        """
        Submit a training job.
        
        Args:
            model_name: Name for the trained model
            training_data: Path to training data file
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Job ID if successful, None otherwise
        """
        if not self._connected:
            logger.error("Not connected to training server")
            return None
        
        training_path = Path(training_data)
        if not training_path.exists():
            logger.error(f"Training data not found: {training_data}")
            return None
        
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self._server_host, self._server_port))
            
            # Send submit request
            file_size = training_path.stat().st_size
            request = {
                "command": "submit_training",
                "model_name": model_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "training_data_size": file_size
            }
            self._send_message(sock, request)
            
            # Get acknowledgment
            response = self._receive_message(sock)
            if not response.get("ready_for_data"):
                sock.close()
                return None
            
            job_id = response.get("job_id")
            
            # Send training data
            with open(training_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    sock.sendall(chunk)
            
            # Get final response
            final_response = self._receive_message(sock)
            sock.close()
            
            if final_response.get("success"):
                # Track job locally
                self._jobs[job_id] = TrainingJob(
                    job_id=job_id,
                    model_name=model_name,
                    epochs_total=epochs,
                    submitted_at=datetime.now().isoformat()
                )
                
                # Start polling if not already
                self._start_polling()
                
                logger.info(f"Training job submitted: {job_id}")
                return job_id
            
            return None
            
        except Exception as e:
            logger.error(f"Submit training error: {e}")
            return None
    
    def get_progress(self, job_id: str) -> Optional[dict]:
        """Get progress of a training job."""
        try:
            response = self._send_command({
                "command": "get_progress",
                "job_id": job_id
            })
            
            if response.get("success"):
                job_data = response.get("job", {})
                # Update local job
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = job_data.get("status", job.status)
                    job.progress = job_data.get("progress", job.progress)
                    job.epochs_done = job_data.get("epochs_done", job.epochs_done)
                return job_data
            
            return None
        except Exception as e:
            logger.error(f"Get progress error: {e}")
            return None
    
    def download_model(self, job_id: str, output_path: str) -> bool:
        """
        Download a trained model.
        
        Args:
            job_id: Job ID
            output_path: Where to save the model
            
        Returns:
            True if successful
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self._server_host, self._server_port))
            
            # Request download
            self._send_message(sock, {
                "command": "download_model",
                "job_id": job_id
            })
            
            # Get response with file size
            response = self._receive_message(sock)
            if not response.get("success"):
                logger.error(f"Download failed: {response.get('error')}")
                sock.close()
                return False
            
            file_size = response.get("file_size", 0)
            
            # Receive file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            received = 0
            with open(output_file, "wb") as f:
                while received < file_size:
                    chunk = sock.recv(min(65536, file_size - received))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
            
            sock.close()
            
            logger.info(f"Model downloaded to {output_path} ({received} bytes)")
            return received == file_size
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def list_jobs(self) -> List[dict]:
        """List all jobs on the server."""
        try:
            response = self._send_command({"command": "list_jobs"})
            return response.get("jobs", [])
        except Exception:
            return []
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        try:
            response = self._send_command({
                "command": "cancel_job",
                "job_id": job_id
            })
            return response.get("success", False)
        except Exception:
            return False
    
    def _send_command(self, command: dict) -> dict:
        """Send a command and get response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._server_host, self._server_port))
        
        self._send_message(sock, command)
        response = self._receive_message(sock)
        
        sock.close()
        return response
    
    def _send_message(self, sock: socket.socket, data: dict):
        """Send a message with length prefix."""
        msg = json.dumps(data).encode()
        sock.sendall(struct.pack(">I", len(msg)) + msg)
    
    def _receive_message(self, sock: socket.socket) -> dict:
        """Receive a message with length prefix."""
        length_data = sock.recv(4)
        if not length_data:
            return {}
        msg_length = struct.unpack(">I", length_data)[0]
        
        data = b""
        while len(data) < msg_length:
            chunk = sock.recv(min(4096, msg_length - len(data)))
            if not chunk:
                break
            data += chunk
        
        return json.loads(data.decode())
    
    def _start_polling(self):
        """Start polling for job progress."""
        if self._polling:
            return
        
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
    
    def _poll_loop(self):
        """Poll for job updates."""
        while self._polling and self._connected:
            for job_id, job in list(self._jobs.items()):
                if job.status in ("pending", "training"):
                    progress = self.get_progress(job_id)
                    if progress:
                        # Emit signals
                        if HAS_PYQT:
                            self.progress_updated.emit(job_id, progress.get("progress", 0))
                            
                            if progress.get("status") == "completed":
                                self.training_completed.emit(job_id)
                            elif progress.get("status") == "failed":
                                self.training_failed.emit(job_id, progress.get("error", ""))
            
            time.sleep(5)  # Poll every 5 seconds
    
    def stop_polling(self):
        """Stop polling."""
        self._polling = False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_trainer: Optional[RemoteTrainer] = None
_server: Optional[TrainingServer] = None


def get_remote_trainer() -> RemoteTrainer:
    """Get or create global remote trainer client."""
    global _trainer
    if _trainer is None:
        _trainer = RemoteTrainer()
    return _trainer


def start_training_server(port: int = 5050) -> TrainingServer:
    """Start a training server."""
    global _server
    if _server is None:
        _server = TrainingServer(port=port)
    _server.start()
    return _server


def stop_training_server():
    """Stop the training server."""
    global _server
    if _server:
        _server.stop()
        _server = None


def submit_remote_training(
    server_address: str,
    model_name: str,
    training_data: str,
    epochs: int = 3
) -> Optional[str]:
    """
    Convenience function to submit a remote training job.
    
    Args:
        server_address: Training server address (host:port)
        model_name: Name for the model
        training_data: Path to training data
        epochs: Number of epochs
        
    Returns:
        Job ID if successful
    """
    trainer = get_remote_trainer()
    if not trainer.connect(server_address):
        return None
    return trainer.submit_training(model_name, training_data, epochs)
