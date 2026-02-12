"""
Distributed Training Network for Multi-Computer Setups

Enables distributed AI training across multiple computers:
- Teacher machine (powerful GPU): Runs large teacher model, generates training data
- Student machine(s) (any GPU): Trains student model, reports quality back
- Works over LAN using the existing network.py foundation

Usage:
    # On Teacher Machine (has powerful GPU like RTX 5090)
    from enigma_engine.learning.distributed_training import TeacherNode
    
    teacher = TeacherNode(port=5100)
    teacher.start()  # Starts serving training data requests
    
    # On Student Machine(s) (can be less powerful)
    from enigma_engine.learning.distributed_training import StudentNode
    
    student = StudentNode(teacher_url="192.168.1.100:5100")
    student.connect()
    student.train_loop()  # Fetches data, trains, reports back
"""

import json
import logging
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrainingTask:
    """A training task from teacher to student."""
    task_id: str
    training_data: List[Dict[str, str]]  # Q&A pairs
    target_loss: float = 1.0
    max_epochs: int = 3
    priority: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "training_data": self.training_data,
            "target_loss": self.target_loss,
            "max_epochs": self.max_epochs,
            "priority": self.priority,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingTask":
        return cls(
            task_id=data["task_id"],
            training_data=data["training_data"],
            target_loss=data.get("target_loss", 1.0),
            max_epochs=data.get("max_epochs", 3),
            priority=data.get("priority", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class TrainingResult:
    """Result from student after training."""
    task_id: str
    student_id: str
    final_loss: float
    epochs_completed: int
    quality_score: float  # 0-10 based on test performance
    duration_seconds: float
    success: bool = True
    error_msg: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "student_id": self.student_id,
            "final_loss": self.final_loss,
            "epochs_completed": self.epochs_completed,
            "quality_score": self.quality_score,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_msg": self.error_msg,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingResult":
        return cls(
            task_id=data["task_id"],
            student_id=data["student_id"],
            final_loss=data["final_loss"],
            epochs_completed=data["epochs_completed"],
            quality_score=data["quality_score"],
            duration_seconds=data["duration_seconds"],
            success=data.get("success", True),
            error_msg=data.get("error_msg", ""),
        )


@dataclass
class GPUInfo:
    """Information about an available GPU."""
    device_id: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: str = ""
    is_available: bool = True
    assigned_task: str = ""  # Current task assigned to this GPU


def detect_gpus() -> List[GPUInfo]:
    """Detect available GPUs and their VRAM."""
    gpus = []
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory // (1024 * 1024)
                free = (props.total_memory - torch.cuda.memory_allocated(i)) // (1024 * 1024)
                
                gpus.append(GPUInfo(
                    device_id=i,
                    name=props.name,
                    vram_total_mb=total,
                    vram_free_mb=free,
                    compute_capability=f"{props.major}.{props.minor}",
                ))
    except Exception as e:
        logger.warning(f"Could not detect GPUs: {e}")
    
    return gpus


class MultiGPUManager:
    """
    Manages multiple GPUs for distributed training.
    
    Use cases:
    - RTX 5090 (primary) for gaming + RTX 2080 (secondary) for training
    - Multiple GPUs for parallel training
    - Teacher on GPU 0, student on GPU 1
    """
    
    def __init__(self):
        self.gpus = detect_gpus()
        self.assignments: Dict[int, str] = {}  # gpu_id -> task name
        self._lock = threading.Lock()
    
    def get_available_gpu(self, min_vram_mb: int = 2000) -> Optional[GPUInfo]:
        """Get a GPU with enough free VRAM."""
        with self._lock:
            for gpu in self.gpus:
                if gpu.is_available and gpu.vram_free_mb >= min_vram_mb:
                    return gpu
        return None
    
    def assign_gpu(self, gpu_id: int, task_name: str):
        """Assign a GPU to a task."""
        with self._lock:
            if gpu_id < len(self.gpus):
                self.gpus[gpu_id].is_available = False
                self.gpus[gpu_id].assigned_task = task_name
                self.assignments[gpu_id] = task_name
                logger.info(f"Assigned GPU {gpu_id} ({self.gpus[gpu_id].name}) to {task_name}")
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU from its task."""
        with self._lock:
            if gpu_id < len(self.gpus):
                self.gpus[gpu_id].is_available = True
                self.gpus[gpu_id].assigned_task = ""
                self.assignments.pop(gpu_id, None)
                logger.info(f"Released GPU {gpu_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all GPUs."""
        return {
            "gpu_count": len(self.gpus),
            "gpus": [
                {
                    "id": g.device_id,
                    "name": g.name,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "available": g.is_available,
                    "task": g.assigned_task,
                }
                for g in self.gpus
            ],
            "assignments": self.assignments,
        }


class TeacherNode:
    """
    Teacher node for distributed training.
    
    Runs on a powerful machine, generates training data using a large model,
    and distributes training tasks to student nodes.
    """
    
    def __init__(
        self,
        port: int = 5100,
        teacher_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ):
        self.port = port
        self.model_id = teacher_model_id
        self.teacher = None  # TeacherAI instance
        self.gpu_manager = MultiGPUManager()
        
        # Task management
        self.pending_tasks: queue.Queue = queue.Queue()
        self.completed_results: List[TrainingResult] = []
        self._max_results = 100
        
        # Connected students
        self.students: Dict[str, Dict] = {}  # student_id -> info
        
        # Server
        self._server_socket = None
        self._running = False
        self._threads: List[threading.Thread] = []
    
    def load_teacher_model(self):
        """Load the teacher model for generating training data."""
        try:
            from scripts.teach_model import TeacherAI
            self.teacher = TeacherAI(model_id=self.model_id)
            self.teacher.load()
            logger.info(f"Teacher model loaded: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
    
    def generate_training_task(
        self,
        topic: str,
        count: int = 30,
        weaknesses: Optional[List[str]] = None,
    ) -> TrainingTask:
        """Generate a training task for students."""
        if self.teacher is None:
            self.load_teacher_model()
        
        # Generate training data
        if weaknesses:
            data = self.teacher.generate_targeted_training(weaknesses, count)
        else:
            data = self.teacher.generate_training_data(topic, count)
        
        task = TrainingTask(
            task_id=f"task_{int(time.time()*1000)}",
            training_data=data,
            target_loss=0.5,
            max_epochs=3,
        )
        
        logger.info(f"Generated task {task.task_id} with {len(data)} training pairs")
        return task
    
    def queue_task(self, task: TrainingTask):
        """Add a task to the queue for students."""
        self.pending_tasks.put(task)
        logger.info(f"Queued task {task.task_id}")
    
    def start(self):
        """Start the teacher server."""
        self._running = True
        
        # Start server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(('', self.port))
        self._server_socket.listen(5)
        
        logger.info(f"Teacher node started on port {self.port}")
        
        # Accept connections
        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        accept_thread.start()
        self._threads.append(accept_thread)
    
    def _accept_loop(self):
        """Accept incoming connections from students."""
        while self._running:
            try:
                self._server_socket.settimeout(1.0)
                client_socket, address = self._server_socket.accept()
                logger.info(f"Student connected from {address}")
                
                # Handle client in new thread
                client_thread = threading.Thread(
                    target=self._handle_student,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                self._threads.append(client_thread)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def _handle_student(self, sock: socket.socket, address):
        """Handle communication with a student node."""
        try:
            while self._running:
                # Receive message length
                length_data = sock.recv(4)
                if not length_data:
                    break
                
                msg_length = struct.unpack('>I', length_data)[0]
                
                # Receive message
                data = b''
                while len(data) < msg_length:
                    chunk = sock.recv(min(msg_length - len(data), 4096))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) != msg_length:
                    break
                
                # Process message
                message = json.loads(data.decode('utf-8'))
                response = self._process_message(message)
                
                # Send response
                resp_data = json.dumps(response).encode('utf-8')
                sock.send(struct.pack('>I', len(resp_data)))
                sock.send(resp_data)
                
        except Exception as e:
            logger.error(f"Student handler error: {e}")
        finally:
            sock.close()
    
    def _process_message(self, message: Dict) -> Dict:
        """Process a message from a student."""
        msg_type = message.get("type", "")
        
        if msg_type == "register":
            # Student registering
            student_id = message.get("student_id", f"student_{int(time.time())}")
            self.students[student_id] = {
                "registered_at": datetime.now().isoformat(),
                "info": message.get("info", {}),
            }
            return {"status": "ok", "message": f"Registered as {student_id}"}
        
        elif msg_type == "get_task":
            # Student requesting work
            try:
                task = self.pending_tasks.get_nowait()
                return {"status": "ok", "task": task.to_dict()}
            except queue.Empty:
                return {"status": "no_tasks"}
        
        elif msg_type == "submit_result":
            # Student submitting training result
            result = TrainingResult.from_dict(message.get("result", {}))
            self.completed_results.append(result)
            if len(self.completed_results) > self._max_results:
                self.completed_results.pop(0)
            
            logger.info(f"Received result from {result.student_id}: loss={result.final_loss:.4f}, quality={result.quality_score:.1f}")
            return {"status": "ok", "message": "Result received"}
        
        elif msg_type == "status":
            # Status request
            return {
                "status": "ok",
                "pending_tasks": self.pending_tasks.qsize(),
                "completed_results": len(self.completed_results),
                "connected_students": len(self.students),
            }
        
        return {"status": "error", "message": "Unknown message type"}
    
    def stop(self):
        """Stop the teacher server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        logger.info("Teacher node stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "running": self._running,
            "port": self.port,
            "pending_tasks": self.pending_tasks.qsize(),
            "completed_results": len(self.completed_results),
            "students": {
                k: v["registered_at"] for k, v in self.students.items()
            },
            "gpu_status": self.gpu_manager.get_status(),
        }


class StudentNode:
    """
    Student node for distributed training.
    
    Connects to a teacher node to receive training tasks,
    trains a local model, and reports results back.
    """
    
    def __init__(
        self,
        student_id: str = None,
        student_size: str = "small",
        teacher_url: str = "localhost:5100",
    ):
        self.student_id = student_id or f"student_{socket.gethostname()}_{int(time.time())}"
        self.student_size = student_size
        self.teacher_host, self.teacher_port = teacher_url.split(':')
        self.teacher_port = int(self.teacher_port)
        
        self.student = None  # StudentAI instance
        self._socket = None
        self._connected = False
        self.gpu_manager = MultiGPUManager()
    
    def connect(self, retries: int = 3) -> bool:
        """Connect to the teacher node."""
        for attempt in range(retries):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.teacher_host, self.teacher_port))
                
                # Register with teacher
                response = self._send_message({
                    "type": "register",
                    "student_id": self.student_id,
                    "info": {
                        "size": self.student_size,
                        "gpus": self.gpu_manager.get_status(),
                    }
                })
                
                if response.get("status") == "ok":
                    self._connected = True
                    logger.info(f"Connected to teacher at {self.teacher_host}:{self.teacher_port}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        
        return False
    
    def _send_message(self, message: Dict) -> Dict:
        """Send a message to teacher and get response."""
        data = json.dumps(message).encode('utf-8')
        
        # Send length + data
        self._socket.send(struct.pack('>I', len(data)))
        self._socket.send(data)
        
        # Receive response
        length_data = self._socket.recv(4)
        msg_length = struct.unpack('>I', length_data)[0]
        
        resp_data = b''
        while len(resp_data) < msg_length:
            chunk = self._socket.recv(min(msg_length - len(resp_data), 4096))
            if not chunk:
                break
            resp_data += chunk
        
        return json.loads(resp_data.decode('utf-8'))
    
    def get_task(self) -> Optional[TrainingTask]:
        """Request a training task from the teacher."""
        if not self._connected:
            return None
        
        response = self._send_message({"type": "get_task"})
        
        if response.get("status") == "ok" and "task" in response:
            return TrainingTask.from_dict(response["task"])
        
        return None
    
    def submit_result(self, result: TrainingResult):
        """Submit training result to teacher."""
        if not self._connected:
            return
        
        self._send_message({
            "type": "submit_result",
            "result": result.to_dict(),
        })
    
    def train_on_task(self, task: TrainingTask) -> TrainingResult:
        """Train on a single task."""
        start_time = time.time()
        
        try:
            # Load student model if needed
            if self.student is None:
                from scripts.teach_model import StudentAI
                self.student = StudentAI(size=self.student_size)
                
                # Use GPU if available
                gpu = self.gpu_manager.get_available_gpu(min_vram_mb=1000)
                if gpu:
                    self.student.set_use_gpu(True)
                    self.gpu_manager.assign_gpu(gpu.device_id, f"training_{task.task_id}")
                
                self.student.load()
            
            # Train
            final_loss = self.student.train_on_data(
                task.training_data,
                epochs=task.max_epochs
            )
            
            # Simple quality test (would use teacher evaluation in full impl)
            quality_score = max(0, 10 - final_loss * 5)
            
            # Save model
            self.student.save()
            
            duration = time.time() - start_time
            
            return TrainingResult(
                task_id=task.task_id,
                student_id=self.student_id,
                final_loss=final_loss,
                epochs_completed=task.max_epochs,
                quality_score=quality_score,
                duration_seconds=duration,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                task_id=task.task_id,
                student_id=self.student_id,
                final_loss=999.0,
                epochs_completed=0,
                quality_score=0.0,
                duration_seconds=time.time() - start_time,
                success=False,
                error_msg=str(e),
            )
    
    def train_loop(self, max_tasks: int = 0, poll_interval: float = 5.0):
        """
        Main training loop.
        
        Args:
            max_tasks: Maximum tasks to complete (0 = unlimited)
            poll_interval: Seconds between task requests
        """
        tasks_completed = 0
        
        logger.info(f"Starting training loop (max_tasks={max_tasks})")
        
        while True:
            # Get next task
            task = self.get_task()
            
            if task:
                logger.info(f"Processing task {task.task_id} with {len(task.training_data)} pairs")
                
                # Train
                result = self.train_on_task(task)
                
                # Submit result
                self.submit_result(result)
                
                tasks_completed += 1
                logger.info(f"Completed task {task.task_id}: loss={result.final_loss:.4f}")
                
                if max_tasks > 0 and tasks_completed >= max_tasks:
                    logger.info(f"Reached max tasks limit ({max_tasks})")
                    break
            else:
                # No tasks available, wait
                logger.debug(f"No tasks available, waiting {poll_interval}s")
                time.sleep(poll_interval)
    
    def disconnect(self):
        """Disconnect from teacher."""
        if self._socket:
            self._socket.close()
        self._connected = False
        logger.info("Disconnected from teacher")


def run_distributed_training(
    mode: str,  # "teacher" or "student"
    **kwargs,
):
    """
    Run distributed training.
    
    Args:
        mode: "teacher" to run as teacher, "student" to run as student
        kwargs: Additional arguments for the node
    """
    if mode == "teacher":
        teacher = TeacherNode(
            port=kwargs.get("port", 5100),
            teacher_model_id=kwargs.get("model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
        )
        teacher.start()
        
        # Generate initial tasks
        topics = kwargs.get("topics", ["general conversation", "helpful responses"])
        for topic in topics:
            task = teacher.generate_training_task(topic, count=30)
            teacher.queue_task(task)
        
        print(f"Teacher running on port {teacher.port}")
        print(f"Queued {len(topics)} initial tasks")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(10)
                stats = teacher.get_stats()
                print(f"Status: {stats['pending_tasks']} pending, {stats['completed_results']} completed")
        except KeyboardInterrupt:
            teacher.stop()
    
    elif mode == "student":
        student = StudentNode(
            teacher_url=kwargs.get("teacher_url", "localhost:5100"),
            student_size=kwargs.get("size", "small"),
        )
        
        if student.connect():
            student.train_loop(max_tasks=kwargs.get("max_tasks", 0))
            student.disconnect()
        else:
            print("Failed to connect to teacher")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed AI Training")
    parser.add_argument("mode", choices=["teacher", "student"], help="Run as teacher or student")
    parser.add_argument("--port", type=int, default=5100, help="Teacher port")
    parser.add_argument("--teacher-url", default="localhost:5100", help="Teacher URL for students")
    parser.add_argument("--size", default="small", help="Student model size")
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks (0=unlimited)")
    
    args = parser.parse_args()
    
    run_distributed_training(
        mode=args.mode,
        port=args.port,
        teacher_url=args.teacher_url,
        size=args.size,
        max_tasks=args.max_tasks,
    )
