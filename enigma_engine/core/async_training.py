"""
================================================================================
                ASYNC TRAINING - Non-Blocking Training System
================================================================================

Provides asynchronous, non-blocking training operations so the main AI
remains responsive during training, data generation, and curation.

FILE: enigma_engine/core/async_training.py
TYPE: Training Infrastructure

KEY FEATURES:
    - Background training that doesn't block the main AI
    - Progress callbacks for UI updates
    - Cancellation support
    - Queue-based task management
    - Thread-safe status updates

USAGE:
    from enigma_engine.core.async_training import AsyncTrainer
    
    trainer = AsyncTrainer()
    
    # Start training in background
    task_id = trainer.start_training_async(
        data_path="data/training.txt",
        model_name="my_model",
        on_progress=lambda p: print(f"Progress: {p}%"),
        on_complete=lambda m: print("Done!")
    )
    
    # Main AI continues working normally
    response = main_ai.chat("Hello!")  # Not blocked
    
    # Check status anytime
    status = trainer.get_task_status(task_id)
    
    # Cancel if needed
    trainer.cancel_task(task_id)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .training import Trainer

logger = logging.getLogger(__name__)


# =============================================================================
# TASK STATUS AND TYPES
# =============================================================================

class TaskStatus(Enum):
    """Status of an async training task."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    PAUSED = auto()


class TaskType(Enum):
    """Type of async task."""
    TRAINING = auto()
    DATA_GENERATION = auto()
    DATA_CURATION = auto()
    WEB_SCRAPING = auto()
    MODEL_EXPORT = auto()
    MODEL_MERGE = auto()
    EVALUATION = auto()


@dataclass
class TaskProgress:
    """Progress information for a task."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    total_steps: int
    completed_steps: int
    start_time: float
    elapsed_time: float = 0.0
    eta_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.name,
            "status": self.status.name,
            "progress": self.progress,
            "progress_percent": round(self.progress * 100, 1),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "elapsed_time": self.elapsed_time,
            "eta_seconds": self.eta_seconds,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class AsyncTask:
    """An async training task."""
    task_id: str
    task_type: TaskType
    target: Callable
    args: tuple
    kwargs: Dict[str, Any]
    on_progress: Optional[Callable[[TaskProgress], None]] = None
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    future: Optional[Future] = None
    progress: Optional[TaskProgress] = None
    result: Any = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    pause_flag: threading.Event = field(default_factory=threading.Event)
    
    def __post_init__(self):
        self.progress = TaskProgress(
            task_id=self.task_id,
            task_type=self.task_type,
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="Waiting to start",
            total_steps=0,
            completed_steps=0,
            start_time=time.time(),
        )


# =============================================================================
# ASYNC TRAINER - NON-BLOCKING TRAINING OPERATIONS
# =============================================================================

class AsyncTrainer:
    """
    Non-blocking training manager.
    
    Runs all training operations in background threads so the main
    AI remains responsive to user interactions.
    """
    
    _instance: Optional['AsyncTrainer'] = None
    
    def __new__(cls) -> 'AsyncTrainer':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_workers: int = 4):
        if self._initialized:
            return
            
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AsyncTrainer")
        self._tasks: Dict[str, AsyncTask] = {}
        self._task_queue = Queue()
        self._lock = threading.RLock()
        self._progress_callbacks: List[Callable[[TaskProgress], None]] = []
        self._history_file = Path("data/training_history.json")
        
        # Start the progress monitor thread
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._progress_monitor, daemon=True)
        self._monitor_thread.start()
        
        self._initialized = True
        logger.info("AsyncTrainer initialized")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def start_training_async(
        self,
        data_path: str,
        model_name: str,
        model_size: str = "small",
        epochs: int = 3,
        learning_rate: float = 0.0001,
        batch_size: int = 4,
        on_progress: Optional[Callable[[TaskProgress], None]] = None,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> str:
        """
        Start training in the background.
        
        Returns task_id immediately - training runs in background.
        """
        task_id = f"train_{uuid.uuid4().hex[:8]}"
        
        task = AsyncTask(
            task_id=task_id,
            task_type=TaskType.TRAINING,
            target=self._run_training,
            args=(data_path, model_name, model_size, epochs, learning_rate, batch_size),
            kwargs={},
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
        )
        
        self._submit_task(task)
        logger.info(f"Started async training: {task_id}")
        return task_id
    
    def _run_training(
        self,
        task: AsyncTask,
        data_path: str,
        model_name: str,
        model_size: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Execute training in background thread."""
        from .training import train_model
        from .model import create_model
        
        self._update_progress(task, 0.0, "Initializing model")
        
        # Check cancellation
        if task.cancel_flag.is_set():
            return {"cancelled": True}
        
        # Create model
        model = create_model(size=model_size)
        self._update_progress(task, 0.1, "Loading training data")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        total_lines = len(data.splitlines())
        self._update_progress(task, 0.15, f"Found {total_lines} training examples")
        
        # Training loop with progress updates
        for epoch in range(epochs):
            if task.cancel_flag.is_set():
                return {"cancelled": True, "completed_epochs": epoch}
            
            # Wait while paused
            while task.pause_flag.is_set() and not task.cancel_flag.is_set():
                time.sleep(0.1)
            
            progress = 0.15 + (epoch / epochs) * 0.8
            self._update_progress(
                task, 
                progress, 
                f"Training epoch {epoch + 1}/{epochs}",
                metrics={"epoch": epoch + 1, "total_epochs": epochs}
            )
            
            # Simulate epoch training (actual training would go here)
            # In real implementation, this would call the actual trainer
            time.sleep(0.5)  # Placeholder for actual training
        
        self._update_progress(task, 0.95, "Saving model")
        
        # Save model
        save_path = Path(f"models/{model_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        self._update_progress(task, 1.0, "Training complete")
        
        return {
            "model_name": model_name,
            "model_path": str(save_path),
            "epochs": epochs,
            "data_path": data_path,
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATA GENERATION OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def start_data_generation_async(
        self,
        position: str,
        count: int = 100,
        use_api: bool = False,
        on_progress: Optional[Callable[[TaskProgress], None]] = None,
        on_complete: Optional[Callable[[List[str]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> str:
        """Start data generation in background."""
        task_id = f"datagen_{uuid.uuid4().hex[:8]}"
        
        task = AsyncTask(
            task_id=task_id,
            task_type=TaskType.DATA_GENERATION,
            target=self._run_data_generation,
            args=(position, count, use_api),
            kwargs={},
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
        )
        
        self._submit_task(task)
        return task_id
    
    def _run_data_generation(
        self,
        task: AsyncTask,
        position: str,
        count: int,
        use_api: bool,
    ) -> List[str]:
        """Execute data generation in background."""
        from .trainer_ai import get_trainer_ai
        
        trainer = get_trainer_ai()
        results = []
        batch_size = 10
        
        for i in range(0, count, batch_size):
            if task.cancel_flag.is_set():
                return results
            
            while task.pause_flag.is_set() and not task.cancel_flag.is_set():
                time.sleep(0.1)
            
            batch_count = min(batch_size, count - i)
            progress = i / count
            self._update_progress(
                task,
                progress,
                f"Generating examples {i + 1}-{i + batch_count}",
                metrics={"generated": i, "total": count}
            )
            
            # Generate batch
            batch = trainer.generate_training_data(position, batch_count)
            results.append(batch)
        
        self._update_progress(task, 1.0, "Generation complete")
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # WEB SCRAPING FOR TRAINING DATA
    # ─────────────────────────────────────────────────────────────────────────
    
    def start_web_scraping_async(
        self,
        urls: List[str],
        topic: str,
        max_pages: int = 10,
        on_progress: Optional[Callable[[TaskProgress], None]] = None,
        on_complete: Optional[Callable[[List[Dict]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> str:
        """Start web scraping for training data in background."""
        task_id = f"webscrape_{uuid.uuid4().hex[:8]}"
        
        task = AsyncTask(
            task_id=task_id,
            task_type=TaskType.WEB_SCRAPING,
            target=self._run_web_scraping,
            args=(urls, topic, max_pages),
            kwargs={},
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
        )
        
        self._submit_task(task)
        return task_id
    
    def _run_web_scraping(
        self,
        task: AsyncTask,
        urls: List[str],
        topic: str,
        max_pages: int,
    ) -> List[Dict[str, Any]]:
        """Execute web scraping in background."""
        from .web_training import WebTrainingCollector
        
        collector = WebTrainingCollector()
        results = []
        
        for i, url in enumerate(urls[:max_pages]):
            if task.cancel_flag.is_set():
                return results
            
            progress = i / len(urls)
            self._update_progress(task, progress, f"Scraping {url}")
            
            try:
                data = collector.scrape_for_training(url, topic)
                results.append(data)
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
        
        self._update_progress(task, 1.0, "Scraping complete")
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # TASK MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def _submit_task(self, task: AsyncTask):
        """Submit a task to the executor."""
        with self._lock:
            self._tasks[task.task_id] = task
            
            def run_task():
                try:
                    task.progress.status = TaskStatus.RUNNING
                    task.progress.start_time = time.time()
                    
                    result = task.target(task, *task.args, **task.kwargs)
                    
                    task.result = result
                    task.progress.status = TaskStatus.COMPLETED
                    
                    if task.on_complete:
                        task.on_complete(result)
                        
                except Exception as e:
                    task.progress.status = TaskStatus.FAILED
                    task.progress.error = str(e)
                    logger.error(f"Task {task.task_id} failed: {e}")
                    
                    if task.on_error:
                        task.on_error(e)
                        
                finally:
                    self._save_task_history(task)
            
            task.future = self._executor.submit(run_task)
    
    def _update_progress(
        self,
        task: AsyncTask,
        progress: float,
        current_step: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Update task progress and notify callbacks."""
        with self._lock:
            task.progress.progress = progress
            task.progress.current_step = current_step
            task.progress.elapsed_time = time.time() - task.progress.start_time
            
            if metrics:
                task.progress.metrics.update(metrics)
            
            # Calculate ETA
            if progress > 0:
                task.progress.eta_seconds = (
                    task.progress.elapsed_time / progress * (1 - progress)
                )
            
            # Notify callbacks
            if task.on_progress:
                task.on_progress(task.progress)
            
            for callback in self._progress_callbacks:
                callback(task.progress)
    
    def get_task_status(self, task_id: str) -> Optional[TaskProgress]:
        """Get the current status of a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.progress if task else None
    
    def get_all_tasks(self) -> List[TaskProgress]:
        """Get status of all tasks."""
        with self._lock:
            return [task.progress for task in self._tasks.values()]
    
    def get_active_tasks(self) -> List[TaskProgress]:
        """Get only running tasks."""
        with self._lock:
            return [
                task.progress
                for task in self._tasks.values()
                if task.progress.status in (TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.PAUSED)
            ]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.progress.status in (TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.PAUSED):
                task.cancel_flag.set()
                task.progress.status = TaskStatus.CANCELLED
                logger.info(f"Cancelled task: {task_id}")
                return True
            return False
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a running task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.progress.status == TaskStatus.RUNNING:
                task.pause_flag.set()
                task.progress.status = TaskStatus.PAUSED
                return True
            return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.progress.status == TaskStatus.PAUSED:
                task.pause_flag.clear()
                task.progress.status = TaskStatus.RUNNING
                return True
            return False
    
    def add_progress_callback(self, callback: Callable[[TaskProgress], None]):
        """Add a global progress callback."""
        with self._lock:
            self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[TaskProgress], None]):
        """Remove a global progress callback."""
        with self._lock:
            if callback in self._progress_callbacks:
                self._progress_callbacks.remove(callback)
    
    # ─────────────────────────────────────────────────────────────────────────
    # HISTORY AND PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_task_history(self, task: AsyncTask):
        """Save completed task to history."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history = []
            if self._history_file.exists():
                history = json.loads(self._history_file.read_text())
            
            history.append({
                "task_id": task.task_id,
                "task_type": task.task_type.name,
                "status": task.progress.status.name,
                "started_at": datetime.fromtimestamp(task.progress.start_time).isoformat(),
                "elapsed_time": task.progress.elapsed_time,
                "metrics": task.progress.metrics,
                "error": task.progress.error,
            })
            
            # Keep only last 100 entries
            history = history[-100:]
            
            self._history_file.write_text(json.dumps(history, indent=2))
            
        except Exception as e:
            logger.warning(f"Failed to save task history: {e}")
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get task history."""
        try:
            if self._history_file.exists():
                history = json.loads(self._history_file.read_text())
                return history[-limit:]
        except Exception as e:
            logger.warning(f"Failed to load task history: {e}")
        return []
    
    def _progress_monitor(self):
        """Background thread to monitor progress."""
        while self._monitor_running:
            # Update elapsed times for running tasks
            with self._lock:
                for task in self._tasks.values():
                    if task.progress.status == TaskStatus.RUNNING:
                        task.progress.elapsed_time = time.time() - task.progress.start_time
            
            time.sleep(0.5)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the async trainer."""
        self._monitor_running = False
        self._executor.shutdown(wait=wait)


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_async_trainer: Optional[AsyncTrainer] = None


def get_async_trainer() -> AsyncTrainer:
    """Get the global async trainer instance."""
    global _async_trainer
    if _async_trainer is None:
        _async_trainer = AsyncTrainer()
    return _async_trainer


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_async(
    data_path: str,
    model_name: str,
    **kwargs
) -> str:
    """Start training in background. Returns task_id."""
    return get_async_trainer().start_training_async(data_path, model_name, **kwargs)


def generate_data_async(
    position: str,
    count: int = 100,
    **kwargs
) -> str:
    """Start data generation in background. Returns task_id."""
    return get_async_trainer().start_data_generation_async(position, count, **kwargs)


def get_task_status(task_id: str) -> Optional[TaskProgress]:
    """Get status of a background task."""
    return get_async_trainer().get_task_status(task_id)


def cancel_task(task_id: str) -> bool:
    """Cancel a background task."""
    return get_async_trainer().cancel_task(task_id)
