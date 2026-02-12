"""
Scheduled Training for Enigma AI Engine

Train models overnight or at scheduled times.

Features:
- Cron-like scheduling
- Training queue
- Resource-aware scheduling
- Email/webhook notifications
- Resume on failure
- Priority system

Usage:
    from enigma_engine.core.scheduled_training import TrainingScheduler, get_scheduler
    
    scheduler = get_scheduler()
    
    # Schedule a training job
    scheduler.schedule(
        "my_training",
        dataset="data/training.txt",
        model_size="small",
        when="tonight at 2am"
    )
    
    # Or with cron syntax
    scheduler.schedule_cron(
        "weekly_training",
        dataset="data/weekly.txt",
        cron="0 2 * * 0"  # Every Sunday at 2am
    )
    
    # Start scheduler daemon
    scheduler.start()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try imports
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    schedule = None


class JobStatus(Enum):
    """Training job status."""
    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class TrainingJob:
    """A scheduled training job."""
    # Identity
    id: str
    name: str
    
    # Training params
    dataset: str
    model_size: str = "small"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.0001
    
    # Model paths
    base_model: Optional[str] = None  # If continuing from existing
    output_dir: Optional[str] = None
    
    # Scheduling
    scheduled_time: Optional[datetime] = None
    cron_expression: Optional[str] = None
    repeat: bool = False
    
    # Priority
    priority: JobPriority = JobPriority.NORMAL
    
    # State
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    current_epoch: int = 0
    current_loss: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    # Error handling
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3
    
    # Notifications
    notify_on_complete: bool = True
    notify_on_error: bool = True
    webhook_url: Optional[str] = None
    email: Optional[str] = None
    
    # Extra config
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "dataset": self.dataset,
            "model_size": self.model_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "base_model": self.base_model,
            "output_dir": self.output_dir,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "cron_expression": self.cron_expression,
            "repeat": self.repeat,
            "priority": self.priority.value,
            "status": self.status.name,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "notify_on_complete": self.notify_on_complete,
            "webhook_url": self.webhook_url,
            "email": self.email,
            "extra_config": self.extra_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        """Create from dictionary."""
        job = cls(
            id=data["id"],
            name=data["name"],
            dataset=data["dataset"],
            model_size=data.get("model_size", "small"),
            epochs=data.get("epochs", 10),
            batch_size=data.get("batch_size", 32),
            learning_rate=data.get("learning_rate", 0.0001),
        )
        
        job.base_model = data.get("base_model")
        job.output_dir = data.get("output_dir")
        job.cron_expression = data.get("cron_expression")
        job.repeat = data.get("repeat", False)
        job.notify_on_complete = data.get("notify_on_complete", True)
        job.webhook_url = data.get("webhook_url")
        job.email = data.get("email")
        job.extra_config = data.get("extra_config", {})
        
        if data.get("scheduled_time"):
            job.scheduled_time = datetime.fromisoformat(data["scheduled_time"])
        if data.get("created_at"):
            job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("priority"):
            job.priority = JobPriority(data["priority"])
        if data.get("status"):
            job.status = JobStatus[data["status"]]
        
        return job


class TrainingScheduler:
    """
    Manages scheduled training jobs.
    """
    
    def __init__(
        self,
        jobs_dir: Optional[Path] = None,
        max_concurrent: int = 1
    ):
        """
        Initialize scheduler.
        
        Args:
            jobs_dir: Directory for job persistence
            max_concurrent: Max concurrent training jobs
        """
        self._jobs_dir = jobs_dir or Path.home() / ".enigma_engine" / "scheduled_jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        
        self._max_concurrent = max_concurrent
        
        # Job storage
        self._jobs: Dict[str, TrainingJob] = {}
        self._job_queue: List[str] = []  # Job IDs in priority order
        
        # State
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_job_start: Optional[Callable[[TrainingJob], None]] = None
        self._on_job_complete: Optional[Callable[[TrainingJob], None]] = None
        self._on_job_error: Optional[Callable[[TrainingJob, Exception], None]] = None
        
        # Currently running
        self._active_jobs: List[str] = []
        
        # Load persisted jobs
        self._load_jobs()
    
    def schedule(
        self,
        name: str,
        dataset: str,
        model_size: str = "small",
        when: str = "now",
        **kwargs
    ) -> TrainingJob:
        """
        Schedule a training job.
        
        Args:
            name: Job name
            dataset: Path to training data
            model_size: Model size preset
            when: Natural language time ("tonight at 2am", "in 1 hour", "now")
            **kwargs: Additional training config
            
        Returns:
            Created job
        """
        # Parse time
        scheduled_time = self._parse_time(when)
        
        # Create job
        job = TrainingJob(
            id=self._generate_id(),
            name=name,
            dataset=dataset,
            model_size=model_size,
            scheduled_time=scheduled_time,
            **kwargs
        )
        
        if scheduled_time:
            job.status = JobStatus.SCHEDULED
        
        # Add to queue
        self._add_job(job)
        
        logger.info(f"Scheduled job: {name} at {scheduled_time or 'immediately'}")
        return job
    
    def schedule_cron(
        self,
        name: str,
        dataset: str,
        cron: str,
        **kwargs
    ) -> TrainingJob:
        """
        Schedule a recurring training job with cron syntax.
        
        Args:
            name: Job name
            dataset: Path to training data
            cron: Cron expression (e.g., "0 2 * * *" for 2am daily)
            **kwargs: Additional training config
            
        Returns:
            Created job
        """
        job = TrainingJob(
            id=self._generate_id(),
            name=name,
            dataset=dataset,
            cron_expression=cron,
            repeat=True,
            status=JobStatus.SCHEDULED,
            **kwargs
        )
        
        # Calculate next run
        job.scheduled_time = self._next_cron_time(cron)
        
        self._add_job(job)
        
        logger.info(f"Scheduled recurring job: {name} ({cron})")
        return job
    
    def _parse_time(self, when: str) -> Optional[datetime]:
        """Parse natural language time."""
        when = when.lower().strip()
        now = datetime.now()
        
        if when in ("now", "immediately", "asap"):
            return None  # Run immediately
        
        # Parse relative times
        if when.startswith("in "):
            parts = when[3:].split()
            if len(parts) >= 2:
                amount = int(parts[0])
                unit = parts[1].rstrip('s')  # Remove plural
                
                if unit in ("minute", "min"):
                    return now + timedelta(minutes=amount)
                elif unit in ("hour", "hr"):
                    return now + timedelta(hours=amount)
                elif unit == "day":
                    return now + timedelta(days=amount)
        
        # Parse specific times
        if "tonight" in when:
            # Default to 2am
            hour = 2
            if "at" in when:
                time_part = when.split("at")[1].strip()
                hour = self._parse_hour(time_part)
            
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target
        
        if "tomorrow" in when:
            hour = 9  # Default
            if "at" in when:
                time_part = when.split("at")[1].strip()
                hour = self._parse_hour(time_part)
            
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            target += timedelta(days=1)
            return target
        
        # Try parsing as time
        if "at" in when:
            time_part = when.split("at")[1].strip()
            hour = self._parse_hour(time_part)
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target
        
        return None
    
    def _parse_hour(self, time_str: str) -> int:
        """Parse hour from time string."""
        time_str = time_str.lower().strip()
        
        # Handle am/pm
        is_pm = "pm" in time_str
        time_str = time_str.replace("am", "").replace("pm", "").strip()
        
        # Extract hour
        if ":" in time_str:
            hour = int(time_str.split(":")[0])
        else:
            hour = int(time_str)
        
        if is_pm and hour < 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0
        
        return hour
    
    def _next_cron_time(self, cron: str) -> datetime:
        """Calculate next run time for cron expression."""
        # Simple cron parsing (minute hour day month weekday)
        parts = cron.split()
        if len(parts) != 5:
            logger.warning(f"Invalid cron expression: {cron}")
            return datetime.now() + timedelta(hours=1)
        
        minute, hour, day, month, weekday = parts
        
        now = datetime.now()
        target = now.replace(second=0, microsecond=0)
        
        # Set hour and minute
        if hour != "*":
            target = target.replace(hour=int(hour))
        if minute != "*":
            target = target.replace(minute=int(minute))
        
        # If time has passed, move to next occurrence
        if target <= now:
            if weekday != "*":
                # Move to next matching weekday
                target_weekday = int(weekday)
                days_ahead = target_weekday - target.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target += timedelta(days=days_ahead)
            else:
                target += timedelta(days=1)
        
        return target
    
    def _generate_id(self) -> str:
        """Generate unique job ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _add_job(self, job: TrainingJob) -> None:
        """Add job to queue."""
        with self._lock:
            self._jobs[job.id] = job
            
            # Insert in priority order
            inserted = False
            for i, job_id in enumerate(self._job_queue):
                existing = self._jobs[job_id]
                if job.priority.value > existing.priority.value:
                    self._job_queue.insert(i, job.id)
                    inserted = True
                    break
            
            if not inserted:
                self._job_queue.append(job.id)
            
            self._save_jobs()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            
            job = self._jobs[job_id]
            
            if job.status == JobStatus.RUNNING:
                # Can't cancel running job here
                return False
            
            job.status = JobStatus.CANCELLED
            
            if job_id in self._job_queue:
                self._job_queue.remove(job_id)
            
            self._save_jobs()
            return True
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def get_jobs(self, status: Optional[JobStatus] = None) -> List[TrainingJob]:
        """Get all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.scheduled_time or j.created_at)
    
    def start(self, blocking: bool = False) -> None:
        """
        Start the scheduler daemon.
        
        Args:
            blocking: If True, run in current thread
        """
        self._running = True
        
        if blocking:
            self._scheduler_loop()
        else:
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True
            )
            self._scheduler_thread.start()
            logger.info("Training scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler daemon."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self._check_jobs()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def _check_jobs(self) -> None:
        """Check and run due jobs."""
        now = datetime.now()
        
        with self._lock:
            # Check if we can run more jobs
            if len(self._active_jobs) >= self._max_concurrent:
                return
            
            # Find due jobs
            for job_id in self._job_queue[:]:
                job = self._jobs.get(job_id)
                if not job:
                    continue
                
                if job.status == JobStatus.CANCELLED:
                    self._job_queue.remove(job_id)
                    continue
                
                # Check if due
                is_due = (
                    job.scheduled_time is None or
                    job.scheduled_time <= now
                )
                
                if is_due and job.status in (JobStatus.PENDING, JobStatus.SCHEDULED):
                    # Start job
                    if len(self._active_jobs) < self._max_concurrent:
                        self._start_job(job)
                        self._job_queue.remove(job_id)
    
    def _start_job(self, job: TrainingJob) -> None:
        """Start a training job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self._active_jobs.append(job.id)
        
        # Callback
        if self._on_job_start:
            self._on_job_start(job)
        
        # Run in thread
        thread = threading.Thread(
            target=self._run_training,
            args=(job,),
            daemon=True
        )
        thread.start()
    
    def _run_training(self, job: TrainingJob) -> None:
        """Run the actual training."""
        try:
            from enigma_engine.core.training import train_model, TrainingConfig
            from enigma_engine.core.model import create_model
            
            logger.info(f"Starting training job: {job.name}")
            
            # Create model
            model = create_model(job.model_size, checkpoint=job.base_model)
            
            # Configure training
            config = TrainingConfig(
                epochs=job.epochs,
                batch_size=job.batch_size,
                learning_rate=job.learning_rate,
                **job.extra_config
            )
            
            # Progress callback
            def on_progress(epoch, total_epochs, loss) -> None:
                job.current_epoch = epoch
                job.progress = epoch / total_epochs
                job.current_loss = loss
                self._save_jobs()
            
            # Train
            train_model(
                model=model,
                data_file=job.dataset,
                config=config,
                output_dir=job.output_dir,
                progress_callback=on_progress
            )
            
            # Success
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            
            logger.info(f"Training job completed: {job.name}")
            
            # Callback
            if self._on_job_complete:
                self._on_job_complete(job)
            
            # Notifications
            self._send_notification(job, "completed")
            
        except Exception as e:
            logger.error(f"Training job failed: {job.name} - {e}")
            job.error_message = str(e)
            job.retry_count += 1
            
            if job.retry_count >= job.max_retries:
                job.status = JobStatus.FAILED
            else:
                job.status = JobStatus.SCHEDULED
                job.scheduled_time = datetime.now() + timedelta(minutes=30)
                self._job_queue.append(job.id)
            
            # Callback
            if self._on_job_error:
                self._on_job_error(job, e)
            
            # Notifications
            if job.notify_on_error:
                self._send_notification(job, "failed")
        
        finally:
            if job.id in self._active_jobs:
                self._active_jobs.remove(job.id)
            
            # Reschedule if recurring
            if job.repeat and job.cron_expression and job.status == JobStatus.COMPLETED:
                job.scheduled_time = self._next_cron_time(job.cron_expression)
                job.status = JobStatus.SCHEDULED
                job.progress = 0
                self._job_queue.append(job.id)
            
            self._save_jobs()
    
    def _send_notification(self, job: TrainingJob, event: str) -> None:
        """Send notification for job event."""
        if not job.notify_on_complete and event == "completed":
            return
        
        message = f"Training job '{job.name}' {event}"
        if event == "failed":
            message += f": {job.error_message}"
        
        # Webhook
        if job.webhook_url:
            try:
                import requests
                requests.post(job.webhook_url, json={
                    "event": event,
                    "job": job.to_dict(),
                    "message": message
                }, timeout=10)
            except Exception as e:
                logger.warning(f"Webhook notification failed: {e}")
        
        logger.info(f"Notification: {message}")
    
    def _save_jobs(self) -> None:
        """Save jobs to disk."""
        try:
            jobs_file = self._jobs_dir / "jobs.json"
            jobs_data = [job.to_dict() for job in self._jobs.values()]
            with open(jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")
    
    def _load_jobs(self) -> None:
        """Load jobs from disk."""
        try:
            jobs_file = self._jobs_dir / "jobs.json"
            if jobs_file.exists():
                with open(jobs_file) as f:
                    jobs_data = json.load(f)
                
                for data in jobs_data:
                    job = TrainingJob.from_dict(data)
                    # Only reload pending/scheduled jobs
                    if job.status in (JobStatus.PENDING, JobStatus.SCHEDULED):
                        self._jobs[job.id] = job
                        self._job_queue.append(job.id)
                
                logger.info(f"Loaded {len(self._jobs)} scheduled jobs")
                
        except Exception as e:
            logger.warning(f"Failed to load jobs: {e}")
    
    def on_job_start(self, callback: Callable[[TrainingJob], None]) -> None:
        """Set callback for job start."""
        self._on_job_start = callback
    
    def on_job_complete(self, callback: Callable[[TrainingJob], None]) -> None:
        """Set callback for job completion."""
        self._on_job_complete = callback
    
    def on_job_error(self, callback: Callable[[TrainingJob, Exception], None]) -> None:
        """Set callback for job errors."""
        self._on_job_error = callback


# Global instance
_scheduler: Optional[TrainingScheduler] = None


def get_scheduler() -> TrainingScheduler:
    """Get or create the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TrainingScheduler()
    return _scheduler


def schedule_training(
    name: str,
    dataset: str,
    when: str = "tonight",
    **kwargs
) -> TrainingJob:
    """Quick function to schedule training."""
    return get_scheduler().schedule(name, dataset, when=when, **kwargs)
