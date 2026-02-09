"""
Training History System for Enigma AI Engine

Tracks all training runs with metrics for analysis and comparison.

Usage:
    from enigma_engine.core.training_history import TrainingHistory, get_training_history
    
    history = get_training_history()
    
    # Record a training run
    run_id = history.start_run(
        model_name="my_model",
        model_size="small",
        training_config={"epochs": 10, "batch_size": 4}
    )
    
    # Log metrics during training
    history.log_metric(run_id, "loss", 0.5, step=100)
    history.log_metric(run_id, "accuracy", 0.85, step=100)
    
    # Complete the run
    history.complete_run(run_id, success=True)
    
    # Get history
    runs = history.get_all_runs()
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetric:
    """A single metric point."""
    name: str
    value: float
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrainingRun:
    """Record of a single training run."""
    id: str
    model_name: str
    model_size: str
    started_at: str
    completed_at: Optional[str] = None
    success: bool = False
    duration_seconds: float = 0.0
    
    # Configuration
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    training_files: List[str] = field(default_factory=list)
    nsfw_enabled: bool = False
    
    # Metrics over time
    metrics: Dict[str, List[TrainingMetric]] = field(default_factory=dict)
    
    # Final stats
    final_loss: Optional[float] = None
    total_steps: int = 0
    total_tokens: int = 0
    
    # Errors
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert metrics to serializable format
        data["metrics"] = {
            name: [asdict(m) for m in metrics]
            for name, metrics in self.metrics.items()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingRun':
        """Create from dictionary."""
        # Convert metrics back
        metrics = {}
        for name, metric_list in data.get("metrics", {}).items():
            metrics[name] = [
                TrainingMetric(**m) for m in metric_list
            ]
        data["metrics"] = metrics
        return cls(**data)


class TrainingHistory:
    """
    Persistent training history storage.
    
    Stores all training runs with:
    - Configuration used
    - Metrics over time (loss, accuracy, etc.)
    - Success/failure status
    - Duration and resource usage
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        from ..config import CONFIG
        
        self.storage_path = storage_path or (
            Path(CONFIG.get("data_dir", "data")) / "training_history"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._runs: Dict[str, TrainingRun] = {}
        self._load_history()
    
    def _load_history(self):
        """Load existing history from disk."""
        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                for run_data in data.get("runs", []):
                    run = TrainingRun.from_dict(run_data)
                    self._runs[run.id] = run
                logger.info(f"Loaded {len(self._runs)} training runs from history")
            except Exception as e:
                logger.warning(f"Could not load training history: {e}")
    
    def _save_history(self):
        """Save history to disk."""
        history_file = self.storage_path / "history.json"
        try:
            data = {
                "version": 1,
                "runs": [run.to_dict() for run in self._runs.values()]
            }
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save training history: {e}")
    
    def start_run(
        self,
        model_name: str,
        model_size: str,
        training_config: Dict[str, Any]
    ) -> str:
        """
        Start a new training run.
        
        Args:
            model_name: Name of the model being trained
            model_size: Size preset (tiny, small, medium, etc.)
            training_config: Training configuration dict
            
        Returns:
            Run ID for tracking
        """
        run_id = str(uuid.uuid4())[:8]
        
        run = TrainingRun(
            id=run_id,
            model_name=model_name,
            model_size=model_size,
            started_at=datetime.now().isoformat(),
            epochs=training_config.get("epochs", 0),
            batch_size=training_config.get("batch_size", 0),
            learning_rate=training_config.get("learning_rate", 0.0),
            training_files=training_config.get("training_files", []),
            nsfw_enabled=training_config.get("nsfw_enabled", False)
        )
        
        self._runs[run_id] = run
        self._save_history()
        
        logger.info(f"Started training run {run_id} for {model_name}")
        return run_id
    
    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        step: int
    ):
        """
        Log a metric value during training.
        
        Args:
            run_id: Training run ID
            metric_name: Name of metric (loss, accuracy, etc.)
            value: Metric value
            step: Training step number
        """
        if run_id not in self._runs:
            logger.warning(f"Unknown run ID: {run_id}")
            return
        
        run = self._runs[run_id]
        
        if metric_name not in run.metrics:
            run.metrics[metric_name] = []
        
        run.metrics[metric_name].append(TrainingMetric(
            name=metric_name,
            value=value,
            step=step
        ))
        
        # Update final loss if this is the loss metric
        if metric_name == "loss":
            run.final_loss = value
            run.total_steps = step
    
    def complete_run(
        self,
        run_id: str,
        success: bool,
        error_message: Optional[str] = None,
        total_tokens: int = 0
    ):
        """
        Mark a training run as complete.
        
        Args:
            run_id: Training run ID
            success: Whether training completed successfully
            error_message: Error message if failed
            total_tokens: Total tokens processed
        """
        if run_id not in self._runs:
            logger.warning(f"Unknown run ID: {run_id}")
            return
        
        run = self._runs[run_id]
        run.completed_at = datetime.now().isoformat()
        run.success = success
        run.error_message = error_message
        run.total_tokens = total_tokens
        
        # Calculate duration
        started = datetime.fromisoformat(run.started_at)
        completed = datetime.fromisoformat(run.completed_at)
        run.duration_seconds = (completed - started).total_seconds()
        
        self._save_history()
        
        status = "successfully" if success else f"with error: {error_message}"
        logger.info(f"Training run {run_id} completed {status}")
    
    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        """Get a specific training run."""
        return self._runs.get(run_id)
    
    def get_all_runs(self) -> List[TrainingRun]:
        """Get all training runs, sorted by start time (newest first)."""
        runs = list(self._runs.values())
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs
    
    def get_recent_runs(self, limit: int = 10) -> List[TrainingRun]:
        """Get most recent training runs."""
        return self.get_all_runs()[:limit]
    
    def get_successful_runs(self) -> List[TrainingRun]:
        """Get all successful training runs."""
        return [r for r in self.get_all_runs() if r.success]
    
    def get_run_metrics(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all metrics for a training run.
        
        Returns:
            Dictionary mapping metric name to list of {step, value, timestamp}
        """
        run = self._runs.get(run_id)
        if not run:
            return {}
        
        return {
            name: [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in metrics]
            for name, metrics in run.metrics.items()
        }
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple training runs.
        
        Returns:
            Comparison data including metrics, configs, and outcomes
        """
        runs = [self._runs[rid] for rid in run_ids if rid in self._runs]
        
        if not runs:
            return {}
        
        return {
            "runs": [
                {
                    "id": r.id,
                    "model_name": r.model_name,
                    "epochs": r.epochs,
                    "batch_size": r.batch_size,
                    "learning_rate": r.learning_rate,
                    "final_loss": r.final_loss,
                    "duration_seconds": r.duration_seconds,
                    "success": r.success
                }
                for r in runs
            ],
            "best_run": (
                min(
                    [r for r in runs if r.final_loss is not None],
                    key=lambda r: r.final_loss or float('inf')
                ) if any(r.final_loss is not None for r in runs) else None
            )
        }
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a training run from history."""
        if run_id in self._runs:
            del self._runs[run_id]
            self._save_history()
            return True
        return False
    
    def clear_history(self):
        """Clear all training history."""
        self._runs.clear()
        self._save_history()
        logger.info("Cleared all training history")
    
    def export_to_csv(self, output_path: Path) -> bool:
        """Export training history to CSV."""
        try:
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Run ID", "Model Name", "Model Size", "Started", "Duration (s)",
                    "Epochs", "Batch Size", "Learning Rate", "Final Loss",
                    "Total Steps", "Success"
                ])
                
                for run in self.get_all_runs():
                    writer.writerow([
                        run.id, run.model_name, run.model_size, run.started_at,
                        run.duration_seconds, run.epochs, run.batch_size,
                        run.learning_rate, run.final_loss, run.total_steps,
                        run.success
                    ])
            
            logger.info(f"Exported training history to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False


# Global instance
_training_history: Optional[TrainingHistory] = None


def get_training_history() -> TrainingHistory:
    """Get or create global training history instance."""
    global _training_history
    if _training_history is None:
        _training_history = TrainingHistory()
    return _training_history
