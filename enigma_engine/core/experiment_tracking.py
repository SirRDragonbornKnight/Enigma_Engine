"""
Experiment Tracking

Weights & Biases and MLflow integration for experiment tracking.
Logs metrics, hyperparameters, artifacts, and model checkpoints.

FILE: enigma_engine/core/experiment_tracking.py
TYPE: Training
MAIN CLASSES: ExperimentTracker, WandBTracker, MLflowTracker
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class TrackerType(Enum):
    """Supported tracking backends."""
    WANDB = "wandb"
    MLFLOW = "mlflow"
    LOCAL = "local"


@dataclass
class RunConfig:
    """Configuration for an experiment run."""
    project: str
    name: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    group: str = ""
    job_type: str = "train"


@dataclass
class Metric:
    """A logged metric."""
    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""
    
    def __init__(self, tracker_type: TrackerType):
        self.tracker_type = tracker_type
        self._run_active = False
        self._metrics: list[Metric] = []
    
    @abstractmethod
    def start_run(self, config: RunConfig):
        """Start a new experiment run."""
    
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int = None):
        """Log metrics to the tracker."""
    
    @abstractmethod
    def log_params(self, params: dict[str, Any]):
        """Log hyperparameters."""
    
    @abstractmethod
    def log_artifact(self, path: Path, name: str = None):
        """Log an artifact file."""
    
    @abstractmethod
    def finish_run(self):
        """Finish the current run."""
    
    @property
    def is_active(self) -> bool:
        """Check if a run is active."""
        return self._run_active


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, api_key: str = None):
        """Initialize W&B tracker."""
        super().__init__(TrackerType.WANDB)
        self._api_key = api_key
        self._run = None
        self._wandb = None
    
    def _init_wandb(self):
        """Initialize wandb module."""
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
                
                if self._api_key:
                    wandb.login(key=self._api_key)
            except ImportError:
                raise ImportError("wandb not installed. Install with: pip install wandb")
    
    def start_run(self, config: RunConfig):
        """Start a new W&B run."""
        self._init_wandb()
        
        self._run = self._wandb.init(
            project=config.project,
            name=config.name,
            tags=config.tags,
            notes=config.notes,
            config=config.config,
            group=config.group or None,
            job_type=config.job_type
        )
        
        self._run_active = True
        logger.info(f"Started W&B run: {self._run.name}")
    
    def log_metrics(self, metrics: dict[str, float], step: int = None):
        """Log metrics to W&B."""
        if not self._run_active:
            logger.warning("No active run to log metrics")
            return
        
        self._wandb.log(metrics, step=step)
        
        # Store locally too
        for name, value in metrics.items():
            self._metrics.append(Metric(name, value, step or 0))
    
    def log_params(self, params: dict[str, Any]):
        """Log params to W&B config."""
        if self._run:
            self._wandb.config.update(params)
    
    def log_artifact(self, path: Path, name: str = None):
        """Log artifact to W&B."""
        if not self._run_active:
            return
        
        path = Path(path)
        artifact_name = name or path.stem
        
        artifact = self._wandb.Artifact(artifact_name, type="model")
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)
        
        logger.info(f"Logged artifact: {artifact_name}")
    
    def log_model(self, path: Path, name: str = "model"):
        """Log model checkpoint."""
        self.log_artifact(path, f"model-{name}")
    
    def log_table(self, name: str, columns: list[str], data: list[list[Any]]):
        """Log a table to W&B."""
        if not self._run_active:
            return
        
        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({name: table})
    
    def finish_run(self):
        """Finish W&B run."""
        if self._run:
            self._run.finish()
            self._run = None
            self._run_active = False
            logger.info("Finished W&B run")


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""
    
    def __init__(self, tracking_uri: str = None):
        """Initialize MLflow tracker."""
        super().__init__(TrackerType.MLFLOW)
        self._tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        self._run_id = None
        self._mlflow = None
    
    def _init_mlflow(self):
        """Initialize mlflow module."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
            except ImportError:
                raise ImportError("mlflow not installed. Install with: pip install mlflow")
    
    def start_run(self, config: RunConfig):
        """Start a new MLflow run."""
        self._init_mlflow()
        
        # Set experiment
        self._mlflow.set_experiment(config.project)
        
        # Start run
        run = self._mlflow.start_run(
            run_name=config.name,
            tags={
                "notes": config.notes,
                "job_type": config.job_type,
                **{f"tag_{i}": tag for i, tag in enumerate(config.tags)}
            }
        )
        
        self._run_id = run.info.run_id
        self._run_active = True
        
        # Log initial config
        self._mlflow.log_params(config.config)
        
        logger.info(f"Started MLflow run: {self._run_id}")
    
    def log_metrics(self, metrics: dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        if not self._run_active:
            return
        
        for name, value in metrics.items():
            self._mlflow.log_metric(name, value, step=step)
            self._metrics.append(Metric(name, value, step or 0))
    
    def log_params(self, params: dict[str, Any]):
        """Log params to MLflow."""
        if not self._run_active:
            return
        
        # MLflow requires flat params
        flat_params = self._flatten_dict(params)
        self._mlflow.log_params(flat_params)
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)
    
    def log_artifact(self, path: Path, name: str = None):
        """Log artifact to MLflow."""
        if not self._run_active:
            return
        
        self._mlflow.log_artifact(str(path))
        logger.info(f"Logged artifact: {path}")
    
    def log_model(self, model, name: str = "model"):
        """Log PyTorch model to MLflow."""
        if not self._run_active:
            return
        
        try:
            self._mlflow.pytorch.log_model(model, name)
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def finish_run(self):
        """Finish MLflow run."""
        if self._run_active:
            self._mlflow.end_run()
            self._run_id = None
            self._run_active = False
            logger.info("Finished MLflow run")


class LocalTracker(ExperimentTracker):
    """Local file-based experiment tracker."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize local tracker."""
        super().__init__(TrackerType.LOCAL)
        self._output_dir = Path(output_dir or "experiments")
        self._run_dir: Optional[Path] = None
        self._config: Optional[RunConfig] = None
        self._step = 0
    
    def start_run(self, config: RunConfig):
        """Start a new local run."""
        self._config = config
        
        # Create run directory
        timestamp = int(time.time())
        self._run_dir = self._output_dir / config.project / f"{config.name}_{timestamp}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self._run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "project": config.project,
                "name": config.name,
                "tags": config.tags,
                "notes": config.notes,
                "config": config.config,
                "start_time": timestamp
            }, f, indent=2)
        
        self._run_active = True
        self._metrics = []
        
        logger.info(f"Started local run: {self._run_dir}")
    
    def log_metrics(self, metrics: dict[str, float], step: int = None):
        """Log metrics to local file."""
        if not self._run_active:
            return
        
        step = step if step is not None else self._step
        self._step = step + 1
        
        for name, value in metrics.items():
            self._metrics.append(Metric(name, value, step))
        
        # Append to metrics file
        metrics_path = self._run_dir / "metrics.jsonl"
        with open(metrics_path, 'a') as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
    
    def log_params(self, params: dict[str, Any]):
        """Log params to local file."""
        if not self._run_active:
            return
        
        params_path = self._run_dir / "params.json"
        
        # Merge with existing params
        existing = {}
        if params_path.exists():
            with open(params_path) as f:
                existing = json.load(f)
        
        existing.update(params)
        
        with open(params_path, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def log_artifact(self, path: Path, name: str = None):
        """Copy artifact to run directory."""
        if not self._run_active:
            return
        
        import shutil
        path = Path(path)
        
        artifacts_dir = self._run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        dest = artifacts_dir / (name or path.name)
        shutil.copy2(path, dest)
        
        logger.info(f"Logged artifact: {dest}")
    
    def finish_run(self):
        """Finish local run."""
        if not self._run_active:
            return
        
        # Save final summary
        summary_path = self._run_dir / "summary.json"
        
        # Calculate final metrics
        metrics_by_name = {}
        for m in self._metrics:
            if m.name not in metrics_by_name:
                metrics_by_name[m.name] = []
            metrics_by_name[m.name].append(m.value)
        
        summary = {
            "total_steps": max((m.step for m in self._metrics), default=0),
            "end_time": time.time(),
            "final_metrics": {
                name: {
                    "last": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values)
                }
                for name, values in metrics_by_name.items()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self._run_active = False
        logger.info(f"Finished local run: {self._run_dir}")


def create_tracker(tracker_type: Union[str, TrackerType],
                   **kwargs) -> ExperimentTracker:
    """
    Create an experiment tracker.
    
    Args:
        tracker_type: Type of tracker (wandb, mlflow, local)
        **kwargs: Tracker-specific arguments
        
    Returns:
        Experiment tracker instance
    """
    if isinstance(tracker_type, str):
        tracker_type = TrackerType(tracker_type.lower())
    
    if tracker_type == TrackerType.WANDB:
        return WandBTracker(api_key=kwargs.get('api_key'))
    elif tracker_type == TrackerType.MLFLOW:
        return MLflowTracker(tracking_uri=kwargs.get('tracking_uri'))
    elif tracker_type == TrackerType.LOCAL:
        return LocalTracker(output_dir=kwargs.get('output_dir'))
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


class ExperimentManager:
    """Manages experiment runs with automatic tracking."""
    
    def __init__(self, tracker: ExperimentTracker):
        """Initialize manager with tracker."""
        self._tracker = tracker
        self._best_metrics: dict[str, float] = {}
        self._checkpoint_dir: Optional[Path] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tracker.finish_run()
    
    def start(self, config: RunConfig, checkpoint_dir: Path = None):
        """Start experiment."""
        self._tracker.start_run(config)
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def log_step(self, metrics: dict[str, float], step: int):
        """Log a training step."""
        self._tracker.log_metrics(metrics, step)
        
        # Track best metrics
        for name, value in metrics.items():
            if name.startswith("val_") or name.startswith("eval_"):
                if name not in self._best_metrics or value < self._best_metrics[name]:
                    self._best_metrics[name] = value
    
    def save_checkpoint(self, model, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self._checkpoint_dir:
            return
        
        import torch
        
        path = self._checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save(model.state_dict(), path)
        self._tracker.log_artifact(path, f"checkpoint_{step}")
        
        if is_best:
            best_path = self._checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            self._tracker.log_artifact(best_path, "best_model")


__all__ = [
    'ExperimentTracker',
    'WandBTracker',
    'MLflowTracker',
    'LocalTracker',
    'RunConfig',
    'Metric',
    'TrackerType',
    'create_tracker',
    'ExperimentManager'
]
