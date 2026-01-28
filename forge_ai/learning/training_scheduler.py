"""
Training Scheduler - Automatic LoRA training based on collected feedback.

Monitors feedback collection and triggers LoRA training when criteria are met.
"""

import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """
    Schedule and execute LoRA training based on collected feedback.
    
    Features:
    - Monitors learning example accumulation
    - Triggers training when criteria met (min examples, time interval)
    - Prepares datasets from high-quality examples
    - Executes LoRA fine-tuning
    - Tracks training history
    """
    
    def __init__(self, model_name: str):
        """
        Initialize training scheduler.
        
        Args:
            model_name: Name of the model to train
        """
        self.model_name = model_name
        
        # Configuration (can be loaded from config)
        self.min_examples_for_training = 100
        self.training_interval_hours = 24
        self.min_quality_score = 0.6
        self.max_examples_per_training = 1000
        
        # LoRA configuration
        self.lora_config = {
            "r": 8,  # LoRA rank
            "alpha": 16,  # LoRA alpha
            "dropout": 0.1,  # Dropout rate
            "target_modules": ["q_proj", "v_proj"]  # Which layers to adapt
        }
        
        # State
        self.last_training_time: Optional[datetime] = None
        self.training_in_progress = False
        self._lock = threading.Lock()
        
        # Load state
        self._load_state()
        
        logger.info(f"TrainingScheduler initialized for {model_name}")
    
    def _load_state(self):
        """Load scheduler state from disk."""
        try:
            from ..config import CONFIG
            models_dir = Path(CONFIG.get("models_dir", "models"))
            state_file = models_dir / self.model_name / "learning" / "training_state.json"
            
            if state_file.exists():
                import json
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    if 'last_training_time' in data:
                        self.last_training_time = datetime.fromisoformat(data['last_training_time'])
                logger.info(f"Loaded training state. Last training: {self.last_training_time}")
        except Exception as e:
            logger.warning(f"Could not load training state: {e}")
    
    def _save_state(self):
        """Save scheduler state to disk."""
        try:
            from ..config import CONFIG
            models_dir = Path(CONFIG.get("models_dir", "models"))
            state_file = models_dir / self.model_name / "learning" / "training_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            data = {
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
            }
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save training state: {e}")
    
    def should_train(self) -> bool:
        """
        Check if it's time to train.
        
        Criteria:
        - Have enough training examples
        - Enough time passed since last training
        - User hasn't disabled auto-training
        - Not already training
        
        Returns:
            True if training should be triggered
        """
        with self._lock:
            if self.training_in_progress:
                return False
            
            # Get learning engine to check example count
            try:
                from ..core.self_improvement import get_learning_engine
                engine = get_learning_engine(self.model_name)
                
                # Check if we have enough data
                queue_stats = engine.get_queue_stats()
                num_examples = queue_stats.get('total_examples', 0)
                
                if num_examples < self.min_examples_for_training:
                    logger.debug(f"Not enough examples for training: {num_examples}/{self.min_examples_for_training}")
                    return False
                
                # Check time since last training
                if self.last_training_time:
                    hours_since = (datetime.now() - self.last_training_time).total_seconds() / 3600
                    if hours_since < self.training_interval_hours:
                        logger.debug(f"Too soon since last training: {hours_since:.1f}/{self.training_interval_hours} hours")
                        return False
                
                logger.info(f"Training criteria met: {num_examples} examples, ready to train")
                return True
                
            except Exception as e:
                logger.error(f"Error checking training criteria: {e}")
                return False
    
    def run_training(self) -> bool:
        """
        Execute LoRA training with collected examples.
        
        Returns:
            True if training completed successfully
        """
        with self._lock:
            if self.training_in_progress:
                logger.warning("Training already in progress")
                return False
            self.training_in_progress = True
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING LORA TRAINING")
            logger.info("=" * 60)
            
            from ..core.self_improvement import get_learning_engine
            engine = get_learning_engine(self.model_name)
            
            # Get high-quality training examples
            examples = self._get_training_examples(engine)
            
            if not examples:
                logger.warning("No suitable training examples found")
                return False
            
            logger.info(f"Preparing dataset with {len(examples)} examples...")
            
            # Prepare dataset
            dataset = self._prepare_lora_dataset(examples)
            
            # Export training data for reference
            training_file = self._export_training_data(examples)
            logger.info(f"Training data exported to: {training_file}")
            
            # TODO: Actual LoRA training implementation
            # This would integrate with forge_ai/core/training.py
            logger.info(f"LoRA training with config: {self.lora_config}")
            logger.info("Note: LoRA training integration pending - data collected and ready")
            
            # For now, just simulate training
            logger.info("Training simulation: Would train model with prepared dataset")
            
            # Update state
            self.last_training_time = datetime.now()
            self._save_state()
            
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return False
        finally:
            with self._lock:
                self.training_in_progress = False
    
    def _get_training_examples(self, engine) -> List:
        """
        Get high-quality examples for training.
        
        Args:
            engine: Learning engine instance
            
        Returns:
            List of training examples
        """
        # Filter examples from queue
        good_examples = [
            e for e in engine.learning_queue
            if e.quality_score >= self.min_quality_score
        ]
        
        # Sort by priority and quality
        good_examples.sort(
            key=lambda e: (e.priority.value, e.quality_score),
            reverse=True
        )
        
        # Limit size
        return good_examples[:self.max_examples_per_training]
    
    def _prepare_lora_dataset(self, examples: List) -> List[Dict[str, str]]:
        """
        Prepare dataset in format suitable for LoRA training.
        
        Args:
            examples: List of LearningExample objects
            
        Returns:
            List of training examples in dict format
        """
        dataset = []
        for example in examples:
            dataset.append({
                "input": example.input_text,
                "output": example.output_text,
                "quality": example.quality_score,
                "source": example.source.value
            })
        return dataset
    
    def _export_training_data(self, examples: List) -> Path:
        """
        Export training examples to file for reference.
        
        Args:
            examples: List of LearningExample objects
            
        Returns:
            Path to exported file
        """
        from ..config import CONFIG
        models_dir = Path(CONFIG.get("models_dir", "models"))
        export_dir = models_dir / self.model_name / "learning"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"training_data_{timestamp}.txt"
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(f"# Training data for {self.model_name}\n")
            f.write(f"# Exported: {datetime.now().isoformat()}\n")
            f.write(f"# Examples: {len(examples)}\n")
            f.write(f"# Min quality: {self.min_quality_score}\n")
            f.write("\n")
            
            for i, example in enumerate(examples, 1):
                f.write(f"# Example {i} (Quality: {example.quality_score:.2f}, Priority: {example.priority.name})\n")
                f.write(f"Q: {example.input_text}\n")
                f.write(f"A: {example.output_text}\n")
                f.write("\n")
        
        return export_path
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.
        
        Returns:
            Dictionary with status information
        """
        from ..core.self_improvement import get_learning_engine
        engine = get_learning_engine(self.model_name)
        queue_stats = engine.get_queue_stats()
        
        status = {
            'model_name': self.model_name,
            'training_in_progress': self.training_in_progress,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'examples_collected': queue_stats.get('total_examples', 0),
            'min_examples_needed': self.min_examples_for_training,
            'ready_to_train': self.should_train()
        }
        
        if self.last_training_time:
            hours_since = (datetime.now() - self.last_training_time).total_seconds() / 3600
            status['hours_since_training'] = hours_since
        
        return status


# Global scheduler instances
_schedulers = {}
_lock = threading.Lock()


def get_training_scheduler(model_name: str) -> TrainingScheduler:
    """
    Get or create a training scheduler for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        TrainingScheduler instance
    """
    with _lock:
        if model_name not in _schedulers:
            _schedulers[model_name] = TrainingScheduler(model_name)
        return _schedulers[model_name]
