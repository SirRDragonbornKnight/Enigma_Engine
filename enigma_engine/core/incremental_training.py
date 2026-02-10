"""
Incremental Training for Enigma AI Engine

Add more training data to existing trained models without starting from scratch.

Features:
- Continue training from checkpoint
- Add new data to existing model
- Preserve existing capabilities
- Elastic weight consolidation (EWC) to prevent forgetting
- Data mixing (old + new)
- Learning rate scheduling for fine-tuning

Usage:
    from enigma_engine.core.incremental_training import IncrementalTrainer
    
    trainer = IncrementalTrainer(model_path="models/my_model")
    
    # Add new training data
    trainer.add_data("data/new_conversations.txt")
    
    # Train incrementally
    trainer.train(
        epochs=5,
        preserve_ratio=0.3,  # Mix in 30% old data
        learning_rate=1e-5   # Lower LR for fine-tuning
    )
    
    # Save updated model
    trainer.save("models/my_model_v2")
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


@dataclass
class IncrementalConfig:
    """Configuration for incremental training."""
    # Training parameters
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    
    # Incremental-specific
    preserve_ratio: float = 0.2      # Ratio of old data to mix in
    ewc_lambda: float = 0.0          # EWC strength (0 = disabled)
    warmup_steps: int = 100
    gradient_accumulation: int = 4
    
    # Forgetting prevention
    max_forgetting_rate: float = 0.1     # Max allowed performance drop
    validation_interval: int = 100        # Steps between validations
    early_stop_patience: int = 3          # Stop if forgetting detected
    
    # Data
    max_length: int = 512
    shuffle_buffer: int = 10000
    
    # Checkpointing
    save_steps: int = 500
    keep_checkpoint_count: int = 3


@dataclass
class TrainingState:
    """State of incremental training."""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    forgetting_count: int = 0
    data_files: List[str] = field(default_factory=list)


class IncrementalTrainer:
    """
    Trainer for adding new data to existing models.
    
    Supports:
    - Continue training from checkpoints
    - Mixing old and new data
    - Elastic weight consolidation
    - Forgetting detection
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[IncrementalConfig] = None
    ):
        """
        Initialize incremental trainer.
        
        Args:
            model_path: Path to existing model (or None for new model)
            config: Training configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for incremental training")
        
        self._model_path = Path(model_path) if model_path else None
        self._config = config or IncrementalConfig()
        self._state = TrainingState()
        
        # Components
        self._model = None
        self._tokenizer = None
        self._optimizer = None
        self._scheduler = None
        
        # Data
        self._new_data: List[str] = []
        self._old_data: List[str] = []
        self._validation_data: List[str] = []
        
        # EWC
        self._fisher_matrix: Optional[Dict[str, torch.Tensor]] = None
        self._optimal_params: Optional[Dict[str, torch.Tensor]] = None
        
        # Callbacks
        self._callbacks: List[Callable] = []
        
        # Load model if path provided
        if self._model_path:
            self._load_model()
        
        logger.info("IncrementalTrainer initialized")
    
    def _load_model(self):
        """Load existing model and tokenizer."""
        try:
            from enigma_engine.core.model import Forge
            from enigma_engine.core.tokenizer import get_tokenizer
            
            model_file = self._model_path / "model.pt"
            if model_file.exists():
                self._model = Forge.from_pretrained(str(self._model_path))
                logger.info(f"Loaded model from {self._model_path}")
            else:
                # Try loading HuggingFace-style
                self._model = Forge.from_pretrained(str(self._model_path))
            
            self._tokenizer = get_tokenizer()
            
            # Compute baseline metrics
            self._compute_baseline()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def add_data(
        self,
        data_source: str | List[str],
        is_validation: bool = False
    ):
        """
        Add training data.
        
        Args:
            data_source: Path to file or list of text samples
            is_validation: Whether this is validation data
        """
        samples = []
        
        if isinstance(data_source, str):
            path = Path(data_source)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into samples
                if '\n\n' in content:
                    samples = [s.strip() for s in content.split('\n\n') if s.strip()]
                else:
                    samples = [s.strip() for s in content.split('\n') if s.strip()]
                
                self._state.data_files.append(str(path))
            else:
                samples = [data_source]
        else:
            samples = data_source
        
        if is_validation:
            self._validation_data.extend(samples)
            logger.info(f"Added {len(samples)} validation samples")
        else:
            self._new_data.extend(samples)
            logger.info(f"Added {len(samples)} training samples")
    
    def load_old_data(
        self,
        data_path: str,
        max_samples: Optional[int] = None
    ):
        """
        Load old training data for mixing.
        
        Args:
            data_path: Path to old training data
            max_samples: Maximum samples to load
        """
        path = Path(data_path)
        if not path.exists():
            logger.warning(f"Old data path not found: {data_path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '\n\n' in content:
            samples = [s.strip() for s in content.split('\n\n') if s.strip()]
        else:
            samples = [s.strip() for s in content.split('\n') if s.strip()]
        
        if max_samples:
            samples = random.sample(samples, min(max_samples, len(samples)))
        
        self._old_data.extend(samples)
        logger.info(f"Loaded {len(samples)} old samples for mixing")
    
    def _prepare_training_data(self) -> List[str]:
        """Prepare mixed training data."""
        config = self._config
        
        # Calculate mix
        new_count = len(self._new_data)
        if config.preserve_ratio > 0 and self._old_data:
            old_count = int(new_count * config.preserve_ratio / (1 - config.preserve_ratio))
            old_samples = random.sample(
                self._old_data,
                min(old_count, len(self._old_data))
            )
        else:
            old_samples = []
        
        # Combine and shuffle
        all_data = self._new_data + old_samples
        random.shuffle(all_data)
        
        logger.info(f"Training data: {len(self._new_data)} new + {len(old_samples)} old = {len(all_data)} total")
        
        return all_data
    
    def _compute_baseline(self):
        """Compute baseline metrics before training."""
        if not self._validation_data or not self._model:
            return
        
        self._model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for sample in self._validation_data[:100]:
                try:
                    tokens = self._tokenizer.encode(sample)
                    if len(tokens) < 2:
                        continue
                    
                    input_ids = torch.tensor([tokens[:-1]])
                    targets = torch.tensor([tokens[1:]])
                    
                    if hasattr(self._model, 'device'):
                        input_ids = input_ids.to(self._model.device)
                        targets = targets.to(self._model.device)
                    
                    outputs = self._model(input_ids)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                    total_loss += loss.item()
                    count += 1
                    
                except Exception:
                    continue
        
        if count > 0:
            self._state.baseline_metrics['loss'] = total_loss / count
            logger.info(f"Baseline loss: {self._state.baseline_metrics['loss']:.4f}")
    
    def _setup_ewc(self):
        """Setup Elastic Weight Consolidation."""
        if self._config.ewc_lambda <= 0 or not self._old_data:
            return
        
        logger.info("Computing Fisher Information Matrix for EWC...")
        
        self._model.eval()
        self._fisher_matrix = {}
        self._optimal_params = {}
        
        # Store optimal params
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self._optimal_params[name] = param.data.clone()
                self._fisher_matrix[name] = torch.zeros_like(param.data)
        
        # Compute Fisher matrix from old data
        num_samples = min(100, len(self._old_data))
        for sample in random.sample(self._old_data, num_samples):
            try:
                tokens = self._tokenizer.encode(sample)
                if len(tokens) < 2:
                    continue
                
                input_ids = torch.tensor([tokens[:-1]])
                targets = torch.tensor([tokens[1:]])
                
                if hasattr(self._model, 'device'):
                    input_ids = input_ids.to(self._model.device)
                    targets = targets.to(self._model.device)
                
                self._model.zero_grad()
                outputs = self._model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                loss.backward()
                
                for name, param in self._model.named_parameters():
                    if param.grad is not None and name in self._fisher_matrix:
                        self._fisher_matrix[name] += param.grad.data ** 2
                        
            except Exception:
                continue
        
        # Average
        for name in self._fisher_matrix:
            self._fisher_matrix[name] /= num_samples
        
        logger.info("EWC Fisher matrix computed")
    
    def _ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if self._fisher_matrix is None:
            return torch.tensor(0.0)
        
        loss = 0.0
        for name, param in self._model.named_parameters():
            if name in self._fisher_matrix:
                fisher = self._fisher_matrix[name]
                optimal = self._optimal_params[name]
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return self._config.ewc_lambda * loss
    
    def train(
        self,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        preserve_ratio: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train incrementally on new data.
        
        Args:
            epochs: Override config epochs
            learning_rate: Override config learning rate
            preserve_ratio: Override config preserve ratio
            callbacks: Training callbacks
            
        Returns:
            Training results dict
        """
        if not self._new_data:
            raise ValueError("No training data added. Use add_data() first.")
        
        if not self._model:
            raise ValueError("No model loaded. Provide model_path or call load_model().")
        
        # Apply overrides
        config = self._config
        if epochs is not None:
            config.epochs = epochs
        if learning_rate is not None:
            config.learning_rate = learning_rate
        if preserve_ratio is not None:
            config.preserve_ratio = preserve_ratio
        if callbacks:
            self._callbacks.extend(callbacks)
        
        # Prepare data
        training_data = self._prepare_training_data()
        
        # Setup EWC if enabled
        self._setup_ewc()
        
        # Setup optimizer
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(training_data) * config.epochs // config.batch_size
        self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=total_steps // config.epochs,
            eta_min=config.learning_rate / 10
        )
        
        # Training loop
        self._model.train()
        results = {
            'epochs_completed': 0,
            'total_steps': 0,
            'final_loss': float('inf'),
            'forgetting_detected': False
        }
        
        try:
            for epoch in range(config.epochs):
                epoch_loss = self._train_epoch(training_data, epoch)
                
                self._state.epoch = epoch + 1
                results['epochs_completed'] = epoch + 1
                results['final_loss'] = epoch_loss
                
                # Check for forgetting
                if self._validation_data:
                    current_loss = self._evaluate()
                    self._state.current_metrics['loss'] = current_loss
                    
                    if 'loss' in self._state.baseline_metrics:
                        forgetting = (current_loss - self._state.baseline_metrics['loss']) / self._state.baseline_metrics['loss']
                        
                        if forgetting > config.max_forgetting_rate:
                            self._state.forgetting_count += 1
                            logger.warning(f"Forgetting detected: {forgetting:.2%} (count: {self._state.forgetting_count})")
                            
                            if self._state.forgetting_count >= config.early_stop_patience:
                                logger.warning("Early stopping due to forgetting")
                                results['forgetting_detected'] = True
                                break
                        else:
                            self._state.forgetting_count = 0
                
                logger.info(f"Epoch {epoch + 1}/{config.epochs} - Loss: {epoch_loss:.4f}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        results['total_steps'] = self._state.global_step
        
        return results
    
    def _train_epoch(self, data: List[str], epoch: int) -> float:
        """Train for one epoch."""
        config = self._config
        total_loss = 0.0
        step_count = 0
        
        # Create batches
        random.shuffle(data)
        
        self._optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for i, sample in enumerate(data):
            try:
                # Tokenize
                tokens = self._tokenizer.encode(sample)
                if len(tokens) < 2:
                    continue
                
                tokens = tokens[:config.max_length]
                
                input_ids = torch.tensor([tokens[:-1]])
                targets = torch.tensor([tokens[1:]])
                
                if hasattr(self._model, 'device'):
                    input_ids = input_ids.to(self._model.device)
                    targets = targets.to(self._model.device)
                
                # Forward
                outputs = self._model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                # Add EWC loss
                ewc_loss = self._ewc_loss()
                loss = loss + ewc_loss
                
                # Scale for accumulation
                loss = loss / config.gradient_accumulation
                loss.backward()
                
                accumulated_loss += loss.item()
                
                # Gradient step
                if (i + 1) % config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad()
                    
                    total_loss += accumulated_loss
                    step_count += 1
                    self._state.global_step += 1
                    accumulated_loss = 0.0
                    
                    # Callbacks
                    for callback in self._callbacks:
                        callback({
                            'epoch': epoch,
                            'step': self._state.global_step,
                            'loss': total_loss / step_count
                        })
                
            except Exception as e:
                logger.debug(f"Sample error: {e}")
                continue
        
        return total_loss / max(1, step_count)
    
    def _evaluate(self) -> float:
        """Evaluate on validation data."""
        if not self._validation_data:
            return float('inf')
        
        self._model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for sample in self._validation_data[:100]:
                try:
                    tokens = self._tokenizer.encode(sample)
                    if len(tokens) < 2:
                        continue
                    
                    input_ids = torch.tensor([tokens[:-1]])
                    targets = torch.tensor([tokens[1:]])
                    
                    if hasattr(self._model, 'device'):
                        input_ids = input_ids.to(self._model.device)
                        targets = targets.to(self._model.device)
                    
                    outputs = self._model(input_ids)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                    total_loss += loss.item()
                    count += 1
                    
                except Exception:
                    continue
        
        self._model.train()
        return total_loss / max(1, count)
    
    def save(
        self,
        output_path: str,
        save_state: bool = True
    ):
        """
        Save the updated model.
        
        Args:
            output_path: Path to save model
            save_state: Also save training state
        """
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self._model, 'save_pretrained'):
            self._model.save_pretrained(str(output))
        else:
            torch.save(self._model.state_dict(), output / "model.pt")
        
        # Save state
        if save_state:
            import json
            state_dict = {
                'epoch': self._state.epoch,
                'global_step': self._state.global_step,
                'best_loss': self._state.best_loss,
                'baseline_metrics': self._state.baseline_metrics,
                'current_metrics': self._state.current_metrics,
                'data_files': self._state.data_files
            }
            with open(output / "training_state.json", 'w') as f:
                json.dump(state_dict, f, indent=2)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_state(self, state_path: str):
        """Load training state to continue."""
        import json
        with open(state_path, 'r') as f:
            state_dict = json.load(f)
        
        self._state.epoch = state_dict.get('epoch', 0)
        self._state.global_step = state_dict.get('global_step', 0)
        self._state.best_loss = state_dict.get('best_loss', float('inf'))
        self._state.baseline_metrics = state_dict.get('baseline_metrics', {})
        self._state.current_metrics = state_dict.get('current_metrics', {})
        self._state.data_files = state_dict.get('data_files', [])
        
        logger.info(f"Loaded training state from {state_path}")


def continue_training(
    model_path: str,
    new_data_path: str,
    epochs: int = 5,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to continue training on new data.
    
    Args:
        model_path: Path to existing model
        new_data_path: Path to new training data
        epochs: Number of epochs
        output_path: Where to save (default: overwrite)
        
    Returns:
        Training results
    """
    trainer = IncrementalTrainer(model_path)
    trainer.add_data(new_data_path)
    
    results = trainer.train(epochs=epochs)
    
    save_path = output_path or model_path
    trainer.save(save_path)
    
    return results
