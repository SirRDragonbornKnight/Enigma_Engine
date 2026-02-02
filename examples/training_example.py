#!/usr/bin/env python3
"""
ForgeAI Model Training Example
==============================

Complete example showing how to train ForgeAI models including:
- Training from text data
- Training specialized models (router, vision, code)
- Configuration options
- Checkpointing and resuming
- Training monitoring

ForgeAI supports training models from nano (1M params) to omega (70B+).
This example covers the training workflow.

Dependencies:
    pip install torch  # PyTorch for training

Run: python examples/training_example.py
"""

import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model architecture
    model_size: str = "small"  # nano, micro, tiny, small, medium, large, etc.
    vocab_size: int = 50257
    max_seq_len: int = 512
    
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 0.0001
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Data
    train_data_path: str = "data/training.txt"
    val_split: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "models/checkpoints"
    save_every_n_steps: int = 1000
    
    # Hardware
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = True
    
    # Logging
    log_every_n_steps: int = 100
    wandb_project: Optional[str] = None


# Model size presets
MODEL_SIZES = {
    "nano":   {"d_model": 64,   "n_heads": 2,  "n_layers": 2,  "params": "~1M"},
    "micro":  {"d_model": 128,  "n_heads": 2,  "n_layers": 4,  "params": "~2M"},
    "tiny":   {"d_model": 256,  "n_heads": 4,  "n_layers": 4,  "params": "~5M"},
    "small":  {"d_model": 512,  "n_heads": 8,  "n_layers": 6,  "params": "~27M"},
    "medium": {"d_model": 768,  "n_heads": 12, "n_layers": 12, "params": "~85M"},
    "large":  {"d_model": 1024, "n_heads": 16, "n_layers": 24, "params": "~300M"},
    "xl":     {"d_model": 2048, "n_heads": 16, "n_layers": 24, "params": "~1B"},
    "xxl":    {"d_model": 4096, "n_heads": 32, "n_layers": 32, "params": "~7B"},
}


# =============================================================================
# Simulated Trainer (For Testing)
# =============================================================================

class TrainerSimulator:
    """
    Simulated trainer for demonstrating the training workflow.
    
    In real usage, use forge_ai.core.training.Trainer which
    implements actual gradient descent and model updates.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        self._training_log: List[Dict] = []
    
    def _log(self, message: str):
        """Log a training message."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def prepare_data(self) -> bool:
        """Load and prepare training data."""
        data_path = Path(self.config.train_data_path)
        
        if not data_path.exists():
            self._log(f"Training data not found: {data_path}")
            self._log("Creating sample training data...")
            
            # Create sample data
            data_path.parent.mkdir(parents=True, exist_ok=True)
            sample_data = """Hello, how can I help you today?
I'm an AI assistant created by ForgeAI.
I can help with many tasks including writing, coding, and analysis.
What would you like to know?
I'm always happy to help!
Let me think about that for a moment.
That's an interesting question.
Here's what I know about that topic."""
            
            with open(data_path, 'w') as f:
                f.write(sample_data)
        
        # Load data
        with open(data_path, 'r') as f:
            text = f.read()
        
        lines = text.strip().split('\n')
        self._log(f"Loaded {len(lines)} lines of training data")
        
        if len(lines) < 100:
            self._log("WARNING: Very small dataset. Need 1000+ lines for good results.")
        
        return True
    
    def create_model(self) -> bool:
        """Create model based on config."""
        size_info = MODEL_SIZES.get(self.config.model_size)
        
        if not size_info:
            self._log(f"Unknown model size: {self.config.model_size}")
            return False
        
        self._log(f"Creating {self.config.model_size} model (~{size_info['params']} params)")
        self._log(f"  d_model: {size_info['d_model']}")
        self._log(f"  n_heads: {size_info['n_heads']}")
        self._log(f"  n_layers: {size_info['n_layers']}")
        
        # In real code, this creates actual model:
        # from forge_ai.core.model import create_model
        # self.model = create_model(size=self.config.model_size)
        
        self.model = {"size": self.config.model_size, **size_info}
        return True
    
    def setup_optimizer(self) -> bool:
        """Configure optimizer and scheduler."""
        self._log(f"Setting up optimizer (lr={self.config.learning_rate})")
        self._log(f"  Weight decay: {self.config.weight_decay}")
        self._log(f"  Warmup steps: {self.config.warmup_steps}")
        
        # In real code:
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.config.learning_rate,
        #     weight_decay=self.config.weight_decay
        # )
        
        self.optimizer = {"lr": self.config.learning_rate}
        return True
    
    def train_step(self, batch_data: Dict) -> float:
        """Execute one training step."""
        self.step += 1
        
        # Simulate training loss (decreasing over time)
        import random
        base_loss = 3.0 - (self.step / 1000) * 0.5
        noise = random.uniform(-0.1, 0.1)
        loss = max(0.1, base_loss + noise)
        
        self._training_log.append({
            'step': self.step,
            'epoch': self.epoch,
            'loss': loss,
            'lr': self.config.learning_rate
        })
        
        return loss
    
    def train_epoch(self, epoch_num: int) -> float:
        """Train for one epoch."""
        self.epoch = epoch_num
        self._log(f"Epoch {epoch_num}/{self.config.num_epochs}")
        
        # Simulate batches (would iterate over real data)
        num_batches = 10  # Simulated
        epoch_loss = 0.0
        
        for batch_idx in range(num_batches):
            batch = {"input_ids": [0], "labels": [0]}  # Simulated
            loss = self.train_step(batch)
            epoch_loss += loss
            
            if self.step % self.config.log_every_n_steps == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                self._log(f"  Step {self.step}: loss={avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if path is None:
            path = checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        
        self._log(f"Saving checkpoint: {path}")
        
        # In real code:
        # torch.save({
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'step': self.step,
        #     'epoch': self.epoch,
        #     'config': self.config
        # }, path)
    
    def load_checkpoint(self, path: str) -> bool:
        """Load checkpoint to resume training."""
        self._log(f"Loading checkpoint: {path}")
        
        if not Path(path).exists():
            self._log("Checkpoint not found")
            return False
        
        # In real code:
        # checkpoint = torch.load(path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.step = checkpoint['step']
        # self.epoch = checkpoint['epoch']
        
        return True
    
    def train(self):
        """Full training loop."""
        self._log("="*50)
        self._log("Starting training")
        self._log("="*50)
        
        # Prepare
        if not self.prepare_data():
            return
        if not self.create_model():
            return
        if not self.setup_optimizer():
            return
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, self.config.num_epochs + 1):
            epoch_loss = self.train_epoch(epoch)
            
            self._log(f"Epoch {epoch} complete - avg loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint()
        
        # Done
        elapsed = time.time() - start_time
        self._log("="*50)
        self._log(f"Training complete!")
        self._log(f"Total time: {elapsed:.1f}s")
        self._log(f"Final loss: {self.best_loss:.4f}")
        self._log("="*50)
    
    def get_training_log(self) -> List[Dict]:
        """Get training history."""
        return self._training_log


# =============================================================================
# Specialized Model Training
# =============================================================================

class SpecializedTrainer:
    """
    Train specialized models for specific tasks:
    - Router model: Classifies user intent
    - Vision model: Describes images
    - Code model: Generates code
    """
    
    @staticmethod
    def train_router_model(data_path: str, output_path: str):
        """
        Train a router model for intent classification.
        
        The router classifies user messages into categories:
        - chat: General conversation
        - code: Code generation request
        - image: Image generation request
        - search: Web search request
        - file: File operation request
        """
        print("\n" + "="*50)
        print("Training Router Model")
        print("="*50)
        
        # Sample router training data format
        sample_data = """[CHAT] Hello, how are you?
[CODE] Write a function to sort a list
[IMAGE] Generate an image of a sunset
[SEARCH] What is the weather today?
[FILE] Read the contents of readme.txt
[CHAT] Tell me a joke
[CODE] Debug this Python code
[IMAGE] Draw a cat sitting on a couch"""
        
        print(f"Training data format:")
        for line in sample_data.split('\n')[:4]:
            print(f"  {line}")
        
        print(f"\nWould train router model and save to: {output_path}")
        
        # In real code:
        # python scripts/train_specialized_model.py --type router --data data/router_training.txt
    
    @staticmethod
    def train_vision_model(data_path: str, output_path: str):
        """
        Train a vision model for image description.
        """
        print("\n" + "="*50)
        print("Training Vision Model")
        print("="*50)
        
        print("Vision model training requires:")
        print("  - Image-caption pairs")
        print("  - GPU with sufficient VRAM")
        print("  - Vision encoder (CLIP, ViT)")
        
        print(f"\nWould train vision model and save to: {output_path}")
    
    @staticmethod
    def train_code_model(data_path: str, output_path: str):
        """
        Train a code generation model.
        """
        print("\n" + "="*50)
        print("Training Code Model")
        print("="*50)
        
        print("Code model training data format:")
        sample = """# Task: Write a function to calculate factorial
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Task: Sort a list of numbers
def sort_numbers(nums):
    return sorted(nums)"""
        
        for line in sample.split('\n')[:6]:
            print(f"  {line}")
        
        print(f"\nWould train code model and save to: {output_path}")


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_training():
    """Basic model training."""
    print("\n" + "="*60)
    print("Example 1: Basic Model Training")
    print("="*60)
    
    config = TrainingConfig(
        model_size="tiny",
        batch_size=4,
        learning_rate=0.0001,
        num_epochs=3,
        train_data_path="data/training.txt"
    )
    
    trainer = TrainerSimulator(config)
    trainer.train()


def example_model_sizes():
    """Show available model sizes."""
    print("\n" + "="*60)
    print("Example 2: Available Model Sizes")
    print("="*60)
    
    print("\nForgeAI supports these model sizes:\n")
    print(f"{'Size':<10} {'d_model':<10} {'Heads':<8} {'Layers':<8} {'Params':<10}")
    print("-" * 50)
    
    for name, info in MODEL_SIZES.items():
        print(f"{name:<10} {info['d_model']:<10} {info['n_heads']:<8} {info['n_layers']:<8} {info['params']:<10}")
    
    print("\nRecommendations:")
    print("  - Raspberry Pi: nano, micro")
    print("  - Desktop CPU: tiny, small")
    print("  - Desktop GPU: small, medium, large")
    print("  - Multi-GPU: xl, xxl")


def example_training_config():
    """Training configuration options."""
    print("\n" + "="*60)
    print("Example 3: Training Configuration")
    print("="*60)
    
    print("\nKey hyperparameters:\n")
    
    print("Learning Rate:")
    print("  - 0.0001 (1e-4): Safe default, slow but stable")
    print("  - 0.0003 (3e-4): Faster, good for small models")
    print("  - 0.00001 (1e-5): For fine-tuning pre-trained models")
    
    print("\nBatch Size:")
    print("  - 1-4: Limited VRAM")
    print("  - 8-16: Standard GPU")
    print("  - 32+: Multi-GPU or gradient accumulation")
    
    print("\nWarmup Steps:")
    print("  - 100-500: For small datasets")
    print("  - 1000-2000: For large datasets")
    
    print("\nWeight Decay:")
    print("  - 0.01: Standard regularization")
    print("  - 0.1: Stronger regularization for overfitting")


def example_checkpointing():
    """Checkpoint and resume training."""
    print("\n" + "="*60)
    print("Example 4: Checkpointing")
    print("="*60)
    
    config = TrainingConfig(
        model_size="small",
        num_epochs=2,
        checkpoint_dir="models/checkpoints",
        save_every_n_steps=500
    )
    
    trainer = TrainerSimulator(config)
    
    print("Checkpointing options:")
    print(f"  - Checkpoint directory: {config.checkpoint_dir}")
    print(f"  - Save every {config.save_every_n_steps} steps")
    
    print("\nTo resume training:")
    print("  trainer.load_checkpoint('models/checkpoints/checkpoint_step_500.pt')")
    print("  trainer.train()  # Continues from step 500")


def example_specialized():
    """Train specialized models."""
    print("\n" + "="*60)
    print("Example 5: Specialized Models")
    print("="*60)
    
    SpecializedTrainer.train_router_model(
        "data/specialized/router_training.txt",
        "models/specialized/router"
    )
    
    SpecializedTrainer.train_vision_model(
        "data/specialized/vision_training.txt",
        "models/specialized/vision"
    )
    
    SpecializedTrainer.train_code_model(
        "data/specialized/code_training.txt",
        "models/specialized/code"
    )


def example_forge_integration():
    """Using actual ForgeAI training."""
    print("\n" + "="*60)
    print("Example 6: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI training:")
    print("""
    from forge_ai.core.training import Trainer, TrainingConfig
    from forge_ai.core.model import create_model
    from forge_ai.core.tokenizer import get_tokenizer
    
    # Create model and tokenizer
    model = create_model(size='small')
    tokenizer = get_tokenizer()
    
    # Configure training
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.0001,
        num_epochs=10,
        train_data_path='data/training.txt'
    )
    
    # Train
    trainer = Trainer(model, tokenizer, config)
    trainer.train()
    
    # Save trained model
    trainer.save_model('models/my_trained_model')
    
    # Command line training:
    python run.py --train
    python run.py --train --size medium --epochs 20
    
    # Train specialized model:
    python scripts/train_specialized_model.py --type router
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Model Training Examples")
    print("="*60)
    
    example_model_sizes()
    example_training_config()
    example_basic_training()
    example_checkpointing()
    example_specialized()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    print("""
Training Workflow:

1. Prepare Data:
   - Text file with training examples
   - Minimum 1000+ lines recommended
   - Clean, well-formatted text

2. Configure Training:
   - Choose model size based on hardware
   - Set learning rate (0.0001 default)
   - Set batch size (4-16 typical)
   - Set epochs (5-20 typical)

3. Train:
   - Monitor loss decreasing
   - Save checkpoints regularly
   - Validate on held-out data

4. Evaluate:
   - Test on sample prompts
   - Check for overfitting
   - Adjust hyperparameters if needed

Key Tips:
   - Start small (tiny/small models)
   - Use GPU if available
   - Don't use too high learning rate (>0.001)
   - Save checkpoints frequently
   - Monitor for overfitting

Command Line:
   python run.py --train                    # Default training
   python run.py --train --size small       # Specific size
   python run.py --train --epochs 20        # More epochs
   python run.py --train --resume           # Resume from checkpoint
""")
