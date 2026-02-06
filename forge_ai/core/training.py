"""
================================================================================
          CHAPTER 4: THE TRAINING GROUNDS - WHERE AIS GROW STRONGER
================================================================================

    "A mind without training is like a sword without an edge."

Welcome to the TRAINING GROUNDS! This is where your AI transforms from a
random number generator into an intelligent conversationalist. Feed it text,
watch it learn, and marvel as patterns emerge from chaos.

WHY THIS FILE MATTERS:
    Your AI starts knowing NOTHING. Through training, it learns language,
    facts, personality, and capabilities. This file orchestrates that
    entire learning process - from raw text to trained intelligence.

THE TRAINING RITUAL:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  1. PREPARE: Load your training text                           │
    │     data/training.txt → "Hello! I am helpful and friendly..."  │
    │                    ↓                                            │
    │  2. TOKENIZE: Convert words to numbers                         │
    │     "Hello" → [15496, 995, ...]                                │
    │                    ↓                                            │
    │  3. FEED: Show batches to the model                            │
    │     Model sees input, predicts next word, checks if correct    │
    │                    ↓                                            │
    │  4. LEARN: Adjust weights based on errors                      │
    │     Got it wrong? Update neurons. Repeat thousands of times.   │
    │                    ↓                                            │
    │  5. SAVE: Store the trained brain                              │
    │     models/forge.pth (your trained AI!)                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

TRAINING WISDOM:
    | Setting           | Too Low              | Too High             |
    |-------------------|----------------------|----------------------|
    | Learning Rate     | Learns nothing       | Forgets everything   |
    | Epochs            | Undertrained         | Overfitted           |
    | Batch Size        | Noisy learning       | Memory crash         |

QUICK START:
    # From command line (easiest)
    python run.py --train

    # From code (more control)
    >>> from forge_ai.core.training import train_model
    >>> train_model("data/training.txt", epochs=30, model_size="small")

YOUR QUEST HERE:
    Training takes time. Start small (5 epochs) to test, then go big.
    Watch the loss - if it stops going down, training is complete.

CONNECTED PATHS:
    Input comes from → data/training.txt (your training data)
    Brain comes from → model.py (Chapter 1: The Forge)
    GUI interface   → gui/tabs/training_tab.py
    Output saved to → models/forge.pth

SEE ALSO:
    • forge_ai/core/trainer.py  - Advanced ForgeTrainer class
    • docs/HOW_TO_TRAIN.md      - Training guide
    • docs/TRAINING_DATA_FORMAT.md - Data format guide
"""
import logging  # For log messages
import math  # For cosine learning rate schedule
import time  # For timing training
from dataclasses import dataclass  # For clean config classes
from pathlib import Path  # For file paths (cross-platform)
from typing import Any, Callable, Dict, List, Optional, Union  # Type hints

# =============================================================================
# IMPORTS - What libraries we need
# =============================================================================
# PyTorch is the deep learning framework that powers everything
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network building blocks
import torch.nn.functional as F  # Functions like cross_entropy loss
from torch.cuda.amp import GradScaler  # Mixed precision scaler (faster on GPU)
from torch.amp import autocast  # Mixed precision autocast
from torch.utils.data import DataLoader, Dataset  # For loading training data

from ..config import CONFIG  # → Global settings
from ..utils.system_messages import (  # → Pretty printing
    info_msg,
    system_msg,
    warning_msg,
)

# Our own modules
from .model import MODEL_PRESETS, create_model  # → Creates the neural network
from .tokenizer import get_tokenizer, train_tokenizer  # → Converts text↔numbers

logger = logging.getLogger(__name__)

# =============================================================================
# 📁 DEFAULT PATHS - Where things are stored
# =============================================================================
# MODELS_DIR: Where trained models are saved (e.g., models/forge.pth)
# DATA_DIR: Where training data lives (e.g., data/training.txt)
MODELS_DIR = Path(CONFIG.get("models_dir", "models"))
DATA_DIR = Path(CONFIG.get("data_dir", "data"))


# =============================================================================
# ⚙️ TRAINING CONFIGURATION - All the knobs you can turn
# =============================================================================
# This class holds ALL the settings for training. Think of it like a recipe.
# You can create one with defaults: config = TrainingConfig()
# Or customize: config = TrainingConfig(epochs=50, learning_rate=0.001)

@dataclass
class TrainingConfig:
    """
    Configuration for training - all the hyperparameters in one place.
    
    🎛️ HYPERPARAMETERS EXPLAINED:
    
    epochs: How many times to go through ALL the training data
        → More epochs = more learning, but too many = overfitting
        → Start with 30, increase if loss is still dropping
        
    batch_size: How many examples to process at once
        → Bigger = faster but needs more memory
        → 8 is safe for most GPUs, try 16/32 if you have VRAM
        
    learning_rate: How big of steps to take when learning
        → Too high = unstable, too low = slow learning
        → 3e-4 (0.0003) is a good default, rarely go above 1e-3
        
    weight_decay: Prevents model from memorizing (regularization)
        → 0.1 is standard for transformers
    """
    
    # ===== CORE TRAINING SETTINGS =====
    epochs: int = 30              # Number of full passes through data
    batch_size: int = 8           # Examples per forward pass
    learning_rate: float = 3e-4   # Step size for optimizer (0.0003)
    weight_decay: float = 0.1     # L2 regularization strength

    # ===== LEARNING RATE SCHEDULE =====
    # Learning rate starts low (warmup), peaks, then slowly decreases (cosine)
    warmup_steps: int = 100       # Steps to ramp up LR from 0
    min_lr: float = 1e-5          # Minimum LR at end of training

    # ===== GRADIENT SETTINGS =====
    # Gradient clipping prevents "exploding gradients" that break training
    grad_clip: float = 1.0        # Max gradient norm (clips if larger)
    grad_accumulation_steps: int = 4  # Accumulate N batches before updating
    # ^ This lets you simulate larger batch sizes without more memory
    # effective_batch_size = batch_size × grad_accumulation_steps

    # ===== MIXED PRECISION (AMP) =====
    # Uses float16 for some operations - 2x faster on modern GPUs!
    use_amp: bool = True          # Enable automatic mixed precision

    # ===== CHECKPOINTING =====
    # Save model periodically so you don't lose progress
    save_every: int = 5           # Save checkpoint every N epochs
    checkpoint_dir: Optional[str] = None  # Where to save checkpoints

    # ===== LOGGING =====
    log_every: int = 10           # Print progress every N steps
    verbose: bool = True          # Show detailed progress

    # ===== SEQUENCE SETTINGS =====
    max_seq_len: int = 512        # Maximum tokens per training example
    # ^ Longer = more context but slower/more memory

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(MODELS_DIR / "checkpoints")

    @classmethod
    def from_device_profile(cls, **overrides) -> 'TrainingConfig':
        """
        Create TrainingConfig with device-aware defaults.
        
        Automatically adjusts batch_size, max_seq_len, use_amp, and grad_accumulation
        based on detected hardware capabilities.
        
        📐 USAGE:
            # Auto-detect best settings for your hardware
            config = TrainingConfig.from_device_profile()
            
            # Override specific settings
            config = TrainingConfig.from_device_profile(epochs=50)
        
        🖥️ DEVICE ADJUSTMENTS:
            EMBEDDED (Pi): batch=1, seq=128, no AMP, high grad_accum
            MOBILE: batch=2, seq=256, no AMP
            LAPTOP_LOW: batch=4, seq=256, AMP disabled
            LAPTOP_MID: batch=4, seq=512, AMP enabled
            DESKTOP_CPU: batch=4, seq=512, no AMP
            DESKTOP_GPU: batch=8, seq=512, AMP enabled
            WORKSTATION: batch=16, seq=1024, AMP enabled
            DATACENTER: batch=32, seq=2048, AMP enabled
        
        Returns:
            TrainingConfig optimized for current hardware
        """
        try:
            from .device_profiles import DeviceClass, get_device_profiler
            
            profiler = get_device_profiler()
            device_class = profiler.classify()
            caps = profiler.detect()  # Returns DeviceCapabilities
            
            # Device-specific defaults
            defaults = {
                DeviceClass.EMBEDDED: {
                    'batch_size': 1,
                    'max_seq_len': 128,
                    'use_amp': False,
                    'grad_accumulation_steps': 16,  # Simulate larger batch
                    'save_every': 10,  # Save less often (slow storage)
                },
                DeviceClass.MOBILE: {
                    'batch_size': 2,
                    'max_seq_len': 256,
                    'use_amp': False,
                    'grad_accumulation_steps': 8,
                },
                DeviceClass.LAPTOP_LOW: {
                    'batch_size': 4,
                    'max_seq_len': 256,
                    'use_amp': False,
                    'grad_accumulation_steps': 4,
                },
                DeviceClass.LAPTOP_MID: {
                    'batch_size': 4,
                    'max_seq_len': 512,
                    'use_amp': True,
                    'grad_accumulation_steps': 4,
                },
                DeviceClass.DESKTOP_CPU: {
                    'batch_size': 4,
                    'max_seq_len': 512,
                    'use_amp': False,
                    'grad_accumulation_steps': 4,
                },
                DeviceClass.DESKTOP_GPU: {
                    'batch_size': 8,
                    'max_seq_len': 512,
                    'use_amp': True,
                    'grad_accumulation_steps': 4,
                },
                DeviceClass.WORKSTATION: {
                    'batch_size': 16,
                    'max_seq_len': 1024,
                    'use_amp': True,
                    'grad_accumulation_steps': 2,
                },
                DeviceClass.DATACENTER: {
                    'batch_size': 32,
                    'max_seq_len': 2048,
                    'use_amp': True,
                    'grad_accumulation_steps': 1,
                },
            }
            
            # Get defaults for detected device
            device_defaults = defaults.get(device_class, defaults[DeviceClass.LAPTOP_LOW])
            
            # Adjust based on actual VRAM if GPU available
            if caps.has_cuda and caps.vram_total_mb:
                vram_gb = caps.vram_total_mb / 1024
                if vram_gb >= 24:
                    device_defaults['batch_size'] = min(32, device_defaults['batch_size'] * 2)
                    device_defaults['max_seq_len'] = min(2048, device_defaults['max_seq_len'] * 2)
                elif vram_gb >= 12:
                    device_defaults['batch_size'] = min(16, device_defaults['batch_size'])
                elif vram_gb < 6:
                    device_defaults['batch_size'] = max(2, device_defaults['batch_size'] // 2)
                    device_defaults['max_seq_len'] = min(256, device_defaults['max_seq_len'])
            
            # Apply user overrides
            device_defaults.update(overrides)
            
            logger.info(f"[Training] Using device-aware config for {device_class.name}: "
                       f"batch={device_defaults.get('batch_size')}, "
                       f"seq_len={device_defaults.get('max_seq_len')}, "
                       f"amp={device_defaults.get('use_amp')}")
            
            return cls(**device_defaults)
            
        except ImportError:
            logger.warning("[Training] Device profiles not available, using defaults")
            return cls(**overrides)


# =============================================================================
# 📊 DATASET CLASSES - How we prepare text for training
# =============================================================================
# Neural networks can't read text directly - they need numbers!
# These classes convert text into sequences of token IDs that the model 
# can learn from.
#
# HOW IT WORKS:
#   "Hello world" → tokenizer → [15496, 995] → Dataset → Model
#
# The model learns to predict the NEXT token given previous tokens:
#   Input:  [Hello]     → Target: [world]
#   Input:  [Hello, world] → Target: [!]

class TextDataset(Dataset):
    """
    Dataset for language model training.
    
    📖 WHAT THIS DOES:
    1. Takes raw text ("Hello world, how are you?")
    2. Converts to token IDs using tokenizer
    3. Splits into overlapping chunks of max_length
    4. Creates input/target pairs for next-token prediction
    
    📐 EXAMPLE with max_length=4, stride=2:
    Text: "The quick brown fox jumps"
    Tokens: [1, 2, 3, 4, 5]
    
    Sequences created:
      Chunk 1: [1, 2, 3, 4, 5] → input=[1,2,3,4], target=[2,3,4,5]
      Chunk 2: [3, 4, 5, ...]  → (overlapping for better learning)
    
    🔗 CONNECTS TO:
      → Uses tokenizer from forge_ai/core/tokenizer.py
      ← Used by Trainer class below
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Initialize dataset.

        Args:
            texts: List of training texts (e.g., loaded from file)
            tokenizer: Tokenizer to convert text→numbers
            max_length: Maximum tokens per sequence (longer = more context)
            stride: Step size when creating sequences (smaller = more overlap)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.sequences = []  # Will hold all our training sequences

        # Process each text document into sequences
        for text in texts:
            self._process_text(text)

        logger.info(f"Created {len(self.sequences)} training sequences")

    def _process_text(self, text: str):
        """
        Process a text into training sequences.
        
        📖 WHAT HAPPENS HERE:
        1. Convert text to token IDs (numbers)
        2. Slide a window across the tokens to create chunks
        3. Each chunk becomes one training example
        
        📐 SLIDING WINDOW EXAMPLE:
        Text tokens: [1, 2, 3, 4, 5, 6, 7, 8]
        max_length=4, stride=2
        
        Window 1: [1, 2, 3, 4, 5] ← positions 0-4
        Window 2: [3, 4, 5, 6, 7] ← positions 2-6 (overlap!)
        Window 3: [5, 6, 7, 8]    ← final chunk
        
        WHY OVERLAP? It helps the model see the same content from
        different positions, improving learning.
        """
        # ─────────────────────────────────────────────────────────────
        # STEP 1: Convert text to token IDs
        # ─────────────────────────────────────────────────────────────
        # Different tokenizers have different interfaces, so we handle both:
        # - .encode() method (most tokenizers like tiktoken, SentencePiece)
        # - callable (HuggingFace tokenizers return dict)
        if hasattr(self.tokenizer, 'encode'):
            # Direct encode method - returns list of integers
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # HuggingFace style - returns dict with 'input_ids'
            enc = self.tokenizer(text, add_special_tokens=False)
            ids = enc['input_ids']
            # Convert from tensor to list if needed
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            # Handle batched output (list of lists)
            if isinstance(ids[0], list):
                ids = ids[0]

        # ─────────────────────────────────────────────────────────────
        # STEP 2: Create overlapping sequences with sliding window
        # ─────────────────────────────────────────────────────────────
        # We add +1 to max_length because we need one extra token for the
        # TARGET (what we're predicting). If max_length=512, we grab 513
        # tokens: 512 for input, last one for target.
        for i in range(0, max(1, len(ids) - self.max_length), self.stride):
            seq = ids[i:i + self.max_length + 1]  # +1 for target token
            if len(seq) > 2:  # Need at least a few tokens to learn anything
                self.sequences.append(seq)

        # ─────────────────────────────────────────────────────────────
        # STEP 3: Don't forget the last chunk!
        # ─────────────────────────────────────────────────────────────
        # The sliding window might not reach the end perfectly,
        # so we grab the final portion of the text too
        if len(ids) > self.max_length:
            seq = ids[-self.max_length - 1:]  # Last max_length+1 tokens
            if len(seq) > 2:
                self.sequences.append(seq)
        elif len(ids) > 2:
            # Short text - just use it all as one sequence
            self.sequences.append(ids)

    def __len__(self) -> int:
        """Return number of training sequences (used by DataLoader)."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        📖 WHAT HAPPENS HERE:
        Takes a sequence like [1, 2, 3, 4, 5] and creates:
        - input_ids: [1, 2, 3, 4] ← What the model sees
        - labels:    [2, 3, 4, 5] ← What the model should predict
        
        This is called "next token prediction" - given tokens 1,2,3,4
        the model learns to predict 2,3,4,5 (each shifted by one).
        """
        seq = self.sequences[idx]

        # ─────────────────────────────────────────────────────────────
        # PADDING: Make all sequences the same length
        # ─────────────────────────────────────────────────────────────
        # Neural networks need fixed-size inputs for batching.
        # Short sequences get padded with a special pad_token_id.
        if len(seq) < self.max_length + 1:
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            seq = seq + [pad_id] * (self.max_length + 1 - len(seq))

        seq = seq[:self.max_length + 1]  # Truncate if somehow too long

        # ─────────────────────────────────────────────────────────────
        # CREATE INPUT/TARGET PAIRS
        # ─────────────────────────────────────────────────────────────
        # Input: all tokens except the last one  [T1, T2, T3, T4]
        # Target: all tokens except the first one [T2, T3, T4, T5]
        # The model learns: given T1 predict T2, given T1,T2 predict T3, etc.
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,   # Model input
            'labels': target_ids       # What model should predict
        }


# =============================================================================
# 📝 Q&A DATASET - For question/answer style training
# =============================================================================
# This is a SPECIALIZED dataset for training chatbots and assistants.
# Instead of learning from raw text, it learns from Q&A pairs.
#
# INPUT FORMAT (in your training file):
#   Q: What is Python?
#   A: Python is a programming language known for its simplicity.
#   
#   Q: How do I learn coding?
#   A: Start with basics, practice daily, build projects.
#
# The model learns the PATTERN of question→answer, making it better
# at responding to user questions.

class QADataset(Dataset):
    """
    Dataset for Q&A format training.
    
    📖 WHAT THIS DOES:
    1. Parses "Q: question\\nA: answer" format from text files
    2. Creates training examples that teach the model to answer questions
    3. Helps the model learn conversational patterns
    
    📐 EXAMPLE:
    Input file:
        Q: What's your name?
        A: I'm Forge, an AI assistant.
        Q: What can you do?
        A: I can chat, generate images, write code, and more!
    
    Creates 2 training examples, each teaching a Q→A pattern.
    
    🔗 CONNECTS TO:
      → Uses tokenizer from forge_ai/core/tokenizer.py
      ← Used by Trainer class for chatbot training
      
    💡 TIP: Use this for:
      - Chatbot personality training
      - FAQ-style knowledge
      - Customer support responses
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 512
    ):
        """
        Initialize Q&A dataset.
        
        Args:
            texts: List of texts containing Q:/A: pairs
            tokenizer: Tokenizer to convert text→numbers
            max_length: Maximum tokens per example
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []  # Will hold parsed Q&A pairs

        # Parse Q&A pairs from each text
        for text in texts:
            self._parse_qa(text)

        logger.info(f"Created {len(self.examples)} Q&A training examples")

    def _parse_qa(self, text: str):
        """
        Parse Q:/A: format into training examples.
        
        📖 PARSING LOGIC:
        1. Split text on "Q:" markers to find questions
        2. Within each chunk, split on "A:" to find the answer
        3. Combine back into "Q: question\\nA: answer" format
        
        This handles various formatting styles:
          Q: question     (with space)
          Q:question      (without space)
          q: question     (lowercase)
        """
        import re

        # ─────────────────────────────────────────────────────────────
        # STEP 1: Split on "Q:" to find all questions
        # ─────────────────────────────────────────────────────────────
        # re.IGNORECASE makes it work with Q:, q:, even Q :
        parts = re.split(r'\n?Q:\s*', text, flags=re.IGNORECASE)

        for part in parts:
            if not part.strip():
                continue  # Skip empty parts

            # ─────────────────────────────────────────────────────────
            # STEP 2: Split each part on "A:" to separate Q from A
            # ─────────────────────────────────────────────────────────
            # maxsplit=1 means only split on FIRST "A:" (answer might contain "A:")
            qa_split = re.split(r'\n?A:\s*', part, maxsplit=1, flags=re.IGNORECASE)

            if len(qa_split) == 2:
                question = qa_split[0].strip()
                answer = qa_split[1].strip()

                if question and answer:
                    # ─────────────────────────────────────────────────
                    # STEP 3: Create the training example
                    # ─────────────────────────────────────────────────
                    # We format it consistently so the model learns
                    # the exact pattern we want
                    full_text = f"Q: {question}\nA: {answer}"
                    self.examples.append(full_text)

    def __len__(self) -> int:
        """Return number of Q&A examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single Q&A training example.
        
        📖 SAME LOGIC AS TextDataset:
        Converts text to tokens, creates input/target pairs for
        next-token prediction. The model learns to generate the
        answer when given the question.
        """
        text = self.examples[idx]

        # ─────────────────────────────────────────────────────────────
        # ENCODE: Convert Q&A text to token IDs
        # ─────────────────────────────────────────────────────────────
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        else:
            enc = self.tokenizer(text, add_special_tokens=True)
            ids = enc['input_ids']
            if hasattr(ids, 'tolist'):
                ids = ids.tolist()
            if isinstance(ids[0], list):
                ids = ids[0]

        # ─────────────────────────────────────────────────────────────
        # PAD/TRUNCATE: Make all sequences the same length
        # ─────────────────────────────────────────────────────────────
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)

        if len(ids) < self.max_length + 1:
            ids = ids + [pad_id] * (self.max_length + 1 - len(ids))
        ids = ids[:self.max_length + 1]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': target_ids
        }


# =============================================================================
# 📈 LEARNING RATE SCHEDULER - Controls how fast the model learns
# =============================================================================
# Learning rate is THE most important hyperparameter in deep learning.
# Too high = model learns garbage (loss explodes)
# Too low = model learns too slowly (training takes forever)
#
# This scheduler uses a proven strategy:
# 1. WARMUP: Start very low, gradually increase (like warming up a car engine)
# 2. COSINE DECAY: After warmup, slowly decrease (fine-tuning gets gentler)
#
#     Learning Rate
#         │
#    max  │        ╭───╮
#         │       ╱     ╲
#         │      ╱       ╲
#         │     ╱         ╲
#    min  │────╱           ╲────
#         └────────────────────→ Steps
#            Warmup  Cosine Decay

class CosineWarmupScheduler:
    """
    Cosine annealing with linear warmup.
    
    📖 WHAT THIS DOES:
    Phase 1 - WARMUP (first N steps):
      Learning rate: 0 → max_lr (linear increase)
      WHY: Prevents early training from being too aggressive
      
    Phase 2 - COSINE DECAY (remaining steps):
      Learning rate: max_lr → min_lr (smooth cosine curve)
      WHY: Gentler learning as model gets better
    
    📐 EXAMPLE with warmup=100, total=1000, max_lr=0.001:
      Step 0:    lr = 0.0        (starting cold)
      Step 50:   lr = 0.0005     (halfway through warmup)
      Step 100:  lr = 0.001      (peak learning rate)
      Step 500:  lr = 0.0005     (halfway through decay)
      Step 1000: lr = 0.00001    (minimum, fine-tuning)
    
    🔗 CONNECTS TO:
      ← Created by Trainer class
      → Updates optimizer learning rate each step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,  # The optimizer whose LR we'll modify
        warmup_steps: int,                  # How many steps to warm up
        total_steps: int,                   # Total training steps
        max_lr: float,                      # Peak learning rate
        min_lr: float = 1e-5                # Minimum learning rate at end
    ):
        """Initialize the scheduler with warmup and decay settings."""
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0  # Tracks where we are in training

    def step(self) -> None:
        """
        Update learning rate for the current step.
        Called once per training step (after optimizer.step()).
        """
        self.current_step += 1
        lr = self.get_lr()  # Calculate new learning rate

        # Update the learning rate in the optimizer
        # (optimizers can have multiple param groups, we update all of them)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """
        Calculate the learning rate for the current step.
        
        📖 THE MATH:
        
        WARMUP PHASE (step < warmup_steps):
          lr = max_lr * (current_step / warmup_steps)
          This is just linear interpolation from 0 to max_lr
          
        COSINE DECAY PHASE (step >= warmup_steps):
          progress = (step - warmup) / (total - warmup)
          lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
          
          The cosine function gives a smooth S-curve from max to min
        """
        if self.current_step < self.warmup_steps:
            # ─────────────────────────────────────────────────────────
            # WARMUP PHASE: Linear increase from 0 to max_lr
            # ─────────────────────────────────────────────────────────
            return self.max_lr * self.current_step / self.warmup_steps

        # ─────────────────────────────────────────────────────────────
        # COSINE DECAY PHASE: Smooth decrease from max_lr to min_lr
        # ─────────────────────────────────────────────────────────────
        # Calculate how far through the decay phase we are (0.0 to 1.0)
        progress = (self.current_step - self.warmup_steps) / \
            max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)  # Cap at 1.0 if we go over

        # Cosine formula: starts at max_lr, smoothly decreases to min_lr
        # math.cos(0) = 1, math.cos(π) = -1
        # So (1 + cos(π*0)) = 2 → full max_lr contribution
        # And (1 + cos(π*1)) = 0 → no max_lr contribution, just min_lr
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# 🏋️ TRAINER CLASS - The main training engine
# =============================================================================
# This is the HEART of model training. It orchestrates everything:
# - Loading data into batches
# - Running forward/backward passes
# - Updating model weights
# - Saving checkpoints
# - Tracking progress
#
# TRAINING LOOP OVERVIEW:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │  for each epoch:                                                 │
#   │    for each batch:                                               │
#   │      1. Forward pass: model(input) → predictions                 │
#   │      2. Calculate loss: how wrong are the predictions?           │
#   │      3. Backward pass: compute gradients (which direction?)      │
#   │      4. Optimizer step: update weights (move that direction)     │
#   │      5. Scheduler step: adjust learning rate                     │
#   └─────────────────────────────────────────────────────────────────┘

class Trainer:
    """
    Production-grade trainer for Forge models.
    
    📖 WHAT THIS DOES:
    Takes a model and training data, runs the training loop,
    and produces a trained model that can generate text.
    
    ⚡ KEY FEATURES:
    
    1. Mixed Precision (AMP):
       Uses 16-bit floats where possible → 2x faster, less memory
       
    2. Gradient Accumulation:
       Simulates larger batches on limited GPU memory
       batch_size=4 with accumulation=8 acts like batch_size=32
       
    3. Cosine Warmup Schedule:
       Learning rate starts low, peaks, then decays
       
    4. Gradient Clipping:
       Prevents exploding gradients from ruining training
       
    5. Checkpointing:
       Saves progress regularly so you can resume if interrupted
    
    📐 EXAMPLE USAGE:
        trainer = Trainer(model, tokenizer)
        results = trainer.train(texts, epochs=3)
        logger.info(f"Final loss: {results['final_loss']}")
    
    🔗 CONNECTS TO:
      → Uses datasets (TextDataset, QADataset) defined above
      → Uses CosineWarmupScheduler for learning rate
      → Model from forge_ai/core/model.py
      → Tokenizer from forge_ai/core/tokenizer.py
      ← Called by train_model() function below
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: The neural network to train (Forge model)
            tokenizer: Tokenizer for text→numbers conversion
            config: Training settings (learning rate, epochs, etc.)
            device: "cuda" for GPU or "cpu" for CPU
        """
        # Use default config if none provided
        self.config = config or TrainingConfig()

        # ─────────────────────────────────────────────────────────────
        # DEVICE SELECTION: GPU is ~10-100x faster than CPU
        # ─────────────────────────────────────────────────────────────
        if device is None:
            # Auto-detect: use GPU if available, otherwise CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ─────────────────────────────────────────────────────────────
        # MODEL SETUP: Move model to the chosen device
        # ─────────────────────────────────────────────────────────────
        self.model = model.to(self.device)  # .to() moves all parameters
        self.tokenizer = tokenizer

        # ─────────────────────────────────────────────────────────────
        # MIXED PRECISION (AMP) SETUP
        # ─────────────────────────────────────────────────────────────
        # GradScaler handles the tricky parts of 16-bit training:
        # - Scales loss up before backward pass (prevents underflow)
        # - Scales gradients down before optimizer step
        # - Automatically adjusts scale if NaN/Inf detected
        # Only works on CUDA (GPU), CPU doesn't support FP16 acceleration
        self.scaler = GradScaler() if self.config.use_amp and self.device.type == "cuda" else None

        # ─────────────────────────────────────────────────────────────
        # TRAINING STATE: Initialized in train(), stored here for resume
        # ─────────────────────────────────────────────────────────────
        self.optimizer = None     # Will be AdamW
        self.scheduler = None     # Will be CosineWarmupScheduler
        self.global_step = 0      # Total steps taken across all epochs
        self.best_loss = float('inf')  # Best loss seen (for checkpointing)

        # Track losses for reporting and graphing
        self.loss_history = []

        # Log configuration for debugging
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  AMP enabled: {self.scaler is not None}")

    def train(
        self,
        texts: list[str],
        epochs: Optional[int] = None,
        dataset_type: str = "auto",
        callback: Optional[Callable[[dict], None]] = None
    ) -> dict[str, Any]:
        """
        Train the model on texts.
        
        📖 THIS IS THE MAIN TRAINING METHOD!
        
        It sets up everything needed for training:
        1. Creates a Dataset from your texts
        2. Creates a DataLoader for batching
        3. Sets up optimizer and scheduler
        4. Runs the training loop for N epochs
        5. Returns metrics about the training run

        Args:
            texts: List of training texts (loaded from file usually)
            epochs: Number of epochs (full passes through data)
            dataset_type: "text", "qa", or "auto" (auto-detects from content)
            callback: Function called after each epoch for progress updates

        Returns:
            Dictionary with training metrics:
            - final_loss: Loss after last epoch
            - best_loss: Lowest loss seen during training
            - loss_history: List of loss per epoch
            - elapsed_time: Total training time in seconds
            - total_steps: Total optimizer steps taken
        """
        epochs = epochs or self.config.epochs

        # ─────────────────────────────────────────────────────────────
        # STEP 1: DETECT DATASET TYPE
        # ─────────────────────────────────────────────────────────────
        # Look at the data to decide if it's Q&A format or plain text
        # Q&A format: "Q: question\nA: answer"
        # Text format: Just regular paragraphs
        if dataset_type == "auto":
            sample = "\n".join(texts[:10])  # Look at first 10 texts
            if "Q:" in sample or "A:" in sample:
                dataset_type = "qa"
            else:
                dataset_type = "text"

        # ─────────────────────────────────────────────────────────────
        # STEP 2: CREATE DATASET
        # ─────────────────────────────────────────────────────────────
        # Convert raw texts into training sequences
        if dataset_type == "qa":
            dataset = QADataset(
                texts,
                self.tokenizer,
                max_length=self.config.max_seq_len
            )
        else:
            dataset = TextDataset(
                texts,
                self.tokenizer,
                max_length=self.config.max_seq_len,
                stride=self.config.max_seq_len // 2  # 50% overlap between chunks
            )

        # ─────────────────────────────────────────────────────────────
        # STEP 3: CREATE DATALOADER
        # ─────────────────────────────────────────────────────────────
        # DataLoader batches sequences together and shuffles them
        # This is more efficient than processing one sequence at a time
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,   # How many sequences per batch
            shuffle=True,                         # Randomize order each epoch
            num_workers=0,  # Keep simple for compatibility (no multiprocessing)
            pin_memory=self.device.type == "cuda"  # Faster GPU transfer
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 4: CALCULATE TRAINING STEPS
        # ─────────────────────────────────────────────────────────────
        steps_per_epoch = len(dataloader)  # Number of batches
        total_steps = steps_per_epoch * epochs  # Total training steps

        # ─────────────────────────────────────────────────────────────
        # STEP 5: INITIALIZE OPTIMIZER (AdamW)
        # ─────────────────────────────────────────────────────────────
        # AdamW is the go-to optimizer for transformers:
        # - Adam: Adaptive learning rates per parameter
        # - W: Weight decay (prevents overfitting)
        # - betas: Momentum parameters (0.9, 0.95) standard for transformers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),           # What to optimize
            lr=self.config.learning_rate,      # Base learning rate
            weight_decay=self.config.weight_decay,  # Regularization
            betas=(0.9, 0.95)                  # Momentum coefficients
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 6: INITIALIZE LEARNING RATE SCHEDULER
        # ─────────────────────────────────────────────────────────────
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=min(self.config.warmup_steps, total_steps // 10),
            total_steps=total_steps,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 7: PRINT TRAINING INFO (if verbose)
        # ─────────────────────────────────────────────────────────────
        if self.config.verbose:
            print("=" * 60)
            print(system_msg("FORGE AI TRAINING"))
            print("=" * 60)
            print(info_msg(f"Device: {self.device}"))
            print(info_msg(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"))
            print(info_msg(f"Dataset size: {len(dataset):,} sequences"))
            print(info_msg(f"Batch size: {self.config.batch_size}"))
            print(info_msg(f"Gradient accumulation: {self.config.grad_accumulation_steps}"))
            effective_batch = self.config.batch_size * self.config.grad_accumulation_steps
            print(info_msg(f"Effective batch size: {effective_batch}"))
            print(info_msg(f"Steps per epoch: {steps_per_epoch}"))
            print(info_msg(f"Total steps: {total_steps}"))
            print(info_msg(f"Epochs: {epochs}"))
            print(info_msg(f"Learning rate: {self.config.learning_rate}"))
            print(info_msg(f"AMP: {self.scaler is not None}"))
            print("=" * 60)

        # ─────────────────────────────────────────────────────────────
        # STEP 8: THE TRAINING LOOP
        # ─────────────────────────────────────────────────────────────
        # This is where the actual learning happens!
        start_time = time.time()
        self.loss_history = []

        for epoch in range(epochs):
            # Train for one epoch (see _train_epoch below)
            epoch_loss = self._train_epoch(dataloader, epoch, epochs)
            self.loss_history.append(epoch_loss)

            # Call the callback if provided (for progress bars, logging, etc.)
            if callback:
                callback({
                    'epoch': epoch + 1,
                    'loss': epoch_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoint periodically (in case of crash)
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch + 1)

            # Track best loss (for selecting best model)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

        # ─────────────────────────────────────────────────────────────
        # STEP 9: TRAINING COMPLETE - RETURN RESULTS
        # ─────────────────────────────────────────────────────────────
        elapsed = time.time() - start_time

        if self.config.verbose:
            print()
            print("=" * 60)
            print(system_msg("TRAINING COMPLETE"))
            print("=" * 60)
            print(info_msg(f"Total time: {elapsed:.1f}s"))
            print(info_msg(f"Final loss: {self.loss_history[-1]:.4f}"))
            print(info_msg(f"Best loss: {self.best_loss:.4f}"))
            print("=" * 60)

        return {
            'final_loss': self.loss_history[-1],
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'elapsed_time': elapsed,
            'total_steps': self.global_step
        }

    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> float:
        """
        Train for one epoch (one full pass through the dataset).
        
        📖 THE CORE TRAINING LOOP:
        This is where the magic happens! For each batch:
        
        1. FORWARD PASS: Input → Model → Predictions
           The model sees the input tokens and predicts the next tokens
           
        2. LOSS CALCULATION: How wrong were the predictions?
           Cross-entropy loss measures the difference between
           predictions and actual targets
           
        3. BACKWARD PASS: Compute gradients
           Backpropagation calculates how to adjust each weight
           to reduce the loss
           
        4. OPTIMIZER STEP: Update weights
           Move weights in the direction that reduces loss
           
        📐 GRADIENT ACCUMULATION EXPLAINED:
        If you want batch_size=32 but only have memory for 4:
        - Process 8 mini-batches of 4
        - Accumulate gradients (don't update yet)
        - After 8 batches, do one big update
        - Effect: Same as batch_size=32!
        
        Returns:
            Average loss for this epoch
        """
        # ─────────────────────────────────────────────────────────────
        # SET MODEL TO TRAINING MODE
        # ─────────────────────────────────────────────────────────────
        # model.train() enables:
        # - Dropout (randomly zeros neurons - prevents overfitting)
        # - BatchNorm training statistics
        # Without this, the model would be in eval mode and not learn!
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        epoch_start = time.time()

        # ─────────────────────────────────────────────────────────────
        # MAIN BATCH LOOP
        # ─────────────────────────────────────────────────────────────
        for step, batch in enumerate(dataloader):
            # ─────────────────────────────────────────────────────────
            # MOVE DATA TO DEVICE (CPU → GPU if available)
            # ─────────────────────────────────────────────────────────
            # Data must be on the same device as the model
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # ─────────────────────────────────────────────────────────
            # FORWARD PASS + LOSS CALCULATION
            # ─────────────────────────────────────────────────────────
            # Two paths: with AMP (faster, less memory) or without
            if self.scaler is not None:
                # AMP PATH: Use 16-bit floats for speed
                with autocast(device_type=self.device.type):  # Automatic mixed precision context
                    # Forward: input_ids → model → logits (predictions)
                    logits = self.model(input_ids)
                    
                    # Calculate cross-entropy loss
                    # - logits.view(-1, vocab_size): flatten to [batch*seq, vocab]
                    # - labels.view(-1): flatten to [batch*seq]
                    # - ignore_index: don't count padding tokens in loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=getattr(self.tokenizer, 'pad_token_id', 0)
                    )
                    # Divide by accumulation steps (will be summed up later)
                    loss = loss / self.config.grad_accumulation_steps

                # BACKWARD PASS with gradient scaling (AMP)
                # Scaler prevents underflow in 16-bit gradients
                self.scaler.scale(loss).backward()
            else:
                # STANDARD PATH: Full precision (32-bit floats)
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=getattr(self.tokenizer, 'pad_token_id', 0)
                )
                loss = loss / self.config.grad_accumulation_steps
                # Standard backward pass - computes gradients
                loss.backward()

            # Track loss for reporting
            total_loss += loss.item() * self.config.grad_accumulation_steps
            num_batches += 1

            # ─────────────────────────────────────────────────────────
            # GRADIENT ACCUMULATION CHECK
            # ─────────────────────────────────────────────────────────
            # Only update weights every N steps (accumulation)
            if (step + 1) % self.config.grad_accumulation_steps == 0:
                if self.scaler is not None:
                    # AMP: Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                    # Update weights (with AMP scaling)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()  # Adjust scale for next iteration
                else:
                    # Standard: Just clip and step
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                    # Update weights
                    self.optimizer.step()

                # Zero gradients for next accumulation cycle
                self.optimizer.zero_grad()
                # Update learning rate
                self.scheduler.step()
                self.global_step += 1

                # ─────────────────────────────────────────────────────
                # LOGGING: Print progress periodically
                # ─────────────────────────────────────────────────────
                if self.config.verbose and self.global_step % self.config.log_every == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Step {self.global_step:,} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        # ─────────────────────────────────────────────────────────────
        # HANDLE REMAINING GRADIENTS
        # ─────────────────────────────────────────────────────────────
        # If dataset size isn't divisible by accumulation steps,
        # there might be leftover gradients to apply
        if len(dataloader) % self.config.grad_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.global_step += 1

        # ─────────────────────────────────────────────────────────────
        # EPOCH COMPLETE - RETURN AVERAGE LOSS
        # ─────────────────────────────────────────────────────────────
        epoch_loss = total_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start

        if self.config.verbose:
            print(info_msg(f"Epoch {epoch + 1}/{total_epochs} | Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s"))

        return epoch_loss

    def _save_checkpoint(self, epoch: int):
        """
        Save training checkpoint for recovery/resume.
        
        📖 WHAT THIS SAVES:
        - epoch: Which epoch we just finished
        - model_state_dict: All model weights
        - optimizer_state_dict: Optimizer state (momentum, etc.)
        - loss: Current loss value
        - global_step: Total steps taken
        - config: Training configuration
        
        💡 WHY CHECKPOINTS MATTER:
        Training can take hours/days. If it crashes, checkpoints
        let you resume from where you left off instead of starting over!
        
        🔗 CONNECTS TO:
          → Saved to config.checkpoint_dir (usually models/checkpoints/)
          ← Called by train() every config.save_every epochs
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Bundle everything needed to resume training
        checkpoint = {
            'epoch': epoch,                                    # Where we stopped
            'model_state_dict': self.model.state_dict(),       # Model weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'loss': self.loss_history[-1] if self.loss_history else None,
            'global_step': self.global_step,                   # Step counter
            'config': self.config.__dict__                     # Settings
        }

        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        logger.info(f"Saved checkpoint to {path}")

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model weights.
        
        📖 WHAT THIS DOES:
        Saves ONLY the model weights (not optimizer, not config).
        This is the final model file you'll use for inference.
        
        💡 TIP:
        Use checkpoints during training, use this for final model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")


# =============================================================================
# 🎯 CONVENIENCE FUNCTIONS - Easy-to-use training interface
# =============================================================================
# These functions wrap the Trainer class with sensible defaults,
# making it easy to train a model with one function call.

def train_model(
    data_path: Optional[Union[str, Path]] = None,
    epochs: int = 30,
    model_size: str = "small",
    output_path: Optional[Union[str, Path]] = None,
    train_tokenizer_first: bool = True,
    force: bool = False,
    **kwargs
) -> dict[str, Any]:
    """
    High-level training function - THE EASY WAY TO TRAIN!
    
    📖 WHAT THIS DOES:
    1. Loads your training data
    2. Trains a tokenizer (optional)
    3. Creates a model
    4. Trains the model
    5. Saves everything
    
    📐 SIMPLE USAGE:
        # Train with defaults
        results = train_model("data/training.txt")
        
        # Custom training
        results = train_model(
            data_path="my_data.txt",
            epochs=50,
            model_size="medium",
            learning_rate=0.0001
        )
    
    🔗 CONNECTS TO:
      → Uses Trainer class (above)
      → Uses create_model from forge_ai/core/model.py
      → Uses get_tokenizer from forge_ai/core/tokenizer.py
      ← Called from run.py --train or GUI

    Args:
        data_path: Path to training data file (default: data/data.txt)
        epochs: Number of training epochs (must be > 0)
        model_size: Model size preset:
            - "nano", "micro", "tiny": Very small, fast training
            - "small": Default, good balance
            - "medium", "large": Better quality, slower
            - "xl", "xxl": Best quality, needs good GPU
        output_path: Where to save the trained model
        train_tokenizer_first: Train a tokenizer on your data (recommended)
        force: Train even if model already exists
        **kwargs: Additional TrainingConfig parameters

    Returns:
        Dictionary with results:
            - status: 'success', 'skipped', or 'failed'
            - model_path: Path to saved model
            - final_loss: Final training loss
            - epochs_completed: Number of epochs completed

    Raises:
        ValueError: If parameters are invalid (epochs, paths)
        TypeError: If parameter types are incorrect
        FileNotFoundError: If data file doesn't exist
        RuntimeError: If training fails
    """
    # ─────────────────────────────────────────────────────────────────
    # VALIDATION: Check inputs before doing anything
    # ─────────────────────────────────────────────────────────────────
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    if not isinstance(model_size, str):
        raise TypeError(f"model_size must be a string, got {type(model_size).__name__}")

    # ─────────────────────────────────────────────────────────────────
    # DEFAULT PATHS: Use sensible defaults if not specified
    # ─────────────────────────────────────────────────────────────────
    if data_path is None:
        data_path = DATA_DIR / "data.txt"
    data_path = Path(data_path)

    if output_path is None:
        output_path = MODELS_DIR / f"{model_size}_forge.pth"
    output_path = Path(output_path)

    # ─────────────────────────────────────────────────────────────────
    # FILE VALIDATION: Make sure data exists and is readable
    # ─────────────────────────────────────────────────────────────────
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Please create a training data file or specify a valid path."
        )

    if not data_path.is_file():
        raise ValueError(f"data_path must be a file, got directory: {data_path}")

    # Check file size (empty files won't train anything useful)
    file_size = data_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Training data file is empty: {data_path}")

    if file_size < 100:
        logger.warning(
            f"Training data file is very small ({file_size} bytes). "
            f"Training may not be effective."
        )

    # ─────────────────────────────────────────────────────────────────
    # CHECK EXISTING MODEL: Skip if already trained (unless force=True)
    # ─────────────────────────────────────────────────────────────────
    if output_path.exists() and not force:
        logger.warning(f"Model already exists at {output_path}")
        logger.info("Use force=True to retrain")
        return {"status": "skipped", "path": str(output_path)}

    # Ensure output directory exists
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Cannot create output directory: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # LOAD TRAINING DATA
    # ─────────────────────────────────────────────────────────────────
    try:
        texts = [data_path.read_text(encoding='utf-8')]
        logger.info(f"Loaded {len(texts[0]):,} characters from {data_path}")
    except (UnicodeDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to read training data: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # TOKENIZER: Train or load
    # ─────────────────────────────────────────────────────────────────
    # Training a tokenizer on YOUR data gives better results than
    # using a generic one, especially for specialized domains
    if train_tokenizer_first:
        logger.info("Training tokenizer...")
        try:
            tokenizer = train_tokenizer(
                data_paths=[str(data_path)],
                vocab_size=8000,
                tokenizer_type="bpe"
            )
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise RuntimeError(f"Tokenizer training failed: {e}") from e
    else:
        try:
            tokenizer = get_tokenizer()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer: {e}\n"
                f"Try setting train_tokenizer_first=True"
            ) from e

    # ─────────────────────────────────────────────────────────────────
    # CREATE MODEL: Initialize with random weights
    # ─────────────────────────────────────────────────────────────────
    logger.info(f"Creating {model_size} model...")
    try:
        model = create_model(model_size, vocab_size=tokenizer.vocab_size)
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(f"Model creation failed: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # TRAIN: This is where the magic happens!
    # ─────────────────────────────────────────────────────────────────
    config = TrainingConfig(
        epochs=epochs,
        **{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)}
    )

    try:
        trainer = Trainer(model, tokenizer, config)
        results = trainer.train(texts)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training failed: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # SAVE EVERYTHING: Model and tokenizer
    # ─────────────────────────────────────────────────────────────────
    try:
        trainer.save_model(output_path)
        logger.info(f"Model saved to {output_path}")
    except OSError as e:
        raise RuntimeError(f"Failed to save model: {e}") from e

    # Save tokenizer alongside model (so they stay together)
    tokenizer_path = output_path.parent / f"{output_path.stem}_tokenizer.json"
    if hasattr(tokenizer, 'save'):
        try:
            tokenizer.save(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

    results['model_path'] = str(output_path)
    results['tokenizer_path'] = str(tokenizer_path)

    return results


def load_trained_model(
    model_path: Union[str, Path],
    device: Optional[str] = None
) -> tuple:
    """
    Load a trained model and tokenizer.
    
    📖 WHAT THIS DOES:
    Loads a previously trained model from disk so you can use it
    for inference (generating text).
    
    📐 USAGE:
        model, tokenizer = load_trained_model("models/small_forge.pth")
        # Now you can generate text with the model
    
    🔗 CONNECTS TO:
      → Model weights saved by Trainer.save_model()
      → Tokenizer saved during train_model()
      ← Used by ForgeEngine in inference.py

    Args:
        model_path: Path to saved model (.pth file)
        device: Device to load to ("cuda" or "cpu")

    Returns:
        (model, tokenizer) tuple ready for inference
    """
    from .model_registry import safe_load_weights
    model_path = Path(model_path)

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─────────────────────────────────────────────────────────────────
    # LOAD MODEL WEIGHTS
    # ─────────────────────────────────────────────────────────────────
    state_dict = safe_load_weights(model_path, map_location=device)

    # ─────────────────────────────────────────────────────────────────
    # INFER MODEL SIZE FROM WEIGHTS
    # ─────────────────────────────────────────────────────────────────
    # We need to know the model architecture to load the weights.
    # We can figure this out from the embedding layer dimensions.
    embed_key = None
    for key in state_dict.keys():
        if 'embed' in key.lower() or 'token' in key.lower():
            embed_key = key
            break

    if embed_key:
        vocab_size, hidden_dim = state_dict[embed_key].shape
    else:
        # Default values if we can't detect
        vocab_size = 8000
        hidden_dim = 512

    # Find matching preset based on hidden dimension
    model_size = "small"  # Default
    for name, preset in MODEL_PRESETS.items():
        preset_dim = preset.dim if hasattr(preset, 'dim') else preset.get('hidden_dim', 512)
        if preset_dim == hidden_dim:
            model_size = name
            break

    # ─────────────────────────────────────────────────────────────────
    # CREATE MODEL AND LOAD WEIGHTS
    # ─────────────────────────────────────────────────────────────────
    model = create_model(model_size, vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    # ─────────────────────────────────────────────────────────────────
    # LOAD MATCHING TOKENIZER
    # ─────────────────────────────────────────────────────────────────
    # Look for tokenizer saved alongside the model
    tokenizer_path = model_path.parent / f"{model_path.stem}_tokenizer.json"
    if tokenizer_path.exists():
        from .advanced_tokenizer import AdvancedBPETokenizer
        tokenizer = AdvancedBPETokenizer(vocab_file=tokenizer_path)
    else:
        # Fall back to default tokenizer
        tokenizer = get_tokenizer()

    return model, tokenizer


# =============================================================================
# 📦 MODULE EXPORTS - What's available when you import this module
# =============================================================================

__all__ = [
    # Main classes
    "Trainer",           # The training engine
    "TrainingConfig",    # Training settings
    "TextDataset",       # Dataset for plain text
    "QADataset",         # Dataset for Q&A format
    "CosineWarmupScheduler",  # Learning rate scheduler

    # Functions
    "train_model",       # High-level training function
    "load_trained_model",  # Load a trained model

    # Constants
    "MODELS_DIR",        # Where models are saved
    "DATA_DIR",          # Where training data lives
]
