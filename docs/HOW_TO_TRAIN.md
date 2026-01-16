# How to Train Your ForgeAI

Complete guide to training and fine-tuning your ForgeAI model.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Data Preparation](#training-data-preparation)
3. [Training Methods](#training-methods)
4. [Training Parameters](#training-parameters)
5. [Training Data Files](#training-data-files)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Command Line Training
```bash
# Basic training with defaults
python run.py --train

# Train specific model size
python run.py --train --model medium

# Train with custom epochs
python run.py --train --epochs 50

# Train with custom data
python run.py --train --data path/to/your_data.txt
```

### GUI Training
1. Launch the GUI: `python run.py --gui`
2. Go to the **Train** tab
3. Select your training data file
4. Configure epochs and learning rate
5. Click "Start Training"
6. Monitor progress in real-time

---

## Training Data Preparation

### Format Requirements

Your training data should be plain text with one of these formats:

#### Conversation Format (Recommended)
```
Q: What is Python?
A: Python is a high-level programming language known for its simplicity and readability.

Q: How do I create a list in Python?
A: You can create a list using square brackets: my_list = [1, 2, 3, 4]
```

#### Dialogue Format
```
User: Tell me about machine learning
Assistant: Machine learning is a subset of artificial intelligence that enables systems to learn from data...

User: What are neural networks?
Assistant: Neural networks are computing systems inspired by biological neural networks...
```

#### Plain Text Format
For general text, just provide clean paragraphs:
```
The ForgeAI is a flexible AI framework. It can be trained on various types of data.
Each model size offers different capabilities and resource requirements.
```

### Data Quality Tips

✅ **DO:**
- Use clear, grammatically correct text
- Include diverse topics for general models
- Focus on specific domains for specialized models
- Aim for 1000+ lines minimum
- Use consistent formatting throughout

❌ **DON'T:**
- Mix multiple formats in one file
- Include garbled or corrupted text
- Use extremely short responses (< 5 words)
- Train on copyrighted content without permission
- Include personally identifiable information (PII)

---

## Training Methods

### 1. Train from Scratch (Build)

Creates a completely new model:

```bash
python run.py --build --model small
```

This will:
1. Build a fresh tokenizer from your data
2. Initialize model weights randomly
3. Train for specified epochs
4. Save the trained model

**When to use:** First time setup or completely new personality

### 2. Continue Training (Fine-tune)

Continues training an existing model:

```bash
python run.py --train --model small
```

This will:
1. Load existing model and tokenizer
2. Continue training from current weights
3. Update the model with new data
4. Save checkpoints periodically

**When to use:** Adding new knowledge, adjusting behavior, personality development

### 3. Transfer Learning

Train a small model, then grow it:

```bash
# Train tiny model first
python run.py --train --model tiny --epochs 100

# Grow to small (transfers knowledge)
# (Use Model Manager in GUI)
```

---

## Training Parameters

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `small` | Model size: nano, micro, tiny, small, medium, large, xl, xxl |
| `--epochs` | `30` | Number of training epochs (full passes through data) |
| `--data` | `data/combined_action_training.txt` | Path to training data file |
| `--force` | `False` | Force retrain even if model exists |

### Advanced Parameters (config.py)

Edit `forge_ai/config.py` for fine-tuning:

```python
# Learning rate (default: 1e-4)
LEARNING_RATE = 0.0001  # Lower = slower but more stable

# Batch size (default: 32)
BATCH_SIZE = 32  # Higher = faster but needs more memory

# Gradient accumulation (default: 4)
GRAD_ACCUM_STEPS = 4  # Simulates larger batch sizes

# Sequence length (default: 512)
MAX_SEQ_LEN = 512  # Longer = more context, more memory

# Mixed precision (default: True)
USE_AMP = True  # Automatic mixed precision for speed

# Checkpoint frequency (default: 100)
SAVE_EVERY = 100  # Save checkpoint every N epochs
```

### Learning Rate Guide

| Use Case | Recommended LR |
|----------|---------------|
| First training | `1e-4` (default) |
| Fine-tuning | `5e-5` to `1e-5` |
| Small adjustments | `1e-5` to `1e-6` |
| Unstable training | `5e-5` (reduce) |

---

## Training Data Files

The `data/` directory contains different types of training data:

### Core Training Data

**`combined_action_training.txt`**
- Complete conversation examples
- Tool usage demonstrations
- Multi-step reasoning
- **Use for:** General-purpose training

### Specialized Training Data

**`tool_training_examples.txt`**
- Comprehensive tool usage examples (69+ examples)
- All available tools with proper formatting
- Success and error handling cases
- **Use for:** Teaching AI to use tools

**`tool_training_data.txt`**
- Additional tool use cases
- Natural language tool requests
- Complex multi-tool scenarios
- **Use for:** Advanced tool usage

**`personality_development.txt`**
- Personality traits and behaviors
- Emotional responses
- Character consistency
- **Use for:** Developing unique AI personalities

**`self_awareness_training.txt`**
- Self-reflection examples
- Understanding of being an AI
- Honest capability assessment
- **Use for:** Self-aware AI behavior

### Creating Custom Training Data

1. Create a new `.txt` file in the `data/` directory
2. Use the conversation format (Q:/A:)
3. Include 1000+ lines minimum
4. Test with: `python run.py --train --data data/your_file.txt`

Example custom file:
```
Q: What is your favorite color?
A: As an AI, I don't perceive colors, but I appreciate the concept of blue for its association with calm and clarity.

Q: Tell me about yourself
A: I'm an AI trained on the ForgeAI framework. I'm designed to be helpful, honest, and continuously learning.
```

---

## Best Practices

### 1. Start Small, Scale Up
- Begin with `tiny` or `small` models
- Test training on subset of data first
- Grow to larger models once confident

### 2. Monitor Training
- Watch the loss curve (should decrease)
- Check generation quality every 10-20 epochs
- Stop if loss plateaus or increases

### 3. Use Checkpoints
- Training saves checkpoints every 100 epochs
- Can resume if training interrupts
- Test intermediate checkpoints

### 4. Combine Data Sources
Merge multiple training files:
```bash
cat data/tool_training_data.txt data/personality_development.txt > data/my_training.txt
python run.py --train --data data/my_training.txt
```

### 5. Incremental Training
- Train in stages with different data
- Add capabilities gradually
- Test between training sessions

---

## Troubleshooting

### Problem: Loss Not Decreasing

**Symptoms:** Loss stays constant or increases
**Solutions:**
- Reduce learning rate to `5e-5` or `1e-5`
- Check data quality (no corrupted text)
- Ensure sufficient data (1000+ lines)
- Try smaller batch size

### Problem: Out of Memory (OOM)

**Symptoms:** CUDA OOM error or system freeze
**Solutions:**
- Reduce `BATCH_SIZE` in config (try 16 or 8)
- Reduce `MAX_SEQ_LEN` (try 256 or 128)
- Use smaller model size
- Close other GPU applications

### Problem: Training Too Slow

**Symptoms:** Epochs taking hours
**Solutions:**
- Enable AMP (mixed precision): `USE_AMP = True`
- Increase batch size if memory allows
- Use GPU instead of CPU
- Reduce `MAX_SEQ_LEN` if not needed

### Problem: Model Not Improving

**Symptoms:** Generated text quality poor after training
**Solutions:**
- Train for more epochs (50-100+)
- Check if data is too diverse or unfocused
- Verify data format is correct
- Try different learning rate

### Problem: Model Forgot Previous Knowledge

**Symptoms:** Model performs worse after training
**Solutions:**
- Lower learning rate significantly (`1e-5`)
- Reduce epochs for fine-tuning (10-20)
- Include previous training data in new training
- Use smaller gradient steps

---

## Training Tips by Model Size

### Nano/Micro (< 5M params)
- **Epochs:** 50-100
- **Data:** 500-1000 lines
- **Time:** Minutes
- **Use:** Testing, embedded devices

### Tiny/Mini (5-10M params)
- **Epochs:** 30-50
- **Data:** 1000-2000 lines
- **Time:** 10-30 minutes
- **Use:** Raspberry Pi, quick experiments

### Small (27M params)
- **Epochs:** 30-50
- **Data:** 2000-5000 lines
- **Time:** 30-60 minutes
- **Use:** Desktop, good balance

### Medium (85M params)
- **Epochs:** 30-40
- **Data:** 5000-10000 lines
- **Time:** 1-2 hours
- **Use:** Mid-range GPU, quality results

### Large+ (200M+ params)
- **Epochs:** 20-30
- **Data:** 10000+ lines
- **Time:** 2-8+ hours
- **Use:** High-end GPU, production quality

---

## Advanced Topics

### Multi-Stage Training

1. **Stage 1: Foundation** (General knowledge)
   ```bash
   python run.py --train --data data/combined_action_training.txt --epochs 30
   ```

2. **Stage 2: Tool Mastery**
   ```bash
   python run.py --train --data data/tool_training_examples.txt --epochs 20
   ```

3. **Stage 3: Personality**
   ```bash
   python run.py --train --data data/personality_development.txt --epochs 15
   ```

### Curriculum Learning

Start with easy examples, progress to complex:
```bash
# Week 1: Simple Q&A
python run.py --train --data data/simple_qa.txt --epochs 30

# Week 2: Add reasoning
python run.py --train --data data/reasoning.txt --epochs 20

# Week 3: Add tools
python run.py --train --data data/tool_training_data.txt --epochs 20
```

### Model Evaluation

Test your model quality:
```python
from forge_ai.core.model_registry import ModelRegistry

registry = ModelRegistry()
model, _ = registry.load_model("your_model")

# Test prompts
prompts = [
    "What is Python?",
    "How do I use a for loop?",
    "Explain machine learning"
]

for prompt in prompts:
    response = model.generate(prompt)
    print(f"Q: {prompt}\nA: {response}\n")
```

---

## Related Documentation

- **[Training Data Format](TRAINING_DATA_FORMAT.md)** - Detailed format specifications
- **[Module Guide](MODULE_GUIDE.md)** - Using modules and capabilities
- **[Tool Use](TOOL_USE.md)** - Teaching AI to use tools
- **[Personality Development](PERSONALITY.md)** - Creating unique AI personalities

---

## Quick Reference

```bash
# Common training commands
python run.py --train                    # Basic training
python run.py --train --model medium     # Train medium model
python run.py --train --epochs 50        # Train for 50 epochs
python run.py --build --model small      # Build from scratch
python run.py --gui                      # Use GUI trainer

# Validation
python -m pytest tests/ -v               # Run tests
python run.py --run                      # Test inference

# Files
data/combined_action_training.txt        # Main training data
data/tool_training_examples.txt          # Tool usage examples
models/sacrifice/                        # Default test model
```

---

**Happy Training!** 🚀

For questions or issues, check the [troubleshooting section](#troubleshooting) or create an issue on GitHub.
