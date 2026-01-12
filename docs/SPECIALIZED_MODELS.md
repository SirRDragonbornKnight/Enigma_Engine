# Specialized Models Guide

## Overview

AI Tester supports training and deploying **specialized smaller AI models** for specific tasks like intent classification, vision captioning, and code generation. All specialized models share the same tokenizer architecture, allowing seamless interoperability across the system.

This approach offers several advantages:
- **Efficiency**: Use tiny models (1-5M params) for simple classification tasks
- **Speed**: Specialized models are faster than routing everything through a large model
- **Modularity**: Train and deploy task-specific models independently
- **Scalability**: Mix local and cloud models based on your needs

## Architecture

### Shared Tokenizer
All models in AI Tester use the same tokenizer, ensuring:
- Consistent vocabulary across all models
- Easy model switching without re-tokenization
- Simplified training pipeline
- Better interoperability

The shared tokenizer is located at `enigma/vocab_model/bpe_vocab.json` (or character tokenizer as fallback).

### Model Types

| Type | Purpose | Recommended Size | Typical Use |
|------|---------|------------------|-------------|
| **router** | Intent classification | nano (~1M params) | Route user requests to appropriate tools |
| **vision** | Image captioning | tiny (~5M params) | Generate descriptions from vision features |
| **code** | Code generation | small (~27M params) | Generate and explain code |
| **math** | Mathematical reasoning | small (~27M params) | Solve math problems |

### Model Sizes

Enigma supports 15 model size presets:

| Size | Parameters | Use Case |
|------|-----------|----------|
| nano | ~1M | Microcontrollers, basic classification |
| micro | ~2M | IoT devices, simple tasks |
| tiny | ~5M | Raspberry Pi, edge devices |
| mini | ~10M | Mobile devices |
| small | ~27M | Entry GPU, good for most tasks |
| medium | ~85M | Mid-range GPU, high quality |
| large+ | 200M+ | Datacenter, production systems |

## Quick Start

### 1. Training a Router Model

The router model classifies user intent (chat, vision, image, code, etc.):

```bash
# Train nano router for intent classification
python scripts/train_specialized_model.py \
    --type router \
    --data data/specialized/router_training.txt \
    --model-size nano \
    --epochs 50
```

**Router Training Data Format** (`data/specialized/router_training.txt`):
```
Q: draw me a picture of a dog
A: [E:tool]image

Q: what do you see in this image
A: [E:tool]vision

Q: write a Python function
A: [E:tool]code

Q: how are you today
A: [E:tool]chat
```

Each Q/A pair trains the model to classify text into tool categories.

### 2. Training a Vision Model

The vision model generates natural language descriptions from vision features:

```bash
# Train tiny vision model for image captioning
python scripts/train_specialized_model.py \
    --type vision \
    --data data/specialized/vision_training.txt \
    --model-size tiny \
    --epochs 40
```

**Vision Training Data Format** (`data/specialized/vision_training.txt`):
```
Q: [E:vision] person, outdoors, smiling, grass
A: I see a person outdoors who is smiling, standing in grass.

Q: [E:vision] cat, sleeping, couch, living room
A: There's a cat sleeping on a couch in what appears to be a living room.

Q: [E:vision] sunset, ocean, orange sky, clouds
A: The image shows a beautiful sunset over the ocean with an orange sky and clouds.
```

The `[E:vision]` prefix indicates vision input, followed by comma-separated features.

### 3. Training a Code Model

The code model specializes in generating and explaining code:

```bash
# Train small code model
python scripts/train_specialized_model.py \
    --type code \
    --data data/specialized/code_training.txt \
    --model-size small \
    --epochs 40
```

**Code Training Data Format** (`data/specialized/code_training.txt`):
```
Q: write a Python function to sort a list
A: Here's a function to sort a list:
```python
def sort_list(items):
    return sorted(items)
```

Q: create a function to reverse a string
A: Here's how to reverse a string:
```python
def reverse_string(text):
    return text[::-1]
```
```

## Using Specialized Models

### In Python Code

```python
from ai_tester.core.inference import AITesterEngine

# Enable routing with specialized models
engine = AITesterEngine(use_routing=True)

# Router automatically detects intent and routes to specialized model
response = engine.generate("draw me a cat")  # Routes to image tool

response = engine.generate("write a Python function to sort a list")  # Uses code model

response = engine.generate("how are you today?")  # Uses main chat model
```

### Direct Router Usage

```python
from ai_tester.core.tool_router import get_router

# Get router with specialized models
router = get_router(use_specialized=True)

# Classify intent
intent = router.classify_intent("draw me a sunset")
# Returns: "image"

# Generate code
code = router.generate_code("write a function to reverse a string")
# Returns: Python code

# Describe image (from features)
description = router.describe_image("cat, sleeping, couch")
# Returns: "There's a cat sleeping on a couch..."
```

### Configuration

Specialized models are configured in `information/specialized_models.json`:

```json
{
  "enabled": true,
  "shared_tokenizer": "enigma/vocab_model/bpe_vocab.json",
  "models": {
    "router": {
      "path": "models/specialized/intent_router_nano.pth",
      "size": "nano",
      "description": "Classifies user intent",
      "max_seq_len": 256
    },
    "vision": {
      "path": "models/specialized/vision_caption_tiny.pth",
      "size": "tiny",
      "description": "Generates descriptions from vision features",
      "max_seq_len": 512
    },
    "code": {
      "path": "models/specialized/code_gen_small.pth",
      "size": "small",
      "description": "Generates and explains code",
      "max_seq_len": 1024
    }
  }
}
```

## Creating Training Data

### Best Practices

1. **Quantity**: Aim for 100+ Q/A pairs minimum, 500+ for good results
2. **Diversity**: Cover various phrasings and edge cases
3. **Consistency**: Use consistent formatting for Q/A pairs
4. **Quality**: Ensure answers are accurate and relevant
5. **Balance**: Include examples for all categories/intents

### Router Training Data Tips

- Include variations: "draw me", "create an image", "generate a picture"
- Cover edge cases: questions that could be ambiguous
- Test with real user queries from your application
- Add more examples for poorly classified intents

Example categories:
- `chat`: General conversation, questions, explanations
- `image`: Drawing, painting, illustration requests
- `vision`: "What do you see", "describe this image"
- `code`: Programming requests, debugging, refactoring
- `video`: Animation, video generation
- `audio`: TTS, music generation

### Vision Training Data Tips

- Use realistic feature sets from actual image classifiers
- Include common objects and scenes
- Vary description styles (short, detailed, poetic)
- Train on your specific use case (screenshots, photos, diagrams)

### Code Training Data Tips

- Include multiple programming languages
- Cover common patterns (sorting, searching, data structures)
- Add edge cases and error handling
- Include both implementations and explanations
- Vary complexity levels

## Advanced Topics

### Model Size Selection

Choose based on your task complexity and hardware:

**Router Model:**
- Use **nano** (1M params) - intent classification is simple
- Training time: ~5-10 minutes on CPU
- Inference: Nearly instant

**Vision Model:**
- Use **tiny** (5M params) - good balance for descriptions
- Training time: ~15-30 minutes on CPU/GPU
- Can handle detailed captions

**Code Model:**
- Use **small** (27M params) - code needs more capacity
- Training time: ~30-60 minutes on GPU
- Better at complex patterns

### Training Parameters

```bash
# Fast prototyping (lower quality)
python scripts/train_specialized_model.py \
    --type router \
    --data data/specialized/router_training.txt \
    --model-size nano \
    --epochs 20 \
    --lr 5e-4

# Production quality (more training)
python scripts/train_specialized_model.py \
    --type code \
    --data data/specialized/code_training.txt \
    --model-size small \
    --epochs 60 \
    --batch-size 16 \
    --lr 3e-4
```

### Monitoring Training

The script provides progress information:
- Loss per epoch
- Training speed (tokens/sec)
- Validation perplexity
- Checkpoint locations

Watch for:
- **Decreasing loss**: Model is learning
- **Plateauing loss**: May need more data or longer training
- **Increasing loss**: Learning rate too high, reduce it

### Fine-tuning Pre-trained Models

You can fine-tune an existing model on new data:

1. Train initial model on broad dataset
2. Save checkpoint
3. Resume training with task-specific data
4. Lower learning rate for fine-tuning

### Multi-task Models

You can train a single model on multiple tasks:

**Combined Training Data:**
```
Q: [TASK:classification] draw me a picture
A: image

Q: [TASK:vision] cat, sleeping, couch
A: There's a cat sleeping on a couch.

Q: [TASK:code] write a sort function
A: def sort_list(items): return sorted(items)
```

## Integration with Module System

Specialized models integrate with AI Tester's module system. To enable via module manager:

```python
from ai_tester.modules.manager import ModuleManager

manager = ModuleManager()

# Load tool router module (includes specialized models)
manager.load('tool_router')

# Router automatically uses specialized models if available
```

The module can be toggled on/off in the GUI Module Manager tab.

## Troubleshooting

### "Model not found" Error

**Cause**: Specialized model hasn't been trained yet

**Solution**: Train the model first:
```bash
python scripts/train_specialized_model.py --type router --data data/specialized/router_training.txt --model-size nano
```

### Poor Classification Accuracy

**Causes & Solutions**:
1. **Too little data**: Add more examples (aim for 100+)
2. **Inconsistent formatting**: Standardize Q/A format
3. **Ambiguous examples**: Clarify intent in training data
4. **Model too small**: Try larger size (tiny instead of nano)

### Slow Training

**Solutions**:
- Use GPU if available (specify `--device cuda`)
- Reduce batch size if running out of memory
- Use smaller model size for prototyping
- Enable mixed precision training (automatic on GPU)

### Model Not Loading

**Check**:
1. Path in `information/specialized_models.json` is correct
2. Model file exists at specified path
3. Tokenizer is available
4. Model was trained with compatible version

### Inconsistent Results

**Causes**:
- Temperature too high (try 0.1-0.3 for classification)
- Insufficient training epochs
- Noisy training data
- Model size mismatch with task complexity

## Performance Tips

1. **Router Model**: Keep it tiny (nano) - classification is simple
2. **Batch Inference**: Process multiple requests together
3. **Model Caching**: Models are cached after first load
4. **Lazy Loading**: Models load only when first used
5. **Shared Tokenizer**: Eliminates re-tokenization overhead

## Example Workflows

### Building a Multi-Model System

```bash
# 1. Train router for intent classification
python scripts/train_specialized_model.py \
    --type router \
    --data data/specialized/router_training.txt \
    --model-size nano

# 2. Train vision for image descriptions
python scripts/train_specialized_model.py \
    --type vision \
    --data data/specialized/vision_training.txt \
    --model-size tiny

# 3. Train code for programming tasks
python scripts/train_specialized_model.py \
    --type code \
    --data data/specialized/code_training.txt \
    --model-size small

# 4. Use the system
python -c "
from ai_tester.core.inference import AITesterEngine
engine = AITesterEngine(use_routing=True)
print(engine.generate('write a hello world function'))
"
```

### Iterative Improvement

1. Start with small training set (50 examples)
2. Train and test model
3. Identify misclassifications
4. Add examples for problem cases
5. Retrain with expanded dataset
6. Repeat until satisfied with accuracy

### Domain-Specific Customization

For your specific application:

1. Collect real user queries
2. Manually classify them
3. Create training data matching your patterns
4. Train specialized models
5. Deploy and monitor
6. Continuously improve based on feedback

## API Reference

### Training Script

```bash
python scripts/train_specialized_model.py [OPTIONS]

Required:
  --type {router,vision,code,math}    Model type
  --data PATH                          Training data file

Optional:
  --model-size {nano,tiny,small,...}  Model size (default: optimal for type)
  --epochs N                           Training epochs (default: recommended)
  --batch-size N                       Batch size (default: 8)
  --lr FLOAT                           Learning rate (default: 3e-4)
  --device {auto,cuda,cpu}            Training device (default: auto)
  --output PATH                        Custom output path
  --quiet                              Reduce verbosity
```

### Python API

**AITesterEngine with Routing:**
```python
engine = AITesterEngine(
    use_routing=True,      # Enable specialized models
    enable_tools=True,     # Enable tool system
    device='cuda'          # Use GPU
)

response = engine.generate(
    prompt="write a sort function",
    max_gen=256,
    temperature=0.7
)
```

**Direct Router Access:**
```python
from ai_tester.core.tool_router import get_router

router = get_router(use_specialized=True)

# Classify intent
intent = router.classify_intent("draw a cat")

# Generate code
code = router.generate_code("reverse a string")

# Describe image
desc = router.describe_image("cat, couch, sleeping")
```

## Conclusion

Specialized models offer a powerful way to enhance your AI system with task-specific intelligence while maintaining efficiency. Start with the router model for intent classification, then expand to vision and code models as needed.

Key takeaways:
- **Start small**: Begin with nano/tiny models
- **Iterate**: Improve training data based on results  
- **Share tokenizer**: Enables seamless model switching
- **Monitor performance**: Track classification accuracy
- **Scale as needed**: Upgrade model sizes for complex tasks

For questions or issues, refer to the main AI Tester documentation or open an issue on GitHub.
