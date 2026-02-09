# Enigma AI Engine Training Quirks & Common Issues

This document explains the **weird things** in Enigma AI Engine that need special attention during training and usage.

## 🔴 Critical Issues

### 1. Model Outputs Raw Tensors/Numbers

**Symptom:** Model outputs something like `tensor([0.234, -0.125, ...])` instead of text.

**Cause:** 
- Model not fully loaded
- Missing tokenizer
- Using wrong inference method

**Fix:**
```python
# Make sure to use the proper inference path
from enigma_engine.core.inference import EnigmaEngine

engine = EnigmaEngine()
engine.load_model("your_model")

# Use generate(), not forward()
response = engine.generate("Hello", max_length=100)
```

### 2. Tool Calls Completely Ignored

**Symptom:** AI responds conversationally but never uses tools even when asked to "search the web" or "generate an image".

**Cause:** Model not trained on tool data OR tool training data too sparse.

**Fix:**
1. Train on `data/tool_training_data.txt`
2. Make sure training data has MANY examples (100+ per tool)
3. Include both positive (use tool) and negative (don't use tool) examples:

```
# POSITIVE - when to use tool
Q: Search the web for cat pictures
A: <tool_call>{"tool": "web_search", "params": {"query": "cat pictures"}}</tool_call>

# NEGATIVE - when NOT to use tool  
Q: What is a cat?
A: A cat is a small domesticated carnivorous mammal. They are popular pets known for...
```

### 3. Repetition Loops

**Symptom:** Model repeats the same phrase over and over: "I can help you I can help you I can help you..."

**Cause:** 
- Temperature too low
- Repetition penalty not enabled
- Training data has repeated patterns

**Fix:**
```python
# Adjust generation parameters
response = engine.generate(
    prompt,
    temperature=0.7,  # Higher = more random
    repetition_penalty=1.1,  # Penalize repetition
    top_p=0.9,  # Nucleus sampling
)
```

### 4. Wrong Model Type for Feature

**Symptom:** Vision tab doesn't work, code is garbage, etc.

**Cause:** Using wrong model type for the task:
- **Vision:** Needs vision-language model (Qwen2-VL, LLaVA), NOT regular text model
- **Code:** Benefits from code-trained model (DeepSeek Coder, CodeLlama)
- **Chat:** Needs instruction-tuned model

**Fix:**
Use the right model for each feature. See Model Manager tab or:
```python
from enigma_engine.core.ai_integration import AIIntegration

ai = AIIntegration()
ai.setup_feature("vision", "huggingface:Qwen/Qwen2-VL-2B-Instruct")
ai.setup_feature("code", "huggingface:deepseek-ai/deepseek-coder-1.3b-instruct")
```

---

## ⚠️ Training Data Quirks

### Format Must Be Exact

**Wrong:**
```
User: Hello
Assistant: Hi there!
```

**Correct:**
```
Q: Hello
A: Hi there!
```

The training code looks for `Q:` and `A:` specifically!

### Tool Call JSON Must Be Valid

**Wrong:**
```
<tool_call>{tool: "search", params: {query: "test"}}</tool_call>
```

**Correct:**
```
<tool_call>{"tool": "search", "params": {"query": "test"}}</tool_call>
```

Use **double quotes** for JSON!

### Comments Are Ignored (Good!)

Lines starting with `#` are ignored:
```
# This is a comment - won't be trained on
Q: Hello
A: Hi!
```

### Empty Lines Between Examples

Good practice:
```
Q: Hello
A: Hi!

Q: How are you?
A: I'm great!
```

---

## 🟡 Module System Quirks

### Can't Load Conflicting Modules

**Issue:** `image_gen_local` and `image_gen_api` both provide "image generation" - can't load both.

**Fix:** Only load one at a time:
```python
manager.load('image_gen_local')  # OR
manager.load('image_gen_api')    # NOT BOTH
```

### Dependencies Must Load First

**Issue:** Loading `inference` before `model` fails.

**Fix:** ModuleManager handles this automatically, but if manually loading:
```python
manager.load('model')      # First
manager.load('tokenizer')  # Second  
manager.load('inference')  # Third
```

---

## 🟢 HuggingFace Model Quirks

### Some Models Need Auth Token

**Issue:** `meta-llama/Llama-2-7b` requires HuggingFace login.

**Fix:**
```bash
huggingface-cli login
```

Or in code:
```python
from huggingface_hub import login
login(token="your_token")
```

### VRAM Requirements

| Model Size | Minimum VRAM |
|------------|--------------|
| 1-2B | 4-6 GB |
| 7B | 8-12 GB |
| 13B | 16-24 GB |
| 70B | 80+ GB |

**Tip:** Use 4-bit quantized models to reduce VRAM:
```python
# In Model Manager, enable "Load in 4-bit"
```

### First Load Is Slow

**Issue:** Model takes 5-10 minutes to download first time.

**Cause:** Downloading from HuggingFace Hub.

**Fix:** Be patient! Cached after first download in `~/.cache/huggingface/`

---

## 🔧 GUI Quirks

### Chat Shows "Thinking..." Before User Message

**Fixed in latest!** Now uses 100ms delay so user message appears first.

### X Button in Quick Chat Opens Menu

**Fixed in latest!** Now closes directly (right-click for menu).

### Text Doesn't Wrap in Quick Chat

**Fixed in latest!** Added proper word wrap mode.

---

## 🎯 Validation Script

Run this to check your training data:

```bash
python scripts/validate_training_data.py
```

This checks:
- Q/A format
- Tool call JSON validity
- Balanced tags
- Dataset size
- Common mistakes

---

## 📊 Minimum Recommended Training Data

| Feature | Min Examples | Ideal |
|---------|--------------|-------|
| Chat personality | 200 Q/A | 1000+ |
| Tool use (per tool) | 50 | 200+ |
| Code generation | 100 | 500+ |

---

## 🐛 Debugging Checklist

1. **Model not responding?**
   - Check if model is loaded (`engine.model is not None`)
   - Check GPU memory (`nvidia-smi`)
   - Try smaller model

2. **Tools not working?**
   - Validate training data
   - Check tool is registered
   - Check model was trained on tools

3. **Vision not working?**
   - Confirm using vision-language model
   - Check image path is valid
   - Try different image format

4. **Training fails?**
   - Check VRAM (reduce batch size)
   - Validate data format
   - Check file encoding (UTF-8)

---

## 📝 Quick Reference

```python
# Check integration status
from enigma_engine.core.ai_integration import print_integration_status
print_integration_status()

# Validate training data
python scripts/validate_training_data.py

# Quick test GUI
python scripts/quick_test_gui.py --no-model

# Train model
python run.py --train

# Run with GUI
python run.py --gui
```
