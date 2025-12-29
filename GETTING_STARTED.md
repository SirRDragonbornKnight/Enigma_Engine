# 🚀 Getting Started with Enigma Engine

## Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the GUI
```bash
python run.py --gui
```

### 3. Create Your First AI
1. Go to **Models** tab
2. Click **Create New Model**
3. Enter a name (e.g., "mybot")
4. Select size: **small** (recommended for beginners)
5. Click **Create**

### 4. Train Your AI
1. Go to **Training** tab
2. Your model should be selected
3. Set **Epochs**: 30 (more = better but slower)
4. Click **Start Training**
5. Wait for training to complete

### 5. Chat!
1. Go to **Chat** tab
2. Type a message and press Enter
3. Your AI will respond!

---

## Understanding Model Sizes

| Size | Parameters | Training Time* | Quality | Best For |
|------|------------|----------------|---------|----------|
| **tiny** | ~2M | 5-10 min | Basic | Testing, Raspberry Pi |
| **small** | ~10M | 20-40 min | Good | Personal chatbot |
| **medium** | ~50M | 1-2 hours | Better | Serious projects |
| **large** | ~150M | 4-8 hours | Great | Production use |

*Times are approximate for CPU training with default data.

### Which Size Should I Choose?

- **Just experimenting?** → Use `tiny`
- **Want a usable chatbot?** → Use `small`
- **Have a GPU and time?** → Use `medium` or `large`

---

## Understanding Training

### What is Loss?

Loss measures how wrong your model is. **Lower is better!**

| Loss Value | What It Means |
|------------|---------------|
| 8-10 | Just started, random guessing |
| 4-6 | Learning basic patterns |
| 2-3 | Getting better, understanding structure |
| 1-2 | Good! Model has learned well |
| <0.5 | May be overfitting (memorizing, not learning) |

### How Many Epochs?

An **epoch** = one complete pass through your training data.

| Epochs | Result |
|--------|--------|
| 10 | Quick test, rough results |
| 30 | Decent for small datasets |
| 50-100 | Good for most uses |
| 200+ | Diminishing returns, may overfit |

**Tip**: Watch the loss. If it stops decreasing, you can stop training early.

---

## Adding Your Own Training Data

### Option 1: Use the Data Editor (Easiest)
1. Go to **Data Editor** tab in GUI
2. Add your Q&A pairs
3. Click **Save**
4. Retrain your model

### Option 2: Edit the Text File
1. Open `data/training_data.txt`
2. Add entries in this format:
```
Q: Your question here?
A: The answer you want the AI to give.

Q: Another question?
A: Another answer.
```
3. Save and retrain

### Tips for Good Training Data
- ✅ Be consistent with formatting
- ✅ Include varied examples
- ✅ 100+ Q&A pairs minimum for decent results
- ✅ 1000+ pairs for good results
- ❌ Don't repeat the same Q&A many times
- ❌ Don't use very short answers only

---

## Troubleshooting

### Model gives gibberish output
- **Cause**: Not enough training
- **Fix**: Train for more epochs (50+)

### Model repeats itself
- **Cause**: Overfitting or not enough variety
- **Fix**: Add more diverse training data

### Training is very slow
- **Cause**: Large model on CPU
- **Fix**: Use a smaller model size, or get a GPU

### Out of memory error
- **Cause**: Model too large for your RAM/VRAM
- **Fix**: Use a smaller model size, reduce batch size in Settings

---

## Example: Create a Customer Service Bot

1. Create a `small` model named "support-bot"

2. Add training data like:
```
Q: What are your hours?
A: We're open Monday to Friday, 9 AM to 5 PM.

Q: How do I reset my password?
A: Go to Settings > Security > Reset Password, then follow the prompts.

Q: I need help with my order
A: I'd be happy to help! Please provide your order number and I'll look into it.
```

3. Train for 50 epochs

4. Test in Chat tab!

---

## Next Steps

- 📖 Read [docs/CODE_TOUR.md](docs/CODE_TOUR.md) for developer info
- 🔧 Check [docs/WHAT_NOT_TO_DO.txt](docs/WHAT_NOT_TO_DO.txt) for common mistakes
- 🌐 See [examples/](examples/) for code examples

Happy training! 🎉
