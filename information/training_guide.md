# Training Guide

Train your own AI models using the FORGE page in the GUI or the CLI.

---

## Training Methods

### Supervised Fine-Tuning (SFT)

The standard training method. Feed the model text data and it learns
to predict the next token.

**Data format:** Plain text (.txt) or JSONL with prompt/completion pairs.

```
python run.py --train data/training.txt --epochs 10 --model-size small
```

### LoRA / QLoRA

Low-Rank Adaptation — train a small adapter instead of the full model.
Much faster and uses less memory.

- **LoRA** — adds small trainable matrices to attention layers
- **QLoRA** — same as LoRA but quantizes the base model to 4-bit

### Evolutionary Training

Run N instances on the same task, score outputs, keep the best,
fine-tune on winners, repeat.

**Scoring methods:**
- Rule-based (do tests pass?)
- Self-evaluation
- Consistency across runs
- Length and format checks
- Perplexity scoring

---

## Model Sizes

| Size | Parameters | Good For |
|------|-----------|----------|
| pi_zero | ~500K | Testing, learning |
| nano | ~1M | Simple tasks |
| tiny | ~5M | Basic chat |
| small | ~27M | General use |
| medium | ~85M | Better quality |
| large | ~200M | Best local quality |

---

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Epochs | 10 | Number of full passes through the data |
| Batch Size | 4 | Samples processed together |
| Learning Rate | 0.0001 | How fast the model learns |
| Vocabulary Size | 8000 | BPE tokenizer vocabulary size |

---

## Tips

1. **Start small** — train a tiny model first to test your data
2. **Clean data matters** — garbage in, garbage out
3. **Watch the loss** — it should decrease over epochs
4. **NaN detection** — training auto-stops if loss becomes NaN
5. **Early stopping** — stops if loss stops improving
6. **Save checkpoints** — the FORGE saves progress automatically

---

## Training a Tokenizer

Before training a model, you may want to train a BPE tokenizer on
your specific data:

```
python run.py --train-tokenizer data/training.txt --vocab-size 8000
```

This creates a tokenizer optimized for your vocabulary.
