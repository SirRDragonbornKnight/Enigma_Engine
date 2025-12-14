"""
Simple tokenizer wrapper using tokenizers / transformers PreTrainedTokenizerFast if available.
Falls back to a whitespace tokenizer if no tokenizers available.
"""
from pathlib import Path
from typing import List
import logging

try:
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast
    HAVE_HF = True
except Exception:
    HAVE_HF = False

logger = logging.getLogger(__name__)

VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab_model"


def build_tokenizer_from_files(data_files: List[str], vocab_size: int = 5000):
    """Train a ByteLevel BPE tokenizer and save to vocab_model/"""
    if not HAVE_HF:
        raise RuntimeError("tokenizers/transformers package required to build tokenizer")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=data_files, vocab_size=vocab_size, min_frequency=2,
                    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"])
    VOCAB_DIR.mkdir(exist_ok=True)
    tokenizer.save_model(str(VOCAB_DIR))
    return str(VOCAB_DIR)


def load_tokenizer():
    """Return a PreTrainedTokenizerFast if trained vocab exists, else return a simple whitespace tokenizer"""
    if HAVE_HF and (VOCAB_DIR / "vocab.json").exists():
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(
            vocab_file=str(VOCAB_DIR / "vocab.json"),
            merges_file=str(VOCAB_DIR / "merges.txt"),
            pad_token="<pad>",
            eos_token="</s>"
        )
        tok.pad_token = tok.eos_token
        return tok
    else:
        # fallback simple tokenizer
        class SimpleTok:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
            def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
                toks = text.split()
                ids = list(range(1, len(toks)+1))
                return {"input_ids": ids}
            def decode(self, ids, skip_special_tokens=False):
                return " ".join(f"<tok{idx}>" for idx in ids)
        logger.warning("Using fallback whitespace tokenizer. Train a real tokenizer for production.")
        return SimpleTok()
