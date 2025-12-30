"""
Simple tokenizer wrapper.
Uses a lightweight character-level tokenizer by default to avoid heavy dependencies.
Can optionally use HuggingFace tokenizers if available and trained.
"""
from pathlib import Path
from typing import List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab_model"

# Don't import transformers by default - it's heavy and causes lag
HAVE_HF = False


class SimpleTokenizer:
    """
    A lightweight character-level tokenizer.
    Works without any external dependencies.
    Good enough for small models and Pi.
    """
    
    def __init__(self, vocab_file: Path = None):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        
        # Special token IDs
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        
        # Load or create vocabulary
        if vocab_file and vocab_file.exists():
            self._load_vocab(vocab_file)
        else:
            self._create_default_vocab()
        
        self.vocab_size = len(self.token_to_id)
    
    def _create_default_vocab(self):
        """Create a basic character + common word vocabulary."""
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Add printable ASCII characters
        for i, c in enumerate(range(32, 127)):
            token = chr(c)
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # Add common words/subwords
        common = ["the", "is", "a", "to", "of", "and", "in", "that", "it", 
                  "for", "you", "was", "with", "on", "are", "be", "have",
                  "this", "will", "your", "from", "or", "by", "not", "but",
                  "what", "all", "were", "we", "when", "can", "there", "an",
                  "which", "their", "if", "has", "more", "also", "do", "no",
                  "my", "one", "so", "our", "they", "been", "would", "how",
                  "her", "him", "his", "its", "may", "new", "now", "old",
                  "see", "way", "who", "did", "get", "just", "know", "take",
                  "come", "could", "good", "some", "them", "very", "after",
                  "most", "make", "should", "still", "over", "such", "much",
                  "then", "first", "any", "only", "other", "into", "year",
                  "hello", "hi", "yes", "no", "please", "thank", "thanks",
                  "sorry", "help", "?", "!", ".", ",", ":", ";", "'", '"',
                  "AI", "I", "You", "What", "How", "Why", "When", "Where"]
        
        for word in common:
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word
    
    def _load_vocab(self, vocab_file: Path):
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def save_vocab(self, vocab_file: Path):
        """Save vocabulary to file."""
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_file, 'w') as f:
            json.dump(self.token_to_id, f)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ids = [self.special_tokens["<s>"]]  # Start token
        
        # Simple tokenization: try words first, then characters
        words = text.split()
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                # Fall back to character-level
                for char in word:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        ids.append(self.special_tokens["<unk>"])
            # Add space token
            if " " in self.token_to_id:
                ids.append(self.token_to_id[" "])
        
        ids.append(self.special_tokens["</s>"])  # End token
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        # Join tokens, handling spaces properly
        result = ""
        prev_was_space = True  # Start true to avoid leading space
        for token in tokens:
            if token == " ":
                if not prev_was_space:
                    result += " "
                prev_was_space = True
            elif len(token) == 1:
                # Single character - just append
                result += token
                prev_was_space = False
            else:
                # Multi-char token (word) - add space before if needed
                if result and not prev_was_space:
                    result += " "
                result += token
                prev_was_space = False
        
        return result.strip()
    
    def __call__(self, text: str, return_tensors: str = None, 
                 padding: bool = None, truncation: bool = None, 
                 max_length: int = None) -> Dict[str, Any]:
        """Tokenize text (HuggingFace-compatible interface)."""
        ids = self.encode(text)
        
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids])}
        
        return {"input_ids": ids}


def build_tokenizer_from_files(data_files: List[str], vocab_size: int = 5000):
    """Train a tokenizer from data files (optional, uses HF if available)."""
    try:
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=data_files, vocab_size=vocab_size, min_frequency=2,
                        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"])
        VOCAB_DIR.mkdir(exist_ok=True)
        tokenizer.save_model(str(VOCAB_DIR))
        return str(VOCAB_DIR)
    except ImportError:
        logger.warning("tokenizers package not available, using simple tokenizer")
        tok = SimpleTokenizer()
        tok.save_vocab(VOCAB_DIR / "simple_vocab.json")
        return str(VOCAB_DIR)


def load_tokenizer(tokenizer_type: str = "auto"):
    """
    Load the best available tokenizer.
    
    Args:
        tokenizer_type: "auto", "bpe", "character", "simple", or "hf"
                       - "auto": Best available (bpe > character > simple)
                       - "bpe": Byte Pair Encoding (learns from data)
                       - "character": Full character-level with dictionary
                       - "simple": Basic character + common words
                       - "hf": HuggingFace tokenizer (if available)
    """
    # Try BPE tokenizer first (best option - learns from data)
    if tokenizer_type in ("auto", "bpe"):
        try:
            from .bpe_tokenizer import BPETokenizer
            bpe_vocab = VOCAB_DIR / "bpe_vocab.json"
            if bpe_vocab.exists():
                tok = BPETokenizer(vocab_file=bpe_vocab)
                return tok
            elif tokenizer_type == "bpe":
                # Return untrained BPE tokenizer (user should train it)
                logger.warning("BPE tokenizer not trained yet. Use train_bpe_tokenizer() first.")
                return BPETokenizer()
        except Exception as e:
            logger.warning(f"Could not load BPE tokenizer: {e}")
            if tokenizer_type == "bpe":
                raise
    
    # Try character tokenizer (good fallback)
    if tokenizer_type in ("auto", "character"):
        try:
            from .char_tokenizer import CharacterTokenizer
            char_vocab = VOCAB_DIR / "char_vocab.json"
            if char_vocab.exists():
                return CharacterTokenizer(vocab_file=char_vocab, use_dictionary=True)
            else:
                # Create new character tokenizer
                tok = CharacterTokenizer(use_dictionary=True)
                VOCAB_DIR.mkdir(parents=True, exist_ok=True)
                tok.save_vocab(char_vocab)
                return tok
        except Exception as e:
            logger.warning(f"Could not load character tokenizer: {e}")
            if tokenizer_type == "character":
                raise
    
    # Try simple tokenizer
    if tokenizer_type in ("auto", "simple"):
        simple_vocab = VOCAB_DIR / "simple_vocab.json"
        if simple_vocab.exists():
            return SimpleTokenizer(simple_vocab)
    
    # Check for HuggingFace trained vocab
    if tokenizer_type in ("auto", "hf"):
        hf_vocab = VOCAB_DIR / "vocab.json"
        if hf_vocab.exists():
            try:
                from transformers import PreTrainedTokenizerFast
                tok = PreTrainedTokenizerFast(
                    vocab_file=str(hf_vocab),
                    merges_file=str(VOCAB_DIR / "merges.txt"),
                    pad_token="<pad>",
                    eos_token="</s>"
                )
                tok.pad_token = tok.eos_token
                return tok
            except ImportError:
                pass
    
    # Default: create and return character tokenizer (best for learning)
    try:
        from .char_tokenizer import CharacterTokenizer
        return CharacterTokenizer(use_dictionary=True)
    except:
        return SimpleTokenizer()
