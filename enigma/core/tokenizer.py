"""
Enigma Tokenizer - Unified Interface
=====================================

Provides a unified interface for all tokenizer types with automatic selection
of the best available tokenizer for the task.

Tokenizer Hierarchy (best to worst):
1. AdvancedBPETokenizer - Byte-level BPE, handles any input, learns from data
2. CharacterTokenizer - Full character coverage with dictionary
3. SimpleTokenizer - Basic character + common words (no dependencies)

Usage:
    from enigma.core.tokenizer import get_tokenizer

    # Auto-select best available
    tokenizer = get_tokenizer()

    # Or specify type
    tokenizer = get_tokenizer("bpe")      # Advanced BPE
    tokenizer = get_tokenizer("char")     # Character-level
    tokenizer = get_tokenizer("simple")   # Simple fallback

    # Use it
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)

# Vocabulary directory (shared across all tokenizers)
VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab_model"


# =============================================================================
# Simple Tokenizer (always available, no dependencies)
# =============================================================================

class SimpleTokenizer:
    """
    Lightweight character-level tokenizer.

    Works without any external dependencies.
    Good for bootstrapping and fallback.
    """

    def __init__(self, vocab_file: Optional[Path] = None):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"

        # Special token IDs (fixed)
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
        }

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        # Load or create vocabulary
        if vocab_file and Path(vocab_file).exists():
            self._load_vocab(vocab_file)
        else:
            self._create_default_vocab()

        self.vocab_size = len(self.token_to_id)

    def _create_default_vocab(self):
        """Create a basic character + common word vocabulary."""
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        # Add printable ASCII characters
        for c in range(32, 127):
            token = chr(c)
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        # Add common words/subwords for better efficiency
        common = [
            "the", "is", "a", "to", "of", "and", "in", "that", "it",
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
            "hello", "hi", "yes", "please", "thank", "thanks",
            "sorry", "help", "AI", "I", "You", "What", "How", "Why",
            "When", "Where", "Q:", "A:", "User:", "Bot:",
        ]

        for word in common:
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word

    def _load_vocab(self, vocab_file: Path):
        """Load vocabulary from file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save_vocab(self, vocab_file: Path):
        """Save vocabulary to file."""
        Path(vocab_file).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        # Tokenize: try words first, then characters
        words = text.split()
        for i, word in enumerate(words):
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                # Fall back to character-level
                for char in word:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        ids.append(self.unk_token_id)

            # Add space token between words
            if i < len(words) - 1 and " " in self.token_to_id:
                ids.append(self.token_to_id[" "])

        if add_special_tokens:
            ids.append(self.eos_token_id)

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

        # Join tokens intelligently
        result = []
        for i, token in enumerate(tokens):
            if token == " ":
                result.append(" ")
            elif len(token) == 1:
                result.append(token)
            else:
                # Word token - add space before if needed
                if result and result[-1] != " ":
                    result.append(" ")
                result.append(token)

        return "".join(result).strip()

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text (HuggingFace-compatible interface)."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)

        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        if padding and max_length and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids])}

        return {"input_ids": ids}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return self.token_to_id.copy()


# =============================================================================
# Tokenizer Loading Functions
# =============================================================================

def get_tokenizer(
    tokenizer_type: str = "auto",
    vocab_path: Optional[Union[str, Path]] = None
) -> Any:
    """
    Get the best available tokenizer.

    Args:
        tokenizer_type: Type of tokenizer to load
            - "auto": Best available (bpe > char > simple)
            - "bpe": Advanced BPE tokenizer
            - "advanced": Alias for "bpe"
            - "char": Character-level tokenizer
            - "simple": Simple character tokenizer
        vocab_path: Optional path to vocabulary file

    Returns:
        Tokenizer instance
    """
    vocab_path = Path(vocab_path) if vocab_path else VOCAB_DIR

    # Try Advanced BPE first (best)
    if tokenizer_type in ("auto", "bpe", "advanced"):
        try:
            from .advanced_tokenizer import AdvancedBPETokenizer

            # Check for saved vocabulary
            bpe_vocab = vocab_path / "bpe_vocab.json" if vocab_path.is_dir() else vocab_path

            if bpe_vocab.exists():
                tok = AdvancedBPETokenizer(vocab_file=bpe_vocab)
                logger.info(f"Loaded BPE tokenizer from {bpe_vocab}")
                return tok
            elif tokenizer_type in ("bpe", "advanced"):
                # Return untrained BPE tokenizer
                logger.warning("BPE tokenizer not trained. Use train_tokenizer() first.")
                return AdvancedBPETokenizer()
        except Exception as e:
            logger.warning(f"Could not load BPE tokenizer: {e}")
            if tokenizer_type in ("bpe", "advanced"):
                raise

    # Try Character tokenizer
    if tokenizer_type in ("auto", "char", "character"):
        try:
            from .char_tokenizer import CharacterTokenizer

            char_vocab = vocab_path / "char_vocab.json" if vocab_path.is_dir() else vocab_path

            if char_vocab.exists():
                tok = CharacterTokenizer(vocab_file=char_vocab, use_dictionary=True)
                logger.info(f"Loaded character tokenizer from {char_vocab}")
                return tok
            else:
                # Create new character tokenizer
                tok = CharacterTokenizer(use_dictionary=True)
                VOCAB_DIR.mkdir(parents=True, exist_ok=True)
                tok.save_vocab(VOCAB_DIR / "char_vocab.json")
                logger.info("Created new character tokenizer")
                return tok
        except Exception as e:
            logger.warning(f"Could not load character tokenizer: {e}")
            if tokenizer_type in ("char", "character"):
                raise

    # Fall back to Simple tokenizer
    simple_vocab = vocab_path / "simple_vocab.json" if vocab_path.is_dir() else vocab_path

    if simple_vocab.exists():
        tok = SimpleTokenizer(simple_vocab)
        logger.info(f"Loaded simple tokenizer from {simple_vocab}")
        return tok

    # Create new simple tokenizer
    tok = SimpleTokenizer()
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_vocab(VOCAB_DIR / "simple_vocab.json")
    logger.info("Created new simple tokenizer")
    return tok


def load_tokenizer(tokenizer_type: str = "auto") -> Any:
    """
    Load the best available tokenizer.

    Alias for get_tokenizer() for backwards compatibility.
    """
    return get_tokenizer(tokenizer_type)


def train_tokenizer(
    data_paths: List[str],
    vocab_size: int = 8000,
    output_path: Optional[str] = None,
    tokenizer_type: str = "bpe"
) -> Any:
    """
    Train a tokenizer on text data.

    Args:
        data_paths: List of paths to training text files
        vocab_size: Target vocabulary size
        output_path: Where to save the trained tokenizer
        tokenizer_type: Type of tokenizer to train ("bpe" or "char")

    Returns:
        Trained tokenizer
    """
    # Load training data
    texts = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            logger.info(f"Loaded training data from {path}")

    if not texts:
        raise ValueError("No training data found!")

    logger.info(f"Training {tokenizer_type} tokenizer on {len(texts)} files...")

    if tokenizer_type in ("bpe", "advanced"):
        from .advanced_tokenizer import AdvancedBPETokenizer

        tokenizer = AdvancedBPETokenizer()
        tokenizer.train(texts, vocab_size=vocab_size, verbose=True)

        # Save
        save_path = Path(output_path) if output_path else VOCAB_DIR / "bpe_vocab.json"
        tokenizer.save(save_path)

        return tokenizer

    elif tokenizer_type in ("char", "character"):
        from .char_tokenizer import CharacterTokenizer

        # Character tokenizer builds vocab from data
        full_text = "\n".join(texts)
        tokenizer = CharacterTokenizer(use_dictionary=True)

        # Add any new characters from training data
        for char in set(full_text):
            if char not in tokenizer.token_to_id:
                idx = len(tokenizer.token_to_id)
                tokenizer.token_to_id[char] = idx
                tokenizer.id_to_token[idx] = char

        tokenizer.vocab_size = len(tokenizer.token_to_id)

        # Save
        save_path = Path(output_path) if output_path else VOCAB_DIR / "char_vocab.json"
        tokenizer.save_vocab(save_path)

        return tokenizer

    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


# =============================================================================
# Backwards Compatibility
# =============================================================================

# Expose SimpleTokenizer as default (always available)
Tokenizer = SimpleTokenizer


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Main functions
    "get_tokenizer",
    "load_tokenizer",
    "train_tokenizer",

    # Classes
    "SimpleTokenizer",
    "Tokenizer",

    # Constants
    "VOCAB_DIR",
]
