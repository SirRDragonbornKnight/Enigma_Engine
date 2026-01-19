"""
================================================================================
🔤 FORGE TOKENIZER - TEXT ↔ NUMBERS CONVERTER
================================================================================

The TRANSLATOR between human text and numbers the AI understands!
Converts sentences into sequences of integers for the neural network.

📍 FILE: forge_ai/core/tokenizer.py
🏷️ TYPE: Text Tokenization
🎯 MAIN FUNCTION: get_tokenizer()
🎯 MAIN CLASSES: SimpleTokenizer, TiktokenWrapper, AdvancedBPETokenizer

┌─────────────────────────────────────────────────────────────────────────────┐
│  TOKENIZATION FLOW:                                                         │
│                                                                             │
│  "Hello world!" → [Tokenizer] → [15496, 995, 0]                            │
│                                                                             │
│  [15496, 995, 0] → [Tokenizer] → "Hello world!"                            │
└─────────────────────────────────────────────────────────────────────────────┘

📊 TOKENIZER HIERARCHY (best to worst):
    1. AdvancedBPETokenizer - Byte-level BPE, handles any input, learns from data
    2. CharacterTokenizer  - Full character coverage with dictionary
    3. SimpleTokenizer     - Basic character + common words (NO dependencies!)

🏷️ SPECIAL TOKENS:
    • <pad> - Padding (ID: 0)
    • <s>   - Start of sequence (ID: 1)
    • </s>  - End of sequence (ID: 2)
    • <unk> - Unknown token (ID: 3)

🔗 CONNECTED FILES:
    → USES:      forge_ai/vocab_model/ (vocabulary files)
    ← USED BY:   forge_ai/core/model.py (needs vocab_size)
    ← USED BY:   forge_ai/core/inference.py (encode/decode text)
    ← USED BY:   forge_ai/core/training.py (prepare training data)

📖 USAGE:
    from forge_ai.core.tokenizer import get_tokenizer
    
    # Auto-select best available
    tokenizer = get_tokenizer()
    
    # Or specify type
    tokenizer = get_tokenizer("bpe")      # Advanced BPE
    tokenizer = get_tokenizer("char")     # Character-level
    tokenizer = get_tokenizer("simple")   # Simple fallback
    
    # Encode/Decode
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)

📁 VOCAB LOCATION: forge_ai/vocab_model/

📖 SEE ALSO:
    • forge_ai/core/bpe_tokenizer.py      - BPE implementation
    • forge_ai/core/char_tokenizer.py     - Character tokenizer
    • forge_ai/core/advanced_tokenizer.py - Advanced tokenizer
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)

# Vocabulary directory (shared across all tokenizers)
VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab_model"


# =============================================================================
# 🔤 SIMPLE TOKENIZER - Always Works, No Dependencies!
# =============================================================================
# This is the FALLBACK tokenizer that works without any external libraries.
# It's simple but reliable - perfect for bootstrapping or when tiktoken fails.

class SimpleTokenizer:
    """
    Lightweight character-level tokenizer.
    
    📖 WHAT THIS DOES:
    Converts text to numbers and back. Simple and reliable!
    
    📐 HOW IT WORKS:
    1. Has a vocabulary: {"a": 0, "b": 1, "the": 50, ...}
    2. encode(): Split text into tokens, look up their IDs
    3. decode(): Look up IDs to get tokens, join them together
    
    💡 TOKENIZATION STRATEGY:
    - First tries to match whole WORDS (like "the", "hello")
    - Falls back to individual CHARACTERS if word not in vocab
    - This is a hybrid word+char approach
    
    📐 EXAMPLE:
        >>> tok = SimpleTokenizer()
        >>> tok.encode("hello world")
        [1, 145, 32, 119, 111, 114, 108, 100, 2]
        #  ↑ <s>  "hello"  space + characters   ↑ </s>
    
    🔗 CONNECTS TO:
      ← Used by get_tokenizer() as fallback
      ← Used when no trained tokenizer is available
    """

    def __init__(self, vocab_file: Optional[Path] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_file: Optional path to saved vocabulary JSON
        """
        # ─────────────────────────────────────────────────────────────────────
        # SPECIAL TOKENS: Reserved tokens with fixed IDs
        # ─────────────────────────────────────────────────────────────────────
        self.pad_token = "<pad>"   # Padding for batching
        self.eos_token = "</s>"    # End of sequence
        self.unk_token = "<unk>"   # Unknown token (fallback)
        self.bos_token = "<s>"     # Beginning of sequence

        # Special token IDs (MUST be fixed for model compatibility)
        self.special_tokens = {
            "<pad>": 0,   # Padding - used to make sequences same length
            "<s>": 1,     # Start - marks beginning of text
            "</s>": 2,    # End - marks end of text
            "<unk>": 3,   # Unknown - used for characters not in vocab
        }

        # Convenient ID lookups
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        # ─────────────────────────────────────────────────────────────────────
        # LOAD OR CREATE VOCABULARY
        # ─────────────────────────────────────────────────────────────────────
        if vocab_file and Path(vocab_file).exists():
            self._load_vocab(vocab_file)
        else:
            self._create_default_vocab()

        self.vocab_size = len(self.token_to_id)

    def _create_default_vocab(self):
        """
        Create a basic character + common word vocabulary.
        
        📖 VOCABULARY STRUCTURE:
        IDs 0-3: Special tokens (<pad>, <s>, </s>, <unk>)
        IDs 4-98: Printable ASCII characters (space, a-z, A-Z, 0-9, etc.)
        IDs 99+: Common English words for efficiency
        """
        # Start with special tokens
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        # ─────────────────────────────────────────────────────────────────────
        # ADD ASCII CHARACTERS (codes 32-126)
        # ─────────────────────────────────────────────────────────────────────
        # This gives us: space, !"#$%&'()*+,-./ 0-9 :;<=>?@ A-Z [\]^_` a-z {|}~
        for c in range(32, 127):
            token = chr(c)
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        # ─────────────────────────────────────────────────────────────────────
        # ADD COMMON WORDS (for efficiency)
        # ─────────────────────────────────────────────────────────────────────
        # Without these, "the" would be 3 tokens: "t", "h", "e"
        # With these, "the" is 1 token - 3x more efficient!
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
        """Load vocabulary from JSON file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save_vocab(self, vocab_file: Path):
        """Save vocabulary to JSON file."""
        Path(vocab_file).parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        📖 ENCODING PROCESS:
        1. Add <s> (start token) if requested
        2. Split text into words
        3. For each word:
           - If word in vocab → use word token
           - Else → use character tokens
        4. Add spaces between words
        5. Add </s> (end token) if requested
        
        Args:
            text: Input string to encode
            add_special_tokens: Whether to add <s> and </s>
        
        Returns:
            List of token IDs
        """
        ids = []

        # Start with beginning-of-sequence token
        if add_special_tokens:
            ids.append(self.bos_token_id)

        # ─────────────────────────────────────────────────────────────────────
        # TOKENIZE: Try words first, fall back to characters
        # ─────────────────────────────────────────────────────────────────────
        words = text.split()
        for i, word in enumerate(words):
            if word in self.token_to_id:
                # Word is in vocabulary - use single token
                ids.append(self.token_to_id[word])
            else:
                # Word not in vocab - tokenize character by character
                for char in word:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        # Character not in vocab - use unknown token
                        ids.append(self.unk_token_id)

            # Add space token between words (not after last word)
            if i < len(words) - 1 and " " in self.token_to_id:
                ids.append(self.token_to_id[" "])

        # End with end-of-sequence token
        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        📖 DECODING PROCESS:
        1. Look up each ID in id_to_token dictionary
        2. Skip special tokens if requested
        3. Join tokens intelligently (handle spaces)
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip <s>, </s>, <pad>, <unk>
        
        Returns:
            Decoded text string
        """
        tokens = []

        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)

        # ─────────────────────────────────────────────────────────────────────
        # JOIN TOKENS INTELLIGENTLY
        # ─────────────────────────────────────────────────────────────────────
        # We need to add spaces between WORDS but not between CHARACTERS
        result = []
        for i, token in enumerate(tokens):
            if token == " ":
                result.append(" ")
            elif len(token) == 1:
                # Single character - don't add extra space
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
        """
        Tokenize text (HuggingFace-compatible interface).
        
        📖 WHAT THIS DOES:
        Same as encode(), but returns a dictionary and optionally
        handles padding, truncation, and tensor conversion.
        This makes the tokenizer work like HuggingFace tokenizers!
        
        Args:
            text: Input text to tokenize
            return_tensors: "pt" for PyTorch tensors, None for lists
            padding: Pad to max_length
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            add_special_tokens: Add <s> and </s>
        
        Returns:
            Dictionary with 'input_ids' key
        """
        ids = self.encode(text, add_special_tokens=add_special_tokens)

        # Truncate if too long
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        # Pad if too short
        if padding and max_length and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        # Convert to PyTorch tensor if requested
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids])}

        return {"input_ids": ids}

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Return copy of vocabulary dictionary."""
        return self.token_to_id.copy()


# =============================================================================
# Tiktoken Tokenizer (Fast GPT-style tokenizer)
# =============================================================================

class TiktokenWrapper:
    """Wrapper for tiktoken (GPT-style fast tokenizer)."""
    
    def __init__(self, encoding: str = "cl100k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self.enc.n_vocab
        
        # Special tokens for compatibility
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        ids = self.enc.encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            # Filter out special tokens
            ids = [i for i in ids if i not in [self.pad_token_id, self.bos_token_id, 
                                                 self.eos_token_id, self.unk_token_id]]
        return self.enc.decode(ids)
    
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
            - "auto": Best available (tiktoken > bpe > char > simple)
            - "tiktoken": Fast GPT-style tokenizer (requires tiktoken)
            - "bpe": Advanced BPE tokenizer
            - "advanced": Alias for "bpe"
            - "char": Character-level tokenizer
            - "simple": Simple character tokenizer
        vocab_path: Optional path to vocabulary file

    Returns:
        Tokenizer instance
    """
    vocab_path = Path(vocab_path) if vocab_path else VOCAB_DIR

    # Try tiktoken first (fastest)
    if tokenizer_type in ("auto", "tiktoken"):
        try:
            import tiktoken
            tok = TiktokenWrapper()
            logger.info("Loaded tiktoken tokenizer (cl100k_base)")
            return tok
        except ImportError:
            if tokenizer_type == "tiktoken":
                logger.error("tiktoken not available. Install with: pip install tiktoken")
                raise
            # If auto, continue to next best option
            logger.debug("tiktoken not available, trying BPE...")

    # Try Advanced BPE (best custom tokenizer)
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
    "TiktokenWrapper",
    "Tokenizer",

    # Constants
    "VOCAB_DIR",
]
