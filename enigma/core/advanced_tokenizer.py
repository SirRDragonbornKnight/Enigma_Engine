"""
Advanced Byte Pair Encoding (BPE) Tokenizer
============================================

A production-grade tokenizer built from scratch. No external dependencies.

Features:
  - Byte-level BPE (handles ANY text, any language, any characters)
  - Learns optimal subword vocabulary from your data
  - Special token handling (Q:, A:, etc.)
  - Efficient caching for fast encoding
  - Regex-based pre-tokenization (GPT-style)
  - Serialization and loading
  - Compatible with trainer interface

This is how GPT-2, GPT-3, and GPT-4 tokenize text.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter, defaultdict
import logging
import unicodedata

logger = logging.getLogger(__name__)


def bytes_to_unicode() -> Dict[int, str]:
    """
    Create a mapping from bytes to unicode characters.

    This is the key innovation of GPT-2's tokenizer - it maps all 256 possible
    byte values to printable unicode characters. This means the tokenizer can
    handle ANY input (any language, any encoding, binary data, etc.).

    Returns a dict mapping byte values (0-255) to unicode characters.
    """
    # Start with printable ASCII (these stay as themselves)
    bs = list(range(ord("!"), ord("~") + 1))  # 33-126
    bs += list(range(ord("¡"), ord("¬") + 1))  # 161-172
    bs += list(range(ord("®"), ord("ÿ") + 1))  # 174-255

    cs = bs[:]
    n = 0

    # Map non-printable bytes to unicode characters starting at 256
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {b: chr(c) for b, c in zip(bs, cs)}


def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Get all adjacent pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class AdvancedBPETokenizer:
    """
    Advanced Byte Pair Encoding Tokenizer.

    This is a production-grade implementation that:
    - Handles any input (byte-level)
    - Uses efficient algorithms
    - Supports special tokens
    - Caches encoding for speed
    - Can be trained on any text data
    """

    # GPT-style regex pattern for pre-tokenization
    # Splits text into words, contractions, numbers, and punctuation
    # Enhanced pattern for better code, numbers, and special character handling
    PAT = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)|"""  # Contractions
        r"""\s?\d+(?:\.\d+)?|"""  # Numbers (including decimals)
        r"""\s?[a-zA-Z]+|"""  # Words
        r"""\s?[^\s\w]+|"""  # Punctuation
        r"""\s+(?!\S)|"""  # Trailing whitespace
        r"""\s+""",  # Other whitespace
        re.IGNORECASE
    )

    def __init__(self, vocab_file: Optional[Path] = None):
        """
        Initialize tokenizer.

        Args:
            vocab_file: Path to saved vocabulary (optional)
        """
        # Byte encoder/decoder for handling any input
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Special tokens with reserved IDs (0-19 reserved)
        self.special_tokens = {
            # Core tokens
            "<|pad|>": 0,
            "<|start|>": 1,
            "<|end|>": 2,
            "<|unk|>": 3,
            "<|sep|>": 4,
            "<|mask|>": 5,
            
            # Legacy conversation tokens
            "<|Q|>": 6,
            "<|A|>": 7,
            
            # Modern conversation tokens
            "<|user|>": 8,
            "<|bot|>": 9,
            "<|assistant|>": 10,
            "<|system|>": 11,
            
            # Tool invocation tokens
            "<|tool_call|>": 12,
            "<|tool_result|>": 13,
            "<|tool_end|>": 14,
            "<|tool_result_end|>": 15,
            
            # Modality tokens
            "<|image|>": 16,
            "<|audio|>": 17,
            "<|video|>": 18,
            "<|vision|>": 19,
            
            # Action/capability tokens
            "<|generate_image|>": 20,
            "<|avatar_action|>": 21,
            "<|speak|>": 22,
            "<|listen|>": 23,
            "<|search_web|>": 24,
            "<|read_file|>": 25,
            "<|write_file|>": 26,
            "<|capture_screen|>": 27,
            "<|run_code|>": 28,
            
            # Formatting tokens
            "<|newline|>": 29,
            "<|tab|>": 30,
            "<|code|>": 31,
            "<|code_end|>": 32,
            
            # Meta tokens
            "<|thinking|>": 33,
            "<|thinking_end|>": 34,
            "<|error|>": 35,
            "<|warning|>": 36,
            "<|success|>": 37,
            "<|info|>": 38,
        }
        self.special_token_ids = {v: k for k, v in self.special_tokens.items()}

        # Standard token names for compatibility
        self.pad_token = "<|pad|>"
        self.eos_token = "<|end|>"
        self.bos_token = "<|start|>"
        self.unk_token = "<|unk|>"

        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3

        # Vocabulary
        self.encoder: Dict[str, int] = {}  # token -> id
        self.decoder: Dict[int, str] = {}  # id -> token

        # BPE merge rules
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        # Encoding cache
        self.cache: Dict[str, str] = {}

        # Load or initialize
        if vocab_file and vocab_file.exists():
            self.load(vocab_file)
        else:
            self._init_base_vocab()

        self.vocab_size = len(self.encoder)
        
        # Compile pattern - try advanced regex, fall back to standard

        # Compile pattern
        try:
            import regex
            self.pat = regex.compile(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            )
        except ImportError:
            # Use the standard re pattern defined as class variable
            self.pat = self.PAT
    
            self.pat = self.PAT_FALLBACK

    def _init_base_vocab(self):
        """Initialize vocabulary with special tokens and byte-level tokens."""
        # Start with special tokens
        self.encoder = dict(self.special_tokens)

        # Add all byte-level tokens (256 base tokens)
        next_id = len(self.special_tokens)
        for byte_char in self.byte_encoder.values():
            if byte_char not in self.encoder:
                self.encoder[byte_char] = next_id
                next_id += 1

        self.decoder = {v: k for k, v in self.encoder.items()}

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
        verbose: bool = True
    ):
        """
        Train BPE on a corpus of texts.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency to merge
            verbose: Print progress
        """
        if verbose:
            print("=" * 60)
            print("TRAINING ADVANCED BPE TOKENIZER")
            print("=" * 60)
            print(f"  Target vocab size: {vocab_size:,}")
            print(f"  Training texts: {len(texts):,}")
            print(f"  Min frequency: {min_frequency}")

        # Reset vocabulary
        self._init_base_vocab()
        self.bpe_ranks = {}
        self.cache = {}

        # Count word frequencies
        word_freqs: Counter = Counter()
        total_chars = 0

        for text in texts:
            total_chars += len(text)
            # Pre-tokenize and convert to byte representation
            for token in self._pre_tokenize(text):
                if token.strip():
                    # Convert to byte-level representation
                    byte_word = tuple(self.byte_encoder[b] for b in token.encode('utf-8'))
                    word_freqs[byte_word] += 1

        if verbose:
            print(f"  Total characters: {total_chars:,}")
            print(f"  Unique words: {len(word_freqs):,}")
            print(f"  Base vocab: {len(self.encoder)}")
            print()
            print("Training BPE merges...")

        # Calculate number of merges needed
        num_merges = vocab_size - len(self.encoder)

        # BPE training loop
        for i in range(num_merges):
            # Count pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                if len(word) < 2:
                    continue
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j + 1])] += freq

            if not pairs:
                if verbose:
                    print(f"  No more pairs at merge {i}")
                break

            # Find best pair
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            if best_freq < min_frequency:
                if verbose:
                    print(f"  Stopping: freq {best_freq} < {min_frequency}")
                break

            # Create new token
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            if new_token not in self.encoder:
                new_id = len(self.encoder)
                self.encoder[new_token] = new_id
                self.decoder[new_id] = new_token

            # Record merge rank
            self.bpe_ranks[best_pair] = len(self.bpe_ranks)

            # Apply merge to all words
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, new_token)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

            # Progress
            if verbose and (i + 1) % 500 == 0:
                print(f"  Merge {i + 1:,}/{num_merges:,}: "
                      f"'{best_pair[0]}'+'{best_pair[1]}' -> '{new_token}' "
                      f"(freq: {best_freq:,})")

        self.vocab_size = len(self.encoder)

        if verbose:
            print()
            print("=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
            print(f"  Final vocab size: {self.vocab_size:,}")
            print(f"  Total merges: {len(self.bpe_ranks):,}")

    def _apply_merge(
        self,
        word: Tuple[str, ...],
        pair: Tuple[str, str],
        new_token: str
    ) -> Tuple[str, ...]:
        """Apply a merge to a word."""
        new_word = []
        i = 0
        while i < len(word):
            if (i < len(word) - 1 and
                word[i] == pair[0] and
                    word[i + 1] == pair[1]):
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Split text into words/tokens before BPE.

        Handles special markers like Q:, A:, etc.
        """
        # Handle special markers first
        result = []

        # Pattern to find special markers
        special_pattern = r'(Q:|A:|User:|Bot:|Human:|Assistant:|System:)'
        parts = re.split(special_pattern, text, flags=re.IGNORECASE)

        for part in parts:
            if not part:
                continue

            # Check for special markers
            lower = part.lower()
            if lower in ('q:', 'a:', 'user:', 'bot:', 'human:', 'assistant:', 'system:'):
                # Map to special token marker (we'll handle this in encode)
                if lower == 'q:':
                    result.append('<|Q|>')
                elif lower == 'a:':
                    result.append('<|A|>')
                elif lower in ('user:', 'human:'):
                    result.append('<|user|>')
                elif lower in ('bot:', 'assistant:'):
                    result.append('<|bot|>')
                elif lower == 'system:':
                    result.append('<|system|>')
            else:
                # Regular text - use pattern
                tokens = self.pat.findall(part)
                result.extend(tokens)

        return result

    def _bpe(self, token: str) -> str:
        """
        Apply BPE to a single token.

        Uses caching for efficiency.
        """
        # Use cache only when dropout is disabled
        if dropout == 0.0 and token in self.cache:
            return self.cache[token]

        # Convert to byte representation
        word = tuple(self.byte_encoder[b] for b in token.encode('utf-8'))

        if len(word) == 1:
            return word[0]

        # Apply merges
        while True:
            # Find the merge with lowest rank (learned earliest = most common)
            pairs = get_pairs(word)
            if not pairs:
                break

            # Get pair with lowest rank
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float('inf'))
                
                # BPE dropout: randomly skip merges during training
                if dropout > 0.0 and random.random() < dropout:
                    continue
                    
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_pair not in self.bpe_ranks:
                break

            # Apply merge
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and
                    word[i] == min_pair[0] and
                        word[i + 1] == min_pair[1]):
                    new_word.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)

            if len(word) == 1:
                break

        result = ' '.join(word)
        
        # Cache only when dropout is disabled
        if dropout == 0.0:
            self.cache[token] = result
        
        return result

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add start/end tokens

        Returns:
            List of token IDs
        """
        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        # Pre-tokenize
        tokens = self._pre_tokenize(text)

        for token in tokens:
            # Check for special tokens
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
                continue

            # Apply BPE
            bpe_tokens = self._bpe(token).split(' ')

            for bpe_token in bpe_tokens:
                if bpe_token in self.encoder:
                    ids.append(self.encoder[bpe_token])
                else:
                    # Unknown token - use UNK
                    ids.append(self.unk_token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokens = []

        for idx in ids:
            if idx in self.decoder:
                token = self.decoder[idx]

                # Handle special tokens
                if token in self.special_tokens:
                    if not skip_special_tokens:
                        tokens.append(token)
                    else:
                        # Convert some special tokens to readable form
                        if token == '<|Q|>':
                            tokens.append('Q: ')
                        elif token == '<|A|>':
                            tokens.append('A: ')
                        elif token == '<|user|>':
                            tokens.append('User: ')
                        elif token == '<|bot|>' or token == '<|assistant|>':
                            tokens.append('Bot: ')
                        elif token == '<|system|>':
                            tokens.append('System: ')
                        elif token == '<|newline|>':
                            tokens.append('\n')
                        elif token == '<|tab|>':
                            tokens.append('\t')
                        # Skip other special tokens when skip_special_tokens=True
                    continue

                tokens.append(token)

        # Join and decode bytes
        text = ''.join(tokens)

        # Convert from byte representation back to text
        try:
            byte_array = bytearray([self.byte_decoder[c] for c in text])
            text = byte_array.decode('utf-8', errors='replace')
        except BaseException:
            pass  # Keep as-is if decoding fails

        return text

    def save(self, path: Path):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'version': '2.1',
            'vocab_size': self.vocab_size,
            'encoder': self.encoder,
            'special_tokens': self.special_tokens,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved tokenizer to {path}")
        print(f"  Vocab size: {self.vocab_size:,}")
        print(f"  Merges: {len(self.bpe_ranks):,}")

    def load(self, path: Path):
        """
        Load tokenizer from file.
        
        Automatically loads merges from separate .merges file if it exists.
        """
        path = Path(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.encoder = data['encoder']
        self.decoder = {int(v) if isinstance(v, str) else v: k
                        for k, v in self.encoder.items()}
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Parse bpe_ranks
        self.bpe_ranks = {}
        for k, v in data.get('bpe_ranks', {}).items():
            parts = k.split('|||')
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = v

        if 'special_tokens' in data:
            self.special_tokens = data['special_tokens']
            self.special_token_ids = {v: k for k, v in self.special_tokens.items()}

        self.vocab_size = len(self.encoder)
        self.cache = {}

        print(f"Loaded tokenizer from {path}")
        print(f"  Vocab size: {self.vocab_size:,}")
        print(f"  Merges: {len(self.bpe_ranks):,}")

    def __call__(
        self,
        text: str,
        return_tensors: str = None,
        padding: bool = None,
        truncation: bool = None,
        max_length: int = None,
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
        return self.encoder.copy()

    @property
    def token_to_id(self) -> Dict[str, int]:
        """Compatibility property."""
        return self.encoder

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Compatibility property."""
        return self.decoder
    
    def set_bpe_dropout(self, dropout: float):
        """
        Set BPE dropout rate for subword regularization.
        
        Args:
            dropout: Dropout probability (0.0 = disabled, 0.1 = 10% dropout)
        """
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"Dropout must be between 0.0 and 1.0, got {dropout}")
        self.bpe_dropout = dropout
    
    def encode_stream(self, text_chunk: str, finalize: bool = False) -> List[int]:
        """
        Encode streaming text efficiently.
        
        Buffers incomplete tokens and only encodes complete words/tokens.
        Call with finalize=True on the last chunk to flush the buffer.
        
        Args:
            text_chunk: New text to encode
            finalize: Whether this is the last chunk (flush buffer)
            
        Returns:
            Token IDs for complete tokens in this chunk
        """
        self._stream_buffer += text_chunk
        
        if not finalize:
            # Only encode complete tokens (wait for whitespace/punctuation)
            # Find last complete token boundary
            last_space = self._stream_buffer.rfind(' ')
            last_newline = self._stream_buffer.rfind('\n')
            last_boundary = max(last_space, last_newline)
            
            if last_boundary == -1:
                # No complete tokens yet
                return []
            
            # Encode up to boundary
            to_encode = self._stream_buffer[:last_boundary + 1]
            self._stream_buffer = self._stream_buffer[last_boundary + 1:]
        else:
            # Final chunk - encode everything
            to_encode = self._stream_buffer
            self._stream_buffer = ""
        
        if not to_encode:
            return []
        
        return self.encode(to_encode, add_special_tokens=False)
    
    def reset_stream(self):
        """Reset streaming buffer."""
        self._stream_buffer = ""
    
    def decode_improved(
        self, 
        ids: List[int], 
        skip_special_tokens: bool = True,
        clean_up_spaces: bool = True
    ) -> str:
        """
        Improved decoding with better space and punctuation handling.
        
        Args:
            ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_spaces: Apply intelligent space cleanup
            
        Returns:
            Decoded text with improved formatting
        """
        # First do standard decode
        text = self.decode(ids, skip_special_tokens=skip_special_tokens)
        
        if not clean_up_spaces:
            return text
        
        # Cleanup common spacing issues
        import re
        
        # Remove spaces before punctuation
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        
        # Remove spaces after opening brackets/quotes
        text = re.sub(r'([\[\(\{"\']) ', r'\1', text)
        
        # Remove spaces before closing brackets/quotes
        text = re.sub(r' ([\]\)\}"\'])', r'\1', text)
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix newline spacing
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        
        # Clean up leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio (chars per token).
        Higher is better - means more efficient encoding.
        
        Args:
            text: Text to analyze
            
        Returns:
            Average characters per token
        """
        if not text:
            return 0.0
        
        ids = self.encode(text, add_special_tokens=False)
        if not ids:
            return 0.0
        
        return len(text) / len(ids)
    
    def tokenize_stats(self, text: str) -> Dict[str, Any]:
        """
        Get detailed tokenization statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with tokenization stats
        """
        ids = self.encode(text, add_special_tokens=False)
        tokens = [self.decoder.get(id, '<unk>') for id in ids]
        
        return {
            'text_length': len(text),
            'token_count': len(ids),
            'unique_tokens': len(set(ids)),
            'compression_ratio': len(text) / len(ids) if ids else 0.0,
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0,
            'tokens': tokens[:50],  # First 50 tokens
        }


def train_tokenizer(
    data_paths: List[str],
    vocab_size: int = 8000,
    output_path: Optional[str] = None
) -> AdvancedBPETokenizer:
    """
    Train a tokenizer on data files.

    Args:
        data_paths: List of text file paths
        vocab_size: Target vocabulary size
        output_path: Where to save the tokenizer

    Returns:
        Trained tokenizer
    """
    # Load texts
    texts = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                texts.append(f.read())

    if not texts:
        raise ValueError("No training data found!")

    # Train
    tokenizer = AdvancedBPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, verbose=True)

    # Save
    if output_path:
        tokenizer.save(Path(output_path))

    return tokenizer
