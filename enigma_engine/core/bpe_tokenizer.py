"""
Byte Pair Encoding (BPE) Tokenizer - Built from Scratch

This is a REAL tokenizer that learns patterns from your data.
No external dependencies. No faking it.

How BPE works:
1. Start with characters as initial vocabulary
2. Count all adjacent pairs in the data
3. Merge the most frequent pair into a new token
4. Repeat until desired vocab size

This creates subword tokens that capture common patterns in YOUR specific data.
"""
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BPETokenizer:
    """
    A real Byte Pair Encoding tokenizer.

    Learns subword patterns directly from your training data.
    No external dependencies - pure Python implementation.
    """

    def __init__(self, vocab_file: Optional[Path] = None):
        # Special tokens with reserved IDs
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<sep>": 4,
            "<mask>": 5,
            "<Q>": 6,
            "<A>": 7,
            "<USER>": 8,
            "<BOT>": 9,
        }

        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"

        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3

        # Vocabulary mappings
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # BPE merge rules (ordered list of merges)
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

        # Cache for encoding
        self.cache: dict[str, list[int]] = {}

        if vocab_file and vocab_file.exists():
            self.load(vocab_file)
        else:
            self._init_base_vocab()

        self.vocab_size = len(self.token_to_id)

    def _init_base_vocab(self):
        """Initialize with special tokens and base characters."""
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        next_id = len(self.special_tokens)

        # Add all printable ASCII as base tokens
        for i in range(256):
            char = bytes([i]).decode('latin-1')
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

    def train(self, texts: list[str], vocab_size: int = 8000, min_frequency: int = 2,
              verbose: bool = True):
        """
        Train BPE on a list of texts.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency to merge
            verbose: Print progress
        """
        if verbose:
            print(f"Training BPE tokenizer...")
            print(f"  Target vocab size: {vocab_size}")
            print(f"  Training texts: {len(texts)}")

        # Reset to base vocabulary
        self._init_base_vocab()
        self.merges = []
        self.merge_ranks = {}
        self.cache = {}

        # Pre-tokenize: split into words, keep track of frequencies
        word_freqs: Counter = Counter()

        for text in texts:
            # Split on whitespace and punctuation, keep the pieces
            words = self._pre_tokenize(text)
            for word in words:
                if word.strip():
                    # Convert word to tuple of characters (with word boundary marker)
                    word_tuple = tuple(word) + ('</w>',)  # End of word marker
                    word_freqs[word_tuple] += 1

        # Add end-of-word marker to vocab
        if '</w>' not in self.token_to_id:
            next_id = len(self.token_to_id)
            self.token_to_id['</w>'] = next_id
            self.id_to_token[next_id] = '</w>'

        if verbose:
            print(f"  Unique words: {len(word_freqs)}")
            print(f"  Base vocab: {len(self.token_to_id)}")

        # BPE training loop
        num_merges = vocab_size - len(self.token_to_id)

        for i in range(num_merges):
            # Count all adjacent pairs
            pair_freqs = self._count_pairs(word_freqs)

            if not pair_freqs:
                if verbose:
                    print(f"  No more pairs to merge at iteration {i}")
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            if best_freq < min_frequency:
                if verbose:
                    print(f"  Stopping: best pair frequency {best_freq} < {min_frequency}")
                break

            # Merge the pair
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            if new_token not in self.token_to_id:
                next_id = len(self.token_to_id)
                self.token_to_id[new_token] = next_id
                self.id_to_token[next_id] = new_token

            # Record the merge
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            # Apply merge to all words
            word_freqs = self._apply_merge(word_freqs, best_pair, new_token)

            if verbose and (i + 1) % 500 == 0:
                print(
                    f"  Merge {i + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' (freq: {best_freq})")

        self.vocab_size = len(self.token_to_id)

        if verbose:
            print(f"  Final vocab size: {self.vocab_size}")
            print(f"  Total merges: {len(self.merges)}")
            print("Training complete!")

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split text into words for BPE processing."""
        result = []

        # Handle special markers first - extract them before regex processing
        # Pattern to match Q:, A:, User:, Bot:, Human:, Assistant:
        special_pattern = r'(Q:|A:|User:|Bot:|Human:|Assistant:)'

        parts = re.split(special_pattern, text)

        for part in parts:
            if not part:
                continue

            # Map markers to special tokens
            if part == 'Q:':
                result.append('<Q>')
            elif part == 'A:':
                result.append('<A>')
            elif part in ('User:', 'Human:'):
                result.append('<USER>')
            elif part in ('Bot:', 'Assistant:'):
                result.append('<BOT>')
            else:
                # Regular text - split into words
                # This regex splits on spaces and separates punctuation
                pattern = r"'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\s\w]+"
                words = re.findall(pattern, part)

                for word in words:
                    word = word.strip()
                    if word and word not in ['', ' ']:
                        result.append(word)

        return result

    def _count_pairs(self, word_freqs: Counter) -> Counter:
        """Count frequency of all adjacent pairs."""
        pair_freqs: Counter = Counter()

        for word_tuple, freq in word_freqs.items():
            if len(word_tuple) < 2:
                continue
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _apply_merge(self, word_freqs: Counter, pair: tuple[str, str],
                     new_token: str) -> Counter:
        """Apply a merge to all words."""
        new_word_freqs: Counter = Counter()

        for word_tuple, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if (i < len(word_tuple) - 1 and
                    word_tuple[i] == pair[0] and
                        word_tuple[i + 1] == pair[1]):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq

        return new_word_freqs

    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word using learned BPE merges."""
        if word in self.cache:
            return list(self.cache[word])

        # Check if it's a special token
        if word in self.special_tokens:
            return [word]

        # Check if the whole word is in vocabulary (common word)
        word_with_end = word + '</w>'
        if word_with_end in self.token_to_id:
            return [word_with_end]

        # Start with characters
        tokens = list(word) + ['</w>']

        # Apply merges in order
        while len(tokens) > 1:
            # Find the best merge (lowest rank = learned earlier = more common)
            best_pair = None
            best_rank = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Apply the merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Cache result (with size limit to prevent memory growth)
        if len(self.cache) > 10000:
            # Clear oldest half of cache
            keys_to_remove = list(self.cache.keys())[:5000]
            for k in keys_to_remove:
                del self.cache[k]
        self.cache[word] = tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        # Pre-tokenize
        words = self._pre_tokenize(text)

        for word in words:
            if not word:
                continue

            # Handle special tokens directly
            if word in self.special_tokens:
                ids.append(self.special_tokens[word])
                continue

            # Tokenize word with BPE
            tokens = self._tokenize_word(word)

            for token in tokens:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    # Unknown - encode character by character
                    for char in token:
                        if char in self.token_to_id:
                            ids.append(self.token_to_id[char])
                        else:
                            ids.append(self.unk_token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []

        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]

                if skip_special_tokens and token in self.special_tokens:
                    # Convert special tokens back to readable form
                    if token == '<Q>':
                        tokens.append('Q: ')
                    elif token == '<A>':
                        tokens.append('A: ')
                    elif token == '<USER>':
                        tokens.append('User: ')
                    elif token == '<BOT>':
                        tokens.append('Bot: ')
                    # Skip other special tokens like <s>, </s>, <pad>
                    continue

                tokens.append(token)

        # Join and clean up
        text = ''.join(tokens)

        # Remove end-of-word markers and add spaces
        text = text.replace('</w>', ' ')

        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def save(self, path: Path):
        """Save tokenizer to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'token_to_id': self.token_to_id,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved tokenizer to {path}")

    def load(self, path: Path):
        """Load tokenizer from file."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(v) if isinstance(v, str) else v: k
                            for k, v in self.token_to_id.items()}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.merges = [tuple(m) for m in data['merges']]
        self.merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}

        if 'special_tokens' in data:
            self.special_tokens = data['special_tokens']

        self.vocab_size = len(self.token_to_id)
        self.cache = {}

        print(
            f"Loaded tokenizer from {path} (vocab: {self.vocab_size}, merges: {len(self.merges)})")

    def __call__(self, text: str, return_tensors: str = None,
                 padding: bool = None, truncation: bool = None,
                 max_length: int = None, add_special_tokens: bool = True) -> dict[str, Any]:
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

    def get_vocab(self) -> dict[str, int]:
        return self.token_to_id.copy()


def train_bpe_tokenizer(data_paths: list[str], vocab_size: int = 8000,
                        output_path: Optional[str] = None) -> BPETokenizer:
    """
    Train a BPE tokenizer on data files.

    Args:
        data_paths: List of text file paths
        vocab_size: Target vocabulary size
        output_path: Where to save the tokenizer

    Returns:
        Trained BPETokenizer
    """
    # Load all texts
    texts = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            with open(p, encoding='utf-8') as f:
                texts.append(f.read())
            print(f"Loaded {p} ({len(texts[-1])} chars)")

    if not texts:
        raise ValueError("No training data found!")

    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, verbose=True)

    # Save if path provided
    if output_path:
        tokenizer.save(Path(output_path))

    return tokenizer
