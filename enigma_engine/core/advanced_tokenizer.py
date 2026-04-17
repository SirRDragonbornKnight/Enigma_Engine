"""
Advanced BPE Tokenizer

Wrapper around BPETokenizer with additional features.
This provides the AdvancedBPETokenizer class expected by inference.py
"""
import json
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class AdvancedBPETokenizer:
    """
    Advanced BPE Tokenizer with additional features.

    Handles multiple tokenizer formats:
    - Standard BPE format (token_to_id, id_to_token, merges)
    - Enigma format (encoder, special_tokens)
    """

    def __init__(self, vocab_file: Optional[Path] = None, **kwargs):
        """
        Initialize the Advanced BPE Tokenizer.

        Args:
            vocab_file: Path to vocabulary JSON file
            **kwargs: Additional arguments
        """
        # Special tokens with reserved IDs
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<think>": 4,
            "</think>": 5,
        }

        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"

        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        self.think_start_id = self.special_tokens["<think>"]
        self.think_end_id = self.special_tokens["</think>"]

        # Vocabulary mappings
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # BPE merge rules
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

        # Cache for encoding (LRU via OrderedDict)
        self._cache_lock = threading.Lock()
        self.cache: OrderedDict[str, list[str]] = OrderedDict()

        if vocab_file and Path(vocab_file).exists():
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

    def load(self, vocab_file: Union[str, Path]) -> None:
        """Load vocabulary from file."""
        vocab_file = Path(vocab_file)

        with open(vocab_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupt vocabulary file {vocab_file}: {exc}"
                ) from exc

        # Handle different formats
        if 'encoder' in data:
            # Enigma format
            self.token_to_id = data['encoder']
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

            # Update special tokens from file
            if 'special_tokens' in data:
                for token, idx in data['special_tokens'].items():
                    self.special_tokens[token] = idx

                # Map standard names to Enigma format
                if '[E:pad]' in data['special_tokens']:
                    self.pad_token_id = data['special_tokens']['[E:pad]']
                    self.pad_token = '[E:pad]'
                if '[E:start]' in data['special_tokens']:
                    self.bos_token_id = data['special_tokens']['[E:start]']
                    self.bos_token = '[E:start]'
                if '[E:end]' in data['special_tokens']:
                    self.eos_token_id = data['special_tokens']['[E:end]']
                    self.eos_token = '[E:end]'
                if '[E:unk]' in data['special_tokens']:
                    self.unk_token_id = data['special_tokens']['[E:unk]']
                    self.unk_token = '[E:unk]'

            self.vocab_size = data.get('vocab_size', len(self.token_to_id))

        elif 'token_to_id' in data:
            # Standard BPE format
            self.token_to_id = data['token_to_id']
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

            if 'merges' in data:
                self.merges = [tuple(m) for m in data['merges']]
                self.merge_ranks = {m: i for i, m in enumerate(self.merges)}

            self.vocab_size = len(self.token_to_id)
        else:
            raise ValueError(f"Unknown tokenizer format in {vocab_file}")

        logger.info(f"Loaded tokenizer from {vocab_file} with {self.vocab_size} tokens")

    def save(self, vocab_file: Union[str, Path]) -> None:
        """Save vocabulary to file."""
        vocab_file = Path(vocab_file)

        data = {
            'version': '2.1',
            'vocab_size': self.vocab_size,
            'encoder': self.token_to_id,
            'special_tokens': self.special_tokens,
        }

        from enigma_engine.core.safe_save import atomic_write_json
        atomic_write_json(vocab_file, data)

    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word using learned BPE merges."""
        # Check cache (LRU: move to end on hit)
        with self._cache_lock:
            if word in self.cache:
                self.cache.move_to_end(word)
                return list(self.cache[word])

        # Check if whole word is in vocabulary
        if word in self.token_to_id:
            return [word]

        # Start with characters
        tokens = list(word)

        # Apply merges in priority order (lowest rank = most common)
        while len(tokens) > 1:
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

        # Cache result (LRU eviction)
        with self._cache_lock:
            if len(self.cache) > 10000:
                for _ in range(5000):
                    self.cache.popitem(last=False)
            self.cache[word] = tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs using BPE merges."""
        if not text:
            return []

        ids = []

        if add_special_tokens:
            if self.bos_token_id in self.id_to_token or \
                    self.bos_token in self.token_to_id:
                ids.append(self.bos_token_id)

        # If we have merges, use BPE encoding
        if self.merges:
            # Build regex with special tokens as alternatives (longest first)
            # so multi-char tokens like <think> match as whole tokens
            special_alts = "|".join(
                re.escape(t)
                for t in sorted(self.special_tokens, key=len, reverse=True)
                if len(t) > 1
            )
            base_pattern = r"'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\s\w]+|\s+"
            pattern = (special_alts + "|" + base_pattern
                       if special_alts else base_pattern)
            words = re.findall(pattern, text)

            for word in words:
                if not word:
                    continue
                # Check special tokens
                if word in self.special_tokens:
                    ids.append(self.special_tokens[word])
                    continue
                # Check if whole word is in vocab
                if word in self.token_to_id:
                    ids.append(self.token_to_id[word])
                    continue
                # Apply BPE merges
                subtokens = self._tokenize_word(word)
                for st in subtokens:
                    if st in self.token_to_id:
                        ids.append(self.token_to_id[st])
                    else:
                        # Fall back to character-level for unknown subtokens
                        for char in st:
                            if char in self.token_to_id:
                                ids.append(self.token_to_id[char])
                            else:
                                ids.append(self.unk_token_id)
        else:
            # No merges loaded — character-level fallback
            for char in text:
                if char in self.token_to_id:
                    ids.append(self.token_to_id[char])
                else:
                    ids.append(self.unk_token_id)

        if add_special_tokens:
            if self.eos_token_id in self.id_to_token or \
                    self.eos_token in self.token_to_id:
                ids.append(self.eos_token_id)

        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        chars = []

        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()

        for tid in token_ids:
            if tid in special_ids:
                continue
            if tid in self.id_to_token:
                chars.append(self.id_to_token[tid])
            else:
                chars.append(self.unk_token)

        return ''.join(chars)

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True
    ) -> list[int]:
        """Encode a list of chat messages."""
        tokens = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if '[E:system]' in self.token_to_id:
                    tokens.append(self.token_to_id['[E:system]'])
                tokens.extend(self.encode(content))
            elif role == "user":
                if '[E:user]' in self.token_to_id:
                    tokens.append(self.token_to_id['[E:user]'])
                tokens.extend(self.encode(content))
            elif role == "assistant":
                if '[E:assistant]' in self.token_to_id:
                    tokens.append(self.token_to_id['[E:assistant]'])
                tokens.extend(self.encode(content))

        if add_generation_prompt:
            if '[E:assistant]' in self.token_to_id:
                tokens.append(self.token_to_id['[E:assistant]'])

        return tokens

    def get_vocab_stats(self) -> dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": list(self.special_tokens.keys()),
            "cache_size": len(self.cache),
        }

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AdvancedBPETokenizer":
        """Load tokenizer from file."""
        return cls(vocab_file=Path(path))


__all__ = ['AdvancedBPETokenizer']
