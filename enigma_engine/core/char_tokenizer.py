"""
Character-Level Tokenizer with Full Dictionary Support

This is a proper character-level tokenizer that:
  - Uses EVERY character as a token (not just common ones)
  - Supports a full English dictionary for word-level optimization
  - Can be extended with any vocabulary
  - Has no external dependencies
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Built-in English dictionary (most common 10,000 words)
# This is loaded lazily to avoid memory usage if not needed
DICTIONARY_PATH = Path(__file__).parent.parent / "data" / "dictionary.txt"


class CharacterTokenizer:
    """
    A proper character-level tokenizer.

    Features:
      - Every Unicode character gets a unique ID
      - Optional word-level tokens for efficiency
      - Dictionary support for common words
      - No external dependencies
      - Fast and memory-efficient
    """

    def __init__(self, vocab_file: Optional[Path] = None, use_dictionary: bool = True):
        # Special tokens - includes Q&A format tokens for training
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,      # Start of sequence
            "</s>": 2,     # End of sequence
            "<unk>": 3,    # Unknown token
            "<sep>": 4,    # Separator
            "<cls>": 5,    # Classification token
            "<mask>": 6,   # Mask for MLM
            "<nl>": 7,     # Newline
            "<tab>": 8,    # Tab
            "<Q>": 9,      # Question marker
            "<A>": 10,     # Answer marker
            "<USER>": 11,  # User turn
            "<BOT>": 12,   # Bot turn
        }

        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"

        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3

        # Initialize vocabulary
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        if vocab_file and vocab_file.exists():
            self._load_vocab(vocab_file)
        else:
            self._build_vocabulary(use_dictionary)

        self.vocab_size = len(self.token_to_id)

    def _build_vocabulary(self, use_dictionary: bool = True):
        """Build vocabulary from scratch."""
        # Start with special tokens
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        next_id = len(self.special_tokens)

        # Add ALL printable ASCII characters (32-126)
        for code in range(32, 127):
            char = chr(code)
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # Add extended ASCII (128-255) for accented characters
        for code in range(128, 256):
            char = chr(code)
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # Add common Unicode ranges
        unicode_ranges = [
            (0x00A0, 0x00FF),  # Latin Extended
            (0x0100, 0x017F),  # Latin Extended-A
            (0x2000, 0x206F),  # General Punctuation
            (0x20A0, 0x20CF),  # Currency Symbols
            (0x2100, 0x214F),  # Letterlike Symbols
            (0x2190, 0x21FF),  # Arrows
            (0x2200, 0x22FF),  # Mathematical Operators
            (0x2300, 0x23FF),  # Miscellaneous Technical
            (0x2500, 0x257F),  # Box Drawing
            (0x2600, 0x26FF),  # Miscellaneous Symbols
            (0x2700, 0x27BF),  # Dingbats
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        ]

        for start, end in unicode_ranges:
            for code in range(start, min(end + 1, start + 200)):  # Limit each range
                try:
                    char = chr(code)
                    if char not in self.token_to_id:
                        self.token_to_id[char] = next_id
                        self.id_to_token[next_id] = char
                        next_id += 1
                except BaseException:
                    pass

        # Add dictionary words if enabled
        if use_dictionary:
            self._add_dictionary_words(next_id)

    def _add_dictionary_words(self, start_id: int):
        """Add common English words to vocabulary."""
        next_id = start_id

        # Built-in common words (always available)
        common_words = [
            # Articles and prepositions
            "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "over",

            # Pronouns
            "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
            "them", "my", "your", "his", "its", "our", "their", "this", "that",
            "these", "those", "who", "what", "which", "whom", "whose",

            # Common verbs
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "must", "can", "need", "want", "like", "know", "think", "see", "come",
            "go", "get", "make", "take", "give", "find", "tell", "ask", "use", "try",
            "say", "help", "show", "call", "keep", "let", "begin", "seem", "leave",
            "put", "mean", "become", "work", "read", "write", "learn", "feel",

            # Common adjectives
            "good", "new", "first", "last", "long", "great", "little", "own", "other",
            "old", "right", "big", "high", "different", "small", "large", "next",
            "early", "young", "important", "few", "public", "bad", "same", "able",

            # Common adverbs
            "not", "just", "also", "very", "often", "however", "too", "usually",
            "really", "early", "never", "always", "sometimes", "together", "likely",
            "simply", "generally", "instead", "actually", "already", "enough",

            # Common nouns
            "time", "year", "people", "way", "day", "man", "thing", "woman", "life",
            "child", "world", "school", "state", "family", "student", "group",
            "country", "problem", "hand", "part", "place", "case", "week", "company",
            "system", "program", "question", "work", "government", "number", "night",
            "point", "home", "water", "room", "mother", "area", "money", "story",

            # Question words
            "how", "why", "when", "where", "what", "which", "who",

            # Conjunctions
            "and", "but", "or", "if", "because", "while", "although", "though",
            "unless", "until", "since", "whether", "so", "yet", "nor", "both",

            # Numbers
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "hundred", "thousand", "million", "billion",

            # AI/Tech related
            "AI", "artificial", "intelligence", "machine", "learning", "neural",
            "network", "model", "data", "algorithm", "computer", "software",
            "hardware", "code", "program", "system", "digital", "technology",
            "robot", "automation", "processing", "memory", "storage", "input",
            "output", "training", "inference", "parameter", "layer", "token",

            # Common phrases (as single tokens for efficiency)
            "hello", "hi", "hey", "thanks", "thank", "please", "sorry", "yes", "no",
            "okay", "sure", "well", "now", "here", "there", "maybe", "perhaps",
        ]

        for word in common_words:
            if word not in self.token_to_id:
                self.token_to_id[word] = next_id
                self.id_to_token[next_id] = word
                next_id += 1

        # Try to load external dictionary file
        if DICTIONARY_PATH.exists():
            try:
                with open(DICTIONARY_PATH, encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word and word not in self.token_to_id and len(word) > 2:
                            self.token_to_id[word] = next_id
                            self.id_to_token[next_id] = word
                            next_id += 1
                            # Limit dictionary size
                            if next_id > 50000:
                                break
                logger.info(f"Loaded dictionary with {next_id - start_id} words")
            except Exception as e:
                logger.warning(f"Could not load dictionary: {e}")

    def add_word(self, word: str) -> int:
        """Add a new word to vocabulary. Returns its ID."""
        if word in self.token_to_id:
            return self.token_to_id[word]

        new_id = len(self.token_to_id)
        self.token_to_id[word] = new_id
        self.id_to_token[new_id] = word
        self.vocab_size = len(self.token_to_id)
        return new_id

    def add_words(self, words: list[str]) -> list[int]:
        """Add multiple words. Returns their IDs."""
        return [self.add_word(word) for word in words]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs.

        Strategy:
        1. Try to match known words first (greedy)
        2. Fall back to character-level for unknown words
        """
        ids = []

        if add_special_tokens:
            ids.append(self.special_tokens["<s>"])

        # Handle special characters and Q&A markers
        text = text.replace('\n', ' <nl> ').replace('\t', ' <tab> ')
        text = text.replace('Q:', ' <Q> ').replace('A:', ' <A> ')
        text = text.replace('User:', ' <USER> ').replace('Bot:', ' <BOT> ')
        text = text.replace('Human:', ' <USER> ').replace('Assistant:', ' <BOT> ')

        # Tokenize
        i = 0
        while i < len(text):
            # Try to find the longest matching token
            best_match = None
            best_len = 0

            # Look for word boundaries
            if text[i].isalnum() or text[i] in "'":
                # Extract potential word
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] in "'"):
                    j += 1
                word = text[i:j]

                # Check if whole word is in vocabulary
                if word in self.token_to_id:
                    ids.append(self.token_to_id[word])
                    i = j
                    continue

                # Check lowercase version
                if word.lower() in self.token_to_id:
                    ids.append(self.token_to_id[word.lower()])
                    i = j
                    continue

            # Single character fallback
            char = text[i]
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            elif char == '\n':
                ids.append(self.special_tokens["<nl>"])
            elif char == '\t':
                ids.append(self.special_tokens["<tab>"])
            else:
                # Unknown character - try to add it dynamically
                new_id = self.add_word(char)
                ids.append(new_id)

            i += 1

        if add_special_tokens:
            ids.append(self.special_tokens["</s>"])

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []

        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]

                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue

                # Handle special markers
                if token == "<nl>":
                    tokens.append("\n")
                elif token == "<tab>":
                    tokens.append("\t")
                elif token == "<Q>":
                    tokens.append("Q:")
                elif token == "<A>":
                    tokens.append("A:")
                elif token == "<USER>":
                    tokens.append("User:")
                elif token == "<BOT>":
                    tokens.append("Bot:")
                else:
                    tokens.append(token)

        # Smart joining - add spaces between words but not characters
        result = []
        for i, token in enumerate(tokens):
            if len(token) == 1 and not token.isalnum():
                # Punctuation - no space before
                result.append(token)
            elif i > 0 and len(tokens[i - 1]) == 1 and tokens[i - 1].isalnum():
                # Previous was single alphanumeric - might be part of word
                result.append(token)
            elif i > 0 and result and not result[-1].endswith(' '):
                # Add space before words
                if len(token) > 1:
                    result.append(' ')
                result.append(token)
            else:
                result.append(token)

        return ''.join(result).strip()

    def _load_vocab(self, vocab_file: Path):
        """Load vocabulary from file."""
        with open(vocab_file, encoding='utf-8') as f:
            data = json.load(f)

        self.token_to_id = data.get('token_to_id', data)
        # Convert string keys to int for id_to_token
        if 'id_to_token' in data:
            self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        else:
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save_vocab(self, vocab_file: Path):
        """Save vocabulary to file."""
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': {str(k): v for k, v in self.id_to_token.items()}
            }, f, ensure_ascii=False, indent=2)

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
        """Return the vocabulary."""
        return self.token_to_id.copy()


def load_char_tokenizer(
        vocab_file: Optional[Path] = None,
        use_dictionary: bool = True) -> CharacterTokenizer:
    """Load or create a character tokenizer."""
    return CharacterTokenizer(vocab_file=vocab_file, use_dictionary=use_dictionary)
