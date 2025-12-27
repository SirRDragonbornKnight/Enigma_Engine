"""
Character-Level Tokenizer with Dictionary Support

A full-featured tokenizer that:
1. Maps every character to a unique ID (character-level)
2. Includes a dictionary of common English words
3. Supports special tokens (PAD, UNK, BOS, EOS)
4. Can be extended with custom vocabulary

This is NOT a lightweight tokenizer - it's a proper implementation
suitable for training real models.
"""
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)

# Directory for vocabulary files
VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab_model"


class CharacterTokenizer:
    """
    Character-level tokenizer with dictionary support.
    
    Every character gets its own token ID, plus common words
    can be added to the vocabulary for efficiency.
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"      # Beginning of sequence
    EOS_TOKEN = "</s>"     # End of sequence
    MASK_TOKEN = "<mask>"
    
    # Special token IDs (reserved)
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    MASK_ID = 4
    
    def __init__(self, vocab_path: Optional[Path] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_path: Optional path to saved vocabulary JSON
        """
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        # For compatibility with transformers
        self.pad_token = self.PAD_TOKEN
        self.eos_token = self.EOS_TOKEN
        self.bos_token = self.BOS_TOKEN
        self.unk_token = self.UNK_TOKEN
        
        self.pad_token_id = self.PAD_ID
        self.eos_token_id = self.EOS_ID
        self.bos_token_id = self.BOS_ID
        self.unk_token_id = self.UNK_ID
        
        self._next_id = 5  # Start after special tokens
        
        if vocab_path and Path(vocab_path).exists():
            self.load_vocab(vocab_path)
        else:
            self._build_default_vocab()
    
    def _build_default_vocab(self):
        """Build default vocabulary with all printable ASCII + extended chars."""
        # Add special tokens first
        self.char_to_id[self.PAD_TOKEN] = self.PAD_ID
        self.char_to_id[self.UNK_TOKEN] = self.UNK_ID
        self.char_to_id[self.BOS_TOKEN] = self.BOS_ID
        self.char_to_id[self.EOS_TOKEN] = self.EOS_ID
        self.char_to_id[self.MASK_TOKEN] = self.MASK_ID
        
        for tok, idx in self.char_to_id.items():
            self.id_to_char[idx] = tok
        
        # Add all printable ASCII characters (32-126)
        for i in range(32, 127):
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = self._next_id
                self.id_to_char[self._next_id] = char
                self._next_id += 1
        
        # Add common special characters and extended ASCII
        special_chars = [
            '\n', '\t', '\r',  # Whitespace
            '©', '®', '™',     # Symbols
            '€', '£', '¥',     # Currency
            '°', '±', '×', '÷',  # Math
            'à', 'á', 'â', 'ã', 'ä', 'å',  # Accented
            'è', 'é', 'ê', 'ë',
            'ì', 'í', 'î', 'ï',
            'ò', 'ó', 'ô', 'õ', 'ö',
            'ù', 'ú', 'û', 'ü',
            'ñ', 'ç', 'ß',
            '…', '—', '–', '"', '"', ''', ''',  # Typography
        ]
        
        for char in special_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = self._next_id
                self.id_to_char[self._next_id] = char
                self._next_id += 1
        
        # Load dictionary words
        self._load_dictionary()
        
        logger.info(f"Built vocabulary with {len(self.char_to_id)} characters and {len(self.word_to_id)} words")
    
    def _load_dictionary(self):
        """Load common English words into vocabulary."""
        # Check for custom dictionary file
        dict_path = VOCAB_DIR / "dictionary.txt"
        
        if dict_path.exists():
            try:
                words = dict_path.read_text().strip().split('\n')
                for word in words:
                    word = word.strip().lower()
                    if word and word not in self.word_to_id:
                        self.word_to_id[word] = self._next_id
                        self.id_to_word[self._next_id] = word
                        self._next_id += 1
                logger.info(f"Loaded {len(words)} words from dictionary.txt")
                return
            except Exception as e:
                logger.warning(f"Could not load dictionary.txt: {e}")
        
        # Default: Add most common English words
        common_words = self._get_common_words()
        
        for word in common_words:
            word = word.lower()
            if word not in self.word_to_id:
                self.word_to_id[word] = self._next_id
                self.id_to_word[self._next_id] = word
                self._next_id += 1
    
    def _get_common_words(self) -> List[str]:
        """Return list of common English words (top ~3000)."""
        return [
            # Articles and determiners
            "the", "a", "an", "this", "that", "these", "those", "my", "your", 
            "his", "her", "its", "our", "their", "some", "any", "no", "every",
            "each", "all", "both", "few", "many", "much", "most", "other",
            # Pronouns
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "myself", "yourself", "himself", "herself", "itself",
            "ourselves", "themselves", "who", "whom", "whose", "which", "what",
            "whoever", "whatever", "whichever",
            # Prepositions
            "in", "on", "at", "to", "for", "with", "by", "from", "about",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "over", "out", "up", "down", "off", "away",
            "around", "among", "within", "without", "along", "across", "behind",
            "beside", "beyond", "near", "toward", "upon",
            # Conjunctions
            "and", "but", "or", "nor", "so", "yet", "because", "although",
            "while", "if", "unless", "until", "when", "where", "whether",
            "though", "since", "as", "than", "once",
            # Common verbs
            "be", "am", "is", "are", "was", "were", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "done",
            "will", "would", "shall", "should", "may", "might", "must", "can",
            "could", "need", "dare", "ought", "used",
            "say", "said", "says", "saying", "get", "got", "gets", "getting",
            "make", "made", "makes", "making", "go", "goes", "went", "going", "gone",
            "take", "took", "takes", "taking", "taken", "come", "came", "comes", "coming",
            "see", "saw", "sees", "seeing", "seen", "know", "knew", "knows", "knowing", "known",
            "think", "thought", "thinks", "thinking", "want", "wanted", "wants", "wanting",
            "give", "gave", "gives", "giving", "given", "use", "uses", "using",
            "find", "found", "finds", "finding", "tell", "told", "tells", "telling",
            "ask", "asked", "asks", "asking", "work", "worked", "works", "working",
            "seem", "seemed", "seems", "seeming", "feel", "felt", "feels", "feeling",
            "try", "tried", "tries", "trying", "leave", "left", "leaves", "leaving",
            "call", "called", "calls", "calling", "keep", "kept", "keeps", "keeping",
            "let", "lets", "letting", "begin", "began", "begins", "beginning", "begun",
            "show", "showed", "shows", "showing", "shown", "hear", "heard", "hears", "hearing",
            "play", "played", "plays", "playing", "run", "ran", "runs", "running",
            "move", "moved", "moves", "moving", "live", "lived", "lives", "living",
            "believe", "believed", "believes", "believing", "hold", "held", "holds", "holding",
            "bring", "brought", "brings", "bringing", "happen", "happened", "happens", "happening",
            "write", "wrote", "writes", "writing", "written", "provide", "provided", "provides",
            "sit", "sat", "sits", "sitting", "stand", "stood", "stands", "standing",
            "lose", "lost", "loses", "losing", "pay", "paid", "pays", "paying",
            "meet", "met", "meets", "meeting", "include", "included", "includes", "including",
            "continue", "continued", "continues", "continuing", "set", "sets", "setting",
            "learn", "learned", "learns", "learning", "change", "changed", "changes", "changing",
            "lead", "led", "leads", "leading", "understand", "understood", "understands",
            "watch", "watched", "watches", "watching", "follow", "followed", "follows",
            "stop", "stopped", "stops", "stopping", "create", "created", "creates", "creating",
            "speak", "spoke", "speaks", "speaking", "spoken", "read", "reads", "reading",
            "allow", "allowed", "allows", "allowing", "add", "added", "adds", "adding",
            "spend", "spent", "spends", "spending", "grow", "grew", "grows", "growing", "grown",
            "open", "opened", "opens", "opening", "walk", "walked", "walks", "walking",
            "win", "won", "wins", "winning", "offer", "offered", "offers", "offering",
            "remember", "remembered", "remembers", "remembering", "love", "loved", "loves", "loving",
            "consider", "considered", "considers", "considering", "appear", "appeared", "appears",
            "buy", "bought", "buys", "buying", "wait", "waited", "waits", "waiting",
            "serve", "served", "serves", "serving", "die", "died", "dies", "dying",
            "send", "sent", "sends", "sending", "expect", "expected", "expects", "expecting",
            "build", "built", "builds", "building", "stay", "stayed", "stays", "staying",
            "fall", "fell", "falls", "falling", "fallen", "cut", "cuts", "cutting",
            "reach", "reached", "reaches", "reaching", "kill", "killed", "kills", "killing",
            "remain", "remained", "remains", "remaining", "suggest", "suggested", "suggests",
            "raise", "raised", "raises", "raising", "pass", "passed", "passes", "passing",
            "sell", "sold", "sells", "selling", "require", "required", "requires", "requiring",
            "report", "reported", "reports", "reporting", "decide", "decided", "decides",
            "pull", "pulled", "pulls", "pulling", "develop", "developed", "develops",
            # Common nouns
            "time", "year", "people", "way", "day", "man", "woman", "child", "children",
            "world", "life", "hand", "part", "place", "case", "week", "company", "system",
            "program", "question", "government", "number", "night", "point", "home",
            "water", "room", "mother", "area", "money", "story", "fact", "month", "lot",
            "right", "study", "book", "eye", "job", "word", "business", "issue", "side",
            "kind", "head", "house", "service", "friend", "father", "power", "hour", "game",
            "line", "end", "member", "law", "car", "city", "community", "name", "president",
            "team", "minute", "idea", "kid", "body", "information", "back", "parent", "face",
            "others", "level", "office", "door", "health", "person", "art", "war", "history",
            "party", "result", "morning", "reason", "research", "girl", "guy",
            "moment", "air", "teacher", "force", "education",
            # Common adjectives
            "good", "new", "first", "last", "long", "great", "little", "own",
            "old", "big", "high", "different", "small", "large", "next", "early",
            "young", "important", "public", "bad", "same", "able", "human", "local",
            "sure", "free", "better", "true", "whole", "real", "best", "political", "social",
            "national", "special", "possible", "particular", "major", "economic", "personal",
            "hard", "black", "white", "strong", "certain", "international",
            "full", "close", "common", "current", "likely", "natural", "happy", "serious",
            "ready", "simple", "physical", "general", "environmental", "financial",
            "past", "wrong", "poor", "nice", "late", "easy", "final", "main", "low",
            "available", "clear", "short", "single", "similar", "recent", "concerned",
            "dead", "central", "necessary", "federal", "private", "present",
            "military", "legal", "religious", "cold", "dark", "medical", "individual",
            "foreign", "beautiful", "various", "difficult", "entire", "united", "popular",
            "traditional", "hot", "cultural", "successful", "southern", "modern", "western",
            # Common adverbs
            "not", "also", "very", "often", "however", "too", "usually", "really",
            "never", "always", "sometimes", "together", "simply", "generally",
            "instead", "actually", "already", "ever", "rather", "almost", "especially",
            "eventually", "probably", "certainly", "clearly", "recently", "quickly",
            "directly", "finally", "exactly", "easily", "carefully", "completely",
            "immediately", "naturally", "suddenly", "particularly", "seriously",
            "merely", "normally", "specifically", "highly", "strongly",
            "obviously", "absolutely", "necessarily", "possibly", "perhaps", "maybe",
            # Numbers
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "hundred", "thousand", "million", "billion", "second", "third",
            "half", "quarter", "percent", "today", "tomorrow", "yesterday", "now", "then",
            "here", "there", "why", "how",
            # AI/tech terms
            "ai", "artificial", "intelligence", "machine", "learning", "neural", "network",
            "model", "data", "training", "algorithm", "computer", "software", "hardware",
            "code", "input", "output", "process", "function",
            "variable", "parameter", "memory", "storage", "database", "server", "client",
            "api", "interface", "user", "digital", "virtual", "online", "internet", "web",
            "file", "folder", "directory", "path", "error", "debug", "test", "deploy",
            # Conversation
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank", "please", "sorry",
            "yes", "no", "okay", "ok", "just", "like",
            "anyway", "therefore", "thus", "hence", "meanwhile", "otherwise",
            # Question words
            "something", "nothing", "everything", "anything", "someone", "anyone",
            "everyone", "nobody", "somebody", "anybody", "everybody", "somewhere", "anywhere",
            "everywhere", "nowhere", "sometime", "anytime",
        ]
    
    def add_word(self, word: str) -> int:
        """Add a word to the vocabulary."""
        word = word.lower()
        if word not in self.word_to_id:
            self.word_to_id[word] = self._next_id
            self.id_to_word[self._next_id] = word
            self._next_id += 1
        return self.word_to_id[word]
    
    def add_character(self, char: str) -> int:
        """Add a character to the vocabulary."""
        if char not in self.char_to_id:
            self.char_to_id[char] = self._next_id
            self.id_to_char[self._next_id] = char
            self._next_id += 1
        return self.char_to_id[char]
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (characters + words + special tokens)."""
        return self._next_id
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        Uses word tokens where possible, falls back to character tokens.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.BOS_ID)
        
        i = 0
        while i < len(text):
            matched_word = False
            if i == 0 or not text[i-1].isalnum():
                for end in range(min(i + 20, len(text)), i, -1):
                    word = text[i:end].lower()
                    if word in self.word_to_id:
                        if end == len(text) or not text[end].isalnum():
                            ids.append(self.word_to_id[word])
                            i = end
                            matched_word = True
                            break
            
            if not matched_word:
                char = text[i]
                if char in self.char_to_id:
                    ids.append(self.char_to_id[char])
                else:
                    self.add_character(char)
                    ids.append(self.char_to_id[char])
                i += 1
        
        if add_special_tokens:
            ids.append(self.EOS_ID)
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID, self.MASK_ID}
        result = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.id_to_char:
                result.append(self.id_to_char[token_id])
            elif token_id in self.id_to_word:
                result.append(self.id_to_word[token_id])
            else:
                result.append(self.unk_token)
        return ''.join(result)
    
    def __call__(
        self, 
        text: Union[str, List[str]], 
        return_tensors: str = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int = None,
        **kwargs
    ) -> Dict:
        """Tokenize text (compatible with HuggingFace interface)."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        all_ids = []
        for t in texts:
            ids = self.encode(t)
            if truncation and max_length:
                ids = ids[:max_length]
            all_ids.append(ids)
        
        if padding and len(all_ids) > 1:
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)
            attention_mask = []
            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids)
                if len(ids) < max_len:
                    pad_len = max_len - len(ids)
                    ids.extend([self.PAD_ID] * pad_len)
                    mask.extend([0] * pad_len)
                all_ids[i] = ids
                attention_mask.append(mask)
        else:
            attention_mask = [[1] * len(ids) for ids in all_ids]
        
        result = {"input_ids": all_ids[0] if isinstance(text, str) else all_ids}
        
        if return_tensors == "pt":
            try:
                import torch
                result["input_ids"] = torch.tensor(result["input_ids"])
                result["attention_mask"] = torch.tensor(attention_mask[0] if isinstance(text, str) else attention_mask)
            except ImportError:
                pass
        return result
    
    def save_vocab(self, path: Optional[Path] = None):
        """Save vocabulary to JSON file."""
        if path is None:
            path = VOCAB_DIR / "char_vocab.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "char_to_id": self.char_to_id,
            "word_to_id": self.word_to_id,
            "next_id": self._next_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved vocabulary to {path}")
    
    def load_vocab(self, path: Path):
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_id = data.get("char_to_id", {})
        self.word_to_id = data.get("word_to_id", {})
        self._next_id = data.get("next_id", 5)
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.id_to_word = {int(v): k for k, v in self.word_to_id.items()}
        logger.info(f"Loaded vocabulary from {path}: {len(self.char_to_id)} chars, {len(self.word_to_id)} words")
    
    def build_from_text(self, text: str, min_word_freq: int = 2):
        """Build vocabulary from a text corpus."""
        from collections import Counter
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if count >= min_word_freq and word not in self.word_to_id:
                self.add_word(word)
        for char in set(text):
            if char not in self.char_to_id:
                self.add_character(char)
        logger.info(f"Built vocabulary from text: {len(self.word_to_id)} words, {len(self.char_to_id)} chars")


# === Compatibility wrapper ===
_tokenizer_instance: Optional[CharacterTokenizer] = None


def load_tokenizer() -> CharacterTokenizer:
    """Load or create the default tokenizer."""
    global _tokenizer_instance
    if _tokenizer_instance is not None:
        return _tokenizer_instance
    
    vocab_path = VOCAB_DIR / "char_vocab.json"
    if vocab_path.exists():
        _tokenizer_instance = CharacterTokenizer(vocab_path)
    else:
        _tokenizer_instance = CharacterTokenizer()
        VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        _tokenizer_instance.save_vocab(vocab_path)
    return _tokenizer_instance


def build_tokenizer_from_files(data_files: List[str], vocab_size: int = 5000) -> str:
    """Build tokenizer vocabulary from training files."""
    tokenizer = CharacterTokenizer()
    all_text = ""
    for filepath in data_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
        except Exception as e:
            logger.warning(f"Could not read {filepath}: {e}")
    
    if all_text:
        from collections import Counter
        import re
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_counts = Counter(words)
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        if len(sorted_words) > vocab_size:
            min_freq = sorted_words[vocab_size - 1][1]
        else:
            min_freq = 1
        tokenizer.build_from_text(all_text, min_word_freq=max(1, min_freq))
    
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab()
    return str(VOCAB_DIR)
