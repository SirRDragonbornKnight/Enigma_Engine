"""
Data Trainer - Extract character data and generate task training datasets.

This lightweight tool provides two main capabilities:
1. **CharacterTrainer**: Extract dialogue, personality traits, speech patterns,
   and vocabulary from training data to create character-specific datasets.
2. **TaskTrainer**: Generate training data for AI tasks (image prompts, avatar
   control, tool usage, code generation) by loading examples from JSON files.

Example Usage:
    from enigma_engine.tools.data_trainer import CharacterTrainer, TaskTrainer
    
    # Character extraction
    trainer = CharacterTrainer()
    characters = trainer.scan_for_characters("data/training/")
    sherlock = trainer.extract_character("Sherlock Holmes", "data/training/")
    trainer.generate_training_dataset("Sherlock Holmes", "data/training/", "outputs/sherlock.txt")
    
    # Task training (load examples from JSON files)
    tasks = TaskTrainer(examples_dir="data/training/examples/")
    tasks.add_example("image", "Create a cat", "Fluffy orange tabby...")
    tasks.generate_all_tasks("outputs/task_training.txt")
    tasks.export_examples("data/training/my_examples.json")
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CharacterProfile:
    """Profile of an extracted character."""
    name: str
    aliases: List[str] = field(default_factory=list)
    dialogue_count: int = 0
    vocabulary: Dict[str, int] = field(default_factory=dict)
    phrases: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)
    speech_patterns: Dict[str, float] = field(default_factory=dict)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    sample_dialogues: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "aliases": self.aliases,
            "dialogue_count": self.dialogue_count,
            "vocabulary": dict(self.vocabulary),
            "phrases": self.phrases[:50],  # Limit for serialization
            "catchphrases": self.catchphrases,
            "speech_patterns": self.speech_patterns,
            "personality_traits": self.personality_traits,
            "topics": self.topics,
            "sample_dialogues": self.sample_dialogues[:20],
            "source_files": self.source_files
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterProfile":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Unknown"),
            aliases=data.get("aliases", []),
            dialogue_count=data.get("dialogue_count", 0),
            vocabulary=data.get("vocabulary", {}),
            phrases=data.get("phrases", []),
            catchphrases=data.get("catchphrases", []),
            speech_patterns=data.get("speech_patterns", {}),
            personality_traits=data.get("personality_traits", {}),
            topics=data.get("topics", []),
            sample_dialogues=data.get("sample_dialogues", []),
            source_files=data.get("source_files", [])
        )


@dataclass
class ExtractionResult:
    """Result of character extraction."""
    success: bool
    character: Optional[CharacterProfile] = None
    dialogue_lines: List[str] = field(default_factory=list)
    qa_pairs: List[Tuple[str, str]] = field(default_factory=list)
    error: Optional[str] = None


class CharacterTrainer:
    """
    Extract character dialogue and traits from training data.
    
    Supports multiple dialogue formats:
    - "CHARACTER: dialogue text"
    - "CHARACTER\nDialogue text"
    - JSON format with speaker field
    - Screenplay format
    """
    
    # Common dialogue patterns
    DIALOGUE_PATTERNS = [
        re.compile(r'^([A-Z][A-Za-z\s]+):\s*(.+)$'),  # CHARACTER: dialogue
        re.compile(r'^([A-Z][A-Z\s]+)\s*\n(.+)$', re.MULTILINE),  # UPPERCASE NAME\n dialogue
        re.compile(r'^"(.+)"\s*said\s+([A-Za-z\s]+)', re.IGNORECASE),  # "dialogue" said Character
        re.compile(r'^([A-Za-z\s]+)\s+said[,:]?\s*"(.+)"', re.IGNORECASE),  # Character said: "dialogue"
    ]
    
    # Personality trait keywords
    TRAIT_KEYWORDS = {
        "analytical": ["deduce", "analyze", "logic", "evidence", "observe", "conclude", "data", "facts"],
        "humorous": ["joke", "laugh", "funny", "wit", "humor", "chuckle", "amusing", "haha"],
        "formal": ["indeed", "quite", "rather", "certainly", "precisely", "therefore", "accordingly"],
        "casual": ["yeah", "cool", "awesome", "like", "kinda", "gonna", "wanna", "hey"],
        "empathetic": ["feel", "understand", "sorry", "care", "concern", "worry", "support"],
        "confident": ["certain", "sure", "obviously", "clearly", "definitely", "absolutely"],
        "mysterious": ["secret", "hidden", "unknown", "strange", "curious", "puzzle", "mystery"],
        "aggressive": ["fight", "attack", "destroy", "battle", "war", "enemy", "conquer"],
        "kind": ["help", "please", "thank", "kind", "gentle", "generous", "love"],
        "sarcastic": ["really", "obviously", "brilliant", "genius", "sure", "right"],
    }
    
    def __init__(self):
        """Initialize character trainer."""
        self.characters: Dict[str, CharacterProfile] = {}
        self._stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> Set[str]:
        """Get common stopwords to filter from vocabulary."""
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "although", "i", "me", "my", "myself",
            "we", "our", "ours", "you", "your", "he", "him", "his", "she", "her",
            "it", "its", "they", "them", "their", "this", "that", "these", "those"
        }
    
    def scan_for_characters(
        self, 
        data_path: str,
        min_dialogue_count: int = 5
    ) -> Dict[str, int]:
        """
        Scan training data for characters.
        
        Args:
            data_path: Path to training data directory or file
            min_dialogue_count: Minimum dialogues to include character
            
        Returns:
            Dictionary of character names to dialogue counts
        """
        path = Path(data_path)
        character_counts: Counter = Counter()
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.txt")) + list(path.rglob("*.json")) + list(path.rglob("*.jsonl"))
        
        logger.info(f"Scanning {len(files)} files for characters...")
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                
                if file_path.suffix in [".json", ".jsonl"]:
                    characters = self._extract_characters_from_json(content)
                else:
                    characters = self._extract_characters_from_text(content)
                
                character_counts.update(characters)
            except Exception as e:
                logger.warning(f"Error scanning {file_path}: {e}")
        
        # Filter by minimum count
        filtered = {
            name: count 
            for name, count in character_counts.items() 
            if count >= min_dialogue_count
        }
        
        logger.info(f"Found {len(filtered)} characters with {min_dialogue_count}+ dialogues")
        return dict(filtered)
    
    def _extract_characters_from_text(self, content: str) -> Counter:
        """Extract character names from plain text."""
        characters: Counter = Counter()
        
        for pattern in self.DIALOGUE_PATTERNS:
            for match in pattern.finditer(content):
                # Pattern might have character name in group 1 or 2
                name = match.group(1).strip()
                if len(name) > 1 and len(name) < 50:
                    # Normalize name
                    name = " ".join(name.split()).title()
                    characters[name] += 1
        
        return characters
    
    def _extract_characters_from_json(self, content: str) -> Counter:
        """Extract character names from JSON/JSONL content."""
        characters: Counter = Counter()
        
        # Try JSONL format first
        lines = content.strip().split("\n")
        for line in lines:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    # Common speaker field names
                    for key in ["speaker", "character", "name", "role", "from"]:
                        if key in data:
                            name = str(data[key]).strip().title()
                            if name and len(name) < 50:
                                characters[name] += 1
                            break
            except json.JSONDecodeError:
                continue
        
        # Try single JSON object
        if not characters:
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key in ["speaker", "character", "name", "role", "from"]:
                                if key in item:
                                    name = str(item[key]).strip().title()
                                    if name and len(name) < 50:
                                        characters[name] += 1
                                    break
            except json.JSONDecodeError:
                pass
        
        return characters
    
    def extract_character(
        self,
        character_name: str,
        data_path: str,
        aliases: Optional[List[str]] = None
    ) -> ExtractionResult:
        """
        Extract all data for a specific character.
        
        Args:
            character_name: Name of character to extract
            data_path: Path to training data
            aliases: Alternative names for the character
            
        Returns:
            ExtractionResult with character profile and dialogues
        """
        path = Path(data_path)
        aliases = aliases or []
        all_names = [character_name] + aliases
        name_pattern = re.compile(
            r'(' + '|'.join(re.escape(n) for n in all_names) + r')[\s:]*(.+)',
            re.IGNORECASE
        )
        
        dialogues: List[str] = []
        qa_pairs: List[Tuple[str, str]] = []
        vocabulary: Counter = Counter()
        source_files: List[str] = []
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.txt")) + list(path.rglob("*.json")) + list(path.rglob("*.jsonl"))
        
        logger.info(f"Extracting '{character_name}' from {len(files)} files...")
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                found_in_file = False
                
                # Extract dialogues
                for line in content.split("\n"):
                    match = name_pattern.match(line.strip())
                    if match:
                        dialogue = match.group(2).strip()
                        if dialogue and len(dialogue) > 5:
                            dialogues.append(dialogue)
                            found_in_file = True
                            
                            # Update vocabulary
                            words = re.findall(r'\b[a-zA-Z]+\b', dialogue.lower())
                            vocabulary.update(
                                w for w in words if w not in self._stopwords
                            )
                
                if found_in_file:
                    source_files.append(str(file_path))
                
                # Try to extract Q&A pairs (look for user/assistant patterns)
                qa_pairs.extend(self._extract_qa_pairs(content, all_names))
                
            except Exception as e:
                logger.warning(f"Error extracting from {file_path}: {e}")
        
        if not dialogues:
            return ExtractionResult(
                success=False,
                error=f"No dialogues found for '{character_name}'"
            )
        
        # Build character profile
        profile = CharacterProfile(
            name=character_name,
            aliases=aliases,
            dialogue_count=len(dialogues),
            vocabulary=dict(vocabulary.most_common(200)),
            phrases=self._extract_phrases(dialogues),
            catchphrases=self._find_catchphrases(dialogues),
            speech_patterns=self._analyze_speech_patterns(dialogues),
            personality_traits=self._analyze_personality(dialogues),
            topics=self._extract_topics(dialogues),
            sample_dialogues=dialogues[:20],
            source_files=source_files
        )
        
        self.characters[character_name] = profile
        
        return ExtractionResult(
            success=True,
            character=profile,
            dialogue_lines=dialogues,
            qa_pairs=qa_pairs
        )
    
    def _extract_qa_pairs(
        self, 
        content: str, 
        character_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Extract Q&A pairs where character responds."""
        pairs = []
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            # Check if this is character speaking
            for name in character_names:
                if line.strip().lower().startswith(name.lower()):
                    # Try to find preceding question
                    if i > 0:
                        prev_line = lines[i-1].strip()
                        if prev_line and ":" in prev_line:
                            # Extract dialogue parts
                            _, q = prev_line.split(":", 1)
                            _, a = line.split(":", 1)
                            if q.strip() and a.strip():
                                pairs.append((q.strip(), a.strip()))
                    break
        
        return pairs
    
    def _extract_phrases(self, dialogues: List[str]) -> List[str]:
        """Extract common phrases from dialogues."""
        phrases: Counter = Counter()
        
        for dialogue in dialogues:
            # Extract 3-5 word phrases
            words = dialogue.split()
            for n in range(3, 6):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n]).lower()
                    # Filter out phrases starting/ending with common words
                    if not any(phrase.startswith(w) or phrase.endswith(w) 
                              for w in ["the", "a", "an", "and", "or", "but"]):
                        phrases[phrase] += 1
        
        # Return phrases that appear multiple times
        return [phrase for phrase, count in phrases.most_common(50) if count >= 2]
    
    def _find_catchphrases(self, dialogues: List[str]) -> List[str]:
        """Find repeated catchphrases unique to this character."""
        # Look for exact repeated phrases
        phrase_counts: Counter = Counter()
        
        for dialogue in dialogues:
            # Normalize and count
            normalized = dialogue.strip().lower()
            if 5 < len(normalized) < 100:
                phrase_counts[normalized] += 1
        
        # Also check for sentence beginnings
        starts: Counter = Counter()
        for dialogue in dialogues:
            words = dialogue.split()[:4]
            if len(words) >= 3:
                start = " ".join(words).lower()
                starts[start] += 1
        
        catchphrases = []
        
        # Add highly repeated full phrases
        for phrase, count in phrase_counts.most_common(10):
            if count >= 3:
                catchphrases.append(phrase)
        
        # Add common starts
        for start, count in starts.most_common(5):
            if count >= 5:
                catchphrases.append(start + "...")
        
        return catchphrases[:10]
    
    def _analyze_speech_patterns(self, dialogues: List[str]) -> Dict[str, float]:
        """Analyze speech patterns in dialogues."""
        total = len(dialogues) or 1
        patterns: Dict[str, float] = {
            "questions": 0.0,
            "exclamations": 0.0,
            "statements": 0.0,
            "interjections": 0.0,
            "avg_sentence_length": 0.0,
            "uses_contractions": 0.0,
            "formal_language": 0.0,
        }
        
        total_words = 0
        for dialogue in dialogues:
            total_words += len(dialogue.split())
            
            if dialogue.endswith("?"):
                patterns["questions"] += 1
            elif dialogue.endswith("!"):
                patterns["exclamations"] += 1
            else:
                patterns["statements"] += 1
            
            # Check for interjections
            if any(word in dialogue.lower() for word in ["oh", "ah", "hmm", "well", "wow"]):
                patterns["interjections"] += 1
            
            # Check for contractions
            if any(c in dialogue for c in ["'t", "'ll", "'ve", "'re", "'m", "'d"]):
                patterns["uses_contractions"] += 1
            
            # Check for formal language
            if any(word in dialogue.lower() for word in ["indeed", "therefore", "furthermore", "however", "nevertheless"]):
                patterns["formal_language"] += 1
        
        # Normalize to percentages
        patterns["questions"] = round(patterns["questions"] / total, 2)
        patterns["exclamations"] = round(patterns["exclamations"] / total, 2)
        patterns["statements"] = round(patterns["statements"] / total, 2)
        patterns["interjections"] = round(patterns["interjections"] / total, 2)
        patterns["uses_contractions"] = round(patterns["uses_contractions"] / total, 2)
        patterns["formal_language"] = round(patterns["formal_language"] / total, 2)
        patterns["avg_sentence_length"] = round(total_words / total, 1)
        
        return patterns
    
    def _analyze_personality(self, dialogues: List[str]) -> Dict[str, float]:
        """Analyze personality traits from dialogues."""
        all_text = " ".join(dialogues).lower()
        total_words = len(all_text.split()) or 1
        
        traits = {}
        for trait, keywords in self.TRAIT_KEYWORDS.items():
            count = sum(all_text.count(keyword) for keyword in keywords)
            # Score from 0-1 based on keyword density
            score = min(1.0, count / (total_words / 100))
            traits[trait] = round(score, 2)
        
        return traits
    
    def _extract_topics(self, dialogues: List[str]) -> List[str]:
        """Extract main topics discussed."""
        # Use noun phrases and common nouns
        all_text = " ".join(dialogues).lower()
        
        # Simple topic extraction via word frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        word_counts = Counter(w for w in words if w not in self._stopwords)
        
        return [word for word, _ in word_counts.most_common(20)]
    
    def generate_training_dataset(
        self,
        character_name: str,
        data_path: str,
        output_path: str,
        format: str = "qa",
        system_prompt: Optional[str] = None
    ) -> bool:
        """
        Generate a training dataset for a character.
        
        Args:
            character_name: Character to generate dataset for
            data_path: Source data path
            output_path: Where to save the dataset
            format: "qa" for Q&A pairs, "chat" for chat format, "raw" for dialogues
            system_prompt: Optional system prompt to prepend
            
        Returns:
            True if successful
        """
        # Extract character if not already done
        if character_name not in self.characters:
            result = self.extract_character(character_name, data_path)
            if not result.success:
                logger.error(result.error)
                return False
        
        profile = self.characters[character_name]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate system prompt if not provided
        if not system_prompt:
            system_prompt = self._generate_system_prompt(profile)
        
        lines = []
        
        if format == "qa":
            # Q&A format with system prompt
            lines.append(f"# System: {system_prompt}\n")
            lines.append("")
            
            # Add sample dialogues as training data
            for dialogue in profile.sample_dialogues:
                # Create synthetic Q&A
                lines.append(f"Q: Respond as {character_name} would.")
                lines.append(f"A: {dialogue}")
                lines.append("")
        
        elif format == "chat":
            # Chat format
            lines.append(f"system: {system_prompt}")
            lines.append("")
            
            for dialogue in profile.sample_dialogues:
                lines.append(f"user: Say something as {character_name}.")
                lines.append(f"assistant: {dialogue}")
                lines.append("")
        
        elif format == "raw":
            # Raw dialogues with character prefix
            for dialogue in profile.sample_dialogues:
                lines.append(f"{character_name}: {dialogue}")
        
        output.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Generated training dataset: {output}")
        return True
    
    def _generate_system_prompt(self, profile: CharacterProfile) -> str:
        """Generate a system prompt from character profile."""
        prompt_parts = [f"You are {profile.name}."]
        
        # Add personality traits
        strong_traits = [
            trait for trait, score in profile.personality_traits.items()
            if score > 0.3
        ]
        if strong_traits:
            prompt_parts.append(f"Your personality is {', '.join(strong_traits)}.")
        
        # Add speech patterns
        if profile.speech_patterns.get("formal_language", 0) > 0.3:
            prompt_parts.append("You speak formally.")
        if profile.speech_patterns.get("uses_contractions", 0) > 0.5:
            prompt_parts.append("You use contractions frequently.")
        if profile.speech_patterns.get("questions", 0) > 0.3:
            prompt_parts.append("You often ask questions.")
        
        # Add catchphrases
        if profile.catchphrases:
            prompt_parts.append(f"Your catchphrases include: '{profile.catchphrases[0]}'")
        
        # Add topics
        if profile.topics:
            prompt_parts.append(f"You often discuss: {', '.join(profile.topics[:5])}.")
        
        return " ".join(prompt_parts)
    
    def save_profile(self, character_name: str, output_path: str) -> bool:
        """Save character profile to JSON."""
        if character_name not in self.characters:
            logger.error(f"Character '{character_name}' not found")
            return False
        
        profile = self.characters[character_name]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        output.write_text(
            json.dumps(profile.to_dict(), indent=2),
            encoding="utf-8"
        )
        logger.info(f"Saved profile: {output}")
        return True
    
    def load_profile(self, path: str) -> Optional[CharacterProfile]:
        """Load character profile from JSON."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            profile = CharacterProfile.from_dict(data)
            self.characters[profile.name] = profile
            return profile
        except Exception as e:
            logger.error(f"Error loading profile: {e}")
            return None


# Convenience function
def get_character_trainer() -> CharacterTrainer:
    """Get a CharacterTrainer instance."""
    return CharacterTrainer()


# ==============================================================================
# TASK-BASED TRAINER - Train AI on specific capabilities
# ==============================================================================

@dataclass
class TaskExample:
    """A single training example for a task."""
    task_type: str
    input_prompt: str
    expected_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    
    def to_training_line(self, format: str = "qa") -> str:
        """Convert to training format."""
        if format == "qa":
            return f"Q: {self.input_prompt}\nA: {self.expected_output}"
        elif format == "chat":
            return f"user: {self.input_prompt}\nassistant: {self.expected_output}"
        else:
            return f"{self.input_prompt} -> {self.expected_output}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "input": self.input_prompt,
            "output": self.expected_output,
            "metadata": self.metadata,
            "difficulty": self.difficulty
        }
    
    @classmethod
    def from_dict(cls, task_type: str, data: Dict[str, Any]) -> "TaskExample":
        """Create from dictionary."""
        return cls(
            task_type=task_type,
            input_prompt=data.get("input", ""),
            expected_output=data.get("output", ""),
            metadata=data.get("metadata", {}),
            difficulty=data.get("difficulty", "medium")
        )


class TaskTrainer:
    """
    Train AI on specific tasks like image generation, avatar control, and tool usage.
    
    This is a lightweight trainer that loads examples from JSON files on demand.
    No hardcoded examples - all training data comes from external files.
    
    Usage:
        trainer = TaskTrainer()
        
        # Load examples from JSON files
        trainer.load_examples_from_file("data/training/image_examples.json")
        trainer.load_examples_from_file("data/training/avatar_examples.json")
        
        # Or load from a directory
        trainer.load_examples_from_directory("data/training/")
        
        # Add custom examples
        trainer.add_example("image", "Create a cat", "Fluffy orange tabby cat...")
        
        # Generate training datasets
        trainer.generate_image_training("outputs/image_training.txt")
        trainer.generate_all_tasks("outputs/complete_training.txt")
        
        # Export examples back to JSON
        trainer.export_examples("data/training/my_examples.json")
    
    Example JSON format for loading:
        {
            "image": [{"input": "prompt", "output": "response", "difficulty": "easy"}],
            "avatar": [{"input": "greet user", "output": "[emotion:happy] Hello!"}],
            "tools": [...],
            "code": [...]
        }
    """
    
    # Default task types
    DEFAULT_TASK_TYPES = ["image", "avatar", "tools", "code", "web", "file"]
    
    def __init__(self, examples_dir: Optional[str] = None):
        """
        Initialize TaskTrainer.
        
        Args:
            examples_dir: Optional directory to auto-load examples from
        """
        self.examples: Dict[str, List[TaskExample]] = {
            task: [] for task in self.DEFAULT_TASK_TYPES
        }
        self._loaded_files: Set[str] = set()
        
        # Auto-load from directory if provided
        if examples_dir:
            self.load_examples_from_directory(examples_dir)
    
    # =========================================================================
    # EXAMPLE LOADING METHODS
    # =========================================================================
    
    def load_examples_from_directory(self, dir_path: str) -> int:
        """
        Load all JSON example files from a directory.
        
        Args:
            dir_path: Path to directory containing JSON files
            
        Returns:
            Total number of examples loaded
        """
        total = 0
        path = Path(dir_path)
        
        if not path.exists():
            logger.warning(f"Examples directory not found: {dir_path}")
            return 0
        
        for json_file in path.glob("*.json"):
            if str(json_file) not in self._loaded_files:
                count = self.load_examples_from_file(str(json_file))
                total += count
        
        logger.info(f"Loaded {total} examples from {dir_path}")
        return total
    
    def export_examples(self, output_path: str, task_types: Optional[List[str]] = None) -> bool:
        """
        Export current examples to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            task_types: Optional list of task types to export (default: all)
            
        Returns:
            True if successful
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        task_types = task_types or list(self.examples.keys())
        
        data = {}
        for task_type in task_types:
            if task_type in self.examples:
                data[task_type] = [ex.to_dict() for ex in self.examples[task_type]]
        
        try:
            output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info(f"Exported {sum(len(v) for v in data.values())} examples to {output}")
            return True
        except Exception as e:
            logger.error(f"Failed to export examples: {e}")
            return False
    
    def clear_examples(self, task_type: Optional[str] = None) -> None:
        """
        Clear loaded examples.
        
        Args:
            task_type: Specific task type to clear, or None for all
        """
        if task_type:
            self.examples[task_type] = []
        else:
            for key in self.examples:
                self.examples[key] = []
        self._loaded_files.clear()
    
    def register_task_type(self, task_type: str) -> None:
        """Register a new task type for custom training."""
        if task_type not in self.examples:
            self.examples[task_type] = []
    
    # =========================================================================
    # TRAINING DATA GENERATION
    # =========================================================================
    
    def generate_image_training(
        self,
        output_path: str,
        format: str = "qa",
        include_system_prompt: bool = True
    ) -> bool:
        """Generate training data for image prompt generation."""
        return self._generate_task_training(
            "image",
            output_path,
            format,
            include_system_prompt,
            system_prompt="""You are an expert at creating detailed image generation prompts.
When given a basic image request, expand it into a detailed, descriptive prompt that includes:
- Subject details (what exactly is being depicted)
- Style (photorealistic, anime, oil painting, etc.)
- Mood and atmosphere
- Lighting conditions
- Color palette
- Composition and perspective
- Any artistic influences or references

Transform vague requests into vivid, detailed descriptions that will produce high-quality images."""
        )
    
    def generate_avatar_training(
        self,
        output_path: str,
        format: str = "qa",
        include_system_prompt: bool = True
    ) -> bool:
        """Generate training data for avatar control."""
        return self._generate_task_training(
            "avatar",
            output_path,
            format,
            include_system_prompt,
            system_prompt="""You are an AI with an expressive avatar that can display emotions and perform gestures.
Use avatar control commands in your responses to express yourself:
- [emotion:X] - Set emotion (happy, sad, surprised, thinking, excited, neutral, curious, apologetic, determined, encouraging)
- [gesture:X] - Perform gesture (wave, nod, shake, point, shrug, clap, thumbs_up, chin_tap, bow, lean_back)
- [action:X] - Do action (think, laugh, listen, explain, work, celebrate)
- [expression:X] - Show expression (smile, frown, gasp, wink, big_smile)

Commands are stripped from displayed text - users only see your words.
Match your avatar expressions to the emotional tone of your responses.
Use multiple commands when appropriate to create natural, expressive interactions."""
        )
    
    def generate_tool_training(
        self,
        output_path: str,
        format: str = "qa",
        include_system_prompt: bool = True
    ) -> bool:
        """Generate training data for tool usage."""
        return self._generate_task_training(
            "tools",
            output_path,
            format,
            include_system_prompt,
            system_prompt="""You are an AI assistant with access to various tools.
When a task requires external capabilities, use the appropriate tool:
- [TOOL:web_search(query='...')] - Search the web for information
- [TOOL:read_file(path='...')] - Read a file's contents
- [TOOL:write_file(path='...', content='...')] - Save content to a file
- [TOOL:list_directory(path='...')] - List files in a directory
- [TOOL:generate_image(prompt='...')] - Generate an image
- [TOOL:execute_code(code='...')] - Run Python code
- [TOOL:analyze_image(source='...')] - Analyze an image

Always explain what you're doing and report the results.
Combine multiple tools when needed for complex tasks."""
        )
    
    def generate_code_training(
        self,
        output_path: str,
        format: str = "qa",
        include_system_prompt: bool = True
    ) -> bool:
        """Generate training data for code generation."""
        return self._generate_task_training(
            "code",
            output_path,
            format,
            include_system_prompt,
            system_prompt="""You are an expert programmer and code assistant.
When writing code:
- Include proper documentation (docstrings, comments)
- Handle edge cases and errors
- Follow best practices for the language
- Explain your implementation approach
- Consider efficiency and maintainability
- Use type hints where appropriate

Format code in markdown code blocks with the language specified."""
        )
    
    def generate_all_tasks(
        self,
        output_path: str,
        format: str = "qa"
    ) -> bool:
        """Generate combined training data for all tasks."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "# Enigma AI Engine Task Training Dataset",
            "# This dataset trains the AI on multiple capabilities",
            "",
            "=" * 60,
            "SECTION: Image Prompt Generation",
            "=" * 60,
            ""
        ]
        
        # Add image examples
        for example in self.examples["image"]:
            lines.append(example.to_training_line(format))
            lines.append("")
        
        lines.extend([
            "=" * 60,
            "SECTION: Avatar Control",
            "=" * 60,
            ""
        ])
        
        # Add avatar examples
        for example in self.examples["avatar"]:
            lines.append(example.to_training_line(format))
            lines.append("")
        
        lines.extend([
            "=" * 60,
            "SECTION: Tool Usage",
            "=" * 60,
            ""
        ])
        
        # Add tool examples
        for example in self.examples["tools"]:
            lines.append(example.to_training_line(format))
            lines.append("")
        
        lines.extend([
            "=" * 60,
            "SECTION: Code Generation",
            "=" * 60,
            ""
        ])
        
        # Add code examples
        for example in self.examples["code"]:
            lines.append(example.to_training_line(format))
            lines.append("")
        
        try:
            output.write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"Generated combined task training: {output}")
            return True
        except Exception as e:
            logger.error(f"Failed to write training data: {e}")
            return False
    
    def _generate_task_training(
        self,
        task_type: str,
        output_path: str,
        format: str,
        include_system_prompt: bool,
        system_prompt: str
    ) -> bool:
        """Generate training data for a specific task type."""
        if task_type not in self.examples:
            logger.error(f"Unknown task type: {task_type}")
            return False
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        
        if include_system_prompt:
            lines.append(f"# System: {system_prompt}")
            lines.append("")
        
        for example in self.examples[task_type]:
            lines.append(example.to_training_line(format))
            lines.append("")
        
        try:
            output.write_text("\n".join(lines), encoding="utf-8")
            logger.info(f"Generated {task_type} training: {output}")
            return True
        except Exception as e:
            logger.error(f"Failed to write training data: {e}")
            return False
    
    def add_example(
        self,
        task_type: str,
        input_prompt: str,
        expected_output: str,
        metadata: Optional[Dict[str, Any]] = None,
        difficulty: str = "medium"
    ) -> bool:
        """Add a custom training example."""
        if task_type not in self.examples:
            self.examples[task_type] = []
        
        self.examples[task_type].append(TaskExample(
            task_type=task_type,
            input_prompt=input_prompt,
            expected_output=expected_output,
            metadata=metadata or {},
            difficulty=difficulty
        ))
        return True
    
    def load_examples_from_file(self, path: str) -> int:
        """
        Load custom examples from a JSON file.
        
        Expected format:
        {
            "image": [{"input": "...", "output": "...", "difficulty": "easy"}],
            "avatar": [...],
            ...
        }
        
        Returns:
            Number of examples loaded
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            count = 0
            
            for task_type, examples in data.items():
                if task_type not in self.examples:
                    self.examples[task_type] = []
                
                for ex in examples:
                    self.examples[task_type].append(TaskExample(
                        task_type=task_type,
                        input_prompt=ex.get("input", ""),
                        expected_output=ex.get("output", ""),
                        metadata=ex.get("metadata", {}),
                        difficulty=ex.get("difficulty", "medium")
                    ))
                    count += 1
            
            logger.info(f"Loaded {count} examples from {path}")
            return count
        except Exception as e:
            logger.error(f"Failed to load examples: {e}")
            return 0
    
    def get_example_count(self, task_type: Optional[str] = None) -> int:
        """Get count of examples, optionally filtered by type."""
        if task_type:
            return len(self.examples.get(task_type, []))
        return sum(len(ex) for ex in self.examples.values())
    
    def get_task_types(self) -> List[str]:
        """Get all available task types."""
        return list(self.examples.keys())


def get_task_trainer() -> TaskTrainer:
    """Get a TaskTrainer instance."""
    return TaskTrainer()
