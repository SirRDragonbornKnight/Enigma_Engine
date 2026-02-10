"""
Gaming & Fun Tools - Games, roleplay, story generation, D&D.

Tools:
  - trivia_game: Play trivia questions
  - word_game: Word games (hangman, word scramble, etc.)
  - number_guess: Number guessing game
  - twenty_questions: 20 questions game
  - character_create: Create an AI character/persona
  - character_chat: Chat as a character
  - story_generate: Generate interactive stories
  - story_continue: Continue a story
  - dnd_roll: Roll dice for D&D
  - dnd_character: Generate D&D character
  - dnd_encounter: Generate D&D encounter
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .tool_registry import RichParameter, Tool

logger = logging.getLogger(__name__)

# Storage paths
GAMES_DIR = Path.home() / ".enigma_engine" / "games"
CHARACTERS_DIR = Path.home() / ".enigma_engine" / "characters"
STORIES_DIR = Path.home() / ".enigma_engine" / "stories"

GAMES_DIR.mkdir(parents=True, exist_ok=True)
CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
STORIES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TRIVIA TOOLS
# ============================================================================

class TriviaGameTool(Tool):
    """Play trivia questions."""
    
    name = "trivia_game"
    description = "Get a trivia question from various categories."
    parameters = {
        "category": "Category: 'general', 'science', 'history', 'geography', 'entertainment', 'sports' (default: random)",
        "difficulty": "Difficulty: 'easy', 'medium', 'hard' (default: medium)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="category",
            type="string",
            description="Trivia category",
            required=False,
            default="random",
            enum=["general", "science", "history", "geography", "entertainment", "sports", "random"]
        ),
        RichParameter(
            name="difficulty",
            type="string",
            description="Question difficulty",
            required=False,
            default="medium",
            enum=["easy", "medium", "hard"]
        ),
    ]
    examples = [
        "trivia_game() - Random trivia question",
        "trivia_game(category='science', difficulty='hard') - Hard science question",
    ]
    
    # Built-in trivia questions
    TRIVIA = {
        "general": [
            {"q": "What is the largest planet in our solar system?", "a": "Jupiter", "options": ["Saturn", "Jupiter", "Neptune", "Uranus"]},
            {"q": "How many continents are there?", "a": "7", "options": ["5", "6", "7", "8"]},
            {"q": "What is the chemical symbol for gold?", "a": "Au", "options": ["Go", "Gd", "Au", "Ag"]},
            {"q": "Which ocean is the largest?", "a": "Pacific", "options": ["Atlantic", "Indian", "Pacific", "Arctic"]},
            {"q": "What year did World War II end?", "a": "1945", "options": ["1943", "1944", "1945", "1946"]},
        ],
        "science": [
            {"q": "What is the speed of light in km/s (approximately)?", "a": "300,000", "options": ["150,000", "300,000", "500,000", "1,000,000"]},
            {"q": "What is the hardest natural substance?", "a": "Diamond", "options": ["Gold", "Iron", "Diamond", "Platinum"]},
            {"q": "What planet is known as the Red Planet?", "a": "Mars", "options": ["Venus", "Mars", "Jupiter", "Mercury"]},
            {"q": "What is H2O commonly known as?", "a": "Water", "options": ["Hydrogen", "Oxygen", "Water", "Helium"]},
            {"q": "How many bones are in the adult human body?", "a": "206", "options": ["186", "206", "226", "256"]},
        ],
        "history": [
            {"q": "Who was the first President of the United States?", "a": "George Washington", "options": ["John Adams", "Thomas Jefferson", "George Washington", "Benjamin Franklin"]},
            {"q": "In what year did the Titanic sink?", "a": "1912", "options": ["1905", "1912", "1920", "1898"]},
            {"q": "Which ancient wonder was located in Alexandria?", "a": "Lighthouse", "options": ["Pyramid", "Lighthouse", "Colossus", "Gardens"]},
            {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci", "options": ["Michelangelo", "Raphael", "Leonardo da Vinci", "Donatello"]},
            {"q": "What year did the Berlin Wall fall?", "a": "1989", "options": ["1987", "1988", "1989", "1990"]},
        ],
        "geography": [
            {"q": "What is the capital of Australia?", "a": "Canberra", "options": ["Sydney", "Melbourne", "Canberra", "Perth"]},
            {"q": "Which river is the longest in the world?", "a": "Nile", "options": ["Amazon", "Nile", "Yangtze", "Mississippi"]},
            {"q": "What is the smallest country in the world?", "a": "Vatican City", "options": ["Monaco", "Vatican City", "San Marino", "Liechtenstein"]},
            {"q": "Which mountain is the tallest?", "a": "Mount Everest", "options": ["K2", "Mount Everest", "Kangchenjunga", "Makalu"]},
            {"q": "What country has the most time zones?", "a": "France", "options": ["Russia", "USA", "France", "China"]},
        ],
        "entertainment": [
            {"q": "What is the highest-grossing film of all time?", "a": "Avatar", "options": ["Titanic", "Avatar", "Avengers: Endgame", "Star Wars"]},
            {"q": "Who wrote Harry Potter?", "a": "J.K. Rowling", "options": ["Stephen King", "J.K. Rowling", "George R.R. Martin", "Suzanne Collins"]},
            {"q": "What band was Freddie Mercury the lead singer of?", "a": "Queen", "options": ["The Beatles", "Queen", "Led Zeppelin", "Pink Floyd"]},
            {"q": "In what year was the first iPhone released?", "a": "2007", "options": ["2005", "2006", "2007", "2008"]},
            {"q": "What is the name of Batman's butler?", "a": "Alfred", "options": ["James", "Alfred", "Bruce", "Gordon"]},
        ],
        "sports": [
            {"q": "How many players are on a soccer team?", "a": "11", "options": ["9", "10", "11", "12"]},
            {"q": "In which sport would you perform a slam dunk?", "a": "Basketball", "options": ["Volleyball", "Basketball", "Tennis", "Football"]},
            {"q": "What country hosted the 2016 Summer Olympics?", "a": "Brazil", "options": ["China", "UK", "Brazil", "Russia"]},
            {"q": "How many holes are on a standard golf course?", "a": "18", "options": ["9", "12", "18", "21"]},
            {"q": "What sport is played at Wimbledon?", "a": "Tennis", "options": ["Cricket", "Tennis", "Golf", "Rugby"]},
        ],
    }
    
    def execute(self, category: str = None, difficulty: str = "medium", **kwargs) -> dict[str, Any]:
        try:
            # Select category
            if not category or category == "random":
                category = random.choice(list(self.TRIVIA.keys()))
            
            if category not in self.TRIVIA:
                return {"success": False, "error": f"Unknown category. Available: {list(self.TRIVIA.keys())}"}
            
            # Get random question
            question = random.choice(self.TRIVIA[category])
            
            # Shuffle options
            options = question["options"].copy()
            random.shuffle(options)
            
            return {
                "success": True,
                "category": category,
                "difficulty": difficulty,
                "question": question["q"],
                "options": options,
                "answer": question["a"],  # In real game, hide this until user answers
                "hint": f"The answer starts with '{question['a'][0]}'",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# WORD GAMES
# ============================================================================

class WordGameTool(Tool):
    """Play word games."""
    
    name = "word_game"
    description = "Play word games like hangman, word scramble, or anagrams."
    parameters = {
        "game_type": "Game: 'hangman', 'scramble', 'anagram' (default: scramble)",
        "difficulty": "Difficulty: 'easy', 'medium', 'hard' (default: medium)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="game_type",
            type="string",
            description="Type of word game to play",
            required=False,
            default="scramble",
            enum=["hangman", "scramble", "anagram"]
        ),
        RichParameter(
            name="difficulty",
            type="string",
            description="Word difficulty",
            required=False,
            default="medium",
            enum=["easy", "medium", "hard"]
        ),
    ]
    examples = [
        "word_game() - Word scramble game",
        "word_game(game_type='hangman', difficulty='hard') - Hard hangman",
    ]
    
    WORDS = {
        "easy": ["cat", "dog", "sun", "moon", "tree", "book", "fish", "bird", "cake", "star"],
        "medium": ["python", "rocket", "garden", "bridge", "castle", "planet", "forest", "island"],
        "hard": ["algorithm", "telescope", "philosophy", "electricity", "mysterious", "adventure"],
    }
    
    def execute(self, game_type: str = "scramble", difficulty: str = "medium", **kwargs) -> dict[str, Any]:
        try:
            words = self.WORDS.get(difficulty, self.WORDS["medium"])
            word = random.choice(words)
            
            if game_type == "scramble":
                # Scramble the word
                letters = list(word)
                random.shuffle(letters)
                scrambled = ''.join(letters)
                
                # Make sure it's actually scrambled
                while scrambled == word and len(word) > 1:
                    random.shuffle(letters)
                    scrambled = ''.join(letters)
                
                return {
                    "success": True,
                    "game": "word_scramble",
                    "scrambled": scrambled,
                    "hint": f"{len(word)} letters",
                    "answer": word,
                }
                
            elif game_type == "hangman":
                return {
                    "success": True,
                    "game": "hangman",
                    "word_length": len(word),
                    "display": "_ " * len(word),
                    "hint": f"Category: common words",
                    "answer": word,
                    "guesses_left": 6,
                }
                
            elif game_type == "anagram":
                # Find words that can be made from the letters
                return {
                    "success": True,
                    "game": "anagram",
                    "letters": word.upper(),
                    "hint": f"Find words using these {len(word)} letters",
                    "main_word": word,
                }
            
            return {"success": False, "error": "Unknown game type"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class NumberGuessTool(Tool):
    """Number guessing game."""
    
    name = "number_guess"
    description = "Start a number guessing game. I think of a number, you guess!"
    parameters = {
        "min_num": "Minimum number (default: 1)",
        "max_num": "Maximum number (default: 100)",
        "guess": "Your guess (optional - start game if not provided)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="min_num",
            type="integer",
            description="Minimum number in the range",
            required=False,
            default=1,
            min_value=1,
        ),
        RichParameter(
            name="max_num",
            type="integer",
            description="Maximum number in the range",
            required=False,
            default=100,
            max_value=1000000,
        ),
        RichParameter(
            name="guess",
            type="integer",
            description="Your guess (omit to start a new game)",
            required=False,
        ),
    ]
    examples = [
        "number_guess() - Start new game 1-100",
        "number_guess(min_num=1, max_num=50) - Custom range game",
        "number_guess(guess=42) - Make a guess",
    ]
    
    # Store active games
    _active_games = {}
    
    def execute(self, min_num: int = 1, max_num: int = 100, guess: int = None, **kwargs) -> dict[str, Any]:
        try:
            game_id = "current"
            
            if guess is None:
                # Start new game
                secret = random.randint(int(min_num), int(max_num))
                self._active_games[game_id] = {
                    "secret": secret,
                    "min": int(min_num),
                    "max": int(max_num),
                    "attempts": 0,
                }
                
                return {
                    "success": True,
                    "message": f"I'm thinking of a number between {min_num} and {max_num}. Make a guess!",
                    "range": [int(min_num), int(max_num)],
                }
            
            # Process guess
            game = self._active_games.get(game_id)
            if not game:
                return {"success": False, "error": "No active game. Start a new one first."}
            
            game["attempts"] += 1
            guess = int(guess)
            secret = game["secret"]
            
            if guess == secret:
                del self._active_games[game_id]
                return {
                    "success": True,
                    "correct": True,
                    "message": f"🎉 Correct! The number was {secret}. You got it in {game['attempts']} attempts!",
                    "attempts": game["attempts"],
                }
            elif guess < secret:
                return {
                    "success": True,
                    "correct": False,
                    "hint": "higher",
                    "message": f"📈 Higher! Try a bigger number.",
                    "attempts": game["attempts"],
                }
            else:
                return {
                    "success": True,
                    "correct": False,
                    "hint": "lower",
                    "message": f"📉 Lower! Try a smaller number.",
                    "attempts": game["attempts"],
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# CHARACTER ROLEPLAY
# ============================================================================

class CharacterManager:
    """Manages AI characters/personas."""
    
    def __init__(self):
        self.characters: dict[str, dict] = {}
        self._load_characters()
    
    def _load_characters(self):
        for file in CHARACTERS_DIR.glob("*.json"):
            try:
                with open(file) as f:
                    char = json.load(f)
                    self.characters[char['name'].lower()] = char
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(f"Could not load character {file}: {e}")
    
    def save_character(self, character: dict):
        name = character['name'].lower()
        self.characters[name] = character
        with open(CHARACTERS_DIR / f"{name}.json", 'w') as f:
            json.dump(character, f, indent=2)
    
    def get_character(self, name: str) -> Optional[dict]:
        return self.characters.get(name.lower())
    
    def list_characters(self) -> list[str]:
        return list(self.characters.keys())
    
    def delete_character(self, name: str):
        name = name.lower()
        if name in self.characters:
            del self.characters[name]
            file = CHARACTERS_DIR / f"{name}.json"
            if file.exists():
                file.unlink()


class CharacterCreateTool(Tool):
    """Create an AI character."""
    
    name = "character_create"
    description = "Create a persistent AI character/persona with personality traits and backstory."
    parameters = {
        "name": "Character name",
        "personality": "Personality description (e.g., 'friendly wizard', 'grumpy detective')",
        "backstory": "Character backstory",
        "speaking_style": "How they speak (e.g., 'formal', 'pirate accent', 'medieval')",
        "traits": "Key traits (comma-separated)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="name",
            type="string",
            description="Character name",
            required=True,
        ),
        RichParameter(
            name="personality",
            type="string",
            description="Personality description",
            required=True,
        ),
        RichParameter(
            name="backstory",
            type="string",
            description="Character backstory or history",
            required=False,
        ),
        RichParameter(
            name="speaking_style",
            type="string",
            description="How the character speaks",
            required=False,
            default="normal",
        ),
        RichParameter(
            name="traits",
            type="string",
            description="Comma-separated personality traits",
            required=False,
        ),
    ]
    examples = [
        "character_create(name='Merlin', personality='wise wizard', speaking_style='formal')",
        "character_create(name='Jack', personality='pirate captain', traits='brave,cunning,loyal')",
    ]
    
    def execute(self, name: str, personality: str, backstory: str = "",
                speaking_style: str = "normal", traits: str = "", **kwargs) -> dict[str, Any]:
        try:
            trait_list = [t.strip() for t in traits.split(',')] if traits else []
            
            character = {
                "name": name,
                "personality": personality,
                "backstory": backstory,
                "speaking_style": speaking_style,
                "traits": trait_list,
                "created": datetime.now().isoformat(),
                "conversation_history": [],
            }
            
            manager = CharacterManager()
            manager.save_character(character)
            
            return {
                "success": True,
                "message": f"Character '{name}' created!",
                "character": character,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CharacterListTool(Tool):
    """List all characters."""
    
    name = "character_list"
    description = "List all created AI characters."
    parameters = {}
    category = "gaming"
    rich_parameters = []  # No parameters
    examples = ["character_list() - List all characters"]
    
    def execute(self, **kwargs) -> dict[str, Any]:
        try:
            manager = CharacterManager()
            characters = []
            
            for name in manager.list_characters():
                char = manager.get_character(name)
                if char:
                    characters.append({
                        "name": char['name'],
                        "personality": char['personality'],
                        "traits": char.get('traits', []),
                    })
            
            return {
                "success": True,
                "count": len(characters),
                "characters": characters,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CharacterChatTool(Tool):
    """Chat as a character."""
    
    name = "character_chat"
    description = "Get the AI to respond as a specific character."
    parameters = {
        "character_name": "Name of the character to use",
        "message": "Message to respond to as the character",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="character_name",
            type="string",
            description="Name of the character to roleplay as",
            required=True,
        ),
        RichParameter(
            name="message",
            type="string",
            description="Message for the character to respond to",
            required=True,
        ),
    ]
    examples = [
        "character_chat(character_name='Merlin', message='Hello wizard!')",
        "character_chat(character_name='Jack', message='Tell me about your ship')",
    ]
    
    def execute(self, character_name: str, message: str, **kwargs) -> dict[str, Any]:
        try:
            manager = CharacterManager()
            character = manager.get_character(character_name)
            
            if not character:
                return {"success": False, "error": f"Character '{character_name}' not found"}
            
            # Build character prompt
            prompt = f"""You are {character['name']}, a {character['personality']}.
Backstory: {character.get('backstory', 'Unknown')}
Speaking style: {character.get('speaking_style', 'normal')}
Traits: {', '.join(character.get('traits', []))}

Respond to this message in character:
User: {message}

{character['name']}:"""
            
            return {
                "success": True,
                "character": character['name'],
                "prompt_for_ai": prompt,
                "personality": character['personality'],
                "note": "Use this prompt with the AI to get an in-character response",
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# STORY GENERATION
# ============================================================================

class StoryManager:
    """Manages interactive stories."""
    
    def __init__(self):
        self.stories: dict[str, dict] = {}
        self._load_stories()
    
    def _load_stories(self):
        for file in STORIES_DIR.glob("*.json"):
            try:
                with open(file) as f:
                    story = json.load(f)
                    self.stories[story['id']] = story
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(f"Could not load story {file}: {e}")
    
    def save_story(self, story: dict):
        self.stories[story['id']] = story
        with open(STORIES_DIR / f"{story['id']}.json", 'w') as f:
            json.dump(story, f, indent=2)
    
    def get_story(self, story_id: str) -> Optional[dict]:
        return self.stories.get(story_id)
    
    def list_stories(self) -> list[dict]:
        return [{"id": s['id'], "title": s['title'], "genre": s.get('genre')} 
                for s in self.stories.values()]


class StoryGenerateTool(Tool):
    """Generate an interactive story."""
    
    name = "story_generate"
    description = "Start a new interactive story with choices."
    parameters = {
        "title": "Story title",
        "genre": "Genre: 'fantasy', 'scifi', 'mystery', 'horror', 'romance', 'adventure'",
        "setting": "Story setting description",
        "protagonist": "Main character description",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="title",
            type="string",
            description="Title for the story",
            required=True,
        ),
        RichParameter(
            name="genre",
            type="string",
            description="Story genre",
            required=False,
            default="fantasy",
            enum=["fantasy", "scifi", "mystery", "horror", "romance", "adventure"]
        ),
        RichParameter(
            name="setting",
            type="string",
            description="Description of the story setting",
            required=False,
        ),
        RichParameter(
            name="protagonist",
            type="string",
            description="Main character description",
            required=False,
            default="our hero",
        ),
    ]
    examples = [
        "story_generate(title='The Lost Kingdom', genre='fantasy')",
        "story_generate(title='Mars Colony', genre='scifi', protagonist='Captain Chen')",
    ]
    
    # Story templates
    OPENINGS = {
        "fantasy": "In a realm where magic flows like rivers and dragons soar through clouded skies, {protagonist} stood at the crossroads of destiny.",
        "scifi": "The year is 2347. Aboard the starship Nexus, {protagonist} received a transmission that would change everything.",
        "mystery": "The rain hadn't stopped for three days when {protagonist} found the envelope under their door. Inside: a single photograph and a cryptic message.",
        "horror": "The house at the end of Willow Lane had been abandoned for decades. When {protagonist} inherited it, the locals warned them to stay away.",
        "romance": "Their eyes met across the crowded café, and {protagonist} felt their heart skip a beat. This was the moment everything changed.",
        "adventure": "The ancient map had been passed down through generations. Now, {protagonist} finally had the chance to discover what lay at the X.",
    }
    
    def execute(self, title: str, genre: str = "fantasy", setting: str = "",
                protagonist: str = "our hero", **kwargs) -> dict[str, Any]:
        try:
            story_id = f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate opening
            opening_template = self.OPENINGS.get(genre, self.OPENINGS["adventure"])
            opening = opening_template.format(protagonist=protagonist)
            
            if setting:
                opening = f"{setting}\n\n{opening}"
            
            # Generate choices
            choices = self._generate_choices(genre)
            
            story = {
                "id": story_id,
                "title": title,
                "genre": genre,
                "protagonist": protagonist,
                "setting": setting,
                "created": datetime.now().isoformat(),
                "chapters": [{
                    "number": 1,
                    "text": opening,
                    "choices": choices,
                }],
                "current_chapter": 1,
            }
            
            manager = StoryManager()
            manager.save_story(story)
            
            return {
                "success": True,
                "story_id": story_id,
                "title": title,
                "chapter": 1,
                "text": opening,
                "choices": choices,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_choices(self, genre: str) -> list[dict]:
        choices_by_genre = {
            "fantasy": [
                {"id": "A", "text": "Follow the mysterious light into the forest"},
                {"id": "B", "text": "Seek counsel from the village elder"},
                {"id": "C", "text": "Ignore the omen and continue your journey"},
            ],
            "scifi": [
                {"id": "A", "text": "Respond to the transmission"},
                {"id": "B", "text": "Report it to the captain"},
                {"id": "C", "text": "Investigate the source coordinates"},
            ],
            "mystery": [
                {"id": "A", "text": "Take the photograph to a detective friend"},
                {"id": "B", "text": "Visit the location shown in the photo"},
                {"id": "C", "text": "Destroy the evidence and pretend nothing happened"},
            ],
            "horror": [
                {"id": "A", "text": "Enter through the front door"},
                {"id": "B", "text": "Search for another way in"},
                {"id": "C", "text": "Wait until morning to explore"},
            ],
            "romance": [
                {"id": "A", "text": "Walk over and introduce yourself"},
                {"id": "B", "text": "Send them a drink from across the room"},
                {"id": "C", "text": "Wait and hope they come to you"},
            ],
            "adventure": [
                {"id": "A", "text": "Set out immediately with minimal supplies"},
                {"id": "B", "text": "Gather a team of trusted companions"},
                {"id": "C", "text": "Research the map's origins first"},
            ],
        }
        return choices_by_genre.get(genre, choices_by_genre["adventure"])


class StoryContinueTool(Tool):
    """Continue a story with a choice."""
    
    name = "story_continue"
    description = "Continue an interactive story by making a choice."
    parameters = {
        "story_id": "ID of the story to continue",
        "choice": "The choice to make (A, B, C, etc.)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="story_id",
            type="string",
            description="ID of the story (from story_generate result)",
            required=True,
        ),
        RichParameter(
            name="choice",
            type="string",
            description="The choice letter to make",
            required=True,
            enum=["A", "B", "C"]
        ),
    ]
    examples = [
        "story_continue(story_id='story_20260209_143022', choice='A')",
    ]
    
    def execute(self, story_id: str, choice: str, **kwargs) -> dict[str, Any]:
        try:
            manager = StoryManager()
            story = manager.get_story(story_id)
            
            if not story:
                return {"success": False, "error": f"Story '{story_id}' not found"}
            
            # Generate continuation based on choice
            continuation = self._generate_continuation(story, choice)
            
            # Add new chapter
            new_chapter = {
                "number": story["current_chapter"] + 1,
                "text": continuation["text"],
                "choices": continuation["choices"],
                "previous_choice": choice,
            }
            
            story["chapters"].append(new_chapter)
            story["current_chapter"] += 1
            manager.save_story(story)
            
            return {
                "success": True,
                "story_id": story_id,
                "chapter": new_chapter["number"],
                "text": continuation["text"],
                "choices": continuation["choices"],
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_continuation(self, story: dict, choice: str) -> dict:
        """Generate story continuation based on choice using AI."""
        protagonist = story.get('protagonist', 'the hero')
        setting = story.get('setting', 'a mysterious land')
        
        # Try AI-generated continuation
        try:
            from ..core.inference import EnigmaEngine
            engine = EnigmaEngine.get_instance()
            
            if engine and engine.model:
                # Get last chapter for context
                last_text = ""
                if story.get("chapters"):
                    last_text = story["chapters"][-1].get("text", "")[:200]
                
                prompt = f"""Continue this interactive story:
Setting: {setting}
Protagonist: {protagonist}
Last scene: {last_text}
Player chose: "{choice}"

Write a SHORT continuation (2-3 sentences) and provide 3 new choices.

Format:
STORY: [continuation]
CHOICE A: [option]
CHOICE B: [option]  
CHOICE C: [option]"""
                
                response = engine.generate(prompt, max_gen=200, temperature=0.9)
                
                # Parse response
                if response and "STORY:" in response:
                    lines = response.split("\n")
                    story_text = ""
                    choices = []
                    
                    for line in lines:
                        if line.startswith("STORY:"):
                            story_text = line.replace("STORY:", "").strip()
                        elif line.startswith("CHOICE"):
                            choice_text = line.split(":", 1)[-1].strip()
                            choice_id = line[7] if len(line) > 7 else chr(65 + len(choices))
                            choices.append({"id": choice_id, "text": choice_text})
                    
                    if story_text and len(choices) >= 2:
                        return {"text": story_text, "choices": choices[:3]}
        except Exception:
            pass
        
        # Fallback: Simple template
        return {
            "text": f"{protagonist} considered the choice carefully. The decision would shape everything that followed.",
            "choices": [
                {"id": "A", "text": "Continue forward cautiously"},
                {"id": "B", "text": "Look for another way"},
                {"id": "C", "text": "Rest and reconsider"},
            ],
        }


# ============================================================================
# D&D TOOLS
# ============================================================================

class DnDRollTool(Tool):
    """Roll dice for D&D with optional animation."""
    
    name = "dnd_roll"
    description = "Roll dice using D&D notation (e.g., '2d6+3', '1d20', '4d6 drop lowest'). Can generate animated GIF!"
    parameters = {
        "dice": "Dice notation (e.g., '2d6', '1d20+5', '4d6kh3' for keep highest 3)",
        "reason": "Optional reason for the roll",
        "animate": "Generate animated dice roll GIF (default: True)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="dice",
            type="string",
            description="Dice notation (NdS, NdS+M, NdSkhX for keep highest)",
            required=True,
        ),
        RichParameter(
            name="reason",
            type="string",
            description="Reason for the roll (e.g., 'Attack roll', 'Perception check')",
            required=False,
        ),
        RichParameter(
            name="animate",
            type="boolean",
            description="Generate animated dice roll GIF",
            required=False,
            default=True,
        ),
    ]
    examples = [
        "dnd_roll(dice='1d20+5', reason='Attack roll')",
        "dnd_roll(dice='2d6+3', reason='Damage')",
        "dnd_roll(dice='4d6kh3', reason='Stat roll')",
    ]
    
    def execute(self, dice: str, reason: str = "", animate: bool = True, **kwargs) -> dict[str, Any]:
        try:
            import re
            
            dice = dice.lower().strip()
            original = dice
            
            # Parse dice notation
            # Format: NdS[kh/kl X][+/-M]
            match = re.match(r'(\d+)d(\d+)(?:(kh|kl)(\d+))?([+-]\d+)?', dice)
            
            if not match:
                # Simple modifier only
                if dice.startswith('+') or dice.startswith('-'):
                    return {
                        "success": True,
                        "notation": dice,
                        "result": int(dice),
                        "reason": reason,
                    }
                return {"success": False, "error": f"Invalid dice notation: {dice}"}
            
            num_dice = int(match.group(1))
            die_size = int(match.group(2))
            keep_type = match.group(3)  # 'kh' or 'kl'
            keep_count = int(match.group(4)) if match.group(4) else None
            modifier = int(match.group(5)) if match.group(5) else 0
            
            # Roll the dice
            rolls = [random.randint(1, die_size) for _ in range(num_dice)]
            
            # Handle keep highest/lowest
            kept_rolls = rolls.copy()
            dropped = []
            if keep_type and keep_count:
                if keep_type == 'kh':
                    kept_rolls = sorted(rolls, reverse=True)[:keep_count]
                    dropped = sorted(rolls, reverse=True)[keep_count:]
                elif keep_type == 'kl':
                    kept_rolls = sorted(rolls)[:keep_count]
                    dropped = sorted(rolls)[keep_count:]
            
            total = sum(kept_rolls) + modifier
            
            # Check for criticals on d20
            crit = None
            if die_size == 20 and num_dice == 1:
                if rolls[0] == 20:
                    crit = "CRITICAL SUCCESS! 🎉"
                elif rolls[0] == 1:
                    crit = "CRITICAL FAILURE! 💀"
            
            result = {
                "success": True,
                "notation": original,
                "rolls": rolls,
                "kept": kept_rolls if keep_type else None,
                "dropped": dropped if dropped else None,
                "modifier": modifier if modifier else None,
                "total": total,
                "reason": reason,
            }
            
            if crit:
                result["critical"] = crit
            
            # Generate animated dice roll GIF
            if animate:
                try:
                    gif_path = self._generate_dice_animation(die_size, rolls, total, crit)
                    if gif_path:
                        result["animation_path"] = gif_path
                        result["type"] = "animation"
                except Exception as e:
                    result["animation_error"] = str(e)
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_dice_animation(self, die_size: int, final_rolls: list, total: int, crit: str = None) -> str:
        """Generate an animated GIF of dice rolling."""
        try:
            import time
            from pathlib import Path

            from PIL import Image, ImageDraw, ImageFont

            # Output directory
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            frames = []
            frame_count = 12  # Number of animation frames
            size = 200  # Frame size
            
            # Colors
            bg_color = (30, 30, 46)  # Dark purple-ish
            dice_color = (205, 214, 244)  # Light text
            crit_success_color = (166, 227, 161)  # Green
            crit_fail_color = (243, 139, 168)  # Red
            accent_color = (137, 180, 250)  # Blue
            
            # Die face patterns for d6
            d6_patterns = {
                1: [(0.5, 0.5)],
                2: [(0.25, 0.25), (0.75, 0.75)],
                3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
                4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
                5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
                6: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5), (0.25, 0.75), (0.75, 0.75)],
            }
            
            try:
                font = ImageFont.truetype("arial.ttf", 24)
                font_large = ImageFont.truetype("arial.ttf", 48)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except OSError:
                # Font file not found, use default
                font = ImageFont.load_default()
                font_large = font
                font_small = font
            
            # Generate rolling frames
            for frame_idx in range(frame_count):
                img = Image.new('RGB', (size, size), bg_color)
                draw = ImageDraw.Draw(img)
                
                # Draw title
                title = f"d{die_size}"
                draw.text((size//2, 15), title, fill=accent_color, font=font_small, anchor="mt")
                
                if frame_idx < frame_count - 3:
                    # Rolling animation - show random numbers
                    random_val = random.randint(1, die_size)
                    
                    if die_size == 6:
                        # Draw d6 with dots
                        dice_x, dice_y = size//2 - 40, size//2 - 40
                        dice_size = 80
                        
                        # Slight rotation effect via position offset
                        offset_x = random.randint(-5, 5)
                        offset_y = random.randint(-5, 5)
                        
                        # Draw die background
                        draw.rounded_rectangle(
                            [dice_x + offset_x, dice_y + offset_y, 
                             dice_x + dice_size + offset_x, dice_y + dice_size + offset_y],
                            radius=8, fill=(69, 71, 90), outline=dice_color, width=2
                        )
                        
                        # Draw dots
                        for px, py in d6_patterns.get(random_val, [(0.5, 0.5)]):
                            dot_x = dice_x + offset_x + int(px * dice_size)
                            dot_y = dice_y + offset_y + int(py * dice_size)
                            draw.ellipse([dot_x-6, dot_y-6, dot_x+6, dot_y+6], fill=dice_color)
                    else:
                        # Draw number for other dice
                        text = str(random_val)
                        draw.text((size//2, size//2), text, fill=dice_color, font=font_large, anchor="mm")
                    
                    # "Rolling..." text
                    dots = "." * ((frame_idx % 3) + 1)
                    draw.text((size//2, size - 25), f"Rolling{dots}", fill=(108, 112, 134), font=font_small, anchor="mt")
                    
                else:
                    # Final result frames
                    final_val = final_rolls[0] if len(final_rolls) == 1 else total
                    
                    # Choose color based on critical
                    text_color = dice_color
                    if crit and "SUCCESS" in crit:
                        text_color = crit_success_color
                    elif crit and "FAILURE" in crit:
                        text_color = crit_fail_color
                    
                    if die_size == 6 and len(final_rolls) == 1:
                        # Draw final d6
                        dice_x, dice_y = size//2 - 40, size//2 - 40
                        dice_size_px = 80
                        
                        outline = crit_success_color if crit and "SUCCESS" in crit else (crit_fail_color if crit and "FAILURE" in crit else dice_color)
                        
                        draw.rounded_rectangle(
                            [dice_x, dice_y, dice_x + dice_size_px, dice_y + dice_size_px],
                            radius=8, fill=(69, 71, 90), outline=outline, width=3
                        )
                        
                        for px, py in d6_patterns.get(final_val, [(0.5, 0.5)]):
                            dot_x = dice_x + int(px * dice_size_px)
                            dot_y = dice_y + int(py * dice_size_px)
                            draw.ellipse([dot_x-6, dot_y-6, dot_x+6, dot_y+6], fill=text_color)
                    else:
                        # Draw final number
                        draw.text((size//2, size//2), str(final_val), fill=text_color, font=font_large, anchor="mm")
                    
                    # Result text
                    if crit:
                        result_text = "NAT 20!" if "SUCCESS" in crit else "NAT 1!"
                        draw.text((size//2, size - 25), result_text, fill=text_color, font=font_small, anchor="mt")
                    else:
                        draw.text((size//2, size - 25), f"Total: {total}", fill=accent_color, font=font_small, anchor="mt")
                
                frames.append(img)
            
            # Add extra final frames to pause on result
            for _ in range(6):
                frames.append(frames[-1].copy())
            
            # Save as GIF
            timestamp = int(time.time())
            gif_path = output_dir / f"dice_roll_{timestamp}.gif"
            
            frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=frames[1:],
                duration=100,  # ms per frame
                loop=0
            )
            
            return str(gif_path)
            
        except ImportError:
            return None
        except Exception as e:
            print(f"Animation generation error: {e}")
            return None


class DnDCharacterTool(Tool):
    """Generate a D&D character."""
    
    name = "dnd_character"
    description = "Generate a random D&D character with stats, race, class, and background."
    parameters = {
        "level": "Character level (default: 1)",
        "race": "Race (optional, random if not specified)",
        "char_class": "Class (optional, random if not specified)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="level",
            type="integer",
            description="Character level",
            required=False,
            default=1,
            min_value=1,
            max_value=20,
        ),
        RichParameter(
            name="race",
            type="string",
            description="Character race",
            required=False,
            enum=["Human", "Elf", "Dwarf", "Halfling", "Dragonborn", "Gnome", "Half-Elf", "Half-Orc", "Tiefling"]
        ),
        RichParameter(
            name="char_class",
            type="string",
            description="Character class",
            required=False,
            enum=["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"]
        ),
    ]
    examples = [
        "dnd_character() - Random level 1 character",
        "dnd_character(level=5, race='Elf', char_class='Wizard') - Level 5 Elf Wizard",
    ]
    
    RACES = ["Human", "Elf", "Dwarf", "Halfling", "Dragonborn", "Gnome", "Half-Elf", "Half-Orc", "Tiefling"]
    CLASSES = ["Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard"]
    BACKGROUNDS = ["Acolyte", "Criminal", "Folk Hero", "Noble", "Sage", "Soldier", "Entertainer", "Outlander", "Hermit", "Guild Artisan"]
    
    NAMES = {
        "Human": ["Marcus", "Elena", "Thomas", "Sarah", "William", "Catherine"],
        "Elf": ["Aelindra", "Thalion", "Caelynn", "Eryn", "Galinndan", "Sariel"],
        "Dwarf": ["Thorin", "Brunhilde", "Grimnar", "Helga", "Durin", "Dagni"],
        "default": ["Zephyr", "Shadow", "Storm", "Raven", "Phoenix", "Wolf"],
    }
    
    def execute(self, level: int = 1, race: str = None, char_class: str = None, **kwargs) -> dict[str, Any]:
        try:
            # Select race and class
            if not race:
                race = random.choice(self.RACES)
            if not char_class:
                char_class = random.choice(self.CLASSES)
            
            # Roll stats (4d6 drop lowest)
            def roll_stat():
                rolls = [random.randint(1, 6) for _ in range(4)]
                return sum(sorted(rolls)[1:])  # Drop lowest
            
            stats = {
                "STR": roll_stat(),
                "DEX": roll_stat(),
                "CON": roll_stat(),
                "INT": roll_stat(),
                "WIS": roll_stat(),
                "CHA": roll_stat(),
            }
            
            # Generate name
            name_pool = self.NAMES.get(race, self.NAMES["default"])
            name = random.choice(name_pool)
            
            # Background
            background = random.choice(self.BACKGROUNDS)
            
            # Hit points (simplified)
            hit_dice = {
                "Barbarian": 12, "Fighter": 10, "Paladin": 10, "Ranger": 10,
                "Bard": 8, "Cleric": 8, "Druid": 8, "Monk": 8, "Rogue": 8, "Warlock": 8,
                "Sorcerer": 6, "Wizard": 6,
            }
            hd = hit_dice.get(char_class, 8)
            con_mod = (stats["CON"] - 10) // 2
            hp = hd + con_mod + ((int(level) - 1) * (hd // 2 + 1 + con_mod))
            
            character = {
                "name": name,
                "race": race,
                "class": char_class,
                "level": int(level),
                "background": background,
                "stats": stats,
                "hit_points": max(1, hp),
                "armor_class": 10 + (stats["DEX"] - 10) // 2,
                "proficiency_bonus": 2 + ((int(level) - 1) // 4),
            }
            
            return {
                "success": True,
                "character": character,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class DnDEncounterTool(Tool):
    """Generate a D&D encounter."""
    
    name = "dnd_encounter"
    description = "Generate a random D&D combat or roleplay encounter."
    parameters = {
        "difficulty": "Difficulty: 'easy', 'medium', 'hard', 'deadly' (default: medium)",
        "environment": "Environment: 'forest', 'dungeon', 'city', 'mountains', 'swamp' (default: random)",
        "party_size": "Number of players (default: 4)",
    }
    category = "gaming"
    rich_parameters = [
        RichParameter(
            name="difficulty",
            type="string",
            description="Encounter difficulty",
            required=False,
            default="medium",
            enum=["easy", "medium", "hard", "deadly"]
        ),
        RichParameter(
            name="environment",
            type="string",
            description="Encounter environment",
            required=False,
            enum=["forest", "dungeon", "city", "mountains", "swamp"]
        ),
        RichParameter(
            name="party_size",
            type="integer",
            description="Number of players",
            required=False,
            default=4,
            min_value=1,
            max_value=10,
        ),
    ]
    examples = [
        "dnd_encounter() - Random medium encounter",
        "dnd_encounter(difficulty='hard', environment='dungeon') - Hard dungeon encounter",
    ]
    
    ENCOUNTERS = {
        "forest": {
            "combat": ["A pack of wolves surrounds you", "Goblins ambush from the trees", "A giant spider drops from above", "Bandits block the path"],
            "roleplay": ["A wounded traveler asks for help", "A mysterious hermit offers cryptic advice", "Fey creatures play tricks on you"],
        },
        "dungeon": {
            "combat": ["Skeletons rise from the darkness", "A gelatinous cube slides toward you", "Kobolds swarm from side tunnels", "A mimic disguised as a chest"],
            "roleplay": ["A prisoner begs for release", "Ancient writings on the wall reveal secrets", "A ghostly apparition appears"],
        },
        "city": {
            "combat": ["Thieves attempt to mug you", "A bar fight breaks out", "Assassins strike from the shadows", "City guards mistake you for criminals"],
            "roleplay": ["A merchant offers a suspicious deal", "A noble requests your aid", "A street performer has information"],
        },
        "mountains": {
            "combat": ["Harpies attack from cliff faces", "An ogre guards a bridge", "Orcs raid from a mountain stronghold", "A young dragon claims territory"],
            "roleplay": ["A dwarven prospector seeks partners", "Mountain monks test your worth", "An avalanche blocks your path"],
        },
        "swamp": {
            "combat": ["Lizardfolk emerge from the murk", "A will-o'-wisp leads you astray", "Crocodiles lurk beneath the surface", "A hydra rises from the bog"],
            "roleplay": ["A hag offers a bargain", "Lost travelers need guidance", "Ancient ruins sink into the mire"],
        },
    }
    
    def execute(self, difficulty: str = "medium", environment: str = None, 
                party_size: int = 4, **kwargs) -> dict[str, Any]:
        try:
            if not environment:
                environment = random.choice(list(self.ENCOUNTERS.keys()))
            
            if environment not in self.ENCOUNTERS:
                environment = "forest"
            
            env_encounters = self.ENCOUNTERS[environment]
            
            # Choose combat or roleplay
            encounter_type = random.choice(["combat", "roleplay"])
            encounter = random.choice(env_encounters[encounter_type])
            
            result = {
                "success": True,
                "environment": environment,
                "type": encounter_type,
                "difficulty": difficulty,
                "encounter": encounter,
                "party_size": party_size,
            }
            
            if encounter_type == "combat":
                # Add initiative roll suggestion
                result["tip"] = "Roll initiative! (1d20 + DEX modifier)"
                
                # Add monster count based on difficulty
                monster_counts = {"easy": "1-2", "medium": "2-4", "hard": "3-5", "deadly": "4-6"}
                result["suggested_enemies"] = monster_counts.get(difficulty, "2-4")
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
