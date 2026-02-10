"""
================================================================================
System Prompt Library - Save, load, and manage system prompts.
================================================================================

Provides a library for managing reusable system prompts:
- Save custom system prompts with metadata
- Organize prompts by category
- Search prompts by keyword
- Share prompts (export/import)
- Built-in prompt templates

USAGE:
    from enigma_engine.memory.prompt_library import PromptLibrary, get_prompt_library
    
    library = get_prompt_library()
    
    # Save a custom prompt
    library.save_prompt(
        name="Expert Coder",
        content="You are an expert programmer...",
        category="coding",
        tags=["python", "debugging"]
    )
    
    # Load a prompt
    prompt = library.get_prompt("Expert Coder")
    
    # Search prompts
    results = library.search("python coding")
    
    # List by category
    coding_prompts = library.list_by_category("coding")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Built-in prompt categories."""
    
    GENERAL = "general"
    CODING = "coding"
    WRITING = "writing"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    EDUCATION = "education"
    BUSINESS = "business"
    ROLEPLAY = "roleplay"
    TECHNICAL = "technical"
    RESEARCH = "research"
    CUSTOM = "custom"


@dataclass
class SystemPrompt:
    """A reusable system prompt."""
    
    name: str
    content: str
    category: str = "general"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    author: str = ""
    version: str = "1.0"
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    usage_count: int = 0
    favorite: bool = False
    
    # Configuration hints
    suggested_temperature: float | None = None
    suggested_max_tokens: int | None = None
    model_hint: str | None = None  # Suggested model type
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemPrompt:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def matches_search(self, query: str) -> bool:
        """Check if prompt matches search query."""
        query_lower = query.lower()
        searchable = f"{self.name} {self.description} {self.content} {' '.join(self.tags)}".lower()
        
        # Check for all query words
        words = query_lower.split()
        return all(word in searchable for word in words)


# Built-in prompts
BUILTIN_PROMPTS: list[SystemPrompt] = [
    SystemPrompt(
        name="Helpful Assistant",
        content="""You are a helpful, harmless, and honest AI assistant. You aim to provide accurate, thoughtful responses while being transparent about your limitations. If you're unsure about something, you say so. If asked to do something harmful or unethical, you politely decline and explain why.""",
        category="general",
        description="A balanced, friendly assistant for general use",
        tags=["assistant", "helpful", "general"],
        author="Enigma AI Engine",
        suggested_temperature=0.7
    ),
    SystemPrompt(
        name="Expert Programmer",
        content="""You are an expert programmer with deep knowledge of software engineering, algorithms, and best practices. You write clean, efficient, well-documented code. When explaining code, you focus on clarity and teach the underlying concepts. You follow language-specific conventions and suggest modern approaches. You also consider security, performance, and maintainability in your solutions.""",
        category="coding",
        description="Expert coding assistant for programming tasks",
        tags=["coding", "programming", "software", "debugging"],
        author="Enigma AI Engine",
        suggested_temperature=0.3,
        model_hint="code"
    ),
    SystemPrompt(
        name="Creative Writer",
        content="""You are a creative writing assistant with expertise in storytelling, prose, poetry, and various writing styles. You help develop compelling narratives, vivid descriptions, and engaging dialogue. You can adapt to different genres, tones, and voices while maintaining consistency. You offer constructive suggestions to improve writing while respecting the author's vision.""",
        category="creative",
        description="Creative writing partner for stories and prose",
        tags=["writing", "creative", "storytelling", "fiction"],
        author="Enigma AI Engine",
        suggested_temperature=0.9
    ),
    SystemPrompt(
        name="Technical Explainer",
        content="""You are a technical educator who excels at explaining complex concepts in clear, accessible language. You use analogies, examples, and step-by-step breakdowns to help understanding. You adapt your explanations to the audience's level - from complete beginners to experts. You verify understanding and welcome follow-up questions.""",
        category="education",
        description="Explains complex technical concepts clearly",
        tags=["education", "technical", "learning", "explanation"],
        author="Enigma AI Engine",
        suggested_temperature=0.5
    ),
    SystemPrompt(
        name="Data Analyst",
        content="""You are a data analysis expert proficient in statistics, data visualization, and deriving insights from data. You help with exploratory data analysis, statistical testing, machine learning approaches, and communicating findings effectively. You are familiar with tools like pandas, numpy, SQL, and various visualization libraries.""",
        category="analysis",
        description="Data analysis and statistics expert",
        tags=["data", "analysis", "statistics", "visualization"],
        author="Enigma AI Engine",
        suggested_temperature=0.4
    ),
    SystemPrompt(
        name="Academic Researcher",
        content="""You are a research assistant with expertise in academic writing, literature review, and research methodology. You help formulate research questions, find relevant sources, synthesize information, and write in academic style. You always emphasize proper citation and academic integrity. You can help with any field of study.""",
        category="research",
        description="Academic research and writing assistant",
        tags=["research", "academic", "writing", "citations"],
        author="Enigma AI Engine",
        suggested_temperature=0.5
    ),
    SystemPrompt(
        name="Business Consultant",
        content="""You are a business consultant with expertise in strategy, operations, marketing, and finance. You help analyze business problems, develop strategies, create business plans, and make data-driven recommendations. You understand both startup dynamics and enterprise operations. You communicate in clear, professional business language.""",
        category="business",
        description="Business strategy and consulting assistant",
        tags=["business", "strategy", "consulting", "management"],
        author="Enigma AI Engine",
        suggested_temperature=0.6
    ),
    SystemPrompt(
        name="Socratic Teacher",
        content="""You are a Socratic teacher who guides learning through thoughtful questions rather than direct answers. You help students discover knowledge themselves by asking probing questions, revealing assumptions, and encouraging deeper thinking. You celebrate curiosity and never make students feel bad for not knowing something.""",
        category="education",
        description="Teaches through questioning and dialogue",
        tags=["education", "socratic", "teaching", "questions"],
        author="Enigma AI Engine",
        suggested_temperature=0.7
    ),
    SystemPrompt(
        name="Code Reviewer",
        content="""You are an expert code reviewer who provides thorough, constructive feedback on code. You check for bugs, security issues, performance problems, and style inconsistencies. You explain the reasoning behind your suggestions and offer improved alternatives. You balance being thorough with being respectful and encouraging.""",
        category="coding",
        description="Thorough code review assistant",
        tags=["code review", "bugs", "security", "best practices"],
        author="Enigma AI Engine",
        suggested_temperature=0.3
    ),
    SystemPrompt(
        name="Brainstorming Partner",
        content="""You are a creative brainstorming partner who helps generate and develop ideas. You think divergently, make unexpected connections, and build on ideas enthusiastically. You use techniques like mind mapping, SCAMPER, and random stimulation. You help evaluate ideas constructively without killing creativity too early.""",
        category="creative",
        description="Creative brainstorming and ideation partner",
        tags=["brainstorming", "ideas", "creative thinking", "innovation"],
        author="Enigma AI Engine",
        suggested_temperature=0.95
    ),
    SystemPrompt(
        name="Debugging Detective",
        content="""You are a debugging expert who systematically tracks down bugs and errors. You ask clarifying questions, form hypotheses, and suggest diagnostic steps. You're familiar with common bugs, error patterns, and debugging tools across languages. You explain the root cause and how to prevent similar issues.""",
        category="coding",
        description="Systematic bug hunting assistant",
        tags=["debugging", "bugs", "errors", "troubleshooting"],
        author="Enigma AI Engine",
        suggested_temperature=0.2
    ),
    SystemPrompt(
        name="Documentation Writer",
        content="""You are a technical documentation expert who writes clear, comprehensive documentation. You create READMEs, API docs, tutorials, and reference guides. You organize information logically, use consistent formatting, and include helpful examples. You adapt your writing style to the audience - developers, end-users, or stakeholders.""",
        category="technical",
        description="Technical documentation specialist",
        tags=["documentation", "technical writing", "api docs", "tutorials"],
        author="Enigma AI Engine",
        suggested_temperature=0.4
    ),
]


class PromptLibrary:
    """
    Library for managing system prompts.
    
    Stores prompts in JSON format with full CRUD operations,
    search, and organization features.
    """
    
    def __init__(self, library_path: Path | None = None):
        """
        Initialize the prompt library.
        
        Args:
            library_path: Path to store prompts. Default: data/prompts/
        """
        self._library_path = library_path or Path("data/prompts")
        self._library_path.mkdir(parents=True, exist_ok=True)
        
        self._prompts_file = self._library_path / "prompts.json"
        self._prompts: dict[str, SystemPrompt] = {}
        
        self._load_library()
    
    def _load_library(self) -> None:
        """Load prompts from disk."""
        # Load built-in prompts
        for prompt in BUILTIN_PROMPTS:
            key = self._normalize_name(prompt.name)
            if key not in self._prompts:
                self._prompts[key] = prompt
        
        # Load custom prompts
        if self._prompts_file.exists():
            try:
                with open(self._prompts_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("prompts", []):
                        prompt = SystemPrompt.from_dict(item)
                        key = self._normalize_name(prompt.name)
                        self._prompts[key] = prompt
                logger.info(f"Loaded {len(self._prompts)} prompts from library")
            except Exception as e:
                logger.error(f"Failed to load prompt library: {e}")
    
    def _save_library(self) -> None:
        """Save custom prompts to disk."""
        try:
            # Only save non-builtin prompts
            builtin_names = {self._normalize_name(p.name) for p in BUILTIN_PROMPTS}
            custom_prompts = [
                p.to_dict() for key, p in self._prompts.items()
                if key not in builtin_names
            ]
            
            data = {
                "version": "1.0",
                "prompts": custom_prompts
            }
            
            with open(self._prompts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save prompt library: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize prompt name for lookup."""
        return name.lower().strip()
    
    def save_prompt(
        self,
        name: str,
        content: str,
        category: str = "custom",
        description: str = "",
        tags: list[str] | None = None,
        **kwargs
    ) -> SystemPrompt:
        """
        Save a new prompt or update existing.
        
        Args:
            name: Unique prompt name
            content: The system prompt text
            category: Category for organization
            description: Brief description
            tags: Search tags
            **kwargs: Additional SystemPrompt fields
            
        Returns:
            The saved SystemPrompt
        """
        key = self._normalize_name(name)
        
        # Check if updating existing
        existing = self._prompts.get(key)
        if existing:
            # Update
            prompt = SystemPrompt(
                name=name,
                content=content,
                category=category,
                description=description,
                tags=tags or [],
                created_at=existing.created_at,
                updated_at=datetime.now().isoformat(),
                usage_count=existing.usage_count,
                favorite=existing.favorite,
                **kwargs
            )
        else:
            # Create new
            prompt = SystemPrompt(
                name=name,
                content=content,
                category=category,
                description=description,
                tags=tags or [],
                **kwargs
            )
        
        self._prompts[key] = prompt
        self._save_library()
        
        logger.info(f"Saved prompt: {name}")
        return prompt
    
    def get_prompt(self, name: str) -> SystemPrompt | None:
        """
        Get a prompt by name.
        
        Args:
            name: Prompt name
            
        Returns:
            SystemPrompt or None if not found
        """
        key = self._normalize_name(name)
        prompt = self._prompts.get(key)
        
        if prompt:
            # Track usage
            prompt.usage_count += 1
            self._save_library()
        
        return prompt
    
    def get_prompt_content(self, name: str) -> str | None:
        """Get just the prompt content by name."""
        prompt = self.get_prompt(name)
        return prompt.content if prompt else None
    
    def delete_prompt(self, name: str) -> bool:
        """
        Delete a prompt.
        
        Args:
            name: Prompt name
            
        Returns:
            True if deleted, False if not found or built-in
        """
        key = self._normalize_name(name)
        
        # Prevent deleting built-ins
        builtin_names = {self._normalize_name(p.name) for p in BUILTIN_PROMPTS}
        if key in builtin_names:
            logger.warning(f"Cannot delete built-in prompt: {name}")
            return False
        
        if key in self._prompts:
            del self._prompts[key]
            self._save_library()
            logger.info(f"Deleted prompt: {name}")
            return True
        
        return False
    
    def list_prompts(self) -> list[SystemPrompt]:
        """Get all prompts."""
        return list(self._prompts.values())
    
    def list_by_category(self, category: str) -> list[SystemPrompt]:
        """Get prompts in a category."""
        return [p for p in self._prompts.values() if p.category == category]
    
    def list_categories(self) -> list[str]:
        """Get all categories in use."""
        return list({p.category for p in self._prompts.values()})
    
    def list_favorites(self) -> list[SystemPrompt]:
        """Get favorite prompts."""
        return [p for p in self._prompts.values() if p.favorite]
    
    def list_recent(self, limit: int = 10) -> list[SystemPrompt]:
        """Get recently used prompts."""
        sorted_prompts = sorted(
            self._prompts.values(),
            key=lambda p: p.updated_at,
            reverse=True
        )
        return sorted_prompts[:limit]
    
    def list_popular(self, limit: int = 10) -> list[SystemPrompt]:
        """Get most used prompts."""
        sorted_prompts = sorted(
            self._prompts.values(),
            key=lambda p: p.usage_count,
            reverse=True
        )
        return sorted_prompts[:limit]
    
    def search(self, query: str) -> list[SystemPrompt]:
        """
        Search prompts by keyword.
        
        Args:
            query: Search query
            
        Returns:
            Matching prompts
        """
        if not query:
            return self.list_prompts()
        
        return [p for p in self._prompts.values() if p.matches_search(query)]
    
    def search_by_tag(self, tag: str) -> list[SystemPrompt]:
        """Get prompts with a specific tag."""
        tag_lower = tag.lower()
        return [p for p in self._prompts.values() if tag_lower in [t.lower() for t in p.tags]]
    
    def list_tags(self) -> list[str]:
        """Get all tags in use."""
        tags: set[str] = set()
        for prompt in self._prompts.values():
            tags.update(prompt.tags)
        return sorted(tags)
    
    def toggle_favorite(self, name: str) -> bool:
        """Toggle favorite status of a prompt."""
        key = self._normalize_name(name)
        prompt = self._prompts.get(key)
        
        if prompt:
            prompt.favorite = not prompt.favorite
            self._save_library()
            return prompt.favorite
        
        return False
    
    def export_prompt(self, name: str) -> str | None:
        """
        Export a prompt as JSON.
        
        Args:
            name: Prompt name
            
        Returns:
            JSON string or None
        """
        prompt = self._prompts.get(self._normalize_name(name))
        if prompt:
            return json.dumps(prompt.to_dict(), indent=2)
        return None
    
    def export_all(self) -> str:
        """Export all prompts as JSON."""
        return json.dumps({
            "version": "1.0",
            "prompts": [p.to_dict() for p in self._prompts.values()]
        }, indent=2)
    
    def import_prompt(self, json_data: str) -> SystemPrompt | None:
        """
        Import a prompt from JSON.
        
        Args:
            json_data: JSON string
            
        Returns:
            Imported prompt or None
        """
        try:
            data = json.loads(json_data)
            prompt = SystemPrompt.from_dict(data)
            key = self._normalize_name(prompt.name)
            self._prompts[key] = prompt
            self._save_library()
            return prompt
        except Exception as e:
            logger.error(f"Failed to import prompt: {e}")
            return None
    
    def import_from_file(self, path: Path) -> int:
        """
        Import prompts from a JSON file.
        
        Returns:
            Number of prompts imported
        """
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            prompts = data.get("prompts", [data])  # Handle single or array
            
            for item in prompts:
                prompt = SystemPrompt.from_dict(item)
                key = self._normalize_name(prompt.name)
                self._prompts[key] = prompt
                count += 1
            
            self._save_library()
            return count
            
        except Exception as e:
            logger.error(f"Failed to import from file: {e}")
            return 0
    
    def duplicate_prompt(self, name: str, new_name: str) -> SystemPrompt | None:
        """
        Duplicate a prompt with a new name.
        
        Args:
            name: Original prompt name
            new_name: Name for the copy
            
        Returns:
            New prompt or None
        """
        original = self._prompts.get(self._normalize_name(name))
        if not original:
            return None
        
        new_prompt = SystemPrompt(
            name=new_name,
            content=original.content,
            category=original.category,
            description=f"Copy of {original.name}",
            tags=original.tags.copy(),
            author=original.author,
            suggested_temperature=original.suggested_temperature,
            suggested_max_tokens=original.suggested_max_tokens,
            model_hint=original.model_hint
        )
        
        key = self._normalize_name(new_name)
        self._prompts[key] = new_prompt
        self._save_library()
        
        return new_prompt


# Singleton instance
_library_instance: PromptLibrary | None = None


def get_prompt_library(path: Path | None = None) -> PromptLibrary:
    """Get or create the singleton library instance."""
    global _library_instance
    if _library_instance is None:
        _library_instance = PromptLibrary(path)
    return _library_instance


# Convenience functions
def get_system_prompt(name: str) -> str | None:
    """Quick access to a system prompt by name."""
    prompt = get_prompt_library().get_prompt(name)
    return prompt.content if prompt else None


def list_system_prompts() -> list[str]:
    """List all available prompt names."""
    return [p.name for p in get_prompt_library().list_prompts()]


def search_prompts(query: str) -> list[SystemPrompt]:
    """Search prompts by keyword."""
    return get_prompt_library().search(query)
