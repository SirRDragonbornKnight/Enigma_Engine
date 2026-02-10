"""
Prompt Library - Create, manage, and share prompt templates

Features:
- Create and organize prompts
- Variables and templates
- Categories and tags
- Import/export prompts
- Community sharing
"""

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class PromptCategory(Enum):
    """Prompt categories"""
    GENERAL = "general"
    CODING = "coding"
    WRITING = "writing"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    BUSINESS = "business"
    EDUCATION = "education"
    ROLEPLAY = "roleplay"
    IMAGE = "image"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    OTHER = "other"


@dataclass
class PromptVariable:
    """Variable in a prompt template"""
    name: str
    description: str = ""
    default: str = ""
    required: bool = True
    options: list[str] = field(default_factory=list)  # For dropdowns


@dataclass
class PromptTemplate:
    """A prompt template with variables"""
    id: str
    title: str
    content: str
    description: str = ""
    category: str = PromptCategory.GENERAL.value
    tags: list[str] = field(default_factory=list)
    variables: list[PromptVariable] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    
    # Community
    public: bool = False
    downloads: int = 0
    likes: int = 0
    
    # Examples
    example_inputs: dict[str, str] = field(default_factory=dict)
    example_output: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        
        # Extract variables from content
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> list[PromptVariable]:
        """Extract {{variable}} patterns from content"""
        pattern = r'\{\{(\w+)(?::([^}]+))?\}\}'
        matches = re.findall(pattern, self.content)
        
        variables = []
        seen = set()
        for name, description in matches:
            if name not in seen:
                seen.add(name)
                variables.append(PromptVariable(
                    name=name,
                    description=description or f"Value for {name}"
                ))
        return variables
    
    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Rendered prompt string
        """
        result = self.content
        
        # Replace variables
        for var in self.variables:
            value = kwargs.get(var.name, var.default)
            if var.required and not value:
                raise ValueError(f"Required variable '{var.name}' not provided")
            
            # Replace both {{var}} and {{var:description}} formats
            # Use lambda to avoid backslash interpretation in replacement
            pattern = r'\{\{' + var.name + r'(?::[^}]+)?\}\}'
            result = re.sub(pattern, lambda m: str(value), result)
        
        return result
    
    def get_variable_names(self) -> list[str]:
        """Get list of variable names"""
        return [v.name for v in self.variables]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['variables'] = [asdict(v) for v in self.variables]
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary"""
        variables = [
            PromptVariable(**v) for v in data.pop('variables', [])
        ]
        return cls(variables=variables, **data)


@dataclass 
class PromptCollection:
    """A collection of related prompts"""
    id: str
    name: str
    description: str = ""
    prompts: list[str] = field(default_factory=list)  # Prompt IDs
    author: str = ""
    public: bool = False
    created_at: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class PromptLibrary:
    """
    Local prompt library manager
    
    Stores and manages prompt templates.
    """
    
    def __init__(self, library_dir: Union[str, Path] = None):
        self.library_dir = Path(library_dir) if library_dir else \
            Path.home() / ".enigma_engine" / "prompts"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        self._prompts: dict[str, PromptTemplate] = {}
        self._collections: dict[str, PromptCollection] = {}
        self._load_library()
    
    def _load_library(self) -> None:
        """Load prompts from disk"""
        prompts_file = self.library_dir / "prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file, encoding='utf-8') as f:
                    data = json.load(f)
                
                for prompt_data in data.get('prompts', []):
                    prompt = PromptTemplate.from_dict(prompt_data)
                    self._prompts[prompt.id] = prompt
                
                for coll_data in data.get('collections', []):
                    coll = PromptCollection(**coll_data)
                    self._collections[coll.id] = coll
                    
            except Exception as e:
                logger.error(f"Failed to load prompt library: {e}")
    
    def _save_library(self) -> None:
        """Save prompts to disk"""
        data = {
            'prompts': [p.to_dict() for p in self._prompts.values()],
            'collections': [asdict(c) for c in self._collections.values()]
        }
        
        prompts_file = self.library_dir / "prompts.json"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    # ==================== Prompt CRUD ====================
    
    def add_prompt(self, prompt: PromptTemplate) -> str:
        """Add a prompt to the library"""
        self._prompts[prompt.id] = prompt
        self._save_library()
        return prompt.id
    
    def create_prompt(self, 
                     title: str,
                     content: str,
                     description: str = "",
                     category: str = "general",
                     tags: list[str] = None,
                     **kwargs) -> PromptTemplate:
        """Create and add a new prompt"""
        prompt = PromptTemplate(
            id="",
            title=title,
            content=content,
            description=description,
            category=category,
            tags=tags or [],
            **kwargs
        )
        self.add_prompt(prompt)
        return prompt
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Get a prompt by ID"""
        return self._prompts.get(prompt_id)
    
    def update_prompt(self, prompt_id: str, **updates) -> Optional[PromptTemplate]:
        """Update a prompt"""
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            return None
        
        for key, value in updates.items():
            if hasattr(prompt, key):
                setattr(prompt, key, value)
        
        prompt.updated_at = datetime.now().isoformat()
        
        # Re-extract variables if content changed
        if 'content' in updates:
            prompt.variables = prompt._extract_variables()
        
        self._save_library()
        return prompt
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt"""
        if prompt_id in self._prompts:
            del self._prompts[prompt_id]
            
            # Remove from collections
            for coll in self._collections.values():
                if prompt_id in coll.prompts:
                    coll.prompts.remove(prompt_id)
            
            self._save_library()
            return True
        return False
    
    def list_prompts(self, 
                    category: str = None,
                    tags: list[str] = None,
                    search: str = None) -> list[PromptTemplate]:
        """List prompts with optional filtering"""
        prompts = list(self._prompts.values())
        
        if category:
            prompts = [p for p in prompts if p.category == category]
        
        if tags:
            prompts = [p for p in prompts 
                      if any(t in p.tags for t in tags)]
        
        if search:
            search_lower = search.lower()
            prompts = [p for p in prompts 
                      if search_lower in p.title.lower() 
                      or search_lower in p.description.lower()
                      or search_lower in p.content.lower()]
        
        return prompts
    
    # ==================== Collections ====================
    
    def create_collection(self, name: str, 
                         description: str = "",
                         prompt_ids: list[str] = None) -> PromptCollection:
        """Create a prompt collection"""
        collection = PromptCollection(
            id="",
            name=name,
            description=description,
            prompts=prompt_ids or []
        )
        self._collections[collection.id] = collection
        self._save_library()
        return collection
    
    def get_collection(self, collection_id: str) -> Optional[PromptCollection]:
        """Get a collection by ID"""
        return self._collections.get(collection_id)
    
    def add_to_collection(self, collection_id: str, prompt_id: str) -> bool:
        """Add a prompt to a collection"""
        collection = self._collections.get(collection_id)
        if collection and prompt_id in self._prompts:
            if prompt_id not in collection.prompts:
                collection.prompts.append(prompt_id)
                self._save_library()
            return True
        return False
    
    def list_collections(self) -> list[PromptCollection]:
        """List all collections"""
        return list(self._collections.values())
    
    # ==================== Import/Export ====================
    
    def export_prompt(self, prompt_id: str, 
                     file_path: Union[str, Path]) -> bool:
        """Export a single prompt to file"""
        prompt = self._prompts.get(prompt_id)
        if not prompt:
            return False
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(prompt.to_dict(), f, indent=2)
        return True
    
    def export_collection(self, collection_id: str,
                         file_path: Union[str, Path]) -> bool:
        """Export a collection to file"""
        collection = self._collections.get(collection_id)
        if not collection:
            return False
        
        prompts = [self._prompts[pid].to_dict() 
                  for pid in collection.prompts 
                  if pid in self._prompts]
        
        data = {
            'collection': asdict(collection),
            'prompts': prompts
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    
    def import_prompt(self, file_path: Union[str, Path]) -> Optional[PromptTemplate]:
        """Import a prompt from file"""
        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
            
            # Generate new ID to avoid conflicts
            data['id'] = ""
            prompt = PromptTemplate.from_dict(data)
            self.add_prompt(prompt)
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to import prompt: {e}")
            return None
    
    def import_collection(self, file_path: Union[str, Path]) -> Optional[PromptCollection]:
        """Import a collection from file"""
        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
            
            # Import prompts
            new_prompt_ids = []
            for prompt_data in data.get('prompts', []):
                prompt_data['id'] = ""  # Generate new ID
                prompt = PromptTemplate.from_dict(prompt_data)
                self.add_prompt(prompt)
                new_prompt_ids.append(prompt.id)
            
            # Create collection
            coll_data = data.get('collection', {})
            coll_data['id'] = ""  # Generate new ID
            coll_data['prompts'] = new_prompt_ids
            
            collection = PromptCollection(**coll_data)
            self._collections[collection.id] = collection
            self._save_library()
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to import collection: {e}")
            return None


class PromptHubClient:
    """
    Client for community prompt sharing hub
    """
    
    DEFAULT_HUB_URL = "https://prompts.Enigma AI Engine.dev/api"  # Placeholder
    
    def __init__(self, hub_url: str = None, api_key: str = None):
        self.hub_url = hub_url or self.DEFAULT_HUB_URL
        self.api_key = api_key
        self._session = None
    
    async def _get_session(self):
        """Get aiohttp session"""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for hub operations")
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def search_prompts(self,
                            query: str = "",
                            category: str = None,
                            tags: list[str] = None,
                            sort_by: str = "downloads",
                            limit: int = 20) -> list[dict]:
        """Search community prompts"""
        params = {
            "q": query,
            "limit": limit,
            "sort": sort_by
        }
        if category:
            params["category"] = category
        if tags:
            params["tags"] = ",".join(tags)
        
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.hub_url}/prompts",
                params=params,
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def get_prompt(self, prompt_id: str) -> Optional[dict]:
        """Get a prompt from the hub"""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.hub_url}/prompts/{prompt_id}",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Failed to get prompt: {e}")
            return None
    
    async def download_prompt(self, prompt_id: str, 
                             library: PromptLibrary) -> Optional[PromptTemplate]:
        """Download and add a prompt to local library"""
        data = await self.get_prompt(prompt_id)
        if not data:
            return None
        
        # Don't copy hub metadata
        data.pop('downloads', None)
        data.pop('likes', None)
        data['id'] = ""  # Generate new local ID
        
        prompt = PromptTemplate.from_dict(data)
        library.add_prompt(prompt)
        return prompt
    
    async def publish_prompt(self, prompt: PromptTemplate) -> Optional[str]:
        """Publish a prompt to the hub"""
        if not self.api_key:
            logger.error("API key required for publishing")
            return None
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.hub_url}/prompts",
                json=prompt.to_dict(),
                headers=self._get_headers()
            ) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    return data.get("id")
                return None
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return None
    
    async def like_prompt(self, prompt_id: str) -> bool:
        """Like a prompt"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.hub_url}/prompts/{prompt_id}/like",
                headers=self._get_headers()
            ) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def get_trending(self, limit: int = 10) -> list[dict]:
        """Get trending prompts"""
        return await self.search_prompts(sort_by="trending", limit=limit)
    
    async def get_recent(self, limit: int = 10) -> list[dict]:
        """Get recently published prompts"""
        return await self.search_prompts(sort_by="created_at", limit=limit)
    
    async def get_categories(self) -> list[str]:
        """Get available categories"""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.hub_url}/categories",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                return [c.value for c in PromptCategory]
        except Exception:
            return [c.value for c in PromptCategory]


# Singleton accessor
_library: Optional[PromptLibrary] = None


def get_prompt_library(library_dir: str = None) -> PromptLibrary:
    """Get the prompt library singleton"""
    global _library
    if _library is None:
        _library = PromptLibrary(library_dir)
    return _library


def get_prompt_hub(api_key: str = None) -> PromptHubClient:
    """Get a prompt hub client"""
    return PromptHubClient(api_key=api_key)


# ==================== Built-in Prompts ====================

BUILTIN_PROMPTS = [
    PromptTemplate(
        id="code-explain",
        title="Explain Code",
        content="""Explain the following {{language:Programming language}} code in detail:

```{{language}}
{{code:The code to explain}}
```

Provide:
1. A high-level overview
2. Step-by-step explanation
3. Any potential issues or improvements""",
        description="Get a detailed explanation of any code",
        category=PromptCategory.CODING.value,
        tags=["code", "explanation", "learning"]
    ),
    
    PromptTemplate(
        id="code-refactor",
        title="Refactor Code",
        content="""Refactor the following {{language:Programming language}} code to be more {{goal:Refactoring goal (readable, efficient, maintainable)}}:

```{{language}}
{{code:The code to refactor}}
```

Explain your changes and provide the refactored version.""",
        description="Improve code quality with AI suggestions",
        category=PromptCategory.CODING.value,
        tags=["code", "refactor", "quality"]
    ),
    
    PromptTemplate(
        id="summarize-text",
        title="Summarize Text",
        content="""Summarize the following text in {{length:Summary length (brief, detailed)}} format:

{{text:The text to summarize}}

Key points to capture:
- Main ideas
- Important details
- Conclusions""",
        description="Create concise summaries of any text",
        category=PromptCategory.SUMMARIZATION.value,
        tags=["summary", "text", "writing"]
    ),
    
    PromptTemplate(
        id="translate",
        title="Translate Text",
        content="""Translate the following text from {{source:Source language}} to {{target:Target language}}:

{{text:The text to translate}}

Maintain the original tone and style.""",
        description="Translate text between languages",
        category=PromptCategory.TRANSLATION.value,
        tags=["translation", "language"]
    ),
    
    PromptTemplate(
        id="creative-story",
        title="Creative Story",
        content="""Write a {{genre:Genre (fantasy, sci-fi, mystery, etc.)}} story with the following elements:
        
Setting: {{setting:Where and when the story takes place}}
Main character: {{character:Description of the protagonist}}
Theme: {{theme:Central theme or message}}

Make it engaging and {{length:Length (short, medium, long)}}.""",
        description="Generate creative fiction stories",
        category=PromptCategory.CREATIVE.value,
        tags=["creative", "story", "writing"]
    ),
    
    PromptTemplate(
        id="email-writer",
        title="Professional Email",
        content="""Write a professional {{tone:Tone (formal, friendly, urgent)}} email for the following situation:

Purpose: {{purpose:What is the email for}}
Recipient: {{recipient:Who is receiving the email}}
Key points: {{points:Main points to include}}

Keep it {{length:Length (brief, detailed)}} and professional.""",
        description="Write professional emails quickly",
        category=PromptCategory.BUSINESS.value,
        tags=["email", "business", "writing"]
    ),
    
    PromptTemplate(
        id="image-prompt",
        title="Image Generation Prompt",
        content="""Create a detailed image generation prompt for:

Subject: {{subject:Main subject of the image}}
Style: {{style:Art style (realistic, anime, oil painting, etc.)}}
Mood: {{mood:Emotional atmosphere}}
Additional details: {{details:Extra details to include}}

Format it as a single, detailed prompt optimized for image generation.""",
        description="Create optimized prompts for image generation",
        category=PromptCategory.IMAGE.value,
        tags=["image", "prompt", "creative"]
    ),
    
    PromptTemplate(
        id="code-review",
        title="Code Review",
        content="""Review the following {{language:Programming language}} code:

```{{language}}
{{code:The code to review}}
```

Provide feedback on:
1. Code quality and readability
2. Potential bugs or issues
3. Performance considerations
4. Best practices
5. Suggestions for improvement""",
        description="Get AI-powered code reviews",
        category=PromptCategory.CODING.value,
        tags=["code", "review", "quality"]
    ),
]


def install_builtin_prompts(library: PromptLibrary = None) -> int:
    """Install built-in prompts to library"""
    library = library or get_prompt_library()
    count = 0
    
    for prompt in BUILTIN_PROMPTS:
        if not library.get_prompt(prompt.id):
            library.add_prompt(prompt)
            count += 1
    
    return count
