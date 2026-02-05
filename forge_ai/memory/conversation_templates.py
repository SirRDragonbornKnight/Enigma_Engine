"""
================================================================================
Conversation Templates - Pre-defined conversation starters.
================================================================================

Provides ready-to-use conversation templates:
- Pre-built templates for common scenarios
- Custom template creation
- Template variables for personalization
- Category organization

USAGE:
    from forge_ai.memory.conversation_templates import TemplateLibrary, get_template_library
    
    library = get_template_library()
    
    # Get available templates
    templates = library.list_templates()
    
    # Get a template with variables filled in
    starter = library.get_starter(
        "code_review",
        language="Python", 
        focus="security"
    )
    
    # Create custom template
    library.save_template(
        name="My Template",
        messages=[
            {"role": "user", "content": "Hello, I need help with {{topic}}"}
        ],
        variables=["topic"]
    )
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ConversationTemplate:
    """A reusable conversation starter template."""
    
    name: str
    description: str
    category: str
    messages: List[Dict[str, str]]  # List of {role, content} dicts
    variables: List[str] = field(default_factory=list)  # {{variable}} placeholders
    
    # Optional system prompt
    system_prompt: Optional[str] = None
    
    # Metadata
    icon: str = ""  # Emoji or icon code
    author: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    usage_count: int = 0
    
    # Configuration hints
    suggested_temperature: Optional[float] = None
    follow_up_suggestions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        
        # Auto-detect variables if not specified
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract {{variable}} placeholders from messages."""
        variables: Set[str] = set()
        pattern = r'\{\{(\w+)\}\}'
        
        for msg in self.messages:
            matches = re.findall(pattern, msg.get("content", ""))
            variables.update(matches)
        
        if self.system_prompt:
            matches = re.findall(pattern, self.system_prompt)
            variables.update(matches)
        
        return list(variables)
    
    def fill_variables(self, **kwargs) -> List[Dict[str, str]]:
        """
        Fill template variables and return messages.
        
        Args:
            **kwargs: Variable values (e.g., topic="Python")
            
        Returns:
            List of messages with variables filled
        """
        filled_messages = []
        
        for msg in self.messages:
            content = msg["content"]
            
            # Replace {{variable}} with values
            for var, value in kwargs.items():
                content = content.replace(f"{{{{{var}}}}}", str(value))
            
            # Remove unfilled variables (replace with empty string)
            content = re.sub(r'\{\{\w+\}\}', '', content)
            
            filled_messages.append({
                "role": msg["role"],
                "content": content
            })
        
        return filled_messages
    
    def get_system_prompt(self, **kwargs) -> Optional[str]:
        """Get system prompt with variables filled."""
        if not self.system_prompt:
            return None
        
        result = self.system_prompt
        for var, value in kwargs.items():
            result = result.replace(f"{{{{{var}}}}}", str(value))
        
        return re.sub(r'\{\{\w+\}\}', '', result)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTemplate':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Built-in templates
BUILTIN_TEMPLATES: List[ConversationTemplate] = [
    # ===== Coding =====
    ConversationTemplate(
        name="Code Review",
        description="Get a thorough code review",
        category="coding",
        icon="",
        messages=[
            {"role": "user", "content": """Please review this {{language}} code for:
- Bugs and potential issues
- Security vulnerabilities
- Performance improvements
- Code style and best practices

```{{language}}
{{code}}
```"""}
        ],
        variables=["language", "code"],
        system_prompt="You are an expert code reviewer. Provide thorough, constructive feedback.",
        suggested_temperature=0.3,
        follow_up_suggestions=[
            "Can you suggest specific improvements?",
            "What are the most critical issues?",
            "How would you refactor this?"
        ]
    ),
    ConversationTemplate(
        name="Debug Helper",
        description="Get help debugging an issue",
        category="coding",
        icon="",
        messages=[
            {"role": "user", "content": """I'm getting this error in my {{language}} code:

Error: {{error}}

Here's the relevant code:
```{{language}}
{{code}}
```

What I expected: {{expected}}
What actually happened: {{actual}}"""}
        ],
        variables=["language", "error", "code", "expected", "actual"],
        system_prompt="You are a debugging expert. Help identify the root cause and provide solutions.",
        suggested_temperature=0.2
    ),
    ConversationTemplate(
        name="Explain Code",
        description="Get an explanation of how code works",
        category="coding",
        icon="",
        messages=[
            {"role": "user", "content": """Please explain how this {{language}} code works step by step:

```{{language}}
{{code}}
```

I'm particularly interested in: {{focus}}"""}
        ],
        variables=["language", "code", "focus"],
        suggested_temperature=0.5
    ),
    ConversationTemplate(
        name="Write Tests",
        description="Generate unit tests for code",
        category="coding",
        icon="",
        messages=[
            {"role": "user", "content": """Write comprehensive unit tests for this {{language}} code:

```{{language}}
{{code}}
```

Use {{framework}} testing framework. Include edge cases and error scenarios."""}
        ],
        variables=["language", "code", "framework"],
        suggested_temperature=0.3
    ),
    
    # ===== Writing =====
    ConversationTemplate(
        name="Improve Writing",
        description="Get help improving your writing",
        category="writing",
        icon="",
        messages=[
            {"role": "user", "content": """Please improve this text for {{purpose}}. Focus on {{focus}}.

Original text:
{{text}}"""}
        ],
        variables=["purpose", "focus", "text"],
        suggested_temperature=0.7
    ),
    ConversationTemplate(
        name="Email Draft",
        description="Help draft a professional email",
        category="writing",
        icon="",
        messages=[
            {"role": "user", "content": """Help me write a {{tone}} email about {{topic}}.

Context: {{context}}
Main points to include:
{{points}}"""}
        ],
        variables=["tone", "topic", "context", "points"],
        system_prompt="You are an expert at writing clear, professional emails.",
        suggested_temperature=0.6
    ),
    ConversationTemplate(
        name="Summarize",
        description="Get a summary of text",
        category="writing",
        icon="",
        messages=[
            {"role": "user", "content": """Summarize the following in {{length}} (bullet points / one paragraph / detailed summary):

{{text}}"""}
        ],
        variables=["length", "text"],
        suggested_temperature=0.4
    ),
    
    # ===== Learning =====
    ConversationTemplate(
        name="Explain Concept",
        description="Get a clear explanation of a concept",
        category="learning",
        icon="",
        messages=[
            {"role": "user", "content": """Explain {{concept}} as if I'm a {{level}}.

Please include:
- What it is
- Why it matters
- A practical example
- Common misconceptions"""}
        ],
        variables=["concept", "level"],
        system_prompt="You are a patient, excellent teacher who explains concepts clearly.",
        suggested_temperature=0.6,
        follow_up_suggestions=[
            "Can you give me another example?",
            "How does this relate to {{related_topic}}?",
            "What should I learn next?"
        ]
    ),
    ConversationTemplate(
        name="Quiz Me",
        description="Test your knowledge on a topic",
        category="learning",
        icon="",
        messages=[
            {"role": "user", "content": """Quiz me on {{topic}}. 

Difficulty level: {{difficulty}}
Number of questions: {{count}}

After each answer, tell me if I'm right and explain why."""}
        ],
        variables=["topic", "difficulty", "count"],
        suggested_temperature=0.7
    ),
    
    # ===== Analysis =====
    ConversationTemplate(
        name="Compare Options",
        description="Get help comparing choices",
        category="analysis",
        icon="",
        messages=[
            {"role": "user", "content": """Help me compare these options for {{purpose}}:

Option A: {{option_a}}
Option B: {{option_b}}

Key criteria: {{criteria}}

Please provide a structured comparison with pros/cons."""}
        ],
        variables=["purpose", "option_a", "option_b", "criteria"],
        suggested_temperature=0.5
    ),
    ConversationTemplate(
        name="Pros and Cons",
        description="Analyze pros and cons of something",
        category="analysis",
        icon="",
        messages=[
            {"role": "user", "content": """Give me a thorough pros and cons analysis of: {{topic}}

Context: {{context}}

Consider: feasibility, cost, risks, benefits, and alternatives."""}
        ],
        variables=["topic", "context"],
        suggested_temperature=0.5
    ),
    
    # ===== Creative =====
    ConversationTemplate(
        name="Brainstorm Ideas",
        description="Generate creative ideas",
        category="creative",
        icon="",
        messages=[
            {"role": "user", "content": """Help me brainstorm ideas for {{topic}}.

Requirements/constraints: {{constraints}}

Give me {{count}} diverse, creative ideas ranging from conventional to wild."""}
        ],
        variables=["topic", "constraints", "count"],
        suggested_temperature=0.9,
        follow_up_suggestions=[
            "Can you combine ideas 2 and 5?",
            "Develop idea {{number}} further",
            "What's the most unique approach?"
        ]
    ),
    ConversationTemplate(
        name="Story Prompt",
        description="Get help with creative writing",
        category="creative",
        icon="",
        messages=[
            {"role": "user", "content": """Help me write a {{genre}} story.

Setting: {{setting}}
Main character: {{character}}
Central conflict: {{conflict}}

Start with an engaging opening scene."""}
        ],
        variables=["genre", "setting", "character", "conflict"],
        suggested_temperature=0.9
    ),
    
    # ===== Productivity =====
    ConversationTemplate(
        name="Plan Project",
        description="Help plan a project",
        category="productivity",
        icon="",
        messages=[
            {"role": "user", "content": """Help me plan this project: {{project}}

Goals: {{goals}}
Timeline: {{timeline}}
Resources available: {{resources}}

Create a structured plan with milestones and tasks."""}
        ],
        variables=["project", "goals", "timeline", "resources"],
        suggested_temperature=0.5
    ),
    ConversationTemplate(
        name="Meeting Agenda",
        description="Create a meeting agenda",
        category="productivity",
        icon="",
        messages=[
            {"role": "user", "content": """Create an agenda for a {{duration}} minute meeting about: {{topic}}

Attendees: {{attendees}}
Goals: {{goals}}

Include time allocations and action items."""}
        ],
        variables=["duration", "topic", "attendees", "goals"],
        suggested_temperature=0.4
    ),
    
    # ===== Technical =====
    ConversationTemplate(
        name="Architecture Design",
        description="Get help designing system architecture",
        category="technical",
        icon="",
        messages=[
            {"role": "user", "content": """Help me design the architecture for: {{system}}

Requirements:
{{requirements}}

Constraints:
{{constraints}}

Please suggest components, their interactions, and technology choices."""}
        ],
        variables=["system", "requirements", "constraints"],
        system_prompt="You are an experienced software architect.",
        suggested_temperature=0.5
    ),
    ConversationTemplate(
        name="API Design",
        description="Help design an API",
        category="technical",
        icon="",
        messages=[
            {"role": "user", "content": """Help me design a REST API for: {{service}}

Main resources: {{resources}}
Operations needed: {{operations}}

Include endpoint definitions, request/response formats, and error handling."""}
        ],
        variables=["service", "resources", "operations"],
        suggested_temperature=0.4
    ),
]


class TemplateLibrary:
    """
    Library for managing conversation templates.
    """
    
    def __init__(self, library_path: Optional[Path] = None):
        """
        Initialize the template library.
        
        Args:
            library_path: Path to store templates. Default: data/templates/
        """
        self._library_path = library_path or Path("data/templates")
        self._library_path.mkdir(parents=True, exist_ok=True)
        
        self._templates_file = self._library_path / "templates.json"
        self._templates: Dict[str, ConversationTemplate] = {}
        
        self._load_library()
    
    def _load_library(self) -> None:
        """Load templates from disk."""
        # Load built-in templates
        for template in BUILTIN_TEMPLATES:
            key = self._normalize_name(template.name)
            self._templates[key] = template
        
        # Load custom templates
        if self._templates_file.exists():
            try:
                with open(self._templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("templates", []):
                        template = ConversationTemplate.from_dict(item)
                        key = self._normalize_name(template.name)
                        self._templates[key] = template
                logger.info(f"Loaded {len(self._templates)} templates")
            except Exception as e:
                logger.error(f"Failed to load template library: {e}")
    
    def _save_library(self) -> None:
        """Save custom templates to disk."""
        try:
            builtin_names = {self._normalize_name(t.name) for t in BUILTIN_TEMPLATES}
            custom_templates = [
                t.to_dict() for key, t in self._templates.items()
                if key not in builtin_names
            ]
            
            data = {
                "version": "1.0",
                "templates": custom_templates
            }
            
            with open(self._templates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save template library: {e}")
    
    def _normalize_name(self, name: str) -> str:
        """Normalize template name for lookup."""
        return name.lower().strip()
    
    def save_template(
        self,
        name: str,
        messages: List[Dict[str, str]],
        description: str = "",
        category: str = "custom",
        **kwargs
    ) -> ConversationTemplate:
        """
        Save a new template or update existing.
        
        Args:
            name: Unique template name
            messages: List of {role, content} message dicts
            description: Brief description
            category: Category for organization
            **kwargs: Additional ConversationTemplate fields
            
        Returns:
            The saved ConversationTemplate
        """
        template = ConversationTemplate(
            name=name,
            messages=messages,
            description=description,
            category=category,
            **kwargs
        )
        
        key = self._normalize_name(name)
        self._templates[key] = template
        self._save_library()
        
        logger.info(f"Saved template: {name}")
        return template
    
    def get_template(self, name: str) -> Optional[ConversationTemplate]:
        """Get a template by name."""
        key = self._normalize_name(name)
        template = self._templates.get(key)
        
        if template:
            template.usage_count += 1
        
        return template
    
    def get_starter(self, name: str, **variables) -> Optional[List[Dict[str, str]]]:
        """
        Get filled-in starter messages.
        
        Args:
            name: Template name
            **variables: Variable values
            
        Returns:
            List of messages with variables filled, or None
        """
        template = self.get_template(name)
        if template:
            return template.fill_variables(**variables)
        return None
    
    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        key = self._normalize_name(name)
        
        builtin_names = {self._normalize_name(t.name) for t in BUILTIN_TEMPLATES}
        if key in builtin_names:
            logger.warning(f"Cannot delete built-in template: {name}")
            return False
        
        if key in self._templates:
            del self._templates[key]
            self._save_library()
            return True
        
        return False
    
    def list_templates(self) -> List[ConversationTemplate]:
        """Get all templates."""
        return list(self._templates.values())
    
    def list_by_category(self, category: str) -> List[ConversationTemplate]:
        """Get templates in a category."""
        return [t for t in self._templates.values() if t.category == category]
    
    def list_categories(self) -> List[str]:
        """Get all categories in use."""
        return list(set(t.category for t in self._templates.values()))
    
    def search(self, query: str) -> List[ConversationTemplate]:
        """Search templates by keyword."""
        if not query:
            return self.list_templates()
        
        query_lower = query.lower()
        results = []
        
        for template in self._templates.values():
            searchable = f"{template.name} {template.description} {' '.join(template.tags)}".lower()
            if query_lower in searchable:
                results.append(template)
        
        return results
    
    def get_required_variables(self, name: str) -> List[str]:
        """Get the variables required by a template."""
        template = self.get_template(name)
        return template.variables if template else []
    
    def export_template(self, name: str) -> Optional[str]:
        """Export a template as JSON."""
        template = self._templates.get(self._normalize_name(name))
        if template:
            return json.dumps(template.to_dict(), indent=2)
        return None
    
    def import_template(self, json_data: str) -> Optional[ConversationTemplate]:
        """Import a template from JSON."""
        try:
            data = json.loads(json_data)
            template = ConversationTemplate.from_dict(data)
            key = self._normalize_name(template.name)
            self._templates[key] = template
            self._save_library()
            return template
        except Exception as e:
            logger.error(f"Failed to import template: {e}")
            return None


# Singleton instance
_template_library_instance: Optional[TemplateLibrary] = None


def get_template_library(path: Optional[Path] = None) -> TemplateLibrary:
    """Get or create the singleton library instance."""
    global _template_library_instance
    if _template_library_instance is None:
        _template_library_instance = TemplateLibrary(path)
    return _template_library_instance


# Convenience functions
def get_conversation_starter(name: str, **variables) -> Optional[List[Dict[str, str]]]:
    """Quick access to a conversation starter."""
    return get_template_library().get_starter(name, **variables)


def list_conversation_templates() -> List[str]:
    """List all available template names."""
    return [t.name for t in get_template_library().list_templates()]


def list_template_categories() -> List[str]:
    """List all template categories."""
    return get_template_library().list_categories()
