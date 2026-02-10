"""
Prompt Templates Library for Enigma AI Engine

Reusable prompt templates for common tasks.

Features:
- Pre-built templates for common tasks
- Variable substitution
- Template composition
- Custom template creation
- Template versioning

Usage:
    from enigma_engine.core.prompts import PromptLibrary, Template
    
    # Use built-in template
    lib = PromptLibrary()
    prompt = lib.get("summarize").format(text="Long text here...")
    
    # Create custom template
    template = Template(
        "Write a {style} story about {topic}",
        variables=["style", "topic"]
    )
    prompt = template.format(style="funny", topic="robots")
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """A prompt template."""
    content: str
    name: str = ""
    description: str = ""
    variables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    
    # Metadata
    author: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Settings
    default_values: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract variables if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        # Match {variable_name} patterns
        pattern = r"\{(\w+)\}"
        return list(set(re.findall(pattern, self.content)))
    
    def format(self, **kwargs) -> str:
        """
        Format template with values.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt
        """
        # Merge with defaults
        values = {**self.default_values, **kwargs}
        
        # Check for missing variables
        missing = set(self.variables) - set(values.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        return self.content.format(**values)
    
    def partial(self, **kwargs) -> "Template":
        """
        Create partial template with some values filled.
        
        Args:
            **kwargs: Values to fill
            
        Returns:
            New template with remaining variables
        """
        new_content = self.content
        new_vars = []
        
        for var in self.variables:
            if var in kwargs:
                new_content = new_content.replace(f"{{{var}}}", str(kwargs[var]))
            else:
                new_vars.append(var)
        
        return Template(
            content=new_content,
            name=f"{self.name}_partial",
            variables=new_vars,
            tags=self.tags
        )
    
    def validate(self) -> bool:
        """Validate template syntax."""
        try:
            # Try to extract variables
            self._extract_variables()
            
            # Check for balanced braces
            if self.content.count("{") != self.content.count("}"):
                return False
            
            # Try formatting with dummy values
            dummy = {var: "test" for var in self.variables}
            self.format(**dummy)
            
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "variables": self.variables,
            "tags": self.tags,
            "version": self.version,
            "default_values": self.default_values
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Template":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            variables=data.get("variables", []),
            tags=data.get("tags", []),
            version=data.get("version", "1.0"),
            default_values=data.get("default_values", {})
        )


# Built-in templates
BUILTIN_TEMPLATES = {
    # Text processing
    "summarize": Template(
        content="""Summarize the following text in {length} sentences:

{text}

Summary:""",
        name="summarize",
        description="Summarize text",
        tags=["text", "summarization"],
        default_values={"length": "3"}
    ),
    
    "explain": Template(
        content="""Explain {topic} in simple terms that a {audience} could understand.

Explanation:""",
        name="explain",
        description="Explain a topic simply",
        tags=["education", "explanation"],
        default_values={"audience": "beginner"}
    ),
    
    "translate": Template(
        content="""Translate the following text from {source_lang} to {target_lang}:

{text}

Translation:""",
        name="translate",
        description="Translate text between languages",
        tags=["translation", "text"]
    ),
    
    "rewrite": Template(
        content="""Rewrite the following text in a {style} style:

{text}

Rewritten:""",
        name="rewrite",
        description="Rewrite text in different style",
        tags=["text", "style"],
        default_values={"style": "professional"}
    ),
    
    # Code
    "code_explain": Template(
        content="""Explain what this {language} code does:

```{language}
{code}
```

Explanation:""",
        name="code_explain",
        description="Explain code",
        tags=["code", "explanation"],
        default_values={"language": "python"}
    ),
    
    "code_review": Template(
        content="""Review this {language} code for potential issues:

```{language}
{code}
```

List any bugs, improvements, or security concerns:""",
        name="code_review",
        description="Review code for issues",
        tags=["code", "review"]
    ),
    
    "code_generate": Template(
        content="""Write {language} code that does the following:

{description}

Requirements:
{requirements}

Code:""",
        name="code_generate",
        description="Generate code from description",
        tags=["code", "generation"],
        default_values={"requirements": "- Clean, readable code\n- Handle edge cases"}
    ),
    
    "code_convert": Template(
        content="""Convert this {source_lang} code to {target_lang}:

```{source_lang}
{code}
```

{target_lang} version:""",
        name="code_convert",
        description="Convert code between languages",
        tags=["code", "conversion"]
    ),
    
    # Chat
    "chat_system": Template(
        content="""You are {name}, {description}.

Your traits:
{traits}

You always:
{behaviors}""",
        name="chat_system",
        description="System prompt for chat persona",
        tags=["chat", "persona"],
        default_values={
            "name": "an AI assistant",
            "description": "a helpful and friendly assistant",
            "traits": "- Knowledgeable\n- Patient\n- Clear",
            "behaviors": "- Answer questions accurately\n- Ask for clarification when needed"
        }
    ),
    
    "few_shot": Template(
        content="""{instructions}

Examples:
{examples}

Now complete:
Input: {input}
Output:""",
        name="few_shot",
        description="Few-shot prompting template",
        tags=["learning", "examples"]
    ),
    
    # Analysis
    "sentiment": Template(
        content="""Analyze the sentiment of this text. Classify as positive, negative, or neutral, and explain why.

Text: {text}

Sentiment:""",
        name="sentiment",
        description="Sentiment analysis",
        tags=["analysis", "sentiment"]
    ),
    
    "extract_entities": Template(
        content="""Extract the following entities from this text:
- People names
- Organizations
- Locations
- Dates

Text: {text}

Entities:""",
        name="extract_entities",
        description="Named entity extraction",
        tags=["analysis", "entities"]
    ),
    
    # Creative
    "story": Template(
        content="""Write a {length} {genre} story about {topic}.

Setting: {setting}
Main character: {character}

Story:""",
        name="story",
        description="Generate a story",
        tags=["creative", "story"],
        default_values={
            "length": "short",
            "genre": "adventure",
            "setting": "a magical forest",
            "character": "a curious young explorer"
        }
    ),
    
    "brainstorm": Template(
        content="""Generate {count} creative ideas for {topic}.

Consider: {constraints}

Ideas:""",
        name="brainstorm",
        description="Brainstorm ideas",
        tags=["creative", "ideation"],
        default_values={"count": "5", "constraints": "practicality and novelty"}
    ),
    
    # Structured output
    "json_output": Template(
        content="""{task}

Respond with valid JSON in this format:
{schema}

Input: {input}

JSON Response:""",
        name="json_output",
        description="Generate structured JSON output",
        tags=["structured", "json"]
    ),
    
    # Chain of thought
    "cot": Template(
        content="""Let's solve this step by step.

Problem: {problem}

Think through this carefully:
Step 1:""",
        name="cot",
        description="Chain of thought reasoning",
        tags=["reasoning", "cot"]
    )
}


class PromptLibrary:
    """
    Library of prompt templates.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize library.
        
        Args:
            templates_dir: Directory to load custom templates from
        """
        self._templates: Dict[str, Template] = {}
        
        # Load built-in templates
        self._templates.update(BUILTIN_TEMPLATES)
        
        # Load custom templates
        if templates_dir:
            self.load_directory(templates_dir)
    
    def get(self, name: str) -> Template:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template
        """
        if name not in self._templates:
            raise KeyError(f"Template not found: {name}")
        return self._templates[name]
    
    def add(self, template: Template):
        """Add a template to the library."""
        if not template.name:
            raise ValueError("Template must have a name")
        self._templates[template.name] = template
    
    def remove(self, name: str):
        """Remove a template."""
        if name in self._templates:
            del self._templates[name]
    
    def list_templates(self, tag: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            tag: Filter by tag
            
        Returns:
            List of template names
        """
        if tag:
            return [
                name for name, tmpl in self._templates.items()
                if tag in tmpl.tags
            ]
        return list(self._templates.keys())
    
    def search(self, query: str) -> List[Template]:
        """
        Search templates by name or description.
        
        Args:
            query: Search query
            
        Returns:
            Matching templates
        """
        query_lower = query.lower()
        results = []
        
        for template in self._templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag for tag in template.tags)):
                results.append(template)
        
        return results
    
    def load_directory(self, path: str):
        """Load templates from a directory."""
        path = Path(path)
        if not path.exists():
            return
        
        for file_path in path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        template = Template.from_dict(item)
                        self.add(template)
                else:
                    template = Template.from_dict(data)
                    self.add(template)
                    
            except Exception as e:
                logger.warning(f"Failed to load template from {file_path}: {e}")
    
    def save_template(self, name: str, path: str):
        """Save a template to file."""
        if name not in self._templates:
            raise KeyError(f"Template not found: {name}")
        
        template = self._templates[name]
        
        with open(path, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
    
    def save_all(self, path: str):
        """Save all custom templates."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for name, template in self._templates.items():
            if name not in BUILTIN_TEMPLATES:
                file_path = path / f"{name}.json"
                with open(file_path, "w") as f:
                    json.dump(template.to_dict(), f, indent=2)
    
    def get_tags(self) -> Set[str]:
        """Get all unique tags."""
        tags = set()
        for template in self._templates.values():
            tags.update(template.tags)
        return tags


class PromptBuilder:
    """
    Fluent builder for complex prompts.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._parts: List[str] = []
        self._variables: Dict[str, str] = {}
    
    def system(self, text: str) -> "PromptBuilder":
        """Add system instruction."""
        self._parts.append(f"System: {text}")
        return self
    
    def context(self, text: str) -> "PromptBuilder":
        """Add context."""
        self._parts.append(f"Context: {text}")
        return self
    
    def instruction(self, text: str) -> "PromptBuilder":
        """Add instruction."""
        self._parts.append(f"Instruction: {text}")
        return self
    
    def example(self, input_text: str, output_text: str) -> "PromptBuilder":
        """Add example."""
        self._parts.append(f"Example:\nInput: {input_text}\nOutput: {output_text}")
        return self
    
    def examples(self, examples: List[tuple]) -> "PromptBuilder":
        """Add multiple examples."""
        for inp, out in examples:
            self.example(inp, out)
        return self
    
    def input(self, text: str) -> "PromptBuilder":
        """Add input."""
        self._parts.append(f"Input: {text}")
        return self
    
    def output_format(self, format_spec: str) -> "PromptBuilder":
        """Specify output format."""
        self._parts.append(f"Output format: {format_spec}")
        return self
    
    def text(self, text: str) -> "PromptBuilder":
        """Add arbitrary text."""
        self._parts.append(text)
        return self
    
    def variable(self, name: str) -> "PromptBuilder":
        """Add a variable placeholder."""
        self._parts.append(f"{{{name}}}")
        return self
    
    def build(self, **kwargs) -> str:
        """
        Build the final prompt.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Complete prompt
        """
        prompt = "\n\n".join(self._parts)
        
        # Substitute variables
        for name, value in {**self._variables, **kwargs}.items():
            prompt = prompt.replace(f"{{{name}}}", str(value))
        
        return prompt
    
    def to_template(self, name: str) -> Template:
        """Convert to a reusable template."""
        content = "\n\n".join(self._parts)
        return Template(content=content, name=name)


# Convenience functions
def prompt(template_name: str, **kwargs) -> str:
    """
    Quick function to format a template.
    
    Args:
        template_name: Built-in template name
        **kwargs: Variable values
        
    Returns:
        Formatted prompt
    """
    lib = PromptLibrary()
    return lib.get(template_name).format(**kwargs)


def build() -> PromptBuilder:
    """Create a new prompt builder."""
    return PromptBuilder()
