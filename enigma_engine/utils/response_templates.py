"""
================================================================================
Response Templates - Format AI responses with consistent structure.
================================================================================

Response formatting features:
- Pre-defined response structures
- Variable substitution
- Conditional sections
- Multiple output formats (plain, markdown, HTML)
- Custom template creation

USAGE:
    from enigma_engine.utils.response_templates import ResponseTemplates, get_response_templates
    
    templates = get_response_templates()
    
    # Use a template
    response = templates.apply("code_explanation", {
        "code": "def hello(): pass",
        "language": "python",
        "explanation": "A simple function"
    })
    
    # Create custom template
    templates.add("my_template", "Answer: {answer}\\n\\nConfidence: {confidence}%")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Output format types."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class ResponseTemplate:
    """A response template definition."""
    id: str
    name: str
    template: str
    description: str = ""
    category: str = "general"
    variables: list[str] = field(default_factory=list)
    required_vars: list[str] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.MARKDOWN
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if not self.variables:
            # Auto-detect variables from template
            self.variables = list(set(re.findall(r'\{(\w+)\}', self.template)))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "category": self.category,
            "variables": self.variables,
            "required_vars": self.required_vars,
            "output_format": self.output_format.value,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseTemplate:
        if "output_format" in data:
            data["output_format"] = OutputFormat(data["output_format"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Built-in templates
BUILTIN_TEMPLATES = [
    ResponseTemplate(
        id="code_explanation",
        name="Code Explanation",
        template="""## Code Analysis

**Language:** {language}

```{language}
{code}
```

### Explanation
{explanation}

### Key Points
{key_points}
""",
        description="Explain code with syntax highlighting",
        category="code",
        required_vars=["code", "language", "explanation"]
    ),
    
    ResponseTemplate(
        id="step_by_step",
        name="Step by Step Guide",
        template="""## {title}

{introduction}

### Steps

{steps}

### Summary
{summary}
""",
        description="Multi-step instructions",
        category="tutorial",
        required_vars=["title", "steps"]
    ),
    
    ResponseTemplate(
        id="comparison",
        name="Comparison",
        template="""## {title}

### Option A: {option_a_name}
{option_a_details}

**Pros:** {option_a_pros}
**Cons:** {option_a_cons}

### Option B: {option_b_name}
{option_b_details}

**Pros:** {option_b_pros}
**Cons:** {option_b_cons}

### Recommendation
{recommendation}
""",
        description="Compare two options",
        category="analysis",
        required_vars=["title", "option_a_name", "option_b_name"]
    ),
    
    ResponseTemplate(
        id="error_solution",
        name="Error Solution",
        template="""## Error: {error_type}

### Problem
{problem_description}

### Cause
{cause}

### Solution

```{language}
{solution_code}
```

### Prevention
{prevention_tips}
""",
        description="Explain and solve errors",
        category="debugging",
        required_vars=["error_type", "problem_description", "solution_code"]
    ),
    
    ResponseTemplate(
        id="summary",
        name="Summary",
        template="""## Summary: {topic}

### Overview
{overview}

### Key Points
{key_points}

### Conclusion
{conclusion}
""",
        description="Summarize a topic",
        category="general",
        required_vars=["topic", "overview"]
    ),
    
    ResponseTemplate(
        id="qa_response",
        name="Q&A Response",
        template="""### Question
{question}

### Answer
{answer}

{additional_info}
""",
        description="Question and answer format",
        category="general",
        required_vars=["question", "answer"]
    ),
    
    ResponseTemplate(
        id="api_documentation",
        name="API Documentation",
        template="""## {function_name}

{description}

### Signature
```{language}
{signature}
```

### Parameters
{parameters}

### Returns
{returns}

### Example
```{language}
{example}
```
""",
        description="Document an API function",
        category="documentation",
        required_vars=["function_name", "signature", "parameters"]
    ),
    
    ResponseTemplate(
        id="review",
        name="Review",
        template="""## Review: {subject}

**Rating:** {rating}/5

### Summary
{summary}

### Pros
{pros}

### Cons
{cons}

### Verdict
{verdict}
""",
        description="Review format",
        category="analysis",
        required_vars=["subject", "rating", "summary"]
    ),
    
    ResponseTemplate(
        id="list_response",
        name="List Response",
        template="""## {title}

{introduction}

{items}

{conclusion}
""",
        description="Numbered or bulleted list",
        category="general",
        required_vars=["title", "items"]
    ),
    
    ResponseTemplate(
        id="definition",
        name="Definition",
        template="""## {term}

**Definition:** {definition}

### Context
{context}

### Examples
{examples}

### Related Terms
{related_terms}
""",
        description="Define a term or concept",
        category="educational",
        required_vars=["term", "definition"]
    ),
    
    ResponseTemplate(
        id="troubleshooting",
        name="Troubleshooting",
        template="""## Troubleshooting: {issue}

### Symptoms
{symptoms}

### Possible Causes
{causes}

### Solutions

{solutions}

### If Issue Persists
{escalation}
""",
        description="Troubleshooting guide",
        category="support",
        required_vars=["issue", "symptoms", "solutions"]
    ),
    
    ResponseTemplate(
        id="changelog",
        name="Changelog Entry",
        template="""## Version {version} ({date})

### Added
{added}

### Changed
{changed}

### Fixed
{fixed}

### Deprecated
{deprecated}
""",
        description="Changelog format",
        category="documentation",
        required_vars=["version", "date"]
    ),
    
    ResponseTemplate(
        id="plain_answer",
        name="Plain Answer",
        template="{answer}",
        description="Simple plain text response",
        category="general",
        output_format=OutputFormat.PLAIN,
        required_vars=["answer"]
    ),
    
    ResponseTemplate(
        id="thinking_process",
        name="Thinking Process",
        template="""### Thinking...

{thought_process}

### Conclusion

{conclusion}
""",
        description="Show reasoning process",
        category="general",
        required_vars=["thought_process", "conclusion"]
    ),
    
    ResponseTemplate(
        id="task_result",
        name="Task Result",
        template="""## Task: {task_name}

**Status:** {status}

### Results
{results}

### Details
{details}

### Next Steps
{next_steps}
""",
        description="Report task completion",
        category="workflow",
        required_vars=["task_name", "status", "results"]
    ),
]


class ResponseTemplates:
    """
    Manage and apply response templates.
    """
    
    def __init__(self, data_path: Path | None = None):
        """
        Initialize response templates.
        
        Args:
            data_path: Path to store custom templates
        """
        self._data_path = data_path or Path("data/templates")
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._templates_file = self._data_path / "response_templates.json"
        
        self._templates: dict[str, ResponseTemplate] = {}
        self._custom_formatters: dict[str, Callable[[str], str]] = {}
        
        # Load built-in templates
        for tmpl in BUILTIN_TEMPLATES:
            self._templates[tmpl.id] = tmpl
        
        # Load custom templates
        self._load_custom_templates()
    
    def _load_custom_templates(self) -> None:
        """Load custom templates from disk."""
        if self._templates_file.exists():
            try:
                with open(self._templates_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for tmpl_data in data.get("templates", []):
                        tmpl = ResponseTemplate.from_dict(tmpl_data)
                        self._templates[tmpl.id] = tmpl
                logger.info(f"Loaded custom response templates")
            except Exception as e:
                logger.error(f"Failed to load custom templates: {e}")
    
    def _save_custom_templates(self) -> None:
        """Save custom templates to disk."""
        try:
            # Only save non-builtin templates
            builtin_ids = {t.id for t in BUILTIN_TEMPLATES}
            custom = [
                t.to_dict() for t in self._templates.values()
                if t.id not in builtin_ids
            ]
            
            with open(self._templates_file, 'w', encoding='utf-8') as f:
                json.dump({"templates": custom}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save custom templates: {e}")
    
    def get(self, template_id: str) -> ResponseTemplate | None:
        """Get a template by ID."""
        return self._templates.get(template_id)
    
    def list_templates(
        self,
        category: str | None = None
    ) -> list[ResponseTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return sorted(templates, key=lambda t: t.name)
    
    def get_categories(self) -> list[str]:
        """Get all template categories."""
        return sorted({t.category for t in self._templates.values()})
    
    def add(
        self,
        template_id: str,
        template: str,
        name: str | None = None,
        description: str = "",
        category: str = "custom",
        required_vars: list[str] | None = None,
        output_format: OutputFormat = OutputFormat.MARKDOWN
    ) -> ResponseTemplate:
        """
        Add a custom template.
        
        Args:
            template_id: Unique template identifier
            template: Template string with {variable} placeholders
            name: Human-readable name
            description: Template description
            category: Category for organization
            required_vars: Required variables
            output_format: Output format
            
        Returns:
            The created template
        """
        tmpl = ResponseTemplate(
            id=template_id,
            name=name or template_id.replace("_", " ").title(),
            template=template,
            description=description,
            category=category,
            required_vars=required_vars or [],
            output_format=output_format
        )
        
        self._templates[template_id] = tmpl
        self._save_custom_templates()
        
        return tmpl
    
    def remove(self, template_id: str) -> bool:
        """Remove a custom template."""
        builtin_ids = {t.id for t in BUILTIN_TEMPLATES}
        if template_id in builtin_ids:
            logger.warning(f"Cannot remove built-in template: {template_id}")
            return False
        
        if template_id in self._templates:
            del self._templates[template_id]
            self._save_custom_templates()
            return True
        return False
    
    def apply(
        self,
        template_id: str,
        variables: dict[str, Any],
        output_format: OutputFormat | None = None,
        strict: bool = False
    ) -> str:
        """
        Apply a template with variables.
        
        Args:
            template_id: Template to use
            variables: Variable values
            output_format: Override output format
            strict: Raise error if required vars missing
            
        Returns:
            Formatted response
        """
        tmpl = self._templates.get(template_id)
        if not tmpl:
            logger.warning(f"Template not found: {template_id}")
            return str(variables.get("content", ""))
        
        # Check required variables
        if strict and tmpl.required_vars:
            missing = [v for v in tmpl.required_vars if v not in variables]
            if missing:
                raise ValueError(f"Missing required variables: {missing}")
        
        # Fill in defaults for missing optional variables
        filled_vars = {v: "" for v in tmpl.variables}
        filled_vars.update(variables)
        
        # Apply template
        result = tmpl.template
        for var, value in filled_vars.items():
            # Handle lists/dicts by converting to string
            if isinstance(value, (list, dict)):
                if isinstance(value, list):
                    value = "\n".join(f"- {item}" for item in value)
                else:
                    value = json.dumps(value, indent=2)
            result = result.replace(f"{{{var}}}", str(value))
        
        # Apply custom formatters
        for formatter_name, formatter in self._custom_formatters.items():
            pattern = f"{{{{format:{formatter_name}:([^}}]+)}}}}"
            for match in re.finditer(pattern, result):
                original = match.group(0)
                content = match.group(1)
                result = result.replace(original, formatter(content))
        
        # Convert format if needed
        target_format = output_format or tmpl.output_format
        result = self._convert_format(result, tmpl.output_format, target_format)
        
        return result.strip()
    
    def _convert_format(
        self,
        content: str,
        from_format: OutputFormat,
        to_format: OutputFormat
    ) -> str:
        """Convert between output formats."""
        if from_format == to_format:
            return content
        
        if from_format == OutputFormat.MARKDOWN and to_format == OutputFormat.HTML:
            return self._markdown_to_html(content)
        
        if from_format == OutputFormat.MARKDOWN and to_format == OutputFormat.PLAIN:
            return self._markdown_to_plain(content)
        
        if from_format == OutputFormat.HTML and to_format == OutputFormat.PLAIN:
            return self._html_to_plain(content)
        
        return content
    
    def _markdown_to_html(self, content: str) -> str:
        """Convert markdown to HTML (basic conversion)."""
        # Headers
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        
        # Bold and italic
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
        
        # Code blocks
        content = re.sub(
            r'```(\w+)?\n(.*?)```',
            r'<pre><code class="\1">\2</code></pre>',
            content,
            flags=re.DOTALL
        )
        content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
        
        # Lists
        lines = content.split('\n')
        in_list = False
        result = []
        for line in lines:
            if line.startswith('- '):
                if not in_list:
                    result.append('<ul>')
                    in_list = True
                result.append(f'<li>{line[2:]}</li>')
            else:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                result.append(line)
        if in_list:
            result.append('</ul>')
        
        # Paragraphs
        content = '\n'.join(result)
        content = re.sub(r'\n\n+', '</p><p>', content)
        content = f'<p>{content}</p>'
        content = re.sub(r'<p></p>', '', content)
        
        return content
    
    def _markdown_to_plain(self, content: str) -> str:
        """Convert markdown to plain text."""
        # Remove headers markers
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        
        # Remove bold/italic
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
        content = re.sub(r'\*(.+?)\*', r'\1', content)
        
        # Remove code block markers
        content = re.sub(r'```\w*\n?', '', content)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        return content
    
    def _html_to_plain(self, content: str) -> str:
        """Convert HTML to plain text."""
        # Remove tags
        content = re.sub(r'<[^>]+>', '', content)
        # Decode entities
        content = content.replace('&lt;', '<').replace('&gt;', '>')
        content = content.replace('&amp;', '&').replace('&quot;', '"')
        return content
    
    def register_formatter(
        self,
        name: str,
        formatter: Callable[[str], str]
    ) -> None:
        """
        Register a custom formatter function.
        
        Args:
            name: Formatter name (used as {{format:name:content}})
            formatter: Function that takes content and returns formatted content
        """
        self._custom_formatters[name] = formatter
    
    def quick_format(
        self,
        content: str,
        format_type: str = "paragraph"
    ) -> str:
        """
        Quick format content without a template.
        
        Args:
            content: Content to format
            format_type: Type of formatting
            
        Returns:
            Formatted content
        """
        formatters = {
            "paragraph": lambda c: c,
            "bullet_list": lambda c: "\n".join(f"- {line}" for line in c.split("\n") if line.strip()),
            "numbered_list": lambda c: "\n".join(f"{i+1}. {line}" for i, line in enumerate(c.split("\n")) if line.strip()),
            "quote": lambda c: "\n".join(f"> {line}" for line in c.split("\n")),
            "code": lambda c: f"```\n{c}\n```",
            "header1": lambda c: f"# {c}",
            "header2": lambda c: f"## {c}",
            "header3": lambda c: f"### {c}",
            "bold": lambda c: f"**{c}**",
            "italic": lambda c: f"*{c}*",
        }
        
        formatter = formatters.get(format_type, formatters["paragraph"])
        return formatter(content)


# Singleton instance
_templates_instance: ResponseTemplates | None = None


def get_response_templates(data_path: Path | None = None) -> ResponseTemplates:
    """Get or create the singleton response templates."""
    global _templates_instance
    if _templates_instance is None:
        _templates_instance = ResponseTemplates(data_path)
    return _templates_instance


# Convenience functions
def apply_template(template_id: str, variables: dict[str, Any]) -> str:
    """Apply a response template."""
    return get_response_templates().apply(template_id, variables)


def list_templates(category: str | None = None) -> list[ResponseTemplate]:
    """List available templates."""
    return get_response_templates().list_templates(category)
