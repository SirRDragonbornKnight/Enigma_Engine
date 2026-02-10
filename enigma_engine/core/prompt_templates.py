"""
Prompt Template System

Structured prompt templates with variables, validation, and composition.
Supports chain-of-thought, few-shot examples, and system prompts.

FILE: enigma_engine/core/prompt_templates.py
TYPE: Core/Inference
MAIN CLASSES: PromptTemplate, TemplateRegistry, PromptChain
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptRole(Enum):
    """Role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class TemplateFormat(Enum):
    """Template string format."""
    PYTHON = "python"  # {variable}
    JINJA2 = "jinja2"  # {{ variable }}
    MUSTACHE = "mustache"  # {{variable}}
    DOLLAR = "dollar"  # $variable or ${variable}


@dataclass
class Variable:
    """A template variable definition."""
    name: str
    description: str = ""
    default: Any = None
    required: bool = True
    validator: Optional[Callable[[Any], bool]] = None
    transformer: Optional[Callable[[Any], str]] = None
    type_hint: str = "str"
    
    def validate(self, value: Any) -> bool:
        """Validate value."""
        if self.validator:
            return self.validator(value)
        return True
    
    def transform(self, value: Any) -> str:
        """Transform value to string."""
        if self.transformer:
            return self.transformer(value)
        return str(value)


@dataclass
class Example:
    """A few-shot example."""
    input: str
    output: str
    explanation: str = ""


@dataclass
class Message:
    """A chat message."""
    role: PromptRole
    content: str
    name: str = ""


class PromptTemplate:
    """
    A structured prompt template.
    
    Supports variable substitution, validation, and formatting
    for consistent prompt generation.
    """
    
    def __init__(
        self,
        template: str,
        name: str = "",
        description: str = "",
        format: TemplateFormat = TemplateFormat.PYTHON,
        variables: dict[str, Variable] = None,
        examples: list[Example] = None,
        system_prompt: str = "",
        metadata: dict[str, Any] = None
    ):
        self.template = template
        self.name = name or self._generate_name()
        self.description = description
        self.format = format
        self.variables = variables or {}
        self.examples = examples or []
        self.system_prompt = system_prompt
        self.metadata = metadata or {}
        
        # Auto-detect variables if not provided
        if not self.variables:
            self._detect_variables()
    
    def _generate_name(self) -> str:
        """Generate a name from template hash."""
        return f"template_{hashlib.md5(self.template.encode()).hexdigest()[:8]}"
    
    def _detect_variables(self):
        """Auto-detect variables from template."""
        patterns = {
            TemplateFormat.PYTHON: r'\{(\w+)\}',
            TemplateFormat.JINJA2: r'\{\{\s*(\w+)\s*\}\}',
            TemplateFormat.MUSTACHE: r'\{\{(\w+)\}\}',
            TemplateFormat.DOLLAR: r'\$\{?(\w+)\}?'
        }
        
        pattern = patterns.get(self.format, patterns[TemplateFormat.PYTHON])
        matches = re.findall(pattern, self.template)
        
        for var_name in set(matches):
            if var_name not in self.variables:
                self.variables[var_name] = Variable(name=var_name)
    
    def get_required_variables(self) -> list[str]:
        """Get list of required variables."""
        return [
            name for name, var in self.variables.items()
            if var.required and var.default is None
        ]
    
    def validate_inputs(self, **kwargs) -> list[str]:
        """
        Validate input variables.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required variables
        for name, var in self.variables.items():
            if var.required and var.default is None:
                if name not in kwargs or kwargs[name] is None:
                    errors.append(f"Missing required variable: {name}")
        
        # Validate provided values
        for name, value in kwargs.items():
            if name in self.variables:
                var = self.variables[name]
                if not var.validate(value):
                    errors.append(f"Invalid value for {name}: {value}")
        
        return errors
    
    def format(self, **kwargs) -> str:
        """
        Format template with variables.
        
        Args:
            **kwargs: Variable values
        
        Returns:
            Formatted prompt string
        """
        # Apply defaults
        values = {}
        for name, var in self.variables.items():
            if name in kwargs:
                values[name] = var.transform(kwargs[name])
            elif var.default is not None:
                values[name] = var.transform(var.default)
        
        # Validate
        errors = self.validate_inputs(**kwargs)
        if errors:
            raise ValueError(f"Validation errors: {', '.join(errors)}")
        
        # Format based on template format
        result = self.template
        
        if self.format == TemplateFormat.PYTHON:
            result = result.format(**values)
        
        elif self.format == TemplateFormat.JINJA2:
            try:
                from jinja2 import Template
                result = Template(result).render(**values)
            except ImportError:
                # Fallback to simple replacement
                for name, value in values.items():
                    result = re.sub(
                        r'\{\{\s*' + name + r'\s*\}\}',
                        value,
                        result
                    )
        
        elif self.format == TemplateFormat.MUSTACHE:
            for name, value in values.items():
                result = result.replace(f'{{{{{name}}}}}', value)
        
        elif self.format == TemplateFormat.DOLLAR:
            for name, value in values.items():
                result = result.replace(f'${name}', value)
                result = result.replace(f'${{{name}}}', value)
        
        return result
    
    def format_with_examples(
        self,
        num_examples: int = 3,
        example_format: str = "Input: {input}\nOutput: {output}",
        **kwargs
    ) -> str:
        """
        Format template with few-shot examples.
        
        Args:
            num_examples: Number of examples to include
            example_format: Format string for each example
            **kwargs: Variable values
        
        Returns:
            Formatted prompt with examples
        """
        parts = []
        
        # Add system prompt if present
        if self.system_prompt:
            parts.append(self.system_prompt)
        
        # Add examples
        if self.examples:
            examples_to_use = self.examples[:num_examples]
            parts.append("\nExamples:\n")
            for i, ex in enumerate(examples_to_use, 1):
                formatted = example_format.format(
                    input=ex.input,
                    output=ex.output,
                    explanation=ex.explanation
                )
                parts.append(f"{i}. {formatted}")
        
        # Add main template
        parts.append("\n" + self.format(**kwargs))
        
        return "\n".join(parts)
    
    def to_messages(self, **kwargs) -> list[dict[str, str]]:
        """
        Convert to chat messages format.
        
        Args:
            **kwargs: Variable values
        
        Returns:
            List of message dicts
        """
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # Add examples as conversation turns
        for ex in self.examples:
            messages.append({
                "role": "user",
                "content": ex.input
            })
            messages.append({
                "role": "assistant",
                "content": ex.output
            })
        
        # Add main prompt
        messages.append({
            "role": "user",
            "content": self.format(**kwargs)
        })
        
        return messages
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "format": self.format.value,
            "system_prompt": self.system_prompt,
            "variables": {
                name: {
                    "description": var.description,
                    "default": var.default,
                    "required": var.required,
                    "type_hint": var.type_hint
                }
                for name, var in self.variables.items()
            },
            "examples": [
                {"input": e.input, "output": e.output, "explanation": e.explanation}
                for e in self.examples
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PromptTemplate':
        """Deserialize from dictionary."""
        variables = {}
        for name, var_data in data.get("variables", {}).items():
            variables[name] = Variable(
                name=name,
                description=var_data.get("description", ""),
                default=var_data.get("default"),
                required=var_data.get("required", True),
                type_hint=var_data.get("type_hint", "str")
            )
        
        examples = [
            Example(
                input=e["input"],
                output=e["output"],
                explanation=e.get("explanation", "")
            )
            for e in data.get("examples", [])
        ]
        
        return cls(
            template=data["template"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            format=TemplateFormat(data.get("format", "python")),
            variables=variables,
            examples=examples,
            system_prompt=data.get("system_prompt", ""),
            metadata=data.get("metadata", {})
        )


class PromptChain:
    """
    Chain of prompts for multi-step reasoning.
    
    Supports chain-of-thought, self-reflection, and
    iterative refinement patterns.
    """
    
    def __init__(
        self,
        name: str = "",
        description: str = ""
    ):
        self.name = name
        self.description = description
        self.steps: list[PromptTemplate] = []
        self._step_names: dict[str, int] = {}
    
    def add_step(
        self,
        template: Union[PromptTemplate, str],
        name: str = "",
        output_key: str = None
    ) -> 'PromptChain':
        """
        Add a step to the chain.
        
        Args:
            template: Template or template string
            name: Step name
            output_key: Key to store step output in context
        
        Returns:
            Self for chaining
        """
        if isinstance(template, str):
            template = PromptTemplate(
                template=template,
                name=name
            )
        
        template.metadata["output_key"] = output_key or f"step_{len(self.steps)}_output"
        self.steps.append(template)
        
        if name:
            self._step_names[name] = len(self.steps) - 1
        
        return self
    
    def format_step(
        self,
        step_index: int,
        context: dict[str, Any]
    ) -> str:
        """Format a specific step with context."""
        if step_index >= len(self.steps):
            raise IndexError(f"Step {step_index} not found")
        
        return self.steps[step_index].format(**context)
    
    def run(
        self,
        executor: Callable[[str], str],
        initial_context: dict[str, Any] = None,
        max_steps: int = None
    ) -> dict[str, Any]:
        """
        Run the chain with an executor function.
        
        Args:
            executor: Function that takes prompt, returns response
            initial_context: Starting context
            max_steps: Maximum steps to run
        
        Returns:
            Final context with all outputs
        """
        context = dict(initial_context or {})
        max_steps = max_steps or len(self.steps)
        
        for i, step in enumerate(self.steps[:max_steps]):
            # Format prompt
            prompt = step.format(**context)
            
            # Execute
            response = executor(prompt)
            
            # Store output
            output_key = step.metadata.get("output_key", f"step_{i}_output")
            context[output_key] = response
        
        return context


class ChainOfThought(PromptChain):
    """Chain-of-thought prompting pattern."""
    
    def __init__(self, task_template: str, system_prompt: str = ""):
        super().__init__(name="chain_of_thought")
        
        # Step 1: Think through the problem
        self.add_step(
            PromptTemplate(
                template=f"{task_template}\n\nLet's think step by step:",
                system_prompt=system_prompt
            ),
            name="reasoning",
            output_key="reasoning"
        )
        
        # Step 2: Synthesize answer
        self.add_step(
            PromptTemplate(
                template="Based on this reasoning:\n{reasoning}\n\nThe final answer is:"
            ),
            name="answer",
            output_key="answer"
        )


class SelfRefine(PromptChain):
    """Self-refinement prompting pattern."""
    
    def __init__(self, task_template: str, max_iterations: int = 3):
        super().__init__(name="self_refine")
        
        # Step 1: Initial attempt
        self.add_step(
            PromptTemplate(template=task_template),
            name="initial",
            output_key="draft"
        )
        
        # Step 2-N: Critique and improve
        for i in range(max_iterations):
            self.add_step(
                PromptTemplate(
                    template="Here is the current draft:\n{draft}\n\n"
                             "Please critique this response and identify improvements:"
                ),
                name=f"critique_{i}",
                output_key="critique"
            )
            
            self.add_step(
                PromptTemplate(
                    template="Original draft:\n{draft}\n\n"
                             "Critique:\n{critique}\n\n"
                             "Please provide an improved version:"
                ),
                name=f"improve_{i}",
                output_key="draft"
            )


class TemplateRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self, templates_dir: Path = None):
        self.templates_dir = templates_dir or Path("data/templates")
        self._templates: dict[str, PromptTemplate] = {}
        self._chains: dict[str, PromptChain] = {}
        
        # Load built-in templates
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in templates."""
        self.register("chat", PromptTemplate(
            template="{user_message}",
            name="chat",
            description="Basic chat prompt",
            system_prompt="You are a helpful AI assistant."
        ))
        
        self.register("code_review", PromptTemplate(
            template="Please review the following code:\n```{language}\n{code}\n```\n\nFocus on: {focus}",
            name="code_review",
            description="Code review prompt",
            variables={
                "language": Variable(name="language", default="python"),
                "code": Variable(name="code", required=True),
                "focus": Variable(name="focus", default="bugs, performance, readability")
            }
        ))
        
        self.register("summarize", PromptTemplate(
            template="Summarize the following text in {style}:\n\n{text}",
            name="summarize",
            description="Text summarization prompt",
            variables={
                "text": Variable(name="text", required=True),
                "style": Variable(name="style", default="3 bullet points")
            }
        ))
        
        self.register("translate", PromptTemplate(
            template="Translate the following from {source_lang} to {target_lang}:\n\n{text}",
            name="translate",
            description="Translation prompt",
            variables={
                "text": Variable(name="text", required=True),
                "source_lang": Variable(name="source_lang", default="English"),
                "target_lang": Variable(name="target_lang", required=True)
            }
        ))
        
        self.register("qa", PromptTemplate(
            template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            name="qa",
            description="Question answering with context",
            variables={
                "context": Variable(name="context", required=True),
                "question": Variable(name="question", required=True)
            }
        ))
    
    def register(self, name: str, template: PromptTemplate):
        """Register a template."""
        template.name = name
        self._templates[name] = template
        logger.debug(f"Registered template: {name}")
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)
    
    def list_templates(self) -> list[str]:
        """List all registered templates."""
        return list(self._templates.keys())
    
    def register_chain(self, name: str, chain: PromptChain):
        """Register a prompt chain."""
        chain.name = name
        self._chains[name] = chain
    
    def get_chain(self, name: str) -> Optional[PromptChain]:
        """Get a chain by name."""
        return self._chains.get(name)
    
    def save_template(self, name: str):
        """Save a template to disk."""
        template = self._templates.get(name)
        if not template:
            raise ValueError(f"Template not found: {name}")
        
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        path = self.templates_dir / f"{name}.json"
        
        with open(path, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)
    
    def load_template(self, path: Path) -> PromptTemplate:
        """Load a template from disk."""
        with open(path) as f:
            data = json.load(f)
        
        template = PromptTemplate.from_dict(data)
        self.register(template.name, template)
        return template
    
    def load_all(self):
        """Load all templates from disk."""
        if not self.templates_dir.exists():
            return
        
        for path in self.templates_dir.glob("*.json"):
            try:
                self.load_template(path)
            except Exception as e:
                logger.error(f"Failed to load template {path}: {e}")


# Global registry instance
_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Get global template registry."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry


def format_prompt(template_name: str, **kwargs) -> str:
    """
    Format a registered template.
    
    Args:
        template_name: Name of registered template
        **kwargs: Variable values
    
    Returns:
        Formatted prompt
    """
    registry = get_template_registry()
    template = registry.get(template_name)
    
    if not template:
        raise ValueError(f"Template not found: {template_name}")
    
    return template.format(**kwargs)
