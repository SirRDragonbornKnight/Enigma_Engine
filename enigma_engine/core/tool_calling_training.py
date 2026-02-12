"""
Tool Calling Training for Enigma AI Engine

Fine-tune models for function/tool calling capabilities.

Features:
- Tool call dataset generation
- Training data formatting
- Fine-tuning pipeline
- Evaluation metrics
- Tool schema handling

Usage:
    from enigma_engine.core.tool_calling_training import ToolCallingTrainer
    
    trainer = ToolCallingTrainer(model_path="models/base")
    
    # Generate training data
    trainer.add_tool_call_example(
        prompt="What's the weather in NYC?",
        tool_name="get_weather",
        arguments={"city": "New York"}
    )
    
    # Train
    trainer.train(epochs=3)
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolSchema:
    """Schema for a tool/function."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }
    
    def to_prompt(self) -> str:
        """Convert to prompt-friendly format."""
        params_str = ", ".join(
            f"{name}: {props.get('type', 'any')}"
            for name, props in self.parameters.items()
        )
        return f"{self.name}({params_str}) - {self.description}"


@dataclass
class ToolCallExample:
    """A single tool call training example."""
    prompt: str
    tool_name: str
    arguments: Dict[str, Any]
    thought: str = ""  # Chain of thought before calling
    result: str = ""  # Tool result for multi-turn


@dataclass
class ToolCallingConfig:
    """Configuration for tool calling training."""
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs: int = 3
    warmup_steps: int = 100
    
    # Data formatting
    system_prompt: str = "You are a helpful assistant with access to tools."
    tool_call_format: str = "json"  # "json" or "xml"
    include_thought: bool = True  # Include chain-of-thought
    
    # Augmentation
    augment_prompts: bool = True
    augment_multiplier: int = 3
    
    # Output
    output_dir: Path = Path("models/tool_trained")
    save_every: int = 500


class ToolCallFormatter:
    """Format tool calls for training."""
    
    def __init__(self, format_type: str = "json"):
        self.format_type = format_type
    
    def format_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        thought: str = ""
    ) -> str:
        """Format a tool call for model output."""
        if self.format_type == "json":
            return self._format_json(tool_name, arguments, thought)
        elif self.format_type == "xml":
            return self._format_xml(tool_name, arguments, thought)
        else:
            return self._format_simple(tool_name, arguments, thought)
    
    def _format_json(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        thought: str
    ) -> str:
        """Format as JSON."""
        output = ""
        
        if thought:
            output += f"<thinking>{thought}</thinking>\n"
        
        tool_call = {
            "tool": tool_name,
            "arguments": arguments
        }
        output += f"<tool_call>{json.dumps(tool_call)}</tool_call>"
        
        return output
    
    def _format_xml(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        thought: str
    ) -> str:
        """Format as XML."""
        output = ""
        
        if thought:
            output += f"<thinking>{thought}</thinking>\n"
        
        output += f"<tool_call>\n  <name>{tool_name}</name>\n  <arguments>\n"
        
        for key, value in arguments.items():
            output += f"    <{key}>{value}</{key}>\n"
        
        output += "  </arguments>\n</tool_call>"
        
        return output
    
    def _format_simple(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        thought: str
    ) -> str:
        """Format as simple text."""
        args_str = ", ".join(f'{k}="{v}"' for k, v in arguments.items())
        
        output = ""
        if thought:
            output += f"Thinking: {thought}\n"
        output += f"Call: {tool_name}({args_str})"
        
        return output
    
    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Parse a tool call from text."""
        if self.format_type == "json":
            return self._parse_json(text)
        elif self.format_type == "xml":
            return self._parse_xml(text)
        else:
            return self._parse_simple(text)
    
    def _parse_json(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Parse JSON format."""
        import re
        match = re.search(r'<tool_call>(.+?)</tool_call>', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return data.get("tool"), data.get("arguments", {})
            except json.JSONDecodeError:
                pass  # Intentionally silent
        return None
    
    def _parse_xml(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Parse XML format."""
        import re
        
        name_match = re.search(r'<name>(.+?)</name>', text)
        if not name_match:
            return None
        
        tool_name = name_match.group(1)
        arguments = {}
        
        # Parse arguments
        args_match = re.search(r'<arguments>(.+?)</arguments>', text, re.DOTALL)
        if args_match:
            for param_match in re.finditer(r'<(\w+)>(.+?)</\1>', args_match.group(1)):
                arguments[param_match.group(1)] = param_match.group(2)
        
        return tool_name, arguments
    
    def _parse_simple(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Parse simple format."""
        import re
        
        match = re.search(r'Call:\s*(\w+)\((.+?)\)', text)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            arguments = {}
            for arg_match in re.finditer(r'(\w+)="([^"]*)"', args_str):
                arguments[arg_match.group(1)] = arg_match.group(2)
            
            return tool_name, arguments
        
        return None


class DatasetGenerator:
    """Generate tool calling training datasets."""
    
    def __init__(self, tools: List[ToolSchema]):
        self.tools = {t.name: t for t in tools}
        
        # Augmentation templates
        self.prompt_templates = [
            "Please {action}",
            "Can you {action}?",
            "I need you to {action}",
            "Could you help me {action}?",
            "{action}",
            "I want to {action}",
            "Help me {action}",
        ]
    
    def generate_negative_examples(
        self,
        count: int = 100
    ) -> List[Dict]:
        """Generate examples where NO tool should be called."""
        negative_prompts = [
            "What do you think about AI?",
            "Tell me a joke",
            "How are you today?",
            "What's the meaning of life?",
            "Explain quantum physics to me",
            "Write a poem about nature",
            "What's your favorite color?",
            "Tell me about yourself",
            "What can you do?",
            "Thank you for your help",
        ]
        
        examples = []
        for _ in range(count):
            prompt = random.choice(negative_prompts)
            examples.append({
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "tool_calls": []  # No tool call
            })
        
        return examples
    
    def augment_prompt(self, prompt: str, action: str) -> List[str]:
        """Augment a prompt with templates."""
        augmented = [prompt]
        
        for template in self.prompt_templates:
            try:
                augmented.append(template.format(action=action))
            except KeyError:
                pass  # Intentionally silent
        
        return augmented


class ToolCallingTrainer:
    """Train models for tool calling capabilities."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: Optional[ToolCallingConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_path: Path to base model
            config: Training configuration
        """
        self.model_path = model_path
        self.config = config or ToolCallingConfig()
        
        self.tools: Dict[str, ToolSchema] = {}
        self.examples: List[ToolCallExample] = []
        self.formatter = ToolCallFormatter(self.config.tool_call_format)
        
        # Training state
        self.model = None
        self.tokenizer = None
    
    def add_tool(self, tool: ToolSchema):
        """Add a tool schema."""
        self.tools[tool.name] = tool
    
    def add_tool_call_example(
        self,
        prompt: str,
        tool_name: str,
        arguments: Dict[str, Any],
        thought: str = "",
        result: str = ""
    ):
        """
        Add a training example.
        
        Args:
            prompt: User prompt
            tool_name: Name of tool to call
            arguments: Tool arguments
            thought: Chain of thought reasoning
            result: Tool result for multi-turn
        """
        self.examples.append(ToolCallExample(
            prompt=prompt,
            tool_name=tool_name,
            arguments=arguments,
            thought=thought,
            result=result
        ))
    
    def build_training_data(self) -> List[Dict]:
        """Build training dataset from examples."""
        training_data = []
        
        # Build tools prompt
        tools_prompt = self._build_tools_prompt()
        
        for example in self.examples:
            # Format the expected output
            expected_output = self.formatter.format_tool_call(
                example.tool_name,
                example.arguments,
                example.thought if self.config.include_thought else ""
            )
            
            # Build messages
            messages = [
                {"role": "system", "content": f"{self.config.system_prompt}\n\n{tools_prompt}"},
                {"role": "user", "content": example.prompt},
                {"role": "assistant", "content": expected_output}
            ]
            
            # Add result turn if available
            if example.result:
                messages.append({"role": "tool", "content": example.result})
            
            training_data.append({
                "messages": messages,
                "tools": [t.to_dict() for t in self.tools.values()]
            })
        
        return training_data
    
    def _build_tools_prompt(self) -> str:
        """Build the tools description for system prompt."""
        lines = ["Available tools:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.to_prompt()}")
        
        lines.append("")
        lines.append("To use a tool, respond with a tool_call in the following format:")
        
        if self.config.tool_call_format == "json":
            lines.append('<tool_call>{"tool": "tool_name", "arguments": {...}}</tool_call>')
        elif self.config.tool_call_format == "xml":
            lines.append("<tool_call><name>tool_name</name><arguments>...</arguments></tool_call>")
        
        return "\n".join(lines)
    
    def load_model(self):
        """Load the base model for fine-tuning."""
        try:
            from ..core.model import create_model
            from ..core.tokenizer import get_tokenizer
            
            self.tokenizer = get_tokenizer()
            self.model = create_model("small")  # Or load from path
            
            if self.model_path and self.model_path.exists():
                import torch
                state_dict = torch.load(
                    self.model_path / "model.pt",
                    map_location="cpu"
                )
                self.model.load_state_dict(state_dict)
            
            logger.info("Model loaded for tool calling training")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(
        self,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ):
        """
        Train the model for tool calling.
        
        Args:
            epochs: Number of epochs (overrides config)
            learning_rate: Learning rate (overrides config)
        """
        epochs = epochs or self.config.epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        # Build training data
        training_data = self.build_training_data()
        
        if not training_data:
            logger.error("No training data available")
            return
        
        logger.info(f"Training on {len(training_data)} examples for {epochs} epochs")
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        try:
            from ..core.training import Trainer, TrainingConfig
            
            # Convert to training format
            texts = []
            for item in training_data:
                # Concatenate messages into training text
                text = ""
                for msg in item["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>{content}<|end|>"
                texts.append(text)
            
            # Train
            trainer_config = TrainingConfig(
                learning_rate=learning_rate,
                batch_size=self.config.batch_size,
                epochs=epochs,
                warmup_steps=self.config.warmup_steps
            )
            
            trainer = Trainer(self.model, self.tokenizer, trainer_config)
            trainer.train(texts)
            
            # Save model
            self.save_model()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the trained model."""
        import torch
        
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            self.model.state_dict(),
            output_dir / "model.pt"
        )
        
        # Save tool schemas
        tools_data = {name: tool.to_dict() for name, tool in self.tools.items()}
        (output_dir / "tools.json").write_text(json.dumps(tools_data, indent=2))
        
        # Save config
        config_data = {
            "tool_call_format": self.config.tool_call_format,
            "include_thought": self.config.include_thought,
            "system_prompt": self.config.system_prompt
        }
        (output_dir / "config.json").write_text(json.dumps(config_data, indent=2))
        
        logger.info(f"Model saved to {output_dir}")
    
    def evaluate(self, test_examples: List[ToolCallExample]) -> Dict[str, float]:
        """
        Evaluate tool calling accuracy.
        
        Args:
            test_examples: Test examples
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            self.load_model()
        
        correct_tool = 0
        correct_args = 0
        total = len(test_examples)
        
        for example in test_examples:
            # Generate model output
            output = self._generate(example.prompt)
            
            # Parse tool call
            parsed = self.formatter.parse_tool_call(output)
            
            if parsed:
                pred_tool, pred_args = parsed
                
                # Check tool name
                if pred_tool == example.tool_name:
                    correct_tool += 1
                    
                    # Check arguments
                    if pred_args == example.arguments:
                        correct_args += 1
        
        return {
            "tool_accuracy": correct_tool / total if total else 0,
            "argument_accuracy": correct_args / total if total else 0,
            "total_examples": total
        }
    
    def _generate(self, prompt: str) -> str:
        """Generate model output."""
        try:
            from ..core.inference import EnigmaEngine
            
            engine = EnigmaEngine(self.model, self.tokenizer)
            
            # Build full prompt with tools
            tools_prompt = self._build_tools_prompt()
            full_prompt = f"{self.config.system_prompt}\n\n{tools_prompt}\n\nUser: {prompt}\nAssistant:"
            
            output = engine.generate(full_prompt, max_tokens=256)
            return output
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def export_dataset(self, path: Path, format: str = "jsonl"):
        """
        Export training data.
        
        Args:
            path: Output path
            format: "jsonl" or "json"
        """
        training_data = self.build_training_data()
        
        if format == "jsonl":
            with open(path, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
        else:
            with open(path, 'w') as f:
                json.dump(training_data, f, indent=2)
        
        logger.info(f"Exported {len(training_data)} examples to {path}")
    
    def import_dataset(self, path: Path):
        """Import training examples from file."""
        if path.suffix == '.jsonl':
            with open(path) as f:
                for line in f:
                    item = json.loads(line)
                    self._import_item(item)
        else:
            with open(path) as f:
                data = json.load(f)
                for item in data:
                    self._import_item(item)
    
    def _import_item(self, item: Dict):
        """Import a single training item."""
        messages = item.get("messages", [])
        
        # Extract user prompt and assistant response
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]
        
        # Parse tool call from assistant response
        parsed = self.formatter.parse_tool_call(assistant_content)
        
        if parsed and user_content:
            tool_name, arguments = parsed
            self.add_tool_call_example(
                prompt=user_content,
                tool_name=tool_name,
                arguments=arguments
            )


# Convenience function
def train_tool_calling(
    model_path: Path,
    examples: List[Dict],
    tools: List[Dict],
    output_dir: Path,
    epochs: int = 3
):
    """
    Train a model for tool calling.
    
    Args:
        model_path: Base model path
        examples: List of {prompt, tool_name, arguments} dicts
        tools: List of tool schema dicts
        output_dir: Where to save trained model
        epochs: Training epochs
    """
    config = ToolCallingConfig(output_dir=output_dir, epochs=epochs)
    trainer = ToolCallingTrainer(model_path, config)
    
    # Add tools
    for tool in tools:
        trainer.add_tool(ToolSchema(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=tool.get("parameters", {}).get("properties", {}),
            required_params=tool.get("parameters", {}).get("required", [])
        ))
    
    # Add examples
    for ex in examples:
        trainer.add_tool_call_example(
            prompt=ex["prompt"],
            tool_name=ex["tool_name"],
            arguments=ex.get("arguments", {}),
            thought=ex.get("thought", "")
        )
    
    # Train
    trainer.train()
    
    return trainer
