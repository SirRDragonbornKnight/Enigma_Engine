"""
Interactive Tutorials

Jupyter notebook template generator for Enigma AI Engine tutorials.
Creates guided learning experiences with code, explanations, and exercises.

FILE: enigma_engine/docs/tutorials.py
TYPE: Documentation
MAIN CLASSES: TutorialGenerator, NotebookBuilder, TutorialConfig
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CellType(Enum):
    """Notebook cell types."""
    CODE = "code"
    MARKDOWN = "markdown"


class TutorialLevel(Enum):
    """Tutorial difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class NotebookCell:
    """A Jupyter notebook cell."""
    cell_type: CellType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict] = field(default_factory=list)
    execution_count: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to notebook format."""
        cell = {
            "cell_type": self.cell_type.value,
            "metadata": self.metadata,
            "source": self.content.split('\n')
        }
        
        if self.cell_type == CellType.CODE:
            cell["outputs"] = self.outputs
            cell["execution_count"] = self.execution_count
        
        return cell


@dataclass
class TutorialConfig:
    """Tutorial configuration."""
    title: str = "Enigma AI Engine Tutorial"
    level: TutorialLevel = TutorialLevel.BEGINNER
    description: str = ""
    author: str = "Enigma AI Engine Team"
    estimated_time: str = "30 minutes"
    prerequisites: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)


class NotebookBuilder:
    """Builds Jupyter notebooks."""
    
    def __init__(self, config: TutorialConfig):
        """
        Initialize notebook builder.
        
        Args:
            config: Tutorial configuration
        """
        self._config = config
        self._cells: list[NotebookCell] = []
    
    def add_markdown(self, content: str, **metadata) -> 'NotebookBuilder':
        """Add a markdown cell."""
        self._cells.append(NotebookCell(
            cell_type=CellType.MARKDOWN,
            content=content,
            metadata=metadata
        ))
        return self
    
    def add_code(self, code: str, **metadata) -> 'NotebookBuilder':
        """Add a code cell."""
        self._cells.append(NotebookCell(
            cell_type=CellType.CODE,
            content=code,
            metadata=metadata
        ))
        return self
    
    def add_header(self, level: int = 1) -> 'NotebookBuilder':
        """Add tutorial header with title and metadata."""
        content = f"""# {self._config.title}

**Level:** {self._config.level.value.capitalize()}
**Estimated Time:** {self._config.estimated_time}
**Author:** {self._config.author}

{self._config.description}
"""
        
        if self._config.learning_objectives:
            content += "\n## Learning Objectives\n\n"
            for obj in self._config.learning_objectives:
                content += f"- {obj}\n"
        
        if self._config.prerequisites:
            content += "\n## Prerequisites\n\n"
            for prereq in self._config.prerequisites:
                content += f"- {prereq}\n"
        
        return self.add_markdown(content)
    
    def add_section(self, title: str, content: str) -> 'NotebookBuilder':
        """Add a section with title and content."""
        return self.add_markdown(f"## {title}\n\n{content}")
    
    def add_exercise(self, title: str, instructions: str, starter_code: str = "",
                    hints: list[str] = None) -> 'NotebookBuilder':
        """Add an exercise with instructions and starter code."""
        exercise_md = f"""### Exercise: {title}

{instructions}
"""
        if hints:
            exercise_md += "\n<details>\n<summary>Hints</summary>\n\n"
            for i, hint in enumerate(hints, 1):
                exercise_md += f"{i}. {hint}\n"
            exercise_md += "\n</details>\n"
        
        self.add_markdown(exercise_md)
        
        if starter_code:
            self.add_code(f"# Your code here\n{starter_code}")
        else:
            self.add_code("# Your code here\n")
        
        return self
    
    def add_quiz(self, question: str, options: dict[str, str],
                 correct: str, explanation: str = "") -> 'NotebookBuilder':
        """Add a quiz question."""
        quiz_md = f"""### Quiz

{question}

"""
        for key, value in options.items():
            quiz_md += f"**{key})** {value}\n\n"
        
        quiz_md += f"""<details>
<summary>Click to reveal answer</summary>

**Answer: {correct}**

{explanation}

</details>
"""
        return self.add_markdown(quiz_md)
    
    def add_info_box(self, content: str, box_type: str = "info") -> 'NotebookBuilder':
        """Add an info/warning/tip box."""
        icons = {
            "info": "information",
            "warning": "warning",
            "tip": "bulb",
            "note": "memo"
        }
        icon = icons.get(box_type, "information")
        
        box = f"""> **{box_type.upper()}**
>
> {content}
"""
        return self.add_markdown(box)
    
    def build(self) -> dict:
        """
        Build the complete notebook.
        
        Returns:
            Jupyter notebook as dictionary
        """
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 4,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                    "mimetype": "text/x-python",
                    "file_extension": ".py"
                },
                "enigma_engine": {
                    "tutorial_level": self._config.level.value,
                    "estimated_time": self._config.estimated_time
                }
            },
            "cells": [cell.to_dict() for cell in self._cells]
        }
        
        return notebook
    
    def save(self, path: Path):
        """Save notebook to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.build(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved tutorial to {path}")


class TutorialGenerator:
    """Generates tutorial notebooks from templates."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize tutorial generator.
        
        Args:
            output_dir: Output directory for tutorials
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_quickstart(self) -> Path:
        """Generate quickstart tutorial."""
        config = TutorialConfig(
            title="Enigma AI Engine Quickstart",
            level=TutorialLevel.BEGINNER,
            description="Get up and running with Enigma AI Engine in minutes.",
            estimated_time="15 minutes",
            learning_objectives=[
                "Install Enigma AI Engine",
                "Run your first inference",
                "Understand basic concepts"
            ]
        )
        
        nb = NotebookBuilder(config)
        nb.add_header()
        
        nb.add_section("Installation", "Let's start by installing Enigma AI Engine.")
        nb.add_code("!pip install forge-ai")
        
        nb.add_section("Import", "Import the main Enigma AI Engine module.")
        nb.add_code("""import enigma_engine
from enigma_engine.core import EnigmaEngine

print(f"Enigma AI Engine version: {enigma_engine.__version__}")""")
        
        nb.add_section("Basic Inference", "Let's run a simple inference.")
        nb.add_code("""# Initialize the engine
engine = EnigmaEngine()

# Generate a response
response = engine.generate("Hello, Enigma AI Engine!")
print(response)""")
        
        nb.add_exercise(
            "Try Custom Prompts",
            "Modify the prompt below to ask Enigma AI Engine a question.",
            'prompt = "What is machine learning?"\nresponse = engine.generate(prompt)\nprint(response)',
            hints=["Try asking about specific topics", "Experiment with different prompt styles"]
        )
        
        nb.add_section("Next Steps", """
Congratulations! You've completed the quickstart tutorial.

**Recommended next tutorials:**
- Training Your Own Model
- Using the Module System
- Building with the API
""")
        
        path = self._output_dir / "01_quickstart.ipynb"
        nb.save(path)
        return path
    
    def generate_training_tutorial(self) -> Path:
        """Generate model training tutorial."""
        config = TutorialConfig(
            title="Training Models with Enigma AI Engine",
            level=TutorialLevel.INTERMEDIATE,
            description="Learn how to train custom models with your data.",
            estimated_time="45 minutes",
            prerequisites=[
                "Completed the Quickstart tutorial",
                "Basic Python knowledge",
                "Understanding of machine learning concepts"
            ],
            learning_objectives=[
                "Prepare training data",
                "Configure training parameters",
                "Train and evaluate models",
                "Export trained models"
            ]
        )
        
        nb = NotebookBuilder(config)
        nb.add_header()
        
        nb.add_section("Data Preparation", """
Training starts with good data. Enigma AI Engine accepts plain text files
where each line is a training sample.
""")
        
        nb.add_code("""# Create sample training data
training_data = '''
What is AI? AI stands for Artificial Intelligence.
Define machine learning: ML is a subset of AI focused on learning from data.
Explain neural networks: Neural networks are computing systems inspired by biological brains.
'''

# Save to file
with open('training.txt', 'w') as f:
    f.write(training_data)

print("Training data saved!")""")
        
        nb.add_section("Configure Training", "Set up training parameters.")
        
        nb.add_code("""from enigma_engine.core import TrainingConfig, Trainer

config = TrainingConfig(
    model_size="small",
    learning_rate=0.0001,
    epochs=10,
    batch_size=4
)

trainer = Trainer(config)
print(f"Training config: {config}")""")
        
        nb.add_info_box(
            "Start with a small model and few epochs to verify everything works before longer training runs.",
            "tip"
        )
        
        nb.add_section("Run Training", "Execute the training loop.")
        
        nb.add_code("""# Train the model
history = trainer.train('training.txt')

# Plot training loss
import matplotlib.pyplot as plt

plt.plot(history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()""")
        
        nb.add_exercise(
            "Customize Training",
            "Modify the training configuration to use different parameters.",
            """# Try different settings
custom_config = TrainingConfig(
    model_size="tiny",  # Change this
    learning_rate=0.001,  # Experiment with learning rate
    epochs=5,
    batch_size=8
)

# What happens with different settings?""",
            hints=[
                "Smaller models train faster but may be less capable",
                "Higher learning rates can speed up training but may be unstable"
            ]
        )
        
        nb.add_quiz(
            "What happens if the learning rate is too high?",
            {
                "A": "Training will be slower",
                "B": "The model will converge more accurately",
                "C": "Training may diverge and loss may increase",
                "D": "Memory usage will increase"
            },
            "C",
            "High learning rates can cause the optimization to overshoot minima, leading to unstable training."
        )
        
        path = self._output_dir / "02_training.ipynb"
        nb.save(path)
        return path
    
    def generate_modules_tutorial(self) -> Path:
        """Generate module system tutorial."""
        config = TutorialConfig(
            title="Enigma AI Engine Module System",
            level=TutorialLevel.INTERMEDIATE,
            description="Master the modular architecture of Enigma AI Engine.",
            estimated_time="30 minutes",
            learning_objectives=[
                "Understand the module system",
                "Load and unload modules",
                "Handle dependencies and conflicts",
                "Create custom modules"
            ]
        )
        
        nb = NotebookBuilder(config)
        nb.add_header()
        
        nb.add_section("Module Basics", """
Enigma AI Engine uses a modular architecture where every capability is a toggleable module.
This prevents conflicts and allows flexible configuration.
""")
        
        nb.add_code("""from enigma_engine.modules import ModuleManager

# Initialize module manager
manager = ModuleManager()

# List available modules
for name, info in manager.list_modules().items():
    print(f"{name}: {info.description}")""")
        
        nb.add_section("Loading Modules", "Load modules to enable capabilities.")
        
        nb.add_code("""# Load core modules
manager.load('model')
manager.load('tokenizer')
manager.load('inference')

# Check loaded modules
print("Loaded modules:", manager.get_loaded())""")
        
        nb.add_info_box(
            "The module manager automatically handles dependencies. When you load a module, its required dependencies are loaded first.",
            "info"
        )
        
        nb.add_section("Using Modules", "Access and use loaded modules.")
        
        nb.add_code("""# Get a loaded module
inference = manager.get_module('inference')

# Use it
response = inference.generate("Hello, world!")
print(response)""")
        
        path = self._output_dir / "03_modules.ipynb"
        nb.save(path)
        return path
    
    def generate_all(self) -> list[Path]:
        """Generate all tutorials."""
        tutorials = [
            self.generate_quickstart(),
            self.generate_training_tutorial(),
            self.generate_modules_tutorial()
        ]
        
        # Generate index
        index_path = self._generate_index(tutorials)
        
        return tutorials + [index_path]
    
    def _generate_index(self, tutorials: list[Path]) -> Path:
        """Generate tutorial index."""
        config = TutorialConfig(
            title="Enigma AI Engine Tutorials",
            description="Interactive tutorials for learning Enigma AI Engine."
        )
        
        nb = NotebookBuilder(config)
        nb.add_markdown(f"""# {config.title}

{config.description}

## Available Tutorials

""")
        
        for i, path in enumerate(tutorials, 1):
            name = path.stem.replace('_', ' ').title()
            nb.add_markdown(f"{i}. [{name}]({path.name})\n")
        
        nb.add_markdown("""
## Getting Help

- [Documentation](https://forge-ai.readthedocs.io)
- [GitHub Issues](https://github.com/forge-ai/forge-ai/issues)
- [Discord Community](https://discord.gg/forge-ai)
""")
        
        path = self._output_dir / "00_index.ipynb"
        nb.save(path)
        return path


def generate_tutorials(output_dir: Path = None) -> list[Path]:
    """
    Generate all tutorial notebooks.
    
    Args:
        output_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    output_dir = output_dir or Path("tutorials")
    generator = TutorialGenerator(output_dir)
    return generator.generate_all()


__all__ = [
    'TutorialGenerator',
    'TutorialConfig',
    'TutorialLevel',
    'NotebookBuilder',
    'NotebookCell',
    'CellType',
    'generate_tutorials'
]
