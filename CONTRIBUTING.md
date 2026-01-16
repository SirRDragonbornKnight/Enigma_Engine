# Contributing to ForgeAI

Thank you for considering contributing to ForgeAI! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Architecture](#project-architecture)
5. [How to Contribute](#how-to-contribute)
6. [Coding Standards](#coding-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Pull Request Process](#pull-request-process)

## Code of Conduct

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Provide helpful feedback and suggestions
- **Be patient** - Remember that everyone is learning
- **Be inclusive** - Welcome contributors of all skill levels

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your feature/fix
4. **Make your changes**
5. **Test thoroughly**
6. **Submit a pull request**

## Development Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest black flake8 mypy
```

### Verify Installation

```bash
# Run test suite
python -m pytest tests/ -v

# Check imports
python -c "from forge_ai.modules import ModuleManager; print('✓ OK')"
```

## Project Architecture

### Module System

Enigma uses a **module-based architecture** where everything is toggleable:

```
┌─────────────────────────────────────────────────────────────┐
│                     MODULE MANAGER                           │
│              Central control for all modules                 │
├─────────────────────────────────────────────────────────────┤
│  CORE          │  GENERATION    │  MEMORY      │  TOOLS     │
│  - model       │  - image_gen   │  - memory    │  - web     │
│  - tokenizer   │  - code_gen    │  - embedding │  - files   │
│  - training    │  - video_gen   │              │            │
│  - inference   │  - audio_gen   │              │            │
└─────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- Everything is a module
- Modules declare dependencies
- Conflict prevention is automatic
- Resources are freed when unloaded

**Read more:** [docs/MODULE_GUIDE.md](docs/MODULE_GUIDE.md)

### Package Structure

```
forge_ai/
├── core/          # AI model, training, inference
├── modules/       # Module system (manager, registry)
├── addons/        # AI generation addons (wrapped as modules)
├── memory/        # Conversation and knowledge storage
├── comms/         # API server, networking
├── gui/           # Desktop interface
├── tools/         # Web, files, vision, etc.
├── voice/         # TTS/STT
└── avatar/        # Visual representation
```

## How to Contribute

### Types of Contributions

#### 🐛 Bug Fixes
- Report bugs via GitHub Issues
- Include: OS, Python version, error message, minimal reproduction
- Submit PR with fix and test case

#### ✨ New Features
- **Discuss first!** Open an issue to propose the feature
- Ensure it fits the project philosophy
- Consider if it should be a module or addon
- Write tests and documentation

#### 📚 Documentation
- Improve README, guides, docstrings
- Add examples and tutorials
- Fix typos and clarify explanations

#### 🧪 Tests
- Add test coverage for untested code
- Improve existing tests
- Add integration tests

#### 🎨 Code Quality
- Refactor for clarity and performance
- Improve error messages
- Add type hints

### Creating a Custom Module

**Example:** Adding a new AI capability

```python
# forge_ai/modules/my_module.py
from forge_ai.modules import Module, ModuleInfo, ModuleCategory

class MyModule(Module):
    INFO = ModuleInfo(
        id="my_module",
        name="My Custom Module",
        description="Adds XYZ capability",
        category=ModuleCategory.EXTENSION,
        requires=["model"],  # Dependencies
        provides=["xyz_capability"],  # What it adds
    )
    
    def load(self) -> bool:
        # Initialize your module
        self._instance = MyImplementation()
        return True
    
    def unload(self) -> bool:
        # Clean up
        if self._instance:
            del self._instance
        return True
```

Then register it in `forge_ai/modules/registry.py`:
```python
MODULE_REGISTRY['my_module'] = MyModule
```

### Adding a New Generation Tab

Generation capabilities now live in GUI tabs. To add a new generator:

```python
# forge_ai/gui/tabs/my_gen_tab.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit
from ..base_tab import BaseTab

class MyGenTab(BaseTab):
    """Custom generation tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.prompt_input = QTextEdit()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate)
        layout.addWidget(self.prompt_input)
        layout.addWidget(self.generate_btn)
    
    def generate(self):
        prompt = self.prompt_input.toPlainText()
        # Your generation logic here
        result = self._do_generation(prompt)
        self.show_result(result)
```

## Coding Standards

### Python Style

- **PEP 8** compliant (with some flexibility)
- **4 spaces** for indentation
- **snake_case** for functions and variables
- **PascalCase** for classes
- **UPPER_CASE** for constants

### Code Quality Requirements

All new code must follow these standards:

#### 1. Type Hints

Add type hints to all function signatures:

```python
# Good ✓
def process_data(input: str, max_length: int = 100) -> Dict[str, Any]:
    return {"result": input[:max_length]}

# Bad ✗
def process_data(input, max_length=100):
    return {"result": input[:max_length]}
```

#### 2. Error Handling

Use specific exception types, never bare `except:`:

```python
# Good ✓
try:
    result = risky_operation()
except (ValueError, KeyError) as e:
    logger.error(f"Operation failed: {e}")
    return None

# Bad ✗
try:
    result = risky_operation()
except:
    pass
```

#### 3. Input Validation

Validate all user inputs at function boundaries:

```python
# Good ✓
def save_file(path: str, content: str) -> bool:
    if not path:
        raise ValueError("Path cannot be empty")
    if not content:
        raise ValueError("Content cannot be empty")
    # ... rest of function

# Bad ✗
def save_file(path, content):
    # No validation, might crash with None values
    with open(path, 'w') as f:
        f.write(content)
```

#### 4. Resource Limits

Add limits to prevent resource exhaustion:

```python
# Good ✓
def read_file(path: str, max_size: int = 100 * 1024 * 1024) -> str:
    file_size = Path(path).stat().st_size
    if file_size > max_size:
        raise ValueError(f"File too large: {file_size} bytes")
    with open(path, 'r') as f:
        return f.read()

# Bad ✗
def read_file(path):
    # Could read a 10GB file into memory!
    with open(path, 'r') as f:
        return f.read()
```

#### 5. Documentation

Document all public APIs:

```python
# Good ✓
def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross-entropy loss between predictions and targets.
    
    Args:
        predictions: Model predictions of shape (batch, vocab_size)
        targets: Ground truth labels of shape (batch,)
        
    Returns:
        Scalar loss tensor
        
    Example:
        >>> preds = torch.randn(32, 1000)
        >>> targets = torch.randint(0, 1000, (32,))
        >>> loss = calculate_loss(preds, targets)
    """
    return F.cross_entropy(predictions, targets)

# Bad ✗
def calculate_loss(predictions, targets):
    # No documentation
    return F.cross_entropy(predictions, targets)
```

### Code Formatting

```bash
# Format code
black forge_ai/

# Check style
flake8 forge_ai/

# Type checking
mypy forge_ai/
```

### Common Patterns

#### Safe File Operations

```python
from pathlib import Path

def safe_read(path: str) -> Optional[str]:
    """Read file with proper error handling."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            logger.error(f"File not found: {p}")
            return None
        return p.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError) as e:
        logger.error(f"Failed to read {path}: {e}")
        return None
```

#### Safe Network Operations

```python
def fetch_url(url: str, timeout: int = 10) -> Optional[bytes]:
    """Fetch URL with timeout and error handling."""
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL scheme")
    
    try:
        response = urllib.request.urlopen(url, timeout=timeout)
        return response.read()
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None
```

#### Module Loading Pattern

```python
def load(self) -> bool:
    """Load module with proper error handling."""
    try:
        # Import and initialize
        from my_package import MyClass
        self._instance = MyClass()
        self.is_loaded = True
        return True
    except ImportError:
        logger.warning("Package not installed: pip install my_package")
        return False
    except Exception as e:
        logger.error(f"Failed to load module: {e}")
        return False
```

### Documentation

- **Docstrings** for all public classes and functions
- **Type hints** for function signatures
- **Comments** for complex logic only
- **Examples** in docstrings where helpful

#### Docstring Format

```python
def my_function(arg1: str, arg2: int = 10) -> bool:
    """
    Brief one-line description.
    
    More detailed explanation if needed. Can span
    multiple lines.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2, defaults to 10
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> my_function("test", 5)
        True
    """
    pass
```

## Testing

### Running Tests

```bash
# All tests
python -m unittest discover tests

# Specific test file
python -m unittest tests.test_modules

# With pytest (if installed)
pytest tests/ -v
```

### Writing Tests

```python
# tests/test_my_module.py
import unittest
from forge_ai.modules import ModuleManager
from forge_ai.modules.my_module import MyModule

class TestMyModule(unittest.TestCase):
    def setUp(self):
        self.manager = ModuleManager()
        self.manager.register(MyModule)
    
    def test_load(self):
        success = self.manager.load('my_module')
        self.assertTrue(success)
    
    def test_functionality(self):
        self.manager.load('my_module')
        module = self.manager.get_interface('my_module')
        result = module.do_something()
        self.assertEqual(result, expected_value)
```

### Test Coverage

- **Unit tests** for individual functions/classes
- **Integration tests** for module interactions
- **System tests** for end-to-end workflows

**Minimum coverage:** Aim for 70%+ on new code

## Documentation

### What to Document

1. **Public APIs** - All public classes, functions, parameters
2. **Module Info** - What each module does, its requirements
3. **Examples** - How to use new features
4. **Architecture** - How components fit together

### Where to Document

- **Docstrings** - In-code documentation
- **README.md** - High-level overview
- **docs/*.md** - Detailed guides and tutorials
- **examples/*.py** - Working code examples

### Documentation Standards

- **Clear and concise** - Avoid jargon
- **Complete** - Cover all parameters and return values
- **Accurate** - Keep docs in sync with code
- **Practical** - Include examples

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] No new warnings or errors
- [ ] Commit messages are clear

### PR Title Format

```
<type>: <description>

Examples:
feat: Add new image generation module
fix: Resolve memory leak in model loading
docs: Update MODULE_GUIDE with examples
test: Add tests for module conflict detection
refactor: Simplify module registration logic
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why this change is needed

## Changes
- Bullet point list of changes
- Another change

## Testing
How you tested the changes

## Screenshots (if applicable)
For UI changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. **Automated checks** run on PR
2. **Maintainer review** - May request changes
3. **Discussion** - Back-and-forth if needed
4. **Approval** - Once changes look good
5. **Merge** - Maintainer merges when ready

### After Merge

- Your contribution is live! 🎉
- You'll be added to contributors
- Consider sharing on social media

## Questions?

- **Issues:** Open a GitHub issue
- **Discussions:** Use GitHub Discussions
- **Email:** Check README for contact info

## Attribution

By contributing, you agree that your contributions will be licensed under the project's MIT License.

Your contributions will be credited in:
- Git commit history
- Contributors list
- Release notes (for significant contributions)

---

**Thank you for making ForgeAI better!** 🚀
