"""
Plugin Templates

Starter templates for creating Enigma AI Engine plugins/extensions.
Includes base classes, scaffolding tools, and example plugins.

FILE: enigma_engine/plugins/templates.py
TYPE: Plugin Development Framework
MAIN CLASSES: PluginTemplate, PluginScaffold, ToolPlugin, TabPlugin, ThemePlugin
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Plugin Base Classes
# ============================================================================

@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    category: str = "general"
    requires: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    config_schema: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "requires": self.requires,
            "conflicts": self.conflicts,
            "config_schema": self.config_schema
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginMetadata':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PluginBase(ABC):
    """Base class for all Enigma AI Engine plugins."""
    
    def __init__(self):
        self._enabled = False
        self._config: dict = {}
        
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
    
    def on_load(self):
        """Called when plugin is loaded."""
    
    def on_unload(self):
        """Called when plugin is unloaded."""
    
    def on_enable(self):
        """Called when plugin is enabled."""
        self._enabled = True
        
    def on_disable(self):
        """Called when plugin is disabled."""
        self._enabled = False
        
    def configure(self, config: dict):
        """Configure the plugin."""
        self._config = config
        
    @property
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled


class ToolPlugin(PluginBase):
    """Base class for tool plugins."""
    
    @abstractmethod
    def get_tools(self) -> list[dict]:
        """
        Return list of tool definitions.
        
        Returns:
            List of tool definition dicts with name, description, parameters
        """
    
    @abstractmethod
    def execute(self, tool_name: str, parameters: dict) -> Any:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool result
        """


class TabPlugin(PluginBase):
    """Base class for GUI tab plugins."""
    
    @abstractmethod
    def get_tab_info(self) -> dict:
        """
        Return tab information.
        
        Returns:
            Dict with 'name', 'icon', 'tooltip'
        """
    
    @abstractmethod
    def create_widget(self, parent=None):
        """
        Create the tab widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            QWidget for the tab
        """


class ThemePlugin(PluginBase):
    """Base class for theme plugins."""
    
    @abstractmethod
    def get_theme(self) -> dict:
        """
        Return theme definition.
        
        Returns:
            Dict with color definitions
        """
    
    @abstractmethod
    def get_stylesheet(self) -> str:
        """
        Return theme stylesheet.
        
        Returns:
            CSS/QSS stylesheet string
        """


class ProcessorPlugin(PluginBase):
    """Base class for text/data processor plugins."""
    
    @abstractmethod
    def process_input(self, text: str) -> str:
        """Pre-process user input."""
    
    @abstractmethod
    def process_output(self, text: str) -> str:
        """Post-process AI output."""


# ============================================================================
# Plugin Scaffolding
# ============================================================================

class PluginScaffold:
    """Generates plugin project scaffolding."""
    
    @staticmethod
    def create_plugin(
        name: str,
        plugin_type: str,
        output_dir: Path,
        author: str = "",
        description: str = ""
    ) -> Path:
        """
        Create a new plugin project from template.
        
        Args:
            name: Plugin name (lowercase, no spaces)
            plugin_type: "tool", "tab", "theme", or "processor"
            output_dir: Directory to create plugin in
            author: Plugin author
            description: Plugin description
            
        Returns:
            Path to created plugin directory
        """
        plugin_dir = output_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_content = PLUGIN_TEMPLATES.get(plugin_type, PLUGIN_TEMPLATES["tool"])
        init_content = init_content.format(
            name=name,
            class_name=name.title().replace("_", ""),
            author=author,
            description=description
        )
        
        (plugin_dir / "__init__.py").write_text(init_content)
        
        # Create plugin.json metadata
        metadata = {
            "name": name,
            "version": "1.0.0",
            "author": author,
            "description": description,
            "type": plugin_type,
            "entry_point": f"{name}.{name.title().replace('_', '')}Plugin"
        }
        (plugin_dir / "plugin.json").write_text(json.dumps(metadata, indent=2))
        
        # Create README
        readme = f"""# {name.replace('_', ' ').title()}

{description}

## Installation

Copy this folder to `enigma_engine/plugins/{name}/`

## Usage

The plugin will be automatically loaded when Enigma AI Engine starts.

## Configuration

Edit `plugin.json` to configure the plugin.

## Author

{author}
"""
        (plugin_dir / "README.md").write_text(readme)
        
        logger.info(f"Created plugin scaffold at {plugin_dir}")
        return plugin_dir


# ============================================================================
# Plugin Templates
# ============================================================================

PLUGIN_TEMPLATES = {
    "tool": '''"""
{name} - A tool plugin for Enigma AI Engine

{description}

Author: {author}
"""

from typing import Dict, List, Any
from enigma_engine.plugins.templates import ToolPlugin, PluginMetadata


class {class_name}Plugin(ToolPlugin):
    """Tool plugin implementation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            category="tools"
        )
    
    def get_tools(self) -> List[Dict]:
        """Define available tools."""
        return [
            {{
                "name": "{name}_action",
                "description": "Perform an action with {name}",
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        "input": {{
                            "type": "string",
                            "description": "Input to process"
                        }}
                    }},
                    "required": ["input"]
                }}
            }}
        ]
    
    def execute(self, tool_name: str, parameters: Dict) -> Any:
        """Execute a tool."""
        if tool_name == "{name}_action":
            input_text = parameters.get("input", "")
            # TODO: Implement your tool logic here
            return {{"result": f"Processed: {{input_text}}"}}
        
        return {{"error": f"Unknown tool: {{tool_name}}"}}


# Export the plugin class
Plugin = {class_name}Plugin
''',

    "tab": '''"""
{name} - A tab plugin for Enigma AI Engine

{description}

Author: {author}
"""

from typing import Dict
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from enigma_engine.plugins.templates import TabPlugin, PluginMetadata


class {class_name}Plugin(TabPlugin):
    """Tab plugin implementation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            category="ui"
        )
    
    def get_tab_info(self) -> Dict:
        """Return tab information."""
        return {{
            "name": "{name}".replace("_", " ").title(),
            "icon": None,  # Optional: QIcon or path
            "tooltip": "{description}"
        }}
    
    def create_widget(self, parent=None) -> QWidget:
        """Create the tab widget."""
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        
        # Header
        header = QLabel("<h2>{name}</h2>")
        layout.addWidget(header)
        
        # Description
        desc = QLabel("{description}")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # TODO: Add your tab content here
        
        # Example button
        btn = QPushButton("Click Me")
        btn.clicked.connect(lambda: print("{name} button clicked!"))
        layout.addWidget(btn)
        
        layout.addStretch()
        return widget


# Export the plugin class
Plugin = {class_name}Plugin
''',

    "theme": '''"""
{name} - A theme plugin for Enigma AI Engine

{description}

Author: {author}
"""

from typing import Dict
from enigma_engine.plugins.templates import ThemePlugin, PluginMetadata


class {class_name}Plugin(ThemePlugin):
    """Theme plugin implementation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            category="themes"
        )
    
    def get_theme(self) -> Dict:
        """Return theme color definitions."""
        return {{
            "name": "{name}",
            "colors": {{
                "background": "#1a1a2e",
                "background_alt": "#16213e",
                "surface": "#0f3460",
                "surface_hover": "#1a5276",
                "primary": "#e94560",
                "secondary": "#0f4c75",
                "accent": "#3282b8",
                "text": "#eaeaea",
                "text_secondary": "#a0a0a0",
                "border": "#2c3e50",
                "selection": "rgba(233, 69, 96, 0.3)",
                "success": "#2ecc71",
                "warning": "#f39c12",
                "error": "#e74c3c",
                "highlight": "#ff6b6b"
            }}
        }}
    
    def get_stylesheet(self) -> str:
        """Return custom stylesheet additions."""
        theme = self.get_theme()
        c = theme["colors"]
        
        return f"""
            /* {name} Theme */
            QMainWindow {{
                background-color: {{c["background"]}};
            }}
            
            QPushButton {{
                background-color: {{c["primary"]}};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }}
            
            QPushButton:hover {{
                background-color: {{c["highlight"]}};
            }}
            
            /* Add more custom styles here */
        """


# Export the plugin class
Plugin = {class_name}Plugin
''',

    "processor": '''"""
{name} - A processor plugin for Enigma AI Engine

{description}

Author: {author}
"""

from enigma_engine.plugins.templates import ProcessorPlugin, PluginMetadata


class {class_name}Plugin(ProcessorPlugin):
    """Processor plugin implementation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            category="processors"
        )
    
    def process_input(self, text: str) -> str:
        """
        Pre-process user input before sending to AI.
        
        Args:
            text: Raw user input
            
        Returns:
            Processed input
        """
        # TODO: Implement input processing
        # Example: Add context, fix typos, expand abbreviations
        return text
    
    def process_output(self, text: str) -> str:
        """
        Post-process AI output before displaying to user.
        
        Args:
            text: Raw AI output
            
        Returns:
            Processed output
        """
        # TODO: Implement output processing
        # Example: Format code, add links, translate
        return text


# Export the plugin class
Plugin = {class_name}Plugin
'''
}


# ============================================================================
# Example Plugins
# ============================================================================

class ExampleToolPlugin(ToolPlugin):
    """Example tool plugin demonstrating the API."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_tool",
            version="1.0.0",
            author="Enigma AI Engine",
            description="Example tool plugin for reference",
            category="examples"
        )
    
    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "example_echo",
                "description": "Echo back the input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo"
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "example_reverse",
                "description": "Reverse a string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to reverse"
                        }
                    },
                    "required": ["text"]
                }
            }
        ]
    
    def execute(self, tool_name: str, parameters: dict) -> Any:
        if tool_name == "example_echo":
            return {"echo": parameters.get("message", "")}
        elif tool_name == "example_reverse":
            text = parameters.get("text", "")
            return {"reversed": text[::-1]}
        return {"error": f"Unknown tool: {tool_name}"}


def create_plugin(name: str, 
                  plugin_type: str = "tool",
                  output_dir: str = "plugins",
                  **kwargs) -> Path:
    """
    Convenience function to create a plugin.
    
    Args:
        name: Plugin name
        plugin_type: "tool", "tab", "theme", or "processor"
        output_dir: Output directory
        **kwargs: Additional metadata (author, description)
        
    Returns:
        Path to created plugin
    """
    return PluginScaffold.create_plugin(
        name=name,
        plugin_type=plugin_type,
        output_dir=Path(output_dir),
        author=kwargs.get("author", ""),
        description=kwargs.get("description", "")
    )


__all__ = [
    'PluginBase',
    'PluginMetadata',
    'ToolPlugin',
    'TabPlugin',
    'ThemePlugin',
    'ProcessorPlugin',
    'PluginScaffold',
    'ExampleToolPlugin',
    'create_plugin',
    'PLUGIN_TEMPLATES'
]
