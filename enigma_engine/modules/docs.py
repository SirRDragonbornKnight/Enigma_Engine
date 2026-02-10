"""
Enigma AI Engine Module Documentation Generator
====================================

Automatically generate documentation from module metadata.

Supports:
- Markdown format
- HTML format
- Dependency graphs (Mermaid)
"""

import logging
from datetime import datetime
from pathlib import Path

from .manager import ModuleCategory, ModuleInfo, ModuleManager

logger = logging.getLogger(__name__)


class ModuleDocGenerator:
    """Generate documentation from module metadata."""
    
    def __init__(self, manager: ModuleManager):
        """
        Initialize documentation generator.
        
        Args:
            manager: ModuleManager instance to generate docs from
        """
        self.manager = manager
    
    def generate_markdown(self, module_id: str) -> str:
        """
        Generate markdown documentation for a single module.
        
        Args:
            module_id: Module ID to document
            
        Returns:
            Markdown string with complete module documentation
        """
        if module_id not in self.manager.module_classes:
            return f"# Error\n\nModule '{module_id}' not found in registry."
        
        module_class = self.manager.module_classes[module_id]
        info = module_class.get_info()
        
        # Build markdown
        lines = []
        
        # Header
        lines.append(f"# {info.name}")
        lines.append("")
        lines.append(f"**Module ID:** `{info.id}`")
        lines.append(f"**Version:** {info.version}")
        lines.append(f"**Category:** {info.category.value}")
        lines.append("")
        
        # Description
        lines.append("## Description")
        lines.append("")
        lines.append(info.description)
        lines.append("")
        
        # Status
        lines.append("## Status")
        lines.append("")
        if module_id in self.manager.modules:
            module = self.manager.modules[module_id]
            lines.append(f"**State:** {module.state.value}")
            if info.load_time:
                lines.append(f"**Load Time:** {info.load_time.isoformat()}")
        else:
            lines.append("**State:** unloaded")
        lines.append("")
        
        # Dependencies
        if info.requires or info.optional:
            lines.append("## Dependencies")
            lines.append("")
            
            if info.requires:
                lines.append("### Required")
                lines.append("")
                for dep in info.requires:
                    lines.append(f"- `{dep}`")
                lines.append("")
            
            if info.optional:
                lines.append("### Optional")
                lines.append("")
                for dep in info.optional:
                    lines.append(f"- `{dep}`")
                lines.append("")
        
        # Conflicts
        if info.conflicts:
            lines.append("## Conflicts")
            lines.append("")
            lines.append("This module conflicts with:")
            lines.append("")
            for conflict in info.conflicts:
                lines.append(f"- `{conflict}`")
            lines.append("")
        
        # Capabilities
        if info.provides:
            lines.append("## Provides")
            lines.append("")
            lines.append("This module provides the following capabilities:")
            lines.append("")
            for capability in info.provides:
                lines.append(f"- `{capability}`")
            lines.append("")
        
        # Hardware Requirements
        if info.min_ram_mb or info.min_vram_mb or info.requires_gpu:
            lines.append("## Hardware Requirements")
            lines.append("")
            
            if info.min_ram_mb:
                lines.append(f"- **RAM:** {info.min_ram_mb} MB minimum")
            if info.min_vram_mb:
                lines.append(f"- **VRAM:** {info.min_vram_mb} MB minimum")
            if info.requires_gpu:
                lines.append(f"- **GPU:** Required")
            if info.supports_distributed:
                lines.append(f"- **Distributed:** Supported")
            lines.append("")
        
        # Cloud Service Warning
        if info.is_cloud_service:
            lines.append("## ⚠️ Cloud Service")
            lines.append("")
            lines.append(
                "This module connects to external cloud services. "
                "API keys and internet connection required. "
                "Data will be sent to third-party servers."
            )
            lines.append("")
        
        # Configuration
        if info.config_schema:
            lines.append("## Configuration")
            lines.append("")
            lines.append("| Parameter | Type | Default | Description |")
            lines.append("|-----------|------|---------|-------------|")
            
            for param_name, param_info in info.config_schema.items():
                param_type = param_info.get('type', 'any')
                default = param_info.get('default', 'N/A')
                
                # Format default value properly
                if default == 'N/A':
                    default_str = default
                elif isinstance(default, str):
                    # Escape for markdown
                    default_str = f'`{repr(default)}`'
                else:
                    # Use repr for proper representation
                    default_str = f'`{repr(default)}`'
                
                # Get options if available
                options = param_info.get('options', [])
                if options:
                    param_type = f"{param_type} ({', '.join(map(str, options))})"
                
                description = param_info.get('description', '')
                
                lines.append(f"| `{param_name}` | {param_type} | {default_str} | {description} |")
            
            lines.append("")
        
        # Usage Example
        lines.append("## Usage Example")
        lines.append("")
        lines.append("```python")
        lines.append("from enigma_engine.modules import ModuleManager")
        lines.append("")
        lines.append("manager = ModuleManager()")
        
        # Show dependencies loading if needed
        if info.requires:
            lines.append("")
            lines.append("# Load dependencies first")
            for dep in info.requires:
                lines.append(f"manager.load('{dep}')")
        
        lines.append("")
        lines.append(f"# Load the module")
        if info.config_schema:
            lines.append(f"config = {{}}")
            lines.append(f"manager.load('{info.id}', config)")
        else:
            lines.append(f"manager.load('{info.id}')")
        
        lines.append("")
        lines.append(f"# Use the module")
        lines.append(f"module = manager.get_module('{info.id}')")
        lines.append(f"interface = manager.get_interface('{info.id}')")
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_all_markdown(self) -> str:
        """
        Generate documentation for all registered modules.
        
        Returns:
            Markdown string with documentation for all modules
        """
        lines = []
        
        # Header
        lines.append("# Enigma AI Engine - Module Documentation")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().isoformat()}*")
        lines.append("")
        
        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(
            f"Total modules registered: {len(self.manager.module_classes)}"
        )
        lines.append(f"Total modules loaded: {len(self.manager.modules)}")
        lines.append("")
        
        # Table of Contents by Category
        lines.append("## Table of Contents")
        lines.append("")
        
        by_category: dict[ModuleCategory, list[ModuleInfo]] = {}
        for module_class in self.manager.module_classes.values():
            info = module_class.get_info()
            if info.category not in by_category:
                by_category[info.category] = []
            by_category[info.category].append(info)
        
        for category in sorted(by_category.keys(), key=lambda c: c.value):
            lines.append(f"### {category.value.upper()}")
            lines.append("")
            for info in sorted(by_category[category], key=lambda i: i.id):
                lines.append(f"- [{info.name}](#{info.id}) - {info.description}")
            lines.append("")
        
        # Individual module documentation
        lines.append("---")
        lines.append("")
        
        for module_id in sorted(self.manager.module_classes.keys()):
            module_doc = self.generate_markdown(module_id)
            lines.append(module_doc)
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_html(self, module_id: str) -> str:
        """
        Generate HTML documentation for a module.
        
        Args:
            module_id: Module ID to document
            
        Returns:
            HTML string
        """
        # Convert markdown to HTML (basic implementation)
        markdown = self.generate_markdown(module_id)
        
        # Simple markdown to HTML conversion
        html_lines = ['<!DOCTYPE html>', '<html>', '<head>',
                     '<meta charset="UTF-8">',
                     '<title>Module Documentation</title>',
                     '<style>',
                     'body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }',
                     'h1 { color: #333; border-bottom: 2px solid #666; }',
                     'h2 { color: #555; margin-top: 30px; }',
                     'code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }',
                     'pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }',
                     'table { border-collapse: collapse; width: 100%; }',
                     'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                     'th { background-color: #f2f2f2; }',
                     '</style>',
                     '</head>', '<body>']
        
        # Basic markdown to HTML (very simplified)
        in_code_block = False
        in_table = False
        
        for line in markdown.split('\n'):
            if line.startswith('```'):
                if in_code_block:
                    html_lines.append('</pre>')
                    in_code_block = False
                else:
                    html_lines.append('<pre><code>')
                    in_code_block = True
            elif in_code_block:
                html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            elif line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('- '):
                html_lines.append(f'<li>{line[2:]}</li>')
            elif line.startswith('|'):
                if not in_table:
                    html_lines.append('<table>')
                    in_table = True
                # Simple table row
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if all(c.startswith('-') for c in cells):
                    continue  # Skip separator row
                row = '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
                html_lines.append(row)
            else:
                if in_table and not line.startswith('|'):
                    html_lines.append('</table>')
                    in_table = False
                if line.strip():
                    html_lines.append(f'<p>{line}</p>')
        
        html_lines.extend(['</body>', '</html>'])
        return '\n'.join(html_lines)
    
    def generate_dependency_graph(self, format: str = 'mermaid') -> str:
        """
        Generate a dependency graph.
        
        Args:
            format: Output format ('mermaid' or 'graphviz')
            
        Returns:
            Graph definition string
        """
        if format == 'mermaid':
            return self._generate_mermaid_graph()
        elif format == 'graphviz':
            return self._generate_graphviz_graph()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_mermaid_graph(self) -> str:
        """Generate Mermaid diagram of module dependencies."""
        lines = ['graph TD']
        
        # Add all modules as nodes
        for module_id, module_class in self.manager.module_classes.items():
            info = module_class.get_info()
            # Use category for styling
            category_class = info.category.value.lower()
            lines.append(f'    {module_id}["{info.name}"]')
            lines.append(f'    class {module_id} {category_class}')
        
        lines.append('')
        
        # Add dependencies as edges
        for module_id, module_class in self.manager.module_classes.items():
            info = module_class.get_info()
            
            # Required dependencies (solid line)
            for dep in info.requires:
                lines.append(f'    {dep} --> {module_id}')
            
            # Optional dependencies (dashed line)
            for dep in info.optional:
                lines.append(f'    {dep} -.-> {module_id}')
            
            # Conflicts (red line)
            for conflict in info.conflicts:
                lines.append(f'    {module_id} ---|conflict| {conflict}')
        
        lines.append('')
        
        # Add styling
        lines.append('    classDef core fill:#e1f5ff,stroke:#0288d1')
        lines.append('    classDef generation fill:#fff3e0,stroke:#f57c00')
        lines.append('    classDef memory fill:#f3e5f5,stroke:#7b1fa2')
        lines.append('    classDef interface fill:#e8f5e9,stroke:#388e3c')
        lines.append('    classDef tools fill:#fce4ec,stroke:#c2185b')
        lines.append('    classDef network fill:#fff9c4,stroke:#f9a825')
        
        return '\n'.join(lines)
    
    def _generate_graphviz_graph(self) -> str:
        """Generate Graphviz DOT format graph."""
        lines = ['digraph ModuleDependencies {']
        lines.append('    rankdir=LR;')
        lines.append('    node [shape=box, style=rounded];')
        lines.append('')
        
        # Add nodes
        for module_id, module_class in self.manager.module_classes.items():
            info = module_class.get_info()
            lines.append(f'    {module_id} [label="{info.name}"];')
        
        lines.append('')
        
        # Add edges
        for module_id, module_class in self.manager.module_classes.items():
            info = module_class.get_info()
            
            for dep in info.requires:
                lines.append(f'    {dep} -> {module_id};')
            
            for dep in info.optional:
                lines.append(f'    {dep} -> {module_id} [style=dashed];')
            
            for conflict in info.conflicts:
                lines.append(
                    f'    {module_id} -> {conflict} '
                    f'[color=red, label="conflicts"];'
                )
        
        lines.append('}')
        return '\n'.join(lines)
    
    def export_to_file(self, path: Path, format: str = 'markdown'):
        """
        Export all documentation to a file.
        
        Args:
            path: Output file path
            format: Output format ('markdown', 'html', 'mermaid', 'graphviz')
        """
        path = Path(path)
        
        logger.info(f"Exporting documentation to {path} (format: {format})")
        
        if format == 'markdown':
            content = self.generate_all_markdown()
        elif format == 'html':
            # Generate HTML for all modules
            content = '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Module Documentation</title></head><body>'
            content += '<h1>Enigma AI Engine - All Modules</h1>'
            for module_id in sorted(self.manager.module_classes.keys()):
                content += self.generate_html(module_id)
                content += '<hr>'
            content += '</body></html>'
        elif format in ['mermaid', 'graphviz']:
            content = self.generate_dependency_graph(format)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Documentation exported to {path}")
