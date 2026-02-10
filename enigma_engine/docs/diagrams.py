"""
Architecture Diagrams

Generates Mermaid and PlantUML diagrams from code structure.
Creates visual architecture documentation automatically.

FILE: enigma_engine/docs/diagrams.py
TYPE: Documentation
MAIN CLASSES: DiagramGenerator, MermaidGenerator, PlantUMLGenerator
"""

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DiagramType(Enum):
    """Types of diagrams."""
    CLASS = "class"
    SEQUENCE = "sequence"
    FLOWCHART = "flowchart"
    COMPONENT = "component"
    ER = "entity_relationship"


class OutputFormat(Enum):
    """Diagram output formats."""
    MERMAID = "mermaid"
    PLANTUML = "plantuml"
    DOT = "dot"


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    module: str
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    docstring: str = ""


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    path: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship between classes."""
    source: str
    target: str
    type: str  # inherits, contains, uses, depends
    label: str = ""


class CodeAnalyzer:
    """Analyzes Python code structure."""
    
    def __init__(self, root_path: Path):
        """
        Initialize code analyzer.
        
        Args:
            root_path: Root path of the codebase
        """
        self._root = Path(root_path)
    
    def analyze_file(self, file_path: Path) -> ModuleInfo:
        """
        Analyze a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            ModuleInfo with extracted information
        """
        with open(file_path, encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return ModuleInfo(name=file_path.stem, path=str(file_path))
        
        module_name = file_path.stem
        rel_path = file_path.relative_to(self._root)
        
        module = ModuleInfo(
            name=module_name,
            path=str(rel_path)
        )
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class(node)
                class_info.module = module_name
                module.classes.append(class_info)
            
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                module.functions.append(node.name)
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module.imports.append(node.module)
        
        return module
    
    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class information from AST node."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attributes.append(item.target.id)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        docstring = ast.get_docstring(node) or ""
        
        return ClassInfo(
            name=node.name,
            module="",
            bases=bases,
            methods=methods,
            attributes=attributes,
            docstring=docstring
        )
    
    def analyze_package(self, package_path: Path) -> list[ModuleInfo]:
        """
        Analyze all Python files in a package.
        
        Args:
            package_path: Path to package directory
            
        Returns:
            List of ModuleInfo objects
        """
        modules = []
        
        for py_file in package_path.rglob("*.py"):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            
            try:
                module = self.analyze_file(py_file)
                modules.append(module)
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
        
        return modules
    
    def find_relationships(self, modules: list[ModuleInfo]) -> list[Relationship]:
        """
        Find relationships between classes.
        
        Args:
            modules: List of analyzed modules
            
        Returns:
            List of relationships
        """
        relationships = []
        all_classes = {cls.name: cls for mod in modules for cls in mod.classes}
        
        for module in modules:
            for cls in module.classes:
                # Inheritance relationships
                for base in cls.bases:
                    if base in all_classes:
                        relationships.append(Relationship(
                            source=cls.name,
                            target=base,
                            type="inherits"
                        ))
                
                # Usage relationships (simplistic - from method names and attributes)
                for method in cls.methods:
                    for other_name in all_classes:
                        if other_name.lower() in method.lower():
                            relationships.append(Relationship(
                                source=cls.name,
                                target=other_name,
                                type="uses"
                            ))
        
        return relationships


class MermaidGenerator:
    """Generates Mermaid diagrams."""
    
    def __init__(self):
        """Initialize Mermaid generator."""
    
    def class_diagram(self, modules: list[ModuleInfo],
                      relationships: list[Relationship] = None) -> str:
        """
        Generate a class diagram.
        
        Args:
            modules: List of module information
            relationships: Optional relationships
            
        Returns:
            Mermaid diagram string
        """
        lines = ["classDiagram"]
        
        for module in modules:
            for cls in module.classes:
                # Class definition
                lines.append(f"    class {cls.name} {{")
                
                for attr in cls.attributes[:5]:  # Limit attributes
                    lines.append(f"        +{attr}")
                
                for method in cls.methods[:5]:  # Limit methods
                    if method.startswith("_"):
                        lines.append(f"        -{method}()")
                    else:
                        lines.append(f"        +{method}()")
                
                lines.append("    }")
        
        # Add relationships
        if relationships:
            for rel in relationships:
                if rel.type == "inherits":
                    lines.append(f"    {rel.target} <|-- {rel.source}")
                elif rel.type == "contains":
                    lines.append(f"    {rel.source} *-- {rel.target}")
                elif rel.type == "uses":
                    lines.append(f"    {rel.source} ..> {rel.target}")
        
        return "\n".join(lines)
    
    def flowchart(self, title: str, steps: list[tuple[str, str, str]]) -> str:
        """
        Generate a flowchart.
        
        Args:
            title: Diagram title
            steps: List of (id, label, next_id) tuples
            
        Returns:
            Mermaid diagram string
        """
        lines = ["flowchart TD"]
        
        for step_id, label, next_id in steps:
            # Define node
            lines.append(f"    {step_id}[{label}]")
            
            # Add connection if specified
            if next_id:
                lines.append(f"    {step_id} --> {next_id}")
        
        return "\n".join(lines)
    
    def sequence_diagram(self, participants: list[str],
                        messages: list[tuple[str, str, str]]) -> str:
        """
        Generate a sequence diagram.
        
        Args:
            participants: List of participant names
            messages: List of (from, to, message) tuples
            
        Returns:
            Mermaid diagram string
        """
        lines = ["sequenceDiagram"]
        
        for participant in participants:
            lines.append(f"    participant {participant}")
        
        for sender, receiver, message in messages:
            lines.append(f"    {sender}->>+{receiver}: {message}")
        
        return "\n".join(lines)
    
    def component_diagram(self, components: dict[str, list[str]]) -> str:
        """
        Generate a component diagram.
        
        Args:
            components: Dict mapping group name to component list
            
        Returns:
            Mermaid diagram string
        """
        lines = ["flowchart TB"]
        
        for group_name, group_components in components.items():
            lines.append(f"    subgraph {group_name}")
            for comp in group_components:
                safe_id = comp.replace(" ", "_").replace("-", "_")
                lines.append(f"        {safe_id}[{comp}]")
            lines.append("    end")
        
        return "\n".join(lines)


class PlantUMLGenerator:
    """Generates PlantUML diagrams."""
    
    def __init__(self):
        """Initialize PlantUML generator."""
    
    def class_diagram(self, modules: list[ModuleInfo],
                      relationships: list[Relationship] = None) -> str:
        """Generate PlantUML class diagram."""
        lines = ["@startuml"]
        
        for module in modules:
            lines.append(f"package {module.name} {{")
            
            for cls in module.classes:
                lines.append(f"    class {cls.name} {{")
                
                for attr in cls.attributes[:5]:
                    lines.append(f"        +{attr}")
                
                for method in cls.methods[:5]:
                    visibility = "-" if method.startswith("_") else "+"
                    lines.append(f"        {visibility}{method}()")
                
                lines.append("    }")
            
            lines.append("}")
        
        # Relationships
        if relationships:
            for rel in relationships:
                if rel.type == "inherits":
                    lines.append(f"{rel.target} <|-- {rel.source}")
                elif rel.type == "contains":
                    lines.append(f"{rel.source} *-- {rel.target}")
                elif rel.type == "uses":
                    lines.append(f"{rel.source} ..> {rel.target}")
        
        lines.append("@enduml")
        return "\n".join(lines)
    
    def component_diagram(self, components: dict[str, list[str]]) -> str:
        """Generate PlantUML component diagram."""
        lines = ["@startuml"]
        
        for group_name, group_components in components.items():
            lines.append(f"package \"{group_name}\" {{")
            for comp in group_components:
                lines.append(f"    [{comp}]")
            lines.append("}")
        
        lines.append("@enduml")
        return "\n".join(lines)


class DiagramGenerator:
    """Main diagram generator coordinating analysis and output."""
    
    def __init__(self, root_path: Path, output_format: OutputFormat = OutputFormat.MERMAID):
        """
        Initialize diagram generator.
        
        Args:
            root_path: Root path of the codebase
            output_format: Output format for diagrams
        """
        self._root = Path(root_path)
        self._format = output_format
        self._analyzer = CodeAnalyzer(root_path)
        
        self._generators = {
            OutputFormat.MERMAID: MermaidGenerator(),
            OutputFormat.PLANTUML: PlantUMLGenerator()
        }
    
    def generate_class_diagram(self, package_path: Path = None) -> str:
        """
        Generate a class diagram for the codebase.
        
        Args:
            package_path: Specific package (uses root if None)
            
        Returns:
            Diagram string in configured format
        """
        path = package_path or self._root
        modules = self._analyzer.analyze_package(path)
        relationships = self._analyzer.find_relationships(modules)
        
        generator = self._generators.get(self._format)
        return generator.class_diagram(modules, relationships)
    
    def generate_architecture_diagram(self) -> str:
        """
        Generate high-level architecture diagram.
        
        Returns:
            Diagram string
        """
        # Enigma AI Engine architecture components
        components = {
            "Core": ["Model", "Tokenizer", "Training", "Inference"],
            "Generation": ["Image Gen", "Code Gen", "Video Gen", "Audio Gen"],
            "Memory": ["Conversation", "Vector DB", "Embeddings"],
            "Interface": ["GUI", "API Server", "Web Dashboard"],
            "Tools": ["Vision", "Web", "File", "Robot"]
        }
        
        generator = self._generators.get(self._format)
        
        if isinstance(generator, MermaidGenerator):
            return generator.component_diagram(components)
        else:
            return generator.component_diagram(components)
    
    def generate_all(self, output_dir: Path) -> dict[str, Path]:
        """
        Generate all architecture diagrams.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dict mapping diagram name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ext = "md" if self._format == OutputFormat.MERMAID else "puml"
        files = {}
        
        # Architecture overview
        arch_diagram = self.generate_architecture_diagram()
        arch_path = output_dir / f"architecture.{ext}"
        self._save_diagram(arch_diagram, arch_path)
        files["architecture"] = arch_path
        
        # Class diagrams per major package
        if (self._root / "enigma_engine").exists():
            for subdir in (self._root / "enigma_engine").iterdir():
                if subdir.is_dir() and not subdir.name.startswith("_"):
                    try:
                        diagram = self.generate_class_diagram(subdir)
                        path = output_dir / f"classes_{subdir.name}.{ext}"
                        self._save_diagram(diagram, path)
                        files[f"classes_{subdir.name}"] = path
                    except Exception as e:
                        logger.warning(f"Failed to generate diagram for {subdir}: {e}")
        
        return files
    
    def _save_diagram(self, diagram: str, path: Path):
        """Save diagram to file."""
        if self._format == OutputFormat.MERMAID:
            content = f"```mermaid\n{diagram}\n```"
        else:
            content = diagram
        
        path.write_text(content)
        logger.info(f"Saved diagram to {path}")


def generate_diagrams(root_path: Path = None, output_dir: Path = None,
                      format: str = "mermaid") -> dict[str, Path]:
    """
    Generate all architecture diagrams.
    
    Args:
        root_path: Codebase root
        output_dir: Output directory
        format: Output format ("mermaid" or "plantuml")
        
    Returns:
        Dict of generated files
    """
    root_path = root_path or Path(".")
    output_dir = output_dir or Path("docs/diagrams")
    
    fmt = OutputFormat.MERMAID if format == "mermaid" else OutputFormat.PLANTUML
    
    generator = DiagramGenerator(root_path, fmt)
    return generator.generate_all(output_dir)


__all__ = [
    'DiagramGenerator',
    'DiagramType',
    'OutputFormat',
    'MermaidGenerator',
    'PlantUMLGenerator',
    'CodeAnalyzer',
    'ClassInfo',
    'ModuleInfo',
    'Relationship',
    'generate_diagrams'
]
