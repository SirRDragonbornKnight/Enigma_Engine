"""
Documentation Generator

Auto-generate documentation from Python source code.
Supports Markdown, HTML, and interactive API docs.

FILE: scripts/generate_docs.py
TYPE: Build/Documentation
MAIN CLASSES: DocGenerator, ClassDoc, ModuleDoc, DocFormatter
"""

import ast
import logging
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocFormat(Enum):
    """Output documentation formats."""
    MARKDOWN = "md"
    HTML = "html"
    JSON = "json"
    RST = "rst"


@dataclass
class ParameterDoc:
    """Documentation for a function/method parameter."""
    name: str
    type_hint: str = ""
    default: str = ""
    description: str = ""


@dataclass
class FunctionDoc:
    """Documentation for a function or method."""
    name: str
    signature: str
    docstring: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    returns: str = ""
    return_type: str = ""
    raises: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    line_number: int = 0


@dataclass
class ClassDoc:
    """Documentation for a class."""
    name: str
    docstring: str = ""
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    attributes: List[ParameterDoc] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_dataclass: bool = False
    is_enum: bool = False
    line_number: int = 0


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    name: str
    path: str
    docstring: str = ""
    classes: List[ClassDoc] = field(default_factory=list)
    functions: List[FunctionDoc] = field(default_factory=list)
    constants: List[ParameterDoc] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


class DocstringParser:
    """Parse docstrings in various formats (Google, NumPy, Sphinx)."""
    
    @staticmethod
    def parse(docstring: str) -> Dict[str, Any]:
        """
        Parse docstring into structured format.
        
        Returns dict with: description, args, returns, raises, examples
        """
        if not docstring:
            return {"description": "", "args": [], "returns": "", "raises": [], "examples": []}
        
        result = {
            "description": "",
            "args": [],
            "returns": "",
            "raises": [],
            "examples": []
        }
        
        lines = docstring.strip().split("\n")
        current_section = "description"
        current_content = []
        
        section_patterns = {
            r"^Args?:": "args",
            r"^Arguments?:": "args",
            r"^Parameters?:": "args",
            r"^Returns?:": "returns",
            r"^Raises?:": "raises",
            r"^Exceptions?:": "raises",
            r"^Examples?:": "examples",
            r"^Notes?:": "notes",
            r"^See Also:": "see_also"
        }
        
        for line in lines:
            stripped = line.strip()
            
            # Check for section headers
            new_section = None
            for pattern, section in section_patterns.items():
                if re.match(pattern, stripped, re.IGNORECASE):
                    new_section = section
                    break
            
            if new_section:
                # Save previous section
                DocstringParser._save_section(result, current_section, current_content)
                current_section = new_section
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        DocstringParser._save_section(result, current_section, current_content)
        
        return result
    
    @staticmethod
    def _save_section(result: Dict, section: str, content: List[str]):
        """Save parsed section content."""
        text = "\n".join(content).strip()
        
        if section == "description":
            result["description"] = text
        elif section == "args":
            result["args"] = DocstringParser._parse_args(text)
        elif section == "returns":
            result["returns"] = text
        elif section == "raises":
            result["raises"] = DocstringParser._parse_raises(text)
        elif section == "examples":
            result["examples"].append(text)
    
    @staticmethod
    def _parse_args(text: str) -> List[Dict]:
        """Parse argument documentation."""
        args = []
        
        # Pattern: name (type): description
        # or: name: description
        pattern = r"^\s*(\w+)\s*(?:\(([^)]+)\))?:\s*(.*)$"
        
        for line in text.split("\n"):
            match = re.match(pattern, line)
            if match:
                args.append({
                    "name": match.group(1),
                    "type": match.group(2) or "",
                    "description": match.group(3)
                })
        
        return args
    
    @staticmethod
    def _parse_raises(text: str) -> List[str]:
        """Parse raises documentation."""
        raises = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                raises.append(line)
        return raises


class ASTDocExtractor(ast.NodeVisitor):
    """Extract documentation from Python AST."""
    
    def __init__(self, source: str):
        self.source = source
        self.module_doc = ModuleDoc(name="", path="")
        self._current_class: Optional[ClassDoc] = None
    
    def extract(self, name: str, path: str) -> ModuleDoc:
        """Extract documentation from source."""
        self.module_doc = ModuleDoc(name=name, path=path)
        
        try:
            tree = ast.parse(self.source)
            self.module_doc.docstring = ast.get_docstring(tree) or ""
            self.visit(tree)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {path}: {e}")
        
        return self.module_doc
    
    def visit_Import(self, node: ast.Import):
        """Record imports."""
        for alias in node.names:
            self.module_doc.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Record from imports."""
        module = node.module or ""
        for alias in node.names:
            self.module_doc.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class documentation."""
        class_doc = ClassDoc(
            name=node.name,
            docstring=ast.get_docstring(node) or "",
            bases=[self._get_name(base) for base in node.bases],
            decorators=[self._get_decorator(d) for d in node.decorator_list],
            line_number=node.lineno
        )
        
        # Check for dataclass/enum
        for dec in class_doc.decorators:
            if "dataclass" in dec:
                class_doc.is_dataclass = True
        if any("Enum" in base for base in class_doc.bases):
            class_doc.is_enum = True
        
        # Process class body
        self._current_class = class_doc
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                method_doc = self._extract_function(item)
                class_doc.methods.append(method_doc)
            elif isinstance(item, ast.AnnAssign) and item.target:
                # Class attribute with annotation
                attr = ParameterDoc(
                    name=self._get_name(item.target),
                    type_hint=self._get_annotation(item.annotation)
                )
                class_doc.attributes.append(attr)
        
        self._current_class = None
        self.module_doc.classes.append(class_doc)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function documentation."""
        if self._current_class is None:
            func_doc = self._extract_function(node)
            self.module_doc.functions.append(func_doc)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Extract async function documentation."""
        if self._current_class is None:
            func_doc = self._extract_function(node)
            func_doc.is_async = True
            self.module_doc.functions.append(func_doc)
    
    def visit_Assign(self, node: ast.Assign):
        """Extract module-level constants."""
        if self._current_class is None:
            for target in node.targets:
                name = self._get_name(target)
                if name and name.isupper():
                    const = ParameterDoc(
                        name=name,
                        default=self._get_value(node.value)
                    )
                    self.module_doc.constants.append(const)
        self.generic_visit(node)
    
    def _extract_function(self, node) -> FunctionDoc:
        """Extract function/method documentation."""
        func_doc = FunctionDoc(
            name=node.name,
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node) or "",
            decorators=[self._get_decorator(d) for d in node.decorator_list],
            line_number=node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )
        
        # Check decorators
        for dec in func_doc.decorators:
            if "classmethod" in dec:
                func_doc.is_classmethod = True
            elif "staticmethod" in dec:
                func_doc.is_staticmethod = True
            elif "property" in dec:
                func_doc.is_property = True
        
        # Extract parameters
        for arg in node.args.args:
            param = ParameterDoc(
                name=arg.arg,
                type_hint=self._get_annotation(arg.annotation)
            )
            func_doc.parameters.append(param)
        
        # Return type
        if node.returns:
            func_doc.return_type = self._get_annotation(node.returns)
        
        return func_doc
    
    def _get_signature(self, node) -> str:
        """Get function signature string."""
        args = []
        
        # Regular args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation(arg.annotation)}"
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        sig = f"{node.name}({', '.join(args)})"
        
        if node.returns:
            sig += f" -> {self._get_annotation(node.returns)}"
        
        return sig
    
    def _get_name(self, node) -> str:
        """Get name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        return ""
    
    def _get_annotation(self, node) -> str:
        """Get type annotation string."""
        if node is None:
            return ""
        return self._get_name(node)
    
    def _get_decorator(self, node) -> str:
        """Get decorator string."""
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return self._get_name(node)
    
    def _get_value(self, node) -> str:
        """Get constant value as string."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.List):
            return "[...]"
        elif isinstance(node, ast.Dict):
            return "{...}"
        return ""


class DocFormatter:
    """Format documentation for output."""
    
    @staticmethod
    def to_markdown(module: ModuleDoc) -> str:
        """Format module documentation as Markdown."""
        lines = []
        
        # Module header
        lines.append(f"# {module.name}")
        lines.append("")
        
        if module.docstring:
            lines.append(module.docstring)
            lines.append("")
        
        # Classes
        if module.classes:
            lines.append("## Classes")
            lines.append("")
            
            for cls in module.classes:
                lines.append(f"### {cls.name}")
                lines.append("")
                
                if cls.bases:
                    lines.append(f"*Inherits from: {', '.join(cls.bases)}*")
                    lines.append("")
                
                if cls.docstring:
                    lines.append(cls.docstring)
                    lines.append("")
                
                # Attributes
                if cls.attributes:
                    lines.append("#### Attributes")
                    lines.append("")
                    for attr in cls.attributes:
                        lines.append(f"- `{attr.name}`: {attr.type_hint}")
                    lines.append("")
                
                # Methods
                if cls.methods:
                    lines.append("#### Methods")
                    lines.append("")
                    for method in cls.methods:
                        if method.name.startswith("_") and not method.name.startswith("__"):
                            continue  # Skip private methods
                        lines.append(f"##### `{method.signature}`")
                        lines.append("")
                        if method.docstring:
                            lines.append(method.docstring)
                            lines.append("")
        
        # Module-level functions
        if module.functions:
            lines.append("## Functions")
            lines.append("")
            
            for func in module.functions:
                if func.name.startswith("_"):
                    continue  # Skip private functions
                
                lines.append(f"### `{func.signature}`")
                lines.append("")
                if func.docstring:
                    lines.append(func.docstring)
                    lines.append("")
        
        # Constants
        if module.constants:
            lines.append("## Constants")
            lines.append("")
            for const in module.constants:
                lines.append(f"- `{const.name}` = {const.default}")
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def to_html(module: ModuleDoc) -> str:
        """Format module documentation as HTML."""
        md = DocFormatter.to_markdown(module)
        
        # Simple markdown to HTML conversion
        html = md
        
        # Headers
        html = re.sub(r"^##### (.+)$", r"<h5>\1</h5>", html, flags=re.MULTILINE)
        html = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        
        # Code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
        
        # Italic
        html = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html)
        
        # Lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        
        # Wrap in basic HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{module.name} Documentation</title>
    <style>
        body {{ font-family: system-ui; max-width: 900px; margin: 0 auto; padding: 20px; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; overflow-x: auto; }}
        h1 {{ border-bottom: 2px solid #333; }}
        h2 {{ border-bottom: 1px solid #ccc; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
        
        return html
    
    @staticmethod
    def to_json(module: ModuleDoc) -> str:
        """Format module documentation as JSON."""
        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        
        return json.dumps(dataclass_to_dict(module), indent=2)


class DocGenerator:
    """
    Main documentation generator.
    
    Scans Python files and generates documentation.
    """
    
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        format: DocFormat = DocFormat.MARKDOWN
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.format = format
        self.modules: List[ModuleDoc] = []
    
    def generate(
        self,
        include_private: bool = False,
        recursive: bool = True
    ) -> List[Path]:
        """
        Generate documentation for all Python files.
        
        Args:
            include_private: Include _private modules
            recursive: Scan subdirectories
        
        Returns:
            List of generated doc file paths
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        files = list(self.source_dir.glob(pattern))
        
        generated = []
        
        for file_path in files:
            if not include_private and file_path.name.startswith("_"):
                if file_path.name != "__init__.py":
                    continue
            
            # Extract documentation
            module_doc = self._extract_module(file_path)
            self.modules.append(module_doc)
            
            # Generate output
            output_path = self._generate_output(module_doc)
            if output_path:
                generated.append(output_path)
                logger.info(f"Generated: {output_path}")
        
        # Generate index
        index_path = self._generate_index()
        if index_path:
            generated.append(index_path)
        
        return generated
    
    def _extract_module(self, file_path: Path) -> ModuleDoc:
        """Extract documentation from a Python file."""
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        
        rel_path = file_path.relative_to(self.source_dir)
        module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        
        extractor = ASTDocExtractor(source)
        return extractor.extract(module_name, str(rel_path))
    
    def _generate_output(self, module: ModuleDoc) -> Optional[Path]:
        """Generate output file for a module."""
        if self.format == DocFormat.MARKDOWN:
            content = DocFormatter.to_markdown(module)
            ext = ".md"
        elif self.format == DocFormat.HTML:
            content = DocFormatter.to_html(module)
            ext = ".html"
        elif self.format == DocFormat.JSON:
            content = DocFormatter.to_json(module)
            ext = ".json"
        else:
            return None
        
        output_name = module.name.replace(".", "_") + ext
        output_path = self.output_dir / output_name
        
        output_path.write_text(content, encoding="utf-8")
        return output_path
    
    def _generate_index(self) -> Optional[Path]:
        """Generate documentation index."""
        if self.format == DocFormat.MARKDOWN:
            lines = ["# API Documentation", "", "## Modules", ""]
            
            for module in sorted(self.modules, key=lambda m: m.name):
                doc_file = module.name.replace(".", "_") + ".md"
                lines.append(f"- [{module.name}]({doc_file})")
                
                # Add class summary
                for cls in module.classes:
                    lines.append(f"  - {cls.name}")
            
            content = "\n".join(lines)
            output_path = self.output_dir / "index.md"
        
        elif self.format == DocFormat.HTML:
            lines = ["<h1>API Documentation</h1>", "<h2>Modules</h2>", "<ul>"]
            
            for module in sorted(self.modules, key=lambda m: m.name):
                doc_file = module.name.replace(".", "_") + ".html"
                lines.append(f'<li><a href="{doc_file}">{module.name}</a>')
                
                if module.classes:
                    lines.append("<ul>")
                    for cls in module.classes:
                        lines.append(f"<li>{cls.name}</li>")
                    lines.append("</ul>")
                
                lines.append("</li>")
            
            lines.append("</ul>")
            
            content = f"""<!DOCTYPE html>
<html>
<head><title>API Documentation</title></head>
<body>{"".join(lines)}</body>
</html>"""
            
            output_path = self.output_dir / "index.html"
        
        else:
            return None
        
        output_path.write_text(content, encoding="utf-8")
        return output_path


def generate_docs(
    source: str = "forge_ai",
    output: str = "docs/api",
    format: str = "md"
) -> List[Path]:
    """
    Generate documentation for ForgeAI.
    
    Args:
        source: Source directory
        output: Output directory
        format: Output format (md, html, json)
    
    Returns:
        List of generated files
    """
    source_path = Path(__file__).parent.parent / source
    output_path = Path(__file__).parent.parent / output
    
    format_map = {
        "md": DocFormat.MARKDOWN,
        "html": DocFormat.HTML,
        "json": DocFormat.JSON
    }
    
    generator = DocGenerator(
        source_path,
        output_path,
        format_map.get(format, DocFormat.MARKDOWN)
    )
    
    return generator.generate()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ForgeAI documentation")
    parser.add_argument("--source", default="forge_ai", help="Source directory")
    parser.add_argument("--output", default="docs/api", help="Output directory")
    parser.add_argument("--format", choices=["md", "html", "json"], default="md")
    
    args = parser.parse_args()
    
    files = generate_docs(args.source, args.output, args.format)
    print(f"Generated {len(files)} documentation files")


if __name__ == "__main__":
    main()
