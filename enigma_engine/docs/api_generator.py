"""
API Reference Documentation Generator

Generates API documentation from docstrings and type hints.
Outputs Markdown, HTML, or structured JSON.

FILE: enigma_engine/docs/api_generator.py
TYPE: Documentation
MAIN CLASSES: APIDocGenerator, APIEndpoint, APISchema
"""

import importlib
import inspect
import json
import logging
import pkgutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, get_type_hints

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class Parameter:
    """API parameter documentation."""
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    location: str = "body"  # body, query, path, header
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": str(self.default) if self.default is not None else None,
            "location": self.location
        }


@dataclass
class Response:
    """API response documentation."""
    status_code: int
    description: str
    schema: Optional[dict] = None
    example: Optional[Any] = None
    
    def to_dict(self) -> dict:
        data = {
            "status_code": self.status_code,
            "description": self.description
        }
        if self.schema:
            data["schema"] = self.schema
        if self.example:
            data["example"] = self.example
        return data


@dataclass
class APIEndpoint:
    """Documentation for a single API endpoint."""
    path: str
    method: HTTPMethod
    summary: str
    description: str = ""
    parameters: list[Parameter] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    deprecated: bool = False
    auth_required: bool = True
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "method": self.method.value,
            "summary": self.summary,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "responses": [r.to_dict() for r in self.responses],
            "tags": self.tags,
            "deprecated": self.deprecated,
            "auth_required": self.auth_required
        }


@dataclass
class FunctionDoc:
    """Documentation for a function."""
    name: str
    module: str
    signature: str
    docstring: str
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str = ""
    return_description: str = ""
    raises: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "signature": self.signature,
            "docstring": self.docstring,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "return_description": self.return_description,
            "raises": self.raises,
            "examples": self.examples
        }


@dataclass
class ClassDoc:
    """Documentation for a class."""
    name: str
    module: str
    docstring: str
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionDoc] = field(default_factory=list)
    attributes: list[Parameter] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "docstring": self.docstring,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "attributes": [a.to_dict() for a in self.attributes]
        }


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    name: str
    path: str
    docstring: str
    classes: list[ClassDoc] = field(default_factory=list)
    functions: list[FunctionDoc] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "docstring": self.docstring,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions]
        }


class DocstringParser:
    """Parses docstrings in various formats."""
    
    @staticmethod
    def parse(docstring: str) -> dict[str, Any]:
        """Parse docstring into components."""
        if not docstring:
            return {"description": "", "params": [], "returns": "", "raises": [], "examples": []}
        
        result = {
            "description": "",
            "params": [],
            "returns": "",
            "raises": [],
            "examples": []
        }
        
        lines = docstring.strip().split("\n")
        current_section = "description"
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for section headers
            if line_stripped.lower().startswith("args:") or line_stripped.lower().startswith("parameters:"):
                if current_content:
                    result["description"] = "\n".join(current_content).strip()
                current_section = "params"
                current_content = []
            elif line_stripped.lower().startswith("returns:"):
                current_section = "returns"
                current_content = []
            elif line_stripped.lower().startswith("raises:") or line_stripped.lower().startswith("exceptions:"):
                current_section = "raises"
                current_content = []
            elif line_stripped.lower().startswith("example:") or line_stripped.lower().startswith("examples:"):
                current_section = "examples"
                current_content = []
            elif current_section == "params" and ":" in line_stripped:
                # Parse parameter
                parts = line_stripped.split(":", 1)
                name = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                result["params"].append({"name": name, "description": desc})
            elif current_section == "returns":
                current_content.append(line_stripped)
            elif current_section == "raises" and line_stripped:
                result["raises"].append(line_stripped)
            elif current_section == "examples":
                current_content.append(line)
            else:
                current_content.append(line_stripped)
        
        # Handle remaining content
        if current_section == "description":
            result["description"] = "\n".join(current_content).strip()
        elif current_section == "returns":
            result["returns"] = "\n".join(current_content).strip()
        elif current_section == "examples":
            result["examples"] = ["\n".join(current_content).strip()]
        
        return result


class APIDocGenerator:
    """Generates API documentation from Flask/FastAPI routes."""
    
    def __init__(self):
        """Initialize generator."""
        self._endpoints: list[APIEndpoint] = []
        self._modules: list[ModuleDoc] = []
    
    def scan_flask_app(self, app):
        """
        Scan a Flask app for routes.
        
        Args:
            app: Flask application instance
        """
        for rule in app.url_map.iter_rules():
            if rule.endpoint == 'static':
                continue
            
            view_func = app.view_functions.get(rule.endpoint)
            if not view_func:
                continue
            
            # Parse docstring
            docstring = view_func.__doc__ or ""
            parsed = DocstringParser.parse(docstring)
            
            # Get methods
            for method in rule.methods:
                if method in ('HEAD', 'OPTIONS'):
                    continue
                
                endpoint = APIEndpoint(
                    path=rule.rule,
                    method=HTTPMethod(method),
                    summary=parsed["description"].split("\n")[0] if parsed["description"] else rule.endpoint,
                    description=parsed["description"],
                    parameters=[
                        Parameter(name=p["name"], type="string", description=p["description"])
                        for p in parsed["params"]
                    ],
                    responses=[
                        Response(status_code=200, description=parsed["returns"] or "Success")
                    ]
                )
                
                self._endpoints.append(endpoint)
    
    def scan_module(self, module_path: str) -> ModuleDoc:
        """
        Scan a Python module for documentation.
        
        Args:
            module_path: Dotted module path
            
        Returns:
            Module documentation
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error(f"Failed to import {module_path}: {e}")
            return ModuleDoc(name=module_path, path="", docstring=f"Import error: {e}")
        
        mod_doc = ModuleDoc(
            name=module_path,
            path=getattr(module, '__file__', ''),
            docstring=module.__doc__ or ""
        )
        
        # Scan classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module_path:
                continue
            
            class_doc = self._document_class(name, obj, module_path)
            mod_doc.classes.append(class_doc)
        
        # Scan functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ != module_path:
                continue
            
            func_doc = self._document_function(name, obj, module_path)
            mod_doc.functions.append(func_doc)
        
        self._modules.append(mod_doc)
        return mod_doc
    
    def scan_package(self, package_path: str):
        """
        Scan an entire package recursively.
        
        Args:
            package_path: Dotted package path
        """
        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.error(f"Failed to import {package_path}: {e}")
            return
        
        # Scan the package itself
        self.scan_module(package_path)
        
        # Scan submodules
        if hasattr(package, '__path__'):
            for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package_path + "."):
                try:
                    self.scan_module(name)
                except Exception as e:
                    logger.warning(f"Failed to scan {name}: {e}")
    
    def _document_class(self, name: str, cls, module: str) -> ClassDoc:
        """Document a class."""
        class_doc = ClassDoc(
            name=name,
            module=module,
            docstring=cls.__doc__ or "",
            bases=[b.__name__ for b in cls.__bases__ if b.__name__ != 'object']
        )
        
        # Document methods
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if method_name.startswith('_') and method_name != '__init__':
                continue
            
            func_doc = self._document_function(method_name, method, module)
            class_doc.methods.append(func_doc)
        
        return class_doc
    
    def _document_function(self, name: str, func: Callable, module: str) -> FunctionDoc:
        """Document a function."""
        # Get signature
        sig = None
        try:
            sig = inspect.signature(func)
            sig_str = f"{name}{sig}"
        except (ValueError, TypeError):
            sig_str = f"{name}(...)"
        
        # Parse docstring
        parsed = DocstringParser.parse(func.__doc__ or "")
        
        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        
        def _get_type_name(type_hint) -> str:
            """Get string name for a type hint, handling generics."""
            if type_hint is None:
                return "None"
            if hasattr(type_hint, '__name__'):
                return type_hint.__name__
            # Handle generic types like Optional[str], List[int], etc.
            return str(type_hint).replace('typing.', '')
        
        # Build parameters
        params = []
        sig_params = sig.parameters.items() if sig is not None and hasattr(sig, 'parameters') else []
        for param_name, param in sig_params:
            if param_name == 'self':
                continue
            
            param_type = _get_type_name(hints.get(param_name, Any)) if param_name in hints else "Any"
            param_desc = ""
            
            for p in parsed["params"]:
                if p["name"] == param_name:
                    param_desc = p["description"]
                    break
            
            params.append(Parameter(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=param.default == inspect.Parameter.empty,
                default=None if param.default == inspect.Parameter.empty else param.default
            ))
        
        return FunctionDoc(
            name=name,
            module=module,
            signature=sig_str,
            docstring=parsed["description"],
            parameters=params,
            return_type=_get_type_name(hints.get('return', Any)) if 'return' in hints else "",
            return_description=parsed["returns"],
            raises=parsed["raises"],
            examples=parsed["examples"]
        )
    
    def export_markdown(self, output_path: Path):
        """
        Export documentation as Markdown.
        
        Args:
            output_path: Output file path
        """
        lines = []
        lines.append("# Enigma AI Engine API Reference\n")
        lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # API Endpoints
        if self._endpoints:
            lines.append("## API Endpoints\n")
            
            for endpoint in self._endpoints:
                lines.append(f"### {endpoint.method.value} {endpoint.path}\n")
                lines.append(f"{endpoint.summary}\n")
                
                if endpoint.description:
                    lines.append(f"\n{endpoint.description}\n")
                
                if endpoint.parameters:
                    lines.append("\n**Parameters:**\n")
                    for param in endpoint.parameters:
                        req = "required" if param.required else "optional"
                        lines.append(f"- `{param.name}` ({param.type}, {req}): {param.description}")
                    lines.append("")
                
                if endpoint.responses:
                    lines.append("\n**Responses:**\n")
                    for resp in endpoint.responses:
                        lines.append(f"- **{resp.status_code}**: {resp.description}")
                    lines.append("")
                
                lines.append("---\n")
        
        # Module Documentation
        if self._modules:
            lines.append("## Modules\n")
            
            for mod in self._modules:
                lines.append(f"### {mod.name}\n")
                
                if mod.docstring:
                    lines.append(f"{mod.docstring.strip()}\n")
                
                # Classes
                for cls in mod.classes:
                    lines.append(f"#### Class: {cls.name}\n")
                    if cls.docstring:
                        lines.append(f"{cls.docstring.strip()}\n")
                    
                    if cls.bases:
                        lines.append(f"*Bases: {', '.join(cls.bases)}*\n")
                    
                    for method in cls.methods:
                        lines.append(f"##### `{method.signature}`\n")
                        if method.docstring:
                            lines.append(f"{method.docstring}\n")
                        
                        if method.parameters:
                            lines.append("\n**Parameters:**")
                            for param in method.parameters:
                                default = f" = {param.default}" if param.default is not None else ""
                                lines.append(f"- `{param.name}: {param.type}{default}` - {param.description}")
                            lines.append("")
                        
                        if method.return_type:
                            lines.append(f"\n**Returns:** `{method.return_type}` - {method.return_description}\n")
                
                # Functions
                for func in mod.functions:
                    lines.append(f"#### `{func.signature}`\n")
                    if func.docstring:
                        lines.append(f"{func.docstring}\n")
                    
                    if func.parameters:
                        lines.append("\n**Parameters:**")
                        for param in func.parameters:
                            default = f" = {param.default}" if param.default is not None else ""
                            lines.append(f"- `{param.name}: {param.type}{default}` - {param.description}")
                        lines.append("")
                    
                    if func.return_type:
                        lines.append(f"\n**Returns:** `{func.return_type}` - {func.return_description}\n")
                
                lines.append("---\n")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
    
    def export_json(self, output_path: Path):
        """Export documentation as JSON."""
        data = {
            "generated_at": time.time(),
            "endpoints": [e.to_dict() for e in self._endpoints],
            "modules": [m.to_dict() for m in self._modules]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_openapi(self, output_path: Path, title: str = "Enigma AI Engine API", version: str = "1.0.0"):
        """Export as OpenAPI 3.0 spec."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": "Enigma AI Engine REST API"
            },
            "paths": {}
        }
        
        for endpoint in self._endpoints:
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}
            
            method = endpoint.method.value.lower()
            spec["paths"][endpoint.path][method] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
                "parameters": [
                    {
                        "name": p.name,
                        "in": p.location,
                        "required": p.required,
                        "schema": {"type": p.type},
                        "description": p.description
                    }
                    for p in endpoint.parameters
                    if p.location != "body"
                ],
                "responses": {
                    str(r.status_code): {
                        "description": r.description
                    }
                    for r in endpoint.responses
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)


def generate_docs(package: str, output_dir: Path):
    """
    Generate documentation for a package.
    
    Args:
        package: Package path to document
        output_dir: Output directory
    """
    generator = APIDocGenerator()
    generator.scan_package(package)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator.export_markdown(output_dir / "API_REFERENCE.md")
    generator.export_json(output_dir / "api_reference.json")


__all__ = [
    'APIDocGenerator',
    'APIEndpoint',
    'Parameter',
    'Response',
    'ModuleDoc',
    'ClassDoc',
    'FunctionDoc',
    'HTTPMethod',
    'DocstringParser',
    'generate_docs'
]
