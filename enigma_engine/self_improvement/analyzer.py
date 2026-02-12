"""
Code Analyzer for Self-Improvement System

Analyzes Python code to extract:
- Class definitions and their methods
- Function signatures and docstrings
- GUI elements (PyQt5 widgets)
- Module relationships
- Feature descriptions

This analysis is used to generate training data for the AI.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    path: str
    line_number: int
    signature: str
    docstring: str = ""
    decorators: List[str] = field(default_factory=list)
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: str = ""
    is_async: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "line_number": self.line_number,
            "signature": self.signature,
            "docstring": self.docstring,
            "decorators": self.decorators,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "is_async": self.is_async,
        }


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    path: str
    line_number: int
    docstring: str = ""
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "line_number": self.line_number,
            "docstring": self.docstring,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "decorators": self.decorators,
            "attributes": self.attributes,
        }


@dataclass
class GUIElementInfo:
    """Information about a GUI element."""
    name: str
    widget_type: str
    path: str
    line_number: int
    parent_class: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "widget_type": self.widget_type,
            "path": self.path,
            "line_number": self.line_number,
            "parent_class": self.parent_class,
            "properties": self.properties,
        }


@dataclass
class CodeChange:
    """Represents a detected code change."""
    path: str
    change_type: str  # "added", "modified", "removed"
    element_type: str  # "class", "function", "gui"
    element_info: Dict[str, Any] = field(default_factory=dict)


class ASTVisitor(ast.NodeVisitor):
    """AST visitor to extract code information."""
    
    def __init__(self, path: str):
        self.path = path
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []
        self.gui_elements: List[GUIElementInfo] = []
        self.imports: List[str] = []
        self._current_class: Optional[str] = None
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Create class info
        class_info = ClassInfo(
            name=node.name,
            path=self.path,
            line_number=node.lineno,
            docstring=docstring,
            bases=bases,
            decorators=decorators,
        )
        
        # Track current class for method extraction
        old_class = self._current_class
        self._current_class = node.name
        
        # Visit children to get methods
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self._extract_function(child, is_method=True)
                class_info.methods.append(method_info)
            elif isinstance(child, ast.Assign):
                # Extract class attributes
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        class_info.attributes.append(target.id)
        
        self._current_class = old_class
        self.classes.append(class_info)
        
        # Check for GUI elements
        self._check_gui_class(class_info)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self._current_class is None:
            # Module-level function
            func_info = self._extract_function(node, is_method=False)
            self.functions.append(func_info)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self._current_class is None:
            func_info = self._extract_function(node, is_method=False)
            func_info.is_async = True
            self.functions.append(func_info)
    
    def _extract_function(self, node, is_method: bool = False) -> FunctionInfo:
        """Extract function information from AST node."""
        # Get parameters
        parameters = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["type"] = self._get_annotation_str(arg.annotation)
            parameters.append(param)
        
        # Get return type
        return_type = ""
        if node.returns:
            return_type = self._get_annotation_str(node.returns)
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Build signature
        params_str = ", ".join(
            f"{p['name']}: {p.get('type', '')}" if p.get('type') else p['name']
            for p in parameters
        )
        ret_str = f" -> {return_type}" if return_type else ""
        signature = f"{node.name}({params_str}){ret_str}"
        
        return FunctionInfo(
            name=node.name,
            path=self.path,
            line_number=node.lineno,
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )
    
    def _check_gui_class(self, class_info: ClassInfo):
        """Check if class is a GUI widget and extract elements."""
        pyqt_bases = {"QWidget", "QMainWindow", "QDialog", "QFrame", 
                      "QPushButton", "QLabel", "QLineEdit", "QTextEdit",
                      "QComboBox", "QListWidget", "QTreeWidget", "QTableWidget"}
        
        if any(base in pyqt_bases for base in class_info.bases):
            gui_element = GUIElementInfo(
                name=class_info.name,
                widget_type=class_info.bases[0] if class_info.bases else "QWidget",
                path=self.path,
                line_number=class_info.line_number,
                properties={"methods": [m.name for m in class_info.methods]},
            )
            self.gui_elements.append(gui_element)
    
    def _get_decorator_name(self, node) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    
    def _get_annotation_str(self, node) -> str:
        """Convert annotation AST node to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_str(node.value)
            if isinstance(node.slice, ast.Tuple):
                slice_str = ", ".join(self._get_annotation_str(e) for e in node.slice.elts)
            else:
                slice_str = self._get_annotation_str(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.BinOp):
            # Union types in Python 3.10+
            left = self._get_annotation_str(node.left)
            right = self._get_annotation_str(node.right)
            return f"{left} | {right}"
        return ""


class CodeAnalyzer:
    """
    Analyzes Python code to extract features and changes.
    
    Usage:
        analyzer = CodeAnalyzer("/path/to/enigma_engine")
        result = analyzer.analyze()
        
        # Get specific file analysis
        file_analysis = analyzer.analyze_file("/path/to/file.py")
    """
    
    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        self._cache: Dict[str, Dict] = {}  # path -> analysis
        self._cache_path = self.engine_path.parent / "data" / "code_analysis_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load analysis cache from disk."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save analysis cache to disk."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def analyze(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze code and return extracted features.
        
        Args:
            paths: Specific paths to analyze, or None for full engine analysis
            
        Returns:
            Dict with classes, functions, gui_elements, etc.
        """
        all_classes = []
        all_functions = []
        all_gui_elements = []
        
        # Get files to analyze
        if paths:
            files = [Path(p) for p in paths if p.endswith('.py')]
        else:
            files = list(self.engine_path.rglob("*.py"))
            files = [f for f in files if "__pycache__" not in str(f)]
        
        # Analyze each file
        for file_path in files:
            try:
                analysis = self.analyze_file(str(file_path))
                all_classes.extend(analysis.get("classes", []))
                all_functions.extend(analysis.get("functions", []))
                all_gui_elements.extend(analysis.get("gui_elements", []))
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Find new elements (compare to cache)
        new_classes = self._find_new_elements(all_classes, "classes")
        new_functions = self._find_new_elements(all_functions, "functions")
        new_gui_elements = self._find_new_elements(all_gui_elements, "gui_elements")
        
        # Update cache
        self._update_cache(all_classes, all_functions, all_gui_elements)
        self._save_cache()
        
        return {
            "all_classes": all_classes,
            "all_functions": all_functions,
            "all_gui_elements": all_gui_elements,
            "new_classes": new_classes,
            "new_functions": new_functions,
            "new_gui_elements": new_gui_elements,
            "total_files": len(files),
        }
    
    def analyze_file(self, path: str) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            visitor = ASTVisitor(path)
            visitor.visit(tree)
            
            return {
                "path": path,
                "classes": [c.to_dict() for c in visitor.classes],
                "functions": [f.to_dict() for f in visitor.functions],
                "gui_elements": [g.to_dict() for g in visitor.gui_elements],
                "imports": visitor.imports,
            }
        except SyntaxError as e:
            logger.warning(f"Syntax error in {path}: {e}")
            return {"path": path, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to analyze {path}: {e}")
            return {"path": path, "error": str(e)}
    
    def _find_new_elements(self, elements: List[Dict], element_type: str) -> List[Dict]:
        """Find elements that are new (not in cache)."""
        cached = self._cache.get(element_type, {})
        new_elements = []
        
        for elem in elements:
            key = f"{elem.get('path')}:{elem.get('name')}"
            if key not in cached:
                new_elements.append(elem)
        
        return new_elements
    
    def _update_cache(self, classes: List[Dict], functions: List[Dict], gui_elements: List[Dict]):
        """Update the analysis cache."""
        self._cache["classes"] = {
            f"{c.get('path')}:{c.get('name')}": c for c in classes
        }
        self._cache["functions"] = {
            f"{f.get('path')}:{f.get('name')}": f for f in functions
        }
        self._cache["gui_elements"] = {
            f"{g.get('path')}:{g.get('name')}": g for g in gui_elements
        }
    
    def get_feature_summary(self, analysis: Dict) -> str:
        """Get a human-readable summary of features."""
        lines = []
        
        if analysis.get("new_classes"):
            lines.append("New Classes:")
            for cls in analysis["new_classes"][:10]:
                doc = cls.get("docstring", "").split("\n")[0][:80]
                lines.append(f"  - {cls['name']}: {doc or 'No description'}")
        
        if analysis.get("new_functions"):
            lines.append("\nNew Functions:")
            for func in analysis["new_functions"][:10]:
                lines.append(f"  - {func['signature']}")
        
        if analysis.get("new_gui_elements"):
            lines.append("\nNew GUI Elements:")
            for gui in analysis["new_gui_elements"][:10]:
                lines.append(f"  - {gui['name']} ({gui['widget_type']})")
        
        return "\n".join(lines) if lines else "No new features detected."


class FeatureExtractor:
    """
    Extracts high-level features from code for training data generation.
    
    Understands common patterns like:
    - Module manager integration
    - GUI tab implementation
    - Tool definitions
    - Training configurations
    """
    
    # Patterns for feature extraction
    PATTERNS = {
        "module_registration": r"MODULE_REGISTRY\[.*\]\s*=",
        "gui_tab": r"class\s+\w+Tab\(.*QWidget",
        "tool_definition": r"ToolDefinition\(",
        "training_config": r"TrainingConfig\(",
        "api_endpoint": r"@app\.route\(",
        "signal_slot": r"\.connect\(",
    }
    
    def __init__(self):
        self.features: Dict[str, List[Dict]] = {
            key: [] for key in self.PATTERNS
        }
    
    def extract_from_file(self, path: str) -> Dict[str, List[Dict]]:
        """Extract features from a file using regex patterns."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            results = {}
            
            for feature_name, pattern in self.PATTERNS.items():
                matches = []
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        matches.append({
                            "line_number": i,
                            "line_content": line.strip(),
                            "path": path,
                        })
                
                if matches:
                    results[feature_name] = matches
            
            return results
        except Exception as e:
            logger.error(f"Feature extraction failed for {path}: {e}")
            return {}
    
    def extract_from_analysis(self, analysis: Dict) -> Dict[str, Any]:
        """Extract high-level features from code analysis."""
        features = {
            "modules": [],
            "gui_components": [],
            "tools": [],
            "api_endpoints": [],
        }
        
        for cls in analysis.get("new_classes", []):
            # Check for module patterns
            if any("Module" in base for base in cls.get("bases", [])):
                features["modules"].append({
                    "name": cls["name"],
                    "docstring": cls.get("docstring", ""),
                    "methods": [m["name"] for m in cls.get("methods", [])],
                })
            
            # Check for GUI patterns
            if any(base in ["QWidget", "QMainWindow", "QDialog"] for base in cls.get("bases", [])):
                features["gui_components"].append({
                    "name": cls["name"],
                    "type": cls.get("bases", ["Unknown"])[0],
                    "docstring": cls.get("docstring", ""),
                })
        
        return features


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Analyzer")
    parser.add_argument("path", nargs="?", help="Path to analyze")
    parser.add_argument("--full", action="store_true", help="Full engine analysis")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Default to enigma_engine
    engine_path = args.path or str(Path(__file__).parent.parent)
    
    analyzer = CodeAnalyzer(engine_path)
    
    if args.full:
        result = analyzer.analyze()
    else:
        result = analyzer.analyze([args.path] if args.path else None)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(analyzer.get_feature_summary(result))
        print(f"\nTotal files analyzed: {result.get('total_files', 0)}")


if __name__ == "__main__":
    main()
