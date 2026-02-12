"""
Code Style Analyzer - Learn code style from user's codebase.

This module analyzes source code to extract style patterns:
- Indentation (tabs vs spaces, size)
- Naming conventions (camelCase, snake_case)
- Documentation style (docstrings, comments)
- Code structure patterns
- Common imports and patterns

The extracted style guide is used to generate code matching the user's preferences.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class CodeStyleGuide:
    """Extracted code style preferences."""
    
    # Indentation
    indent_style: str = "spaces"  # "spaces" or "tabs"
    indent_size: int = 4
    
    # Naming conventions
    function_style: str = "snake_case"  # snake_case, camelCase, PascalCase
    variable_style: str = "snake_case"
    class_style: str = "PascalCase"
    constant_style: str = "UPPER_SNAKE_CASE"
    
    # Documentation
    docstring_style: str = "google"  # google, numpy, sphinx, none
    has_type_hints: bool = True
    comment_density: float = 0.1  # Comments per line of code
    
    # Imports
    import_style: str = "grouped"  # grouped, alphabetical, mixed
    common_imports: List[str] = field(default_factory=list)
    
    # Structure
    avg_function_length: int = 20
    max_line_length: int = 88
    uses_blank_lines: bool = True
    
    # Language-specific
    language: str = "python"
    framework_hints: List[str] = field(default_factory=list)
    
    def to_prompt_context(self) -> str:
        """Convert style guide to prompt context for code generation."""
        lines = [
            "Follow this code style:",
            f"- Use {self.indent_size} {self.indent_style} for indentation",
            f"- Function names: {self.function_style}",
            f"- Variable names: {self.variable_style}",
            f"- Class names: {self.class_style}",
            f"- Constants: {self.constant_style}",
        ]
        
        if self.has_type_hints:
            lines.append("- Include type hints for parameters and return values")
        
        if self.docstring_style != "none":
            lines.append(f"- Use {self.docstring_style}-style docstrings")
        
        if self.max_line_length:
            lines.append(f"- Keep lines under {self.max_line_length} characters")
        
        if self.common_imports:
            lines.append(f"- Common imports to use: {', '.join(self.common_imports[:5])}")
        
        if self.framework_hints:
            lines.append(f"- Framework context: {', '.join(self.framework_hints)}")
        
        return "\n".join(lines)


class CodeStyleAnalyzer:
    """Analyze code style from source files."""
    
    # File extensions to analyze
    EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".java": "java",
        ".go": "go",
    }
    
    def __init__(self):
        self.style_guide = CodeStyleGuide()
        self._analyzed_files: Set[str] = set()
    
    def analyze_directory(self, directory: Path, max_files: int = 100) -> CodeStyleGuide:
        """
        Analyze all source files in a directory to extract code style.
        
        Args:
            directory: Path to the codebase
            max_files: Maximum number of files to analyze
            
        Returns:
            CodeStyleGuide with extracted preferences
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return self.style_guide
        
        files_analyzed = 0
        all_stats = []
        
        for ext, lang in self.EXTENSIONS.items():
            for file_path in directory.rglob(f"*{ext}"):
                if files_analyzed >= max_files:
                    break
                
                # Skip common non-user directories
                if any(skip in str(file_path) for skip in [
                    "__pycache__", "node_modules", ".git", "venv", ".venv",
                    "build", "dist", "target", "vendor"
                ]):
                    continue
                
                try:
                    stats = self._analyze_file(file_path, lang)
                    if stats:
                        all_stats.append(stats)
                        files_analyzed += 1
                except Exception as e:
                    logger.debug(f"Error analyzing {file_path}: {e}")
        
        if all_stats:
            self._aggregate_stats(all_stats)
        
        logger.info(f"Analyzed {files_analyzed} files from {directory}")
        return self.style_guide
    
    def analyze_file(self, file_path: Path) -> CodeStyleGuide:
        """Analyze a single file."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        lang = self.EXTENSIONS.get(ext, "text")
        
        stats = self._analyze_file(file_path, lang)
        if stats:
            self._aggregate_stats([stats])
        
        return self.style_guide
    
    def _analyze_file(self, file_path: Path, language: str) -> Optional[Dict]:
        """Extract statistics from a single file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return None
        
        lines = content.split('\n')
        if len(lines) < 5:  # Skip very small files
            return None
        
        stats = {
            "language": language,
            "lines": len(lines),
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "indents": [],
            "names": {"functions": [], "variables": [], "classes": [], "constants": []},
            "imports": [],
            "max_line_length": 0,
            "has_type_hints": False,
            "docstring_style": None,
            "frameworks": set(),
        }
        
        for line in lines:
            stripped = line.strip()
            
            # Track line types
            if not stripped:
                stats["blank_lines"] += 1
            elif self._is_comment(stripped, language):
                stats["comment_lines"] += 1
            else:
                stats["code_lines"] += 1
            
            # Track indentation
            if stripped and line != stripped:
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    stats["indents"].append((indent, '\t' in line[:indent]))
            
            # Track line length
            stats["max_line_length"] = max(stats["max_line_length"], len(line))
        
        # Analyze language-specific patterns
        if language == "python":
            self._analyze_python(content, stats)
        elif language in ("javascript", "typescript"):
            self._analyze_javascript(content, stats)
        elif language == "rust":
            self._analyze_rust(content, stats)
        elif language == "java":
            self._analyze_java(content, stats)
        
        return stats
    
    def _is_comment(self, line: str, language: str) -> bool:
        """Check if a line is a comment."""
        if language in ("python", "rust"):
            return line.startswith("#") or line.startswith("//")
        elif language in ("javascript", "typescript", "java", "cpp", "go"):
            return line.startswith("//") or line.startswith("/*") or line.startswith("*")
        return False
    
    def _analyze_python(self, content: str, stats: Dict):
        """Analyze Python-specific patterns."""
        # Type hints
        stats["has_type_hints"] = bool(re.search(r'def\s+\w+\([^)]*:\s*\w+', content))
        
        # Docstring style
        if '"""' in content or "'''" in content:
            if re.search(r'""".*Args:', content, re.DOTALL):
                stats["docstring_style"] = "google"
            elif re.search(r'""".*Parameters\s*\n\s*-+', content, re.DOTALL):
                stats["docstring_style"] = "numpy"
            elif re.search(r'""".*:param\s+\w+:', content, re.DOTALL):
                stats["docstring_style"] = "sphinx"
            else:
                stats["docstring_style"] = "simple"
        
        # Function names
        for match in re.findall(r'def\s+(\w+)\s*\(', content):
            if match != "__init__":
                stats["names"]["functions"].append(match)
        
        # Class names
        for match in re.findall(r'class\s+(\w+)\s*[\(:]', content):
            stats["names"]["classes"].append(match)
        
        # Constants (UPPER_CASE at module level)
        for match in re.findall(r'^([A-Z][A-Z_0-9]+)\s*=', content, re.MULTILINE):
            stats["names"]["constants"].append(match)
        
        # Variable names (simple assignment)
        for match in re.findall(r'^\s*(\w+)\s*=', content, re.MULTILINE):
            if not match.isupper() and not match.startswith('_'):
                stats["names"]["variables"].append(match)
        
        # Imports
        for match in re.findall(r'^(?:from\s+(\S+)|import\s+(\S+))', content, re.MULTILINE):
            imp = match[0] or match[1]
            if imp:
                stats["imports"].append(imp.split('.')[0])
        
        # Framework detection
        framework_markers = {
            "flask": ["Flask", "flask"],
            "django": ["django", "Django"],
            "fastapi": ["FastAPI", "fastapi"],
            "pytorch": ["torch", "nn.Module"],
            "tensorflow": ["tensorflow", "tf."],
            "numpy": ["numpy", "np."],
            "pandas": ["pandas", "pd."],
            "pytest": ["pytest", "@pytest"],
            "asyncio": ["asyncio", "async def"],
            "qt": ["PyQt", "PySide", "QWidget"],
        }
        
        for framework, markers in framework_markers.items():
            if any(marker in content for marker in markers):
                stats["frameworks"].add(framework)
    
    def _analyze_javascript(self, content: str, stats: Dict):
        """Analyze JavaScript/TypeScript patterns."""
        # Type hints (TypeScript)
        stats["has_type_hints"] = bool(re.search(r':\s*(string|number|boolean|any)\b', content))
        
        # Function names
        for match in re.findall(r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\()', content):
            name = match[0] or match[1]
            if name:
                stats["names"]["functions"].append(name)
        
        # Class names
        for match in re.findall(r'class\s+(\w+)', content):
            stats["names"]["classes"].append(match)
        
        # Imports
        for match in re.findall(r"(?:import.*from\s+['\"]([^'\"]+)|require\s*\(['\"]([^'\"]+))", content):
            imp = match[0] or match[1]
            if imp:
                stats["imports"].append(imp.split('/')[0])
        
        # Framework detection
        framework_markers = {
            "react": ["React", "useState", "useEffect", "jsx"],
            "vue": ["Vue", "ref(", "reactive("],
            "angular": ["@Component", "@Injectable"],
            "express": ["express()", "app.get(", "app.post("],
            "node": ["require(", "process.", "Buffer"],
            "typescript": [": string", ": number", "interface "],
        }
        
        for framework, markers in framework_markers.items():
            if any(marker in content for marker in markers):
                stats["frameworks"].add(framework)
    
    def _analyze_rust(self, content: str, stats: Dict):
        """Analyze Rust patterns."""
        # Function names
        for match in re.findall(r'fn\s+(\w+)', content):
            stats["names"]["functions"].append(match)
        
        # Struct/enum names
        for match in re.findall(r'(?:struct|enum)\s+(\w+)', content):
            stats["names"]["classes"].append(match)
        
        # Framework detection
        if "tokio" in content:
            stats["frameworks"].add("tokio")
        if "serde" in content:
            stats["frameworks"].add("serde")
        if "actix" in content:
            stats["frameworks"].add("actix")
    
    def _analyze_java(self, content: str, stats: Dict):
        """Analyze Java patterns."""
        # Method names  
        for match in re.findall(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', content):
            stats["names"]["functions"].append(match)
        
        # Class names
        for match in re.findall(r'class\s+(\w+)', content):
            stats["names"]["classes"].append(match)
        
        # Framework detection
        if "@SpringBoot" in content or "springframework" in content:
            stats["frameworks"].add("spring")
    
    def _aggregate_stats(self, all_stats: List[Dict]):
        """Aggregate statistics from multiple files into style guide."""
        if not all_stats:
            return
        
        # Language (most common)
        languages = [s["language"] for s in all_stats]
        self.style_guide.language = Counter(languages).most_common(1)[0][0]
        
        # Indentation
        all_indents = []
        for s in all_stats:
            all_indents.extend(s.get("indents", []))
        
        if all_indents:
            # Count tabs vs spaces
            tabs = sum(1 for _, is_tab in all_indents if is_tab)
            spaces = len(all_indents) - tabs
            self.style_guide.indent_style = "tabs" if tabs > spaces else "spaces"
            
            # Common indent size
            space_indents = [size for size, is_tab in all_indents if not is_tab]
            if space_indents:
                # Find most common indent increment
                increments = []
                for size in space_indents:
                    if 2 <= size <= 8:
                        increments.append(size)
                if increments:
                    self.style_guide.indent_size = Counter(increments).most_common(1)[0][0]
        
        # Naming conventions
        all_functions = []
        all_variables = []
        all_classes = []
        
        for s in all_stats:
            all_functions.extend(s.get("names", {}).get("functions", []))
            all_variables.extend(s.get("names", {}).get("variables", []))
            all_classes.extend(s.get("names", {}).get("classes", []))
        
        self.style_guide.function_style = self._detect_naming_style(all_functions)
        self.style_guide.variable_style = self._detect_naming_style(all_variables)
        self.style_guide.class_style = self._detect_naming_style(all_classes)
        
        # Type hints
        type_hints = [s.get("has_type_hints", False) for s in all_stats]
        self.style_guide.has_type_hints = sum(type_hints) > len(type_hints) / 2
        
        # Docstring style
        docstrings = [s.get("docstring_style") for s in all_stats if s.get("docstring_style")]
        if docstrings:
            self.style_guide.docstring_style = Counter(docstrings).most_common(1)[0][0]
        
        # Comment density
        total_code = sum(s.get("code_lines", 0) for s in all_stats)
        total_comments = sum(s.get("comment_lines", 0) for s in all_stats)
        if total_code > 0:
            self.style_guide.comment_density = total_comments / total_code
        
        # Line length
        max_lengths = [s.get("max_line_length", 80) for s in all_stats]
        # Use 95th percentile as typical max
        max_lengths.sort()
        idx = int(len(max_lengths) * 0.95)
        self.style_guide.max_line_length = max_lengths[idx] if max_lengths else 88
        
        # Common imports
        all_imports = []
        for s in all_stats:
            all_imports.extend(s.get("imports", []))
        if all_imports:
            self.style_guide.common_imports = [
                imp for imp, _ in Counter(all_imports).most_common(10)
            ]
        
        # Frameworks
        all_frameworks = set()
        for s in all_stats:
            all_frameworks.update(s.get("frameworks", set()))
        self.style_guide.framework_hints = list(all_frameworks)
    
    def _detect_naming_style(self, names: List[str]) -> str:
        """Detect the dominant naming convention."""
        if not names:
            return "snake_case"
        
        styles = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "kebab-case": 0,
        }
        
        for name in names:
            if "_" in name:
                if name.isupper():
                    continue  # Constants
                styles["snake_case"] += 1
            elif "-" in name:
                styles["kebab-case"] += 1
            elif name[0].isupper() and any(c.islower() for c in name):
                styles["PascalCase"] += 1
            elif name[0].islower() and any(c.isupper() for c in name):
                styles["camelCase"] += 1
            else:
                styles["snake_case"] += 1  # Default/simple names
        
        return max(styles, key=styles.get)


# Global analyzer instance
_analyzer: Optional[CodeStyleAnalyzer] = None


def get_style_analyzer() -> CodeStyleAnalyzer:
    """Get or create the global style analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodeStyleAnalyzer()
    return _analyzer


def analyze_project(directory: str) -> CodeStyleGuide:
    """Convenience function to analyze a project directory."""
    analyzer = get_style_analyzer()
    return analyzer.analyze_directory(Path(directory))


def get_style_context(directory: Optional[str] = None) -> str:
    """Get style context string for code generation prompts."""
    analyzer = get_style_analyzer()
    if directory:
        analyzer.analyze_directory(Path(directory), max_files=50)
    return analyzer.style_guide.to_prompt_context()


@dataclass
class ProjectContext:
    """Project structure and context for code generation."""
    
    name: str = ""
    root_path: str = ""
    structure: List[str] = field(default_factory=list)  # Key files/folders
    entry_points: List[str] = field(default_factory=list)  # main.py, index.js, etc.
    packages: List[str] = field(default_factory=list)  # Detected packages
    readme_summary: str = ""  # Summary from README if present
    
    def to_prompt_context(self) -> str:
        """Convert to prompt context for code generation."""
        lines = ["Project context:"]
        
        if self.name:
            lines.append(f"- Project: {self.name}")
        
        if self.structure:
            lines.append(f"- Key structure: {', '.join(self.structure[:8])}")
        
        if self.entry_points:
            lines.append(f"- Entry points: {', '.join(self.entry_points)}")
        
        if self.packages:
            lines.append(f"- Dependencies: {', '.join(self.packages[:10])}")
        
        if self.readme_summary:
            lines.append(f"- Purpose: {self.readme_summary[:200]}")
        
        return "\n".join(lines)


class ProjectContextExtractor:
    """Extract project context from a codebase."""
    
    # Important files to detect project type
    PROJECT_FILES = {
        "python": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"],
        "javascript": ["package.json", "tsconfig.json", "webpack.config.js"],
        "rust": ["Cargo.toml"],
        "java": ["pom.xml", "build.gradle"],
        "go": ["go.mod"],
    }
    
    # Entry point patterns by language
    ENTRY_POINTS = {
        "python": ["main.py", "app.py", "run.py", "__main__.py", "cli.py"],
        "javascript": ["index.js", "index.ts", "main.js", "app.js", "server.js"],
        "rust": ["main.rs", "lib.rs"],
        "java": ["Main.java", "Application.java", "App.java"],
        "go": ["main.go"],
    }
    
    def __init__(self):
        self.context = ProjectContext()
    
    def analyze(self, directory: Path, max_depth: int = 3) -> ProjectContext:
        """
        Analyze a project directory to extract context.
        
        Args:
            directory: Project root path
            max_depth: How deep to scan the directory tree
            
        Returns:
            ProjectContext with extracted information
        """
        directory = Path(directory)
        if not directory.exists():
            return self.context
        
        self.context = ProjectContext(
            name=directory.name,
            root_path=str(directory)
        )
        
        # Detect project type and read metadata
        self._detect_project_type(directory)
        
        # Extract key structure
        self._extract_structure(directory, max_depth)
        
        # Find entry points
        self._find_entry_points(directory)
        
        # Read README summary
        self._read_readme(directory)
        
        return self.context
    
    def _detect_project_type(self, directory: Path):
        """Detect project type and extract dependencies."""
        # Check for Python project
        pyproject = directory / "pyproject.toml"
        requirements = directory / "requirements.txt"
        setup_py = directory / "setup.py"
        
        if pyproject.exists():
            self._parse_pyproject(pyproject)
        elif requirements.exists():
            self._parse_requirements(requirements)
        elif setup_py.exists():
            self.context.packages.append("setuptools")
        
        # Check for JavaScript/Node project
        package_json = directory / "package.json"
        if package_json.exists():
            self._parse_package_json(package_json)
        
        # Check for Rust project
        cargo_toml = directory / "Cargo.toml"
        if cargo_toml.exists():
            self._parse_cargo_toml(cargo_toml)
    
    def _parse_pyproject(self, path: Path):
        """Parse pyproject.toml for dependencies."""
        try:
            import tomllib
            content = path.read_text(encoding='utf-8')
            data = tomllib.loads(content)
            
            # Get project name
            if "project" in data:
                self.context.name = data["project"].get("name", self.context.name)
                deps = data["project"].get("dependencies", [])
                self.context.packages.extend([d.split()[0] for d in deps[:15]])
                
        except Exception:
            # Fallback to basic parsing
            try:
                content = path.read_text(encoding='utf-8')
                for line in content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        pkg = line.split('=')[0].strip().strip('"').strip("'")
                        if pkg and not pkg.startswith('['):
                            self.context.packages.append(pkg)
            except Exception:
                pass  # Intentionally silent
    
    def _parse_requirements(self, path: Path):
        """Parse requirements.txt for dependencies."""
        try:
            content = path.read_text(encoding='utf-8')
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before version specifier)
                    pkg = re.split(r'[<>=!~\[]', line)[0].strip()
                    if pkg:
                        self.context.packages.append(pkg)
        except Exception:
            pass  # Intentionally silent
    
    def _parse_package_json(self, path: Path):
        """Parse package.json for dependencies."""
        try:
            import json
            data = json.loads(path.read_text(encoding='utf-8'))
            
            self.context.name = data.get("name", self.context.name)
            
            # Get dependencies
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
            
            self.context.packages.extend(list(deps.keys())[:10])
            self.context.packages.extend(list(dev_deps.keys())[:5])
        except Exception:
            pass  # Intentionally silent
    
    def _parse_cargo_toml(self, path: Path):
        """Parse Cargo.toml for dependencies."""
        try:
            content = path.read_text(encoding='utf-8')
            in_deps = False
            
            for line in content.split('\n'):
                if '[dependencies]' in line:
                    in_deps = True
                    continue
                elif line.startswith('[') and in_deps:
                    in_deps = False
                elif in_deps and '=' in line:
                    pkg = line.split('=')[0].strip()
                    if pkg:
                        self.context.packages.append(pkg)
        except Exception:
            pass  # Intentionally silent
    
    def _extract_structure(self, directory: Path, max_depth: int):
        """Extract key directory structure."""
        important_dirs = []
        important_files = []
        
        def scan(path: Path, depth: int):
            if depth > max_depth:
                return
            
            try:
                for item in path.iterdir():
                    # Skip hidden and common ignored directories
                    if item.name.startswith('.'):
                        continue
                    if item.name in ['__pycache__', 'node_modules', 'venv', '.venv', 
                                     'build', 'dist', 'target', '.git', 'vendor']:
                        continue
                    
                    rel_path = item.relative_to(directory)
                    
                    if item.is_dir():
                        important_dirs.append(str(rel_path))
                        if depth < max_depth:
                            scan(item, depth + 1)
                    elif item.suffix in ['.py', '.js', '.ts', '.rs', '.go', '.java']:
                        important_files.append(str(rel_path))
            except PermissionError:
                pass  # Intentionally silent
        
        scan(directory, 0)
        
        # Combine and limit
        self.context.structure = important_dirs[:6] + important_files[:4]
    
    def _find_entry_points(self, directory: Path):
        """Find likely entry point files."""
        for lang, entries in self.ENTRY_POINTS.items():
            for entry in entries:
                entry_path = directory / entry
                if entry_path.exists():
                    self.context.entry_points.append(entry)
                
                # Also check in src/
                src_entry = directory / "src" / entry
                if src_entry.exists():
                    self.context.entry_points.append(f"src/{entry}")
    
    def _read_readme(self, directory: Path):
        """Read and summarize README if present."""
        readme_names = ["README.md", "README.txt", "README.rst", "README"]
        
        for name in readme_names:
            readme_path = directory / name
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding='utf-8', errors='ignore')
                    # Get first meaningful paragraph
                    lines = content.split('\n')
                    summary_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        # Skip headers and empty lines
                        if line.startswith('#') or not line:
                            if summary_lines:
                                break
                            continue
                        summary_lines.append(line)
                        if len(' '.join(summary_lines)) > 200:
                            break
                    
                    self.context.readme_summary = ' '.join(summary_lines)[:300]
                    break
                except Exception:
                    pass  # Intentionally silent


# Global extractor instance
_extractor: Optional[ProjectContextExtractor] = None


def get_project_extractor() -> ProjectContextExtractor:
    """Get or create the global project context extractor."""
    global _extractor
    if _extractor is None:
        _extractor = ProjectContextExtractor()
    return _extractor


def extract_project_context(directory: str) -> ProjectContext:
    """Convenience function to extract project context."""
    extractor = get_project_extractor()
    return extractor.analyze(Path(directory))


def get_full_code_context(directory: str) -> str:
    """Get both style and project context for code generation."""
    style_analyzer = get_style_analyzer()
    style_guide = style_analyzer.analyze_directory(Path(directory), max_files=30)
    
    extractor = get_project_extractor()
    project_ctx = extractor.analyze(Path(directory))
    
    parts = []
    if project_ctx.name:
        parts.append(project_ctx.to_prompt_context())
    parts.append(style_guide.to_prompt_context())
    
    return "\n\n".join(parts)
