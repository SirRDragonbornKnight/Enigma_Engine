"""
================================================================================
Tab Consistency Analyzer - Ensure all tabs follow unified patterns.
================================================================================

This script analyzes all GUI tabs for consistency and generates a report with
suggestions for fixes. It checks for:

1. Base class usage (should inherit from BaseGenerationTab or use common patterns)
2. Style consistency (unified_patterns, button styles, etc.)
3. Error handling patterns
4. Thread/worker patterns for background operations
5. Device-aware configurations
6. Builtin fallback integration

USAGE:
    python -m enigma_engine.scripts.analyze_tabs
    
    # Or programmatically:
    from enigma_engine.scripts.analyze_tabs import TabAnalyzer
    analyzer = TabAnalyzer()
    report = analyzer.analyze_all()
    print(report)
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TabAnalysisResult:
    """Results of analyzing a single tab."""
    filepath: Path
    tab_name: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    patterns_used: set[str] = field(default_factory=set)
    base_classes: list[str] = field(default_factory=list)
    has_error_handling: bool = False
    has_worker_thread: bool = False
    has_device_awareness: bool = False
    has_builtin_fallback: bool = False
    score: int = 100  # Start at 100, deduct for issues


class TabAnalyzer:
    """Analyzes GUI tabs for consistency with coding standards."""
    
    def __init__(self, tabs_dir: Path = None):
        if tabs_dir is None:
            # Default to enigma_engine/gui/tabs
            script_dir = Path(__file__).parent
            tabs_dir = script_dir.parent / "gui" / "tabs"
        
        self.tabs_dir = tabs_dir
        self.results: dict[str, TabAnalysisResult] = {}
        
        # Expected patterns
        self.expected_imports = {
            "PyQt5.QtWidgets",
            "PyQt5.QtCore",
        }
        
        self.style_patterns = {
            "unified_patterns",
            "get_button_style",
            "get_style_config",
            "Colors",
        }
        
        self.base_tab_classes = {
            "BaseGenerationTab",
            "QWidget",
        }
        
        self.worker_patterns = {
            "QThread",
            "BaseGenerationWorker",
            "pyqtSignal",
        }
    
    def analyze_all(self) -> str:
        """Analyze all tabs and return a formatted report."""
        if not self.tabs_dir.exists():
            return f"Tabs directory not found: {self.tabs_dir}"
        
        # Find all Python files that define tabs
        tab_files = list(self.tabs_dir.glob("*_tab.py"))
        
        for filepath in tab_files:
            self._analyze_file(filepath)
        
        return self._generate_report()
    
    def _analyze_file(self, filepath: Path):
        """Analyze a single tab file."""
        tab_name = filepath.stem
        result = TabAnalysisResult(filepath=filepath, tab_name=tab_name)
        
        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Check imports
            self._check_imports(tree, source, result)
            
            # Check class definitions
            self._check_classes(tree, source, result)
            
            # Check for patterns
            self._check_patterns(source, result)
            
            # Calculate score
            result.score -= len(result.issues) * 10
            result.score -= len(result.warnings) * 5
            result.score = max(0, result.score)
            
        except SyntaxError as e:
            result.issues.append(f"Syntax error: {e}")
            result.score = 0
        except Exception as e:
            result.issues.append(f"Analysis error: {e}")
        
        self.results[tab_name] = result
    
    def _check_imports(self, tree: ast.AST, source: str, result: TabAnalysisResult):
        """Check import statements."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                for alias in node.names:
                    if node.module:
                        imports.add(f"{node.module}.{alias.name}")
                    else:
                        imports.add(alias.name)
        
        # Check for PyQt5
        has_pyqt = any("PyQt5" in imp for imp in imports)
        if not has_pyqt:
            result.warnings.append("No PyQt5 imports found - may be headless-only tab")
        
        # Check for unified patterns
        has_unified = any(p in source for p in ["unified_patterns", "get_button_style", "get_style_config"])
        if has_unified:
            result.patterns_used.add("unified_patterns")
        else:
            result.suggestions.append("Consider using unified_patterns for consistent styling")
        
        # Check for device profiles
        has_device = "device_profiles" in source or "DeviceClass" in source
        if has_device:
            result.has_device_awareness = True
            result.patterns_used.add("device_profiles")
        else:
            result.suggestions.append("Consider using device_profiles for cross-device compatibility")
        
        # Check for builtin fallbacks
        has_builtin = "builtin" in source.lower() or "BuiltinFallback" in source
        if has_builtin or "Builtin" in source:
            result.has_builtin_fallback = True
            result.patterns_used.add("builtin_fallbacks")
    
    def _check_classes(self, tree: ast.AST, source: str, result: TabAnalysisResult):
        """Check class definitions and inheritance."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Record base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        result.base_classes.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        result.base_classes.append(base.attr)
                
                # Check for Tab in name
                if "Tab" in node.name:
                    # Check if inherits from good base
                    if "BaseGenerationTab" in result.base_classes:
                        result.patterns_used.add("BaseGenerationTab")
                    elif "QWidget" in result.base_classes:
                        if node.name not in ["BaseGenerationTab"]:
                            result.suggestions.append(
                                f"{node.name} could inherit from BaseGenerationTab for consistency"
                            )
                    
                    # Check for try/except in methods
                    has_try = any(
                        isinstance(child, ast.Try)
                        for child in ast.walk(node)
                    )
                    if has_try:
                        result.has_error_handling = True
                        result.patterns_used.add("error_handling")
                
                # Check for Worker classes
                if "Worker" in node.name or "Thread" in node.name:
                    result.has_worker_thread = True
                    result.patterns_used.add("worker_threads")
        
        # Check for QThread usage
        if "QThread" in source:
            result.has_worker_thread = True
            result.patterns_used.add("QThread")
    
    def _check_patterns(self, source: str, result: TabAnalysisResult):
        """Check for common patterns in source."""
        
        # Check button styles
        if "BUTTON_STYLE_PRIMARY" in source or "get_button_style" in source:
            result.patterns_used.add("button_styles")
        elif "setStyleSheet" in source and "QPushButton" in source:
            result.warnings.append("Uses inline button styles - consider unified_patterns")
        
        # Check progress bar patterns
        if "QProgressBar" in source or "progress_bar" in source:
            result.patterns_used.add("progress_bar")
        
        # Check for hardcoded colors
        hardcoded_colors = source.count("#") - source.count("# ")  # Rough estimate
        if hardcoded_colors > 20:
            result.warnings.append(f"Found many hardcoded colors ({hardcoded_colors}+) - consider Colors class")
        
        # Check for tooltips (accessibility)
        if "setToolTip" in source:
            result.patterns_used.add("tooltips")
        else:
            result.suggestions.append("Add tooltips for better accessibility")
        
        # Check for gaming mode awareness
        if "gaming_mode" in source or "GamingMode" in source:
            result.patterns_used.add("gaming_mode")
        
        # Check for distributed/network awareness
        if "distributed" in source or "network" in source.lower():
            result.patterns_used.add("distributed")
    
    def _generate_report(self) -> str:
        """Generate a formatted analysis report."""
        lines = [
            "=" * 80,
            "TAB CONSISTENCY ANALYSIS REPORT",
            "=" * 80,
            f"\nAnalyzed {len(self.results)} tabs in {self.tabs_dir}\n",
        ]
        
        # Sort by score (worst first)
        sorted_results = sorted(
            self.results.values(),
            key=lambda r: r.score
        )
        
        # Summary statistics
        total_score = sum(r.score for r in sorted_results)
        avg_score = total_score / len(sorted_results) if sorted_results else 0
        
        lines.append(f"Average consistency score: {avg_score:.1f}/100")
        lines.append(f"Tabs with issues: {sum(1 for r in sorted_results if r.issues)}")
        lines.append(f"Tabs with warnings: {sum(1 for r in sorted_results if r.warnings)}")
        lines.append("")
        
        # Pattern usage summary
        all_patterns = set()
        for r in sorted_results:
            all_patterns.update(r.patterns_used)
        
        lines.append("Pattern usage across all tabs:")
        for pattern in sorted(all_patterns):
            count = sum(1 for r in sorted_results if pattern in r.patterns_used)
            pct = count / len(sorted_results) * 100
            lines.append(f"  {pattern}: {count}/{len(sorted_results)} ({pct:.0f}%)")
        lines.append("")
        
        # Individual tab reports
        lines.append("-" * 80)
        lines.append("INDIVIDUAL TAB ANALYSIS")
        lines.append("-" * 80)
        
        for result in sorted_results:
            lines.append(f"\n{result.tab_name} (Score: {result.score}/100)")
            lines.append(f"  Base classes: {', '.join(result.base_classes) or 'None'}")
            lines.append(f"  Patterns used: {', '.join(sorted(result.patterns_used)) or 'None'}")
            
            if result.issues:
                lines.append("  ISSUES:")
                for issue in result.issues:
                    lines.append(f"    - {issue}")
            
            if result.warnings:
                lines.append("  WARNINGS:")
                for warning in result.warnings:
                    lines.append(f"    - {warning}")
            
            if result.suggestions:
                lines.append("  SUGGESTIONS:")
                for suggestion in result.suggestions:
                    lines.append(f"    - {suggestion}")
        
        # Recommendations
        lines.append("\n" + "=" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 80)
        
        # Find tabs needing most work
        critical_tabs = [r for r in sorted_results if r.score < 70]
        if critical_tabs:
            lines.append("\nHigh priority tabs to update:")
            for r in critical_tabs[:5]:
                lines.append(f"  - {r.tab_name} (score: {r.score})")
        
        # Pattern recommendations
        if sum(1 for r in sorted_results if "unified_patterns" in r.patterns_used) < len(sorted_results) * 0.5:
            lines.append("\n- Many tabs don't use unified_patterns - consider updating")
        
        if sum(1 for r in sorted_results if r.has_device_awareness) < len(sorted_results) * 0.3:
            lines.append("- Device-awareness is low - important for Pi/phone/PC compatibility")
        
        if sum(1 for r in sorted_results if r.has_builtin_fallback) < len(sorted_results) * 0.3:
            lines.append("- Builtin fallback usage is low - needed for zero-dependency mode")
        
        return "\n".join(lines)
    
    def get_worst_tabs(self, n: int = 5) -> list[TabAnalysisResult]:
        """Get the n worst-scoring tabs."""
        return sorted(self.results.values(), key=lambda r: r.score)[:n]
    
    def get_issues_by_type(self) -> dict[str, list[str]]:
        """Group all issues by type."""
        issues_by_type: dict[str, list[str]] = {}
        
        for result in self.results.values():
            for issue in result.issues:
                issue_type = issue.split(":")[0] if ":" in issue else "Other"
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(f"{result.tab_name}: {issue}")
        
        return issues_by_type


def main():
    """Run tab analysis from command line."""
    print("Enigma AI Engine Tab Consistency Analyzer")
    print("-" * 40)
    
    # Find the tabs directory
    script_dir = Path(__file__).parent
    tabs_dir = script_dir.parent / "gui" / "tabs"
    
    if not tabs_dir.exists():
        # Try alternative path
        tabs_dir = Path("enigma_engine/gui/tabs")
    
    if not tabs_dir.exists():
        print(f"Error: Could not find tabs directory")
        print(f"Tried: {tabs_dir}")
        return 1
    
    analyzer = TabAnalyzer(tabs_dir)
    report = analyzer.analyze_all()
    print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
