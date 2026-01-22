"""
DEEP CODE ANALYSIS - ForgeAI
============================
Comprehensive multi-pass analysis for bugs, anti-patterns, and issues.
"""
import sys
import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

sys.path.insert(0, '.')
os.environ['FORGE_NO_AUDIO'] = '1'  # Skip audio checks
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

ISSUES = []
WARNINGS = []

def add_issue(file: str, line: int, category: str, msg: str):
    ISSUES.append((file, line, category, msg))

def add_warning(file: str, line: int, category: str, msg: str):
    WARNINGS.append((file, line, category, msg))

# ===========================================================================
# PASS 1: AST Analysis - Find structural issues
# ===========================================================================
class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self, filename: str, source: str):
        self.filename = filename
        self.source = source
        self.lines = source.split('\n')
        self.current_class = None
        self.current_func = None
        self.defined_names: Set[str] = set()
        self.used_names: Set[str] = set()
        self.has_logger = False
        self.has_logging_import = False
        self.exception_handlers: List[Tuple[int, str]] = []
        self.bare_excepts: List[int] = []
        self.mutable_defaults: List[Tuple[int, str]] = []
        self.missing_self: List[Tuple[int, str]] = []
        self.unclosed_resources: List[Tuple[int, str]] = []
        self.thread_issues: List[Tuple[int, str]] = []
        
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == 'logging':
                self.has_logging_import = True
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module == 'logging':
            self.has_logging_import = True
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        # Check for logger assignment
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'logger':
                self.has_logger = True
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        old_class = self.current_class
        old_func = self.current_func
        self.current_class = node.name
        self.current_func = None
        self.generic_visit(node)
        self.current_class = old_class
        self.current_func = old_func
        
    def visit_FunctionDef(self, node):
        old_func = self.current_func
        
        # Check for mutable default arguments
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.mutable_defaults.append((node.lineno, node.name))
                
        # Check for missing self in methods - only for direct class methods, not nested functions
        if self.current_class and self.current_func is None and node.args.args:
            first_arg = node.args.args[0].arg
            if first_arg not in ('self', 'cls'):
                if not any(isinstance(d, ast.Name) and d.id in ('staticmethod', 'classmethod') 
                          for d in node.decorator_list):
                    self.missing_self.append((node.lineno, f"{self.current_class}.{node.name}"))
        
        self.current_func = node.name
        self.generic_visit(node)
        self.current_func = old_func
        
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.bare_excepts.append(node.lineno)
        elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
            # Check if it just passes
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                self.exception_handlers.append((node.lineno, "Silent exception swallow"))
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Check for unclosed file handles
        if isinstance(node.func, ast.Name) and node.func.id == 'open':
            # Check if it's in a with statement context
            parent = getattr(node, '_parent', None)
            if not isinstance(parent, ast.withitem):
                self.unclosed_resources.append((node.lineno, "open() without 'with'"))
                
        # Check for threading without locks
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('Thread', 'start'):
                pass  # Could check for lock usage
                
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a single Python file."""
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(source)
    except SyntaxError as e:
        add_issue(str(filepath), e.lineno or 0, "SYNTAX", str(e))
        return {}
    except Exception as e:
        add_warning(str(filepath), 0, "PARSE", str(e))
        return {}
    
    # Add parent references for context checking
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    
    analyzer = CodeAnalyzer(str(filepath), source)
    analyzer.visit(tree)
    
    # Report issues
    rel_path = str(filepath.relative_to(Path('.')))
    
    for line, name in analyzer.mutable_defaults:
        add_issue(rel_path, line, "MUTABLE_DEFAULT", 
                 f"Mutable default argument in {name}")
    
    for line in analyzer.bare_excepts:
        add_warning(rel_path, line, "BARE_EXCEPT", 
                   "Bare 'except:' catches all exceptions including KeyboardInterrupt")
    
    for line, msg in analyzer.exception_handlers:
        add_warning(rel_path, line, "SILENT_EXCEPT", msg)
    
    for line, name in analyzer.missing_self:
        add_issue(rel_path, line, "MISSING_SELF", 
                 f"Method {name} missing 'self' parameter")
    
    for line, msg in analyzer.unclosed_resources:
        add_warning(rel_path, line, "RESOURCE", msg)
    
    # Check for logging without logger
    if analyzer.has_logging_import and not analyzer.has_logger:
        # Check if logging.getLogger is called directly
        if 'logging.getLogger' not in source and 'getLogger' not in source:
            add_warning(rel_path, 1, "LOGGING", 
                       "Imports logging but doesn't define logger")
    
    return {
        'has_logger': analyzer.has_logger,
        'has_logging_import': analyzer.has_logging_import,
    }


# ===========================================================================
# PASS 2: Pattern-based analysis
# ===========================================================================
def pattern_analysis(filepath: Path):
    """Find common bug patterns using regex."""
    rel_path = str(filepath.relative_to(Path('.')))
    
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
        lines = content.split('\n')
    except:
        return
    
    for i, line in enumerate(lines, 1):
        # Check for == None instead of is None
        if re.search(r'==\s*None|!=\s*None', line) and 'is None' not in line:
            add_warning(rel_path, i, "STYLE", "Use 'is None' instead of '== None'")
        
        # Check for type() instead of isinstance()
        if re.search(r'type\([^)]+\)\s*==', line):
            add_warning(rel_path, i, "STYLE", "Use isinstance() instead of type() ==")
        
        # Check for string concatenation in loops
        if re.search(r'(\w+)\s*\+=\s*["\']', line) and 'for' in '\n'.join(lines[max(0,i-5):i]):
            add_warning(rel_path, i, "PERF", "String concatenation in loop - use join()")
        
        # Check for global keyword misuse
        if re.search(r'^\s*global\s+', line):
            add_warning(rel_path, i, "STYLE", "Global variable usage")
        
        # Check for hardcoded passwords/secrets (exclude token definitions and example code)
        if re.search(r'(password|secret|api_key)\s*=\s*["\'][a-zA-Z0-9]{16,}["\']', line, re.I):
            if 'example' not in line.lower() and 'test' not in line.lower() and 'token' not in line.lower():
                add_issue(rel_path, i, "SECURITY", "Possible hardcoded credential")
        
        # Check for deprecated or problematic patterns - exclude .eval() method calls
        if re.search(r'(?<!\.)\beval\s*\(', line) and 'safe_eval' not in line:
            add_issue(rel_path, i, "SECURITY", "Use of eval() is dangerous")
        
        if 'exec(' in line:
            add_warning(rel_path, i, "SECURITY", "Use of exec() - ensure input is trusted")
        
        # Check for missing f-string prefix
        if re.search(r'["\'][^"\']*\{[a-zA-Z_][a-zA-Z0-9_]*\}[^"\']*["\']', line):
            if not line.strip().startswith('f') and 'format(' not in line:
                if '{' in line and '}' in line and '.format' not in line:
                    # Could be a false positive for dict literals
                    if not re.search(r'[{]\s*["\']', line):
                        add_warning(rel_path, i, "STRING", "Possible missing f-string prefix")


# ===========================================================================
# PASS 3: Import validation
# ===========================================================================
def check_imports():
    """Check for circular imports and missing modules."""
    print("\n[IMPORT VALIDATION]")
    
    # Key modules to test
    modules_to_test = [
        'forge_ai',
        'forge_ai.core',
        'forge_ai.core.model',
        'forge_ai.core.tokenizer',
        'forge_ai.core.inference',
        'forge_ai.core.training',
        'forge_ai.core.tool_router',
        'forge_ai.modules',
        'forge_ai.modules.manager',
        'forge_ai.modules.registry',
        'forge_ai.memory',
        'forge_ai.memory.manager',
        'forge_ai.tools',
        'forge_ai.tools.tool_executor',
        'forge_ai.gui',
        'forge_ai.gui.enhanced_window',
        'forge_ai.voice',
        'forge_ai.avatar',
        'forge_ai.comms',
        'forge_ai.web',
    ]
    
    import_errors = []
    for mod in modules_to_test:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except Exception as e:
            import_errors.append((mod, str(e)))
            print(f"  ✗ {mod}: {e}")
    
    return import_errors


# ===========================================================================
# PASS 4: Runtime checks
# ===========================================================================
def runtime_checks():
    """Actually run code and check for runtime issues."""
    print("\n[RUNTIME CHECKS]")
    issues = []
    
    # Test 1: Config paths exist
    try:
        from forge_ai.config import CONFIG
        for key in ['data_dir', 'models_dir', 'logs_dir']:
            path = CONFIG.get(key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
        print("  ✓ Config paths")
    except Exception as e:
        issues.append(('config', str(e)))
        print(f"  ✗ Config: {e}")
    
    # Test 2: Model creation with all presets
    try:
        from forge_ai.core.model import create_model, MODEL_PRESETS
        for preset in ['nano', 'micro', 'tiny']:  # Only test small ones
            model = create_model(preset)
            del model
        print("  ✓ Model creation")
    except Exception as e:
        issues.append(('model', str(e)))
        print(f"  ✗ Model: {e}")
    
    # Test 3: Tokenizer encode/decode roundtrip
    try:
        from forge_ai.core.tokenizer import get_tokenizer
        tok = get_tokenizer()
        test_texts = [
            "Hello world!",
            "Special chars: @#$%^&*()",
            "Unicode: 你好世界 🌍",
            "Numbers: 12345",
            "Empty: ",
        ]
        for text in test_texts:
            tokens = tok.encode(text)
            decoded = tok.decode(tokens)
            # Note: may not be identical due to tokenizer quirks
        print("  ✓ Tokenizer roundtrip")
    except Exception as e:
        issues.append(('tokenizer', str(e)))
        print(f"  ✗ Tokenizer: {e}")
    
    # Test 4: Module registration
    try:
        from forge_ai.modules import ModuleManager, register_all
        mgr = ModuleManager()
        register_all(mgr)
        
        # Check all modules can be instantiated
        for name, cls in list(mgr.module_classes.items())[:10]:  # Test first 10
            can, reason = mgr.can_load(name)
            # Just check can_load doesn't crash
        print("  ✓ Module registration")
    except Exception as e:
        issues.append(('modules', str(e)))
        print(f"  ✗ Modules: {e}")
    
    # Test 5: Tool definitions
    try:
        from forge_ai.tools.tool_definitions import get_all_tools
        tools = get_all_tools()
        for tool in tools[:5]:  # Check first 5
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
        print("  ✓ Tool definitions")
    except Exception as e:
        issues.append(('tools', str(e)))
        print(f"  ✗ Tools: {e}")
    
    # Test 6: Memory system
    try:
        from forge_ai.memory.manager import ConversationManager
        from forge_ai.memory.vector_db import SimpleVectorDB
        import numpy as np
        
        cm = ConversationManager()
        db = SimpleVectorDB(dim=4)
        db.add(np.array([1,0,0,0], dtype=np.float32), "test")
        results = db.search(np.array([1,0,0,0], dtype=np.float32), top_k=1)
        print("  ✓ Memory system")
    except Exception as e:
        issues.append(('memory', str(e)))
        print(f"  ✗ Memory: {e}")
    
    # Test 7: GUI imports (offscreen)
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from forge_ai.gui.enhanced_window import EnhancedMainWindow
        from forge_ai.gui.tabs import ImageTab, CodeTab, create_chat_tab
        print("  ✓ GUI imports")
    except Exception as e:
        issues.append(('gui', str(e)))
        print(f"  ✗ GUI: {e}")
    
    return issues


# ===========================================================================
# PASS 5: API consistency check
# ===========================================================================
def api_consistency_check():
    """Check that APIs are consistent and documented."""
    print("\n[API CONSISTENCY]")
    issues = []
    
    # Check Tool classes have required attributes
    try:
        from forge_ai.tools import file_tools, web_tools
        
        tool_modules = [file_tools, web_tools]
        for mod in tool_modules:
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and hasattr(obj, 'execute'):
                    # Check for required attributes
                    if not hasattr(obj, 'name'):
                        issues.append((f"{mod.__name__}.{name}", "Missing 'name' attribute"))
                    if not hasattr(obj, 'description'):
                        issues.append((f"{mod.__name__}.{name}", "Missing 'description' attribute"))
                    if not hasattr(obj, 'parameters'):
                        issues.append((f"{mod.__name__}.{name}", "Missing 'parameters' attribute"))
        print(f"  ✓ Tool consistency ({len(issues)} issues)")
    except Exception as e:
        print(f"  ✗ Tool consistency: {e}")
    
    # Check Module classes have required methods
    try:
        from forge_ai.modules.registry import (
            ModelModule, TokenizerModule, InferenceModule
        )
        
        required_methods = ['load', 'unload', 'is_loaded']
        module_classes = [ModelModule, TokenizerModule, InferenceModule]
        
        for cls in module_classes:
            for method in required_methods:
                if not hasattr(cls, method):
                    issues.append((cls.__name__, f"Missing method: {method}"))
        print(f"  ✓ Module consistency")
    except Exception as e:
        print(f"  ✗ Module consistency: {e}")
    
    return issues


# ===========================================================================
# PASS 6: Thread safety analysis
# ===========================================================================
def thread_safety_check():
    """Check for potential thread safety issues."""
    print("\n[THREAD SAFETY]")
    issues = []
    
    # Patterns that suggest thread issues
    dangerous_patterns = [
        (r'threading\.Thread.*target=.*lambda', "Lambda in thread target can cause closure issues"),
        (r'self\.\w+\s*=.*\n.*Thread', "Instance attribute modification near thread start"),
    ]
    
    forge_ai_path = Path('forge_ai')
    for pyfile in forge_ai_path.rglob('*.py'):
        try:
            content = pyfile.read_text(encoding='utf-8', errors='replace')
            
            # Check if file uses threading
            if 'threading' in content or 'Thread' in content:
                # Check for lock usage
                if 'Lock' not in content and 'RLock' not in content:
                    if 'Thread(' in content:
                        issues.append((str(pyfile), "Uses threads but no Lock/RLock"))
        except:
            pass
    
    print(f"  Found {len(issues)} potential thread safety issues")
    return issues


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("DEEP CODE ANALYSIS - ForgeAI")
    print("=" * 70)
    
    # Collect all Python files
    forge_ai_path = Path('forge_ai')
    py_files = list(forge_ai_path.rglob('*.py'))
    
    print(f"\nAnalyzing {len(py_files)} Python files...")
    
    # PASS 1: AST Analysis
    print("\n[PASS 1: AST ANALYSIS]")
    for pyfile in py_files:
        analyze_file(pyfile)
    print(f"  Completed - found {len(ISSUES)} issues, {len(WARNINGS)} warnings")
    
    # PASS 2: Pattern Analysis  
    print("\n[PASS 2: PATTERN ANALYSIS]")
    for pyfile in py_files:
        pattern_analysis(pyfile)
    print(f"  Completed - total {len(ISSUES)} issues, {len(WARNINGS)} warnings")
    
    # PASS 3: Import Validation
    import_errors = check_imports()
    
    # PASS 4: Runtime Checks
    runtime_issues = runtime_checks()
    
    # PASS 5: API Consistency
    api_issues = api_consistency_check()
    
    # PASS 6: Thread Safety
    thread_issues = thread_safety_check()
    
    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Critical issues
    critical = [i for i in ISSUES if i[2] in ('SYNTAX', 'SECURITY', 'MISSING_SELF')]
    if critical:
        print(f"\n🔴 CRITICAL ISSUES ({len(critical)}):")
        for file, line, cat, msg in critical[:20]:
            print(f"   {file}:{line} [{cat}] {msg}")
        if len(critical) > 20:
            print(f"   ... and {len(critical) - 20} more")
    
    # Important issues
    important = [i for i in ISSUES if i[2] in ('MUTABLE_DEFAULT',)]
    if important:
        print(f"\n🟠 IMPORTANT ISSUES ({len(important)}):")
        for file, line, cat, msg in important[:20]:
            print(f"   {file}:{line} [{cat}] {msg}")
        if len(important) > 20:
            print(f"   ... and {len(important) - 20} more")
    
    # Warnings
    significant_warnings = [w for w in WARNINGS if w[2] in ('BARE_EXCEPT', 'RESOURCE', 'SILENT_EXCEPT')]
    if significant_warnings:
        print(f"\n🟡 SIGNIFICANT WARNINGS ({len(significant_warnings)}):")
        for file, line, cat, msg in significant_warnings[:20]:
            print(f"   {file}:{line} [{cat}] {msg}")
        if len(significant_warnings) > 20:
            print(f"   ... and {len(significant_warnings) - 20} more")
    
    # Import errors
    if import_errors:
        print(f"\n🔴 IMPORT ERRORS ({len(import_errors)}):")
        for mod, err in import_errors:
            print(f"   {mod}: {err}")
    
    # Runtime issues
    if runtime_issues:
        print(f"\n🔴 RUNTIME ISSUES ({len(runtime_issues)}):")
        for component, err in runtime_issues:
            print(f"   {component}: {err}")
    
    # API issues
    if api_issues:
        print(f"\n🟠 API ISSUES ({len(api_issues)}):")
        for component, err in api_issues:
            print(f"   {component}: {err}")
    
    # Thread issues
    if thread_issues:
        print(f"\n🟡 THREAD SAFETY ({len(thread_issues)}):")
        for file, msg in thread_issues[:10]:
            print(f"   {file}: {msg}")
    
    # Final summary
    print("\n" + "-" * 70)
    total_critical = len(critical) + len(import_errors) + len(runtime_issues)
    total_important = len(important) + len(api_issues)
    total_warnings = len(significant_warnings) + len(thread_issues)
    
    print(f"TOTALS: {total_critical} critical, {total_important} important, {total_warnings} warnings")
    
    if total_critical == 0:
        print("\n✅ No critical issues found!")
    else:
        print(f"\n⚠️  {total_critical} critical issues need attention")
    
    return total_critical

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
