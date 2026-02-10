"""
Security Audit Utility

Tools for auditing file operations, path traversal vulnerabilities,
and security-sensitive operations.

Usage:
    from enigma_engine.utils.security_audit import SecurityAuditor, audit_path
    
    # Check if a path is safe
    if audit_path("/some/path", base_dir="/allowed/dir"):
        # Safe to use
        ...
    
    # Run full security audit
    auditor = SecurityAuditor()
    report = auditor.audit_codebase("enigma_engine/")
    print(report.summary())
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """A security issue found during audit."""
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str  # "path_traversal", "command_injection", etc.
    file_path: str
    line_number: int
    code_snippet: str
    description: str
    recommendation: str
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.category}: {self.description} ({self.file_path}:{self.line_number})"


@dataclass
class AuditReport:
    """Security audit report."""
    timestamp: datetime = field(default_factory=datetime.now)
    files_scanned: int = 0
    issues: list[SecurityIssue] = field(default_factory=list)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "high")
    
    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "medium")
    
    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "low")
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "SECURITY AUDIT REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Files Scanned: {self.files_scanned}",
            "",
            "Issue Summary:",
            f"  Critical: {self.critical_count}",
            f"  High:     {self.high_count}",
            f"  Medium:   {self.medium_count}",
            f"  Low:      {self.low_count}",
            "",
        ]
        
        if self.issues:
            lines.append("Issues Found:")
            for issue in sorted(self.issues, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity)):
                lines.append(f"  - {issue}")
        else:
            lines.append("No issues found.")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "files_scanned": self.files_scanned,
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count
            },
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "file": i.file_path,
                    "line": i.line_number,
                    "description": i.description,
                    "recommendation": i.recommendation
                }
                for i in self.issues
            ]
        }


def audit_path(path: str, base_dir: str, allow_symlinks: bool = False) -> tuple[bool, str]:
    """
    Audit a path for traversal vulnerabilities.
    
    Args:
        path: The path to check
        base_dir: The allowed base directory
        allow_symlinks: Whether to allow symbolic links
        
    Returns:
        Tuple of (is_safe, reason)
    """
    try:
        # Resolve to absolute path
        resolved = Path(path).resolve()
        base = Path(base_dir).resolve()
        
        # Check for path traversal
        if ".." in str(path):
            # Check if resolved path is still within base
            try:
                resolved.relative_to(base)
            except ValueError:
                return False, f"Path traversal detected: {path} resolves outside {base_dir}"
        
        # Check if within base directory
        try:
            resolved.relative_to(base)
        except ValueError:
            return False, f"Path {resolved} is outside allowed directory {base}"
        
        # Check for symlinks
        if not allow_symlinks:
            # Check all components
            current = Path(path)
            while current != current.parent:
                if current.is_symlink():
                    return False, f"Symlink detected at {current}"
                current = current.parent
        
        return True, "Path is safe"
        
    except Exception as e:
        return False, f"Error validating path: {e}"


def is_path_blocked(path: str, blocked_patterns: Optional[list[str]] = None) -> bool:
    """
    Check if a path matches blocked patterns.
    
    Args:
        path: Path to check
        blocked_patterns: List of regex patterns to block
        
    Returns:
        True if path is blocked
    """
    if blocked_patterns is None:
        blocked_patterns = [
            r".*\.ssh.*",           # SSH keys
            r".*\.aws.*",           # AWS credentials
            r".*\.env$",            # Environment files
            r".*password.*",        # Password files
            r".*secret.*",          # Secret files
            r".*/etc/passwd.*",     # System files
            r".*/etc/shadow.*",
            r".*\.pem$",            # Certificate files
            r".*\.key$",
        ]
    
    path_lower = path.lower()
    for pattern in blocked_patterns:
        if re.match(pattern, path_lower, re.IGNORECASE):
            return True
    return False


class SecurityAuditor:
    """
    Security auditor for Python codebase.
    
    Scans for common security vulnerabilities including:
    - Path traversal in file operations
    - Command injection via subprocess/os.system
    - SQL injection patterns
    - Hardcoded credentials
    - Insecure deserialization
    """
    
    DANGEROUS_PATTERNS = {
        "path_traversal": [
            # Open with user input
            (r'open\s*\([^)]*\+', "medium", "Dynamic path construction with open()"),
            (r'Path\s*\([^)]*\+', "low", "Dynamic Path construction"),
        ],
        "command_injection": [
            (r'os\.system\s*\(', "high", "os.system() can lead to command injection"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "high", "subprocess with shell=True"),
            (r'subprocess\.Popen\s*\([^)]*shell\s*=\s*True', "high", "Popen with shell=True"),
            (r'eval\s*\(', "critical", "eval() can execute arbitrary code"),
            (r'exec\s*\(', "critical", "exec() can execute arbitrary code"),
        ],
        "sql_injection": [
            (r'execute\s*\([^)]*%', "high", "String formatting in SQL query"),
            (r'execute\s*\([^)]*\.format', "high", "String formatting in SQL query"),
            (r'execute\s*\([^)]*\+', "high", "String concatenation in SQL query"),
        ],
        "hardcoded_credentials": [
            (r'password\s*=\s*["\'][^"\']+["\']', "medium", "Possible hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "medium", "Possible hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "medium", "Possible hardcoded secret"),
        ],
        "insecure_deserialization": [
            (r'pickle\.loads?\s*\(', "high", "Pickle can execute arbitrary code"),
            (r'marshal\.loads?\s*\(', "high", "Marshal can be dangerous"),
            (r'yaml\.load\s*\([^)]*Loader\s*=\s*None', "medium", "yaml.load without safe Loader"),
            (r'yaml\.load\s*\([^)]*(?!Loader)', "medium", "yaml.load should use safe_load"),
        ],
        "insecure_hash": [
            (r'hashlib\.md5\s*\(', "low", "MD5 is cryptographically weak"),
            (r'hashlib\.sha1\s*\(', "low", "SHA1 is cryptographically weak"),
        ],
        "debug_mode": [
            (r'debug\s*=\s*True', "low", "Debug mode enabled"),
            (r'DEBUG\s*=\s*True', "low", "Debug mode enabled"),
        ],
    }
    
    def __init__(self, exclusions: Optional[list[str]] = None):
        """
        Initialize auditor.
        
        Args:
            exclusions: List of path patterns to exclude
        """
        self.exclusions = exclusions or [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.git/**",
        ]
    
    def audit_codebase(self, root_path: str) -> AuditReport:
        """
        Audit an entire codebase.
        
        Args:
            root_path: Root directory to scan
            
        Returns:
            AuditReport with findings
        """
        report = AuditReport()
        root = Path(root_path)
        
        if not root.exists():
            logger.error(f"Path does not exist: {root_path}")
            return report
        
        # Find all Python files
        python_files = list(root.rglob("*.py"))
        
        # Filter exclusions
        for exclusion in self.exclusions:
            python_files = [f for f in python_files if not f.match(exclusion)]
        
        # Scan each file
        for file_path in python_files:
            try:
                issues = self.audit_file(file_path)
                report.issues.extend(issues)
                report.files_scanned += 1
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {e}")
        
        return report
    
    def audit_file(self, file_path: Path) -> list[SecurityIssue]:
        """
        Audit a single file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of security issues found
        """
        issues = []
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return issues
        
        # Pattern-based scanning
        for category, patterns in self.DANGEROUS_PATTERNS.items():
            for pattern, severity, description in patterns:
                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity=severity,
                            category=category,
                            file_path=str(file_path),
                            line_number=line_num,
                            code_snippet=line.strip()[:100],
                            description=description,
                            recommendation=self._get_recommendation(category)
                        ))
        
        return issues
    
    def _get_recommendation(self, category: str) -> str:
        """Get recommendation for a category."""
        recommendations = {
            "path_traversal": "Use Path.resolve() and validate paths against a base directory",
            "command_injection": "Use subprocess with a list of arguments instead of shell=True",
            "sql_injection": "Use parameterized queries with placeholders",
            "hardcoded_credentials": "Use environment variables or a secure secrets manager",
            "insecure_deserialization": "Use json or safe_load for YAML",
            "insecure_hash": "Use SHA-256 or better for security-sensitive hashing",
            "debug_mode": "Disable debug mode in production",
        }
        return recommendations.get(category, "Review and fix the security issue")


def quick_audit(path: str) -> AuditReport:
    """
    Quick security audit of a path.
    
    Args:
        path: File or directory to audit
        
    Returns:
        AuditReport
    """
    auditor = SecurityAuditor()
    p = Path(path)
    
    if p.is_file():
        report = AuditReport(files_scanned=1)
        report.issues = auditor.audit_file(p)
        return report
    else:
        return auditor.audit_codebase(path)
