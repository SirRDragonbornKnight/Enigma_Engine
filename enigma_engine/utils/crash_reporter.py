"""
Crash Reporting for Enigma AI Engine

Anonymous error collection for debugging.

Features:
- Exception capturing
- Stack trace formatting
- Anonymous submissions
- Error categorization
- Rate limiting

Usage:
    from enigma_engine.utils.crash_reporter import CrashReporter
    
    reporter = CrashReporter()
    reporter.enable()  # Enable automatic crash reporting
    
    # Or manually report
    try:
        risky_operation()
    except Exception as e:
        reporter.report(e)
"""

import hashlib
import json
import logging
import platform
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrashReport:
    """A crash report."""
    report_id: str
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    system_info: Dict[str, str]
    context: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""


class CrashReporter:
    """Anonymous crash reporter."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        reports_dir: Optional[Path] = None,
        enabled: bool = False
    ):
        """
        Initialize crash reporter.
        
        Args:
            endpoint: Remote endpoint for reports (optional)
            reports_dir: Local directory for reports
            enabled: Whether auto-reporting is enabled
        """
        self.endpoint = endpoint
        self.reports_dir = reports_dir or Path("logs/crash_reports")
        self._enabled = enabled
        
        # Rate limiting
        self._reports_count = 0
        self._max_reports_per_hour = 10
        self._last_reset = datetime.now()
        
        # Hooks
        self._before_report: List[Callable[[CrashReport], CrashReport]] = []
        self._after_report: List[Callable[[CrashReport], None]] = []
        
        # Original exception hook
        self._original_hook = sys.excepthook
        
        # Ensure reports directory
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def enable(self):
        """Enable automatic crash reporting."""
        self._enabled = True
        sys.excepthook = self._exception_hook
        logger.info("Crash reporting enabled")
    
    def disable(self):
        """Disable automatic crash reporting."""
        self._enabled = False
        sys.excepthook = self._original_hook
        logger.info("Crash reporting disabled")
    
    @property
    def is_enabled(self) -> bool:
        """Check if reporting is enabled."""
        return self._enabled
    
    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """Custom exception hook."""
        # Call original hook
        self._original_hook(exc_type, exc_value, exc_traceback)
        
        # Report the crash
        if self._enabled:
            self.report_exception(exc_type, exc_value, exc_traceback)
    
    def report(self, exception: Exception, context: Optional[Dict] = None) -> Optional[CrashReport]:
        """
        Report an exception.
        
        Args:
            exception: The exception to report
            context: Additional context
            
        Returns:
            Crash report if submitted
        """
        return self.report_exception(
            type(exception),
            exception,
            exception.__traceback__,
            context
        )
    
    def report_exception(
        self,
        exc_type,
        exc_value,
        exc_traceback,
        context: Optional[Dict] = None
    ) -> Optional[CrashReport]:
        """
        Report an exception with full details.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
            context: Additional context
            
        Returns:
            Crash report if submitted
        """
        # Rate limiting
        if not self._check_rate_limit():
            logger.debug("Crash report rate limited")
            return None
        
        # Build report
        report = self._build_report(exc_type, exc_value, exc_traceback, context)
        
        # Run before hooks
        for hook in self._before_report:
            try:
                report = hook(report)
            except Exception as e:
                logger.error(f"Before hook error: {e}")
        
        # Scrub sensitive data
        report = self._scrub_sensitive(report)
        
        # Save locally
        self._save_local(report)
        
        # Submit to endpoint
        if self.endpoint:
            self._submit_remote(report)
        
        # Run after hooks
        for hook in self._after_report:
            try:
                hook(report)
            except Exception as e:
                logger.error(f"After hook error: {e}")
        
        return report
    
    def _build_report(
        self,
        exc_type,
        exc_value,
        exc_traceback,
        context: Optional[Dict]
    ) -> CrashReport:
        """Build crash report."""
        # Format stack trace
        if exc_traceback:
            stack_trace = "".join(traceback.format_tb(exc_traceback))
        else:
            stack_trace = traceback.format_exc()
        
        # System info (anonymized)
        system_info = {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_version": platform.release(),
            "arch": platform.machine()
        }
        
        # Create fingerprint for deduplication
        fingerprint = self._create_fingerprint(
            exc_type.__name__ if exc_type else "Unknown",
            str(exc_value),
            stack_trace
        )
        
        return CrashReport(
            report_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            error_type=exc_type.__name__ if exc_type else "Unknown",
            error_message=str(exc_value)[:500],
            stack_trace=stack_trace[:5000],
            system_info=system_info,
            context=context or {},
            fingerprint=fingerprint
        )
    
    def _create_fingerprint(
        self,
        error_type: str,
        error_message: str,
        stack_trace: str
    ) -> str:
        """Create fingerprint for error deduplication."""
        # Extract key parts of stack trace
        trace_lines = stack_trace.split("\n")
        key_lines = [l for l in trace_lines if "File" in l][:5]
        
        content = f"{error_type}:{':'.join(key_lines)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _scrub_sensitive(self, report: CrashReport) -> CrashReport:
        """Remove sensitive information from report."""
        sensitive_patterns = [
            "password", "secret", "token", "key", "auth",
            "credential", "api_key", "apikey"
        ]
        
        # Scrub context
        for key in list(report.context.keys()):
            if any(p in key.lower() for p in sensitive_patterns):
                report.context[key] = "[REDACTED]"
        
        # Scrub stack trace paths (remove usernames)
        # This is basic - production would need more sophisticated scrubbing
        
        return report
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        
        # Reset counter every hour
        if (now - self._last_reset).total_seconds() > 3600:
            self._reports_count = 0
            self._last_reset = now
        
        if self._reports_count >= self._max_reports_per_hour:
            return False
        
        self._reports_count += 1
        return True
    
    def _save_local(self, report: CrashReport):
        """Save report locally."""
        try:
            filename = f"crash_{report.report_id}_{report.timestamp[:10]}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump({
                    "report_id": report.report_id,
                    "timestamp": report.timestamp,
                    "error_type": report.error_type,
                    "error_message": report.error_message,
                    "stack_trace": report.stack_trace,
                    "system_info": report.system_info,
                    "fingerprint": report.fingerprint
                }, f, indent=2)
            
            logger.debug(f"Crash report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save crash report: {e}")
    
    def _submit_remote(self, report: CrashReport):
        """Submit report to remote endpoint."""
        if not self.endpoint:
            return
        
        try:
            import urllib.request
            import urllib.error
            
            data = json.dumps({
                "report_id": report.report_id,
                "timestamp": report.timestamp,
                "error_type": report.error_type,
                "error_message": report.error_message,
                "fingerprint": report.fingerprint,
                "system_info": report.system_info
            }).encode()
            
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.debug("Crash report submitted")
                else:
                    logger.warning(f"Crash report submission failed: {response.status}")
                    
        except Exception as e:
            logger.debug(f"Failed to submit crash report: {e}")
    
    def add_before_hook(self, hook: Callable[[CrashReport], CrashReport]):
        """Add hook to run before reporting."""
        self._before_report.append(hook)
    
    def add_after_hook(self, hook: Callable[[CrashReport], None]):
        """Add hook to run after reporting."""
        self._after_report.append(hook)
    
    def get_local_reports(self) -> List[CrashReport]:
        """Get all local crash reports."""
        reports = []
        
        for filepath in sorted(self.reports_dir.glob("crash_*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    reports.append(CrashReport(**data))
            except Exception as e:
                logger.error(f"Failed to load report {filepath}: {e}")
        
        return reports
    
    def get_recent_reports(self, limit: int = 10) -> List[CrashReport]:
        """Get most recent crash reports."""
        reports = self.get_local_reports()
        return sorted(reports, key=lambda r: r.timestamp, reverse=True)[:limit]
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of crash reports."""
        reports = self.get_local_reports()
        
        error_counts: Dict[str, int] = {}
        for report in reports:
            error_counts[report.error_type] = error_counts.get(report.error_type, 0) + 1
        
        return {
            "total_reports": len(reports),
            "unique_errors": len(error_counts),
            "error_breakdown": error_counts,
            "most_recent": reports[-1].timestamp if reports else None
        }
    
    def clear_reports(self, older_than_days: int = 30):
        """Clear old crash reports."""
        cutoff = datetime.now()
        count = 0
        
        for filepath in self.reports_dir.glob("crash_*.json"):
            try:
                # Parse date from filename
                parts = filepath.stem.split("_")
                if len(parts) >= 3:
                    report_date = datetime.fromisoformat(parts[2])
                    if (cutoff - report_date).days > older_than_days:
                        filepath.unlink()
                        count += 1
            except Exception:
                pass  # Intentionally silent
        
        logger.info(f"Cleared {count} old crash reports")


# Global instance
_reporter: Optional[CrashReporter] = None


def get_crash_reporter() -> CrashReporter:
    """Get or create global crash reporter."""
    global _reporter
    if _reporter is None:
        _reporter = CrashReporter()
    return _reporter


def enable_crash_reporting(endpoint: Optional[str] = None):
    """Enable crash reporting globally."""
    reporter = get_crash_reporter()
    if endpoint:
        reporter.endpoint = endpoint
    reporter.enable()


def disable_crash_reporting():
    """Disable crash reporting globally."""
    get_crash_reporter().disable()


def report_crash(exception: Exception, context: Optional[Dict] = None):
    """Report a crash."""
    return get_crash_reporter().report(exception, context)
