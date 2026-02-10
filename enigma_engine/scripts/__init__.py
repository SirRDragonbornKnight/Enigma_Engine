"""
Scripts Package

Development and analysis utilities.

Provides:
- TabAnalyzer: Analyze GUI tab structure and detect issues
"""

from .analyze_tabs import TabAnalysisResult, TabAnalyzer

__all__ = [
    "TabAnalysisResult",
    "TabAnalyzer",
]
