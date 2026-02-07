"""
Enigma AI Engine Data Management

Dataset versioning, annotation, and data processing tools.
"""

from .versioning import (
    ChangeType,
    DataChange,
    DataDiff,
    DatasetVersion,
    FileInfo,
    VersionManager,
    create_version_manager,
)

__all__ = [
    'VersionManager',
    'DatasetVersion',
    'DataDiff',
    'DataChange',
    'ChangeType',
    'FileInfo',
    'create_version_manager'
]
