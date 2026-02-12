"""
Self-Improvement System for Enigma AI Engine

Enables fully autonomous self-training:
1. Code Change Detection - monitors for file changes
2. Self-Analysis - reads and understands new code
3. Self-Generated Training Data - creates Q&A pairs from changes
4. Self-Training - trains itself on new features
5. Self-Testing - verifies it learned correctly
6. Logging & Rollback - safety measures

This makes the AI truly self-improving - it learns about its own new features automatically!
"""

from .watcher import SelfImprovementDaemon, FileChangeEvent, WatcherConfig
from .analyzer import CodeAnalyzer, CodeChange, FeatureExtractor, ClassInfo, FunctionInfo
from .data_generator import TrainingDataGenerator, TrainingPair
from .self_trainer import SelfTrainer, TrainingConfig, TrainingResult, LoRAAdapter
from .self_tester import SelfTester, TestResult, TestSuiteResult, TestCase
from .rollback import RollbackManager, Backup, RollbackConfig

__all__ = [
    # Main daemon
    "SelfImprovementDaemon",
    "FileChangeEvent",
    "WatcherConfig",
    
    # Code analysis
    "CodeAnalyzer",
    "CodeChange", 
    "FeatureExtractor",
    "ClassInfo",
    "FunctionInfo",
    
    # Training data generation
    "TrainingDataGenerator",
    "TrainingPair",
    
    # Self-training
    "SelfTrainer",
    "TrainingConfig",
    "TrainingResult",
    "LoRAAdapter",
    
    # Self-testing
    "SelfTester",
    "TestResult",
    "TestSuiteResult",
    "TestCase",
    
    # Rollback / Safety
    "RollbackManager",
    "Backup",
    "RollbackConfig",
]
