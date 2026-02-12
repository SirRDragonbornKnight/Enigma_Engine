"""
File Change Watcher for Self-Improvement System

Monitors the enigma_engine folder for code changes and triggers
the self-improvement pipeline when new features are detected.

Usage:
    from enigma_engine.self_improvement.watcher import SelfImprovementDaemon
    
    daemon = SelfImprovementDaemon()
    daemon.start()  # Starts watching for changes
    
    # Manually trigger analysis
    daemon.trigger_analysis()
    
    # Stop daemon
    daemon.stop()
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """Represents a detected file change."""
    path: str
    change_type: str  # "created", "modified", "deleted"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    old_hash: str = ""
    new_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "change_type": self.change_type,
            "timestamp": self.timestamp,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
        }


@dataclass 
class WatcherConfig:
    """Configuration for the file watcher."""
    watch_paths: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".pyc", ".pyo", ".git", ".venv", "venv",
        "node_modules", ".pytest_cache", "*.egg-info", "build", "dist",
    ])
    file_extensions: List[str] = field(default_factory=lambda: [".py"])
    poll_interval: float = 5.0  # seconds between checks
    debounce_seconds: float = 2.0  # wait before triggering analysis
    auto_analyze: bool = True  # automatically analyze changes
    auto_train: bool = False  # automatically train (requires explicit enable)
    log_changes: bool = True
    state_file: str = "self_improvement_state.json"


class FileWatcher:
    """
    Watches files for changes using polling.
    
    Cross-platform compatible (doesn't require watchdog).
    """
    
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.file_hashes: Dict[str, str] = {}  # path -> hash
        self.callbacks: List[Callable[[List[FileChangeEvent]], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pending_events: List[FileChangeEvent] = []
        self._last_event_time: float = 0
    
    def add_callback(self, callback: Callable[[List[FileChangeEvent]], None]):
        """Add a callback for file changes."""
        self.callbacks.append(callback)
    
    def start(self):
        """Start watching for changes."""
        self._running = True
        
        # Initial scan to build hash map
        self._scan_all_files()
        
        # Start polling thread
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"File watcher started, tracking {len(self.file_hashes)} files")
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("File watcher stopped")
    
    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        path_lower = path.lower()
        for pattern in self.config.ignore_patterns:
            if pattern.lower() in path_lower:
                return True
        return False
    
    def _should_watch(self, path: str) -> bool:
        """Check if file should be watched."""
        if self._should_ignore(path):
            return False
        
        ext = Path(path).suffix.lower()
        return ext in self.config.file_extensions
    
    def _hash_file(self, path: str) -> str:
        """Calculate MD5 hash of file."""
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _scan_all_files(self):
        """Scan all watched paths and build hash map."""
        for watch_path in self.config.watch_paths:
            watch_path = Path(watch_path)
            if not watch_path.exists():
                continue
            
            for ext in self.config.file_extensions:
                for file_path in watch_path.rglob(f"*{ext}"):
                    str_path = str(file_path)
                    if self._should_watch(str_path):
                        self.file_hashes[str_path] = self._hash_file(str_path)
    
    def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                changes = self._detect_changes()
                
                if changes:
                    self._pending_events.extend(changes)
                    self._last_event_time = time.time()
                
                # Debounce: trigger callbacks after quiet period
                if self._pending_events and \
                   (time.time() - self._last_event_time) > self.config.debounce_seconds:
                    self._trigger_callbacks()
                
            except Exception as e:
                logger.error(f"Poll error: {e}")
            
            time.sleep(self.config.poll_interval)
    
    def _detect_changes(self) -> List[FileChangeEvent]:
        """Detect file changes since last scan."""
        changes = []
        current_files: Set[str] = set()
        
        for watch_path in self.config.watch_paths:
            watch_path = Path(watch_path)
            if not watch_path.exists():
                continue
            
            for ext in self.config.file_extensions:
                for file_path in watch_path.rglob(f"*{ext}"):
                    str_path = str(file_path)
                    if not self._should_watch(str_path):
                        continue
                    
                    current_files.add(str_path)
                    new_hash = self._hash_file(str_path)
                    
                    if str_path not in self.file_hashes:
                        # New file
                        changes.append(FileChangeEvent(
                            path=str_path,
                            change_type="created",
                            new_hash=new_hash,
                        ))
                        self.file_hashes[str_path] = new_hash
                        
                    elif self.file_hashes[str_path] != new_hash:
                        # Modified file
                        changes.append(FileChangeEvent(
                            path=str_path,
                            change_type="modified",
                            old_hash=self.file_hashes[str_path],
                            new_hash=new_hash,
                        ))
                        self.file_hashes[str_path] = new_hash
        
        # Check for deleted files
        for path in list(self.file_hashes.keys()):
            if path not in current_files:
                changes.append(FileChangeEvent(
                    path=path,
                    change_type="deleted",
                    old_hash=self.file_hashes.pop(path),
                ))
        
        return changes
    
    def _trigger_callbacks(self):
        """Trigger callbacks with pending events."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        
        for callback in self.callbacks:
            try:
                callback(events)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class SelfImprovementDaemon:
    """
    Main self-improvement daemon.
    
    Orchestrates the entire self-improvement pipeline:
    1. Watches for code changes
    2. Analyzes changes to understand new features
    3. Generates training data for new features
    4. Trains the AI on new features
    5. Tests the AI learned correctly
    6. Manages rollbacks if quality degrades
    """
    
    def __init__(
        self,
        engine_path: Optional[str] = None,
        config: Optional[WatcherConfig] = None,
        auto_start: bool = False,
    ):
        # Default to enigma_engine folder
        if engine_path is None:
            engine_path = str(Path(__file__).parent.parent)
        
        self.engine_path = Path(engine_path)
        
        # Configuration
        if config is None:
            config = WatcherConfig()
        
        if not config.watch_paths:
            config.watch_paths = [str(self.engine_path)]
        
        self.config = config
        
        # Components (lazy loaded)
        self._watcher = FileWatcher(config)
        self._analyzer = None
        self._data_generator = None
        self._trainer = None
        self._tester = None
        self._rollback_manager = None
        
        # State
        self.change_history: List[Dict] = []
        self._max_history = 100
        self._running = False
        self._state_path = self.engine_path.parent / config.state_file
        
        # Load previous state
        self._load_state()
        
        # Set up callback
        self._watcher.add_callback(self._on_changes)
        
        if auto_start:
            self.start()
    
    def _load_state(self):
        """Load previous state from file."""
        if self._state_path.exists():
            try:
                with open(self._state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.change_history = state.get("change_history", [])[-self._max_history:]
                logger.info(f"Loaded state with {len(self.change_history)} history entries")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save current state to file."""
        try:
            state = {
                "change_history": self.change_history[-self._max_history:],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self._state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def start(self):
        """Start the self-improvement daemon."""
        self._running = True
        self._watcher.start()
        logger.info("Self-improvement daemon started")
    
    def stop(self):
        """Stop the daemon."""
        self._running = False
        self._watcher.stop()
        self._save_state()
        logger.info("Self-improvement daemon stopped")
    
    def _on_changes(self, events: List[FileChangeEvent]):
        """Handle detected file changes."""
        if not events:
            return
        
        # Log changes
        if self.config.log_changes:
            for event in events:
                logger.info(f"File {event.change_type}: {event.path}")
        
        # Store in history
        self.change_history.append({
            "timestamp": datetime.now().isoformat(),
            "events": [e.to_dict() for e in events],
        })
        
        # Trim history
        if len(self.change_history) > self._max_history:
            self.change_history = self.change_history[-self._max_history:]
        
        # Save state
        self._save_state()
        
        # Auto-analyze if enabled
        if self.config.auto_analyze:
            self.trigger_analysis(events)
    
    def trigger_analysis(self, events: Optional[List[FileChangeEvent]] = None):
        """
        Trigger the self-improvement analysis pipeline.
        
        Args:
            events: Specific events to analyze, or None for full analysis
        """
        logger.info("Starting self-improvement analysis...")
        
        # Lazy load analyzer
        if self._analyzer is None:
            from .analyzer import CodeAnalyzer
            self._analyzer = CodeAnalyzer(str(self.engine_path))
        
        if events:
            # Analyze specific changes
            paths = [e.path for e in events if e.change_type != "deleted"]
        else:
            # Full analysis
            paths = None
        
        # Analyze code
        analysis = self._analyzer.analyze(paths)
        
        logger.info(f"Analysis found: {len(analysis.get('new_classes', []))} new classes, "
                   f"{len(analysis.get('new_functions', []))} new functions, "
                   f"{len(analysis.get('new_gui_elements', []))} new GUI elements")
        
        # Generate training data
        if analysis.get('new_classes') or analysis.get('new_functions') or analysis.get('new_gui_elements'):
            self._generate_training_data(analysis)
        
        # Auto-train if enabled
        if self.config.auto_train:
            self._run_training()
    
    def _generate_training_data(self, analysis: Dict):
        """Generate training data from analysis."""
        logger.info("Generating training data from analysis...")
        
        if self._data_generator is None:
            from .data_generator import TrainingDataGenerator
            self._data_generator = TrainingDataGenerator()
        
        # Generate Q&A pairs
        pairs = self._data_generator.generate_from_analysis(analysis)
        
        logger.info(f"Generated {len(pairs)} training pairs")
        
        # Save to training data file
        output_path = self.engine_path.parent / "data" / "self_improvement_training.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for pair in pairs:
                f.write(f"Q: {pair.question}\n")
                f.write(f"A: {pair.answer}\n\n")
        
        logger.info(f"Saved training data to {output_path}")
    
    def _run_training(self):
        """Run self-training on generated data."""
        logger.info("Starting self-training...")
        
        if self._rollback_manager is None:
            from .rollback import RollbackManager
            self._rollback_manager = RollbackManager()
        
        # Create backup before training
        backup_id = self._rollback_manager.create_backup("pre_self_train")
        logger.info(f"Created backup: {backup_id}")
        
        if self._trainer is None:
            from .self_trainer import SelfTrainer
            self._trainer = SelfTrainer()
        
        # Train
        result = self._trainer.train_incremental(
            data_path=str(self.engine_path.parent / "data" / "self_improvement_training.txt")
        )
        
        logger.info(f"Training complete: loss={result.final_loss}")
        
        # Test
        if self._tester is None:
            from .self_tester import SelfTester
            self._tester = SelfTester()
        
        test_result = self._tester.test()
        
        if test_result.quality_score < 0.5:
            logger.warning(f"Quality degraded! Score: {test_result.quality_score:.2f}")
            logger.info("Rolling back to backup...")
            self._rollback_manager.restore_backup(backup_id)
        else:
            logger.info(f"Self-improvement successful! Quality: {test_result.quality_score:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        return {
            "running": self._running,
            "engine_path": str(self.engine_path),
            "files_tracked": len(self._watcher.file_hashes) if self._watcher else 0,
            "change_history_count": len(self.change_history),
            "auto_analyze": self.config.auto_analyze,
            "auto_train": self.config.auto_train,
        }
    
    def get_recent_changes(self, limit: int = 10) -> List[Dict]:
        """Get recent change history."""
        return self.change_history[-limit:]
    
    def enable_auto_training(self, enable: bool = True):
        """Enable or disable automatic training."""
        self.config.auto_train = enable
        logger.info(f"Auto-training {'enabled' if enable else 'disabled'}")


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Improvement Daemon")
    parser.add_argument("command", choices=["start", "analyze", "status"],
                       help="Command to run")
    parser.add_argument("--auto-train", action="store_true",
                       help="Enable automatic training")
    parser.add_argument("--path", help="Path to enigma_engine folder")
    
    args = parser.parse_args()
    
    config = WatcherConfig()
    config.auto_train = args.auto_train
    
    daemon = SelfImprovementDaemon(
        engine_path=args.path,
        config=config,
    )
    
    if args.command == "start":
        daemon.start()
        print("Self-improvement daemon started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(10)
                status = daemon.get_status()
                print(f"Status: {status['files_tracked']} files tracked, "
                     f"{status['change_history_count']} changes recorded")
        except KeyboardInterrupt:
            daemon.stop()
            
    elif args.command == "analyze":
        daemon.trigger_analysis()
        
    elif args.command == "status":
        status = daemon.get_status()
        print(json.dumps(status, indent=2))
        
        recent = daemon.get_recent_changes(5)
        if recent:
            print("\nRecent changes:")
            for change in recent:
                print(f"  {change['timestamp']}: {len(change['events'])} events")


if __name__ == "__main__":
    main()
