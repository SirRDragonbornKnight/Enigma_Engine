"""
Tests for new core features and QoL improvements
"""

import unittest
import tempfile
import json
from pathlib import Path


class TestInteractiveTools(unittest.TestCase):
    """Test interactive tools (checklists, tasks, reminders)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
    
    def test_checklist_creation(self):
        """Test creating a checklist."""
        from enigma.tools.interactive_tools import ChecklistManager
        
        manager = ChecklistManager(self.storage_path / "checklists.json")
        result = manager.create_checklist("Test List", ["Item 1", "Item 2", "Item 3"])
        
        self.assertTrue(result['success'])
        self.assertEqual(result['name'], "Test List")
        self.assertEqual(result['items'], 3)
    
    def test_task_addition(self):
        """Test adding a task."""
        from enigma.tools.interactive_tools import TaskScheduler
        
        scheduler = TaskScheduler(self.storage_path / "tasks.json")
        result = scheduler.add_task(
            "Test Task",
            "This is a test task",
            due_date="2025-12-31T17:00:00",
            priority="high"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['title'], "Test Task")
    
    def test_reminder_setting(self):
        """Test setting a reminder."""
        from enigma.tools.interactive_tools import ReminderSystem
        
        system = ReminderSystem(self.storage_path / "reminders.json")
        result = system.set_reminder(
            "Test reminder",
            "2025-12-31T17:00:00"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], "Test reminder")


class TestErrorHandler(unittest.TestCase):
    """Test graceful error handling."""
    
    def test_file_not_found_error(self):
        """Test handling of FileNotFoundError."""
        from enigma.utils.error_handler import ErrorHandler
        
        try:
            raise FileNotFoundError("test.txt not found")
        except Exception as e:
            result = ErrorHandler.handle_error(e, "test context")
        
        self.assertFalse(result['success'])
        self.assertIn('friendly_message', result)
        self.assertIn('suggestions', result)
    
    def test_graceful_file_read(self):
        """Test graceful file reading."""
        from enigma.utils.error_handler import GracefulFileHandler
        
        result = GracefulFileHandler.read_file("/nonexistent/file.txt")
        
        self.assertFalse(result['success'])
        self.assertIn('friendly_message', result)


class TestPersonas(unittest.TestCase):
    """Test persona system."""
    
    def test_persona_loading(self):
        """Test loading predefined personas."""
        from enigma.utils.personas import PersonaManager
        
        manager = PersonaManager()
        personas = manager.list_personas()
        
        self.assertIn('teacher', personas)
        self.assertIn('assistant', personas)
        self.assertIn('tech_expert', personas)
    
    def test_get_persona(self):
        """Test getting a specific persona."""
        from enigma.utils.personas import PersonaManager
        
        manager = PersonaManager()
        teacher = manager.get_persona('teacher')
        
        self.assertIsNotNone(teacher)
        self.assertEqual(teacher.name, 'Teacher')
        self.assertIn('supportive', teacher.traits)


class TestShortcutsBasic(unittest.TestCase):
    """Test keyboard shortcuts system (without PyQt5)."""
    
    def test_undo_redo_manager(self):
        """Test undo/redo functionality."""
        from enigma.utils.shortcuts import UndoRedoManager
        
        manager = UndoRedoManager()
        
        # Add action
        state = []
        
        def do_something():
            state.append("action")
        
        def undo_something():
            if state:
                state.pop()
        
        manager.push_action({
            'description': 'Test action',
            'undo_func': undo_something,
            'redo_func': do_something
        })
        
        self.assertTrue(manager.can_undo())
        self.assertFalse(manager.can_redo())


class TestFeedback(unittest.TestCase):
    """Test feedback collection system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "feedback.json"
    
    def test_add_rating(self):
        """Test adding a rating."""
        from enigma.utils.feedback import FeedbackCollector
        
        collector = FeedbackCollector(self.storage_path)
        result = collector.add_rating(
            "test_1",
            rating=5,
            prompt="Test prompt",
            response="Test response"
        )
        
        self.assertEqual(result['rating'], 5)
    
    def test_feedback_statistics(self):
        """Test statistics calculation."""
        from enigma.utils.feedback import FeedbackCollector
        
        collector = FeedbackCollector(self.storage_path)
        
        # Add some ratings
        collector.add_rating("test_1", 5, "prompt1", "response1")
        collector.add_rating("test_2", 4, "prompt2", "response2")
        collector.add_rating("test_3", 3, "prompt3", "response3")
        
        stats = collector.get_statistics()
        
        self.assertEqual(stats['total_ratings'], 3)
        self.assertEqual(stats['average_rating'], 4.0)


class TestDiscoveryMode(unittest.TestCase):
    """Test discovery mode and auto-save."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
    
    def test_discovery_topic_generation(self):
        """Test topic generation."""
        from enigma.utils.discovery_mode import DiscoveryMode
        
        discovery = DiscoveryMode(self.storage_path / "discoveries.json")
        topic = discovery.get_discovery_topic()
        
        self.assertIsInstance(topic, str)
        self.assertGreater(len(topic), 0)
    
    def test_auto_save_conversation(self):
        """Test auto-saving conversations."""
        from enigma.utils.discovery_mode import AutoSaveManager
        
        autosave = AutoSaveManager(self.storage_path / "autosave")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]
        
        success = autosave.save_conversation("test_conv", messages)
        self.assertTrue(success)
        
        # Recover
        recovered = autosave.recover_conversation("test_conv")
        self.assertEqual(len(recovered), 2)


class TestTrainingValidator(unittest.TestCase):
    """Test training data validator."""
    
    def test_validate_good_data(self):
        """Test validation of good training data."""
        from enigma.utils.training_validator import TrainingDataValidator
        
        validator = TrainingDataValidator()
        
        good_data = """Q: What is Python?
A: Python is a programming language.

Q: How do I install Python?
A: Download from python.org and run the installer.

Q: What is a variable?
A: A variable stores data in memory.

Q: What are loops?
A: Loops repeat code multiple times.

Q: What is a function?
A: A function is reusable code.

Q: What are classes?
A: Classes define objects.

Q: What is inheritance?
A: Inheritance allows code reuse.

Q: What are modules?
A: Modules organize code.

Q: What is pip?
A: Pip installs packages.

Q: What are lists?
A: Lists store multiple items.
"""
        
        result = validator.validate_text(good_data)
        # Should have no critical issues
        self.assertEqual(len(result['issues']), 0)
    
    def test_validate_bad_data(self):
        """Test validation of bad training data."""
        from enigma.utils.training_validator import TrainingDataValidator
        
        validator = TrainingDataValidator()
        
        bad_data = "This is just random text without any structure."
        
        result = validator.validate_text(bad_data)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['issues']), 0)


class TestResourceAllocator(unittest.TestCase):
    """Test resource allocation system."""
    
    def test_mode_setting(self):
        """Test setting resource mode."""
        from enigma.utils.resource_allocator import ResourceAllocator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp:
            allocator = ResourceAllocator(storage_path=Path(tmp) / "settings.json")
            success = allocator.set_mode('balanced')
            
            self.assertTrue(success)
            self.assertEqual(allocator.current_mode, 'balanced')
    
    def test_system_info(self):
        """Test getting system information."""
        from enigma.utils.resource_allocator import ResourceAllocator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp:
            allocator = ResourceAllocator(storage_path=Path(tmp) / "settings.json")
            info = allocator.get_system_info()
            
            self.assertIn('cpu_count', info)
            self.assertIn('memory_total_gb', info)
    
    def test_generation_params(self):
        """Test generation parameter selection."""
        from enigma.utils.resource_allocator import ResourceAllocator
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp:
            allocator = ResourceAllocator(storage_path=Path(tmp) / "settings.json")
            
            # Test speed mode
            allocator.set_speed_vs_quality('speed')
            params = allocator.get_generation_params()
            self.assertLess(params['max_tokens'], 200)
            
            # Test quality mode
            allocator.set_speed_vs_quality('quality')
            params = allocator.get_generation_params()
            self.assertGreater(params['max_tokens'], 300)


class TestToolIntegration(unittest.TestCase):
    """Test that new tools are registered."""
    
    def test_interactive_tools_registered(self):
        """Test that interactive tools are in registry."""
        from enigma.tools.tool_registry import ToolRegistry
        
        registry = ToolRegistry()
        tools = registry.list_tools()
        tool_names = [t['name'] for t in tools]
        
        self.assertIn('create_checklist', tool_names)
        self.assertIn('add_task', tool_names)
        self.assertIn('set_reminder', tool_names)
    
    def test_checklist_tool_execution(self):
        """Test executing checklist tool."""
        from enigma.tools.interactive_tools import CreateChecklistTool
        
        # Test directly
        tool = CreateChecklistTool()
        result = tool.execute(name='Test List', items='Item 1, Item 2')
        
        self.assertTrue(result['success'])


if __name__ == '__main__':
    unittest.main()
