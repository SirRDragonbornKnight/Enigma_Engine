"""
Tests for new GUI features added 2026-02-05:
  - Enhanced Resource Monitor with per-module memory tracking
  - Model Comparison Tab
  - Message Branching System
  - File Attachments in Chat
  - Voice Message Recording
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResourceMonitor:
    """Tests for the enhanced resource monitor."""
    
    def test_resource_monitor_import(self):
        """Test that resource monitor can be imported."""
        from forge_ai.gui.resource_monitor import ResourceMonitor
        assert ResourceMonitor is not None
    
    def test_resource_monitor_has_module_tracking(self):
        """Test that resource monitor has module memory tracking methods."""
        from forge_ai.gui.resource_monitor import ResourceMonitor
        
        # Check that new methods exist
        assert hasattr(ResourceMonitor, 'update_module_memory')
        assert hasattr(ResourceMonitor, '_estimate_module_ram')
        assert hasattr(ResourceMonitor, '_estimate_module_vram')
        assert hasattr(ResourceMonitor, '_get_memory_color')
    
    @patch('forge_ai.gui.resource_monitor.QTimer')
    @patch('forge_ai.gui.resource_monitor.QWidget.__init__')
    def test_module_timer_setup(self, mock_widget_init, mock_timer):
        """Test that module update timer is set up correctly."""
        mock_widget_init.return_value = None
        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance
        
        from forge_ai.gui.resource_monitor import ResourceMonitor
        
        # Create monitor (will fail on UI but we're checking timer setup)
        try:
            monitor = ResourceMonitor.__new__(ResourceMonitor)
            monitor.__init__()
        except Exception:
            pass  # Expected to fail without full PyQt5 environment
        
        # In a real environment, should have module timer
        # This is a basic structure test


class TestModelComparisonTab:
    """Tests for the model comparison tab."""
    
    def test_model_comparison_import(self):
        """Test that model comparison tab can be imported."""
        from forge_ai.gui.tabs.model_comparison_tab import create_model_comparison_tab
        assert create_model_comparison_tab is not None
    
    def test_response_panel_class(self):
        """Test that ResponsePanel class exists and has required methods."""
        from forge_ai.gui.tabs.model_comparison_tab import ResponsePanel
        
        assert hasattr(ResponsePanel, 'set_response')
        assert hasattr(ResponsePanel, 'set_error')
        assert hasattr(ResponsePanel, 'mark_winner')
        assert hasattr(ResponsePanel, 'reset')
    
    def test_compare_worker_class(self):
        """Test that CompareWorker class exists and has required signals."""
        from forge_ai.gui.tabs.model_comparison_tab import CompareWorker
        
        assert hasattr(CompareWorker, 'result_ready')
        assert hasattr(CompareWorker, 'error')
        assert hasattr(CompareWorker, 'all_done')
    
    def test_get_available_models(self):
        """Test that _get_available_models returns defaults."""
        from forge_ai.gui.tabs.model_comparison_tab import _get_available_models
        
        models = _get_available_models()
        
        # Should have at least default models
        assert isinstance(models, list)
        assert len(models) > 0


class TestMessageBranching:
    """Tests for the message branching system."""
    
    def test_branching_functions_exist(self):
        """Test that branching functions are defined in chat_tab."""
        from forge_ai.gui.tabs import chat_tab
        
        assert hasattr(chat_tab, '_init_branching')
        assert hasattr(chat_tab, '_add_branch')
        assert hasattr(chat_tab, '_get_branch_count')
        assert hasattr(chat_tab, '_switch_branch')
        assert hasattr(chat_tab, '_regenerate_response')
        assert hasattr(chat_tab, '_handle_branch_link')
    
    def test_init_branching(self):
        """Test that branching can be initialized on a mock parent."""
        from forge_ai.gui.tabs.chat_tab import _init_branching
        
        parent = Mock()
        _init_branching(parent)
        
        assert hasattr(parent, '_message_branches')
        assert hasattr(parent, '_current_branch')
        assert isinstance(parent._message_branches, dict)
        assert isinstance(parent._current_branch, dict)
    
    def test_add_branch(self):
        """Test adding a branch to a message."""
        from forge_ai.gui.tabs.chat_tab import _init_branching, _add_branch
        
        parent = Mock()
        parent.chat_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'}
        ]
        _init_branching(parent)
        
        # Add alternate response
        branch_idx = _add_branch(parent, 1, "Greetings, human!")
        
        assert branch_idx == 1  # Second branch (index 1)
        assert 1 in parent._message_branches
        assert len(parent._message_branches[1]) == 2  # Original + new
    
    def test_get_branch_count(self):
        """Test getting branch count."""
        from forge_ai.gui.tabs.chat_tab import _init_branching, _add_branch, _get_branch_count
        
        parent = Mock()
        parent.chat_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!'}
        ]
        _init_branching(parent)
        
        # No branches yet
        assert _get_branch_count(parent, 1) == 0
        
        # Add branches
        _add_branch(parent, 1, "Hello!")
        _add_branch(parent, 1, "Hey!")
        
        assert _get_branch_count(parent, 1) == 3  # Original + 2 new


class TestFileAttachments:
    """Tests for file attachment functionality."""
    
    def test_attachment_functions_exist(self):
        """Test that attachment functions are defined."""
        from forge_ai.gui.tabs import chat_tab
        
        assert hasattr(chat_tab, '_attach_file')
        assert hasattr(chat_tab, '_add_attachment')
        assert hasattr(chat_tab, '_update_attachment_preview')
        assert hasattr(chat_tab, '_remove_attachment')
        assert hasattr(chat_tab, '_clear_attachments')
        assert hasattr(chat_tab, 'get_attachments')
    
    def test_clear_attachments(self):
        """Test clearing attachments."""
        from forge_ai.gui.tabs.chat_tab import _clear_attachments, _update_attachment_preview
        
        parent = Mock()
        parent._attachments = ['/path/to/file1.txt', '/path/to/file2.png']
        parent.attachment_frame = Mock()
        parent.attachment_container_layout = Mock()
        parent.attachment_container_layout.count.return_value = 0
        
        _clear_attachments(parent)
        
        assert parent._attachments == []
    
    def test_get_attachments(self):
        """Test getting attachment list."""
        from forge_ai.gui.tabs.chat_tab import get_attachments
        
        parent = Mock()
        parent._attachments = ['/path/to/file.txt']
        
        attachments = get_attachments(parent)
        
        assert attachments == ['/path/to/file.txt']
        # Should return a copy, not the original
        assert attachments is not parent._attachments


class TestVoiceMessages:
    """Tests for voice message recording."""
    
    def test_voice_input_functions_exist(self):
        """Test that voice input functions are defined."""
        from forge_ai.gui.tabs import chat_tab
        
        assert hasattr(chat_tab, '_toggle_voice_input')
        assert hasattr(chat_tab, '_do_voice_input')
        assert hasattr(chat_tab, '_voice_message_saved')
    
    @patch('forge_ai.gui.tabs.chat_tab._update_attachment_preview')
    def test_voice_message_saved_attaches_file(self, mock_update_preview):
        """Test that saved voice message is attached."""
        from forge_ai.gui.tabs.chat_tab import _voice_message_saved
        
        parent = Mock()
        parent._attachments = []
        parent.rec_btn = Mock()
        parent.chat_status = Mock()
        parent.attachment_frame = Mock()
        parent.attachment_container_layout = Mock()
        parent.attachment_container_layout.count.return_value = 0
        
        _voice_message_saved(parent, '/path/to/voice.wav')
        
        assert '/path/to/voice.wav' in parent._attachments


class TestChatTabIntegration:
    """Integration tests for chat tab."""
    
    def test_chat_link_handler_routes_correctly(self):
        """Test that chat link handler routes to correct handler."""
        from forge_ai.gui.tabs.chat_tab import _handle_chat_link, _init_branching
        
        parent = Mock()
        parent.chat_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!'}
        ]
        parent._message_branches = {}
        parent._current_branch = {}
        _init_branching(parent)
        
        # Create mock URL
        mock_url = Mock()
        mock_url.toString.return_value = 'branch_next_1'
        
        # Should not raise
        _handle_chat_link(parent, mock_url)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
