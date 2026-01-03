"""
Tests for enhanced web safety and theme system.
"""
import pytest
from pathlib import Path
import tempfile
import json

from enigma.tools.url_safety import URLSafety, ContentFilter
from enigma.gui.theme_system import (
    Theme,
    ThemeColors,
    ThemeManager
)


class TestEnhancedURLSafety:
    """Test enhanced URL safety features."""
    
    def test_basic_safety_check(self):
        """Test basic URL safety checking."""
        safety = URLSafety()
        
        # Safe URL
        assert safety.is_safe('https://github.com/test')
        
        # Unsafe pattern
        assert not safety.is_safe('http://example.com/download.exe')
    
    def test_add_blocked_domain(self):
        """Test manually adding blocked domain."""
        safety = URLSafety()
        
        safety.add_blocked_domain('badsite.com')
        
        assert not safety.is_safe('https://badsite.com/page')
    
    def test_remove_blocked_domain(self):
        """Test removing blocked domain."""
        safety = URLSafety()
        
        safety.add_blocked_domain('testsite.com')
        assert not safety.is_safe('https://testsite.com')
        
        safety.remove_blocked_domain('testsite.com')
        assert safety.is_safe('https://testsite.com')
    
    def test_import_blocklist(self):
        """Test importing blocklist from file."""
        import uuid
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique domain names to avoid cache collisions
            unique1 = f"test-{uuid.uuid4().hex[:8]}.com"
            unique2 = f"test-{uuid.uuid4().hex[:8]}.com"
            
            # Create blocklist file
            blocklist_path = Path(tmpdir) / 'blocklist.txt'
            blocklist_path.write_text(f'{unique1}\n{unique2}\n')
            
            safety = URLSafety()
            added = safety.import_blocklist_from_file(blocklist_path)
            
            # Should add 2 unique domains
            assert added == 2
            assert not safety.is_safe(f'https://{unique1}')
            assert not safety.is_safe(f'https://{unique2}')
    
    def test_import_json_blocklist(self):
        """Test importing JSON format blocklist."""
        import uuid
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use unique domain names
            unique1 = f"json-{uuid.uuid4().hex[:8]}.com"
            unique2 = f"json-{uuid.uuid4().hex[:8]}.com"
            
            blocklist_path = Path(tmpdir) / 'blocklist.json'
            data = {'domains': [unique1, unique2]}
            blocklist_path.write_text(json.dumps(data))
            
            safety = URLSafety()
            added = safety.import_blocklist_from_file(blocklist_path)
            
            assert added == 2
    
    def test_cache_persistence(self):
        """Test that blocklist cache persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'cache.json'
            
            # First instance
            safety1 = URLSafety()
            safety1.blocklist_cache_path = cache_path
            safety1.add_blocked_domain('cached-site.com')
            safety1._save_cached_blocklist()
            
            # Second instance should load cache
            safety2 = URLSafety()
            safety2.blocklist_cache_path = cache_path
            safety2._load_cached_blocklist()
            
            assert not safety2.is_safe('https://cached-site.com')
    
    def test_statistics(self):
        """Test getting blocklist statistics."""
        safety = URLSafety()
        safety.add_blocked_domain('test.com')
        
        stats = safety.get_statistics()
        
        assert 'total_blocked_domains' in stats
        assert stats['total_blocked_domains'] > 0


class TestContentFilter:
    """Test content filtering."""
    
    def test_ad_detection(self):
        """Test ad content detection."""
        filter = ContentFilter()
        
        assert filter.is_ad_content('Click here to subscribe now!')
        assert filter.is_ad_content('Limited time offer - buy now!')
        assert not filter.is_ad_content('This is regular content.')
    
    def test_filter_content(self):
        """Test filtering ad content from text."""
        filter = ContentFilter()
        
        text = "Good content\nClick here to buy now\nMore good content"
        filtered = filter.filter_content(text)
        
        assert 'buy now' not in filtered.lower()
        assert 'Good content' in filtered


class TestThemeSystem:
    """Test theme system functionality."""
    
    def test_theme_creation(self):
        """Test creating a theme."""
        colors = ThemeColors(
            bg_primary='#000000',
            text_primary='#ffffff'
        )
        
        theme = Theme('test-theme', colors, 'Test theme')
        
        assert theme.name == 'test-theme'
        assert theme.colors.bg_primary == '#000000'
    
    def test_stylesheet_generation(self):
        """Test generating stylesheet from theme."""
        colors = ThemeColors()
        theme = Theme('test', colors)
        
        stylesheet = theme.generate_stylesheet()
        
        assert 'QMainWindow' in stylesheet
        assert colors.bg_primary in stylesheet
        assert colors.text_primary in stylesheet
    
    def test_save_and_load_theme(self):
        """Test saving and loading themes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            colors = ThemeColors(bg_primary='#123456')
            theme = Theme('test-theme', colors)
            
            # Save
            path = Path(tmpdir) / 'theme.json'
            theme.save(path)
            
            # Load
            loaded = Theme.load(path)
            
            assert loaded.name == 'test-theme'
            assert loaded.colors.bg_primary == '#123456'
    
    def test_theme_manager_presets(self):
        """Test theme manager presets."""
        manager = ThemeManager()
        
        # Should have presets
        themes = manager.list_themes()
        assert 'dark' in themes
        assert 'light' in themes
        assert 'high_contrast' in themes
    
    def test_theme_switching(self):
        """Test switching themes."""
        manager = ThemeManager()
        
        assert manager.set_theme('dark')
        assert manager.current_theme.name == 'Dark (Catppuccin Mocha)'
        
        assert manager.set_theme('light')
        assert manager.current_theme.name == 'Light'
    
    def test_custom_theme_creation(self):
        """Test creating custom theme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ThemeManager(Path(tmpdir))
            
            colors = ThemeColors(bg_primary='#ff0000')
            theme = manager.create_custom_theme(
                'my-theme',
                colors,
                'My custom theme'
            )
            
            assert theme.name == 'my-theme'
            assert 'my-theme' in manager.custom_themes
    
    def test_custom_theme_deletion(self):
        """Test deleting custom theme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ThemeManager(Path(tmpdir))
            
            colors = ThemeColors()
            manager.create_custom_theme('deleteme', colors)
            
            assert 'deleteme' in manager.custom_themes
            
            assert manager.delete_custom_theme('deleteme')
            assert 'deleteme' not in manager.custom_themes
    
    def test_get_current_stylesheet(self):
        """Test getting current stylesheet."""
        manager = ThemeManager()
        manager.set_theme('dark')
        
        stylesheet = manager.get_current_stylesheet()
        
        assert len(stylesheet) > 0
        assert 'QMainWindow' in stylesheet


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
