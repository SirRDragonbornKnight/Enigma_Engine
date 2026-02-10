"""
Layout Presets for Enigma AI Engine GUI.

Save and restore window layouts including:
- Window size and position
- Tab order and visibility
- Splitter positions
- Panel states (collapsed/expanded)
"""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QSplitter,
    QTabWidget,
)

logger = logging.getLogger(__name__)


@dataclass
class WindowGeometry:
    """Window position and size."""
    x: int = 100
    y: int = 100
    width: int = 1200
    height: int = 800
    maximized: bool = False
    fullscreen: bool = False


@dataclass
class SplitterState:
    """State of a splitter widget."""
    name: str
    sizes: list[int]
    orientation: str = "horizontal"


@dataclass
class TabState:
    """State of a tab widget."""
    name: str
    current_index: int = 0
    visible_tabs: list[str] = None
    tab_order: list[str] = None


@dataclass
class DockState:
    """State of a dock widget."""
    name: str
    visible: bool = True
    floating: bool = False
    area: str = "left"  # left, right, top, bottom


@dataclass 
class LayoutPreset:
    """Complete layout preset."""
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    
    window: WindowGeometry = None
    splitters: list[SplitterState] = None
    tabs: list[TabState] = None
    docks: list[DockState] = None
    custom_data: dict[str, Any] = None
    
    def __post_init__(self):
        if self.window is None:
            self.window = WindowGeometry()
        if self.splitters is None:
            self.splitters = []
        if self.tabs is None:
            self.tabs = []
        if self.docks is None:
            self.docks = []
        if self.custom_data is None:
            self.custom_data = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "window": asdict(self.window),
            "splitters": [asdict(s) for s in self.splitters],
            "tabs": [asdict(t) for t in self.tabs],
            "docks": [asdict(d) for d in self.docks],
            "custom_data": self.custom_data,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LayoutPreset':
        return cls(
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            window=WindowGeometry(**data.get("window", {})),
            splitters=[SplitterState(**s) for s in data.get("splitters", [])],
            tabs=[TabState(**t) for t in data.get("tabs", [])],
            docks=[DockState(**d) for d in data.get("docks", [])],
            custom_data=data.get("custom_data", {}),
        )


class LayoutPresetManager:
    """
    Manager for saving and restoring window layouts.
    
    Usage:
        manager = LayoutPresetManager()
        
        # Save current layout
        manager.save_preset("my_layout", main_window)
        
        # Restore layout
        manager.load_preset("my_layout", main_window)
        
        # List presets
        for name, preset in manager.list_presets().items():
            print(f"{name}: {preset.description}")
    """
    
    # Built-in presets
    BUILTIN_PRESETS = {
        "default": LayoutPreset(
            name="Default",
            description="Standard layout with sidebar",
            window=WindowGeometry(width=1200, height=800),
        ),
        "compact": LayoutPreset(
            name="Compact",
            description="Minimal layout for smaller screens",
            window=WindowGeometry(width=900, height=600),
        ),
        "wide": LayoutPreset(
            name="Wide",
            description="Extra wide layout for ultrawide monitors",
            window=WindowGeometry(width=1920, height=900),
        ),
        "presentation": LayoutPreset(
            name="Presentation",
            description="Clean layout for demos and presentations",
            window=WindowGeometry(width=1280, height=720),
        ),
    }
    
    def __init__(self, presets_dir: Optional[Path] = None):
        """
        Initialize the layout preset manager.
        
        Args:
            presets_dir: Directory for storing presets
        """
        self.presets_dir = presets_dir or Path("data/layouts")
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        self.presets: dict[str, LayoutPreset] = {}
        self._load_presets()
    
    def _load_presets(self) -> None:
        """Load presets from disk."""
        # Load built-in presets
        self.presets.update(self.BUILTIN_PRESETS)
        
        # Load user presets
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file) as f:
                    data = json.load(f)
                preset = LayoutPreset.from_dict(data)
                self.presets[preset.name.lower()] = preset
                logger.debug(f"Loaded layout preset: {preset.name}")
            except Exception as e:
                logger.error(f"Failed to load preset {preset_file}: {e}")
    
    def _save_preset_file(self, preset: LayoutPreset) -> None:
        """Save a preset to disk."""
        filename = preset.name.lower().replace(" ", "_") + ".json"
        filepath = self.presets_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
        
        logger.info(f"Saved layout preset: {preset.name}")
    
    def capture_layout(self, main_window: QMainWindow, name: str, description: str = "") -> LayoutPreset:
        """
        Capture the current layout of a window.
        
        Args:
            main_window: Window to capture layout from
            name: Preset name
            description: Preset description
            
        Returns:
            LayoutPreset with current state
        """
        preset = LayoutPreset(name=name, description=description)
        
        # Capture window geometry
        geo = main_window.geometry()
        preset.window = WindowGeometry(
            x=geo.x(),
            y=geo.y(),
            width=geo.width(),
            height=geo.height(),
            maximized=main_window.isMaximized(),
            fullscreen=main_window.isFullScreen(),
        )
        
        # Capture splitter states
        for splitter in main_window.findChildren(QSplitter):
            obj_name = splitter.objectName() or f"splitter_{id(splitter)}"
            preset.splitters.append(SplitterState(
                name=obj_name,
                sizes=splitter.sizes(),
                orientation="vertical" if splitter.orientation() == Qt.Vertical else "horizontal"
            ))
        
        # Capture tab widget states
        for tab_widget in main_window.findChildren(QTabWidget):
            obj_name = tab_widget.objectName() or f"tabs_{id(tab_widget)}"
            
            # Get tab names
            tab_names = [tab_widget.tabText(i) for i in range(tab_widget.count())]
            
            preset.tabs.append(TabState(
                name=obj_name,
                current_index=tab_widget.currentIndex(),
                visible_tabs=tab_names,
                tab_order=tab_names,
            ))
        
        # Capture dock widget states
        for dock in main_window.findChildren(QDockWidget):
            obj_name = dock.objectName() or f"dock_{id(dock)}"
            
            # Determine dock area
            area = main_window.dockWidgetArea(dock)
            area_name = {
                Qt.LeftDockWidgetArea: "left",
                Qt.RightDockWidgetArea: "right",
                Qt.TopDockWidgetArea: "top",
                Qt.BottomDockWidgetArea: "bottom",
            }.get(area, "left")
            
            preset.docks.append(DockState(
                name=obj_name,
                visible=dock.isVisible(),
                floating=dock.isFloating(),
                area=area_name,
            ))
        
        preset.updated_at = datetime.now().isoformat()
        return preset
    
    def save_preset(
        self,
        name: str,
        main_window: QMainWindow,
        description: str = ""
    ) -> LayoutPreset:
        """
        Save the current layout as a preset.
        
        Args:
            name: Preset name
            main_window: Window to capture
            description: Optional description
            
        Returns:
            Saved LayoutPreset
        """
        preset = self.capture_layout(main_window, name, description)
        
        # Store in memory
        self.presets[name.lower()] = preset
        
        # Save to disk
        self._save_preset_file(preset)
        
        return preset
    
    def load_preset(self, name: str, main_window: QMainWindow) -> bool:
        """
        Load and apply a layout preset.
        
        Args:
            name: Preset name
            main_window: Window to apply layout to
            
        Returns:
            True if successful
        """
        preset = self.presets.get(name.lower())
        if not preset:
            logger.warning(f"Layout preset not found: {name}")
            return False
        
        return self.apply_preset(preset, main_window)
    
    def apply_preset(self, preset: LayoutPreset, main_window: QMainWindow) -> bool:
        """
        Apply a layout preset to a window.
        
        Args:
            preset: Preset to apply
            main_window: Window to configure
            
        Returns:
            True if successful
        """
        try:
            # Apply window geometry
            if preset.window:
                if preset.window.fullscreen:
                    main_window.showFullScreen()
                elif preset.window.maximized:
                    main_window.showMaximized()
                else:
                    main_window.setGeometry(
                        preset.window.x,
                        preset.window.y,
                        preset.window.width,
                        preset.window.height
                    )
                    main_window.showNormal()
            
            # Apply splitter states
            splitter_map = {
                s.objectName(): s
                for s in main_window.findChildren(QSplitter)
                if s.objectName()
            }
            
            for splitter_state in preset.splitters:
                splitter = splitter_map.get(splitter_state.name)
                if splitter and splitter_state.sizes:
                    splitter.setSizes(splitter_state.sizes)
            
            # Apply tab widget states
            tab_map = {
                t.objectName(): t
                for t in main_window.findChildren(QTabWidget)
                if t.objectName()
            }
            
            for tab_state in preset.tabs:
                tab_widget = tab_map.get(tab_state.name)
                if tab_widget:
                    if 0 <= tab_state.current_index < tab_widget.count():
                        tab_widget.setCurrentIndex(tab_state.current_index)
            
            # Apply dock widget states
            dock_map = {
                d.objectName(): d
                for d in main_window.findChildren(QDockWidget)
                if d.objectName()
            }
            
            dock_areas = {
                "left": Qt.LeftDockWidgetArea,
                "right": Qt.RightDockWidgetArea,
                "top": Qt.TopDockWidgetArea,
                "bottom": Qt.BottomDockWidgetArea,
            }
            
            for dock_state in preset.docks:
                dock = dock_map.get(dock_state.name)
                if dock:
                    dock.setVisible(dock_state.visible)
                    dock.setFloating(dock_state.floating)
                    if not dock_state.floating:
                        area = dock_areas.get(dock_state.area, Qt.LeftDockWidgetArea)
                        main_window.addDockWidget(area, dock)
            
            logger.info(f"Applied layout preset: {preset.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply preset: {e}")
            return False
    
    def list_presets(self) -> dict[str, LayoutPreset]:
        """Get all available presets."""
        return self.presets.copy()
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a user preset.
        
        Args:
            name: Preset name
            
        Returns:
            True if deleted
        """
        key = name.lower()
        
        # Don't allow deleting built-in presets
        if key in self.BUILTIN_PRESETS:
            logger.warning(f"Cannot delete built-in preset: {name}")
            return False
        
        if key not in self.presets:
            return False
        
        # Remove from memory
        del self.presets[key]
        
        # Remove file
        filename = name.lower().replace(" ", "_") + ".json"
        filepath = self.presets_dir / filename
        if filepath.exists():
            filepath.unlink()
        
        logger.info(f"Deleted layout preset: {name}")
        return True
    
    def export_preset(self, name: str, filepath: Path) -> bool:
        """Export a preset to a file."""
        preset = self.presets.get(name.lower())
        if not preset:
            return False
        
        with open(filepath, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
        return True
    
    def import_preset(self, filepath: Path) -> Optional[LayoutPreset]:
        """Import a preset from a file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            preset = LayoutPreset.from_dict(data)
            self.presets[preset.name.lower()] = preset
            self._save_preset_file(preset)
            
            return preset
        except Exception as e:
            logger.error(f"Failed to import preset: {e}")
            return None


# Global instance
_manager_instance: Optional[LayoutPresetManager] = None


def get_layout_manager() -> LayoutPresetManager:
    """Get the global layout preset manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LayoutPresetManager()
    return _manager_instance


def save_layout(name: str, main_window: QMainWindow, description: str = "") -> bool:
    """Convenience function to save current layout."""
    manager = get_layout_manager()
    try:
        manager.save_preset(name, main_window, description)
        return True
    except Exception as e:
        logger.error(f"Failed to save layout: {e}")
        return False


def load_layout(name: str, main_window: QMainWindow) -> bool:
    """Convenience function to load a layout."""
    manager = get_layout_manager()
    return manager.load_preset(name, main_window)
