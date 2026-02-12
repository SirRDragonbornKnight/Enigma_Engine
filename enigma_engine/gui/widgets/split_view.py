"""
Split View Panel

Provides side-by-side split view for chat + documents/code/notes.
Supports horizontal/vertical splitting, resizable panes, and content types.

FILE: enigma_engine/gui/widgets/split_view.py
TYPE: GUI Widget
MAIN CLASSES: SplitView, SplitPane, ContentType
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QFrame,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be displayed in a split pane."""
    CHAT = "chat"
    DOCUMENT = "document"
    CODE = "code"
    NOTES = "notes"
    PREVIEW = "preview"
    EMPTY = "empty"


class SplitOrientation(Enum):
    """Split orientation."""
    HORIZONTAL = "horizontal"  # Side by side
    VERTICAL = "vertical"      # Stacked


@dataclass
class PaneState:
    """State of a split pane."""
    content_type: ContentType
    content: str = ""
    file_path: Optional[str] = None
    scroll_position: int = 0
    cursor_position: int = 0


class SplitPane(QFrame):
    """A single pane in the split view."""
    
    content_changed = pyqtSignal(str)  # Emits content when changed
    close_requested = pyqtSignal()
    
    def __init__(self, content_type: ContentType = ContentType.EMPTY,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._content_type = content_type
        self._file_path: Optional[Path] = None
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup pane UI."""
        self.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header toolbar
        self._toolbar = QToolBar()
        self._toolbar.setIconSize(QSize(16, 16))
        self._toolbar.setMaximumHeight(32)
        
        # Content type selector
        self._type_combo = QComboBox()
        self._type_combo.addItems([t.value.title() for t in ContentType if t != ContentType.EMPTY])
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        self._toolbar.addWidget(self._type_combo)
        
        self._toolbar.addSeparator()
        
        # File actions
        self._open_action = QAction("Open", self)
        self._open_action.triggered.connect(self._open_file)
        self._toolbar.addAction(self._open_action)
        
        self._save_action = QAction("Save", self)
        self._save_action.triggered.connect(self._save_file)
        self._toolbar.addAction(self._save_action)
        
        self._toolbar.addSeparator()
        
        # Close button
        self._close_action = QAction("Close", self)
        self._close_action.triggered.connect(self.close_requested.emit)
        self._toolbar.addAction(self._close_action)
        
        layout.addWidget(self._toolbar)
        
        # Content area - stacked widget for different content types
        self._content_stack = QStackedWidget()
        
        # Empty state
        empty_widget = QWidget()
        empty_layout = QVBoxLayout(empty_widget)
        empty_label = QLabel("Select content type or drop a file here")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("color: #888;")
        empty_layout.addWidget(empty_label)
        self._content_stack.addWidget(empty_widget)
        
        # Document editor
        self._doc_editor = QTextEdit()
        self._doc_editor.setPlaceholderText("Enter document text...")
        self._doc_editor.textChanged.connect(lambda: self.content_changed.emit(self._doc_editor.toPlainText()))
        self._content_stack.addWidget(self._doc_editor)
        
        # Code editor
        self._code_editor = QPlainTextEdit()
        self._code_editor.setPlaceholderText("Enter code...")
        self._code_editor.setFont(QFont("Consolas", 11))
        self._code_editor.textChanged.connect(lambda: self.content_changed.emit(self._code_editor.toPlainText()))
        self._content_stack.addWidget(self._code_editor)
        
        # Notes area
        self._notes_editor = QTextEdit()
        self._notes_editor.setPlaceholderText("Enter notes...")
        self._notes_editor.textChanged.connect(lambda: self.content_changed.emit(self._notes_editor.toPlainText()))
        self._content_stack.addWidget(self._notes_editor)
        
        # Preview area (read-only)
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._content_stack.addWidget(self._preview)
        
        layout.addWidget(self._content_stack, 1)
        
        # Set initial state
        self._set_content_type(self._content_type)
        
    def _on_type_changed(self, type_text: str):
        """Handle content type change."""
        try:
            ct = ContentType(type_text.lower())
            self._set_content_type(ct)
        except ValueError:
            pass  # Intentionally silent
            
    def _set_content_type(self, content_type: ContentType):
        """Set the content type and show appropriate editor."""
        self._content_type = content_type
        
        # Update combo
        idx = self._type_combo.findText(content_type.value.title())
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        
        # Show appropriate widget
        widget_map = {
            ContentType.EMPTY: 0,
            ContentType.CHAT: 0,  # Chat is external
            ContentType.DOCUMENT: 1,
            ContentType.CODE: 2,
            ContentType.NOTES: 3,
            ContentType.PREVIEW: 4
        }
        self._content_stack.setCurrentIndex(widget_map.get(content_type, 0))
        
        # Update toolbar state
        can_edit = content_type in [ContentType.DOCUMENT, ContentType.CODE, ContentType.NOTES]
        self._save_action.setEnabled(can_edit)
        
    def _open_file(self):
        """Open a file into the pane."""
        filters = {
            ContentType.DOCUMENT: "Documents (*.txt *.md *.html *.rtf);;All Files (*)",
            ContentType.CODE: "Code Files (*.py *.js *.ts *.java *.cpp *.c *.h *.json *.xml);;All Files (*)",
            ContentType.NOTES: "Notes (*.txt *.md);;All Files (*)"
        }
        
        filter_str = filters.get(self._content_type, "All Files (*)")
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", filter_str)
        
        if path:
            self._file_path = Path(path)
            try:
                content = self._file_path.read_text(encoding='utf-8')
                self.set_content(content)
                logger.info(f"Opened file: {path}")
            except Exception as e:
                logger.error(f"Failed to open file: {e}")
                
    def _save_file(self):
        """Save pane content to file."""
        if self._file_path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*)")
            if path:
                self._file_path = Path(path)
            else:
                return
        
        try:
            content = self.get_content()
            self._file_path.write_text(content, encoding='utf-8')
            logger.info(f"Saved file: {self._file_path}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            
    def set_content(self, content: str):
        """Set the pane content."""
        if self._content_type == ContentType.DOCUMENT:
            self._doc_editor.setPlainText(content)
        elif self._content_type == ContentType.CODE:
            self._code_editor.setPlainText(content)
        elif self._content_type == ContentType.NOTES:
            self._notes_editor.setPlainText(content)
        elif self._content_type == ContentType.PREVIEW:
            self._preview.setHtml(content)
            
    def get_content(self) -> str:
        """Get the pane content."""
        if self._content_type == ContentType.DOCUMENT:
            return self._doc_editor.toPlainText()
        elif self._content_type == ContentType.CODE:
            return self._code_editor.toPlainText()
        elif self._content_type == ContentType.NOTES:
            return self._notes_editor.toPlainText()
        elif self._content_type == ContentType.PREVIEW:
            return self._preview.toPlainText()
        return ""
    
    def get_state(self) -> PaneState:
        """Get current pane state."""
        return PaneState(
            content_type=self._content_type,
            content=self.get_content(),
            file_path=str(self._file_path) if self._file_path else None
        )
    
    def restore_state(self, state: PaneState):
        """Restore pane state."""
        self._set_content_type(state.content_type)
        self.set_content(state.content)
        if state.file_path:
            self._file_path = Path(state.file_path)


class SplitView(QWidget):
    """Split view widget with resizable panes."""
    
    layout_changed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._orientation = SplitOrientation.HORIZONTAL
        self._panes: list[SplitPane] = []
        self._chat_widget: Optional[QWidget] = None
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup split view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top toolbar
        toolbar = QToolBar()
        toolbar.setMaximumHeight(36)
        
        # Split buttons
        split_h_btn = QPushButton("Split H")
        split_h_btn.setToolTip("Split horizontally (side by side)")
        split_h_btn.clicked.connect(lambda: self.split(SplitOrientation.HORIZONTAL))
        toolbar.addWidget(split_h_btn)
        
        split_v_btn = QPushButton("Split V")
        split_v_btn.setToolTip("Split vertically (stacked)")
        split_v_btn.clicked.connect(lambda: self.split(SplitOrientation.VERTICAL))
        toolbar.addWidget(split_v_btn)
        
        toolbar.addSeparator()
        
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset to single view")
        reset_btn.clicked.connect(self.reset_layout)
        toolbar.addWidget(reset_btn)
        
        layout.addWidget(toolbar)
        
        # Main splitter
        self._splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self._splitter, 1)
        
        # Create initial pane
        self._add_pane()
        
    def _add_pane(self, content_type: ContentType = ContentType.EMPTY) -> SplitPane:
        """Add a new pane to the split view."""
        pane = SplitPane(content_type)
        pane.close_requested.connect(lambda p=pane: self._remove_pane(p))
        self._panes.append(pane)
        self._splitter.addWidget(pane)
        return pane
    
    def _remove_pane(self, pane: SplitPane):
        """Remove a pane from the split view."""
        if len(self._panes) > 1:
            self._panes.remove(pane)
            pane.deleteLater()
            self.layout_changed.emit()
        
    def split(self, orientation: SplitOrientation = SplitOrientation.HORIZONTAL):
        """Split the view, adding a new pane."""
        self._orientation = orientation
        self._splitter.setOrientation(
            Qt.Horizontal if orientation == SplitOrientation.HORIZONTAL else Qt.Vertical
        )
        self._add_pane()
        self.layout_changed.emit()
        
    def reset_layout(self):
        """Reset to single pane layout."""
        # Remove all but first pane
        while len(self._panes) > 1:
            pane = self._panes.pop()
            pane.deleteLater()
        
        # Reset orientation
        self._splitter.setOrientation(Qt.Horizontal)
        self.layout_changed.emit()
        
    def set_chat_widget(self, widget: QWidget):
        """Set the chat widget to display in first pane.
        
        Args:
            widget: Chat widget (typically from chat_tab)
        """
        self._chat_widget = widget
        if self._panes:
            # Add chat to first pane
            first_pane = self._panes[0]
            first_pane._content_stack.insertWidget(1, widget)
            first_pane._set_content_type(ContentType.CHAT)
            
    def get_panes(self) -> list[SplitPane]:
        """Get all panes."""
        return self._panes.copy()
    
    def get_pane_count(self) -> int:
        """Get number of panes."""
        return len(self._panes)
    
    def set_sizes(self, sizes: list[int]):
        """Set pane sizes."""
        self._splitter.setSizes(sizes)
        
    def get_sizes(self) -> list[int]:
        """Get current pane sizes."""
        return self._splitter.sizes()
    
    def get_state(self) -> dict:
        """Get split view state for persistence."""
        return {
            "orientation": self._orientation.value,
            "sizes": self.get_sizes(),
            "panes": [p.get_state().__dict__ for p in self._panes]
        }
    
    def restore_state(self, state: dict):
        """Restore split view state."""
        # Reset first
        self.reset_layout()
        
        # Set orientation
        orientation = SplitOrientation(state.get("orientation", "horizontal"))
        self._splitter.setOrientation(
            Qt.Horizontal if orientation == SplitOrientation.HORIZONTAL else Qt.Vertical
        )
        
        # Restore panes
        pane_states = state.get("panes", [])
        for i, pane_data in enumerate(pane_states):
            if i >= len(self._panes):
                self._add_pane()
            
            pane_state = PaneState(
                content_type=ContentType(pane_data.get("content_type", "empty")),
                content=pane_data.get("content", ""),
                file_path=pane_data.get("file_path")
            )
            self._panes[i].restore_state(pane_state)
        
        # Restore sizes
        sizes = state.get("sizes", [])
        if sizes:
            self._splitter.setSizes(sizes)


def get_split_view(parent: Optional[QWidget] = None) -> SplitView:
    """Factory function to create a split view."""
    return SplitView(parent)


__all__ = [
    'SplitView',
    'SplitPane',
    'ContentType',
    'SplitOrientation',
    'PaneState',
    'get_split_view'
]
