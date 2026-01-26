"""
Notes Tab - Quick notes and bookmarks viewer.

Features:
  - Create, edit, delete notes
  - Tag-based organization
  - Search notes
  - View saved bookmarks
  - Markdown preview
"""

import json
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QInputDialog, QMessageBox,
    QTabWidget, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from .shared_components import NoScrollComboBox

# Storage paths
NOTES_DIR = Path.home() / ".forge_ai" / "notes"
BOOKMARKS_FILE = Path.home() / ".forge_ai" / "bookmarks.json"

NOTES_DIR.mkdir(parents=True, exist_ok=True)


class NotesManager:
    """Manages notes storage and retrieval."""
    
    def __init__(self):
        self.notes_dir = NOTES_DIR
    
    def list_notes(self) -> list:
        """List all notes with metadata."""
        notes = []
        for note_file in sorted(self.notes_dir.glob("*.json"), reverse=True):
            try:
                with open(note_file, 'r') as f:
                    note = json.load(f)
                    notes.append(note)
            except:
                pass
        return notes
    
    def get_note(self, name: str) -> dict:
        """Get a note by name."""
        note_file = self.notes_dir / f"{name}.json"
        if note_file.exists():
            with open(note_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_note(self, name: str, content: str, tags: list = None):
        """Save a note."""
        note = {
            "name": name,
            "content": content,
            "tags": tags or [],
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
        }
        
        # Check if exists (update modified time only)
        note_file = self.notes_dir / f"{name}.json"
        if note_file.exists():
            try:
                with open(note_file, 'r') as f:
                    existing = json.load(f)
                    note["created"] = existing.get("created", note["created"])
            except:
                pass
        
        with open(note_file, 'w') as f:
            json.dump(note, f, indent=2)
        
        return note
    
    def delete_note(self, name: str):
        """Delete a note."""
        note_file = self.notes_dir / f"{name}.json"
        if note_file.exists():
            note_file.unlink()
    
    def search_notes(self, query: str) -> list:
        """Search notes by content or tags."""
        results = []
        query_lower = query.lower()
        
        for note in self.list_notes():
            if query_lower in note.get("name", "").lower():
                results.append(note)
            elif query_lower in note.get("content", "").lower():
                results.append(note)
            elif any(query_lower in tag.lower() for tag in note.get("tags", [])):
                results.append(note)
        
        return results


class BookmarksManager:
    """Manages bookmarks storage and retrieval."""
    
    def __init__(self):
        self.bookmarks_file = BOOKMARKS_FILE
    
    def _load(self) -> list:
        if self.bookmarks_file.exists():
            try:
                with open(self.bookmarks_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save(self, bookmarks: list):
        with open(self.bookmarks_file, 'w') as f:
            json.dump(bookmarks, f, indent=2)
    
    def list_bookmarks(self) -> list:
        return self._load()
    
    def add_bookmark(self, url: str, title: str, tags: list = None):
        bookmarks = self._load()
        bookmark = {
            "id": len(bookmarks) + 1,
            "url": url,
            "title": title,
            "tags": tags or [],
            "created": datetime.now().isoformat(),
        }
        bookmarks.append(bookmark)
        self._save(bookmarks)
        return bookmark
    
    def delete_bookmark(self, bookmark_id: int):
        bookmarks = self._load()
        bookmarks = [b for b in bookmarks if b.get("id") != bookmark_id]
        self._save(bookmarks)
    
    def search_bookmarks(self, query: str) -> list:
        query_lower = query.lower()
        results = []
        for bookmark in self._load():
            if query_lower in bookmark.get("title", "").lower():
                results.append(bookmark)
            elif query_lower in bookmark.get("url", "").lower():
                results.append(bookmark)
            elif any(query_lower in tag.lower() for tag in bookmark.get("tags", [])):
                results.append(bookmark)
        return results


class NotesTab(QWidget):
    """Main Notes tab with notes and bookmarks."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes_manager = NotesManager()
        self.bookmarks_manager = BookmarksManager()
        self.current_note = None
        self._setup_ui()
        self._refresh_lists()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Notes & Bookmarks")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header)
        
        # Tabs for Notes and Bookmarks
        self.tabs = QTabWidget()
        
        # Notes tab
        notes_widget = self._create_notes_widget()
        self.tabs.addTab(notes_widget, "Notes")
        
        # Bookmarks tab
        bookmarks_widget = self._create_bookmarks_widget()
        self.tabs.addTab(bookmarks_widget, "Bookmarks")
        
        layout.addWidget(self.tabs)
    
    def _create_notes_widget(self) -> QWidget:
        """Create the notes management widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Search and controls
        controls = QHBoxLayout()
        
        self.notes_search = QLineEdit()
        self.notes_search.setPlaceholderText("Search notes...")
        self.notes_search.textChanged.connect(self._search_notes)
        controls.addWidget(self.notes_search)
        
        new_btn = QPushButton("New Note")
        new_btn.setToolTip("Create a new note")
        new_btn.clicked.connect(self._new_note)
        controls.addWidget(new_btn)
        
        layout.addLayout(controls)
        
        # Splitter for list and editor
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Notes list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.notes_list = QListWidget()
        self.notes_list.currentItemChanged.connect(self._on_note_selected)
        left_layout.addWidget(self.notes_list)
        
        splitter.addWidget(left_widget)
        
        # Right: Note editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Note title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.note_title = QLineEdit()
        self.note_title.setPlaceholderText("Note title...")
        title_layout.addWidget(self.note_title)
        right_layout.addLayout(title_layout)
        
        # Tags
        tags_layout = QHBoxLayout()
        tags_layout.addWidget(QLabel("Tags:"))
        self.note_tags = QLineEdit()
        self.note_tags.setPlaceholderText("tag1, tag2, tag3...")
        tags_layout.addWidget(self.note_tags)
        right_layout.addLayout(tags_layout)
        
        # Content editor
        self.note_editor = QTextEdit()
        self.note_editor.setPlaceholderText("Write your note here...\n\nSupports plain text.")
        self.note_editor.setFont(QFont("Consolas", 10))
        right_layout.addWidget(self.note_editor)
        
        # Save/Delete buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current note")
        save_btn.clicked.connect(self._save_note)
        btn_layout.addWidget(save_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setToolTip("Delete selected note")
        delete_btn.clicked.connect(self._delete_note)
        btn_layout.addWidget(delete_btn)
        
        right_layout.addLayout(btn_layout)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([250, 550])
        
        layout.addWidget(splitter)
        
        return widget
    
    def _create_bookmarks_widget(self) -> QWidget:
        """Create the bookmarks management widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Search and controls
        controls = QHBoxLayout()
        
        self.bookmarks_search = QLineEdit()
        self.bookmarks_search.setPlaceholderText("Search bookmarks...")
        self.bookmarks_search.textChanged.connect(self._search_bookmarks)
        controls.addWidget(self.bookmarks_search)
        
        add_btn = QPushButton("Add Bookmark")
        add_btn.setToolTip("Add a new bookmark")
        add_btn.clicked.connect(self._add_bookmark)
        controls.addWidget(add_btn)
        
        layout.addLayout(controls)
        
        # Bookmarks list
        self.bookmarks_list = QListWidget()
        self.bookmarks_list.itemDoubleClicked.connect(self._open_bookmark)
        layout.addWidget(self.bookmarks_list)
        
        # Info panel
        info_group = QGroupBox("Bookmark Info")
        info_layout = QVBoxLayout(info_group)
        
        self.bookmark_info = QLabel("Select a bookmark to view details")
        self.bookmark_info.setWordWrap(True)
        info_layout.addWidget(self.bookmark_info)
        
        btn_layout = QHBoxLayout()
        
        open_btn = QPushButton("Open in Browser")
        open_btn.setToolTip("Open bookmark in default browser")
        open_btn.clicked.connect(self._open_selected_bookmark)
        btn_layout.addWidget(open_btn)
        
        copy_btn = QPushButton("Copy URL")
        copy_btn.setToolTip("Copy bookmark URL to clipboard")
        copy_btn.clicked.connect(self._copy_bookmark_url)
        btn_layout.addWidget(copy_btn)
        
        delete_bm_btn = QPushButton("Delete")
        delete_bm_btn.setToolTip("Delete selected bookmark")
        delete_bm_btn.clicked.connect(self._delete_bookmark)
        btn_layout.addWidget(delete_bm_btn)
        
        info_layout.addLayout(btn_layout)
        layout.addWidget(info_group)
        
        # Connect selection
        self.bookmarks_list.currentItemChanged.connect(self._on_bookmark_selected)
        
        return widget
    
    def _refresh_lists(self):
        """Refresh both notes and bookmarks lists."""
        self._refresh_notes()
        self._refresh_bookmarks()
    
    def _refresh_notes(self, notes=None):
        """Refresh the notes list."""
        self.notes_list.clear()
        
        if notes is None:
            notes = self.notes_manager.list_notes()
        
        for note in notes:
            tags_str = ", ".join(note.get("tags", []))
            display = f"{note['name']}"
            if tags_str:
                display += f" [{tags_str}]"
            
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, note["name"])
            
            # Color by tags
            if "important" in [t.lower() for t in note.get("tags", [])]:
                item.setForeground(QColor("#f44336"))
            elif "todo" in [t.lower() for t in note.get("tags", [])]:
                item.setForeground(QColor("#ff9800"))
            
            self.notes_list.addItem(item)
    
    def _refresh_bookmarks(self, bookmarks=None):
        """Refresh the bookmarks list."""
        self.bookmarks_list.clear()
        
        if bookmarks is None:
            bookmarks = self.bookmarks_manager.list_bookmarks()
        
        for bookmark in bookmarks:
            display = f"[*] {bookmark['title']}"
            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, bookmark.get("id"))
            item.setToolTip(bookmark.get("url", ""))
            self.bookmarks_list.addItem(item)
    
    def _search_notes(self, query: str):
        """Search and filter notes."""
        if query:
            notes = self.notes_manager.search_notes(query)
        else:
            notes = self.notes_manager.list_notes()
        self._refresh_notes(notes)
    
    def _search_bookmarks(self, query: str):
        """Search and filter bookmarks."""
        if query:
            bookmarks = self.bookmarks_manager.search_bookmarks(query)
        else:
            bookmarks = self.bookmarks_manager.list_bookmarks()
        self._refresh_bookmarks(bookmarks)
    
    def _new_note(self):
        """Create a new note."""
        self.current_note = None
        self.note_title.clear()
        self.note_tags.clear()
        self.note_editor.clear()
        self.note_title.setFocus()
    
    def _on_note_selected(self, current, previous):
        """Handle note selection."""
        if not current:
            return
        
        note_name = current.data(Qt.UserRole)
        note = self.notes_manager.get_note(note_name)
        
        if note:
            self.current_note = note_name
            self.note_title.setText(note.get("name", ""))
            self.note_tags.setText(", ".join(note.get("tags", [])))
            self.note_editor.setPlainText(note.get("content", ""))
    
    def _save_note(self):
        """Save the current note."""
        title = self.note_title.text().strip()
        if not title:
            QMessageBox.warning(self, "Error", "Please enter a note title.")
            return
        
        content = self.note_editor.toPlainText()
        tags = [t.strip() for t in self.note_tags.text().split(",") if t.strip()]
        
        self.notes_manager.save_note(title, content, tags)
        self.current_note = title
        self._refresh_notes()
        
        QMessageBox.information(self, "Saved", f"Note '{title}' saved!")
    
    def _delete_note(self):
        """Delete the current note."""
        if not self.current_note:
            return
        
        reply = QMessageBox.question(
            self, "Delete Note",
            f"Delete note '{self.current_note}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.notes_manager.delete_note(self.current_note)
            self._new_note()
            self._refresh_notes()
    
    def _add_bookmark(self):
        """Add a new bookmark."""
        url, ok = QInputDialog.getText(self, "Add Bookmark", "URL:")
        if not ok or not url:
            return
        
        title, ok = QInputDialog.getText(self, "Add Bookmark", "Title:")
        if not ok:
            title = url
        
        tags_str, ok = QInputDialog.getText(self, "Add Bookmark", "Tags (comma-separated):")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if ok else []
        
        self.bookmarks_manager.add_bookmark(url, title, tags)
        self._refresh_bookmarks()
    
    def _on_bookmark_selected(self, current, previous):
        """Handle bookmark selection."""
        if not current:
            return
        
        bookmark_id = current.data(Qt.UserRole)
        bookmarks = self.bookmarks_manager.list_bookmarks()
        
        for bookmark in bookmarks:
            if bookmark.get("id") == bookmark_id:
                info = f"<b>{bookmark['title']}</b><br>"
                info += f"<a href='{bookmark['url']}'>{bookmark['url']}</a><br>"
                if bookmark.get("tags"):
                    info += f"Tags: {', '.join(bookmark['tags'])}<br>"
                info += f"Added: {bookmark.get('created', 'Unknown')[:10]}"
                self.bookmark_info.setText(info)
                break
    
    def _open_bookmark(self, item):
        """Open bookmark on double-click."""
        self._open_selected_bookmark()
    
    def _open_selected_bookmark(self):
        """Open selected bookmark in browser."""
        current = self.bookmarks_list.currentItem()
        if not current:
            return
        
        url = current.toolTip()
        if url:
            import webbrowser
            webbrowser.open(url)
    
    def _copy_bookmark_url(self):
        """Copy bookmark URL to clipboard."""
        current = self.bookmarks_list.currentItem()
        if not current:
            return
        
        url = current.toolTip()
        if url:
            from PyQt5.QtWidgets import QApplication
            QApplication.clipboard().setText(url)
            QMessageBox.information(self, "Copied", "URL copied to clipboard!")
    
    def _delete_bookmark(self):
        """Delete selected bookmark."""
        current = self.bookmarks_list.currentItem()
        if not current:
            return
        
        bookmark_id = current.data(Qt.UserRole)
        
        reply = QMessageBox.question(
            self, "Delete Bookmark",
            "Delete this bookmark?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.bookmarks_manager.delete_bookmark(bookmark_id)
            self._refresh_bookmarks()


def create_notes_tab(parent=None):
    """Factory function to create notes tab."""
    return NotesTab(parent)
