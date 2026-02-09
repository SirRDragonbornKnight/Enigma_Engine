"""
Custom Dataset Builder GUI for Enigma AI Engine

Interactive dataset creation and management.

Features:
- Import from multiple sources (text, JSON, CSV)
- Manual entry with formatting
- Data cleaning and deduplication
- Preview and validation
- Export in training formats
- Template-based generation

Usage:
    from enigma_engine.gui.dataset_builder import DatasetBuilderDialog
    
    dialog = DatasetBuilderDialog()
    dialog.exec_()
    
    # Get dataset
    dataset = dialog.get_dataset()
"""

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Qt imports
try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
        QPushButton, QLineEdit, QTextEdit, QPlainTextEdit,
        QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
        QTabWidget, QGroupBox, QComboBox, QSpinBox, QCheckBox,
        QFileDialog, QMessageBox, QProgressBar, QSplitter,
        QMenu, QAction, QToolBar, QStatusBar, QHeaderView
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QColor
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@dataclass
class DataEntry:
    """A single training data entry."""
    # Core data
    text: str
    
    # For instruction tuning
    instruction: str = ""
    response: str = ""
    
    # For conversation format
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    source: str = ""
    tags: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    
    # State
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_plain(self) -> str:
        """Convert to plain text format."""
        if self.instruction and self.response:
            return f"### Instruction:\n{self.instruction}\n\n### Response:\n{self.response}"
        elif self.messages:
            lines = []
            for msg in self.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lines.append(f"<{role}>\n{content}\n</{role}>")
            return "\n\n".join(lines)
        else:
            return self.text
    
    def to_chat_format(self) -> Dict[str, Any]:
        """Convert to chat format (OpenAI style)."""
        if self.messages:
            return {"messages": self.messages}
        elif self.instruction and self.response:
            return {
                "messages": [
                    {"role": "user", "content": self.instruction},
                    {"role": "assistant", "content": self.response}
                ]
            }
        else:
            return {
                "messages": [
                    {"role": "user", "content": self.text}
                ]
            }
    
    def to_alpaca_format(self) -> Dict[str, str]:
        """Convert to Alpaca format."""
        return {
            "instruction": self.instruction or self.text,
            "input": "",
            "output": self.response or ""
        }


class DatasetBuilder:
    """
    Dataset builder for creating training data.
    """
    
    def __init__(self):
        self._entries: List[DataEntry] = []
        self._templates: Dict[str, str] = {}
        
        # Stats
        self._total_tokens = 0
        self._duplicates_removed = 0
    
    @property
    def entries(self) -> List[DataEntry]:
        """Get all entries."""
        return self._entries
    
    @property
    def count(self) -> int:
        """Get entry count."""
        return len(self._entries)
    
    def add_entry(self, entry: DataEntry) -> bool:
        """Add a single entry."""
        # Validate
        self._validate_entry(entry)
        
        if entry.is_valid:
            self._entries.append(entry)
            return True
        return False
    
    def add_text(self, text: str, **kwargs) -> DataEntry:
        """Add plain text entry."""
        entry = DataEntry(text=text.strip(), **kwargs)
        self.add_entry(entry)
        return entry
    
    def add_instruction(
        self,
        instruction: str,
        response: str,
        **kwargs
    ) -> DataEntry:
        """Add instruction-response pair."""
        entry = DataEntry(
            text="",
            instruction=instruction.strip(),
            response=response.strip(),
            **kwargs
        )
        self.add_entry(entry)
        return entry
    
    def add_conversation(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> DataEntry:
        """Add conversation."""
        entry = DataEntry(text="", messages=messages, **kwargs)
        self.add_entry(entry)
        return entry
    
    def import_text_file(self, path: str, separator: str = "\n\n") -> int:
        """
        Import from text file.
        
        Args:
            path: File path
            separator: Entry separator
            
        Returns:
            Number of entries imported
        """
        path = Path(path)
        if not path.exists():
            return 0
        
        content = path.read_text(encoding='utf-8')
        entries = content.split(separator)
        
        count = 0
        for text in entries:
            text = text.strip()
            if text:
                self.add_text(text, source=path.name)
                count += 1
        
        return count
    
    def import_json_file(self, path: str) -> int:
        """
        Import from JSON file (various formats supported).
        
        Supported formats:
        - Array of strings
        - Array of {text: "..."} objects
        - Array of {instruction: "...", response: "..."} objects
        - Array of {messages: [...]} objects
        """
        path = Path(path)
        if not path.exists():
            return 0
        
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        count = 0
        for item in data:
            if isinstance(item, str):
                self.add_text(item, source=path.name)
                count += 1
            elif isinstance(item, dict):
                if "messages" in item:
                    self.add_conversation(item["messages"], source=path.name)
                    count += 1
                elif "instruction" in item:
                    self.add_instruction(
                        item.get("instruction", ""),
                        item.get("response", item.get("output", "")),
                        source=path.name
                    )
                    count += 1
                elif "text" in item:
                    self.add_text(item["text"], source=path.name)
                    count += 1
        
        return count
    
    def import_csv_file(
        self,
        path: str,
        text_column: str = "text",
        instruction_column: str = "instruction",
        response_column: str = "response"
    ) -> int:
        """Import from CSV file."""
        path = Path(path)
        if not path.exists():
            return 0
        
        count = 0
        with open(path, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if text_column in row and row[text_column]:
                    self.add_text(row[text_column], source=path.name)
                    count += 1
                elif instruction_column in row:
                    self.add_instruction(
                        row.get(instruction_column, ""),
                        row.get(response_column, ""),
                        source=path.name
                    )
                    count += 1
        
        return count
    
    def _validate_entry(self, entry: DataEntry):
        """Validate an entry."""
        entry.validation_errors = []
        
        # Check content
        has_content = bool(
            entry.text.strip() or
            entry.instruction.strip() or
            entry.messages
        )
        
        if not has_content:
            entry.validation_errors.append("Entry has no content")
        
        # Check response for instruction format
        if entry.instruction and not entry.response:
            entry.validation_errors.append("Instruction has no response")
        
        # Check message format
        if entry.messages:
            for msg in entry.messages:
                if "role" not in msg or "content" not in msg:
                    entry.validation_errors.append("Invalid message format")
                    break
        
        entry.is_valid = len(entry.validation_errors) == 0
    
    def deduplicate(self, threshold: float = 0.95) -> int:
        """
        Remove duplicate entries.
        
        Args:
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            Number of duplicates removed
        """
        if not self._entries:
            return 0
        
        # Simple exact match dedup first
        seen = set()
        unique = []
        
        for entry in self._entries:
            key = entry.to_plain().strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        
        removed = len(self._entries) - len(unique)
        self._entries = unique
        self._duplicates_removed += removed
        
        return removed
    
    def clean(self) -> int:
        """
        Clean dataset entries.
        
        Returns:
            Number of entries modified
        """
        modified = 0
        
        for entry in self._entries:
            original_text = entry.text
            original_inst = entry.instruction
            original_resp = entry.response
            
            # Clean text
            entry.text = self._clean_text(entry.text)
            entry.instruction = self._clean_text(entry.instruction)
            entry.response = self._clean_text(entry.response)
            
            # Clean messages
            for msg in entry.messages:
                if "content" in msg:
                    msg["content"] = self._clean_text(msg["content"])
            
            if (entry.text != original_text or
                entry.instruction != original_inst or
                entry.response != original_resp):
                modified += 1
        
        return modified
    
    def _clean_text(self, text: str) -> str:
        """Clean a text string."""
        if not text:
            return text
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove control characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
        
        return text
    
    def filter_by_length(
        self,
        min_length: int = 10,
        max_length: int = 10000
    ) -> int:
        """
        Filter entries by length.
        
        Returns:
            Number removed
        """
        original_count = len(self._entries)
        
        self._entries = [
            e for e in self._entries
            if min_length <= len(e.to_plain()) <= max_length
        ]
        
        return original_count - len(self._entries)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_chars = sum(len(e.to_plain()) for e in self._entries)
        
        # Estimate tokens (rough)
        estimated_tokens = total_chars // 4
        
        return {
            "total_entries": len(self._entries),
            "valid_entries": sum(1 for e in self._entries if e.is_valid),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "duplicates_removed": self._duplicates_removed,
            "sources": list(set(e.source for e in self._entries if e.source)),
            "avg_length": total_chars // max(1, len(self._entries)),
        }
    
    def export_text(self, path: str, separator: str = "\n\n---\n\n"):
        """Export to plain text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        texts = [e.to_plain() for e in self._entries if e.is_valid]
        content = separator.join(texts)
        
        path.write_text(content, encoding='utf-8')
        logger.info(f"Exported {len(texts)} entries to {path}")
    
    def export_json(self, path: str, format: str = "chat"):
        """
        Export to JSON file.
        
        Args:
            path: Output path
            format: "chat", "alpaca", or "plain"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "chat":
            data = [e.to_chat_format() for e in self._entries if e.is_valid]
        elif format == "alpaca":
            data = [e.to_alpaca_format() for e in self._entries if e.is_valid]
        else:
            data = [{"text": e.to_plain()} for e in self._entries if e.is_valid]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} entries to {path}")
    
    def export_csv(self, path: str):
        """Export to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["instruction", "response", "text"])
            
            for entry in self._entries:
                if entry.is_valid:
                    writer.writerow([
                        entry.instruction,
                        entry.response,
                        entry.text
                    ])


if QT_AVAILABLE:
    class DatasetBuilderDialog(QDialog):
        """
        GUI dialog for building datasets.
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Dataset Builder")
            self.setMinimumSize(900, 700)
            
            self._builder = DatasetBuilder()
            self._setup_ui()
            self._connect_signals()
        
        def _setup_ui(self):
            """Set up the UI."""
            layout = QVBoxLayout(self)
            
            # Toolbar
            toolbar = QToolBar()
            toolbar.addAction("Import", self._import_file)
            toolbar.addAction("Export", self._export_file)
            toolbar.addSeparator()
            toolbar.addAction("Clean", self._clean_data)
            toolbar.addAction("Deduplicate", self._deduplicate)
            toolbar.addSeparator()
            toolbar.addAction("Clear All", self._clear_all)
            layout.addWidget(toolbar)
            
            # Main splitter
            splitter = QSplitter(Qt.Horizontal)
            
            # Left panel - Entry list
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            
            left_layout.addWidget(QLabel("Entries:"))
            self._entry_list = QListWidget()
            self._entry_list.setSelectionMode(QListWidget.ExtendedSelection)
            left_layout.addWidget(self._entry_list)
            
            # Entry controls
            entry_btns = QHBoxLayout()
            add_btn = QPushButton("Add")
            add_btn.clicked.connect(self._add_entry)
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(self._remove_entries)
            entry_btns.addWidget(add_btn)
            entry_btns.addWidget(remove_btn)
            left_layout.addLayout(entry_btns)
            
            splitter.addWidget(left_panel)
            
            # Right panel - Entry editor
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            # Tabs for different entry types
            self._tabs = QTabWidget()
            
            # Plain text tab
            text_tab = QWidget()
            text_layout = QVBoxLayout(text_tab)
            text_layout.addWidget(QLabel("Plain Text:"))
            self._text_edit = QPlainTextEdit()
            self._text_edit.setPlaceholderText("Enter training text here...")
            text_layout.addWidget(self._text_edit)
            self._tabs.addTab(text_tab, "Text")
            
            # Instruction tab
            inst_tab = QWidget()
            inst_layout = QVBoxLayout(inst_tab)
            inst_layout.addWidget(QLabel("Instruction:"))
            self._instruction_edit = QPlainTextEdit()
            self._instruction_edit.setPlaceholderText("Enter instruction/prompt...")
            self._instruction_edit.setMaximumHeight(150)
            inst_layout.addWidget(self._instruction_edit)
            inst_layout.addWidget(QLabel("Response:"))
            self._response_edit = QPlainTextEdit()
            self._response_edit.setPlaceholderText("Enter expected response...")
            inst_layout.addWidget(self._response_edit)
            self._tabs.addTab(inst_tab, "Instruction")
            
            # Conversation tab
            conv_tab = QWidget()
            conv_layout = QVBoxLayout(conv_tab)
            
            self._conv_table = QTableWidget()
            self._conv_table.setColumnCount(2)
            self._conv_table.setHorizontalHeaderLabels(["Role", "Content"])
            self._conv_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            conv_layout.addWidget(self._conv_table)
            
            conv_btns = QHBoxLayout()
            add_msg_btn = QPushButton("Add Message")
            add_msg_btn.clicked.connect(self._add_message)
            conv_btns.addWidget(add_msg_btn)
            conv_layout.addLayout(conv_btns)
            
            self._tabs.addTab(conv_tab, "Conversation")
            
            right_layout.addWidget(self._tabs)
            
            # Metadata
            meta_group = QGroupBox("Metadata")
            meta_layout = QHBoxLayout(meta_group)
            meta_layout.addWidget(QLabel("Source:"))
            self._source_edit = QLineEdit()
            meta_layout.addWidget(self._source_edit)
            meta_layout.addWidget(QLabel("Tags:"))
            self._tags_edit = QLineEdit()
            self._tags_edit.setPlaceholderText("comma, separated, tags")
            meta_layout.addWidget(self._tags_edit)
            right_layout.addWidget(meta_group)
            
            # Save entry button
            save_btn = QPushButton("Save Entry")
            save_btn.clicked.connect(self._save_entry)
            right_layout.addWidget(save_btn)
            
            splitter.addWidget(right_panel)
            splitter.setSizes([300, 600])
            
            layout.addWidget(splitter)
            
            # Stats bar
            stats_layout = QHBoxLayout()
            self._stats_label = QLabel("Entries: 0")
            stats_layout.addWidget(self._stats_label)
            stats_layout.addStretch()
            layout.addLayout(stats_layout)
            
            # Dialog buttons
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(self.reject)
            ok_btn = QPushButton("Done")
            ok_btn.clicked.connect(self.accept)
            btn_layout.addWidget(cancel_btn)
            btn_layout.addWidget(ok_btn)
            layout.addLayout(btn_layout)
        
        def _connect_signals(self):
            """Connect signals."""
            self._entry_list.currentRowChanged.connect(self._load_entry)
        
        def _update_stats(self):
            """Update statistics display."""
            stats = self._builder.get_stats()
            self._stats_label.setText(
                f"Entries: {stats['total_entries']} | "
                f"Characters: {stats['total_characters']:,} | "
                f"Est. Tokens: {stats['estimated_tokens']:,}"
            )
        
        def _refresh_list(self):
            """Refresh entry list."""
            self._entry_list.clear()
            for i, entry in enumerate(self._builder.entries):
                preview = entry.to_plain()[:50].replace('\n', ' ')
                if len(entry.to_plain()) > 50:
                    preview += "..."
                
                item = QListWidgetItem(f"{i+1}. {preview}")
                if not entry.is_valid:
                    item.setForeground(QColor("red"))
                
                self._entry_list.addItem(item)
            
            self._update_stats()
        
        def _add_entry(self):
            """Add new entry from current tab."""
            self._save_entry()
        
        def _save_entry(self):
            """Save current entry."""
            current_tab = self._tabs.currentIndex()
            
            if current_tab == 0:  # Text
                text = self._text_edit.toPlainText().strip()
                if text:
                    self._builder.add_text(
                        text,
                        source=self._source_edit.text(),
                        tags=self._get_tags()
                    )
            
            elif current_tab == 1:  # Instruction
                instruction = self._instruction_edit.toPlainText().strip()
                response = self._response_edit.toPlainText().strip()
                if instruction:
                    self._builder.add_instruction(
                        instruction,
                        response,
                        source=self._source_edit.text(),
                        tags=self._get_tags()
                    )
            
            elif current_tab == 2:  # Conversation
                messages = self._get_messages()
                if messages:
                    self._builder.add_conversation(
                        messages,
                        source=self._source_edit.text(),
                        tags=self._get_tags()
                    )
            
            self._clear_inputs()
            self._refresh_list()
        
        def _get_tags(self) -> List[str]:
            """Get tags from input."""
            text = self._tags_edit.text().strip()
            if not text:
                return []
            return [t.strip() for t in text.split(',') if t.strip()]
        
        def _get_messages(self) -> List[Dict[str, str]]:
            """Get messages from conversation table."""
            messages = []
            for row in range(self._conv_table.rowCount()):
                role_item = self._conv_table.item(row, 0)
                content_item = self._conv_table.item(row, 1)
                
                if role_item and content_item:
                    messages.append({
                        "role": role_item.text(),
                        "content": content_item.text()
                    })
            
            return messages
        
        def _add_message(self):
            """Add message row to conversation table."""
            row = self._conv_table.rowCount()
            self._conv_table.insertRow(row)
            
            # Default role alternating
            role = "user" if row % 2 == 0 else "assistant"
            self._conv_table.setItem(row, 0, QTableWidgetItem(role))
            self._conv_table.setItem(row, 1, QTableWidgetItem(""))
        
        def _clear_inputs(self):
            """Clear input fields."""
            self._text_edit.clear()
            self._instruction_edit.clear()
            self._response_edit.clear()
            self._conv_table.setRowCount(0)
        
        def _load_entry(self, index: int):
            """Load entry into editor."""
            if index < 0 or index >= len(self._builder.entries):
                return
            
            entry = self._builder.entries[index]
            
            self._text_edit.setPlainText(entry.text)
            self._instruction_edit.setPlainText(entry.instruction)
            self._response_edit.setPlainText(entry.response)
            
            # Load messages
            self._conv_table.setRowCount(0)
            for msg in entry.messages:
                row = self._conv_table.rowCount()
                self._conv_table.insertRow(row)
                self._conv_table.setItem(row, 0, QTableWidgetItem(msg.get("role", "")))
                self._conv_table.setItem(row, 1, QTableWidgetItem(msg.get("content", "")))
            
            self._source_edit.setText(entry.source)
            self._tags_edit.setText(", ".join(entry.tags))
        
        def _remove_entries(self):
            """Remove selected entries."""
            indices = [item.row() for item in self._entry_list.selectedItems()]
            indices.sort(reverse=True)
            
            for idx in indices:
                if 0 <= idx < len(self._builder.entries):
                    del self._builder.entries[idx]
            
            self._refresh_list()
        
        def _import_file(self):
            """Import from file."""
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Dataset",
                "",
                "All Files (*.*);;Text (*.txt);;JSON (*.json);;CSV (*.csv)"
            )
            
            if not path:
                return
            
            path = Path(path)
            count = 0
            
            if path.suffix == '.json':
                count = self._builder.import_json_file(str(path))
            elif path.suffix == '.csv':
                count = self._builder.import_csv_file(str(path))
            else:
                count = self._builder.import_text_file(str(path))
            
            QMessageBox.information(
                self,
                "Import Complete",
                f"Imported {count} entries from {path.name}"
            )
            
            self._refresh_list()
        
        def _export_file(self):
            """Export to file."""
            if not self._builder.entries:
                QMessageBox.warning(self, "No Data", "No entries to export")
                return
            
            path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export Dataset",
                "dataset",
                "JSON Chat (*.json);;JSON Alpaca (*.json);;Text (*.txt);;CSV (*.csv)"
            )
            
            if not path:
                return
            
            if "Chat" in selected_filter:
                self._builder.export_json(path, format="chat")
            elif "Alpaca" in selected_filter:
                self._builder.export_json(path, format="alpaca")
            elif "CSV" in selected_filter:
                self._builder.export_csv(path)
            else:
                self._builder.export_text(path)
            
            QMessageBox.information(self, "Export Complete", f"Exported to {path}")
        
        def _clean_data(self):
            """Clean dataset."""
            modified = self._builder.clean()
            QMessageBox.information(
                self,
                "Cleaning Complete",
                f"Modified {modified} entries"
            )
            self._refresh_list()
        
        def _deduplicate(self):
            """Remove duplicates."""
            removed = self._builder.deduplicate()
            QMessageBox.information(
                self,
                "Deduplication Complete",
                f"Removed {removed} duplicate entries"
            )
            self._refresh_list()
        
        def _clear_all(self):
            """Clear all entries."""
            reply = QMessageBox.question(
                self,
                "Clear All",
                "Are you sure you want to clear all entries?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._builder = DatasetBuilder()
                self._refresh_list()
                self._clear_inputs()
        
        def get_dataset(self) -> DatasetBuilder:
            """Get the built dataset."""
            return self._builder
        
        def get_builder(self) -> DatasetBuilder:
            """Get the builder instance."""
            return self._builder


def build_dataset_gui() -> Optional[DatasetBuilder]:
    """Open dataset builder dialog and return result."""
    if not QT_AVAILABLE:
        logger.error("PyQt5 not available")
        return None
    
    dialog = DatasetBuilderDialog()
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_dataset()
    return None
