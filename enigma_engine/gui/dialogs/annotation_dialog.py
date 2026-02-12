"""
Data Annotation Tool

GUI and backend for annotating training data.
Supports text labeling, classification, and entity annotation.

FILE: enigma_engine/gui/dialogs/annotation_dialog.py
TYPE: GUI/Data Management
MAIN CLASSES: AnnotationDialog, Annotator, AnnotatedSample
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor
    from PyQt5.QtWidgets import (
        QComboBox,
        QDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt5 not available - AnnotationDialog disabled")


class AnnotationType(Enum):
    """Types of annotations."""
    CLASSIFICATION = "classification"  # Single label
    MULTI_LABEL = "multi_label"       # Multiple labels
    NAMED_ENTITY = "named_entity"     # Entity spans
    SEQUENCE = "sequence"             # Token-level labels
    SENTIMENT = "sentiment"           # Sentiment score
    QA = "qa"                         # Question-answer
    SUMMARY = "summary"               # Text summarization


@dataclass
class EntitySpan:
    """A labeled span in text."""
    start: int
    end: int
    label: str
    text: str = ""


@dataclass
class AnnotatedSample:
    """An annotated data sample."""
    id: str
    text: str
    annotation_type: AnnotationType
    labels: list[str] = field(default_factory=list)
    entities: list[EntitySpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    annotated_at: float = 0.0
    annotator_id: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "annotation_type": self.annotation_type.value,
            "labels": self.labels,
            "entities": [
                {"start": e.start, "end": e.end, "label": e.label, "text": e.text}
                for e in self.entities
            ],
            "metadata": self.metadata,
            "annotated_at": self.annotated_at,
            "annotator_id": self.annotator_id,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnnotatedSample':
        return cls(
            id=data["id"],
            text=data["text"],
            annotation_type=AnnotationType(data.get("annotation_type", "classification")),
            labels=data.get("labels", []),
            entities=[
                EntitySpan(e["start"], e["end"], e["label"], e.get("text", ""))
                for e in data.get("entities", [])
            ],
            metadata=data.get("metadata", {}),
            annotated_at=data.get("annotated_at", 0),
            annotator_id=data.get("annotator_id", ""),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class AnnotationProject:
    """Annotation project configuration."""
    name: str
    annotation_type: AnnotationType
    labels: list[str] = field(default_factory=list)
    entity_labels: list[str] = field(default_factory=list)
    guidelines: str = ""
    created_at: float = field(default_factory=time.time)
    samples_total: int = 0
    samples_annotated: int = 0


class Annotator:
    """Backend for data annotation."""
    
    def __init__(self, project: AnnotationProject, storage_path: Path):
        """
        Initialize annotator.
        
        Args:
            project: Annotation project config
            storage_path: Path for storing annotations
        """
        self._project = project
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self._samples: list[AnnotatedSample] = []
        self._current_index = 0
        self._annotator_id = "default"
        
        self._load()
    
    def load_data(self, samples: list[dict]):
        """
        Load samples to annotate.
        
        Args:
            samples: List of sample dicts with 'id' and 'text'
        """
        for sample in samples:
            self._samples.append(AnnotatedSample(
                id=sample.get("id", str(len(self._samples))),
                text=sample["text"],
                annotation_type=self._project.annotation_type
            ))
        
        self._project.samples_total = len(self._samples)
        self._save_project()
    
    def get_current_sample(self) -> Optional[AnnotatedSample]:
        """Get the current sample to annotate."""
        if 0 <= self._current_index < len(self._samples):
            return self._samples[self._current_index]
        return None
    
    def next_sample(self) -> Optional[AnnotatedSample]:
        """Move to next sample."""
        if self._current_index < len(self._samples) - 1:
            self._current_index += 1
        return self.get_current_sample()
    
    def previous_sample(self) -> Optional[AnnotatedSample]:
        """Move to previous sample."""
        if self._current_index > 0:
            self._current_index -= 1
        return self.get_current_sample()
    
    def go_to_sample(self, index: int) -> Optional[AnnotatedSample]:
        """Go to specific sample."""
        if 0 <= index < len(self._samples):
            self._current_index = index
        return self.get_current_sample()
    
    def annotate_current(self,
                         labels: list[str] = None,
                         entities: list[EntitySpan] = None,
                         confidence: float = 1.0):
        """
        Annotate the current sample.
        
        Args:
            labels: Classification labels
            entities: Entity spans
            confidence: Annotation confidence
        """
        sample = self.get_current_sample()
        if not sample:
            return
        
        was_annotated = sample.annotated_at > 0
        
        sample.labels = labels or []
        sample.entities = entities or []
        sample.annotated_at = time.time()
        sample.annotator_id = self._annotator_id
        sample.confidence = confidence
        
        if not was_annotated:
            self._project.samples_annotated += 1
        
        self._save_sample(sample)
        self._save_project()
    
    def skip_current(self):
        """Skip current sample without annotating."""
        self.next_sample()
    
    def get_unannotated(self) -> list[int]:
        """Get indices of unannotated samples."""
        return [i for i, s in enumerate(self._samples) if s.annotated_at == 0]
    
    def get_progress(self) -> float:
        """Get annotation progress (0-1)."""
        if not self._samples:
            return 0.0
        return self._project.samples_annotated / len(self._samples)
    
    def export_annotations(self, format: str = "jsonl") -> Path:
        """
        Export annotations.
        
        Args:
            format: Export format ('jsonl', 'json', 'csv')
            
        Returns:
            Path to exported file
        """
        annotated = [s for s in self._samples if s.annotated_at > 0]
        
        if format == "jsonl":
            export_path = self._storage_path / "annotations.jsonl"
            with open(export_path, 'w') as f:
                for sample in annotated:
                    f.write(json.dumps(sample.to_dict()) + "\n")
        
        elif format == "json":
            export_path = self._storage_path / "annotations.json"
            with open(export_path, 'w') as f:
                json.dump([s.to_dict() for s in annotated], f, indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return export_path
    
    def _save_sample(self, sample: AnnotatedSample):
        """Save a single sample."""
        samples_path = self._storage_path / "samples.jsonl"
        
        # Rewrite all samples (simple approach)
        with open(samples_path, 'w') as f:
            for s in self._samples:
                f.write(json.dumps(s.to_dict()) + "\n")
    
    def _save_project(self):
        """Save project config."""
        project_path = self._storage_path / "project.json"
        
        project_data = {
            "name": self._project.name,
            "annotation_type": self._project.annotation_type.value,
            "labels": self._project.labels,
            "entity_labels": self._project.entity_labels,
            "guidelines": self._project.guidelines,
            "created_at": self._project.created_at,
            "samples_total": self._project.samples_total,
            "samples_annotated": self._project.samples_annotated
        }
        
        with open(project_path, 'w') as f:
            json.dump(project_data, f, indent=2)
    
    def _load(self):
        """Load project and samples."""
        project_path = self._storage_path / "project.json"
        samples_path = self._storage_path / "samples.jsonl"
        
        if project_path.exists():
            with open(project_path) as f:
                data = json.load(f)
            self._project = AnnotationProject(
                name=data["name"],
                annotation_type=AnnotationType(data["annotation_type"]),
                labels=data.get("labels", []),
                entity_labels=data.get("entity_labels", []),
                guidelines=data.get("guidelines", ""),
                created_at=data.get("created_at", time.time()),
                samples_total=data.get("samples_total", 0),
                samples_annotated=data.get("samples_annotated", 0)
            )
        
        if samples_path.exists():
            with open(samples_path) as f:
                for line in f:
                    if line.strip():
                        self._samples.append(AnnotatedSample.from_dict(json.loads(line)))


# GUI Dialog
if HAS_PYQT:
    
    class LabelButton(QPushButton):
        """Button for selecting a label."""
        
        def __init__(self, label: str, color: str = None):
            super().__init__(label)
            self._label = label
            self._selected = False
            
            if color:
                self.setStyleSheet(f"background-color: {color};")
            
            self.setCheckable(True)
            self.clicked.connect(self._on_click)
        
        def _on_click(self):
            self._selected = self.isChecked()
        
        @property
        def label(self) -> str:
            return self._label
        
        @property
        def is_selected(self) -> bool:
            return self._selected
    
    
    class EntityHighlighter:
        """Highlights entity spans in text."""
        
        COLORS = {
            "PER": "#FF6B6B",     # Person - red
            "ORG": "#4ECDC4",     # Organization - teal
            "LOC": "#45B7D1",     # Location - blue
            "DATE": "#96CEB4",    # Date - green
            "MISC": "#FFEAA7",    # Miscellaneous - yellow
        }
        
        @classmethod
        def get_color(cls, label: str) -> str:
            return cls.COLORS.get(label.upper(), "#E0E0E0")
    
    
    class AnnotationDialog(QDialog):
        """Dialog for annotating data samples."""
        
        annotation_saved = pyqtSignal(AnnotatedSample)
        
        def __init__(self, annotator: Annotator, parent=None):
            super().__init__(parent)
            self._annotator = annotator
            self._current_entities: list[EntitySpan] = []
            self._selected_labels: set[str] = set()
            
            self._setup_ui()
            self._load_current_sample()
            
            # Apply transparency
            try:
                from ..ui_settings import apply_dialog_transparency
                apply_dialog_transparency(self)
            except ImportError:
                pass  # Intentionally silent
        
        def _setup_ui(self):
            """Setup the dialog UI."""
            self.setWindowTitle("Data Annotation")
            self.setMinimumSize(900, 600)
            
            layout = QVBoxLayout(self)
            
            # Progress bar
            self._progress = QProgressBar()
            self._progress.setTextVisible(True)
            layout.addWidget(self._progress)
            
            # Main splitter
            splitter = QSplitter(Qt.Horizontal)
            
            # Left side: Text and annotations
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            
            # Sample ID and navigation
            nav_layout = QHBoxLayout()
            
            self._prev_btn = QPushButton("Previous")
            self._prev_btn.clicked.connect(self._on_previous)
            nav_layout.addWidget(self._prev_btn)
            
            self._sample_label = QLabel("Sample 0/0")
            self._sample_label.setAlignment(Qt.AlignCenter)
            nav_layout.addWidget(self._sample_label)
            
            self._next_btn = QPushButton("Next")
            self._next_btn.clicked.connect(self._on_next)
            nav_layout.addWidget(self._next_btn)
            
            left_layout.addLayout(nav_layout)
            
            # Text area
            text_group = QGroupBox("Text")
            text_layout = QVBoxLayout(text_group)
            
            self._text_edit = QTextEdit()
            self._text_edit.setReadOnly(True)
            text_layout.addWidget(self._text_edit)
            
            left_layout.addWidget(text_group)
            
            # Entity list (for NER)
            entity_group = QGroupBox("Entities")
            entity_layout = QVBoxLayout(entity_group)
            
            self._entity_list = QListWidget()
            entity_layout.addWidget(self._entity_list)
            
            entity_btn_layout = QHBoxLayout()
            self._add_entity_btn = QPushButton("Add Entity")
            self._add_entity_btn.clicked.connect(self._on_add_entity)
            entity_btn_layout.addWidget(self._add_entity_btn)
            
            self._remove_entity_btn = QPushButton("Remove Entity")
            self._remove_entity_btn.clicked.connect(self._on_remove_entity)
            entity_btn_layout.addWidget(self._remove_entity_btn)
            
            entity_layout.addLayout(entity_btn_layout)
            
            left_layout.addWidget(entity_group)
            
            splitter.addWidget(left_widget)
            
            # Right side: Labels and controls
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            
            # Labels (for classification)
            labels_group = QGroupBox("Labels")
            self._labels_layout = QVBoxLayout(labels_group)
            
            # Will be populated based on project
            self._label_buttons: list[LabelButton] = []
            
            right_layout.addWidget(labels_group)
            
            # Entity label selector (for NER)
            entity_label_group = QGroupBox("Entity Label")
            entity_label_layout = QVBoxLayout(entity_label_group)
            
            self._entity_label_combo = QComboBox()
            entity_label_layout.addWidget(self._entity_label_combo)
            
            right_layout.addWidget(entity_label_group)
            
            # Confidence slider
            conf_group = QGroupBox("Confidence")
            conf_layout = QVBoxLayout(conf_group)
            
            conf_btn_layout = QHBoxLayout()
            self._conf_low = QPushButton("Low")
            self._conf_low.clicked.connect(lambda: self._set_confidence(0.5))
            conf_btn_layout.addWidget(self._conf_low)
            
            self._conf_med = QPushButton("Medium")
            self._conf_med.clicked.connect(lambda: self._set_confidence(0.75))
            conf_btn_layout.addWidget(self._conf_med)
            
            self._conf_high = QPushButton("High")
            self._conf_high.clicked.connect(lambda: self._set_confidence(1.0))
            self._conf_high.setStyleSheet("background-color: #4ECDC4;")
            conf_btn_layout.addWidget(self._conf_high)
            
            conf_layout.addLayout(conf_btn_layout)
            
            self._confidence = 1.0
            
            right_layout.addWidget(conf_group)
            
            # Guidelines
            guidelines_group = QGroupBox("Guidelines")
            guidelines_layout = QVBoxLayout(guidelines_group)
            
            self._guidelines_text = QTextEdit()
            self._guidelines_text.setReadOnly(True)
            self._guidelines_text.setMaximumHeight(100)
            guidelines_layout.addWidget(self._guidelines_text)
            
            right_layout.addWidget(guidelines_group)
            
            right_layout.addStretch()
            
            # Action buttons
            action_layout = QHBoxLayout()
            
            self._skip_btn = QPushButton("Skip")
            self._skip_btn.clicked.connect(self._on_skip)
            action_layout.addWidget(self._skip_btn)
            
            self._save_btn = QPushButton("Save")
            self._save_btn.setStyleSheet("background-color: #4ECDC4;")
            self._save_btn.clicked.connect(self._on_save)
            action_layout.addWidget(self._save_btn)
            
            right_layout.addLayout(action_layout)
            
            splitter.addWidget(right_widget)
            splitter.setSizes([600, 300])
            
            layout.addWidget(splitter)
            
            # Setup project-specific UI
            self._setup_project_ui()
        
        def _setup_project_ui(self):
            """Setup UI based on project config."""
            project = self._annotator._project
            
            # Add classification labels
            for label in project.labels:
                btn = LabelButton(label)
                self._label_buttons.append(btn)
                self._labels_layout.addWidget(btn)
            
            # Add entity labels
            for label in project.entity_labels:
                self._entity_label_combo.addItem(label)
            
            # Set guidelines
            self._guidelines_text.setText(project.guidelines or "No guidelines provided.")
            
            # Update progress
            self._update_progress()
        
        def _load_current_sample(self):
            """Load the current sample into the UI."""
            sample = self._annotator.get_current_sample()
            if not sample:
                self._text_edit.setText("No samples loaded")
                return
            
            # Update text
            self._text_edit.setText(sample.text)
            
            # Update sample counter
            total = self._annotator._project.samples_total
            current = self._annotator._current_index + 1
            self._sample_label.setText(f"Sample {current}/{total}")
            
            # Load existing annotations
            self._selected_labels = set(sample.labels)
            self._current_entities = sample.entities.copy()
            
            # Update label buttons
            for btn in self._label_buttons:
                btn.setChecked(btn.label in self._selected_labels)
            
            # Update entity list
            self._entity_list.clear()
            for entity in self._current_entities:
                self._entity_list.addItem(f"{entity.label}: {entity.text} [{entity.start}:{entity.end}]")
            
            # Highlight entities in text
            self._highlight_entities()
            
            self._update_progress()
        
        def _highlight_entities(self):
            """Highlight entity spans in the text."""
            cursor = self._text_edit.textCursor()
            cursor.select(QTextCursor.Document)
            
            # Reset formatting
            default_format = QTextCharFormat()
            cursor.setCharFormat(default_format)
            
            # Highlight each entity
            for entity in self._current_entities:
                cursor.setPosition(entity.start)
                cursor.setPosition(entity.end, QTextCursor.KeepAnchor)
                
                fmt = QTextCharFormat()
                color = EntityHighlighter.get_color(entity.label)
                fmt.setBackground(QColor(color))
                cursor.setCharFormat(fmt)
        
        def _on_add_entity(self):
            """Add entity from text selection."""
            cursor = self._text_edit.textCursor()
            
            if not cursor.hasSelection():
                QMessageBox.warning(self, "No Selection", "Select text to create an entity")
                return
            
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
            text = cursor.selectedText()
            label = self._entity_label_combo.currentText()
            
            if not label:
                QMessageBox.warning(self, "No Label", "Select an entity label first")
                return
            
            entity = EntitySpan(start=start, end=end, label=label, text=text)
            self._current_entities.append(entity)
            
            self._entity_list.addItem(f"{label}: {text} [{start}:{end}]")
            self._highlight_entities()
        
        def _on_remove_entity(self):
            """Remove selected entity."""
            current = self._entity_list.currentRow()
            if current >= 0:
                self._entity_list.takeItem(current)
                del self._current_entities[current]
                self._highlight_entities()
        
        def _set_confidence(self, value: float):
            """Set confidence level."""
            self._confidence = value
            
            # Update button styles
            self._conf_low.setStyleSheet("" if value != 0.5 else "background-color: #FF6B6B;")
            self._conf_med.setStyleSheet("" if value != 0.75 else "background-color: #FFEAA7;")
            self._conf_high.setStyleSheet("" if value != 1.0 else "background-color: #4ECDC4;")
        
        def _on_save(self):
            """Save current annotation."""
            # Collect selected labels
            labels = [btn.label for btn in self._label_buttons if btn.is_selected]
            
            self._annotator.annotate_current(
                labels=labels,
                entities=self._current_entities,
                confidence=self._confidence
            )
            
            sample = self._annotator.get_current_sample()
            if sample:
                self.annotation_saved.emit(sample)
            
            self._annotator.next_sample()
            self._load_current_sample()
        
        def _on_skip(self):
            """Skip current sample."""
            self._annotator.skip_current()
            self._load_current_sample()
        
        def _on_previous(self):
            """Go to previous sample."""
            self._annotator.previous_sample()
            self._load_current_sample()
        
        def _on_next(self):
            """Go to next sample."""
            self._annotator.next_sample()
            self._load_current_sample()
        
        def _update_progress(self):
            """Update progress bar."""
            progress = self._annotator.get_progress()
            self._progress.setValue(int(progress * 100))
            self._progress.setFormat(f"{int(progress * 100)}% annotated")


# Factory functions
def create_classification_project(name: str, labels: list[str], guidelines: str = "") -> AnnotationProject:
    """Create a classification annotation project."""
    return AnnotationProject(
        name=name,
        annotation_type=AnnotationType.CLASSIFICATION,
        labels=labels,
        guidelines=guidelines
    )


def create_ner_project(name: str, entity_labels: list[str], guidelines: str = "") -> AnnotationProject:
    """Create a named entity recognition project."""
    return AnnotationProject(
        name=name,
        annotation_type=AnnotationType.NAMED_ENTITY,
        entity_labels=entity_labels,
        guidelines=guidelines
    )


__all__ = [
    'AnnotationType',
    'AnnotatedSample',
    'AnnotationProject',
    'EntitySpan',
    'Annotator',
    'create_classification_project',
    'create_ner_project'
]

if HAS_PYQT:
    __all__.extend(['AnnotationDialog', 'LabelButton', 'EntityHighlighter'])
