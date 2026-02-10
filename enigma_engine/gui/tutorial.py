"""
Interactive Tutorial System
===========================

In-app guided tours for new users to learn the Enigma Engine interface.
Supports step-by-step highlights, tooltips, and interactive exercises.

Usage:
    from enigma_engine.gui.tutorial import TutorialManager, TutorialStep
    
    # Create and run a tutorial
    manager = TutorialManager(main_window)
    manager.start_tutorial('quick_start')
    
    # Or create custom tutorials
    steps = [
        TutorialStep(
            target='chat_input',
            title='Chat Input',
            description='Type your message here',
            position='bottom'
        ),
        ...
    ]
    manager.run_custom_tutorial(steps)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import (
        Qt, QTimer, QPoint, QRect, pyqtSignal, QObject
    )
    from PyQt5.QtWidgets import (
        QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QFrame, QGraphicsDropShadowEffect, QApplication
    )
    from PyQt5.QtGui import QColor, QPainter, QPainterPath
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False


class StepPosition(Enum):
    """Position of tutorial tooltip relative to target."""
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    AUTO = "auto"


@dataclass
class TutorialStep:
    """A single step in a tutorial."""
    # Target widget (by objectName or reference)
    target: str
    
    # Content
    title: str
    description: str
    
    # Position and display
    position: StepPosition = StepPosition.AUTO
    highlight: bool = True
    
    # Actions
    action: Optional[str] = None  # 'click', 'type', 'hover'
    action_data: Optional[str] = None  # Text to type, etc.
    wait_for_action: bool = False  # Wait for user to perform action
    
    # Navigation
    can_skip: bool = True
    auto_advance: bool = False
    auto_advance_delay: int = 3000  # ms
    
    # Callbacks
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    
    # Validation
    validate: Optional[Callable[[], bool]] = None


@dataclass
class Tutorial:
    """A complete tutorial with multiple steps."""
    id: str
    name: str
    description: str
    steps: List[TutorialStep]
    
    # Metadata
    category: str = "general"
    difficulty: str = "beginner"  # beginner, intermediate, advanced
    estimated_time: int = 5  # minutes
    
    # Requirements
    requires_model: bool = False
    requires_training_data: bool = False
    
    # Progress tracking
    completed: bool = False
    progress: int = 0


if PYQT5_AVAILABLE:
    
    class HighlightOverlay(QWidget):
        """
        Overlay that darkens everything except the highlighted widget.
        Creates a spotlight effect on the target.
        """
        
        def __init__(self, parent: QWidget):
            super().__init__(parent)
            self.target_rect: Optional[QRect] = None
            self.padding = 8
            
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            
        def set_target(self, widget: Optional[QWidget]) -> None:
            """Set the widget to highlight."""
            if widget is None:
                self.target_rect = None
            else:
                # Get global position and map to overlay coordinates
                global_pos = widget.mapToGlobal(QPoint(0, 0))
                local_pos = self.mapFromGlobal(global_pos)
                self.target_rect = QRect(
                    local_pos.x() - self.padding,
                    local_pos.y() - self.padding,
                    widget.width() + self.padding * 2,
                    widget.height() + self.padding * 2
                )
            self.update()
        
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Dark overlay
            painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
            
            if self.target_rect:
                # Cut out the highlight area with rounded corners
                path = QPainterPath()
                path.addRoundedRect(
                    float(self.target_rect.x()),
                    float(self.target_rect.y()),
                    float(self.target_rect.width()),
                    float(self.target_rect.height()),
                    8, 8
                )
                
                # Clear the highlight area
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.fillPath(path, QColor(0, 0, 0, 0))
                
                # Draw border around highlight
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                painter.setPen(QColor(100, 200, 255, 200))
                painter.drawPath(path)
    
    
    class TutorialTooltip(QFrame):
        """
        Tutorial step tooltip with title, description, and navigation.
        """
        
        next_clicked = pyqtSignal()
        prev_clicked = pyqtSignal()
        skip_clicked = pyqtSignal()
        
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setup_ui()
            self._setup_shadow()
            
        def setup_ui(self):
            self.setObjectName("TutorialTooltip")
            self.setMinimumWidth(300)
            self.setMaximumWidth(400)
            
            # Style
            self.setStyleSheet("""
                #TutorialTooltip {
                    background-color: #2d2d2d;
                    border: 1px solid #3d3d3d;
                    border-radius: 8px;
                    padding: 16px;
                }
                QLabel#title {
                    color: #ffffff;
                    font-size: 14px;
                    font-weight: bold;
                }
                QLabel#description {
                    color: #b0b0b0;
                    font-size: 12px;
                }
                QLabel#progress {
                    color: #808080;
                    font-size: 11px;
                }
                QPushButton {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #4d4d4d;
                }
                QPushButton#next {
                    background-color: #4a90d9;
                }
                QPushButton#next:hover {
                    background-color: #5aa0e9;
                }
                QPushButton#skip {
                    background-color: transparent;
                    color: #808080;
                }
            """)
            
            # Layout
            layout = QVBoxLayout(self)
            layout.setSpacing(12)
            
            # Title
            self.title_label = QLabel()
            self.title_label.setObjectName("title")
            layout.addWidget(self.title_label)
            
            # Description
            self.desc_label = QLabel()
            self.desc_label.setObjectName("description")
            self.desc_label.setWordWrap(True)
            layout.addWidget(self.desc_label)
            
            # Progress
            self.progress_label = QLabel()
            self.progress_label.setObjectName("progress")
            layout.addWidget(self.progress_label)
            
            # Buttons
            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(8)
            
            self.skip_btn = QPushButton("Skip Tutorial")
            self.skip_btn.setObjectName("skip")
            self.skip_btn.clicked.connect(self.skip_clicked.emit)
            btn_layout.addWidget(self.skip_btn)
            
            btn_layout.addStretch()
            
            self.prev_btn = QPushButton("Previous")
            self.prev_btn.clicked.connect(self.prev_clicked.emit)
            btn_layout.addWidget(self.prev_btn)
            
            self.next_btn = QPushButton("Next")
            self.next_btn.setObjectName("next")
            self.next_btn.clicked.connect(self.next_clicked.emit)
            btn_layout.addWidget(self.next_btn)
            
            layout.addLayout(btn_layout)
        
        def _setup_shadow(self):
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 100))
            shadow.setOffset(0, 4)
            self.setGraphicsEffect(shadow)
        
        def set_content(
            self,
            title: str,
            description: str,
            step: int,
            total: int,
            can_skip: bool = True,
            is_first: bool = False,
            is_last: bool = False
        ):
            """Set tooltip content."""
            self.title_label.setText(title)
            self.desc_label.setText(description)
            self.progress_label.setText(f"Step {step} of {total}")
            
            self.skip_btn.setVisible(can_skip)
            self.prev_btn.setVisible(not is_first)
            self.next_btn.setText("Finish" if is_last else "Next")
        
        def position_near(
            self,
            target: QWidget,
            position: StepPosition = StepPosition.AUTO
        ):
            """Position tooltip near target widget."""
            if position == StepPosition.AUTO:
                position = self._auto_position(target)
            
            target_global = target.mapToGlobal(QPoint(0, 0))
            target_rect = QRect(target_global, target.size())
            
            # Calculate position
            margin = 16
            tooltip_size = self.sizeHint()
            
            if position == StepPosition.BOTTOM:
                x = target_rect.center().x() - tooltip_size.width() // 2
                y = target_rect.bottom() + margin
            elif position == StepPosition.TOP:
                x = target_rect.center().x() - tooltip_size.width() // 2
                y = target_rect.top() - tooltip_size.height() - margin
            elif position == StepPosition.LEFT:
                x = target_rect.left() - tooltip_size.width() - margin
                y = target_rect.center().y() - tooltip_size.height() // 2
            elif position == StepPosition.RIGHT:
                x = target_rect.right() + margin
                y = target_rect.center().y() - tooltip_size.height() // 2
            else:  # CENTER
                screen = QApplication.primaryScreen().geometry()
                x = (screen.width() - tooltip_size.width()) // 2
                y = (screen.height() - tooltip_size.height()) // 2
            
            # Keep on screen
            screen = QApplication.primaryScreen().geometry()
            x = max(margin, min(x, screen.width() - tooltip_size.width() - margin))
            y = max(margin, min(y, screen.height() - tooltip_size.height() - margin))
            
            self.move(x, y)
        
        def _auto_position(self, target: QWidget) -> StepPosition:
            """Automatically determine best position."""
            screen = QApplication.primaryScreen().geometry()
            target_global = target.mapToGlobal(QPoint(0, 0))
            target_rect = QRect(target_global, target.size())
            tooltip_size = self.sizeHint()
            
            # Check available space in each direction
            space_bottom = screen.bottom() - target_rect.bottom()
            space_top = target_rect.top()
            space_right = screen.right() - target_rect.right()
            space_left = target_rect.left()
            
            # Prefer bottom, then top, then right, then left
            if space_bottom >= tooltip_size.height() + 20:
                return StepPosition.BOTTOM
            elif space_top >= tooltip_size.height() + 20:
                return StepPosition.TOP
            elif space_right >= tooltip_size.width() + 20:
                return StepPosition.RIGHT
            elif space_left >= tooltip_size.width() + 20:
                return StepPosition.LEFT
            else:
                return StepPosition.BOTTOM
    
    
    class TutorialManager(QObject):
        """
        Manages in-app tutorials.
        
        Usage:
            manager = TutorialManager(main_window)
            manager.start_tutorial('quick_start')
        """
        
        tutorial_started = pyqtSignal(str)  # tutorial_id
        tutorial_completed = pyqtSignal(str)  # tutorial_id
        tutorial_skipped = pyqtSignal(str)  # tutorial_id
        step_changed = pyqtSignal(int, int)  # current, total
        
        def __init__(self, main_window: QWidget):
            super().__init__(main_window)
            self.main_window = main_window
            
            # Current state
            self.current_tutorial: Optional[Tutorial] = None
            self.current_step_index: int = 0
            
            # UI components
            self.overlay: Optional[HighlightOverlay] = None
            self.tooltip: Optional[TutorialTooltip] = None
            
            # Built-in tutorials
            self.tutorials: Dict[str, Tutorial] = {}
            self._register_builtin_tutorials()
            
            # Progress tracking
            self.completed_tutorials: set = set()
            self.tutorial_progress: Dict[str, int] = {}
        
        def _register_builtin_tutorials(self):
            """Register built-in tutorials."""
            
            # Quick Start Tutorial
            self.register_tutorial(Tutorial(
                id="quick_start",
                name="Quick Start",
                description="Learn the basics of Enigma Engine in 5 minutes",
                category="getting_started",
                difficulty="beginner",
                estimated_time=5,
                steps=[
                    TutorialStep(
                        target="sidebar",
                        title="Welcome to Enigma Engine",
                        description="This sidebar lets you navigate between different features. Let's explore the main areas.",
                        position=StepPosition.RIGHT,
                    ),
                    TutorialStep(
                        target="chat_tab",
                        title="Chat Interface",
                        description="This is where you can have conversations with your AI. Try typing a message!",
                        position=StepPosition.BOTTOM,
                        action="click",
                    ),
                    TutorialStep(
                        target="chat_input",
                        title="Message Input",
                        description="Type your message here and press Enter or click Send. The AI will respond based on the loaded model.",
                        position=StepPosition.TOP,
                    ),
                    TutorialStep(
                        target="settings_tab",
                        title="Settings",
                        description="Configure your preferences, API keys, and model settings here.",
                        position=StepPosition.BOTTOM,
                        action="click",
                    ),
                    TutorialStep(
                        target="modules_tab",
                        title="Module Manager",
                        description="Enable or disable different AI capabilities. Only load what you need to save resources.",
                        position=StepPosition.BOTTOM,
                    ),
                ]
            ))
            
            # Training Tutorial
            self.register_tutorial(Tutorial(
                id="training_basics",
                name="Training Your Model",
                description="Learn how to train and fine-tune your AI model",
                category="training",
                difficulty="intermediate",
                estimated_time=10,
                requires_training_data=True,
                steps=[
                    TutorialStep(
                        target="training_tab",
                        title="Training Tab",
                        description="This is where you configure and run model training.",
                        position=StepPosition.BOTTOM,
                    ),
                    TutorialStep(
                        target="data_selection",
                        title="Training Data",
                        description="Select your training data files. Supported formats include .txt, .json, and .jsonl",
                        position=StepPosition.RIGHT,
                    ),
                    TutorialStep(
                        target="training_config",
                        title="Training Configuration",
                        description="Adjust learning rate, batch size, and other hyperparameters. The defaults work well for most cases.",
                        position=StepPosition.LEFT,
                    ),
                    TutorialStep(
                        target="start_training_btn",
                        title="Start Training",
                        description="Click here to begin training. You can monitor progress in real-time.",
                        position=StepPosition.TOP,
                    ),
                ]
            ))
            
            # Voice Setup Tutorial
            self.register_tutorial(Tutorial(
                id="voice_setup",
                name="Voice Setup",
                description="Configure text-to-speech and voice input",
                category="voice",
                difficulty="beginner",
                estimated_time=3,
                steps=[
                    TutorialStep(
                        target="voice_tab",
                        title="Voice Settings",
                        description="Configure how Enigma speaks and listens.",
                        position=StepPosition.BOTTOM,
                    ),
                    TutorialStep(
                        target="tts_settings",
                        title="Text-to-Speech",
                        description="Choose a voice and adjust speed, pitch, and volume.",
                        position=StepPosition.RIGHT,
                    ),
                    TutorialStep(
                        target="stt_settings",
                        title="Speech Recognition",
                        description="Enable voice input to speak to your AI instead of typing.",
                        position=StepPosition.RIGHT,
                    ),
                ]
            ))
        
        def register_tutorial(self, tutorial: Tutorial) -> None:
            """Register a tutorial."""
            self.tutorials[tutorial.id] = tutorial
        
        def get_available_tutorials(self) -> List[Tutorial]:
            """Get list of available tutorials."""
            return list(self.tutorials.values())
        
        def start_tutorial(self, tutorial_id: str) -> bool:
            """Start a tutorial by ID."""
            if tutorial_id not in self.tutorials:
                logger.error(f"Tutorial not found: {tutorial_id}")
                return False
            
            tutorial = self.tutorials[tutorial_id]
            
            # Resume from saved progress
            start_step = self.tutorial_progress.get(tutorial_id, 0)
            
            self._start_tutorial_internal(tutorial, start_step)
            return True
        
        def _start_tutorial_internal(self, tutorial: Tutorial, start_step: int = 0):
            """Internal method to start a tutorial."""
            self.current_tutorial = tutorial
            self.current_step_index = start_step
            
            # Create overlay and tooltip
            self._create_ui()
            
            # Emit signal
            self.tutorial_started.emit(tutorial.id)
            
            # Show first step
            self._show_current_step()
        
        def _create_ui(self):
            """Create tutorial UI components."""
            # Overlay covers the main window
            self.overlay = HighlightOverlay(self.main_window)
            self.overlay.setGeometry(self.main_window.rect())
            self.overlay.show()
            
            # Tooltip is a top-level window
            self.tooltip = TutorialTooltip()
            self.tooltip.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.tooltip.next_clicked.connect(self._next_step)
            self.tooltip.prev_clicked.connect(self._prev_step)
            self.tooltip.skip_clicked.connect(self._skip_tutorial)
            self.tooltip.show()
        
        def _cleanup_ui(self):
            """Clean up tutorial UI."""
            if self.overlay:
                self.overlay.close()
                self.overlay = None
            if self.tooltip:
                self.tooltip.close()
                self.tooltip = None
        
        def _show_current_step(self):
            """Display the current tutorial step."""
            if not self.current_tutorial or not self.tooltip:
                return
            
            steps = self.current_tutorial.steps
            if self.current_step_index >= len(steps):
                self._complete_tutorial()
                return
            
            step = steps[self.current_step_index]
            
            # Run on_enter callback
            if step.on_enter:
                try:
                    step.on_enter()
                except Exception as e:
                    logger.error(f"Step on_enter error: {e}")
            
            # Find target widget
            target_widget = self._find_widget(step.target)
            
            # Update overlay highlight
            if self.overlay and step.highlight:
                self.overlay.set_target(target_widget)
            elif self.overlay:
                self.overlay.set_target(None)
            
            # Update tooltip
            self.tooltip.set_content(
                title=step.title,
                description=step.description,
                step=self.current_step_index + 1,
                total=len(steps),
                can_skip=step.can_skip,
                is_first=self.current_step_index == 0,
                is_last=self.current_step_index == len(steps) - 1
            )
            
            if target_widget:
                self.tooltip.position_near(target_widget, step.position)
            
            # Emit signal
            self.step_changed.emit(self.current_step_index + 1, len(steps))
            
            # Auto-advance if configured
            if step.auto_advance:
                QTimer.singleShot(step.auto_advance_delay, self._next_step)
        
        def _find_widget(self, target: str) -> Optional[QWidget]:
            """Find a widget by objectName."""
            if not target:
                return None
            
            # Search in main window
            widget = self.main_window.findChild(QWidget, target)
            if widget:
                return widget
            
            # Try common attribute names
            if hasattr(self.main_window, target):
                attr = getattr(self.main_window, target)
                if isinstance(attr, QWidget):
                    return attr
            
            logger.warning(f"Target widget not found: {target}")
            return None
        
        def _next_step(self):
            """Go to next step."""
            if not self.current_tutorial:
                return
            
            step = self.current_tutorial.steps[self.current_step_index]
            
            # Validate if required
            if step.validate and not step.validate():
                return
            
            # Run on_exit callback
            if step.on_exit:
                try:
                    step.on_exit()
                except Exception as e:
                    logger.error(f"Step on_exit error: {e}")
            
            self.current_step_index += 1
            
            # Save progress
            self.tutorial_progress[self.current_tutorial.id] = self.current_step_index
            
            self._show_current_step()
        
        def _prev_step(self):
            """Go to previous step."""
            if self.current_step_index > 0:
                self.current_step_index -= 1
                self._show_current_step()
        
        def _skip_tutorial(self):
            """Skip the current tutorial."""
            if self.current_tutorial:
                self.tutorial_skipped.emit(self.current_tutorial.id)
            self._cleanup_ui()
            self.current_tutorial = None
        
        def _complete_tutorial(self):
            """Mark tutorial as complete."""
            if self.current_tutorial:
                tutorial_id = self.current_tutorial.id
                self.completed_tutorials.add(tutorial_id)
                if tutorial_id in self.tutorial_progress:
                    del self.tutorial_progress[tutorial_id]
                self.tutorial_completed.emit(tutorial_id)
            
            self._cleanup_ui()
            self.current_tutorial = None
        
        def run_custom_tutorial(self, steps: List[TutorialStep], name: str = "Custom Tutorial") -> None:
            """Run a custom tutorial with provided steps."""
            tutorial = Tutorial(
                id="custom",
                name=name,
                description="Custom tutorial",
                steps=steps
            )
            self._start_tutorial_internal(tutorial)
        
        def is_tutorial_completed(self, tutorial_id: str) -> bool:
            """Check if a tutorial has been completed."""
            return tutorial_id in self.completed_tutorials
        
        def reset_progress(self, tutorial_id: Optional[str] = None):
            """Reset tutorial progress."""
            if tutorial_id:
                self.completed_tutorials.discard(tutorial_id)
                if tutorial_id in self.tutorial_progress:
                    del self.tutorial_progress[tutorial_id]
            else:
                self.completed_tutorials.clear()
                self.tutorial_progress.clear()


else:
    # Stubs when PyQt5 not available
    class TutorialManager:
        def __init__(self, *args, **kwargs):
            pass
        
        def start_tutorial(self, tutorial_id: str) -> bool:
            logger.warning("Tutorials require PyQt5")
            return False
    
    class HighlightOverlay:
        pass
    
    class TutorialTooltip:
        pass


# Export
__all__ = [
    'TutorialManager',
    'Tutorial',
    'TutorialStep',
    'StepPosition',
    'HighlightOverlay',
    'TutorialTooltip',
]
