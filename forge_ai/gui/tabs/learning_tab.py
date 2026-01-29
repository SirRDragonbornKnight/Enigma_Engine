"""
Learning Tab - Self-improvement metrics and progress dashboard.

Displays learning statistics, training examples, and progress charts.
"""

import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QPushButton, QProgressBar, QTextEdit, QCheckBox, QMessageBox,
    QGridLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer

logger = logging.getLogger(__name__)


class LearningTab(QWidget):
    """
    Tab showing self-improvement metrics and progress.
    
    Features:
    - Real-time metrics display
    - Training examples count
    - Learning progress visualization
    - Manual training trigger
    - Autonomous learning toggle
    """
    
    def __init__(self, parent=None):
        """
        Initialize learning tab.
        
        Args:
            parent: Parent window
        """
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
        
        # Start refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_metrics)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Header
        header = QLabel("Self-Improvement System")
        header.setStyleSheet("""
            font-size: 12px;
            font-weight: bold;
            color: #89b4fa;
            padding: 8px;
        """)
        layout.addWidget(header)
        
        # Status indicator
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            color: #f9e2af;
            font-size: 11px;
            padding: 4px 8px;
            background: rgba(249, 226, 175, 0.1);
            border-radius: 4px;
        """)
        layout.addWidget(self.status_label)
        
        # Metrics group
        metrics_group = QGroupBox("Learning Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #a6e3a1;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(8)
        
        # Create metric displays
        self.metrics_widgets = {}
        metrics = [
            ("conversations", "Conversations", "#89b4fa"),
            ("examples", "Training Examples", "#a6e3a1"),
            ("feedback_ratio", "Positive Feedback", "#f9e2af"),
            ("health_score", "Health Score", "#cba6f7"),
            ("avg_quality", "Avg Quality", "#94e2d5")
        ]
        
        row = 0
        col = 0
        for key, label, color in metrics:
            metric_widget = self._create_metric_widget(label, "0", color)
            self.metrics_widgets[key] = metric_widget
            metrics_layout.addWidget(metric_widget, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Training status group
        training_group = QGroupBox("Training Status")
        training_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #f38ba8;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        training_layout = QVBoxLayout()
        
        # Progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                background: #313244;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #89b4fa, stop:1 #a6e3a1);
                border-radius: 3px;
            }
        """)
        self.training_progress.setFormat("%v / %m examples collected")
        training_layout.addWidget(self.training_progress)
        
        # Training info
        self.training_info = QLabel("Collecting examples...")
        self.training_info.setStyleSheet("""
            color: #a6adc8;
            font-size: 11px;
            padding: 4px;
        """)
        self.training_info.setWordWrap(True)
        training_layout.addWidget(self.training_info)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #fab387;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        controls_layout = QHBoxLayout()
        
        # Enable autonomous learning checkbox
        self.enable_learning_checkbox = QCheckBox("Enable Autonomous Learning")
        self.enable_learning_checkbox.setChecked(False)
        self.enable_learning_checkbox.setStyleSheet("""
            QCheckBox {
                color: #cdd6f4;
                font-size: 11px;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #6c7086;
                border-radius: 3px;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                background: #a6e3a1;
                border-color: #a6e3a1;
            }
        """)
        self.enable_learning_checkbox.stateChanged.connect(self.on_learning_toggle)
        controls_layout.addWidget(self.enable_learning_checkbox)
        
        controls_layout.addStretch()
        
        # Train now button
        self.train_now_button = QPushButton("Train Now")
        self.train_now_button.setFixedWidth(100)
        self.train_now_button.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #94e2d5;
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #f38ba8;
                border: 2px dashed #f38ba8;
            }
        """)
        self.train_now_button.clicked.connect(self.on_train_now)
        controls_layout.addWidget(self.train_now_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Recent activity log
        log_group = QGroupBox("Recent Activity")
        log_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #cba6f7;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)
        log_layout = QVBoxLayout()
        
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setStyleSheet("""
            QTextEdit {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        log_layout.addWidget(self.activity_log)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Initial refresh
        self.refresh_metrics()
    
    def _create_metric_widget(self, label: str, value: str, color: str) -> QFrame:
        """Create a metric display widget."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: rgba(49, 50, 68, 0.5);
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"""
            color: {color};
            font-size: 10px;
            font-weight: bold;
        """)
        label_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_widget)
        
        value_widget = QLabel(value)
        value_widget.setObjectName("value")
        value_widget.setStyleSheet(f"""
            color: #cdd6f4;
            font-size: 12px;
            font-weight: bold;
        """)
        value_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_widget)
        
        frame.setLayout(layout)
        return frame
    
    def refresh_metrics(self):
        """Refresh all metrics displays."""
        try:
            # Get model name from parent window
            model_name = getattr(self.parent_window, 'current_model_name', None)
            if not model_name:
                model_name = 'forge_ai'  # Default fallback
            
            # Get learning engine
            from forge_ai.core.self_improvement import get_learning_engine
            engine = get_learning_engine(model_name)
            
            # Get metrics
            metrics = engine.get_metrics()
            queue_stats = engine.get_queue_stats()
            
            # Update metric displays
            self._update_metric('conversations', str(metrics.total_conversations))
            self._update_metric('examples', str(queue_stats.get('total_examples', 0)))
            self._update_metric('feedback_ratio', f"{metrics.feedback_ratio():.0%}")
            self._update_metric('health_score', f"{metrics.health_score():.0%}")
            self._update_metric('avg_quality', f"{metrics.avg_response_quality:.2f}")
            
            # Update training progress
            from forge_ai.learning.training_scheduler import get_training_scheduler
            scheduler = get_training_scheduler(model_name)
            status = scheduler.get_status()
            
            examples_collected = status['examples_collected']
            min_needed = status['min_examples_needed']
            self.training_progress.setMaximum(min_needed)
            self.training_progress.setValue(examples_collected)
            
            # Update training info
            if status['training_in_progress']:
                self.training_info.setText("Training in progress...")
                self.train_now_button.setEnabled(False)
            elif status['ready_to_train']:
                self.training_info.setText(f"Ready to train! {examples_collected} examples collected.")
                self.train_now_button.setEnabled(True)
            else:
                self.training_info.setText(
                    f"Collecting examples... {examples_collected}/{min_needed} needed. "
                    f"Next training available in {24 - status.get('hours_since_training', 24):.1f} hours."
                )
                self.train_now_button.setEnabled(False)
            
            # Update status
            if metrics.total_conversations == 0:
                self.status_label.setText("Waiting for conversations to begin learning...")
            else:
                health = metrics.health_score()
                if health > 0.7:
                    status_text = "System healthy - learning actively"
                    color = "#a6e3a1"
                elif health > 0.4:
                    status_text = "System learning - moderate progress"
                    color = "#f9e2af"
                else:
                    status_text = "System needs more feedback"
                    color = "#f38ba8"
                
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet(f"""
                    color: {color};
                    font-size: 11px;
                    padding: 4px 8px;
                    background: rgba(249, 226, 175, 0.1);
                    border-radius: 4px;
                """)
        
        except Exception as e:
            logger.error(f"Error refreshing metrics: {e}", exc_info=True)
            self.status_label.setText(f"Error loading metrics: {str(e)}")
    
    def _update_metric(self, key: str, value: str):
        """Update a metric display value."""
        if key in self.metrics_widgets:
            value_label = self.metrics_widgets[key].findChild(QLabel, "value")
            if value_label:
                value_label.setText(value)
    
    def on_learning_toggle(self, state):
        """Handle autonomous learning toggle."""
        enabled = state == Qt.Checked
        
        try:
            model_name = getattr(self.parent_window, 'current_model_name', 'forge_ai')
            
            # Import and control autonomous learning system
            try:
                from ..core.autonomous import (
                    get_autonomous_learner,
                    AutonomousLearner,
                    AutonomousConfig
                )
                
                learner = get_autonomous_learner(model_name)
                
                if enabled:
                    # Start autonomous learning
                    if not learner.is_active():
                        config = AutonomousConfig(
                            enable_reflection=True,
                            enable_practice=True,
                            enable_dreaming=False,  # Can be resource intensive
                            practice_interval_s=300,  # 5 minutes
                        )
                        learner.start(config)
                        self.log_activity("Autonomous learning enabled - learner started")
                    else:
                        self.log_activity("Autonomous learning already active")
                else:
                    # Stop autonomous learning
                    if learner.is_active():
                        learner.stop()
                        self.log_activity("Autonomous learning disabled - learner stopped")
                    else:
                        self.log_activity("Autonomous learning already inactive")
                        
            except ImportError:
                # Fallback: just log the state change
                if enabled:
                    self.log_activity("Autonomous learning enabled (core module not loaded)")
                else:
                    self.log_activity("Autonomous learning disabled")
            
            if enabled:
                QMessageBox.information(
                    self,
                    "Autonomous Learning",
                    "Autonomous learning has been enabled.\n\n"
                    "The AI will now:\n"
                    "- Reflect on conversations\n"
                    "- Practice responses\n"
                    "- Learn from feedback\n"
                    "- Evolve personality based on interactions"
                )
        
        except Exception as e:
            logger.error(f"Error toggling learning: {e}")
            QMessageBox.warning(self, "Error", f"Could not toggle learning: {e}")
    
    def on_train_now(self):
        """Manually trigger training."""
        try:
            model_name = getattr(self.parent_window, 'current_model_name', 'forge_ai')
            
            from forge_ai.learning.training_scheduler import get_training_scheduler
            scheduler = get_training_scheduler(model_name)
            
            # Confirm with user
            reply = QMessageBox.question(
                self,
                "Start Training",
                "Start LoRA training now with collected examples?\n\n"
                "This will:\n"
                "- Use all high-quality training examples\n"
                "- Fine-tune the model with LoRA\n"
                "- Save the trained adapter\n\n"
                "Training may take several minutes.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.log_activity("Starting manual training...")
                self.train_now_button.setEnabled(False)
                
                # Run training in background
                import threading
                def train():
                    success = scheduler.run_training()
                    if success:
                        self.log_activity("Training completed successfully!")
                    else:
                        self.log_activity("Training failed - check logs")
                
                thread = threading.Thread(target=train, daemon=True)
                thread.start()
        
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            QMessageBox.critical(self, "Training Error", f"Could not start training: {e}")
    
    def log_activity(self, message: str):
        """Add message to activity log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.append(f"[{timestamp}] {message}")
