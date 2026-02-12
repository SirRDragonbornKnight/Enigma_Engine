"""
Task Offloading Tab - GUI for distributing work across devices

Features:
  - Per-task device assignment (image gen, code gen, video gen, etc.)
  - Task queue visualization
  - Real-time distribution monitoring
  - Load balancing configuration
  - Task routing rules
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be offloaded."""
    CHAT_INFERENCE = "Chat / Inference"
    IMAGE_GENERATION = "Image Generation"
    CODE_GENERATION = "Code Generation"
    VIDEO_GENERATION = "Video Generation"
    AUDIO_GENERATION = "Audio / TTS"
    THREE_D_GENERATION = "3D Model Generation"
    EMBEDDINGS = "Embeddings / Search"
    VISION_ANALYSIS = "Vision / Image Analysis"
    TRAINING = "Model Training"
    DATA_PROCESSING = "Data Processing"


class RoutingMode(Enum):
    """How tasks are routed to devices."""
    AUTO = "Automatic (Load Balanced)"
    ROUND_ROBIN = "Round Robin"
    LEAST_LOADED = "Least Loaded Device"
    FASTEST = "Fastest Response Time"
    SPECIFIC = "Specific Device"
    CAPABILITY = "By Capability"


@dataclass
class TaskConfig:
    """Configuration for a task type."""
    task_type: TaskType
    enabled: bool = True
    routing_mode: RoutingMode = RoutingMode.AUTO
    specific_device: Optional[str] = None
    fallback_local: bool = True
    priority: int = 5  # 1-10
    timeout_seconds: int = 300
    max_retries: int = 2


@dataclass
class QueuedTask:
    """A task in the queue."""
    id: str
    task_type: TaskType
    device: str
    status: str  # pending, running, completed, failed
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_preview: str = ""
    output_preview: str = ""
    error: Optional[str] = None


class DeviceCapabilityWidget(QWidget):
    """Shows capabilities of a device with checkboxes."""
    
    def __init__(self, device_id: str, device_info: dict, parent=None):
        super().__init__(parent)
        self.device_id = device_id
        self.device_info = device_info
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Device header
        header = QHBoxLayout()
        
        # Status indicator
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self._update_status()
        header.addWidget(self.status_dot)
        
        # Name
        name = self.device_info.get("name", self.device_id)
        name_label = QLabel(f"<b>{name}</b>")
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Load indicator
        load = self.device_info.get("load", 0)
        self.load_bar = QProgressBar()
        self.load_bar.setFixedWidth(60)
        self.load_bar.setFixedHeight(12)
        self.load_bar.setValue(int(load * 100))
        self.load_bar.setFormat("")
        header.addWidget(self.load_bar)
        
        layout.addLayout(header)
        
        # Capabilities grid
        caps_layout = QGridLayout()
        caps_layout.setSpacing(4)
        
        capabilities = [
            ("gpu", "GPU"),
            ("chat", "Chat"),
            ("image", "Image"),
            ("code", "Code"),
            ("video", "Video"),
            ("audio", "Audio"),
            ("3d", "3D"),
            ("vision", "Vision"),
            ("training", "Train"),
        ]
        
        col = 0
        for cap_key, cap_name in capabilities:
            has_cap = self.device_info.get("capabilities", {}).get(cap_key, False)
            label = QLabel(cap_name)
            if has_cap:
                label.setStyleSheet("color: #a6e3a1; font-size: 9pt;")
            else:
                label.setStyleSheet("color: #585b70; font-size: 9pt;")
            caps_layout.addWidget(label, 0, col)
            col += 1
        
        layout.addLayout(caps_layout)
    
    def _update_status(self):
        health = self.device_info.get("health", "unknown")
        colors = {
            "healthy": "#a6e3a1",
            "degraded": "#f9e2af",
            "unhealthy": "#f38ba8",
            "unknown": "#585b70",
        }
        color = colors.get(health, "#585b70")
        self.status_dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;"
        )


class TaskRouteWidget(QWidget):
    """Widget for configuring routing of a single task type."""
    
    changed = pyqtSignal(TaskType, TaskConfig)
    
    def __init__(self, task_type: TaskType, devices: list[str], parent=None):
        super().__init__(parent)
        self.task_type = task_type
        self.devices = devices
        self.config = TaskConfig(task_type=task_type)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Enable checkbox
        self.enable_check = QCheckBox()
        self.enable_check.setChecked(self.config.enabled)
        self.enable_check.stateChanged.connect(self._on_changed)
        layout.addWidget(self.enable_check)
        
        # Task name
        name_label = QLabel(self.task_type.value)
        name_label.setMinimumWidth(150)
        layout.addWidget(name_label)
        
        # Routing mode
        self.routing_combo = QComboBox()
        for mode in RoutingMode:
            self.routing_combo.addItem(mode.value, mode)
        self.routing_combo.currentIndexChanged.connect(self._on_routing_changed)
        layout.addWidget(self.routing_combo)
        
        # Specific device (shown when routing=SPECIFIC)
        self.device_combo = QComboBox()
        self.device_combo.addItem("(Any)", None)
        for device in self.devices:
            self.device_combo.addItem(device, device)
        self.device_combo.setEnabled(False)
        self.device_combo.currentIndexChanged.connect(self._on_changed)
        layout.addWidget(self.device_combo)
        
        # Priority
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(1, 10)
        self.priority_spin.setValue(self.config.priority)
        self.priority_spin.setFixedWidth(50)
        self.priority_spin.valueChanged.connect(self._on_changed)
        layout.addWidget(QLabel("Priority:"))
        layout.addWidget(self.priority_spin)
        
        # Fallback checkbox
        self.fallback_check = QCheckBox("Fallback to local")
        self.fallback_check.setChecked(self.config.fallback_local)
        self.fallback_check.stateChanged.connect(self._on_changed)
        layout.addWidget(self.fallback_check)
        
        layout.addStretch()
    
    def _on_routing_changed(self, index: int):
        mode = self.routing_combo.currentData()
        self.device_combo.setEnabled(mode == RoutingMode.SPECIFIC)
        self._on_changed()
    
    def _on_changed(self):
        self.config.enabled = self.enable_check.isChecked()
        self.config.routing_mode = self.routing_combo.currentData()
        self.config.specific_device = self.device_combo.currentData()
        self.config.priority = self.priority_spin.value()
        self.config.fallback_local = self.fallback_check.isChecked()
        
        self.changed.emit(self.task_type, self.config)
    
    def update_devices(self, devices: list[str]):
        """Update device list."""
        current = self.device_combo.currentData()
        self.device_combo.clear()
        self.device_combo.addItem("(Any)", None)
        for device in devices:
            self.device_combo.addItem(device, device)
        
        # Restore selection
        if current:
            idx = self.device_combo.findData(current)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)


class TaskQueueWidget(QWidget):
    """Shows the current task queue."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tasks: list[QueuedTask] = []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Task Queue</b>"))
        header.addStretch()
        
        self.clear_btn = QPushButton("Clear Completed")
        self.clear_btn.setFixedHeight(24)
        self.clear_btn.clicked.connect(self._clear_completed)
        header.addWidget(self.clear_btn)
        
        layout.addLayout(header)
        
        # Task table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Type", "Device", "Status", "Progress", "Input", "Started"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(3, 80)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.table)
    
    def add_task(self, task: QueuedTask):
        """Add a task to the queue display."""
        self._tasks.append(task)
        self._refresh_table()
    
    def update_task(self, task_id: str, **updates):
        """Update a task in the queue."""
        for task in self._tasks:
            if task.id == task_id:
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                break
        self._refresh_table()
    
    def remove_task(self, task_id: str):
        """Remove a task from the queue."""
        self._tasks = [t for t in self._tasks if t.id != task_id]
        self._refresh_table()
    
    def _clear_completed(self):
        """Clear completed tasks."""
        self._tasks = [t for t in self._tasks if t.status not in ("completed", "failed")]
        self._refresh_table()
    
    def _refresh_table(self):
        """Refresh the task table."""
        self.table.setRowCount(len(self._tasks))
        
        for row, task in enumerate(self._tasks):
            # Type
            type_item = QTableWidgetItem(task.task_type.value)
            self.table.setItem(row, 0, type_item)
            
            # Device
            device_item = QTableWidgetItem(task.device or "Pending")
            self.table.setItem(row, 1, device_item)
            
            # Status
            status_item = QTableWidgetItem(task.status.capitalize())
            status_colors = {
                "pending": QColor("#cdd6f4"),
                "running": QColor("#89b4fa"),
                "completed": QColor("#a6e3a1"),
                "failed": QColor("#f38ba8"),
            }
            status_item.setForeground(QBrush(status_colors.get(task.status, QColor("#cdd6f4"))))
            self.table.setItem(row, 2, status_item)
            
            # Progress
            progress_item = QTableWidgetItem(f"{int(task.progress * 100)}%")
            self.table.setItem(row, 3, progress_item)
            
            # Input preview
            input_item = QTableWidgetItem(task.input_preview[:30] + "..." if len(task.input_preview) > 30 else task.input_preview)
            self.table.setItem(row, 4, input_item)
            
            # Started time
            if task.started_at:
                started = task.started_at.strftime("%H:%M:%S")
            else:
                started = "-"
            time_item = QTableWidgetItem(started)
            self.table.setItem(row, 5, time_item)


class LoadBalancingWidget(QWidget):
    """Load balancing configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QFormLayout(self)
        
        # Strategy
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "Weighted Round Robin",
            "Least Connections",
            "Least Response Time",
            "Random",
            "Resource Based",
        ])
        layout.addRow("Strategy:", self.strategy_combo)
        
        # Health check interval
        self.health_interval = QSpinBox()
        self.health_interval.setRange(1, 60)
        self.health_interval.setValue(5)
        self.health_interval.setSuffix(" sec")
        layout.addRow("Health Check:", self.health_interval)
        
        # Failure threshold
        self.failure_threshold = QSpinBox()
        self.failure_threshold.setRange(1, 10)
        self.failure_threshold.setValue(3)
        layout.addRow("Failure Threshold:", self.failure_threshold)
        
        # Max concurrent per device
        self.max_concurrent = QSpinBox()
        self.max_concurrent.setRange(1, 100)
        self.max_concurrent.setValue(10)
        layout.addRow("Max Concurrent/Device:", self.max_concurrent)
        
        # Sticky sessions
        self.sticky_sessions = QCheckBox("Enable session affinity")
        layout.addRow("", self.sticky_sessions)
        
        # Weight slider
        weight_layout = QHBoxLayout()
        self.local_weight = QSlider(Qt.Horizontal)
        self.local_weight.setRange(0, 100)
        self.local_weight.setValue(50)
        self.local_weight.valueChanged.connect(self._on_weight_changed)
        weight_layout.addWidget(QLabel("Local"))
        weight_layout.addWidget(self.local_weight)
        weight_layout.addWidget(QLabel("Remote"))
        
        self.weight_label = QLabel("50%")
        weight_layout.addWidget(self.weight_label)
        
        layout.addRow("Local/Remote Balance:", weight_layout)
    
    def _on_weight_changed(self, value: int):
        self.weight_label.setText(f"{value}%")


class TaskOffloadingTab(QWidget):
    """
    Task Offloading Tab - Distribute work across connected devices.
    
    Features:
    - Per-task type routing configuration
    - Real-time task queue monitoring
    - Device capability overview
    - Load balancing settings
    - Statistics and metrics
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._devices: dict[str, dict] = {}
        self._task_configs: dict[TaskType, TaskConfig] = {}
        self._setup_ui()
        self._setup_refresh_timer()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Top: Device overview
        devices_group = QGroupBox("Connected Devices")
        devices_layout = QVBoxLayout(devices_group)
        
        # Device scroll area
        self.device_scroll = QScrollArea()
        self.device_scroll.setWidgetResizable(True)
        self.device_scroll.setFixedHeight(100)
        self.device_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.device_container = QWidget()
        self.device_flow = QHBoxLayout(self.device_container)
        self.device_flow.setAlignment(Qt.AlignLeft)
        self.device_scroll.setWidget(self.device_container)
        
        devices_layout.addWidget(self.device_scroll)
        layout.addWidget(devices_group)
        
        # Main content in tabs
        main_tabs = QTabWidget()
        
        # Tab 1: Task Routing
        routing_tab = QWidget()
        routing_layout = QVBoxLayout(routing_tab)
        
        # Routing header
        routing_header = QHBoxLayout()
        routing_header.addWidget(QLabel("<b>Task Routing Rules</b>"))
        routing_header.addStretch()
        
        self.enable_all_btn = QPushButton("Enable All")
        self.enable_all_btn.clicked.connect(self._enable_all_tasks)
        routing_header.addWidget(self.enable_all_btn)
        
        self.disable_all_btn = QPushButton("Disable All")
        self.disable_all_btn.clicked.connect(self._disable_all_tasks)
        routing_header.addWidget(self.disable_all_btn)
        
        routing_layout.addLayout(routing_header)
        
        # Task route widgets
        self.route_widgets: dict[TaskType, TaskRouteWidget] = {}
        
        for task_type in TaskType:
            widget = TaskRouteWidget(task_type, list(self._devices.keys()))
            widget.changed.connect(self._on_task_config_changed)
            self.route_widgets[task_type] = widget
            routing_layout.addWidget(widget)
        
        routing_layout.addStretch()
        main_tabs.addTab(routing_tab, "Task Routing")
        
        # Tab 2: Queue
        queue_tab = QWidget()
        queue_layout = QVBoxLayout(queue_tab)
        self.queue_widget = TaskQueueWidget()
        queue_layout.addWidget(self.queue_widget)
        main_tabs.addTab(queue_tab, "Task Queue")
        
        # Tab 3: Load Balancing
        lb_tab = QWidget()
        lb_layout = QVBoxLayout(lb_tab)
        self.lb_widget = LoadBalancingWidget()
        lb_layout.addWidget(self.lb_widget)
        lb_layout.addStretch()
        main_tabs.addTab(lb_tab, "Load Balancing")
        
        # Tab 4: Statistics
        stats_tab = self._create_stats_tab()
        main_tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(main_tabs, 1)
        
        # Bottom: Action buttons
        actions = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Configuration")
        self.apply_btn.clicked.connect(self._apply_config)
        actions.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_config)
        actions.addWidget(self.reset_btn)
        
        actions.addStretch()
        
        self.export_btn = QPushButton("Export Config")
        self.export_btn.clicked.connect(self._export_config)
        actions.addWidget(self.export_btn)
        
        self.import_btn = QPushButton("Import Config")
        self.import_btn.clicked.connect(self._import_config)
        actions.addWidget(self.import_btn)
        
        layout.addLayout(actions)
    
    def _create_stats_tab(self) -> QWidget:
        """Create statistics tab."""
        tab = QWidget()
        layout = QGridLayout(tab)
        
        # Local vs Remote distribution
        dist_group = QGroupBox("Distribution")
        dist_layout = QFormLayout(dist_group)
        
        self.stat_local = QLabel("0")
        dist_layout.addRow("Local Tasks:", self.stat_local)
        
        self.stat_remote = QLabel("0")
        dist_layout.addRow("Remote Tasks:", self.stat_remote)
        
        self.stat_pending = QLabel("0")
        dist_layout.addRow("Pending:", self.stat_pending)
        
        layout.addWidget(dist_group, 0, 0)
        
        # Performance
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.stat_avg_latency = QLabel("0 ms")
        perf_layout.addRow("Avg Latency:", self.stat_avg_latency)
        
        self.stat_throughput = QLabel("0 tasks/min")
        perf_layout.addRow("Throughput:", self.stat_throughput)
        
        self.stat_success_rate = QLabel("0%")
        perf_layout.addRow("Success Rate:", self.stat_success_rate)
        
        layout.addWidget(perf_group, 0, 1)
        
        # Errors
        error_group = QGroupBox("Errors")
        error_layout = QFormLayout(error_group)
        
        self.stat_failures = QLabel("0")
        error_layout.addRow("Failed Tasks:", self.stat_failures)
        
        self.stat_timeouts = QLabel("0")
        error_layout.addRow("Timeouts:", self.stat_timeouts)
        
        self.stat_fallbacks = QLabel("0")
        error_layout.addRow("Fallbacks:", self.stat_fallbacks)
        
        layout.addWidget(error_group, 1, 0)
        
        # Per-device stats
        device_group = QGroupBox("Per-Device")
        device_layout = QVBoxLayout(device_group)
        
        self.device_stats_table = QTableWidget()
        self.device_stats_table.setColumnCount(4)
        self.device_stats_table.setHorizontalHeaderLabels([
            "Device", "Tasks", "Avg Time", "Success"
        ])
        self.device_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        device_layout.addWidget(self.device_stats_table)
        
        layout.addWidget(device_group, 1, 1)
        
        return tab
    
    def _setup_refresh_timer(self):
        """Setup periodic refresh."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start(5000)  # 5 seconds
    
    def _refresh(self):
        """Refresh device list and stats."""
        self._refresh_devices()
        self._refresh_stats()
    
    def _refresh_devices(self):
        """Update device list."""
        try:
            from enigma_engine.comms.discovery import DeviceDiscovery
            
            discovery = DeviceDiscovery("Enigma AI Engine", 5000)
            self._devices = discovery.discovered or {}
            
            # Clear device container
            for i in reversed(range(self.device_flow.count())):
                widget = self.device_flow.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # Add device widgets
            for device_id, info in self._devices.items():
                widget = DeviceCapabilityWidget(device_id, info)
                widget.setFixedWidth(180)
                self.device_flow.addWidget(widget)
            
            # Add "No devices" label if empty
            if not self._devices:
                label = QLabel("No devices connected. Go to Network tab to add devices.")
                label.setStyleSheet("color: #6c7086;")
                self.device_flow.addWidget(label)
            
            # Update route widgets
            device_list = list(self._devices.keys())
            for widget in self.route_widgets.values():
                widget.update_devices(device_list)
                
        except Exception as e:
            logger.debug(f"Device refresh: {e}")
    
    def _refresh_stats(self):
        """Update statistics."""
        try:
            from enigma_engine.network import get_inference_gateway
            gateway = get_inference_gateway()
            stats = gateway.get_stats()
            
            self.stat_local.setText(str(stats.get("local_requests", 0)))
            self.stat_remote.setText(str(stats.get("remote_requests", 0)))
            self.stat_pending.setText(str(stats.get("pending", 0)))
            self.stat_failures.setText(str(stats.get("failures", 0)))
            self.stat_timeouts.setText(str(stats.get("timeouts", 0)))
            self.stat_fallbacks.setText(str(stats.get("fallbacks", 0)))
            
            avg_latency = stats.get("avg_latency_ms", 0)
            self.stat_avg_latency.setText(f"{avg_latency:.0f} ms")
            
            throughput = stats.get("throughput", 0)
            self.stat_throughput.setText(f"{throughput:.1f} tasks/min")
            
            success_rate = stats.get("success_rate", 1.0)
            self.stat_success_rate.setText(f"{success_rate * 100:.1f}%")
            
        except Exception:
            pass  # Intentionally silent
    
    def _on_task_config_changed(self, task_type: TaskType, config: TaskConfig):
        """Handle task configuration change."""
        self._task_configs[task_type] = config
        logger.debug(f"Task config changed: {task_type.value} -> {config.routing_mode.value}")
    
    def _enable_all_tasks(self):
        """Enable all tasks for offloading."""
        for widget in self.route_widgets.values():
            widget.enable_check.setChecked(True)
    
    def _disable_all_tasks(self):
        """Disable all tasks for offloading."""
        for widget in self.route_widgets.values():
            widget.enable_check.setChecked(False)
    
    def _apply_config(self):
        """Apply current configuration."""
        try:
            from enigma_engine.network import get_inference_gateway
            gateway = get_inference_gateway()
            
            # Apply task configs
            for task_type, config in self._task_configs.items():
                # Convert to gateway format
                gateway.set_task_routing(
                    task_type=task_type.name.lower(),
                    mode=config.routing_mode.name.lower(),
                    device=config.specific_device,
                    fallback=config.fallback_local,
                    priority=config.priority,
                )
            
            logger.info("Task offloading configuration applied")
            
        except Exception as e:
            logger.error(f"Failed to apply config: {e}")
    
    def _reset_config(self):
        """Reset to default configuration."""
        for widget in self.route_widgets.values():
            widget.enable_check.setChecked(True)
            widget.routing_combo.setCurrentIndex(0)  # Auto
            widget.priority_spin.setValue(5)
            widget.fallback_check.setChecked(True)
    
    def _export_config(self):
        """Export configuration to file."""
        import json
        from PyQt5.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "", "JSON files (*.json)"
        )
        
        if path:
            config = {}
            for task_type, task_config in self._task_configs.items():
                config[task_type.name] = {
                    "enabled": task_config.enabled,
                    "routing_mode": task_config.routing_mode.name,
                    "specific_device": task_config.specific_device,
                    "fallback_local": task_config.fallback_local,
                    "priority": task_config.priority,
                }
            
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
    
    def _import_config(self):
        """Import configuration from file."""
        import json
        from PyQt5.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Configuration", "", "JSON files (*.json)"
        )
        
        if path:
            try:
                with open(path) as f:
                    config = json.load(f)
                
                for task_name, task_config in config.items():
                    try:
                        task_type = TaskType[task_name]
                        widget = self.route_widgets.get(task_type)
                        if widget:
                            widget.enable_check.setChecked(task_config.get("enabled", True))
                            
                            mode = RoutingMode[task_config.get("routing_mode", "AUTO")]
                            idx = widget.routing_combo.findData(mode)
                            if idx >= 0:
                                widget.routing_combo.setCurrentIndex(idx)
                            
                            widget.priority_spin.setValue(task_config.get("priority", 5))
                            widget.fallback_check.setChecked(task_config.get("fallback_local", True))
                    except (KeyError, ValueError):
                        pass  # Intentionally silent
                        
            except Exception as e:
                logger.error(f"Failed to import config: {e}")
