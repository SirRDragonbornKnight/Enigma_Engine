"""
Dashboard Tab - Visual system overview and quick actions.

Features:
  - System resource monitoring (CPU, RAM, Disk, Network, Temp)
  - Historical usage charts
  - Module status overview with more modules
  - System alerts for high temp/RAM/CPU
  - Quick action buttons (customizable)
  - Recent activity feed
  - Visual gauges and progress indicators
  - Pi-friendly - no AI dependencies
"""

import platform
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# === Storage for dashboard settings ===
DASHBOARD_DIR = Path.home() / ".enigma_engine" / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_FILE = DASHBOARD_DIR / "alerts.json"
CUSTOM_ACTIONS_FILE = DASHBOARD_DIR / "custom_actions.json"


class QuickActionButton(QPushButton):
    """A styled quick action button with icon and color."""
    
    def __init__(self, text: str, icon_text: str = "", color: str = "#89b4fa", parent=None):
        super().__init__(parent)
        # Show both icon and text if icon provided
        display_text = f"{icon_text} {text}" if icon_text else text
        self.setText(display_text)
        self.setToolTip(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {color};
                filter: brightness(1.1);
            }}
            QPushButton:pressed {{
                background-color: {color};
                filter: brightness(0.9);
            }}
        """)


class CircularGauge(QWidget):
    """A circular gauge widget for displaying percentages."""
    
    def __init__(self, title: str = "", value: int = 0, color: str = "#89b4fa", parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.color = QColor(color)
        self.setMinimumSize(100, 120)
        self.setMaximumSize(130, 150)
    
    def set_value(self, value: int):
        """Update the gauge value (0-100)."""
        self.value = max(0, min(100, value))
        self.update()
    
    def set_color(self, color: str):
        """Update the gauge color."""
        self.color = QColor(color)
        self.update()
    
    def paintEvent(self, event):
        """Draw the circular gauge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        size = min(width, height - 25)
        x = (width - size) // 2
        y = 5
        
        # Background circle
        painter.setPen(QPen(QColor("#45475a"), 6))
        painter.drawArc(x + 8, y + 8, size - 16, size - 16, 0, 360 * 16)
        
        # Value arc
        if self.value > 0:
            if self.value < 50:
                arc_color = self.color
            elif self.value < 75:
                arc_color = QColor("#f9e2af")
            else:
                arc_color = QColor("#f38ba8")
            
            painter.setPen(QPen(arc_color, 6, Qt.SolidLine, Qt.RoundCap))
            start_angle = 90 * 16
            span_angle = -int(self.value * 3.6 * 16)
            painter.drawArc(x + 8, y + 8, size - 16, size - 16, start_angle, span_angle)
        
        # Percentage text
        painter.setPen(QColor("#cdd6f4"))
        font = QFont("Segoe UI", 14, QFont.Bold)
        painter.setFont(font)
        painter.drawText(x, y, size, size, Qt.AlignCenter, f"{self.value}%")
        
        # Title
        painter.setPen(QColor("#bac2de"))
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.drawText(0, size + 2, width, 20, Qt.AlignCenter, self.title)


class MiniLineChart(QWidget):
    """A mini line chart for historical data."""
    
    def __init__(self, title: str = "", color: str = "#89b4fa", max_points: int = 30, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = QColor(color)
        self.data = deque(maxlen=max_points)
        self.setMinimumSize(150, 80)
        self.setMaximumHeight(100)
    
    def add_value(self, value: float):
        """Add a new value to the chart."""
        self.data.append(max(0, min(100, value)))
        self.update()
    
    def paintEvent(self, event):
        """Draw the line chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        margin = 5
        chart_height = height - 25
        chart_width = width - margin * 2
        
        # Background
        painter.fillRect(margin, margin, chart_width, chart_height, QColor("#1e1e2e"))
        
        # Grid lines
        painter.setPen(QPen(QColor("#313244"), 1))
        for i in range(5):
            y = margin + (chart_height * i // 4)
            painter.drawLine(margin, y, width - margin, y)
        
        # Draw line
        if len(self.data) > 1:
            painter.setPen(QPen(self.color, 2))
            points = list(self.data)
            step = chart_width / (len(points) - 1) if len(points) > 1 else chart_width
            
            path = QPainterPath()
            for i, val in enumerate(points):
                x = margin + i * step
                y = margin + chart_height - (val / 100 * chart_height)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.drawPath(path)
            
            # Fill under line
            fill_path = QPainterPath(path)
            fill_path.lineTo(margin + (len(points) - 1) * step, margin + chart_height)
            fill_path.lineTo(margin, margin + chart_height)
            fill_path.closeSubpath()
            
            fill_color = QColor(self.color)
            fill_color.setAlpha(50)
            painter.fillPath(fill_path, fill_color)
        
        # Title
        painter.setPen(QColor("#bac2de"))
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.drawText(0, height - 18, width, 18, Qt.AlignCenter, self.title)


class AlertBanner(QFrame):
    """A banner for showing system alerts."""
    
    dismissed = pyqtSignal(str)
    
    def __init__(self, message: str, level: str = "warning", parent=None):
        super().__init__(parent)
        self.message = message
        self.level = level
        
        colors = {
            "info": ("#89b4fa", "#1e1e2e"),
            "warning": ("#f9e2af", "#1e1e2e"),
            "critical": ("#f38ba8", "#1e1e2e"),
        }
        bg, fg = colors.get(level, colors["warning"])
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        
        icons = {"info": "i", "warning": "!", "critical": "!!"}
        icon_label = QLabel(icons.get(level, "!"))
        layout.addWidget(icon_label)
        
        msg_label = QLabel(message)
        msg_label.setStyleSheet(f"color: {fg}; font-weight: bold; font-size: 11px;")
        layout.addWidget(msg_label, stretch=1)
        
        dismiss_btn = QPushButton("X")
        dismiss_btn.setFixedSize(20, 20)
        dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {fg};
                border: none;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: rgba(0,0,0,0.2);
                border-radius: 10px;
            }}
        """)
        dismiss_btn.clicked.connect(lambda: self.dismissed.emit(message))
        layout.addWidget(dismiss_btn)


class StatusCard(QFrame):
    """A status card showing module or system status."""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, title: str, status: str = "Unknown", 
                 icon: str = "[*]", color: str = "#6c7086", parent=None):
        super().__init__(parent)
        self.title = title
        self.status = status
        self.color = color
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        layout.addWidget(self.icon_label)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: bold;")
        text_layout.addWidget(self.title_label)
        
        self.status_label = QLabel(status)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        text_layout.addWidget(self.status_label)
        
        layout.addLayout(text_layout)
        layout.addStretch()
    
    def _update_style(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                border-left: 3px solid {self.color};
            }}
            QFrame:hover {{
                background-color: #3b3d52;
            }}
        """)
    
    def set_status(self, status: str, color: str = None):
        self.status = status
        if color:
            self.color = color
            self.icon_label.setStyleSheet(f"color: {color}; font-size: 12px;")
            self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")
            self._update_style()
        self.status_label.setText(status)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.title)
        super().mousePressEvent(event)


class ActivityItem(QFrame):
    """A single activity log item."""
    
    def __init__(self, message: str, timestamp: str, icon: str = ">", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: none;
                border-bottom: 1px solid #313244;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(icon_label)
        
        msg_label = QLabel(message)
        msg_label.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label, stretch=1)
        
        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        layout.addWidget(time_label)


class DashboardTab(QWidget):
    """Main dashboard widget showing system overview."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.activity_items = []
        self.alerts = []
        
        # Historical data storage
        self.cpu_history = deque(maxlen=30)
        self.ram_history = deque(maxlen=30)
        self.net_history = deque(maxlen=30)
        
        # Previous network stats for calculating speed
        self.prev_net_io = psutil.net_io_counters()
        self.prev_net_time = datetime.now()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_high': 85,
            'ram_high': 85,
            'temp_high': 75,
            'disk_high': 90,
        }
        self.dismissed_alerts = set()
        
        self._setup_ui()
        self._start_monitoring()
    
    def _setup_ui(self):
        """Set up the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # === Header ===
        header_layout = QHBoxLayout()
        
        title = QLabel("Dashboard")
        title.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Uptime
        self.uptime_label = QLabel("Up: --")
        self.uptime_label.setStyleSheet("color: #bac2de; font-size: 12px;")
        header_layout.addWidget(self.uptime_label)
        
        header_layout.addSpacing(15)
        
        refresh_btn = QuickActionButton("Refresh", "R", "#a6e3a1")
        refresh_btn.clicked.connect(self._refresh_all)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # === Alerts Container ===
        self.alerts_container = QVBoxLayout()
        self.alerts_container.setSpacing(5)
        layout.addLayout(self.alerts_container)
        
        # === System Info Banner ===
        info_banner = QFrame()
        info_banner.setStyleSheet("""
            QFrame {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 8px;
            }
        """)
        info_layout = QHBoxLayout(info_banner)
        info_layout.setContentsMargins(12, 8, 12, 8)
        
        self.hostname_label = QLabel(f"Host: {platform.node()}")
        self.hostname_label.setStyleSheet("color: #89b4fa; font-size: 11px; font-weight: bold;")
        info_layout.addWidget(self.hostname_label)
        
        info_layout.addSpacing(20)
        
        self.os_label = QLabel(f"OS: {platform.system()} {platform.release()[:20]}")
        self.os_label.setStyleSheet("color: #a6e3a1; font-size: 12px;")
        info_layout.addWidget(self.os_label)
        
        info_layout.addSpacing(20)
        
        self.python_label = QLabel(f"Python {platform.python_version()}")
        self.python_label.setStyleSheet("color: #f9e2af; font-size: 12px;")
        info_layout.addWidget(self.python_label)
        
        info_layout.addStretch()
        
        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #bac2de; font-size: 12px;")
        info_layout.addWidget(self.time_label)
        
        layout.addWidget(info_banner)
        
        # === Main Content ===
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        
        # LEFT COLUMN
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        
        # Gauges Row
        gauges_group = self._create_group("System Resources")
        gauges_layout = QHBoxLayout(gauges_group)
        gauges_layout.setSpacing(5)
        
        self.cpu_gauge = CircularGauge("CPU", 0, "#89b4fa")
        self.ram_gauge = CircularGauge("RAM", 0, "#a6e3a1")
        self.disk_gauge = CircularGauge("Disk", 0, "#f9e2af")
        self.temp_gauge = CircularGauge("Temp", 0, "#f38ba8")
        
        gauges_layout.addWidget(self.cpu_gauge)
        gauges_layout.addWidget(self.ram_gauge)
        gauges_layout.addWidget(self.disk_gauge)
        gauges_layout.addWidget(self.temp_gauge)
        
        left_col.addWidget(gauges_group)
        
        # Historical Charts
        charts_group = self._create_group("History (30 samples)")
        charts_layout = QHBoxLayout(charts_group)
        charts_layout.setSpacing(10)
        
        self.cpu_chart = MiniLineChart("CPU History", "#89b4fa")
        self.ram_chart = MiniLineChart("RAM History", "#a6e3a1")
        self.net_chart = MiniLineChart("Network (KB/s)", "#cba6f7")
        
        charts_layout.addWidget(self.cpu_chart)
        charts_layout.addWidget(self.ram_chart)
        charts_layout.addWidget(self.net_chart)
        
        left_col.addWidget(charts_group)
        
        # Details
        details_group = self._create_group("Details")
        details_layout = QGridLayout(details_group)
        details_layout.setSpacing(8)
        
        # Row 0: CPU
        details_layout.addWidget(QLabel("CPU:"), 0, 0)
        self.cpu_cores_label = self._detail_label("Cores: --")
        details_layout.addWidget(self.cpu_cores_label, 0, 1)
        self.cpu_freq_label = self._detail_label("Freq: --")
        details_layout.addWidget(self.cpu_freq_label, 0, 2)
        
        # Row 1: RAM
        details_layout.addWidget(QLabel("RAM:"), 1, 0)
        self.ram_used_label = self._detail_label("Used: --")
        details_layout.addWidget(self.ram_used_label, 1, 1)
        self.ram_total_label = self._detail_label("Total: --")
        details_layout.addWidget(self.ram_total_label, 1, 2)
        
        # Row 2: Disk
        details_layout.addWidget(QLabel("Disk:"), 2, 0)
        self.disk_used_label = self._detail_label("Used: --")
        details_layout.addWidget(self.disk_used_label, 2, 1)
        self.disk_total_label = self._detail_label("Total: --")
        details_layout.addWidget(self.disk_total_label, 2, 2)
        
        # Row 3: Network
        details_layout.addWidget(QLabel("Net:"), 3, 0)
        self.net_sent_label = self._detail_label("Sent: --")
        details_layout.addWidget(self.net_sent_label, 3, 1)
        self.net_recv_label = self._detail_label("Recv: --")
        details_layout.addWidget(self.net_recv_label, 3, 2)
        
        # Row 4: Temperature
        details_layout.addWidget(QLabel("Temp:"), 4, 0)
        self.temp_label = self._detail_label("Temp: --")
        details_layout.addWidget(self.temp_label, 4, 1, 1, 2)
        
        left_col.addWidget(details_group)
        left_col.addStretch()
        
        content_layout.addLayout(left_col, stretch=1)
        
        # RIGHT COLUMN
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        
        # Module Status (expanded)
        status_group = self._create_group("Module Status")
        status_layout = QGridLayout(status_group)
        status_layout.setSpacing(6)
        
        self.status_cards = {}
        modules_to_show = [
            ("Model", "[*]"),
            ("Tokenizer", "[*]"),
            ("Memory", "[*]"),
            ("API Server", "[*]"),
            ("Voice Input", "[*]"),
            ("Voice Output", "[*]"),
        ]
        
        for i, (name, icon) in enumerate(modules_to_show):
            card = StatusCard(name, "Not loaded", icon, "#6c7086")
            card.clicked.connect(self._on_card_clicked)
            self.status_cards[name] = card
            status_layout.addWidget(card, i // 2, i % 2)
        
        right_col.addWidget(status_group)
        
        # Activity Feed
        activity_group = self._create_group("Recent Activity")
        activity_inner = QVBoxLayout(activity_group)
        
        self.activity_container = QVBoxLayout()
        self.activity_container.setSpacing(0)
        
        self._add_activity("Dashboard initialized", "*")
        self._add_activity("System monitoring started", "*")
        
        activity_inner.addLayout(self.activity_container)
        activity_inner.addStretch()
        
        right_col.addWidget(activity_group)
        right_col.addStretch()
        
        content_layout.addLayout(right_col, stretch=1)
        
        layout.addLayout(content_layout)
    
    def _create_group(self, title: str) -> QGroupBox:
        """Create a styled group box."""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                color: #cdd6f4;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
        """)
        return group
    
    def _detail_label(self, text: str) -> QLabel:
        """Create a styled detail label."""
        label = QLabel(text)
        label.setStyleSheet("color: #bac2de; font-size: 12px;")
        return label
    
    def _start_monitoring(self):
        """Start the system monitoring timer."""
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self._update_stats)
        self.monitor_timer.start(2000)
        
        self._update_stats()
        self._update_time()
        
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)
        
        # Uptime timer
        self.uptime_timer = QTimer(self)
        self.uptime_timer.timeout.connect(self._update_uptime)
        self.uptime_timer.start(60000)
        self._update_uptime()
    
    def _update_time(self):
        """Update the current time display."""
        now = datetime.now()
        self.time_label.setText(now.strftime("Time: %H:%M:%S"))
    
    def _update_uptime(self):
        """Update system uptime."""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)
            self.uptime_label.setText(f"Up {hours}h {minutes}m")
        except (OSError, AttributeError):
            pass  # psutil not available or uptime unavailable
    
    def _update_stats(self):
        """Update system statistics."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_gauge.set_value(int(cpu_percent))
            self.cpu_history.append(cpu_percent)
            self.cpu_chart.add_value(cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.cpu_cores_label.setText(f"Cores: {cpu_count}")
            
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    self.cpu_freq_label.setText(f"Freq: {cpu_freq.current:.0f} MHz")
            except (AttributeError, OSError):
                self.cpu_freq_label.setText("Freq: N/A")  # CPU freq not available on this platform
            
            # RAM
            ram = psutil.virtual_memory()
            self.ram_gauge.set_value(int(ram.percent))
            self.ram_history.append(ram.percent)
            self.ram_chart.add_value(ram.percent)
            self.ram_used_label.setText(f"Used: {ram.used / (1024**3):.1f} GB")
            self.ram_total_label.setText(f"Total: {ram.total / (1024**3):.1f} GB")
            
            # Disk
            disk = psutil.disk_usage('/')
            self.disk_gauge.set_value(int(disk.percent))
            self.disk_used_label.setText(f"Used: {disk.used / (1024**3):.1f} GB")
            self.disk_total_label.setText(f"Total: {disk.total / (1024**3):.1f} GB")
            
            # Network
            net_io = psutil.net_io_counters()
            now = datetime.now()
            elapsed = (now - self.prev_net_time).total_seconds()
            
            if elapsed > 0:
                sent_speed = (net_io.bytes_sent - self.prev_net_io.bytes_sent) / elapsed / 1024
                recv_speed = (net_io.bytes_recv - self.prev_net_io.bytes_recv) / elapsed / 1024
                
                self.net_sent_label.setText(f"UP: {sent_speed:.1f} KB/s")
                self.net_recv_label.setText(f"DN: {recv_speed:.1f} KB/s")
                
                # Normalize for chart (cap at 1000 KB/s = 100%)
                net_percent = min(100, (sent_speed + recv_speed) / 10)
                self.net_history.append(net_percent)
                self.net_chart.add_value(net_percent)
            
            self.prev_net_io = net_io
            self.prev_net_time = now
            
            # Temperature
            temp = self._get_cpu_temp()
            if temp is not None:
                temp_percent = min(100, int(temp))
                self.temp_gauge.set_value(temp_percent)
                
                temp_color = "#a6e3a1" if temp < 60 else "#f9e2af" if temp < 75 else "#f38ba8"
                self.temp_label.setText(f"Temp: {temp:.1f} C")
                self.temp_label.setStyleSheet(f"color: {temp_color}; font-size: 12px;")
            else:
                self.temp_gauge.set_value(0)
                self.temp_label.setText("Temp: N/A")
            
            # Check for alerts
            self._check_alerts(cpu_percent, ram.percent, disk.percent, temp)
            
            # Update module status
            self._update_module_status()
            
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def _get_cpu_temp(self):
        """Get CPU temperature."""
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                with open(temp_path) as f:
                    return float(f.read().strip()) / 1000
            
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except (AttributeError, OSError, FileNotFoundError, PermissionError):
            pass  # Temperature sensors not available on this platform
        return None
    
    def _check_alerts(self, cpu: float, ram: float, disk: float, temp: float):
        """Check thresholds and show alerts."""
        alerts_to_show = []
        
        if cpu > self.alert_thresholds['cpu_high']:
            alerts_to_show.append(("critical" if cpu > 95 else "warning", f"High CPU usage: {cpu:.0f}%"))
        
        if ram > self.alert_thresholds['ram_high']:
            alerts_to_show.append(("critical" if ram > 95 else "warning", f"High RAM usage: {ram:.0f}%"))
        
        if disk > self.alert_thresholds['disk_high']:
            alerts_to_show.append(("warning", f"Disk space low: {100-disk:.0f}% free"))
        
        if temp and temp > self.alert_thresholds['temp_high']:
            alerts_to_show.append(("critical" if temp > 80 else "warning", f"High temperature: {temp:.1f} C"))
        
        # Show new alerts
        for level, message in alerts_to_show:
            if message not in self.dismissed_alerts:
                self._show_alert(message, level)
    
    def _show_alert(self, message: str, level: str = "warning"):
        """Show an alert banner."""
        # Check if already showing
        for alert in self.alerts:
            if alert.message == message:
                return
        
        alert = AlertBanner(message, level)
        alert.dismissed.connect(self._dismiss_alert)
        self.alerts.append(alert)
        self.alerts_container.addWidget(alert)
    
    def _dismiss_alert(self, message: str):
        """Dismiss an alert."""
        self.dismissed_alerts.add(message)
        for alert in self.alerts:
            if alert.message == message:
                self.alerts_container.removeWidget(alert)
                alert.deleteLater()
                self.alerts.remove(alert)
                break
    
    def _update_module_status(self):
        """Update module status cards."""
        if not self.parent_window:
            return
        
        try:
            if hasattr(self.parent_window, 'module_manager') and self.parent_window.module_manager:
                mm = self.parent_window.module_manager
                loaded_modules = mm.list_loaded() if hasattr(mm, 'list_loaded') else []
                
                status_map = {
                    "Model": ('model', "Loaded", "Not loaded"),
                    "Tokenizer": ('tokenizer', "Loaded", "Not loaded"),
                    "Memory": ('memory', "Active", "Inactive"),
                    "API Server": ('api_server', "Running", "Stopped"),
                    "Voice Input": ('voice_input', "Active", "Inactive"),
                    "Voice Output": ('voice_output', "Active", "Inactive"),
                }
                
                for card_name, (module_id, loaded_status, unloaded_status) in status_map.items():
                    if card_name in self.status_cards:
                        if module_id in loaded_modules:
                            self.status_cards[card_name].set_status(loaded_status, "#a6e3a1")
                        else:
                            self.status_cards[card_name].set_status(unloaded_status, "#6c7086")
            else:
                if hasattr(self.parent_window, 'engine') and self.parent_window.engine:
                    self.status_cards["Model"].set_status("Loaded", "#a6e3a1")
                
        except Exception:
            pass  # Silently handle errors
    
    def _add_activity(self, message: str, icon: str = ">"):
        """Add an activity item to the feed."""
        timestamp = datetime.now().strftime("%H:%M")
        item = ActivityItem(message, timestamp, icon)
        
        if len(self.activity_items) >= 8:
            old_item = self.activity_items.pop(0)
            self.activity_container.removeWidget(old_item)
            old_item.deleteLater()
        
        self.activity_items.append(item)
        self.activity_container.addWidget(item)
    
    def _refresh_all(self):
        """Manually refresh all stats."""
        self._update_stats()
        self._add_activity("Manual refresh", ">")
    
    def _on_card_clicked(self, title: str):
        """Handle status card click."""
        self._add_activity(f"Clicked: {title}", ">")
        
        if self.parent_window and hasattr(self.parent_window, 'switch_to_tab'):
            tab_map = {
                "Model": "modules",
                "Tokenizer": "modules",
                "Memory": "modules",
                "API Server": "network",
                "Voice Input": "modules",
                "Voice Output": "modules",
            }
            if title in tab_map:
                self.parent_window.switch_to_tab(tab_map[title])
    
    def log_activity(self, message: str, icon: str = ">"):
        """Public method to log activity from other parts of the app."""
        self._add_activity(message, icon)


def create_dashboard_tab(parent=None):
    """Factory function to create dashboard tab."""
    return DashboardTab(parent)
