"""
Analytics Tab - Usage statistics, performance metrics, and insights.

Features:
  - Tool usage statistics
  - Chat session metrics
  - Model performance tracking
  - Training history
  - Visual charts and graphs
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .shared_components import NoScrollComboBox

logger = logging.getLogger(__name__)

# Config paths
ANALYTICS_DIR = Path.home() / ".enigma_engine" / "analytics"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

TOOL_USAGE_FILE = ANALYTICS_DIR / "tool_usage.json"
SESSION_STATS_FILE = ANALYTICS_DIR / "session_stats.json"
RESPONSE_TIMES_FILE = ANALYTICS_DIR / "response_times.json"


# Global analytics recorder for use throughout the codebase
class AnalyticsRecorder:
    """Global analytics recording utility."""
    
    _instance: Optional['AnalyticsRecorder'] = None
    
    def __init__(self):
        self._ensure_files()
    
    @classmethod
    def get_instance(cls) -> 'AnalyticsRecorder':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensure_files(self):
        """Ensure analytics files exist."""
        ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
        for file in [TOOL_USAGE_FILE, SESSION_STATS_FILE, RESPONSE_TIMES_FILE]:
            if not file.exists():
                file.write_text('{}')
    
    def record_tool_usage(self, tool_name: str, success: bool = True, 
                          duration_ms: float = 0, category: str = ""):
        """Record a tool usage event."""
        try:
            entries = []
            if TOOL_USAGE_FILE.exists():
                try:
                    data = json.loads(TOOL_USAGE_FILE.read_text())
                    entries = data.get("entries", [])
                except json.JSONDecodeError:
                    pass
            
            entries.append({
                "tool": tool_name,
                "success": success,
                "duration_ms": duration_ms,
                "category": category,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Keep last 10000 entries
            entries = entries[-10000:]
            TOOL_USAGE_FILE.write_text(json.dumps({"entries": entries}, indent=2))
        except Exception as e:
            logger.debug(f"Could not record tool usage: {e}")
    
    def record_response_time(self, response_time_ms: float, model: str = "", 
                             tokens: int = 0):
        """Record a response time measurement."""
        try:
            entries = []
            if RESPONSE_TIMES_FILE.exists():
                try:
                    data = json.loads(RESPONSE_TIMES_FILE.read_text())
                    entries = data.get("entries", [])
                except json.JSONDecodeError:
                    pass
            
            entries.append({
                "response_time_ms": response_time_ms,
                "model": model,
                "tokens": tokens,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Keep last 5000 entries
            entries = entries[-5000:]
            RESPONSE_TIMES_FILE.write_text(json.dumps({"entries": entries}, indent=2))
        except Exception as e:
            logger.debug(f"Could not record response time: {e}")
    
    def record_session_message(self, is_user: bool = True):
        """Record a message in session stats."""
        try:
            stats = {"total_messages": 0, "session_count": 0, 
                     "hourly_activity": {}, "models_trained": 0}
            
            if SESSION_STATS_FILE.exists():
                try:
                    stats = json.loads(SESSION_STATS_FILE.read_text())
                except json.JSONDecodeError:
                    pass
            
            stats["total_messages"] = stats.get("total_messages", 0) + 1
            
            # Track hourly activity
            hour = f"{datetime.now().hour}:00"
            hourly = stats.get("hourly_activity", {})
            hourly[hour] = hourly.get(hour, 0) + 1
            stats["hourly_activity"] = hourly
            
            SESSION_STATS_FILE.write_text(json.dumps(stats, indent=2))
        except Exception as e:
            logger.debug(f"Could not record session message: {e}")
    
    def record_training(self, model: str, epochs: int, final_loss: float, 
                        duration_min: float):
        """Record a training session."""
        try:
            stats = {}
            if SESSION_STATS_FILE.exists():
                try:
                    stats = json.loads(SESSION_STATS_FILE.read_text())
                except json.JSONDecodeError:
                    pass
            
            stats["models_trained"] = stats.get("models_trained", 0) + 1
            
            # Track training history
            history = stats.get("training_history", [])
            history.append({
                "model": model,
                "epochs": epochs,
                "final_loss": final_loss,
                "duration_min": duration_min,
                "timestamp": datetime.now().isoformat(),
            })
            stats["training_history"] = history[-100:]  # Keep last 100
            
            SESSION_STATS_FILE.write_text(json.dumps(stats, indent=2))
        except Exception as e:
            logger.debug(f"Could not record training: {e}")
    
    def get_avg_response_time(self) -> float:
        """Get average response time in ms."""
        try:
            if RESPONSE_TIMES_FILE.exists():
                data = json.loads(RESPONSE_TIMES_FILE.read_text())
                entries = data.get("entries", [])
                if entries:
                    times = [e.get("response_time_ms", 0) for e in entries[-100:]]
                    return sum(times) / len(times) if times else 0
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return 0


def get_analytics_recorder() -> AnalyticsRecorder:
    """Get the global analytics recorder."""
    return AnalyticsRecorder.get_instance()


# Convenience functions for easy importing
def record_tool_usage(tool_name: str, success: bool = True, 
                      duration_ms: float = 0, category: str = ""):
    """Record tool usage - convenience function."""
    get_analytics_recorder().record_tool_usage(tool_name, success, duration_ms, category)


def record_response_time(response_time_ms: float, model: str = "", tokens: int = 0):
    """Record response time - convenience function."""
    get_analytics_recorder().record_response_time(response_time_ms, model, tokens)


def record_session_message(is_user: bool = True):
    """Record session message - convenience function."""
    get_analytics_recorder().record_session_message(is_user)


class StatCard(QFrame):
    """A card displaying a single statistic."""
    
    def __init__(self, title: str, value: str, subtitle: str = "", color: str = "#2196f3"):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 8px;
                border-left: 4px solid {color};
            }}
        """)
        self.setMinimumWidth(150)
        self.setMinimumHeight(90)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #bac2de; font-size: 12px;")
        layout.addWidget(title_label)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")
        layout.addWidget(self.value_label)
        
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setStyleSheet("color: #bac2de; font-size: 12px;")
            layout.addWidget(self.subtitle_label)
        else:
            self.subtitle_label = None
    
    def set_value(self, value: str, subtitle: str = None):
        """Update the displayed value."""
        self.value_label.setText(value)
        if subtitle and self.subtitle_label:
            self.subtitle_label.setText(subtitle)


class SimpleBarChart(QWidget):
    """Simple bar chart widget."""
    
    def __init__(self, data: dict = None, title: str = ""):
        super().__init__()
        self.data = data or {}
        self.title = title
        self.setMinimumHeight(200)
    
    def set_data(self, data: dict):
        """Update chart data."""
        self.data = data
        self.update()
    
    def paintEvent(self, event):
        """Draw the bar chart."""
        if not self.data:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        margin = 40
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        
        # Title
        if self.title:
            painter.setPen(QPen(QColor("#333")))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(margin, 20, self.title)
        
        if not self.data:
            return
        
        max_value = max(self.data.values()) if self.data.values() else 1
        bar_width = chart_width / len(self.data) - 10
        
        colors = ["#2196f3", "#4caf50", "#ff9800", "#f44336", "#9c27b0", "#00bcd4"]
        
        x = margin
        for i, (label, value) in enumerate(self.data.items()):
            bar_height = (value / max_value) * (chart_height - 30)
            
            # Bar
            color = QColor(colors[i % len(colors)])
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(
                int(x + 5),
                int(margin + chart_height - bar_height - 20),
                int(bar_width),
                int(bar_height)
            )
            
            # Value
            painter.setPen(QPen(QColor("#333")))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(
                int(x + 5),
                int(margin + chart_height - bar_height - 25),
                int(bar_width),
                15,
                Qt.AlignCenter,
                str(value)
            )
            
            # Label (rotated)
            painter.save()
            painter.translate(x + bar_width / 2 + 5, height - 5)
            painter.rotate(-45)
            painter.drawText(0, 0, label[:12])
            painter.restore()
            
            x += bar_width + 10


class AnalyticsTab(QWidget):
    """Main Analytics tab with usage statistics and insights."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._setup_ui()
        self._load_analytics()
        self._start_refresh_timer()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        header = QLabel("Analytics Dashboard")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Date range
        header_layout.addWidget(QLabel("Period:"))
        self.period_combo = NoScrollComboBox()
        self.period_combo.addItems(["Today", "This Week", "This Month", "All Time"])
        self.period_combo.setToolTip("Select time period for analytics")
        self.period_combo.currentIndexChanged.connect(self._load_analytics)
        header_layout.addWidget(self.period_combo)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload analytics data")
        refresh_btn.clicked.connect(self._load_analytics)
        header_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export")
        export_btn.setToolTip("Export analytics to file")
        export_btn.clicked.connect(self._export_analytics)
        header_layout.addWidget(export_btn)
        
        layout.addLayout(header_layout)
        
        # Scroll area for dashboard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Overview cards
        cards_group = QGroupBox("Overview")
        cards_layout = QHBoxLayout(cards_group)
        
        self.total_messages_card = StatCard("Total Messages", "0", "lifetime", "#2196f3")
        cards_layout.addWidget(self.total_messages_card)
        
        self.active_sessions_card = StatCard("Chat Sessions", "0", "this period", "#4caf50")
        cards_layout.addWidget(self.active_sessions_card)
        
        self.tools_used_card = StatCard("Tool Calls", "0", "this period", "#ff9800")
        cards_layout.addWidget(self.tools_used_card)
        
        self.avg_response_card = StatCard("Avg Response", "0ms", "response time", "#9c27b0")
        cards_layout.addWidget(self.avg_response_card)
        
        self.models_trained_card = StatCard("Models Trained", "0", "total", "#f44336")
        cards_layout.addWidget(self.models_trained_card)
        
        scroll_layout.addWidget(cards_group)
        
        # Charts section
        charts_layout = QHBoxLayout()
        
        # Tool usage chart
        tool_group = QGroupBox("Tool Usage")
        tool_layout = QVBoxLayout(tool_group)
        self.tool_chart = SimpleBarChart(title="Most Used Tools")
        tool_layout.addWidget(self.tool_chart)
        charts_layout.addWidget(tool_group)
        
        # Session activity chart
        activity_group = QGroupBox("Activity by Hour")
        activity_layout = QVBoxLayout(activity_group)
        self.activity_chart = SimpleBarChart(title="Messages by Hour")
        activity_layout.addWidget(self.activity_chart)
        charts_layout.addWidget(activity_group)
        
        scroll_layout.addLayout(charts_layout)
        
        # Detailed tables in tabs
        detail_tabs = QTabWidget()
        
        # Tool usage table
        tool_table_widget = QWidget()
        tool_table_layout = QVBoxLayout(tool_table_widget)
        
        self.tool_table = QTableWidget()
        self.tool_table.setColumnCount(5)
        self.tool_table.setHorizontalHeaderLabels(["Tool Name", "Category", "Calls", "Success Rate", "Last Used"])
        self.tool_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tool_table_layout.addWidget(self.tool_table)
        
        detail_tabs.addTab(tool_table_widget, "Tool Details")
        
        # Session history table
        session_widget = QWidget()
        session_layout = QVBoxLayout(session_widget)
        
        self.session_table = QTableWidget()
        self.session_table.setColumnCount(5)
        self.session_table.setHorizontalHeaderLabels(["Date", "Duration", "Messages", "Tools Used", "Model"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        session_layout.addWidget(self.session_table)
        
        detail_tabs.addTab(session_widget, "Sessions")
        
        # Model performance table
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(5)
        self.model_table.setHorizontalHeaderLabels(["Model", "Size", "Inference Time", "Memory Usage", "Last Used"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        model_layout.addWidget(self.model_table)
        
        detail_tabs.addTab(model_widget, "Models")
        
        # Training history
        training_widget = QWidget()
        training_layout = QVBoxLayout(training_widget)
        
        self.training_table = QTableWidget()
        self.training_table.setColumnCount(5)
        self.training_table.setHorizontalHeaderLabels(["Date", "Model", "Epochs", "Final Loss", "Duration"])
        self.training_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        training_layout.addWidget(self.training_table)
        
        detail_tabs.addTab(training_widget, "Training")
        
        scroll_layout.addWidget(detail_tabs)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def _load_analytics(self):
        """Load and display analytics data."""
        period = self.period_combo.currentText()
        
        # Calculate date range
        today = datetime.now()
        if period == "Today":
            start_date = today.replace(hour=0, minute=0, second=0)
        elif period == "This Week":
            start_date = today - timedelta(days=today.weekday())
        elif period == "This Month":
            start_date = today.replace(day=1)
        else:
            start_date = datetime(2000, 1, 1)
        
        # Load tool usage
        tool_usage = self._load_tool_usage(start_date)
        session_stats = self._load_session_stats(start_date)
        
        # Update cards
        self.total_messages_card.set_value(str(session_stats.get("total_messages", 0)))
        self.active_sessions_card.set_value(str(session_stats.get("session_count", 0)))
        self.tools_used_card.set_value(str(sum(tool_usage.values())))
        self.avg_response_card.set_value(f"{session_stats.get('avg_response_ms', 0):.0f}ms")
        self.models_trained_card.set_value(str(session_stats.get("models_trained", 0)))
        
        # Update tool chart
        top_tools = dict(sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:8])
        self.tool_chart.set_data(top_tools)
        
        # Update activity chart
        activity_data = session_stats.get("hourly_activity", {})
        if not activity_data:
            activity_data = {f"{h}:00": 0 for h in range(0, 24, 3)}
        self.activity_chart.set_data(activity_data)
        
        # Update tool table
        self._update_tool_table(tool_usage)
        
        # Update training table
        self._update_training_table(session_stats.get("training_history", []))
    
    def _load_tool_usage(self, start_date: datetime) -> dict:
        """Load tool usage statistics."""
        usage = defaultdict(int)
        success_count = defaultdict(int)
        total_count = defaultdict(int)
        last_used = {}
        
        if TOOL_USAGE_FILE.exists():
            try:
                with open(TOOL_USAGE_FILE) as f:
                    data = json.load(f)
                    for entry in data.get("entries", []):
                        entry_date = datetime.fromisoformat(entry.get("timestamp", "2000-01-01"))
                        tool = entry.get("tool", "unknown")
                        if entry_date >= start_date:
                            usage[tool] += 1
                            total_count[tool] += 1
                            if entry.get("success", True):
                                success_count[tool] += 1
                            last_used[tool] = entry.get("timestamp", "")
            except Exception as e:
                logger.debug(f"Could not load tool usage file: {e}")
        
        # Store extra data for table display
        self._tool_success_rates = {}
        self._tool_last_used = {}
        for tool in usage:
            if total_count[tool] > 0:
                self._tool_success_rates[tool] = int(success_count[tool] / total_count[tool] * 100)
            else:
                self._tool_success_rates[tool] = 100
            self._tool_last_used[tool] = last_used.get(tool, "")
        
        return dict(usage)
    
    def _load_session_stats(self, start_date: datetime) -> dict:
        """Load session statistics."""
        stats = {
            "total_messages": 0,
            "session_count": 0,
            "avg_response_ms": 0,
            "models_trained": 0,
            "hourly_activity": {},
            "training_history": [],
        }
        
        if SESSION_STATS_FILE.exists():
            try:
                with open(SESSION_STATS_FILE) as f:
                    loaded = json.load(f)
                    stats.update(loaded)
            except Exception as e:
                logger.debug(f"Could not load session stats: {e}")
        
        # Get real average response time
        stats["avg_response_ms"] = get_analytics_recorder().get_avg_response_time()
        
        # Default sample data only if truly empty
        if stats["total_messages"] == 0 and not stats.get("hourly_activity"):
            stats = {
                "total_messages": 0,
                "session_count": 0,
                "avg_response_ms": 0,
                "models_trained": 0,
                "hourly_activity": {
                    "0:00": 0, "3:00": 0, "6:00": 0, "9:00": 0,
                    "12:00": 0, "15:00": 0, "18:00": 0, "21:00": 0
                },
                "training_history": [],
            }
        
        return stats
    
    def _update_tool_table(self, tool_usage: dict):
        """Update the tool details table."""
        self.tool_table.setRowCount(0)
        
        categories = {
            "chat": "Core", "web_search": "Web", "file_read": "Files",
            "image_gen": "Generation", "code_gen": "Generation",
            "screen_capture": "Vision", "analyze_image": "Vision",
        }
        
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
            row = self.tool_table.rowCount()
            self.tool_table.insertRow(row)
            
            # Get success rate
            success_rate = self._tool_success_rates.get(tool, 100) if hasattr(self, '_tool_success_rates') else 100
            
            # Get last used time
            last_used_str = "Unknown"
            if hasattr(self, '_tool_last_used') and tool in self._tool_last_used:
                ts = self._tool_last_used[tool]
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        # Show relative time
                        delta = datetime.now() - dt
                        if delta.days > 0:
                            last_used_str = f"{delta.days}d ago"
                        elif delta.seconds > 3600:
                            last_used_str = f"{delta.seconds // 3600}h ago"
                        elif delta.seconds > 60:
                            last_used_str = f"{delta.seconds // 60}m ago"
                        else:
                            last_used_str = "Just now"
                    except ValueError:
                        last_used_str = ts[:16]
            
            self.tool_table.setItem(row, 0, QTableWidgetItem(tool))
            self.tool_table.setItem(row, 1, QTableWidgetItem(categories.get(tool, "Other")))
            self.tool_table.setItem(row, 2, QTableWidgetItem(str(count)))
            self.tool_table.setItem(row, 3, QTableWidgetItem(f"{success_rate}%"))
            self.tool_table.setItem(row, 4, QTableWidgetItem(last_used_str))
    
    def _update_training_table(self, training_history: list):
        """Update the training history table."""
        self.training_table.setRowCount(0)
        
        for entry in reversed(training_history):  # Most recent first
            row = self.training_table.rowCount()
            self.training_table.insertRow(row)
            
            # Parse timestamp
            ts = entry.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    date_str = ts[:16]
            else:
                date_str = "Unknown"
            
            self.training_table.setItem(row, 0, QTableWidgetItem(date_str))
            self.training_table.setItem(row, 1, QTableWidgetItem(entry.get("model", "unknown")))
            self.training_table.setItem(row, 2, QTableWidgetItem(str(entry.get("epochs", 0))))
            self.training_table.setItem(row, 3, QTableWidgetItem(f"{entry.get('final_loss', 0):.4f}"))
            self.training_table.setItem(row, 4, QTableWidgetItem(f"{entry.get('duration_min', 0):.1f}m"))
    
    def _export_analytics(self):
        """Export analytics to file."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Analytics", 
            str(Path.home() / "forge_analytics.json"),
            "JSON Files (*.json)"
        )
        
        if filename:
            data = {
                "exported": datetime.now().isoformat(),
                "tool_usage": self._load_tool_usage(datetime(2000, 1, 1)),
                "session_stats": self._load_session_stats(datetime(2000, 1, 1)),
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            QMessageBox.information(self, "Exported", f"Analytics exported to {filename}")
    
    def _start_refresh_timer(self):
        """Start auto-refresh timer."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._load_analytics)
        self.refresh_timer.start(60000)  # Refresh every minute
    
    def record_tool_usage(self, tool_name: str, success: bool = True):
        """Record a tool usage event (instance method for backwards compatibility)."""
        get_analytics_recorder().record_tool_usage(tool_name, success)


def create_analytics_tab(parent=None):
    """Factory function to create analytics tab."""
    return AnalyticsTab(parent)
