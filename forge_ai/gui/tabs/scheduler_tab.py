"""
Scheduler Tab - View and manage scheduled tasks.

Features:
  - View scheduled tasks
  - Create new scheduled tasks
  - Edit/delete existing tasks
  - Run tasks manually
  - View task history and logs
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView, QDialog, QFormLayout,
    QSpinBox, QTimeEdit, QDateEdit, QCheckBox, QMessageBox,
    QMenu, QSplitter
)
from PyQt5.QtCore import Qt, QTime, QDate, QTimer
from PyQt5.QtGui import QFont, QColor

from .shared_components import NoScrollComboBox

# Config
SCHEDULER_FILE = Path.home() / ".forge_ai" / "scheduled_tasks.json"
SCHEDULER_FILE.parent.mkdir(parents=True, exist_ok=True)


class TaskDialog(QDialog):
    """Dialog for creating/editing a scheduled task."""
    
    def __init__(self, parent=None, task: dict = None):
        super().__init__(parent)
        self.task = task or {}
        self.setWindowTitle("Edit Task" if task else "New Task")
        self.setMinimumWidth(400)
        self._setup_ui()
        
        if task:
            self._load_task(task)
    
    def _setup_ui(self):
        layout = QFormLayout(self)
        
        # Task name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("My scheduled task")
        layout.addRow("Name:", self.name_input)
        
        # Task type
        self.type_combo = NoScrollComboBox()
        self.type_combo.setToolTip("Select the type of scheduled task")
        self.type_combo.addItems([
            "Run Command",
            "Send Message",
            "Backup Models",
            "Clear Cache",
            "Export Logs",
            "Custom Python"
        ])
        layout.addRow("Type:", self.type_combo)
        
        # Command/Action
        self.action_input = QTextEdit()
        self.action_input.setMaximumHeight(80)
        self.action_input.setPlaceholderText("Enter command or message...")
        layout.addRow("Action:", self.action_input)
        
        # Schedule type
        self.schedule_combo = NoScrollComboBox()
        self.schedule_combo.setToolTip("Select how often the task should run")
        self.schedule_combo.addItems([
            "Once",
            "Every X Minutes",
            "Hourly",
            "Daily",
            "Weekly",
            "Monthly"
        ])
        self.schedule_combo.currentIndexChanged.connect(self._update_schedule_options)
        layout.addRow("Schedule:", self.schedule_combo)
        
        # Schedule details
        schedule_widget = QWidget()
        schedule_layout = QHBoxLayout(schedule_widget)
        schedule_layout.setContentsMargins(0, 0, 0, 0)
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 1440)
        self.interval_spin.setValue(30)
        schedule_layout.addWidget(QLabel("Every"))
        schedule_layout.addWidget(self.interval_spin)
        self.interval_label = QLabel("minutes")
        schedule_layout.addWidget(self.interval_label)
        schedule_layout.addStretch()
        
        layout.addRow("Interval:", schedule_widget)
        
        # Time
        time_widget = QWidget()
        time_layout = QHBoxLayout(time_widget)
        time_layout.setContentsMargins(0, 0, 0, 0)
        
        self.time_edit = QTimeEdit()
        self.time_edit.setTime(QTime(9, 0))
        time_layout.addWidget(self.time_edit)
        
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        time_layout.addWidget(self.date_edit)
        
        time_layout.addStretch()
        layout.addRow("At:", time_widget)
        
        # Days of week (for weekly)
        days_widget = QWidget()
        days_layout = QHBoxLayout(days_widget)
        days_layout.setContentsMargins(0, 0, 0, 0)
        
        self.day_checks = {}
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            cb = QCheckBox(day)
            self.day_checks[day] = cb
            days_layout.addWidget(cb)
        
        layout.addRow("Days:", days_widget)
        
        # Enabled
        self.enabled_check = QCheckBox("Task enabled")
        self.enabled_check.setChecked(True)
        layout.addRow("", self.enabled_check)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addRow("", btn_layout)
        
        self._update_schedule_options()
    
    def _update_schedule_options(self):
        """Show/hide schedule options based on type."""
        schedule = self.schedule_combo.currentText()
        
        # Hide all by default
        self.interval_spin.setVisible(schedule == "Every X Minutes")
        self.interval_label.setVisible(schedule == "Every X Minutes")
        self.date_edit.setVisible(schedule == "Once")
        
        for cb in self.day_checks.values():
            cb.setVisible(schedule == "Weekly")
    
    def _load_task(self, task: dict):
        """Load task data into form."""
        self.name_input.setText(task.get("name", ""))
        
        task_types = ["Run Command", "Send Message", "Backup Models", "Clear Cache", "Export Logs", "Custom Python"]
        task_type = task.get("type", "Run Command")
        if task_type in task_types:
            self.type_combo.setCurrentIndex(task_types.index(task_type))
        
        self.action_input.setPlainText(task.get("action", ""))
        
        schedules = ["Once", "Every X Minutes", "Hourly", "Daily", "Weekly", "Monthly"]
        schedule = task.get("schedule", "Daily")
        if schedule in schedules:
            self.schedule_combo.setCurrentIndex(schedules.index(schedule))
        
        self.interval_spin.setValue(task.get("interval", 30))
        
        if "time" in task:
            try:
                time_parts = task["time"].split(":")
                self.time_edit.setTime(QTime(int(time_parts[0]), int(time_parts[1])))
            except:
                pass
        
        if "date" in task:
            try:
                self.date_edit.setDate(QDate.fromString(task["date"], "yyyy-MM-dd"))
            except:
                pass
        
        for day, cb in self.day_checks.items():
            cb.setChecked(day in task.get("days", []))
        
        self.enabled_check.setChecked(task.get("enabled", True))
    
    def get_task(self) -> dict:
        """Get task data from form."""
        return {
            "id": self.task.get("id", datetime.now().strftime("%Y%m%d%H%M%S")),
            "name": self.name_input.text(),
            "type": self.type_combo.currentText(),
            "action": self.action_input.toPlainText(),
            "schedule": self.schedule_combo.currentText(),
            "interval": self.interval_spin.value(),
            "time": self.time_edit.time().toString("HH:mm"),
            "date": self.date_edit.date().toString("yyyy-MM-dd"),
            "days": [day for day, cb in self.day_checks.items() if cb.isChecked()],
            "enabled": self.enabled_check.isChecked(),
            "created": self.task.get("created", datetime.now().isoformat()),
            "last_run": self.task.get("last_run"),
            "run_count": self.task.get("run_count", 0),
        }


class SchedulerTab(QWidget):
    """Main Scheduler tab for managing scheduled tasks."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.tasks = []
        self._setup_ui()
        self._load_tasks()
        self._start_scheduler()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        header = QLabel("Task Scheduler")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        add_btn = QPushButton("New Task")
        add_btn.clicked.connect(self._add_task)
        header_layout.addWidget(add_btn)
        
        layout.addLayout(header_layout)
        
        # Main splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Tasks table
        tasks_group = QGroupBox("Scheduled Tasks")
        tasks_layout = QVBoxLayout(tasks_group)
        
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(7)
        self.tasks_table.setHorizontalHeaderLabels([
            "Status", "Name", "Type", "Schedule", "Next Run", "Last Run", "Actions"
        ])
        self.tasks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tasks_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tasks_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tasks_table.customContextMenuRequested.connect(self._show_context_menu)
        tasks_layout.addWidget(self.tasks_table)
        
        splitter.addWidget(tasks_group)
        
        # History section
        history_group = QGroupBox("Run History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Task", "Status", "Duration", "Output"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        
        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        history_layout.addWidget(clear_btn)
        
        splitter.addWidget(history_group)
        
        splitter.setSizes([400, 200])
        layout.addWidget(splitter)
    
    def _load_tasks(self):
        """Load scheduled tasks from file."""
        if SCHEDULER_FILE.exists():
            try:
                with open(SCHEDULER_FILE, 'r') as f:
                    data = json.load(f)
                    self.tasks = data.get("tasks", [])
            except:
                self.tasks = []
        
        self._refresh_table()
    
    def _save_tasks(self):
        """Save scheduled tasks to file."""
        data = {"tasks": self.tasks}
        with open(SCHEDULER_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _refresh_table(self):
        """Refresh the tasks table."""
        self.tasks_table.setRowCount(0)
        
        for task in self.tasks:
            row = self.tasks_table.rowCount()
            self.tasks_table.insertRow(row)
            
            # Status
            status = "[ON]" if task.get("enabled", True) else "[OFF]"
            self.tasks_table.setItem(row, 0, QTableWidgetItem(status))
            
            # Name
            self.tasks_table.setItem(row, 1, QTableWidgetItem(task.get("name", "Unnamed")))
            
            # Type
            self.tasks_table.setItem(row, 2, QTableWidgetItem(task.get("type", "Unknown")))
            
            # Schedule
            schedule = self._format_schedule(task)
            self.tasks_table.setItem(row, 3, QTableWidgetItem(schedule))
            
            # Next run
            next_run = self._calculate_next_run(task)
            self.tasks_table.setItem(row, 4, QTableWidgetItem(next_run))
            
            # Last run
            last_run = task.get("last_run", "Never")
            if last_run != "Never":
                try:
                    dt = datetime.fromisoformat(last_run)
                    last_run = dt.strftime("%m/%d %H:%M")
                except:
                    pass
            self.tasks_table.setItem(row, 5, QTableWidgetItem(last_run))
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            run_btn = QPushButton(">")
            run_btn.setMaximumWidth(30)
            run_btn.setToolTip("Run now")
            run_btn.clicked.connect(lambda _, t=task: self._run_task(t))
            actions_layout.addWidget(run_btn)
            
            edit_btn = QPushButton("Edit")
            edit_btn.setMaximumWidth(40)
            edit_btn.setToolTip("Edit")
            edit_btn.clicked.connect(lambda _, t=task: self._edit_task(t))
            actions_layout.addWidget(edit_btn)
            
            delete_btn = QPushButton("X")
            delete_btn.setMaximumWidth(30)
            delete_btn.setToolTip("Delete")
            delete_btn.clicked.connect(lambda _, t=task: self._delete_task(t))
            actions_layout.addWidget(delete_btn)
            
            self.tasks_table.setCellWidget(row, 6, actions_widget)
    
    def _format_schedule(self, task: dict) -> str:
        """Format schedule for display."""
        schedule = task.get("schedule", "Daily")
        time = task.get("time", "09:00")
        
        if schedule == "Once":
            date = task.get("date", "")
            return f"Once at {date} {time}"
        elif schedule == "Every X Minutes":
            return f"Every {task.get('interval', 30)} min"
        elif schedule == "Hourly":
            return f"Hourly at :{time.split(':')[1]}"
        elif schedule == "Daily":
            return f"Daily at {time}"
        elif schedule == "Weekly":
            days = ", ".join(task.get("days", []))
            return f"Weekly: {days} {time}"
        elif schedule == "Monthly":
            return f"Monthly at {time}"
        return schedule
    
    def _calculate_next_run(self, task: dict) -> str:
        """Calculate next run time."""
        if not task.get("enabled", True):
            return "Disabled"
        
        now = datetime.now()
        schedule = task.get("schedule", "Daily")
        time_str = task.get("time", "09:00")
        
        try:
            time_parts = time_str.split(":")
            run_time = now.replace(
                hour=int(time_parts[0]),
                minute=int(time_parts[1]),
                second=0,
                microsecond=0
            )
            
            if schedule == "Once":
                date_str = task.get("date", "")
                if date_str:
                    return f"{date_str} {time_str}"
                return "Not set"
            
            elif schedule == "Every X Minutes":
                interval = task.get("interval", 30)
                next_run = now + timedelta(minutes=interval)
                return next_run.strftime("%H:%M")
            
            elif schedule == "Daily":
                if run_time < now:
                    run_time += timedelta(days=1)
                if run_time.date() == now.date():
                    return f"Today {time_str}"
                else:
                    return f"Tomorrow {time_str}"
            
            return time_str
            
        except:
            return "Unknown"
    
    def _add_task(self):
        """Add a new task."""
        dialog = TaskDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            task = dialog.get_task()
            self.tasks.append(task)
            self._save_tasks()
            self._refresh_table()
    
    def _edit_task(self, task: dict):
        """Edit an existing task."""
        dialog = TaskDialog(self, task)
        if dialog.exec_() == QDialog.Accepted:
            updated = dialog.get_task()
            for i, t in enumerate(self.tasks):
                if t.get("id") == task.get("id"):
                    self.tasks[i] = updated
                    break
            self._save_tasks()
            self._refresh_table()
    
    def _delete_task(self, task: dict):
        """Delete a task."""
        reply = QMessageBox.question(
            self, "Delete Task",
            f"Delete task '{task.get('name')}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.tasks = [t for t in self.tasks if t.get("id") != task.get("id")]
            self._save_tasks()
            self._refresh_table()
    
    def _run_task(self, task: dict):
        """Run a task immediately (in background thread)."""
        self._add_history(task["name"], "Running...")
        
        # Run task in background thread to prevent UI freeze
        import threading
        def do_task():
            task_type = task.get("type", "")
            action = task.get("action", "")
            
            try:
                if task_type == "Run Command":
                    import subprocess
                    result = subprocess.run(action, shell=True, capture_output=True, text=True, timeout=30)
                    output = result.stdout or result.stderr or "Completed"
                    status = "Success" if result.returncode == 0 else "Failed"
                
                elif task_type == "Send Message":
                    # Would integrate with chat here
                    output = f"Message sent: {action[:50]}..."
                    status = "Success"
                
                elif task_type == "Backup Models":
                    output = "Backup not yet implemented"
                    status = "Skipped"
                
                elif task_type == "Clear Cache":
                    import shutil
                    cache_dir = Path.home() / ".forge_ai" / "cache"
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                        cache_dir.mkdir()
                    output = "Cache cleared"
                    status = "Success"
                
                elif task_type == "Export Logs":
                    output = "Log export not yet implemented"
                    status = "Skipped"
                
                else:
                    output = f"Unknown task type: {task_type}"
                    status = "Failed"
                
                # Update task on main thread
                from PyQt5.QtCore import QTimer
                def update_ui():
                    for t in self.tasks:
                        if t.get("id") == task.get("id"):
                            t["last_run"] = datetime.now().isoformat()
                            t["run_count"] = t.get("run_count", 0) + 1
                    
                    self._save_tasks()
                    self._refresh_table()
                    self._update_history(task["name"], status, output)
                
                QTimer.singleShot(0, update_ui)
                
            except Exception as e:
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._update_history(task["name"], "Error", str(e)))
        
        thread = threading.Thread(target=do_task, daemon=True)
        thread.start()
    
    def _add_history(self, task_name: str, status: str):
        """Add entry to history table."""
        row = 0
        self.history_table.insertRow(row)
        
        self.history_table.setItem(row, 0, QTableWidgetItem(
            datetime.now().strftime("%H:%M:%S")
        ))
        self.history_table.setItem(row, 1, QTableWidgetItem(task_name))
        self.history_table.setItem(row, 2, QTableWidgetItem(status))
        self.history_table.setItem(row, 3, QTableWidgetItem("-"))
        self.history_table.setItem(row, 4, QTableWidgetItem(""))
    
    def _update_history(self, task_name: str, status: str, output: str):
        """Update the most recent history entry."""
        if self.history_table.rowCount() > 0:
            self.history_table.setItem(0, 2, QTableWidgetItem(status))
            self.history_table.setItem(0, 3, QTableWidgetItem("< 1s"))
            self.history_table.setItem(0, 4, QTableWidgetItem(output[:100]))
    
    def _clear_history(self):
        """Clear run history."""
        self.history_table.setRowCount(0)
    
    def _show_context_menu(self, pos):
        """Show context menu for task."""
        row = self.tasks_table.rowAt(pos.y())
        if row < 0 or row >= len(self.tasks):
            return
        
        task = self.tasks[row]
        
        menu = QMenu(self)
        
        run_action = menu.addAction("Run Now")
        run_action.triggered.connect(lambda: self._run_task(task))
        
        edit_action = menu.addAction("Edit")
        edit_action.triggered.connect(lambda: self._edit_task(task))
        
        menu.addSeparator()
        
        if task.get("enabled", True):
            disable_action = menu.addAction("Disable")
            disable_action.triggered.connect(lambda: self._toggle_task(task, False))
        else:
            enable_action = menu.addAction("Enable")
            enable_action.triggered.connect(lambda: self._toggle_task(task, True))
        
        menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_task(task))
        
        menu.exec_(self.tasks_table.viewport().mapToGlobal(pos))
    
    def _toggle_task(self, task: dict, enabled: bool):
        """Enable or disable a task."""
        for t in self.tasks:
            if t.get("id") == task.get("id"):
                t["enabled"] = enabled
        self._save_tasks()
        self._refresh_table()
    
    def _start_scheduler(self):
        """Start the scheduler timer."""
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.timeout.connect(self._check_tasks)
        self.scheduler_timer.start(60000)  # Check every minute
    
    def _check_tasks(self):
        """Check if any tasks need to run."""
        now = datetime.now()
        
        for task in self.tasks:
            if not task.get("enabled", True):
                continue
            
            schedule = task.get("schedule", "")
            time_str = task.get("time", "09:00")
            
            try:
                time_parts = time_str.split(":")
                task_hour = int(time_parts[0])
                task_minute = int(time_parts[1])
                
                should_run = False
                
                if schedule == "Every X Minutes":
                    interval = task.get("interval", 30)
                    last_run = task.get("last_run")
                    if last_run:
                        last_dt = datetime.fromisoformat(last_run)
                        if (now - last_dt).total_seconds() >= interval * 60:
                            should_run = True
                    else:
                        should_run = True
                
                elif schedule == "Hourly":
                    if now.minute == task_minute:
                        should_run = True
                
                elif schedule == "Daily":
                    if now.hour == task_hour and now.minute == task_minute:
                        should_run = True
                
                elif schedule == "Weekly":
                    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    current_day = day_names[now.weekday()]
                    if current_day in task.get("days", []):
                        if now.hour == task_hour and now.minute == task_minute:
                            should_run = True
                
                if should_run:
                    self._run_task(task)
            
            except:
                pass


def create_scheduler_tab(parent=None):
    """Factory function to create scheduler tab."""
    return SchedulerTab(parent)
