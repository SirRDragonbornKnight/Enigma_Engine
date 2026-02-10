"""
Interactive Tools - Checklist, Task Scheduler, and Reminder System

Tools for personal assistant functionality:
  - create_checklist: Create and manage checklists
  - add_task: Add a task with optional due date/reminder
  - list_tasks: View all tasks and their status
  - complete_task: Mark a task as complete
  - set_reminder: Set a reminder for a specific time
  - list_reminders: View all active reminders
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .tool_registry import Tool, RichParameter


class ChecklistManager:
    """Manages checklists and tasks."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "checklists.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.checklists = self._load()
    
    def _load(self) -> dict:
        """Load checklists from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
    
    def _save(self):
        """Save checklists to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.checklists, f, indent=2)
    
    def create_checklist(self, name: str, items: list[str]) -> dict[str, Any]:
        """Create a new checklist."""
        checklist_id = f"checklist_{len(self.checklists) + 1}"
        self.checklists[checklist_id] = {
            'name': name,
            'created': datetime.now().isoformat(),
            'items': [{'text': item, 'completed': False} for item in items]
        }
        self._save()
        return {
            'success': True,
            'id': checklist_id,
            'name': name,
            'items': len(items)
        }
    
    def get_checklist(self, checklist_id: str) -> Optional[dict]:
        """Get a specific checklist."""
        return self.checklists.get(checklist_id)
    
    def list_checklists(self) -> list[dict]:
        """List all checklists."""
        return [
            {
                'id': cid,
                'name': data['name'],
                'created': data['created'],
                'total_items': len(data['items']),
                'completed_items': sum(1 for item in data['items'] if item['completed'])
            }
            for cid, data in self.checklists.items()
        ]
    
    def update_item(self, checklist_id: str, item_index: int, completed: bool) -> bool:
        """Update checklist item status."""
        if checklist_id in self.checklists:
            items = self.checklists[checklist_id]['items']
            if 0 <= item_index < len(items):
                items[item_index]['completed'] = completed
                self._save()
                return True
        return False
    
    def delete_checklist(self, checklist_id: str) -> bool:
        """Delete a checklist."""
        if checklist_id in self.checklists:
            del self.checklists[checklist_id]
            self._save()
            return True
        return False


class TaskScheduler:
    """Manages tasks with due dates and priorities."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "tasks.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks = self._load()
    
    def _load(self) -> dict:
        """Load tasks from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
    
    def _save(self):
        """Save tasks to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.tasks, f, indent=2)
    
    def add_task(self, title: str, description: str = "", 
                 due_date: Optional[str] = None, 
                 priority: str = "medium") -> dict[str, Any]:
        """Add a new task."""
        task_id = f"task_{len(self.tasks) + 1}"
        self.tasks[task_id] = {
            'title': title,
            'description': description,
            'created': datetime.now().isoformat(),
            'due_date': due_date,
            'priority': priority,
            'completed': False,
            'completed_at': None
        }
        self._save()
        return {
            'success': True,
            'id': task_id,
            'title': title
        }
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get a specific task."""
        return self.tasks.get(task_id)
    
    def list_tasks(self, show_completed: bool = False, 
                   priority: Optional[str] = None) -> list[dict]:
        """List tasks with optional filtering."""
        tasks = []
        for tid, task in self.tasks.items():
            if not show_completed and task['completed']:
                continue
            if priority and task['priority'] != priority:
                continue
            
            task_info = {
                'id': tid,
                'title': task['title'],
                'description': task['description'],
                'created': task['created'],
                'due_date': task['due_date'],
                'priority': task['priority'],
                'completed': task['completed']
            }
            
            # Calculate if overdue
            if task['due_date'] and not task['completed']:
                try:
                    due = datetime.fromisoformat(task['due_date'])
                    task_info['overdue'] = due < datetime.now()
                except (ValueError, TypeError):
                    task_info['overdue'] = False
            
            tasks.append(task_info)
        
        return tasks
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as complete."""
        if task_id in self.tasks:
            self.tasks[task_id]['completed'] = True
            self.tasks[task_id]['completed_at'] = datetime.now().isoformat()
            self._save()
            return True
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save()
            return True
        return False


class ReminderSystem:
    """Manages reminders with notifications."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "reminders.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.reminders = self._load()
    
    def _load(self) -> dict:
        """Load reminders from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
    
    def _save(self):
        """Save reminders to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.reminders, f, indent=2)
    
    def set_reminder(self, message: str, remind_at: str, 
                     repeat: Optional[str] = None) -> dict[str, Any]:
        """Set a new reminder."""
        reminder_id = f"reminder_{len(self.reminders) + 1}"
        self.reminders[reminder_id] = {
            'message': message,
            'created': datetime.now().isoformat(),
            'remind_at': remind_at,
            'repeat': repeat,  # None, 'daily', 'weekly', 'monthly'
            'active': True,
            'triggered': False
        }
        self._save()
        return {
            'success': True,
            'id': reminder_id,
            'message': message,
            'remind_at': remind_at
        }
    
    def get_reminder(self, reminder_id: str) -> Optional[dict]:
        """Get a specific reminder."""
        return self.reminders.get(reminder_id)
    
    def list_reminders(self, active_only: bool = True) -> list[dict]:
        """List all reminders."""
        reminders = []
        for rid, reminder in self.reminders.items():
            if active_only and not reminder['active']:
                continue
            
            reminders.append({
                'id': rid,
                'message': reminder['message'],
                'remind_at': reminder['remind_at'],
                'repeat': reminder['repeat'],
                'active': reminder['active'],
                'triggered': reminder['triggered']
            })
        
        return reminders
    
    def check_due_reminders(self) -> list[dict]:
        """Check for reminders that are due."""
        due_reminders = []
        now = datetime.now()
        
        for rid, reminder in self.reminders.items():
            if not reminder['active']:
                continue
            
            try:
                remind_time = datetime.fromisoformat(reminder['remind_at'])
                if remind_time <= now and not reminder['triggered']:
                    due_reminders.append({
                        'id': rid,
                        'message': reminder['message']
                    })
                    
                    # Handle repeating reminders
                    if reminder['repeat'] == 'daily':
                        reminder['remind_at'] = (remind_time + timedelta(days=1)).isoformat()
                    elif reminder['repeat'] == 'weekly':
                        reminder['remind_at'] = (remind_time + timedelta(weeks=1)).isoformat()
                    elif reminder['repeat'] == 'monthly':
                        reminder['remind_at'] = (remind_time + timedelta(days=30)).isoformat()
                    else:
                        reminder['triggered'] = True
                        reminder['active'] = False
            except (ValueError, KeyError, TypeError):
                continue
        
        if due_reminders:
            self._save()
        
        return due_reminders
    
    def cancel_reminder(self, reminder_id: str) -> bool:
        """Cancel a reminder."""
        if reminder_id in self.reminders:
            self.reminders[reminder_id]['active'] = False
            self._save()
            return True
        return False


# Tool implementations
class CreateChecklistTool(Tool):
    """Create a new checklist."""
    
    name = "create_checklist"
    description = "Create a new checklist with items. Good for organizing tasks."
    parameters = {
        "name": "Name of the checklist",
        "items": "List of checklist items (comma-separated or list)",
    }
    category = "interactive"
    rich_parameters = [
        RichParameter(name="name", type="string", description="Name of the checklist", required=True),
        RichParameter(name="items", type="string", description="Comma-separated checklist items", required=True),
    ]
    examples = ["create_checklist(name='Shopping', items='milk, eggs, bread')"]
    
    def __init__(self):
        self.manager = ChecklistManager()
    
    def execute(self, name: str, items, **kwargs) -> dict[str, Any]:
        try:
            # Parse items
            if isinstance(items, str):
                items_list = [item.strip() for item in items.split(',')]
            elif isinstance(items, list):
                items_list = items
            else:
                return {"success": False, "error": "Items must be a string or list"}
            
            result = self.manager.create_checklist(name, items_list)
            result['checklist'] = self.manager.get_checklist(result['id'])
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListChecklistsTool(Tool):
    """List all checklists."""
    
    name = "list_checklists"
    description = "List all created checklists and their status."
    parameters = {}
    category = "interactive"
    rich_parameters = []
    examples = ["list_checklists()"]
    
    def __init__(self):
        self.manager = ChecklistManager()
    
    def execute(self, **kwargs) -> dict[str, Any]:
        try:
            checklists = self.manager.list_checklists()
            return {
                "success": True,
                "checklists": checklists,
                "count": len(checklists)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class AddTaskTool(Tool):
    """Add a new task."""
    
    name = "add_task"
    description = "Add a task with optional due date and priority."
    parameters = {
        "title": "Task title",
        "description": "Task description (optional)",
        "due_date": "Due date in ISO format like '2024-12-31T17:00:00' (optional)",
        "priority": "Priority: low, medium, or high (default: medium)",
    }
    category = "interactive"
    rich_parameters = [
        RichParameter(name="title", type="string", description="Task title", required=True),
        RichParameter(name="description", type="string", description="Task description", required=False),
        RichParameter(name="due_date", type="string", description="Due date (ISO format)", required=False),
        RichParameter(name="priority", type="string", description="Task priority", required=False, default="medium", enum=["low", "medium", "high"]),
    ]
    examples = ["add_task(title='Review PR', priority='high')", "add_task(title='Meeting', due_date='2024-12-31T10:00:00')"]
    
    def __init__(self):
        self.scheduler = TaskScheduler()
    
    def execute(self, title: str, description: str = "", 
                due_date: Optional[str] = None, 
                priority: str = "medium", **kwargs) -> dict[str, Any]:
        try:
            return self.scheduler.add_task(title, description, due_date, priority)
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListTasksTool(Tool):
    """List all tasks."""
    
    name = "list_tasks"
    description = "List all tasks. Can filter by priority or show completed tasks."
    parameters = {
        "show_completed": "Whether to show completed tasks (default: false)",
        "priority": "Filter by priority: low, medium, or high (optional)",
    }
    category = "interactive"
    rich_parameters = [
        RichParameter(name="show_completed", type="boolean", description="Show completed tasks", required=False, default=False),
        RichParameter(name="priority", type="string", description="Filter by priority", required=False, enum=["low", "medium", "high"]),
    ]
    examples = ["list_tasks()", "list_tasks(priority='high')", "list_tasks(show_completed=True)"]
    
    def __init__(self):
        self.scheduler = TaskScheduler()
    
    def execute(self, show_completed: bool = False, 
                priority: Optional[str] = None, **kwargs) -> dict[str, Any]:
        try:
            tasks = self.scheduler.list_tasks(show_completed, priority)
            return {
                "success": True,
                "tasks": tasks,
                "count": len(tasks)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class CompleteTaskTool(Tool):
    """Mark a task as complete."""
    
    name = "complete_task"
    description = "Mark a task as complete."
    parameters = {
        "task_id": "The ID of the task to complete",
    }
    category = "interactive"
    rich_parameters = [
        RichParameter(name="task_id", type="string", description="The task ID to complete", required=True),
    ]
    examples = ["complete_task(task_id='task_abc123')"]
    
    def __init__(self):
        self.scheduler = TaskScheduler()
    
    def execute(self, task_id: str, **kwargs) -> dict[str, Any]:
        try:
            success = self.scheduler.complete_task(task_id)
            if success:
                return {"success": True, "message": f"Task {task_id} completed"}
            return {"success": False, "error": "Task not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class SetReminderTool(Tool):
    """Set a reminder."""
    
    name = "set_reminder"
    description = "Set a reminder for a specific time."
    parameters = {
        "message": "The reminder message",
        "remind_at": "When to remind in ISO format like '2024-12-31T17:00:00'",
        "repeat": "Repeat frequency: daily, weekly, monthly, or none (optional)",
    }
    category = "interactive"
    rich_parameters = [
        RichParameter(name="message", type="string", description="Reminder message", required=True),
        RichParameter(name="remind_at", type="string", description="Time to remind (ISO format)", required=True),
        RichParameter(name="repeat", type="string", description="Repeat frequency", required=False, enum=["none", "daily", "weekly", "monthly"]),
    ]
    examples = ["set_reminder(message='Call mom', remind_at='2024-12-31T17:00:00')", "set_reminder(message='Take meds', remind_at='2024-12-31T08:00:00', repeat='daily')"]
    
    def __init__(self):
        self.system = ReminderSystem()
    
    def execute(self, message: str, remind_at: str, 
                repeat: Optional[str] = None, **kwargs) -> dict[str, Any]:
        try:
            return self.system.set_reminder(message, remind_at, repeat)
        except Exception as e:
            return {"success": False, "error": str(e)}


class ListRemindersTool(Tool):
    """List all reminders."""
    
    name = "list_reminders"
    description = "List all active reminders."
    parameters = {
        "active_only": "Whether to show only active reminders (default: true)",
    }
    category = "interactive"
    rich_parameters = [
        RichParameter(name="active_only", type="boolean", description="Show only active reminders", required=False, default=True),
    ]
    examples = ["list_reminders()", "list_reminders(active_only=False)"]
    
    def __init__(self):
        self.system = ReminderSystem()
    
    def execute(self, active_only: bool = True, **kwargs) -> dict[str, Any]:
        try:
            reminders = self.system.list_reminders(active_only)
            return {
                "success": True,
                "reminders": reminders,
                "count": len(reminders)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class CheckRemindersTool(Tool):
    """Check for due reminders."""
    
    name = "check_reminders"
    description = "Check for reminders that are due right now."
    parameters = {}
    category = "interactive"
    rich_parameters = []
    examples = ["check_reminders()"]
    
    def __init__(self):
        self.system = ReminderSystem()
    
    def execute(self, **kwargs) -> dict[str, Any]:
        try:
            due = self.system.check_due_reminders()
            return {
                "success": True,
                "due_reminders": due,
                "count": len(due)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test the tools
    import json

    # Test checklist
    tool = CreateChecklistTool()
    result = tool.execute("Shopping List", "Milk, Bread, Eggs, Cheese")
    print("Checklist created:")
    print(json.dumps(result, indent=2))
    
    # Test task
    task_tool = AddTaskTool()
    result = task_tool.execute(
        "Finish project report",
        "Complete the quarterly project report",
        "2024-12-31T17:00:00",
        "high"
    )
    print("\nTask added:")
    print(json.dumps(result, indent=2))
    
    # List tasks
    list_tool = ListTasksTool()
    result = list_tool.execute()
    print("\nTasks:")
    print(json.dumps(result, indent=2))
