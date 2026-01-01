# New Features Documentation

This document describes all the new features and quality-of-life improvements added to Enigma AI Engine.

## Table of Contents

1. [Interactive Tools](#interactive-tools)
2. [Error Handling](#error-handling)
3. [Persona System](#persona-system)
4. [Keyboard Shortcuts](#keyboard-shortcuts)
5. [Feedback System](#feedback-system)
6. [Discovery Mode](#discovery-mode)
7. [Training Validator](#training-validator)
8. [Resource Allocator](#resource-allocator)

---

## Interactive Tools

New personal assistant capabilities including checklists, tasks, and reminders.

### Checklists

Create and manage checklists for organizing tasks.

```python
from enigma.tools.interactive_tools import ChecklistManager

manager = ChecklistManager()

# Create a checklist
result = manager.create_checklist(
    "Shopping List",
    ["Milk", "Bread", "Eggs", "Cheese"]
)

# List all checklists
checklists = manager.list_checklists()

# Update item status
manager.update_item("checklist_1", item_index=0, completed=True)
```

**Tool Usage:**
```python
from enigma.tools import tool_registry

# Create checklist
tool_registry.execute("create_checklist", 
    name="Weekend Plans",
    items="Clean house, Buy groceries, Call family"
)

# List checklists
tool_registry.execute("list_checklists")
```

### Task Scheduler

Manage tasks with due dates and priorities.

```python
from enigma.tools.interactive_tools import TaskScheduler

scheduler = TaskScheduler()

# Add a task
task = scheduler.add_task(
    title="Finish project report",
    description="Complete Q1 project report",
    due_date="2025-12-31T17:00:00",
    priority="high"
)

# List tasks
tasks = scheduler.list_tasks(priority="high")

# Complete a task
scheduler.complete_task("task_1")
```

**Tool Usage:**
```python
# Add task
tool_registry.execute("add_task",
    title="Review pull request",
    description="Check PR #123",
    due_date="2025-12-20T15:00:00",
    priority="medium"
)

# List tasks
tool_registry.execute("list_tasks", show_completed=False)

# Complete task
tool_registry.execute("complete_task", task_id="task_1")
```

### Reminders

Set reminders with optional repeat functionality.

```python
from enigma.tools.interactive_tools import ReminderSystem

system = ReminderSystem()

# Set a reminder
reminder = system.set_reminder(
    message="Team meeting",
    remind_at="2025-12-20T14:00:00",
    repeat="weekly"
)

# Check for due reminders
due = system.check_due_reminders()

# List all reminders
reminders = system.list_reminders(active_only=True)
```

**Tool Usage:**
```python
# Set reminder
tool_registry.execute("set_reminder",
    message="Doctor appointment",
    remind_at="2025-12-25T10:00:00",
    repeat="monthly"
)

# Check reminders
tool_registry.execute("check_reminders")

# List reminders
tool_registry.execute("list_reminders", active_only=True)
```

---

## Error Handling

Graceful error handling with user-friendly messages and recovery suggestions.

### Error Handler

```python
from enigma.utils.error_handler import ErrorHandler

try:
    # Some operation that might fail
    open("/nonexistent/file.txt")
except Exception as e:
    result = ErrorHandler.handle_error(e, "file reading")
    print(result['friendly_message'])
    print("Suggestions:")
    for suggestion in result['suggestions']:
        print(f"  - {suggestion}")
```

### Graceful File Operations

```python
from enigma.utils.error_handler import GracefulFileHandler

# Read file with error handling
result = GracefulFileHandler.read_file("document.txt")
if result['success']:
    content = result['content']
else:
    print(result['friendly_message'])
    for suggestion in result['suggestions']:
        print(f"  - {suggestion}")

# Write file with error handling
result = GracefulFileHandler.write_file(
    "output.txt",
    "Hello, world!",
    mode="w"
)
```

### Decorator for Error Handling

```python
from enigma.utils.error_handler import graceful_errors

@graceful_errors("processing user data")
def process_user_data(data):
    # Process data
    return result

# Automatically handles errors and returns friendly messages
result = process_user_data(user_input)
```

---

## Persona System

Predefined AI personality templates for different use cases.

### Available Personas

1. **Teacher** - Patient educator who explains concepts clearly
2. **Assistant** - Helpful organizer focused on tasks and productivity
3. **Tech Expert** - Technical specialist with deep knowledge
4. **Friend** - Casual, empathetic conversational partner
5. **Researcher** - Analytical thinker who examines evidence
6. **Creative** - Imaginative thinker who generates ideas

### Using Personas

```python
from enigma.utils.personas import PersonaManager

manager = PersonaManager()

# Get a persona
teacher = manager.get_persona('teacher')
print(teacher.system_prompt)
print(teacher.traits)

# Apply to config
config = {}
config = manager.apply_persona_to_config('assistant', config)

# Create custom persona
manager.create_custom_persona(
    name="Coach",
    description="Motivational fitness coach",
    system_prompt="You are a motivational fitness coach...",
    tone="energetic",
    traits=["motivational", "encouraging", "goal-oriented"]
)
```

### Persona Example

```python
# Teacher persona example
teacher = manager.get_persona('teacher')

# System prompt includes:
# - Be patient and supportive
# - Explain step-by-step
# - Use examples and analogies
# - Check for understanding
# - Encourage questions

# Example responses demonstrate the teaching style
for example in teacher.example_responses:
    print(f"User: {example['user']}")
    print(f"AI: {example['ai']}\n")
```

---

## Keyboard Shortcuts

Customizable keyboard shortcuts for faster GUI navigation.

### Default Shortcuts

**Navigation:**
- `Ctrl+L` - Focus input field
- `Ctrl+1` - Chat tab
- `Ctrl+2` - Training tab
- `Ctrl+3` - Vision tab
- `Ctrl+4` - Settings tab
- `Ctrl+Tab` - Next tab
- `Ctrl+Shift+Tab` - Previous tab

**Actions:**
- `Ctrl+Return` - Send message
- `Ctrl+N` - New conversation
- `Ctrl+S` - Save conversation
- `Ctrl+T` - Start training
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo

**Application:**
- `F1` - Help
- `Ctrl+Q` - Quit
- `F11` - Fullscreen

### Using Shortcuts

```python
from enigma.utils.shortcuts import ShortcutManager
from PyQt5.QtWidgets import QMainWindow

window = QMainWindow()
manager = ShortcutManager(window)

# Register a shortcut
def on_send():
    print("Sending message...")

manager.register('send_message', on_send)

# Update a shortcut
manager.update_shortcut('send_message', 'Ctrl+Enter')

# Get all shortcuts
shortcuts = manager.get_all_shortcuts()

# Format for display
print(manager.format_for_display())
```

### Undo/Redo Manager

```python
from enigma.utils.shortcuts import UndoRedoManager

manager = UndoRedoManager()

# Track an action
def do_action():
    # Perform action
    pass

def undo_action():
    # Undo the action
    pass

manager.push_action({
    'description': 'Added message',
    'undo_func': undo_action,
    'redo_func': do_action
})

# Undo
if manager.can_undo():
    manager.undo()

# Redo
if manager.can_redo():
    manager.redo()
```

---

## Feedback System

Collect user ratings and feedback on AI responses.

### Adding Ratings

```python
from enigma.utils.feedback import FeedbackCollector

collector = FeedbackCollector()

# Add star rating (1-5)
collector.add_rating(
    response_id="resp_123",
    rating=5,
    prompt="Explain quantum computing",
    response="Quantum computing uses quantum mechanics...",
    categories={
        "helpful": 5,
        "accurate": 5,
        "clear": 4,
        "complete": 5
    },
    text_feedback="Great explanation!"
)

# Add thumbs up/down
collector.add_thumbs(
    response_id="resp_124",
    thumbs_up=True,
    prompt="Tell me a joke",
    response="Why did the programmer quit?...",
    reason="Made me laugh!"
)
```

### Getting Statistics

```python
# Get overall statistics
stats = collector.get_statistics()
print(f"Average rating: {stats['average_rating']}")
print(f"Thumbs up: {stats['thumbs_up_percent']}%")
print(f"Total feedback: {stats['total_feedback']}")

# Category averages
for category, avg in stats['category_averages'].items():
    print(f"{category}: {avg}")
```

### Analyzing Feedback

```python
# Get low-rated responses for review
low_rated = collector.get_recent_low_ratings(threshold=2, limit=10)
for feedback in low_rated:
    print(f"Rating: {feedback['rating']}")
    print(f"Prompt: {feedback['prompt']}")
    print(f"Feedback: {feedback.get('text_feedback', 'N/A')}")

# Get high-quality responses
high_rated = collector.get_high_performing_responses(threshold=4, limit=10)

# Export for training
collector.export_for_training(
    Path("training_data.txt"),
    min_rating=4
)
```

---

## Discovery Mode

Autonomous exploration and learning when the AI is idle.

### Enabling Discovery Mode

```python
from enigma.utils.discovery_mode import DiscoveryMode

discovery = DiscoveryMode()

# Enable with idle threshold (seconds)
discovery.enable(idle_threshold=300)  # 5 minutes

# Check if idle
if discovery.is_idle():
    topic = discovery.get_discovery_topic()
    query = discovery.suggest_research_query(topic)
    # Research the topic...
    discovery.log_discovery(topic, findings, related_topics)

# Disable
discovery.disable()
```

### Discovery Topics

The system explores diverse topics:
- Science & Technology
- Arts & Culture
- History & Society
- Nature & Environment
- Practical Skills

### Managing Discoveries

```python
# Get recent discoveries
recent = discovery.get_recent_discoveries(limit=10)

# Get summary
summary = discovery.get_discovery_summary()
print(f"Total discoveries: {summary['total_discoveries']}")
print(f"Topics explored: {', '.join(summary['topics_explored'])}")

# Export discoveries
discovery.export_discoveries(Path("discoveries.txt"))
```

### Auto-Save Manager

```python
from enigma.utils.discovery_mode import AutoSaveManager

autosave = AutoSaveManager()

# Enable auto-save
autosave.enable(interval=60)  # Save every 60 seconds

# Save conversation
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
]
autosave.save_conversation("conv_123", messages)

# Save training state
state = {"epoch": 10, "loss": 0.5}
autosave.save_training_state("my_model", state)

# Recover conversation
recovered = autosave.recover_conversation("conv_123")

# List all auto-saves
saves = autosave.list_auto_saves()

# Clean old saves
autosave.clean_old_saves(days=7)
```

---

## Training Validator

Validate training data before training to ensure quality.

### Basic Validation

```python
from enigma.utils.training_validator import TrainingDataValidator

validator = TrainingDataValidator()

# Validate file
result = validator.validate_file("training_data.txt")

# Or validate text directly
data = """
Q: What is Python?
A: Python is a programming language.

Q: What are variables?
A: Variables store data.
"""

result = validator.validate_text(data)

# Check results
if result['valid']:
    print("✓ Data is valid!")
else:
    print("✗ Data has issues:")
    for issue in result['issues']:
        print(f"  - {issue}")

# Generate report
report = validator.generate_report(result)
print(report)
```

### Understanding Results

```python
result = {
    'valid': True/False,
    'issues': [...],  # Must fix
    'warnings': [...],  # Should fix
    'suggestions': [...],  # Recommendations
    'statistics': {
        'estimated_pairs': 100,
        'avg_prompt_length': 50,
        'avg_response_length': 150,
        'duplicate_rate': 0.05
    }
}
```

### Training Data Formatter

```python
from enigma.utils.training_validator import TrainingDataFormatter

# Format Q&A pair
formatted = TrainingDataFormatter.format_qa_pair(
    "What is AI?",
    "AI is artificial intelligence..."
)

# Convert formats
converted = TrainingDataFormatter.convert_to_format(
    content,
    target_format="qa"  # or "conversation"
)
```

---

## Resource Allocator

Manage CPU, RAM, and GPU usage with different performance modes.

### Performance Modes

1. **Minimal** - Low resource usage, runs in background
2. **Balanced** - Good balance (recommended)
3. **Performance** - Fast processing, uses more resources
4. **Maximum** - All available resources

### Using Resource Allocator

```python
from enigma.utils.resource_allocator import ResourceAllocator

allocator = ResourceAllocator()

# Set mode
allocator.set_mode('balanced')

# Get current mode
mode = allocator.get_current_mode()
print(f"Mode: {mode['mode']}")
print(f"Settings: {mode['settings']}")

# Get system info
info = allocator.get_system_info()
print(f"CPU cores: {info['cpu_count']}")
print(f"RAM: {info['memory_total_gb']} GB")
print(f"GPU: {info.get('gpu_available', False)}")

# Get recommended mode
recommended = allocator.get_recommended_mode()
```

### Speed vs Quality

```python
# Set preference
allocator.set_speed_vs_quality('quality')

# Get generation parameters
params = allocator.get_generation_params()
print(params)
# {
#     'max_tokens': 500,
#     'temperature': 0.8,
#     'num_beams': 4,
#     ...
# }

# Get training parameters
train_params = allocator.get_training_params()

# Check if model fits
fits = allocator.check_if_model_fits('large')
```

### Performance Monitoring

```python
from enigma.utils.resource_allocator import PerformanceMonitor

monitor = PerformanceMonitor()

# Start monitoring
monitor.start()

# Record operations
monitor.record("loading model")
# ... do work ...
monitor.record("generating response")

# Get summary
summary = monitor.get_summary()
print(f"Total time: {summary['total_time']}s")
print(f"Average CPU: {summary['avg_cpu']}%")
print(f"Max memory: {summary['max_memory']}%")
```

---

## Quick Reference

### Import Paths

```python
# Interactive tools
from enigma.tools.interactive_tools import (
    ChecklistManager, TaskScheduler, ReminderSystem,
    CreateChecklistTool, AddTaskTool, SetReminderTool
)

# Error handling
from enigma.utils.error_handler import (
    ErrorHandler, GracefulFileHandler, graceful_errors
)

# Personas
from enigma.utils.personas import PersonaManager, PREDEFINED_PERSONAS

# Shortcuts
from enigma.utils.shortcuts import ShortcutManager, UndoRedoManager

# Feedback
from enigma.utils.feedback import FeedbackCollector

# Discovery & Auto-save
from enigma.utils.discovery_mode import DiscoveryMode, AutoSaveManager

# Training validator
from enigma.utils.training_validator import TrainingDataValidator

# Resource allocation
from enigma.utils.resource_allocator import ResourceAllocator

# Tool registry
from enigma.tools.tool_registry import tool_registry
```

### Running the Demo

```bash
python demo_core_features.py
```

### Running Tests

```bash
python -m unittest tests.test_core_features_qol
```

---

## Best Practices

1. **Error Handling**: Always use `GracefulFileHandler` for file operations
2. **Personas**: Choose appropriate persona for your use case
3. **Feedback**: Collect feedback regularly to improve model quality
4. **Training**: Always validate training data before training
5. **Resources**: Set appropriate resource mode based on hardware
6. **Auto-save**: Enable auto-save for long-running operations

---

## See Also

- Main README: `README.md`
- API Documentation: `docs/`
- Examples: `examples/`
- Tests: `tests/test_core_features_qol.py`
