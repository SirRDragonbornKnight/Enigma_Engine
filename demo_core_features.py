"""
Demo Script - New Core Features and QoL Improvements

This script demonstrates all the new features added to Enigma AI Engine.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_interactive_tools():
    """Demo interactive tools: checklists, tasks, reminders."""
    print("\n" + "=" * 70)
    print("INTERACTIVE TOOLS DEMO")
    print("=" * 70)
    
    from enigma.tools.interactive_tools import ChecklistManager, TaskScheduler, ReminderSystem
    
    # Checklists
    print("\n1. Creating a Checklist:")
    manager = ChecklistManager()
    result = manager.create_checklist(
        "Morning Routine",
        ["Wake up", "Exercise", "Breakfast", "Check emails"]
    )
    print(f"   Created: {result['name']} with {result['items']} items")
    
    checklists = manager.list_checklists()
    for cl in checklists:
        print(f"   - {cl['name']}: {cl['completed_items']}/{cl['total_items']} complete")
    
    # Tasks
    print("\n2. Adding Tasks:")
    scheduler = TaskScheduler()
    task = scheduler.add_task(
        "Finish project proposal",
        "Complete the Q1 project proposal document",
        due_date="2025-12-31T17:00:00",
        priority="high"
    )
    print(f"   Created task: {task['title']}")
    
    tasks = scheduler.list_tasks()
    print(f"   Total active tasks: {len(tasks)}")
    
    # Reminders
    print("\n3. Setting Reminders:")
    reminder_sys = ReminderSystem()
    reminder = reminder_sys.set_reminder(
        "Call mom",
        "2025-12-25T15:00:00",
        repeat="weekly"
    )
    print(f"   Set reminder: {reminder['message']}")
    print(f"   Reminds at: {reminder['remind_at']}")


def demo_error_handling():
    """Demo graceful error handling."""
    print("\n" + "=" * 70)
    print("ERROR HANDLING DEMO")
    print("=" * 70)
    
    from enigma.utils.error_handler import ErrorHandler, GracefulFileHandler
    
    # Test file error
    print("\n1. Handling Missing File:")
    result = GracefulFileHandler.read_file("/nonexistent/file.txt")
    if not result['success']:
        print(f"   ✗ {result['friendly_message']}")
        print("   Suggestions:")
        for i, suggestion in enumerate(result['suggestions'], 1):
            print(f"      {i}. {suggestion}")
    
    # Test error recovery
    print("\n2. Error Recovery Example:")
    try:
        raise ValueError("Invalid input: expected number, got text")
    except Exception as e:
        result = ErrorHandler.handle_error(e, "user input validation")
        print(f"   Error: {result['friendly_message']}")
        print(f"   Type: {result['error_type']}")


def demo_personas():
    """Demo persona system."""
    print("\n" + "=" * 70)
    print("PERSONA SYSTEM DEMO")
    print("=" * 70)
    
    from enigma.utils.personas import PersonaManager
    
    manager = PersonaManager()
    personas = manager.list_personas()
    
    print(f"\n{len(personas)} Available Personas:")
    for name, persona in list(personas.items())[:3]:
        print(f"\n   {persona.name}:")
        print(f"   {persona.description}")
        print(f"   Tone: {persona.tone}")
        print(f"   Traits: {', '.join(persona.traits)}")


def demo_feedback():
    """Demo feedback collection."""
    print("\n" + "=" * 70)
    print("FEEDBACK SYSTEM DEMO")
    print("=" * 70)
    
    from enigma.utils.feedback import FeedbackCollector
    
    collector = FeedbackCollector()
    
    # Add some ratings
    print("\n1. Adding Ratings:")
    collector.add_rating(
        "response_1",
        rating=5,
        prompt="Explain quantum computing",
        response="Quantum computing uses quantum mechanics...",
        categories={"helpful": 5, "accurate": 5, "clear": 4}
    )
    print("   Added 5-star rating")
    
    collector.add_thumbs(
        "response_2",
        thumbs_up=True,
        prompt="Tell me a joke",
        response="Why did the programmer quit? No arrays!"
    )
    print("   Added thumbs up")
    
    # Get statistics
    print("\n2. Statistics:")
    stats = collector.get_statistics()
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Average rating: {stats.get('average_rating', 0)}")
    print(f"   Thumbs up: {stats.get('thumbs_up_percent', 0)}%")


def demo_discovery_mode():
    """Demo discovery mode and auto-save."""
    print("\n" + "=" * 70)
    print("DISCOVERY MODE & AUTO-SAVE DEMO")
    print("=" * 70)
    
    from enigma.utils.discovery_mode import DiscoveryMode, AutoSaveManager
    
    # Discovery mode
    print("\n1. Discovery Mode:")
    discovery = DiscoveryMode()
    discovery.enable(idle_threshold=300)
    
    topic = discovery.get_discovery_topic()
    query = discovery.suggest_research_query(topic)
    print(f"   Topic: {topic}")
    print(f"   Query: {query}")
    
    # Auto-save
    print("\n2. Auto-Save:")
    autosave = AutoSaveManager()
    autosave.enable(interval=60)
    
    messages = [
        {"role": "user", "content": "Hello AI"},
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
    
    success = autosave.save_conversation("demo_conv", messages)
    print(f"   Conversation saved: {success}")
    
    saves = autosave.list_auto_saves()
    print(f"   Total auto-saves: {len(saves)}")


def demo_training_validator():
    """Demo training data validator."""
    print("\n" + "=" * 70)
    print("TRAINING DATA VALIDATOR DEMO")
    print("=" * 70)
    
    from enigma.utils.training_validator import TrainingDataValidator
    
    validator = TrainingDataValidator()
    
    # Good data
    print("\n1. Validating Good Data:")
    good_data = """Q: What is Python?
A: Python is a high-level programming language.

Q: What are variables?
A: Variables store data in memory.

Q: What is a function?
A: A function is reusable code."""
    
    result = validator.validate_text(good_data)
    print(f"   Valid: {result['valid']}")
    print(f"   Issues: {len(result['issues'])}")
    print(f"   Warnings: {len(result['warnings'])}")
    print(f"   Estimated pairs: {result['statistics']['estimated_pairs']}")
    
    # Bad data
    print("\n2. Validating Bad Data:")
    bad_data = "Just some random text without any structure"
    
    result = validator.validate_text(bad_data)
    print(f"   Valid: {result['valid']}")
    print(f"   Issues: {len(result['issues'])}")
    if result['issues']:
        print("   First issue:", result['issues'][0])


def demo_resource_allocator():
    """Demo resource allocation (if psutil available)."""
    print("\n" + "=" * 70)
    print("RESOURCE ALLOCATOR DEMO")
    print("=" * 70)
    
    try:
        from enigma.utils.resource_allocator import ResourceAllocator
        
        allocator = ResourceAllocator()
        
        # Show modes
        print("\n1. Available Modes:")
        modes = allocator.get_all_modes()
        for mode_name, settings in modes.items():
            print(f"   {settings['name']}: {settings['description']}")
        
        # System info
        print("\n2. System Information:")
        info = allocator.get_system_info()
        print(f"   CPU cores: {info.get('cpu_count', 'N/A')}")
        print(f"   Total RAM: {info.get('memory_total_gb', 0):.1f} GB")
        print(f"   GPU available: {info.get('gpu_available', False)}")
        
        # Speed vs Quality
        print("\n3. Speed vs Quality Settings:")
        for pref in ['speed', 'balanced', 'quality']:
            allocator.set_speed_vs_quality(pref)
            params = allocator.get_generation_params()
            print(f"   {pref.capitalize()}: max_tokens={params['max_tokens']}, "
                  f"temperature={params['temperature']}")
    
    except ImportError:
        print("\n   ⚠ Requires psutil package (pip install psutil)")


def demo_tool_integration():
    """Demo tool registry integration."""
    print("\n" + "=" * 70)
    print("TOOL INTEGRATION DEMO")
    print("=" * 70)
    
    from enigma.tools.tool_registry import ToolRegistry
    
    registry = ToolRegistry()
    tools = registry.list_tools()
    
    # Filter new tools
    new_tools = [t for t in tools if 'checklist' in t['name'] or 'task' in t['name'] or 'reminder' in t['name']]
    
    print(f"\n{len(new_tools)} New Interactive Tools:")
    for tool in new_tools:
        print(f"   - {tool['name']}: {tool['description']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("ENIGMA AI ENGINE - NEW FEATURES DEMO")
    print("=" * 70)
    print("\nThis demo showcases the new core features and QoL improvements")
    
    try:
        demo_interactive_tools()
        demo_error_handling()
        demo_personas()
        demo_feedback()
        demo_discovery_mode()
        demo_training_validator()
        demo_resource_allocator()
        demo_tool_integration()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE!")
        print("=" * 70)
        print("\nAll new features demonstrated successfully! 🎉")
        print("\nNext steps:")
        print("  1. Explore the code in enigma/tools/ and enigma/utils/")
        print("  2. Run tests: python -m unittest tests.test_core_features_qol")
        print("  3. Try the GUI: python run.py --gui")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
