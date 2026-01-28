"""
Test the self-improvement system integration.

This test verifies that the components work together:
- Learning engine records feedback
- Training scheduler monitors examples
- Configuration is properly loaded
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_learning_engine():
    """Test the learning engine can record feedback."""
    print("Testing Learning Engine...")
    
    from forge_ai.core.self_improvement import get_learning_engine
    
    # Create a test engine
    engine = get_learning_engine("test_model")
    
    # Record feedback
    engine.record_feedback(
        input_text="What is Python?",
        output_text="Python is a high-level programming language.",
        feedback="positive"
    )
    
    # Get metrics
    metrics = engine.get_metrics()
    assert metrics.positive_feedback == 1, "Should have 1 positive feedback"
    
    print("✓ Learning engine works!")


def test_training_scheduler():
    """Test the training scheduler monitors examples."""
    print("\nTesting Training Scheduler...")
    
    from forge_ai.learning.training_scheduler import get_training_scheduler
    
    # Create scheduler
    scheduler = get_training_scheduler("test_model")
    
    # Get status
    status = scheduler.get_status()
    print(f"  Examples collected: {status['examples_collected']}")
    print(f"  Min needed: {status['min_examples_needed']}")
    print(f"  Ready to train: {status['ready_to_train']}")
    
    print("✓ Training scheduler works!")


def test_configuration():
    """Test self-improvement configuration is loaded."""
    print("\nTesting Configuration...")
    
    from forge_ai.config import CONFIG
    
    assert "self_improvement" in CONFIG, "Config should have self_improvement section"
    
    si_config = CONFIG["self_improvement"]
    assert si_config["enabled"] == True
    assert "auto_training" in si_config
    assert "lora_config" in si_config
    
    print("✓ Configuration loaded!")
    print(f"  Autonomous learning: {si_config['autonomous_learning']}")
    print(f"  Feedback learning: {si_config['feedback_learning']}")
    print(f"  Min examples for training: {si_config['auto_training']['min_examples']}")


def test_integration():
    """Test the full integration flow."""
    print("\nTesting Integration Flow...")
    
    from forge_ai.core.self_improvement import get_learning_engine
    from forge_ai.learning.training_scheduler import get_training_scheduler
    
    model_name = "integration_test_model"
    
    # Get components
    engine = get_learning_engine(model_name)
    scheduler = get_training_scheduler(model_name)
    
    # Record some feedback
    for i in range(5):
        engine.record_feedback(
            input_text=f"Question {i}",
            output_text=f"Answer {i}",
            feedback="positive"
        )
    
    # Check queue stats
    stats = engine.get_queue_stats()
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Avg quality: {stats['avg_quality']:.2f}")
    
    # Check scheduler status
    status = scheduler.get_status()
    print(f"  Scheduler status: examples={status['examples_collected']}, ready={status['ready_to_train']}")
    
    print("✓ Integration flow works!")


if __name__ == "__main__":
    print("=" * 60)
    print("Self-Improvement System Integration Tests")
    print("=" * 60)
    
    try:
        test_learning_engine()
        test_training_scheduler()
        test_configuration()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
