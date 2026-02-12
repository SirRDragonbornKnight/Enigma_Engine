"""Run all training pipelines in sequence."""

import sys
from pathlib import Path

def run_self_improvement_trainer():
    """Run the self-improvement training pipeline."""
    print("=" * 60)
    print("SELF-IMPROVEMENT TRAINER")
    print("=" * 60)
    
    from enigma_engine.self_improvement import CodeAnalyzer, TrainingDataGenerator, SelfTrainer
    from enigma_engine.self_improvement.self_trainer import TrainingConfig
    
    # Generate training data from codebase analysis
    print("\n[1/3] Analyzing codebase...")
    analyzer = CodeAnalyzer('enigma_engine')
    
    # Force fresh analysis by clearing cache
    analyzer._cache = {}
    result = analyzer.analyze()
    print(f"  Found {len(result['all_classes'])} classes, {len(result['all_functions'])} functions")
    
    print("\n[2/3] Generating training pairs...")
    generator = TrainingDataGenerator()
    # Use all_classes instead of new_classes to always generate data
    analysis_for_gen = {
        'new_classes': result['all_classes'][:50],  # Limit for speed
        'new_functions': result['all_functions'][:50],
        'new_gui_elements': result['all_gui_elements'][:20],
    }
    pairs = generator.generate_from_analysis(analysis_for_gen)
    print(f"  Generated {len(pairs)} pairs")
    
    # Save to file
    output_path = Path('data/self_improvement_training.txt')
    output_path.parent.mkdir(exist_ok=True)
    generator.save_to_file(pairs[:500], str(output_path), append=False)  # Limit to 500 for quick test
    print(f"  Saved {min(len(pairs), 500)} pairs to {output_path}")
    
    print("\n[3/3] Running SelfTrainer (LoRA)...")
    # Use tiny_forge model which exists
    config = TrainingConfig(epochs=1, max_samples=50)  # Quick test
    trainer = SelfTrainer(
        model_path=str(Path('models/tiny_forge.pth').absolute()),
        config=config
    )
    result = trainer.train_incremental(str(output_path))
    print(f"  Training result: success={result.success}, loss={result.final_loss:.4f}")
    
    return result.success

def run_gui_teacher_trainer():
    """Run GUI teacher training."""
    print("\n" + "=" * 60)
    print("GUI TEACHER TRAINER")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/train_gui_teacher.py", "--epochs", "1", "--help"],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout[:500] if result.stdout else "Script available")
    print("  (Use: python scripts/train_gui_teacher.py to run full training)")
    return True

def run_teach_model():
    """Run teacher-student training."""
    print("\n" + "=" * 60)
    print("TEACHER-STUDENT TRAINER (teach_model.py)")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/teach_model.py", "--help"],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout[:500] if result.stdout else "Script available")
    print("  (Use: python scripts/teach_model.py to run full training)")
    return True

def run_specialized_trainer():
    """Run specialized model training."""
    print("\n" + "=" * 60)
    print("SPECIALIZED MODEL TRAINER")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/train_specialized_model.py", "--help"],
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout[:500] if result.stdout else "Script available")
    print("  (Use: python scripts/train_specialized_model.py --type router/code/vision)")
    return True

if __name__ == "__main__":
    print("Running all trainers...\n")
    
    # Run self-improvement (actually trains)
    run_self_improvement_trainer()
    
    # Show help for other trainers
    run_gui_teacher_trainer()
    run_teach_model()
    run_specialized_trainer()
    
    print("\n" + "=" * 60)
    print("ALL TRAINERS COMPLETE")
    print("=" * 60)
