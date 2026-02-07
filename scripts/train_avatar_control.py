"""
Train Enigma AI Engine Model for Avatar Bone Control

Trains the AI to understand bone control commands and execute them naturally.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enigma_engine.core.training import train_model, TrainingConfig
from enigma_engine.config import CONFIG

def main():
    print("=" * 80)
    print("Training Enigma AI Engine for Avatar Bone Control")
    print("=" * 80)
    
    # Training data file
    training_file = project_root / "data" / "specialized" / "avatar_control_training.txt"
    
    if not training_file.exists():
        print(f"Error: Training file not found: {training_file}")
        return
    
    print(f"\nTraining file: {training_file}")
    print(f"Lines in file: {len(training_file.read_text().splitlines())}")
    
    # Training configuration
    config = TrainingConfig(
        model_name="avatar_control",
        data_file=str(training_file),
        model_size="small",  # Small model for fast avatar control
        epochs=10,
        batch_size=4,
        learning_rate=0.0001,
        save_every=2,
        use_mixed_precision=False,  # More stable for small models
    )
    
    print("\nTraining Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Size: {config.model_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # Train the model
    try:
        train_model(config)
        
        print("\n" + "=" * 80)
        print("✓ Training complete!")
        print("=" * 80)
        print(f"\nModel saved to: {CONFIG['models_dir']}/{config.model_name}/")
        print("\nTo use this model:")
        print("  1. Load it in the GUI Model Manager")
        print("  2. Enable the 'avatar' module")
        print("  3. Upload a rigged 3D avatar (GLB/GLTF with bones)")
        print("  4. Chat with AI: 'Wave hello' or 'Nod your head'")
        print("\nThe AI will now control the avatar through bone animations!")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
