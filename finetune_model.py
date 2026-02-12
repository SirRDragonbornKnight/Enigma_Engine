#!/usr/bin/env python3
"""Fine-tune the existing Enigma model on training data."""

import sys
import os

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from enigma_engine.core.incremental_training import continue_training

def main():
    print("Starting fine-tuning on existing model...")
    print("=" * 50)
    
    try:
        results = continue_training(
            model_path="models/enigma_engine",
            new_data_path="data/training.txt", 
            epochs=10,
            output_path="models/enigma_engine"
        )
        
        print("=" * 50)
        print("Training complete!")
        print(f"Epochs completed: {results['epochs_completed']}")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Total steps: {results['total_steps']}")
        
        if results.get('forgetting_detected'):
            print("WARNING: Some forgetting was detected during training")
            
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Trying alternative model path...")
        
        # Try with the .pth file directly
        results = continue_training(
            model_path="models",
            new_data_path="data/training.txt", 
            epochs=10,
            output_path="models/enigma_engine"
        )
        print(f"Training complete! Final loss: {results['final_loss']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
