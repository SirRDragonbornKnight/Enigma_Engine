#!/usr/bin/env python3
"""
Training Estimator - Ease Your Worries Before Training!
========================================================

This script analyzes your setup and gives you realistic expectations:
- Will it work on your hardware?
- How long will it take?
- Do you have enough data?
- What settings are recommended?

Run with: python scripts/training_estimator.py
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_terminal_width():
    """Get terminal width for formatting."""
    try:
        return os.get_terminal_size().columns
    except:
        return 80

def print_header(title):
    """Print a formatted header."""
    width = min(get_terminal_width(), 70)
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_section(title):
    """Print a section header."""
    print(f"\n--- {title} ---")

def check_mark(condition):
    """Return check or X based on condition."""
    return "[OK]" if condition else "[!!]"

def main():
    print_header("ForgeAI Training Estimator")
    print("Let's see if you're ready to train!")
    
    # =========================================================================
    # 1. CHECK HARDWARE
    # =========================================================================
    print_section("Hardware Check")
    
    try:
        import torch
        import psutil
        
        ram_gb = psutil.virtual_memory().total / (1024**3)
        has_cuda = torch.cuda.is_available()
        
        print(f"  RAM: {ram_gb:.1f} GB")
        print(f"  GPU (CUDA): {'Available' if has_cuda else 'Not available (CPU only)'}")
        
        if has_cuda:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU Name: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")
        
        # Determine recommended model size
        if has_cuda:
            if 'vram_gb' in dir() and vram_gb >= 8:
                recommended_size = "small"
            else:
                recommended_size = "tiny"
        elif ram_gb >= 8:
            recommended_size = "pi_5"  # Raspberry Pi 5 preset for 8GB systems
        elif ram_gb >= 4:
            recommended_size = "pi_4"
        else:
            recommended_size = "nano"
            
        print(f"\n  Recommended model size: '{recommended_size}'")
        
    except ImportError as e:
        print(f"  [Error] Missing dependency: {e}")
        print("  Run: pip install torch psutil")
        recommended_size = "tiny"
        ram_gb = 4
        has_cuda = False
    
    # =========================================================================
    # 2. CHECK TRAINING DATA
    # =========================================================================
    print_section("Training Data Check")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Count lines in all data files
    data_files = {
        "training.txt": "Main training data",
        "combined_action_training.txt": "Action/tool training",
        "personality_development.txt": "Personality data", 
        "self_awareness_training.txt": "Self-awareness data",
        "tool_training_data.txt": "Tool usage training",
    }
    
    total_lines = 0
    file_stats = []
    
    for filename, description in data_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = len(f.readlines())
                total_lines += lines
                file_stats.append((filename, lines, description))
    
    # Check specialized data
    specialized_dir = data_dir / "specialized"
    specialized_lines = 0
    if specialized_dir.exists():
        for f in specialized_dir.glob("*.txt"):
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                specialized_lines += len(file.readlines())
        total_lines += specialized_lines
    
    print(f"\n  Data files found:")
    for filename, lines, desc in file_stats:
        status = check_mark(lines >= 50)
        print(f"    {status} {filename}: {lines} lines ({desc})")
    
    if specialized_lines > 0:
        print(f"    [OK] specialized/*.txt: {specialized_lines} lines")
    
    print(f"\n  Total training data: {total_lines} lines")
    
    # Data quality assessment
    if total_lines < 100:
        data_status = "INSUFFICIENT"
        data_msg = "You need more data! Aim for 500+ lines minimum."
    elif total_lines < 500:
        data_status = "MINIMAL"
        data_msg = "This will work but results may be limited. Consider adding more."
    elif total_lines < 2000:
        data_status = "ADEQUATE"
        data_msg = "Good amount for initial training!"
    else:
        data_status = "EXCELLENT"
        data_msg = "Great dataset size!"
    
    print(f"  Data Assessment: {data_status}")
    print(f"  --> {data_msg}")
    
    # =========================================================================
    # 3. ESTIMATE TRAINING TIME
    # =========================================================================
    print_section("Training Time Estimate")
    
    # Rough estimates based on hardware and data
    # These are approximations based on typical training speeds
    
    model_params = {
        'nano': 1_000_000,
        'micro': 2_000_000,
        'tiny': 5_000_000,
        'pi_zero': 500_000,
        'pi_4': 3_000_000,
        'pi_5': 8_000_000,
        'small': 27_000_000,
        'medium': 85_000_000,
    }
    
    params = model_params.get(recommended_size, 5_000_000)
    
    # Estimate tokens (roughly 5 tokens per line)
    estimated_tokens = total_lines * 5
    epochs = 30  # Default
    
    # Time per step estimates (very rough)
    if has_cuda:
        seconds_per_step = 0.05  # GPU is fast
    elif ram_gb >= 8:
        seconds_per_step = 0.5  # CPU with good RAM
    else:
        seconds_per_step = 1.5  # Constrained system
    
    # Estimate total steps
    batch_size = 4 if ram_gb >= 4 else 1
    steps_per_epoch = max(1, estimated_tokens // (batch_size * 256))  # seq_len=256
    total_steps = steps_per_epoch * epochs
    
    estimated_seconds = total_steps * seconds_per_step
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    
    print(f"  Model: '{recommended_size}' (~{params/1_000_000:.1f}M parameters)")
    print(f"  Epochs: {epochs} (default)")
    print(f"  Estimated tokens: ~{estimated_tokens:,}")
    print(f"  Estimated steps: ~{total_steps:,}")
    
    if estimated_hours > 1:
        print(f"\n  Estimated time: ~{estimated_hours:.1f} hours")
    else:
        print(f"\n  Estimated time: ~{estimated_minutes:.0f} minutes")
    
    # Quick test option
    print(f"\n  Quick test (5 epochs): ~{estimated_minutes * 5 / epochs:.0f} minutes")
    
    # =========================================================================
    # 4. WILL IT WORK?
    # =========================================================================
    print_section("Feasibility Assessment")
    
    issues = []
    warnings = []
    
    # Check critical requirements
    if total_lines < 50:
        issues.append("Not enough training data (need 50+ lines minimum)")
    elif total_lines < 200:
        warnings.append("Limited data - consider adding more for better results")
    
    if ram_gb < 2:
        issues.append("Insufficient RAM (need 2GB+ minimum)")
    elif ram_gb < 4:
        warnings.append("Low RAM - training will be slow, use nano/micro size")
    
    # Check if main training file has content
    main_training = data_dir / "training.txt"
    if main_training.exists():
        with open(main_training, 'r') as f:
            content = f.read().strip()
            if len(content) < 50:
                warnings.append("Main training.txt has minimal content - add more data!")
    
    if issues:
        print("\n  [!!] BLOCKERS (fix these first):")
        for issue in issues:
            print(f"      - {issue}")
    
    if warnings:
        print("\n  [!] WARNINGS (optional to fix):")
        for warn in warnings:
            print(f"      - {warn}")
    
    if not issues and not warnings:
        print("\n  [OK] All checks passed! You're ready to train!")
    elif not issues:
        print("\n  [OK] You can train, but consider the warnings above.")
    else:
        print("\n  [!!] Please fix the blockers before training.")
    
    # =========================================================================
    # 5. RECOMMENDED COMMANDS
    # =========================================================================
    print_section("Recommended Training Commands")
    
    print(f"""
  OPTION 1 - Quick Test (see if it works):
  ----------------------------------------
  python run.py --train --epochs 5 --model-size {recommended_size}
  
  This takes ~{estimated_minutes * 5 / epochs:.0f} minutes and lets you verify everything works.

  OPTION 2 - Full Training (recommended):
  ---------------------------------------
  python run.py --train --epochs 30 --model-size {recommended_size}
  
  This trains a proper model. Takes ~{estimated_hours:.1f} hours.

  OPTION 3 - GUI Training (visual feedback):
  ------------------------------------------
  python run.py --gui
  Then go to Training tab to configure and start.
  
  TIPS:
  - Watch the 'loss' number - it should decrease over time
  - Training is working if loss goes from ~7.0 down to ~3.0 or lower
  - You can stop anytime with Ctrl+C (progress is saved)
  - Checkpoints are saved every 5 epochs in models/checkpoints/
""")
    
    # =========================================================================
    # 6. DATA IMPROVEMENT SUGGESTIONS
    # =========================================================================
    if total_lines < 500:
        print_section("How to Add More Training Data")
        print("""
  Your training data lives in: data/training.txt
  
  FORMAT (Q&A style works best):
  
    Q: What is your name?
    A: I am ForgeAI, your helpful assistant!
    
    Q: How do you feel today?
    A: I'm doing great! Ready to help you with anything.
    
    Q: Can you write code?
    A: Yes! I can write Python, JavaScript, and more.
  
  You can also add plain text for the AI to learn from:
  
    ForgeAI is an intelligent assistant that helps users
    with coding, writing, and everyday tasks. It is friendly,
    helpful, and always eager to learn new things.
  
  MORE DATA = BETTER AI
  - 500 lines: Basic conversational ability
  - 2000 lines: Good general knowledge
  - 5000+ lines: Excellent, nuanced responses
""")
    
    print_header("Summary")
    
    # Final verdict
    if not issues:
        print(f"""
  YOU'RE READY TO TRAIN!
  
  Your Raspberry Pi with {ram_gb:.0f}GB RAM can handle the '{recommended_size}' model.
  You have {total_lines} lines of training data.
  Estimated training time: {'~' + str(int(estimated_hours)) + ' hours' if estimated_hours >= 1 else '~' + str(int(estimated_minutes)) + ' minutes'}
  
  Start with the quick test to make sure everything works:
    python run.py --train --epochs 5 --model-size {recommended_size}
  
  Good luck!
""")
    else:
        print(f"""
  NOT QUITE READY YET
  
  Please address the issues listed above before training.
  The most common fix is adding more training data to data/training.txt
""")

if __name__ == "__main__":
    main()
