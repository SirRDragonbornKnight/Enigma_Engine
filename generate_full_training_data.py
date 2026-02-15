"""Generate full training dataset from codebase analysis."""
from enigma_engine.self_improvement.analyzer import CodeAnalyzer
from enigma_engine.self_improvement.data_generator import TrainingDataGenerator
from pathlib import Path
from datetime import datetime

print("Analyzing codebase...")
analyzer = CodeAnalyzer(Path("enigma_engine"))
analysis = analyzer.analyze()
print(f"Found {len(analysis['all_classes'])} classes, {len(analysis['all_functions'])} functions")

# Convert 'all_*' to 'new_*' so generator processes everything
full_analysis = {
    "new_classes": analysis["all_classes"],
    "new_functions": analysis["all_functions"],
    "new_gui_elements": analysis["all_gui_elements"],
}

print("Generating training pairs...")
generator = TrainingDataGenerator()
pairs = generator.generate_from_analysis(full_analysis)
print(f"Generated {len(pairs)} Q&A pairs")

# Save ALL pairs
output = Path("data/full_training_data.txt")
output.parent.mkdir(exist_ok=True)
with open(output, "w", encoding="utf-8") as f:
    f.write("# Auto-generated training data from Enigma Engine codebase\n")
    f.write(f"# Generated: {datetime.now().isoformat()}\n")
    f.write(f"# Total pairs: {len(pairs)}\n\n")
    for pair in pairs:
        f.write(f"Q: {pair.question}\n")
        f.write(f"A: {pair.answer}\n\n")

print(f"Saved to {output}")
print(f"File size: {output.stat().st_size / 1024:.1f} KB")
