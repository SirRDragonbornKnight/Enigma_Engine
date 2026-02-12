"""Quick integration test for self-improvement system."""

from enigma_engine.self_improvement import CodeAnalyzer, TrainingDataGenerator

# Test code analysis
print("Testing CodeAnalyzer...")
analyzer = CodeAnalyzer('enigma_engine')
result = analyzer.analyze()
print(f"  Analyzed {result['total_files']} files")
print(f"  Classes: {len(result['all_classes'])}")
print(f"  Functions: {len(result['all_functions'])}")
print(f"  GUI elements: {len(result['all_gui_elements'])}")

# Test training data generation
print("\nTesting TrainingDataGenerator...")
generator = TrainingDataGenerator()
pairs = generator.generate_from_analysis(result)
print(f"  Generated {len(pairs)} training pairs")
if pairs:
    print(f"  Example Q: {pairs[0].question[:70]}...")
    print(f"  Example A: {pairs[0].answer[:100]}...")

print("\nSelf-improvement pipeline OK!")
