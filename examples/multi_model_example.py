"""
EXAMPLE: Creating and Training Multiple Named AI Models

This demonstrates the workflow for:
  1. Creating different named AI models
  2. Training each on different data
  3. Loading and using them independently

NO NEED TO COPY THE ENGINE - one engine, many models!
"""

from enigma.core.model_registry import ModelRegistry
from enigma.core.model_config import print_model_info
from enigma.core.trainer import EnigmaTrainer, train_model_by_name

# =============================================================================
# STEP 1: See available model sizes
# =============================================================================

print("\n" + "="*70)
print("AVAILABLE MODEL SIZES")
print("="*70)
print_model_info()


# =============================================================================
# STEP 2: Create the Model Registry
# =============================================================================

registry = ModelRegistry()

# See what models exist
registry.list_models()


# =============================================================================
# STEP 3: Create Your First AI
# =============================================================================

# Create a model named "artemis" - starts as a blank slate
# It will learn ONLY from the data you feed it

print("\n" + "="*70)
print("CREATING FIRST MODEL")
print("="*70)

# Uncomment to create:
# model_artemis = registry.create_model(
#     name="artemis",
#     size="small",          # or "tiny", "medium", "large", etc.
#     vocab_size=32000,
#     description="My first AI - trained on philosophical texts"
# )


# =============================================================================
# STEP 4: Create Another AI with Different Purpose
# =============================================================================

print("\n" + "="*70)
print("CREATING SECOND MODEL")
print("="*70)

# Uncomment to create:
# model_apollo = registry.create_model(
#     name="apollo",
#     size="small",
#     vocab_size=32000,
#     description="Technical assistant - trained on code and documentation"
# )


# =============================================================================
# STEP 5: Train Each Model on Different Data
# =============================================================================

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Train artemis on philosophical text:
# trainer_artemis = EnigmaTrainer(
#     model=model_artemis,
#     model_name="artemis",
#     registry=registry,
#     data_path="data/philosophy.txt",  # Your training data
#     use_multi_gpu=True,               # Use all GPUs
#     batch_size=8,
#     learning_rate=1e-4,
# )
# trainer_artemis.train(epochs=100, save_every=10)

# Train apollo on technical text:
# trainer_apollo = EnigmaTrainer(
#     model=model_apollo,
#     model_name="apollo",
#     registry=registry,
#     data_path="data/technical.txt",
#     use_multi_gpu=True,
# )
# trainer_apollo.train(epochs=100)


# =============================================================================
# STEP 6: Load and Use Any Model
# =============================================================================

print("\n" + "="*70)
print("LOADING AND USING MODELS")
print("="*70)

# Load artemis:
# artemis, config = registry.load_model("artemis")
# print(artemis)

# Load apollo:
# apollo, config = registry.load_model("apollo")

# You can also load from a specific checkpoint:
# artemis_v2, _ = registry.load_model("artemis", checkpoint="epoch_50")


# =============================================================================
# STEP 7: The Quick Way (All-in-One)
# =============================================================================

print("\n" + "="*70)
print("QUICK METHOD")
print("="*70)

# This creates AND trains in one call:
# train_model_by_name(
#     name="athena",
#     data_path="data/my_data.txt",
#     epochs=100,
#     size="medium",
#     use_multi_gpu=True
# )


# =============================================================================
# KEY POINTS
# =============================================================================

print("""
================================================================================
                              KEY POINTS
================================================================================

1. ONE ENGINE, MANY MODELS
   - Don't copy the engine folder
   - Each model is saved in: models/<name>/
   - The engine loads whichever model you specify

2. MODELS START AS BLANK SLATES
   - No emotions, no personality hardcoded
   - They learn patterns ONLY from your training data
   - Want a philosophical AI? Train on philosophy
   - Want a coding AI? Train on code

3. EACH MODEL HAS:
   - models/<name>/config.json      - Architecture settings
   - models/<name>/metadata.json    - Training history
   - models/<name>/weights.pth      - The trained brain
   - models/<name>/checkpoints/     - Saved progress points

4. TRAINING DATA MATTERS
   - The AI becomes what it eats
   - More data = better learning
   - Diverse data = more capable
   - Clean data = cleaner outputs

5. MODEL SIZES
   - tiny:   Testing/mobile, ~2M params
   - small:  Learning experiments, ~15M params
   - medium: Real use, needs GPU, ~50M params
   - large:  Serious, needs good GPU, ~125M params (GPT-2 small)
   - xl:     High quality, ~350M params (GPT-2 medium)
   - xxl:    Production, ~770M params (GPT-2 large)
   - xxxl:   Maximum, ~1.5B params (GPT-2 XL)

================================================================================
""")

if __name__ == "__main__":
    print("Run this file to see the workflow.")
    print("Uncomment sections to actually create/train models.")
