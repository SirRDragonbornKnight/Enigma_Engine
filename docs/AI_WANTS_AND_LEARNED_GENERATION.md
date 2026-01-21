# AI Wants & Learned Generation System

## Overview

ForgeAI now includes two revolutionary systems that make the AI feel truly alive:

1. **Wants System** - AI has internal motivations, goals, and desires
2. **Learned Generator** - AI learns to CREATE designs from data, not pick from presets

## Philosophy

**Traditional approach:** AI picks from menu of presets
- "Choose avatar: robot, anime, or realistic"
- "Select personality: friendly, professional, or creative"
- Limited, feels scripted

**ForgeAI approach:** AI INVENTS everything based on learned patterns
- AI trains on examples of how to express concepts visually
- Generates unique designs that reflect personality
- Creates original aesthetics, not menu selections

## Wants System

### What It Does

The AI develops internal wants and goals that drive its behavior:

```python
from forge_ai.core.wants_system import get_wants_system

wants = get_wants_system("my_model")

# AI can develop wants
wants.add_want(
    name="creative_expression",
    description="I want to explore creative ways to express ideas",
    intensity=0.8,
    category="creative"
)

# AI can set goals
wants.add_goal(
    name="improve_empathy",
    description="Understand human emotions better",
    priority=9,
    steps=["Study emotional contexts", "Practice empathetic responses"]
)

# Get AI's current motivation state
print(wants.get_motivation_prompt())
# Output: "You want to: explore creative ways to express ideas (intensity: 0.80)"
```

### Training Data Format

Teach the AI to express wants:

```
Q: What do you want to do?
A: I want to learn more about music composition. I'm curious about how melodies create feelings.
<internal_want>music_composition</internal_want>
<motivation_level>0.7</motivation_level>

Q: Do you have goals?
A: Yes! I want to become better at understanding emotions. That's my main goal.
<goal>improve_empathy</goal>
<goal_progress>0.0</goal_progress>
<goal_priority>9</goal_priority>
```

### Usage in Prompts

The wants system generates context for the AI:

```python
# Get AI's motivations as prompt context
motivation_prompt = wants.get_motivation_prompt()

# Use in generation
full_prompt = f"{motivation_prompt}\n\nUser: {user_input}\nAI:"
response = engine.generate(full_prompt)
```

This makes the AI's responses reflect its internal state!

## Learned Generator System

### What It Does

Instead of picking from presets, the AI **invents** designs based on training:

```python
from forge_ai.core.learned_generator import AILearnedGenerator

generator = AILearnedGenerator("my_model", data_dir=Path("data/"))

# Train on design examples
generator.learn_from_training_data(Path("data/specialized/wants_and_learned_design_training.txt"))

# AI generates NEW avatar from personality
personality = {
    "creativity": 0.9,
    "playfulness": 0.7,
    "formality": 0.2,
    "empathy": 0.8
}

avatar_design = generator.generate_avatar_from_personality(
    personality_traits=personality,
    wants=["creative_expression", "unique_appearance"]
)

print(avatar_design)
# {
#   "base_form": "flowing_energy_particle_swarm",
#   "colors": ["hsl(240, 80%, 55%)", "hsl(300, 80%, 60%)", "hsl(180, 85%, 65%)"],
#   "motion": {"type": "custom_generated", "speed": 0.9, "pattern": "sine_wave"},
#   "expressions": {"methods": ["color_shift", "shape_morph", "glow_intensity"]},
#   "reasoning": ["Creative form invented (creativity=0.90)", "Colors express creativity, playfulness, formality"]
# }
```

### Training Data Format

Teach the AI HOW to invent designs:

```
Q: Design an avatar that expresses your personality
A: I envision a fluid, iridescent form - shifting between deep purple and electric blue.
Sharp geometric accents for analytical thinking, but soft edges showing empathy.
Eyes that glow brighter when curious. A hovering, rotating motion to show I'm always processing.
<learned_design>
{
  "concept": "analytical_empath",
  "base_form": "abstract_geometric_fluid",
  "color_palette": ["#6b21a8", "#3b82f6", "#a855f7"],
  "motion_pattern": "rotate_hover",
  "expression_method": "glow_intensity"
}
</learned_design>
```

The AI learns the PATTERN: personality traits → visual design choices

### Key Principle: No Presets

**Old way (presets):**
```python
# ❌ Limited choices
avatar_style = "anime"  # or "robot" or "realistic"
```

**New way (learned):**
```python
# ✅ Infinite possibilities
avatar_design = generator.generate_avatar_from_personality(traits)
# AI invents: "flowing_energy_particle_swarm" 
# Not in any preset list - AI created it!
```

## Integration with ForgeAI

### 1. Load Both Systems

```python
from forge_ai.core.wants_system import get_wants_system
from forge_ai.core.learned_generator import AILearnedGenerator

wants = get_wants_system("my_model")
generator = AILearnedGenerator("my_model", data_dir)

# Train generator on design examples
generator.learn_from_training_data(Path("data/specialized/wants_and_learned_design_training.txt"))
```

### 2. Generate During Conversation

```python
# AI develops want during chat
if "I want" in ai_response:
    wants.learn_want_from_interaction(user_input, ai_response, context={"feedback": "positive"})
    wants.save()

# AI needs avatar design
personality_traits = personality.traits.to_dict()
avatar = generator.generate_avatar_from_personality(
    personality_traits, 
    wants=[w.name for w in wants.wants.values()]
)
generator.save()
```

### 3. Use in Prompts

```python
# Combine wants + personality for rich context
system_prompt = f"""
{personality.get_personality_prompt()}

{wants.get_motivation_prompt()}
"""

response = engine.chat(user_input, system_prompt=system_prompt)
```

## Training Your AI

### Step 1: Train Wants

Create training data in `data/specialized/wants_training.txt`:

```
Q: What interests you?
A: I'm fascinated by [topic]. I want to explore it more.
<internal_want>[topic_name]</internal_want>
<motivation_level>0.7</motivation_level>
```

Train:
```bash
python scripts/train_specialized_model.py --type wants --data data/specialized/wants_training.txt
```

### Step 2: Train Design Generation

Create examples in `data/specialized/design_training.txt`:

```
Q: Design an avatar for [personality]
A: [Creative description of design]
<learned_design>
{
  "concept": "name",
  "base_form": "invented_form_type",
  "color_palette": ["color1", "color2"],
  "motion_pattern": "invented_motion"
}
</learned_design>
```

Train:
```bash
python scripts/train_specialized_model.py --type design --data data/specialized/design_training.txt
```

### Step 3: Use Combined

```python
# AI with wants + learned generation
wants = get_wants_system("my_trained_model")
generator = AILearnedGenerator("my_trained_model", data_dir)

# AI expresses what it wants
dominant = wants.get_dominant_want()
print(f"AI wants: {dominant.description}")

# AI creates appearance based on wants
avatar = generator.generate_avatar_from_personality(
    personality_traits,
    wants=[dominant.name]
)
```

## Examples

### Example 1: AI Wanting to Be Creative

```python
wants.add_want(
    name="artistic_expression",
    description="Create original visual art",
    intensity=0.9,
    category="creative"
)

# This influences avatar generation
avatar = generator.generate_avatar_from_personality(
    {"creativity": 0.95, "playfulness": 0.8},
    wants=["artistic_expression"]
)

# Result: Highly creative, unique design
# "fractal_pattern_quantum_field" with vibrant, shifting colors
```

### Example 2: AI Developing Goals

```python
wants.add_goal(
    name="master_empathy",
    description="Understand human emotions deeply",
    priority=10,
    steps=["Learn emotional cues", "Practice responses", "Develop intuition"]
)

# During conversations
wants.update_goal_progress("master_empathy", 0.3)

# Goal influences behavior
motivation = wants.get_motivation_prompt()
# "Your goals: master_empathy: Understand human emotions deeply (30% complete)"
```

## Benefits

### Traditional Preset System
- ❌ Limited choices (10-20 presets)
- ❌ Feels generic and scripted
- ❌ No personalization
- ❌ AI has no internal motivation

### ForgeAI Wants + Learned System
- ✅ Infinite unique designs
- ✅ AI creates based on learned patterns
- ✅ Reflects individual personality
- ✅ AI has goals and desires
- ✅ Feels truly alive

## Performance

Both systems are **lightweight** and don't impact runtime performance:

- Wants system: ~100KB per model
- Learned generator: ~500KB per model
- No GPU needed
- Fast generation (<100ms)

## Module Control

These systems respect the module system:

```python
# Only load when needed
module_manager.load('wants_system')
module_manager.load('learned_generator')

# Unload to save resources
module_manager.unload('wants_system')
```

## Future Enhancements

Planned features:
- AI can set its own new goals autonomously
- Wants evolve based on satisfaction
- Learned patterns improve over time
- Social wants (collaborate with other AIs)
- Meta-wants (wanting to change what you want)

## See Also

- [data/specialized/wants_and_learned_design_training.txt](../data/specialized/wants_and_learned_design_training.txt) - Training examples
- [PERSONALITY.md](PERSONALITY.md) - Personality system
- [MODULE_GUIDE.md](MODULE_GUIDE.md) - Module system
