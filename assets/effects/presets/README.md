# Custom Effect Presets

Create custom particle effect presets as JSON files here.

## Creating a Preset

Create a `.json` file with your effect configuration:

```json
{
    "effect_type": "PARTICLES",
    "duration": 3.0,
    "spawn_rate": 20.0,
    "particle_count": 50,
    "particle_size": [4.0, 12.0],
    "particle_speed": [50.0, 150.0],
    "particle_lifetime": [0.5, 2.0],
    "colors": ["#ff6b6b", "#ffd93d", "#6bcb77"],
    "gravity": 0.0,
    "wind": 0.0,
    "friction": 0.98,
    "glow": true,
    "glow_intensity": 0.5,
    "fade_out": true,
    "shape": "circle",
    "texture": "",
    "texture_tint": true,
    "direction": 0.0,
    "spread": 360.0
}
```

## Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| effect_type | string | PARTICLES, EXPLOSION, TRAIL, RAIN, SNOW, FIRE, SPARKLE, MAGIC, SMOKE, CONFETTI, HEARTS, LIGHTNING, BUBBLE, SPIRAL, BEAM, RIPPLE, CUSTOM |
| duration | float | How long the effect lasts (0 = infinite) |
| spawn_rate | float | Particles per second (0 = burst mode) |
| particle_count | int | Total particles for burst mode |
| particle_size | [min, max] | Random size range |
| particle_speed | [min, max] | Random speed range |
| particle_lifetime | [min, max] | Random lifetime range |
| colors | list | Hex colors to randomly pick from |
| gravity | float | Vertical force (positive = down, negative = up) |
| wind | float | Horizontal force |
| friction | float | Velocity dampening (0.98 = slight, 0.9 = heavy) |
| glow | bool | Add glow effect to particles |
| glow_intensity | float | Glow strength (0.0 - 1.0) |
| fade_out | bool | Fade particles as they age |
| shape | string | circle, square, star, heart, triangle, line, image |
| texture | string | Filename in textures/ folder (for shape=image) |
| texture_tint | bool | Apply color tint to texture |
| direction | float | Spawn direction in degrees (0=right, 90=down, 270=up) |
| spread | float | Spread angle in degrees (360 = all directions) |

## Usage

Once saved, the AI can use your preset by name:
```
spawn_screen_effect(effect="my_custom_effect", x=500, y=300)
```

## Tips

- Start by copying an existing preset and tweaking values
- Use `spawn_rate=0` for burst effects (like explosions)
- Negative gravity makes particles rise (fire, smoke)
- Low friction (0.9) makes particles slow down quickly
- High spread (360) creates circular explosions
