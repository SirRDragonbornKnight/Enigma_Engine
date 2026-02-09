# Effect Textures

Place custom particle texture images here for use with `spawn_screen_effect`.

## Supported Formats
- PNG (recommended - supports transparency)
- JPG/JPEG
- GIF
- BMP
- WebP

## Best Practices
- Use transparent backgrounds (PNG with alpha)
- Keep textures small: 32x32 to 128x128 pixels is ideal
- Square images work best
- Simple shapes read better as particles

## Example Textures to Add
- `star.png` - Star shape for sparkle effects
- `leaf.png` - Falling leaves
- `coin.png` - Coin rain
- `snowflake.png` - Detailed snowflakes
- `petal.png` - Flower petals
- `feather.png` - Floating feathers
- `orb.png` - Glowing orb with soft edges

## Usage

AI can use textures via the `spawn_screen_effect` tool:
```
spawn_screen_effect(effect="sparkle", texture="star.png", x=500, y=300)
```

Users can also create custom presets with textures - see `../presets/README.md`
