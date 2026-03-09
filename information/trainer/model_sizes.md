# Model Sizes

How the model size system works in Enigma Engine.

---

## Specifying Size

When creating or training a model, type your target parameter count:

| Input | Meaning |
|-------|---------|
| `8b` | 8 billion parameters |
| `1.5b` | 1.5 billion parameters |
| `500m` | 500 million parameters |
| `27m` | 27 million parameters |

The engine matches your target to the closest architecture preset.

---

## Available Presets (auto-matched)

| Preset | Parameters | Best For |
|--------|-----------|----------|
| pi_zero | ~500K | Testing, Raspberry Pi Zero |
| nano | ~1M | Microcontrollers, basic tasks |
| tiny | ~5M | Edge devices, simple chat |
| small | ~27M | Entry GPU, learning |
| medium | ~85M | Mid-range GPU |
| large | ~200M | RTX 3080+ |
| xl | ~600M | RTX 4090 |
| xxl | ~1.5B | Multi-GPU |
| huge | ~3B | Server GPU |
| giant | ~7B | Multi-node datacenter |
| colossal | ~13B | Distributed cloud |
| titan | ~30B | Full datacenter |
| omega | ~70B+ | Research frontier |

---

## How Matching Works

1. You type a target like `8b`
2. The engine calculates estimated parameters for each preset
3. It picks the preset closest to your target
4. The matched preset name is shown in the training log

You can also type preset names directly (e.g. `small`, `giant`).

---

## Hardware Requirements

| Size | Minimum VRAM |
|------|-------------|
| Under 100M | CPU or any GPU |
| 100M - 500M | 4 GB VRAM |
| 500M - 2B | 8 GB VRAM |
| 2B - 7B | 16 GB VRAM |
| 7B - 13B | 24 GB+ VRAM |
| 13B+ | Multi-GPU / 48 GB+ |
