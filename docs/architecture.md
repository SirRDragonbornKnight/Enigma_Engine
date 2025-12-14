# Enigma Engine - Architecture (minimal)

- **enigma.core**: tokenizer, model, training, inference for a toy transformer.
- **enigma.memory**: sqlite persistence and simple vector utilities.
- **enigma.voice**: TTS / STT stubs.
- **enigma.comms**: local API server and remote client wrapper.
- **enigma.avatar**: stub controller for an avatar engine.
- **enigma.tools**: OS and screenshot helpers.
- **enigma.utils**: IO helpers.

This layout keeps responsibilities isolated so you can replace modules independently.
