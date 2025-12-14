# Enigma Engine (minimal starter)

This repository contains a minimal, modular skeleton for the Enigma engine:
- small transformer-style toy model (PyTorch) for experimentation
- tokenizer utilities
- memory storage (sqlite + vectors)
- simple TTS/STT wrappers (placeholder)
- API server and remote client stubs
- avatar API stub
- utility tools for screenshots and OS tasks

This is intended as a starting point. Replace components with production libraries (Hugging Face, ONNX, TFLite, Coqui, etc.) as needed.

## Quickstart
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
2. Train tiny model (toy):
    python run.py --train
3. Run a demo server:
    python run.py --serve
4. Run interactive demo:
    python run.py --run

## Structure
See the repository tree in the project description.

---

### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="enigma_engine",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # kept minimal; see requirements.txt
    ],
    entry_points={
        'console_scripts': [
            'enigma = run:main'
        ]
    }
)

