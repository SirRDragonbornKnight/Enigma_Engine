# Credits & Attributions

Enigma Engine is built on the shoulders of giants. This project uses and is inspired by several excellent open-source libraries and research papers.

## Core Dependencies

### PyTorch
**License:** BSD-3-Clause  
**Website:** https://pytorch.org/  
**Purpose:** Neural network training and inference  
PyTorch is the foundation of our transformer model implementation.

### Transformers (Hugging Face)
**License:** Apache-2.0  
**Website:** https://huggingface.co/transformers  
**Purpose:** Tokenizer implementations and model utilities  
We use concepts and patterns from the transformers library.

## GUI Framework

### PyQt5
**License:** GPL v3 / Commercial  
**Website:** https://www.riverbankcomputing.com/software/pyqt/  
**Purpose:** Desktop GUI interface  
PyQt5 provides the rich graphical interface for the desktop application.

## AI Generation Libraries (Optional)

### Diffusers (Hugging Face)
**License:** Apache-2.0  
**Website:** https://github.com/huggingface/diffusers  
**Purpose:** Stable Diffusion image generation  
Used for local image generation when the `image_gen_local` module is enabled.

### Sentence Transformers
**License:** Apache-2.0  
**Website:** https://www.sbert.net/  
**Purpose:** Text embeddings and semantic search  
Powers the local embeddings module for vector search capabilities.

## Speech & Audio

### pyttsx3
**License:** MPL-2.0  
**Website:** https://github.com/nateshmbhat/pyttsx3  
**Purpose:** Text-to-speech synthesis  
Cross-platform TTS engine for voice output.

### SpeechRecognition
**License:** BSD-3-Clause  
**Website:** https://github.com/Uberi/speech_recognition  
**Purpose:** Speech-to-text recognition  
Enables voice input capabilities.

## Utilities

### Flask
**License:** BSD-3-Clause  
**Website:** https://flask.palletsprojects.com/  
**Purpose:** API server and web interface  
Powers the REST API for remote access.

### SQLAlchemy
**License:** MIT  
**Website:** https://www.sqlalchemy.org/  
**Purpose:** Database management  
Used for structured memory storage.

### Pillow (PIL)
**License:** HPND  
**Website:** https://python-pillow.org/  
**Purpose:** Image processing  
Handles image capture, manipulation, and format conversions.

### MSS (Multiple Screen Shots)
**License:** MIT  
**Website:** https://github.com/BoboTiG/python-mss  
**Purpose:** Screen capture  
Fast cross-platform screen capture for vision features.

## Research & Inspiration

This project implements concepts from various research papers and open-source projects:

### Transformer Architecture
- **"Attention Is All You Need"** (Vaswani et al., 2017)
- Original transformer architecture that revolutionized NLP

### Rotary Position Embeddings (RoPE)
- **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (Su et al., 2021)
- More efficient positional encoding used in modern LLMs

### RMSNorm
- **"Root Mean Square Layer Normalization"** (Zhang & Sennrich, 2019)
- Faster normalization technique used in LLaMA and other models

### SwiGLU Activation
- **"GLU Variants Improve Transformer"** (Shazeer, 2020)
- Activation function used in PaLM and LLaMA models

### Grouped Query Attention (GQA)
- **"GQA: Training Generalized Multi-Query Transformer Models"** (Ainslie et al., 2023)
- Memory-efficient attention mechanism

### LLaMA Architecture
- **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023)
- Architectural decisions and best practices for transformer models

## Design Patterns

### Modular Architecture
The module system design is inspired by:
- **Plugin architectures** from Eclipse, VSCode, and WordPress
- **Dependency injection** patterns from modern frameworks
- **Capability-based security** models

## Community Contributions

Special thanks to:
- The PyTorch team for an incredible deep learning framework
- Hugging Face for democratizing AI with open models and tools
- The open-source AI community for sharing knowledge and code

## No External Code Copied

**Important:** While Enigma Engine is inspired by and uses libraries from the above projects, **all core code is original**:

- ✅ Transformer model implementation is written from scratch
- ✅ Module manager system is original design
- ✅ Training loops and inference engine are custom
- ✅ GUI and all tabs are original implementations
- ✅ Tool integrations are custom wrappers

We follow best practices and architectural patterns from the community but do not copy-paste code from other projects (except when using their libraries as intended via pip install).

## License Compliance

All dependencies are used in accordance with their respective licenses. Enigma Engine itself is released under the MIT License, allowing free use, modification, and distribution.

---

**If you believe we've used your code or ideas without proper attribution, please open an issue and we'll correct it immediately.**
