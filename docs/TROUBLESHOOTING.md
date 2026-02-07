# Enigma AI Engine Troubleshooting Guide

Quick solutions to common issues. Can't find your problem? Check the [GitHub Issues](https://github.com/SirRDragonbornKnight/enigma_engine/issues).

---

## Installation Problems

### "No module named 'torch'"

**Cause**: PyTorch not installed or wrong version.

**Fix**:
```bash
# CPU only (all platforms)
pip install torch torchvision torchaudio

# CUDA 11.8 (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (newer NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "No module named 'PyQt5'"

**Cause**: GUI dependencies missing.

**Fix**:
```bash
pip install PyQt5
```

**Linux users** may also need:
```bash
sudo apt-get install python3-pyqt5 libxcb-xinerama0
```

### PowerShell "cannot be loaded because running scripts is disabled"

**Cause**: PowerShell execution policy blocks the venv activation script.

**Fix**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Training Problems

### "Model gives gibberish output"

**Cause**: Insufficient training.

**Fixes**:
1. Train for more epochs (50+ recommended)
2. Add more training data (500+ Q&A pairs)
3. Check your training data format:
```
Q: What is your name?
A: I'm Forge, your AI assistant.
```

### "Loss stuck at high value (8+)"

**Cause**: Learning rate too high or data issues.

**Fixes**:
1. Reduce learning rate: `0.0001` → `0.00005`
2. Check training data isn't corrupted
3. Verify data format matches expected pattern

### "Loss goes to 0 immediately"

**Cause**: Data leakage or overfitting on tiny dataset.

**Fixes**:
1. Add more diverse training data
2. Check you're not accidentally training on test data
3. Reduce number of epochs

### "Training is very slow"

**Cause**: Large model on CPU or inefficient settings.

**Fixes**:
1. Use smaller model size (`tiny` or `small`)
2. Reduce batch size to 1-2
3. Enable GPU if available (check Settings → Hardware)
4. Reduce max sequence length

---

## Memory Problems

### "CUDA out of memory"

**Cause**: Model too large for your GPU VRAM.

**Fixes**:
1. Use smaller model size
2. Reduce batch size to 1
3. Enable gradient checkpointing (Settings → Training)
4. Close other GPU-using applications

### "Killed" or system becomes unresponsive

**Cause**: Running out of system RAM.

**Fixes**:
1. Use smaller model size
2. Reduce batch size
3. Close other applications
4. Add swap space (Linux/Mac)

---

## GUI Problems

### GUI doesn't start / crashes immediately

**Fixes**:
1. Check PyQt5 is installed: `pip install PyQt5`
2. Try running from terminal to see error: `python run.py --gui`
3. Update graphics drivers
4. Try: `QT_QPA_PLATFORM=offscreen python run.py --gui` (Linux)

### GUI is blurry on high-DPI display

**Fix**: Set environment variable before running:
```bash
# Windows
set QT_SCALE_FACTOR=1.5
python run.py --gui

# Linux/Mac
QT_SCALE_FACTOR=1.5 python run.py --gui
```

### Avatar not showing

**Fixes**:
1. Enable avatar in Settings → Avatar → Enable
2. Check you have avatar images in `data/avatar/`
3. Try a different avatar style

---

## Voice Problems

### "No speech output"

**Fixes**:
1. Install TTS: `pip install pyttsx3`
2. Check system audio is working
3. On Linux: `sudo apt-get install espeak`

### "Speech recognition not working"

**Fixes**:
1. Install speech recognition: `pip install SpeechRecognition pyaudio`
2. Check microphone permissions
3. Try VOSK (offline): `pip install vosk`

### PyAudio installation fails (Windows)

**Fix**: Use pre-built wheel:
```bash
pip install pipwin
pipwin install pyaudio
```

---

## Model Problems

### "Model file not found"

**Cause**: Model hasn't been trained yet.

**Fix**: Train a model first:
1. Go to Training tab
2. Select model size
3. Click "Start Training"

### "Tokenizer mismatch"

**Cause**: Model was trained with different tokenizer.

**Fixes**:
1. Retrain the model
2. Check `vocab_model/` folder exists with correct files

### Model outputs repeat forever

**Cause**: Repetition penalty too low or model undertrained.

**Fixes**:
1. Increase repetition penalty in Settings → Inference
2. Add more diverse training data
3. Train for more epochs

---

## Performance Tips

### Faster Training
- Use GPU if available
- Use `tiny` or `small` model for testing
- Reduce sequence length
- Use mixed precision (Settings → Training → Enable AMP)

### Faster Inference
- Enable KV-cache (default)
- Use smaller context window
- Try quantization: `pip install bitsandbytes`

### Lower Memory Usage
- Use smaller model
- Reduce batch size to 1
- Enable gradient checkpointing
- Use CPU instead of GPU (slower but uses system RAM)

---

## Getting Help

1. **Check logs**: Look in `logs/` folder for detailed error messages
2. **GitHub Issues**: [Report a bug](https://github.com/SirRDragonbornKnight/enigma_engine/issues)
3. **Debug mode**: Run with `--debug` flag for verbose output:
   ```bash
   python run.py --gui --debug
   ```

## Quick Diagnostic

Run this to check your setup:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5: OK')"
python -c "from enigma_engine.core.model import Forge; print('Enigma AI Engine: OK')"
```
