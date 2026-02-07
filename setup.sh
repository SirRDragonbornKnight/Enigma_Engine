#!/bin/bash
# Enigma AI Engine Quick Setup Script
# Works on Linux, macOS, and WSL

set -e

echo "========================================"
echo "  Enigma AI Engine Quick Setup"
echo "========================================"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    # Check for Raspberry Pi
    if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
        OS="pi"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate virtual environment
if [[ "$OS" == "windows" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install system dependencies (Linux/Pi only)
if [[ "$OS" == "linux" ]] || [[ "$OS" == "pi" ]]; then
    echo ""
    echo "You may need system packages. Run these if needed:"
    echo "  sudo apt update"
    echo "  sudo apt install python3-pyqt5 portaudio19-dev libespeak-ng1 espeak-ng"
    echo ""
fi

# Run the installer
echo "Running Enigma AI Engine installer..."
$PYTHON install.py --standard

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
if [[ "$OS" == "windows" ]]; then
    echo "  venv\\Scripts\\activate"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "To run Enigma AI Engine:"
echo "  python run.py --gui    # GUI mode"
echo "  python run.py --run    # CLI mode"
echo "  python run.py --train  # Training mode"
echo ""
