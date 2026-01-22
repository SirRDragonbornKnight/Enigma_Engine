#!/bin/bash
# ForgeAI Launcher for Linux/macOS
# Run with: bash ForgeAI.sh or ./ForgeAI.sh

echo ""
echo "  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗     █████╗ ██╗"
echo "  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝    ██╔══██╗██║"
echo "  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗      ███████║██║"
echo "  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝      ██╔══██║██║"
echo "  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗    ██║  ██║██║"
echo "  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝"
echo ""
echo "  Starting ForgeAI..."
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python is not installed."
        echo ""
        echo "Install Python 3.9+ using your package manager:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "  Fedora: sudo dnf install python3 python3-pip"
        echo "  Arch: sudo pacman -S python python-pip"
        echo "  macOS: brew install python3"
        exit 1
    fi
    PYTHON=python
else
    PYTHON=python3
fi

echo "[INFO] Using Python: $($PYTHON --version)"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if dependencies are installed
if ! $PYTHON -c "import torch" &> /dev/null; then
    echo "[INFO] Installing dependencies... This may take a few minutes."
    echo ""
    $PYTHON -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi
fi

# Check for display (for GUI)
if [ -z "$DISPLAY" ] && [ "$(uname)" != "Darwin" ]; then
    echo "[WARNING] No display detected. Starting in CLI mode..."
    $PYTHON run.py --run
else
    echo "[INFO] Launching ForgeAI GUI..."
    $PYTHON run.py --gui
fi
