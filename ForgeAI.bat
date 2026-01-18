@echo off
:: ForgeAI Launcher for Windows
:: Double-click this to start ForgeAI

title ForgeAI
echo.
echo  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗     █████╗ ██╗
echo  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝    ██╔══██╗██║
echo  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗      ███████║██║
echo  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝      ██╔══██║██║
echo  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗    ██║  ██║██║
echo  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝
echo.
echo  Starting ForgeAI...
echo.

:: Change to script directory
cd /d "%~dp0"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Check if dependencies are installed (quick check for torch)
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies... This may take a few minutes.
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: Launch ForgeAI GUI
echo [INFO] Launching ForgeAI...
python run.py --gui

:: If we get here, the app closed
if errorlevel 1 (
    echo.
    echo [ERROR] ForgeAI encountered an error.
    pause
)
