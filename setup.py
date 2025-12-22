"""
Enigma Engine Setup Script

Install with: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="enigma-engine",
    version="1.0.0",
    description="Personal AI Framework - Train and deploy your own AI",
    author="SirRDragonbornKnight",
    author_email="sirknighth3@gmail.com",
    url="https://github.com/SirRDragonbornKnight/enigma_engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core (all offline-capable)
        "numpy",
        
        # Optional but recommended
        # "torch",  # Install separately based on platform
        # "PyQt5",  # Use system package on Pi: sudo apt install python3-pyqt5
    ],
    extras_require={
        "full": [
            "torch",
            "flask",
            "flask-cors",
            "pyttsx3",
            "SpeechRecognition",
            "PyAudio",
            "Pillow",
            "mss",
        ],
        "web": [
            "flask",
            "flask-cors",
            "flask-socketio",
        ],
        "voice": [
            "pyttsx3",
            "SpeechRecognition",
            "PyAudio",
        ],
        "vision": [
            "Pillow",
            "mss",
            "easyocr",  # Optional: better OCR
        ],
    },
    entry_points={
        "console_scripts": [
            "enigma=run:main",
            "enigma-train=run:train",
            "enigma-serve=run:serve",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
