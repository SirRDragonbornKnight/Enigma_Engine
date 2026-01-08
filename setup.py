"""
Enigma Engine Setup Script

Install with: pip install -e .
Or for full install: pip install -e .[full]
"""
from setuptools import setup, find_packages

# Read requirements.txt for dependencies
def read_requirements():
    """Read base requirements from requirements.txt."""
    reqs = []
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and optional packages
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Remove inline comments and version specifiers for comparison
                    pkg = line.split(";")[0].strip()
                    if pkg:
                        reqs.append(pkg)
    except FileNotFoundError:
        pass
    return reqs

setup(
    name="enigma-engine",
    version="1.1.0",
    description="Personal AI Framework - Train and deploy your own AI with GUI, voice, vision, and more",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="SirRDragonbornKnight",
    author_email="sirknighth3@gmail.com",
    url="https://github.com/SirRDragonbornKnight/Enigma_Engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core (minimal - all offline-capable)
        "numpy>=1.21.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "full": [
            "torch>=1.12",
            "transformers>=4.20.0",
            "huggingface-hub>=0.20.0",
            "flask>=2.0.0",
            "flask-cors>=3.0.10",
            "flask-socketio>=5.0.0",
            "pyttsx3>=2.90",
            "SpeechRecognition>=3.8.0",
            "sounddevice>=0.4.0",
            "Pillow>=9.0.0",
            "mss>=6.1.0",
            "psutil>=5.9.0",
            "safetensors>=0.4.0",
            "opencv-python>=4.5.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-cors>=3.0.10",
            "flask-socketio>=5.0.0",
        ],
        "voice": [
            "pyttsx3>=2.90",
            "SpeechRecognition>=3.8.0",
            "sounddevice>=0.4.0",
            "vosk>=0.3.0",
        ],
        "vision": [
            "Pillow>=9.0.0",
            "mss>=6.1.0",
            "opencv-python>=4.5.0",
        ],
        "camera": [
            "opencv-python>=4.5.0",
        ],
        "huggingface": [
            "transformers>=4.20.0",
            "huggingface-hub>=0.20.0",
            "tokenizers>=0.13.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "enigma=run:main",
            "enigma-train=run:train",
            "enigma-serve=run:serve",
            "enigma-gui=run:gui",
        ],
    },
    include_package_data=True,
    package_data={
        "enigma": [
            "vocab_model/*",
            "data/*",
            "py.typed",
        ],
        "": [
            "*.txt",
            "*.md",
            "*.json",
        ],
    },
    data_files=[
        ("", ["README.md", "LICENSE", "requirements.txt"]),
        ("information", [
            "information/instructions.txt",
            "information/help.txt",
        ]),
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Graphics",
    ],
)
