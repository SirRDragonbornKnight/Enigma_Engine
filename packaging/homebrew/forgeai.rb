# Homebrew Formula for ForgeAI
# Install with: brew install forgeai

class Forgeai < Formula
  include Language::Python::Virtualenv

  desc "Fully modular AI framework for local and cloud AI"
  homepage "https://github.com/forgeai/forge_ai"
  url "https://github.com/forgeai/forge_ai/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256_HASH"
  license "MIT"
  head "https://github.com/forgeai/forge_ai.git", branch: "main"

  depends_on "python@3.11"
  depends_on "cmake" => :build
  depends_on "rust" => :build

  # Optional dependencies
  depends_on "portaudio" => :optional  # For voice features
  depends_on "ffmpeg" => :optional     # For video/audio processing

  resource "torch" do
    url "https://files.pythonhosted.org/packages/torch-2.1.0.tar.gz"
    sha256 "PLACEHOLDER_TORCH_SHA256"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/numpy-1.26.0.tar.gz"
    sha256 "PLACEHOLDER_NUMPY_SHA256"
  end

  resource "PyQt5" do
    url "https://files.pythonhosted.org/packages/PyQt5-5.15.9.tar.gz"
    sha256 "PLACEHOLDER_PYQT5_SHA256"
  end

  def install
    virtualenv_install_with_resources

    # Install the package
    system Formula["python@3.11"].opt_bin/"python3", "-m", "pip", "install", "."

    # Install shell completion
    bash_completion.install "completion/forgeai.bash" => "forgeai"
    zsh_completion.install "completion/forgeai.zsh" => "_forgeai"

    # Create wrapper scripts
    (bin/"forgeai").write <<~EOS
      #!/bin/bash
      exec "#{libexec}/bin/python" -m forge_ai.run "$@"
    EOS

    (bin/"forgeai-gui").write <<~EOS
      #!/bin/bash
      exec "#{libexec}/bin/python" -m forge_ai.run --gui "$@"
    EOS
  end

  def caveats
    <<~EOS
      ForgeAI has been installed!

      To start the GUI:
        forgeai-gui

      To start the CLI:
        forgeai

      For GPU acceleration, install PyTorch with CUDA/MPS support:
        pip install torch torchvision torchaudio

      Configuration is stored in:
        ~/.config/forgeai/

      Models are stored in:
        ~/.local/share/forgeai/models/
    EOS
  end

  test do
    system "#{bin}/forgeai", "--version"
  end
end
