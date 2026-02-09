#!/usr/bin/env python3
"""
================================================================================
Forge CLI - Simple Command-Line Interface
================================================================================

Ollama-style CLI for enigma_engine. Simple commands that just work.

COMMANDS:
    forge pull <model>      - Download a model
    forge run <model>       - Chat with a model
    forge serve             - Start OpenAI-compatible API server
    forge list              - List available models
    forge create <name>     - Create a model from a Modelfile
    forge rm <model>        - Remove a model
    forge show <model>      - Show model info
    forge train             - Train a model
    forge quantize          - Quantize a model

EXAMPLES:
    forge pull forge-small
    forge run forge-small
    forge serve --port 8000
    forge list

📍 FILE: enigma_engine/cli/main.py
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# =============================================================================
# CLI Commands
# =============================================================================

def cmd_pull(args):
    """Download a model."""
    model_name = args.model
    
    print(f"\nPulling model: {model_name}")
    print("-" * 40)
    
    # Check for HuggingFace model
    if "/" in model_name:
        # HuggingFace format: org/model
        print(f"Downloading from HuggingFace: {model_name}")
        try:
            from ..core.huggingface_loader import load_huggingface_model
            model, tokenizer = load_huggingface_model(model_name)
            
            # Save to models directory
            from ..config import CONFIG
            models_dir = Path(CONFIG.get("models_dir", "models"))
            models_dir.mkdir(exist_ok=True)
            
            safe_name = model_name.replace("/", "_")
            save_path = models_dir / f"{safe_name}.pth"
            
            import torch
            torch.save(model.state_dict(), save_path)
            
            print(f"\n[OK] Model saved to: {save_path}")
            return 0
            
        except Exception as e:
            print(f"\n[ERROR] Failed to download: {e}")
            return 1
    
    # Check for GGUF URL
    if model_name.endswith(".gguf") or "gguf" in model_name.lower():
        print(f"Downloading GGUF model: {model_name}")
        try:
            from ..core.model_registry import ModelRegistry
            registry = ModelRegistry()
            registry.pull_gguf(model_name)
            print(f"\n[OK] GGUF model downloaded")
            return 0
        except Exception as e:
            print(f"\n[ERROR] Failed to download GGUF: {e}")
            return 1
    
    # Built-in models
    builtin_models = {
        "forge-nano": "nano",
        "forge-micro": "micro",
        "forge-tiny": "tiny",
        "forge-small": "small",
        "forge-medium": "medium",
        "forge-large": "large",
        "forge": "small",  # Default
    }
    
    if model_name.lower() in builtin_models:
        size = builtin_models[model_name.lower()]
        print(f"Creating {size} model...")
        
        try:
            import torch

            from ..config import CONFIG
            from ..core.model import create_model
            
            model = create_model(size)
            
            models_dir = Path(CONFIG.get("models_dir", "models"))
            models_dir.mkdir(exist_ok=True)
            save_path = models_dir / f"{model_name.lower()}.pth"
            
            torch.save(model.state_dict(), save_path)
            
            print(f"\n[OK] Model created: {save_path}")
            print(f"    Size: {size}")
            print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return 0
            
        except Exception as e:
            print(f"\n[ERROR] Failed to create model: {e}")
            return 1
    
    print(f"\n[ERROR] Unknown model: {model_name}")
    print("\nAvailable built-in models:")
    for name in builtin_models:
        print(f"  - {name}")
    print("\nOr use HuggingFace format: organization/model-name")
    return 1


def cmd_run(args):
    """Chat with a model."""
    model_name = args.model
    
    print(f"\n{'='*50}")
    print(f"  Forge Chat - {model_name}")
    print(f"{'='*50}")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    try:
        from ..config import CONFIG
        from ..core.inference import EnigmaEngine

        # Find the model
        models_dir = Path(CONFIG.get("models_dir", "models"))
        model_path = None
        
        # Check various locations
        candidates = [
            models_dir / f"{model_name}.pth",
            models_dir / f"{model_name.lower()}.pth",
            models_dir / model_name,
            Path(model_name),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                break
        
        # Also check for GGUF
        if model_path is None:
            for candidate in [
                models_dir / f"{model_name}.gguf",
                models_dir / model_name,
            ]:
                if candidate.exists() and str(candidate).endswith(".gguf"):
                    # Use GGUF loader
                    from ..core.gguf_loader import GGUFModel
                    print(f"Loading GGUF model: {candidate}")
                    model = GGUFModel(str(candidate), n_gpu_layers=args.gpu_layers or 0)
                    model.load()
                    
                    # GGUF chat loop
                    while True:
                        try:
                            user_input = input("\nYou: ").strip()
                            if user_input.lower() in ['exit', 'quit', 'q']:
                                print("\nGoodbye!")
                                return 0
                            if not user_input:
                                continue
                            
                            print("\nAssistant: ", end="", flush=True)
                            response = model.generate(
                                user_input,
                                max_tokens=args.max_tokens or 256,
                                temperature=args.temperature or 0.7,
                                stream=True
                            )
                            print()
                            
                        except KeyboardInterrupt:
                            print("\n\nGoodbye!")
                            return 0
                    return 0
        
        # Load PyTorch model
        if model_path is None:
            # Try default model
            default_path = models_dir / "forge.pth"
            if default_path.exists():
                model_path = default_path
            else:
                print(f"\n[ERROR] Model not found: {model_name}")
                print(f"Run 'forge pull {model_name}' first.")
                return 1
        
        print(f"Loading model: {model_path}")
        engine = EnigmaEngine(model_path=str(model_path))
        
        # Chat loop
        conversation = []
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    return 0
                if not user_input:
                    continue
                
                print("\nAssistant: ", end="", flush=True)
                
                # Use streaming for better UX
                response_text = ""
                for token in engine.stream_generate(
                    user_input,
                    max_gen=args.max_tokens or 256,
                    temperature=args.temperature or 0.7
                ):
                    print(token, end="", flush=True)
                    response_text += token
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                return 0
        
    except Exception as e:
        print(f"\n[ERROR] Failed to run model: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_serve(args):
    """Start API server."""
    print(f"\nStarting Forge API server...")
    
    api_type = args.api or "openai"
    
    if api_type == "openai":
        from ..comms.openai_api import create_openai_server
        create_openai_server(
            host=args.host or "0.0.0.0",
            port=args.port or 8000,
            debug=args.debug
        )
    else:
        from ..comms.api_server import create_api_server
        create_api_server(
            host=args.host or "0.0.0.0",
            port=args.port or 5000,
            debug=args.debug
        )
    
    return 0


def cmd_list(args):
    """List available models."""
    from ..config import CONFIG
    
    models_dir = Path(CONFIG.get("models_dir", "models"))
    
    print(f"\n{'='*50}")
    print("  Available Models")
    print(f"{'='*50}\n")
    
    if not models_dir.exists():
        print("No models found. Run 'forge pull <model>' to download one.")
        return 0
    
    models = []
    
    # Find PyTorch models
    for model_file in models_dir.glob("*.pth"):
        size = model_file.stat().st_size / (1024 * 1024)  # MB
        models.append({
            "name": model_file.stem,
            "type": "PyTorch",
            "size": f"{size:.1f} MB",
            "path": str(model_file)
        })
    
    # Find GGUF models
    for model_file in models_dir.glob("*.gguf"):
        size = model_file.stat().st_size / (1024 * 1024 * 1024)  # GB
        models.append({
            "name": model_file.stem,
            "type": "GGUF",
            "size": f"{size:.2f} GB",
            "path": str(model_file)
        })
    
    if not models:
        print("No models found. Run 'forge pull <model>' to download one.\n")
        print("Available models to pull:")
        print("  forge pull forge-small    # ~27M params")
        print("  forge pull forge-medium   # ~85M params")
        print("  forge pull forge-large    # ~300M params")
        return 0
    
    # Print table
    print(f"{'NAME':<30} {'TYPE':<10} {'SIZE':<15}")
    print("-" * 55)
    for model in models:
        print(f"{model['name']:<30} {model['type']:<10} {model['size']:<15}")
    
    print(f"\nTotal: {len(models)} model(s)")
    return 0


def cmd_show(args):
    """Show model information."""
    model_name = args.model
    
    import torch

    from ..config import CONFIG
    
    models_dir = Path(CONFIG.get("models_dir", "models"))
    model_path = models_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        # Try GGUF
        gguf_path = models_dir / f"{model_name}.gguf"
        if gguf_path.exists():
            print(f"\n{'='*50}")
            print(f"  Model: {model_name}")
            print(f"{'='*50}\n")
            print(f"Type: GGUF")
            print(f"Path: {gguf_path}")
            print(f"Size: {gguf_path.stat().st_size / (1024*1024*1024):.2f} GB")
            return 0
        
        print(f"\n[ERROR] Model not found: {model_name}")
        return 1
    
    print(f"\n{'='*50}")
    print(f"  Model: {model_name}")
    print(f"{'='*50}\n")
    
    # Load and inspect
    # Note: weights_only=False needed for full checkpoint inspection, only inspect trusted models
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}
    
    # Calculate parameters
    total_params = sum(t.numel() for t in state_dict.values())
    
    print(f"Type: PyTorch")
    print(f"Path: {model_path}")
    print(f"Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Parameters: {total_params:,}")
    
    if config:
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    return 0


def cmd_rm(args):
    """Remove a model."""
    model_name = args.model
    
    from ..config import CONFIG
    
    models_dir = Path(CONFIG.get("models_dir", "models"))
    
    # Try different extensions
    removed = False
    for ext in [".pth", ".gguf", ""]:
        model_path = models_dir / f"{model_name}{ext}"
        if model_path.exists():
            if not args.force:
                confirm = input(f"Remove {model_path}? [y/N]: ")
                if confirm.lower() != 'y':
                    print("Cancelled.")
                    return 0
            
            model_path.unlink()
            print(f"[OK] Removed: {model_path}")
            removed = True
            break
    
    if not removed:
        print(f"[ERROR] Model not found: {model_name}")
        return 1
    
    return 0


def cmd_train(args):
    """Train a model."""
    print(f"\n{'='*50}")
    print("  Forge Training")
    print(f"{'='*50}\n")
    
    from ..config import CONFIG
    from ..core.training import train_model
    
    data_path = args.data or Path(CONFIG.get("data_dir", "data")) / "training.txt"
    model_size = args.size or "small"
    epochs = args.epochs or 30
    output = args.output
    
    print(f"Model size: {model_size}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print()
    
    results = train_model(
        data_path=str(data_path),
        epochs=epochs,
        model_size=model_size,
        output_path=output,
        train_tokenizer_first=True,
        force=args.force
    )
    
    print(f"\n[OK] Training complete!")
    print(f"Final loss: {results.get('final_loss', 'N/A')}")
    
    return 0


def cmd_quantize(args):
    """Quantize a model."""
    model_name = args.model
    bits = args.bits or 8
    
    print(f"\n{'='*50}")
    print(f"  Quantizing: {model_name}")
    print(f"{'='*50}\n")
    
    import torch

    from ..config import CONFIG
    from ..core.quantization import QuantConfig, quantize_model
    
    models_dir = Path(CONFIG.get("models_dir", "models"))
    model_path = models_dir / f"{model_name}.pth"
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_name}")
        return 1
    
    print(f"Loading model...")
    from ..core.model import create_model

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {"size": "small"}
    
    # Create model and load weights
    model = create_model(config.get("size", "small"))
    model.load_state_dict(state_dict)
    
    # Quantize
    print(f"Quantizing to {bits}-bit...")
    quant_config = QuantConfig(bits=bits)
    quantized_model = quantize_model(model, config=quant_config)
    
    # Save
    output_path = models_dir / f"{model_name}_q{bits}.pth"
    torch.save({
        "model_state_dict": quantized_model.state_dict(),
        "config": config,
        "quantization": {"bits": bits}
    }, output_path)
    
    # Compare sizes
    orig_size = model_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n[OK] Quantized model saved: {output_path}")
    print(f"Original size: {orig_size:.1f} MB")
    print(f"Quantized size: {new_size:.1f} MB")
    print(f"Reduction: {(1 - new_size/orig_size) * 100:.1f}%")
    
    return 0


def cmd_create(args):
    """Create a model from a Modelfile."""
    name = args.name
    modelfile = args.file or "Modelfile"
    
    print(f"\nCreating model: {name}")
    
    if not Path(modelfile).exists():
        print(f"[ERROR] Modelfile not found: {modelfile}")
        print("\nCreate a Modelfile with:")
        print("  FROM forge-small")
        print("  SYSTEM You are a helpful assistant.")
        return 1
    
    # Parse Modelfile
    config = {"base": "forge-small", "system": ""}
    
    with open(modelfile) as f:
        for line in f:
            line = line.strip()
            if line.startswith("FROM "):
                config["base"] = line[5:].strip()
            elif line.startswith("SYSTEM "):
                config["system"] = line[7:].strip()
            elif line.startswith("PARAMETER "):
                parts = line[10:].split(" ", 1)
                if len(parts) == 2:
                    config[parts[0]] = parts[1]
    
    print(f"Base model: {config['base']}")
    
    # Copy base model with new config
    import json

    import torch

    from ..config import CONFIG
    
    models_dir = Path(CONFIG.get("models_dir", "models"))
    base_path = models_dir / f"{config['base']}.pth"
    
    if not base_path.exists():
        print(f"\n[ERROR] Base model not found: {config['base']}")
        print("Run 'forge pull {config['base']}' first.")
        return 1
    
    # Load base model
    checkpoint = torch.load(base_path, map_location="cpu")
    
    # Add custom config
    if isinstance(checkpoint, dict):
        checkpoint["custom_config"] = config
    else:
        checkpoint = {"model_state_dict": checkpoint, "custom_config": config}
    
    # Save new model
    output_path = models_dir / f"{name}.pth"
    torch.save(checkpoint, output_path)
    
    print(f"\n[OK] Model created: {output_path}")
    
    return 0


# =============================================================================
# Main CLI Entry Point
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="enigma_engine - Build and run AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  forge pull forge-small           Download a model
  forge run forge-small            Chat with a model  
  forge serve                      Start OpenAI-compatible API
  forge list                       List available models
  forge train --data data.txt      Train a model
  forge quantize mymodel --bits 4  Quantize a model
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # pull command
    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model name or HuggingFace ID")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Chat with a model")
    run_parser.add_argument("model", nargs="?", default="forge", help="Model name")
    run_parser.add_argument("--temperature", "-t", type=float, help="Sampling temperature")
    run_parser.add_argument("--max-tokens", "-n", type=int, help="Max tokens to generate")
    run_parser.add_argument("--gpu-layers", "-g", type=int, help="GPU layers (GGUF)")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port")
    serve_parser.add_argument("--api", choices=["openai", "simple"], default="openai", 
                              help="API type (default: openai)")
    serve_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # show command
    show_parser = subparsers.add_parser("show", help="Show model info")
    show_parser.add_argument("model", help="Model name")
    
    # rm command
    rm_parser = subparsers.add_parser("rm", help="Remove a model")
    rm_parser.add_argument("model", help="Model name")
    rm_parser.add_argument("-f", "--force", action="store_true", help="Force removal")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", "-d", help="Training data file")
    train_parser.add_argument("--size", "-s", default="small", 
                              choices=["nano", "micro", "tiny", "small", "medium", "large"])
    train_parser.add_argument("--epochs", "-e", type=int, default=30, help="Epochs")
    train_parser.add_argument("--output", "-o", help="Output path")
    train_parser.add_argument("--force", action="store_true", help="Force retrain")
    
    # quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quant_parser.add_argument("model", help="Model name")
    quant_parser.add_argument("--bits", "-b", type=int, default=8, choices=[4, 8],
                              help="Quantization bits (default: 8)")
    
    # create command
    create_parser = subparsers.add_parser("create", help="Create model from Modelfile")
    create_parser.add_argument("name", help="Model name")
    create_parser.add_argument("-f", "--file", default="Modelfile", help="Modelfile path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    commands = {
        "pull": cmd_pull,
        "run": cmd_run,
        "serve": cmd_serve,
        "list": cmd_list,
        "show": cmd_show,
        "rm": cmd_rm,
        "train": cmd_train,
        "quantize": cmd_quantize,
        "create": cmd_create,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
