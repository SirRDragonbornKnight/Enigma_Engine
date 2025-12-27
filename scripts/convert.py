#!/usr/bin/env python3
"""
Model conversion and export utilities.

Usage:
    python -m scripts.convert --model my_model --to onnx
    python -m scripts.convert --model my_model --grow large
    python -m scripts.convert --model big_model --shrink tiny --output small_model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Convert and export Enigma models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Source model name"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output model name (for grow/shrink)"
    )
    
    # Conversion options (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--to", type=str, choices=["onnx", "torchscript"],
        help="Export to format"
    )
    action_group.add_argument(
        "--grow", type=str, choices=["small", "medium", "large", "xl", "xxl", "xxxl"],
        help="Grow model to larger size"
    )
    action_group.add_argument(
        "--shrink", type=str, choices=["tiny", "small", "medium", "large", "xl", "xxl"],
        help="Shrink model to smaller size"
    )
    action_group.add_argument(
        "--info", action="store_true",
        help="Show model info"
    )
    
    args = parser.parse_args()
    
    from enigma.core.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    
    if args.model not in registry.list_models():
        print(f"Error: Model '{args.model}' not found")
        print(f"Available models: {registry.list_models()}")
        sys.exit(1)
    
    if args.info:
        info = registry.get_model_info(args.model)
        print(f"Model: {args.model}")
        print(f"  Size: {info.get('size', 'unknown')}")
        print(f"  Created: {info.get('created', 'unknown')}")
        print(f"  Status: {info.get('status', 'unknown')}")
        return
    
    if args.grow:
        from enigma.core.model_scaling import grow_model
        
        output_name = args.output or f"{args.model}_{args.grow}"
        print(f"Growing {args.model} to {args.grow} as {output_name}...")
        
        # Load source model
        model = registry.load_model(args.model)
        
        # Grow
        vocab_size = model.vocab_size
        new_model = grow_model(model, args.grow, vocab_size)
        
        # Save
        registry.create_model(output_name, size=args.grow)
        registry.save_model(output_name, new_model)
        
        print(f"Created {output_name}")
    
    elif args.shrink:
        from enigma.core.model_scaling import shrink_model
        
        output_name = args.output or f"{args.model}_{args.shrink}"
        print(f"Shrinking {args.model} to {args.shrink} as {output_name}...")
        
        model = registry.load_model(args.model)
        vocab_size = model.vocab_size
        new_model = shrink_model(model, args.shrink, vocab_size)
        
        registry.create_model(output_name, size=args.shrink)
        registry.save_model(output_name, new_model)
        
        print(f"Created {output_name}")
    
    elif args.to == "onnx":
        print("ONNX export not yet implemented")
    
    elif args.to == "torchscript":
        import torch
        
        model = registry.load_model(args.model)
        model.eval()
        
        # Trace
        dummy_input = torch.randint(0, 1000, (1, 10))
        traced = torch.jit.trace(model, dummy_input)
        
        output_path = f"{args.model}.pt"
        traced.save(output_path)
        print(f"Saved TorchScript model to {output_path}")


if __name__ == "__main__":
    main()
