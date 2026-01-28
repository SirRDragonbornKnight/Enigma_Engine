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
    
    from forge_ai.core.model_registry import ModelRegistry
    
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
        from forge_ai.core.model_scaling import grow_model
        
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
        from forge_ai.core.model_scaling import shrink_model
        
        output_name = args.output or f"{args.model}_{args.shrink}"
        print(f"Shrinking {args.model} to {args.shrink} as {output_name}...")
        
        model = registry.load_model(args.model)
        vocab_size = model.vocab_size
        new_model = shrink_model(model, args.shrink, vocab_size)
        
        registry.create_model(output_name, size=args.shrink)
        registry.save_model(output_name, new_model)
        
        print(f"Created {output_name}")
    
    elif args.to == "onnx":
        import torch
        
        try:
            import onnx
        except ImportError:
            print("Error: ONNX not installed. Run: pip install onnx")
            sys.exit(1)
        
        print(f"Exporting {args.model} to ONNX format...")
        
        model = registry.load_model(args.model)
        model.eval()
        
        # Get model info for dynamic axes
        vocab_size = model.vocab_size
        
        # Create dummy input (batch_size=1, seq_len=10)
        dummy_input = torch.randint(0, vocab_size, (1, 10))
        
        output_path = args.output or f"{args.model}.onnx"
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                verbose=False
            )
            
            # Validate the exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            print(f"Saved ONNX model to {output_path}")
            print(f"Model inputs: {[inp.name for inp in onnx_model.graph.input]}")
            print(f"Model outputs: {[out.name for out in onnx_model.graph.output]}")
            
            # Try to simplify if onnx-simplifier is available
            try:
                import onnxsim
                simplified_model, check = onnxsim.simplify(onnx_model)
                if check:
                    simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                    onnx.save(simplified_model, simplified_path)
                    print(f"Saved simplified model to {simplified_path}")
            except ImportError:
                pass
            
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("Tip: Some model operations may not be supported by ONNX.")
            print("     Try using TorchScript export instead.")
            sys.exit(1)
    
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
