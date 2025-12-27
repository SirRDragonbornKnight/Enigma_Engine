#!/usr/bin/env python3
"""
Generate text with an Enigma model from command line.

Usage:
    python -m scripts.generate --prompt "Hello, world"
    python -m scripts.generate --model my_model --prompt "Once upon a time"
    echo "Hello" | python -m scripts.generate --stdin
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with an Enigma model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--prompt", "-p", type=str, default=None,
        help="Prompt text to continue"
    )
    input_group.add_argument(
        "--stdin", action="store_true",
        help="Read prompt from stdin"
    )
    input_group.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive mode (chat)"
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model", "-m", type=str, default=None,
        help="Model name (uses default if not specified)"
    )
    model_group.add_argument(
        "--device", type=str, default=None,
        help="Device to use"
    )
    
    # Generation arguments
    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument(
        "--max-tokens", "-n", type=int, default=100,
        help="Maximum tokens to generate"
    )
    gen_group.add_argument(
        "--temperature", "-t", type=float, default=0.8,
        help="Sampling temperature"
    )
    gen_group.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling"
    )
    gen_group.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p (nucleus) sampling"
    )
    gen_group.add_argument(
        "--repetition-penalty", type=float, default=1.1,
        help="Repetition penalty"
    )
    
    # Output
    parser.add_argument(
        "--stream", "-s", action="store_true",
        help="Stream output token by token"
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="Don't include prompt in output"
    )
    
    args = parser.parse_args()
    
    # Get prompt
    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    elif args.interactive:
        prompt = None  # Will prompt later
    else:
        parser.error("Please provide --prompt, --stdin, or --interactive")
        return
    
    # Import after parsing
    from enigma.core.inference import EnigmaEngine
    
    # Create engine
    engine = EnigmaEngine(device=args.device)
    
    if args.interactive:
        print("Enigma Interactive Mode (type 'quit' to exit)")
        print("-" * 50)
        history = []
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue
                
                response = engine.chat(
                    user_input,
                    history=history,
                    max_gen=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                
                print(f"Assistant: {response}")
                print()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    elif args.stream:
        for token in engine.stream_generate(
            prompt,
            max_gen=args.max_tokens,
            temperature=args.temperature,
        ):
            print(token, end="", flush=True)
        print()
    else:
        output = engine.generate(
            prompt,
            max_gen=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        
        if args.no_prompt:
            output = output[len(prompt):]
        
        print(output)


if __name__ == "__main__":
    main()
