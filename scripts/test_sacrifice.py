"""
Test the Sacrifice Model Interactively
======================================

Test the trained sacrifice model with an interactive chat interface.

Usage:
    python scripts/test_sacrifice.py
    python scripts/test_sacrifice.py --model models/sacrifice
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from enigma.core.model import Enigma, create_model, MODEL_PRESETS
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer
from enigma.core.inference import EnigmaEngine

# Constants
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9

# Paths
MODELS_DIR = Path(__file__).parent.parent / 'models'
VOCAB_DIR = Path(__file__).parent.parent / 'enigma' / 'vocab_model'


def load_sacrifice_model(model_dir: Path = None) -> Tuple[Enigma, AdvancedBPETokenizer, str]:
    """Load the sacrifice model and tokenizer.
    
    Args:
        model_dir: Path to model directory (default: models/sacrifice)
    
    Returns:
        tuple: (model, tokenizer, device)
    
    Raises:
        FileNotFoundError: If model files are not found
    """
    if model_dir is None:
        model_dir = MODELS_DIR / 'sacrifice'
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer_path = model_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = VOCAB_DIR / 'bpe_vocab.json'
    
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
    
    tokenizer = AdvancedBPETokenizer(vocab_file=tokenizer_path)
    print(f"  Tokenizer: {tokenizer.vocab_size:,} tokens")
    
    # Load model
    model_path = model_dir / 'model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load state dict to infer model size
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Infer hidden dimension from state dict
    hidden_dim = None
    for key, tensor in state_dict.items():
        if 'embed' in key.lower() and tensor.dim() == 2:
            hidden_dim = tensor.shape[1]
            break
    
    if hidden_dim is None:
        raise ValueError("Could not infer model dimensions from state dict")
    
    # Find matching preset
    model_size = "small"
    for name, preset in MODEL_PRESETS.items():
        preset_dim = preset.dim if hasattr(preset, 'dim') else preset.get('hidden_dim', 512)
        if preset_dim == hidden_dim:
            model_size = name
            break
    
    # Create model
    model = create_model(model_size, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(state_dict)
    
    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {num_params:,} parameters")
    print(f"  Size: {model_size}")
    print(f"  Device: {device}")
    
    return model, tokenizer, device


def generate_response(
    model: Enigma,
    tokenizer: AdvancedBPETokenizer,
    prompt: str,
    device: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = True,
) -> str:
    """Generate a response from the model.
    
    Args:
        model: The Enigma model
        tokenizer: Tokenizer for encoding/decoding
        prompt: User input prompt
        device: Device to run on ('cuda' or 'cpu')
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stream: Whether to stream output
    
    Returns:
        str: Generated response
    """
    # Format as Q&A
    formatted = f"Q: {prompt}\nA:"
    
    # Encode
    input_ids = torch.tensor(
        [tokenizer.encode(formatted)], 
        dtype=torch.long, 
        device=device
    )
    
    with torch.no_grad():
        if stream:
            # Streaming generation - print as we generate
            print("AI: ", end="", flush=True)
            
            generated = input_ids
            response_started = False
            response_text = ""
            
            for _ in range(max_tokens):
                # Get logits for next token
                logits = model(generated)
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                values, indices = torch.topk(next_logits, DEFAULT_TOP_K)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)
                
                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Decode and print token
                full_text = tokenizer.decode(generated[0].tolist())
                
                # Check if we've reached the answer section
                if "A:" in full_text and not response_started:
                    response_started = True
                    response_text = full_text.split("A:", 1)[-1]
                    print(response_text, end="", flush=True)
                elif response_started:
                    new_response = full_text.split("A:", 1)[-1]
                    # Only print the new part
                    new_part = new_response[len(response_text):]
                    print(new_part, end="", flush=True)
                    response_text = new_response
                
                # Check for end conditions
                if next_token[0, 0].item() == tokenizer.eos_token_id:
                    break
                if "\nQ:" in response_text:
                    break
            
            print()  # Newline at end
            
            # Clean up response
            if "\nQ:" in response_text:
                response_text = response_text.split("\nQ:")[0]
            
            return response_text.strip()
        else:
            # Non-streaming generation
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=DEFAULT_TOP_K,
                top_p=DEFAULT_TOP_P,
            )
            
            full_response = tokenizer.decode(output_ids[0].tolist())
            
            # Extract answer
            if "A:" in full_response:
                answer = full_response.split("A:", 1)[-1]
                # Clean up
                if "\nQ:" in answer:
                    answer = answer.split("\nQ:")[0]
                return answer.strip()
            
            return full_response


def main():
    """Main entry point for interactive testing."""
    parser = argparse.ArgumentParser(description="Test Sacrifice Model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model directory")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true",
                       help="Disable streaming output")
    args = parser.parse_args()
    
    # Load model
    try:
        model_dir = Path(args.model) if args.model else None
        model, tokenizer, device = load_sacrifice_model(model_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train a model first using: python scripts/build_sacrifice.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("SACRIFICE MODEL - Interactive Test")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  quit, exit, q - Exit the program")
                print("  help - Show this message")
                print(f"  temp <value> - Set temperature (current: {args.temperature:.1f})")
                print()
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    args.temperature = float(user_input.split()[1])
                    print(f"Temperature set to {args.temperature}")
                except (IndexError, ValueError):
                    print("Invalid temperature value. Usage: temp 0.7")
                continue
            
            if not user_input:
                continue
            
            # Generate response
            generate_response(
                model, tokenizer, user_input, device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=not args.no_stream,
            )
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
