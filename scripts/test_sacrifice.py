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

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from enigma.core.model import Enigma, create_model, MODEL_PRESETS
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer
from enigma.core.inference import EnigmaEngine

# Paths
MODELS_DIR = Path(__file__).parent.parent / 'models'
VOCAB_DIR = Path(__file__).parent.parent / 'enigma' / 'vocab_model'


def load_sacrifice_model(model_dir: Path = None):
    """Load the sacrifice model and tokenizer."""
    if model_dir is None:
        model_dir = MODELS_DIR / 'sacrifice'
    
    model_dir = Path(model_dir)
    
    print("Loading model...")
    
    # Load tokenizer
    tokenizer_path = model_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        tokenizer_path = VOCAB_DIR / 'bpe_vocab.json'
    
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
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = True,
) -> str:
    """Generate a response from the model."""
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
            # Streaming generation
            print("AI: ", end="", flush=True)
            
            generated = input_ids
            response_started = False
            
            for _ in range(max_tokens):
                # Get logits for next token
                logits = model(generated)
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                top_k = 50
                values, indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)
                
                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Decode token
                token_text = tokenizer.decode([next_token[0, 0].item()])
                
                # Skip until we're past "A:"
                full_text = tokenizer.decode(generated[0].tolist())
                if "A:" in full_text and not response_started:
                    response_started = True
                    # Print what comes after A:
                    response_part = full_text.split("A:", 1)[-1]
                    print(response_part, end="", flush=True)
                elif response_started:
                    print(token_text, end="", flush=True)
                
                # Check for end
                if next_token[0, 0].item() == tokenizer.eos_token_id:
                    break
                if "\nQ:" in full_text.split("A:", 1)[-1] if "A:" in full_text else False:
                    break
            
            print()  # Newline at end
            
            return tokenizer.decode(generated[0].tolist())
        else:
            # Non-streaming generation
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
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
    parser = argparse.ArgumentParser(description="Test Sacrifice Model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model directory")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--no-stream", action="store_true",
                       help="Disable streaming output")
    args = parser.parse_args()
    
    # Load model
    model_dir = Path(args.model) if args.model else None
    model, tokenizer, device = load_sacrifice_model(model_dir)
    
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
                print("  temp <value> - Set temperature (current: {:.1f})".format(args.temperature))
                print()
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    args.temperature = float(user_input.split()[1])
                    print(f"Temperature set to {args.temperature}")
                except:
                    print("Invalid temperature value")
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


if __name__ == "__main__":
    main()
