"""
Test the trained sacrifice model interactively.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from enigma.core.advanced_tokenizer import AdvancedBPETokenizer
from enigma.core.advanced_model import EnigmaModel, EnigmaConfig

# Paths
MODELS_DIR = Path(__file__).parent.parent / 'models'
VOCAB_DIR = Path(__file__).parent.parent / 'enigma' / 'vocab_model'


def main():
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AdvancedBPETokenizer()
    tokenizer.load(VOCAB_DIR / 'advanced_bpe')
    print(f"Loaded tokenizer with {tokenizer.vocab_size} tokens")
    
    # Load model
    model_dir = MODELS_DIR / 'sacrifice'
    checkpoint = torch.load(model_dir / 'model.pt', map_location='cpu')
    
    config = EnigmaConfig(**checkpoint['config'])
    model = EnigmaModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model with {model.count_parameters():,} parameters")
    print(f"Device: {device}")
    print()
    print("=" * 50)
    print("SACRIFICE MODEL - Interactive Test")
    print("Type 'quit' to exit")
    print("=" * 50)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Format as question
            prompt = f"Q: {user_input}\nA:"
            
            # Encode
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
            
            # Decode
            full_response = tokenizer.decode(output_ids[0].tolist())
            
            # Extract just the answer
            if "A:" in full_response:
                # Find the first A: after Q:
                parts = full_response.split("A:", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Cut at next Q: if present
                    if "Q:" in answer:
                        answer = answer.split("Q:")[0].strip()
                    print(f"AI: {answer}")
                else:
                    print(f"AI: {full_response}")
            else:
                print(f"AI: {full_response}")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == '__main__':
    main()
