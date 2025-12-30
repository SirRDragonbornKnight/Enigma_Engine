"""Quick test of sacrifice model"""
import torch
import torch.nn.functional as F
from enigma.core.model_registry import ModelRegistry
from enigma.core.tokenizer import load_tokenizer

registry = ModelRegistry()
model, config = registry.load_model('sacrifice_medium')
tokenizer = load_tokenizer()
model.eval()

# Get device from model
device = next(model.parameters()).device

def sample_with_temperature(logits, temperature=0.8, top_k=40):
    """Sample from logits with temperature and top-k."""
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1).item()
    return next_token

tests = ['Hello', 'What is your name?', 'How are you?']
for user_input in tests:
    # Format as Q&A prompt like training data
    prompt = f"Q: {user_input}\nA:"
    
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    with torch.no_grad():
        for _ in range(100):
            output = model(input_tensor)
            logits = output[0, -1, :]
            next_token = sample_with_temperature(logits, temperature=0.7, top_k=50)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]]).to(device)], dim=1)
            # Stop at newline (end of answer)
            decoded_char = tokenizer.decode([next_token])
            if '\n' in decoded_char or next_token == tokenizer.eos_token_id:
                break
    
    # Decode full response and extract the answer
    full_response = tokenizer.decode(input_tensor[0].tolist())
    # Extract just the answer part
    if 'A:' in full_response:
        answer = full_response.split('A:')[-1].strip()
        # Remove any trailing Q: that might have started
        if 'Q:' in answer:
            answer = answer.split('Q:')[0].strip()
    else:
        answer = full_response
    
    print(f'Q: {user_input}')
    print(f'A: {answer}')
    print()
