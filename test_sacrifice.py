"""Quick test of sacrifice model"""
import torch
import torch.nn.functional as F
from enigma.core.model_registry import ModelRegistry
from enigma.core.tokenizer import load_tokenizer

# Constants for sampling
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 50
MAX_GENERATION_TOKENS = 100

# Load model and tokenizer
registry = ModelRegistry()
model, config = registry.load_model('sacrifice_medium')
tokenizer = load_tokenizer()
model.eval()

# Get device from model
device = next(model.parameters()).device

def sample_with_temperature(logits, temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K):
    """Sample from logits with temperature and top-k filtering.
    
    Args:
        logits: Logits tensor for next token prediction
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider (0 = disabled)
    
    Returns:
        int: Selected token ID
    """
    # Apply temperature scaling
    logits = logits / temperature
    
    # Apply top-k filtering to focus on most likely tokens
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        # Mask out tokens below top-k threshold
        logits = torch.where(
            logits < values[-1],
            torch.full_like(logits, float('-inf')),
            logits
        )
    
    # Sample from probability distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token

def generate_answer(prompt_text, max_tokens=MAX_GENERATION_TOKENS):
    """Generate answer for a given prompt.
    
    Args:
        prompt_text: The input question text
        max_tokens: Maximum number of tokens to generate
    
    Returns:
        str: Generated answer text
    """
    # Format as Q&A prompt
    prompt = f"Q: {prompt_text}\nA:"
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)
    
    # Generate response
    with torch.no_grad():
        for _ in range(max_tokens):
            output = model(input_tensor)
            logits = output[0, -1, :]
            next_token = sample_with_temperature(logits)
            
            # Append new token
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            
            # Check stopping conditions
            decoded_char = tokenizer.decode([next_token])
            if '\n' in decoded_char or next_token == tokenizer.eos_token_id:
                break
    
    # Decode and extract answer
    full_response = tokenizer.decode(input_tensor[0].tolist())
    
    if 'A:' in full_response:
        answer = full_response.split('A:')[-1].strip()
        # Remove trailing Q: if it started generating next question
        if 'Q:' in answer:
            answer = answer.split('Q:')[0].strip()
    else:
        answer = full_response
    
    return answer

# Test cases
test_questions = ['Hello', 'What is your name?', 'How are you?']

print("Testing Sacrifice Model")
print("=" * 50)

for question in test_questions:
    answer = generate_answer(question)
    print(f'Q: {question}')
    print(f'A: {answer}')
    print()
