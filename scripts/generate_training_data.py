"""
Training Data Generator using Large HuggingFace Models

Uses a large AI (Mistral-7B, Llama-3-8B, etc.) to generate high-quality
training data for smaller Enigma models.

Usage:
    python scripts/generate_training_data.py --topic "general knowledge" --count 100
    python scripts/generate_training_data.py --topics-file topics.txt --count 50
    
Requirements:
    pip install transformers accelerate bitsandbytes
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Recommended models for training data generation
# Sorted by quality (best first), with VRAM requirements
RECOMMENDED_MODELS = {
    # Large - needs 32GB+ VRAM (4-bit quantized)
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",    # 70B - Best quality

    # Medium - needs 16-20GB VRAM (4-bit quantized)
    "deepseek-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # 32B - Excellent reasoning

    # Small-Medium - needs 6-10GB VRAM (4-bit quantized)
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",      # 8B - Great quality
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",               # 7B - Good alternative
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",   # 7B - Fast, reliable
    "phi3-mini": "microsoft/Phi-3.5-mini-instruct",       # 3.8B - Best for low VRAM
}

# Fallback chain: tries each until one fits in VRAM
FALLBACK_CHAIN = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class TrainingDataGenerator:
    """
    Generate training data using a large language model.
    
    The generated data is formatted for Enigma Engine training:
    - Q&A pairs in "Question: ... Answer: ..." format
    - Conversational exchanges
    - Instruction-following examples
    """
    
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        use_4bit: bool = True,
        device: str = "auto",
    ):
        """
        Initialize the generator.
        
        Args:
            model_id: HuggingFace model ID
            use_4bit: Use 4-bit quantization (recommended for CPU/low VRAM)
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.model_id = model_id
        self.use_4bit = use_4bit
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load(self) -> bool:
        """Load the model and tokenizer. Falls back to smaller models if VRAM is insufficient."""
        if self._loaded:
            return True

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Build list of models to try
        models_to_try = [self.model_id]
        for fallback in FALLBACK_CHAIN:
            if fallback != self.model_id and fallback not in models_to_try:
                models_to_try.append(fallback)

        for model_id in models_to_try:
            try:
                logger.info(f"Loading model: {model_id}")
                logger.info("This may take a few minutes on first run (downloading model)...")

                # Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Quantization config for 4-bit
                quant_config = None
                if self.use_4bit:
                    try:
                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        logger.info("Using 4-bit quantization")
                    except Exception as e:
                        logger.warning(f"4-bit quantization not available: {e}")
                        quant_config = None

                # Load model
                model_kwargs: dict = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }

                if quant_config:
                    model_kwargs["quantization_config"] = quant_config
                    model_kwargs["device_map"] = "auto"
                elif torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
                else:
                    logger.info("Running on CPU - this will be slower")
                    model_kwargs["torch_dtype"] = torch.float32

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs
                )

                self.model_id = model_id  # Update to the model that actually loaded
                self._loaded = True
                param_count = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Loaded {model_id} ({param_count / 1e9:.1f}B params)")
                return True

            except Exception as e:
                logger.warning(f"Failed to load {model_id}: {e}")
                if model_id != models_to_try[-1]:
                    logger.info("Trying smaller fallback model...")
                continue

        logger.error("All models failed to load. Check VRAM and installations.")
        return False
    
    def generate_qa_pairs(
        self,
        topic: str,
        count: int = 10,
        style: str = "informative",
    ) -> List[dict]:
        """
        Generate Q&A pairs on a topic.
        
        Args:
            topic: The topic to generate Q&A about
            count: Number of Q&A pairs to generate
            style: Style of responses ("informative", "conversational", "technical")
            
        Returns:
            List of {"question": ..., "answer": ...} dicts
        """
        if not self._loaded:
            if not self.load():
                return []
        
        style_instructions = {
            "informative": "Give clear, factual, educational answers.",
            "conversational": "Give friendly, natural conversational answers.",
            "technical": "Give detailed technical explanations.",
        }
        
        prompt = f"""Generate {count} diverse question-answer pairs about: {topic}

Instructions:
- {style_instructions.get(style, style_instructions["informative"])}
- Make questions varied (what, how, why, when, explain, describe, etc.)
- Answers should be 1-3 sentences, helpful and accurate
- Format each pair as:
Q: [question]
A: [answer]

Generate the Q&A pairs now:"""

        response = self._generate(prompt, max_tokens=2000)
        
        # Parse Q&A pairs from response (handles multi-line answers, code blocks, and <think> tags)
        pairs = []
        lines = response.strip().split('\n')
        current_q = None
        current_a_lines = []
        in_answer = False  # Track if we're collecting an answer (even if first line was empty)
        
        for line in lines:
            stripped = line.strip()
            # Check for new question
            if stripped.startswith(('Q:', 'Question:')):
                # Save previous Q&A if exists
                if current_q and in_answer:
                    answer = '\n'.join(current_a_lines).strip()
                    # Strip <think>...</think> blocks from DeepSeek-R1 reasoning
                    import re
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
                    if answer:
                        pairs.append({"question": current_q, "answer": answer})
                # Start new question
                current_q = stripped.split(':', 1)[1].strip()
                current_a_lines = []
                in_answer = False
            # Check for answer start
            elif stripped.startswith(('A:', 'Answer:')) and current_q:
                in_answer = True  # Mark that we're now in answer mode
                # First line of answer (may be empty if answer is on next line)
                first_line = stripped.split(':', 1)[1].strip()
                if first_line:
                    current_a_lines.append(first_line)
            # Continue collecting answer lines (including code blocks, reasoning, etc.)
            elif current_q and in_answer:
                # Keep collecting until we hit next Q: or Question:
                if not stripped.startswith(('Q:', 'Question:')):
                    # Only add non-empty lines or code-block markers
                    if stripped or line.strip().startswith('```'):
                        current_a_lines.append(line.rstrip())  # Preserve indentation for code
        
        # Don't forget the last Q&A pair
        if current_q and in_answer:
            answer = '\n'.join(current_a_lines).strip()
            # Strip <think>...</think> blocks from DeepSeek-R1 reasoning
            import re
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            if answer:
                pairs.append({"question": current_q, "answer": answer})
        
        logger.info(f"Generated {len(pairs)} Q&A pairs for topic: {topic}")
        return pairs
    
    def generate_conversations(
        self,
        scenario: str,
        count: int = 5,
        turns: int = 4,
    ) -> List[List[dict]]:
        """
        Generate multi-turn conversations.
        
        Args:
            scenario: Scenario description
            count: Number of conversations
            turns: Turns per conversation
            
        Returns:
            List of conversations, each being a list of {"role": ..., "content": ...}
        """
        if not self._loaded:
            if not self.load():
                return []
        
        prompt = f"""Generate {count} natural conversations about: {scenario}

Each conversation should have {turns} exchanges between User and Assistant.
Make them realistic and helpful.

Format:
---CONVERSATION---
User: [message]
Assistant: [response]
User: [follow-up]
Assistant: [response]
---END---

Generate the conversations:"""

        response = self._generate(prompt, max_tokens=3000)
        
        # Parse conversations
        conversations = []
        current_conv = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('User:'):
                current_conv.append({"role": "user", "content": line[5:].strip()})
            elif line.startswith('Assistant:'):
                current_conv.append({"role": "assistant", "content": line[10:].strip()})
            elif '---END---' in line or '---CONVERSATION---' in line:
                if current_conv:
                    conversations.append(current_conv)
                    current_conv = []
        
        if current_conv:
            conversations.append(current_conv)
        
        logger.info(f"Generated {len(conversations)} conversations")
        return conversations
    
    def generate_instructions(
        self,
        task_type: str,
        count: int = 10,
    ) -> List[dict]:
        """
        Generate instruction-following examples.
        
        Args:
            task_type: Type of task (e.g., "writing", "coding", "analysis")
            count: Number of examples
            
        Returns:
            List of {"instruction": ..., "response": ...} dicts
        """
        if not self._loaded:
            if not self.load():
                return []
        
        prompt = f"""Generate {count} instruction-following examples for: {task_type}

Each example should have:
- A clear instruction/request from a user
- A helpful, complete response

Format:
INSTRUCTION: [user request]
RESPONSE: [helpful response]

Generate varied examples:"""

        response = self._generate(prompt, max_tokens=3000)
        
        # Parse instructions
        examples = []
        lines = response.split('\n')
        current_inst = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('INSTRUCTION:'):
                current_inst = line[12:].strip()
            elif line.startswith('RESPONSE:') and current_inst:
                resp = line[9:].strip()
                if current_inst and resp:
                    examples.append({"instruction": current_inst, "response": resp})
                current_inst = None
        
        logger.info(f"Generated {len(examples)} instruction examples")
        return examples
    
    def _generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from prompt."""
        import torch
        
        # Format for instruct models
        if "instruct" in self.model_id.lower() or "chat" in self.model_id.lower():
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                formatted = f"[INST] {prompt} [/INST]"
        else:
            formatted = prompt
        
        inputs = self.tokenizer(formatted, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if formatted in response:
            response = response[len(formatted):].strip()
        
        return response
    
    def save_training_data(
        self,
        data: List[dict],
        output_path: str,
        format: str = "enigma",
    ):
        """
        Save generated data in training format.
        
        Args:
            data: List of Q&A pairs or conversations
            output_path: Output file path
            format: Output format ("enigma", "jsonl", "json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "enigma":
            # Enigma training format: Q: ... A: ...
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    if "question" in item:
                        f.write(f"Q: {item['question']}\n")
                        f.write(f"A: {item['answer']}\n\n")
                    elif "instruction" in item:
                        f.write(f"Q: {item['instruction']}\n")
                        f.write(f"A: {item['response']}\n\n")
                    elif isinstance(item, list):  # Conversation
                        for turn in item:
                            role = "Q" if turn["role"] == "user" else "A"
                            f.write(f"{role}: {turn['content']}\n")
                        f.write("\n")
            logger.info(f"Saved {len(data)} examples to {output_path}")
            
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved {len(data)} examples to {output_path}")
            
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} examples to {output_path}")


def _run_comprehensive(model_id: str, args):
    """Generate comprehensive training data across all categories."""

    COMPREHENSIVE_TOPICS = [
        # General conversation & personality
        ("general conversation and greetings", 40, "conversational"),
        ("emotions, empathy, and supportive responses", 30, "conversational"),
        ("declining inappropriate requests politely", 20, "conversational"),
        ("creative hypotheticals and thought experiments", 25, "conversational"),
        # Knowledge & reasoning
        ("explaining science and technology concepts simply", 30, "informative"),
        ("step-by-step math and logic problem solving", 30, "informative"),
        ("comparisons, pros/cons, and decision making", 25, "informative"),
        ("common sense and cause-effect reasoning", 25, "informative"),
        # Code generation
        ("Python programming, file I/O, data structures, and algorithms", 40, "technical"),
        ("debugging Python code, fixing errors, explaining bugs", 30, "technical"),
        ("web development with HTML CSS JavaScript and APIs", 25, "technical"),
        ("PyTorch machine learning, training loops, tensors, GPU usage", 30, "technical"),
        ("programming best practices, git, testing, clean code", 25, "technical"),
        # Tool usage
        ("when to use web search, file tools, vision, image generation", 30, "informative"),
        ("multi-step workflows combining multiple AI tools", 25, "informative"),
        # Avatar control
        ("AI avatar expressions: smile, wave, nod, thinking, happy, sad", 30, "conversational"),
        ("avatar reacting to conversations naturally", 25, "conversational"),
        # Self-awareness (Enigma AI Engine identity)
        ("AI assistant identity, capabilities, and limitations", 25, "informative"),
        ("AI ethics, safety, privacy, and responsible use", 20, "informative"),
        # Instruction following
        ("multi-step tasks, project setup, deployment workflows", 30, "informative"),
        ("writing emails, documentation, READMEs, commit messages", 25, "informative"),
        ("data transformation, CSV, JSON, cleaning, formatting", 20, "technical"),
        # GUI and training help
        ("using an AI desktop app with chat, image, code, video tabs", 25, "informative"),
        ("training AI models: hyperparameters, data prep, evaluation", 25, "technical"),
        ("fine-tuning, LoRA, transfer learning, data quality", 20, "technical"),
    ]

    total = sum(c for _, c, _ in COMPREHENSIVE_TOPICS)
    print(f"\n{'=' * 60}", flush=True)
    print("COMPREHENSIVE TRAINING DATA GENERATION", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"Categories: {len(COMPREHENSIVE_TOPICS)}", flush=True)
    print(f"Target total: ~{total} Q&A pairs", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    generator = TrainingDataGenerator(
        model_id=model_id,
        use_4bit=not args.no_4bit,
    )

    all_data: list = []
    start = time.time()
    
    # Use incremental save file to preserve progress
    incremental_file = args.output.replace('.txt', '_incremental.txt')
    
    # Check for existing incremental progress
    completed_topics = set()
    if os.path.exists(incremental_file):
        with open(incremental_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Count existing pairs
            existing_pairs = content.count('\nQ: ')
            if existing_pairs > 0:
                print(f"[!] Found {existing_pairs} existing pairs in {incremental_file}")
                print(f"    Resuming from where we left off...")
                # Load existing data
                for line in content.split('\n'):
                    if line.startswith('# COMPLETED: '):
                        completed_topics.add(line[13:].strip())

    for topic, count, style in COMPREHENSIVE_TOPICS:
        # Skip already completed topics
        if topic in completed_topics:
            print(f"\n--- {topic} (SKIPPED - already done) ---", flush=True)
            continue
            
        print(f"\n--- {topic} (target: {count}) ---", flush=True)
        data = generator.generate_qa_pairs(topic, count, style)
        all_data.extend(data)
        print(f"  Got {len(data)} pairs (running total: {len(all_data)})", flush=True)
        
        # Save incrementally after each topic
        if data:
            with open(incremental_file, 'a', encoding='utf-8') as f:
                f.write(f"# COMPLETED: {topic}\n")
                for item in data:
                    if 'question' in item:
                        f.write(f"Q: {item['question']}\n")
                        f.write(f"A: {item['answer']}\n\n")
            print(f"  [Saved to {incremental_file}]", flush=True)

    elapsed = time.time() - start

    # Consolidate incremental file into final output
    if os.path.exists(incremental_file):
        # Read all Q&A pairs from incremental file (strips # COMPLETED markers)
        with open(incremental_file, 'r', encoding='utf-8') as f:
            incremental_content = f.read()
        
        # Write clean version to final output
        with open(args.output, 'w', encoding='utf-8') as f:
            for line in incremental_content.split('\n'):
                if not line.startswith('# COMPLETED:'):
                    f.write(line + '\n')
        
        # Count total pairs
        total_pairs = incremental_content.count('\nQ: ') + (1 if incremental_content.startswith('Q: ') else 0)
        
        print(f"\n{'=' * 60}")
        print(f"Done! Generated {total_pairs} total Q&A pairs in {elapsed:.0f}s")
        print(f"Saved to: {args.output}")
        print(f"Incremental backup: {incremental_file}")
        print(f"\nNext steps:")
        print(f"  1. Review data: head -100 {args.output}")
        print(f"  2. Merge with existing: python scripts/merge_training_data.py")
        print(f"  3. Train: python run.py --train --data data/training.txt --epochs 20")
        print(f"{'=' * 60}\n")
    elif all_data:
        generator.save_training_data(all_data, args.output, args.format)
        print(f"\n{'=' * 60}")
        print(f"Done! Generated {len(all_data)} total Q&A pairs in {elapsed:.0f}s")
        print(f"Saved to: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Review data: head -100 {args.output}")
        print(f"  2. Merge with existing: python scripts/merge_training_data.py")
        print(f"  3. Train: python run.py --train --data data/training.txt --epochs 20")
        print(f"{'=' * 60}\n")
    else:
        print("\nNo data generated. Check logs for errors.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using a large AI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate Q&A about a topic
    python generate_training_data.py --topic "Python programming" --count 50
    
    # Generate from multiple topics
    python generate_training_data.py --topics "math,science,history" --count 20
    
    # Generate conversations
    python generate_training_data.py --topic "customer support" --type conversations
    
    # Use a specific model
    python generate_training_data.py --model "microsoft/Phi-3-mini-4k-instruct" --topic "AI"

Available models (sorted by CPU speed):
    tinyllama   : TinyLlama-1.1B  (fastest, ~10 tok/s on CPU)
    phi2        : Phi-2 2.7B     (fast, good quality)
    phi3-mini   : Phi-3-mini 3.8B (DEFAULT - best balance)
    mistral-7b  : Mistral-7B     (slower but excellent)
    qwen2-7b    : Qwen2-7B       (good alternative)
    llama3-8b   : Llama-3-8B     (best quality, needs HF access)
"""
    )
    
    parser.add_argument(
        "--model", "-m",
        default="phi3-mini",
        help="Model to use (name or HuggingFace ID). Default: phi3-mini"
    )
    parser.add_argument(
        "--topic", "-t",
        help="Topic to generate data about"
    )
    parser.add_argument(
        "--topics",
        help="Comma-separated list of topics"
    )
    parser.add_argument(
        "--topics-file",
        help="File with topics (one per line)"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=20,
        help="Number of examples per topic (default: 20)"
    )
    parser.add_argument(
        "--type",
        choices=["qa", "conversations", "instructions"],
        default="qa",
        help="Type of data to generate (default: qa)"
    )
    parser.add_argument(
        "--style",
        choices=["informative", "conversational", "technical"],
        default="informative",
        help="Style of responses (default: informative)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/generated_training.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["enigma", "jsonl", "json"],
        default="enigma",
        help="Output format (default: enigma)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Don't use 4-bit quantization (uses more memory)"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate comprehensive training data across all categories "
             "(conversation, reasoning, code, tools, avatar, identity, GUI, training)"
    )
    
    args = parser.parse_args()
    
    # Resolve model ID
    model_id = RECOMMENDED_MODELS.get(args.model, args.model)
    
    # Comprehensive mode: generate data across all categories
    if args.comprehensive:
        _run_comprehensive(model_id, args)
        return

    # Collect topics
    topics = []
    if args.topic:
        topics.append(args.topic)
    if args.topics:
        topics.extend([t.strip() for t in args.topics.split(',')])
    if args.topics_file:
        with open(args.topics_file) as f:
            topics.extend([line.strip() for line in f if line.strip()])
    
    if not topics:
        parser.error("Please provide at least one topic (--topic, --topics, or --topics-file)")
    
    # Initialize generator
    print(f"\n{'='*60}")
    print(f"Training Data Generator")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Topics: {', '.join(topics)}")
    print(f"Count per topic: {args.count}")
    print(f"Type: {args.type}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")
    
    # CPU warning
    import torch
    if not torch.cuda.is_available():
        print("\n[!] Running on CPU - generation will be slower")
        print("    Estimated time per topic:")
        if "tiny" in model_id.lower() or "1.1b" in model_id.lower():
            print("    - ~2-5 minutes for 20 Q&A pairs")
        elif "phi-2" in model_id.lower() or "2.7" in model_id.lower():
            print("    - ~5-10 minutes for 20 Q&A pairs")
        elif "phi-3" in model_id.lower() or "3.8" in model_id.lower():
            print("    - ~10-20 minutes for 20 Q&A pairs")
        else:  # 7B+ models
            print("    - ~30-60 minutes for 20 Q&A pairs")
        print("    Use --model tinyllama or phi2 for faster generation\n")
    
    generator = TrainingDataGenerator(
        model_id=model_id,
        use_4bit=not args.no_4bit,
    )
    
    # Generate data
    all_data = []
    
    for topic in topics:
        print(f"\nGenerating data for: {topic}")
        print("-" * 40)
        
        if args.type == "qa":
            data = generator.generate_qa_pairs(topic, args.count, args.style)
        elif args.type == "conversations":
            data = generator.generate_conversations(topic, args.count)
        elif args.type == "instructions":
            data = generator.generate_instructions(topic, args.count)
        
        all_data.extend(data)
        print(f"Generated {len(data)} examples")
    
    # Save
    if all_data:
        generator.save_training_data(all_data, args.output, args.format)
        print(f"\n{'='*60}")
        print(f"Done! Generated {len(all_data)} total examples")
        print(f"Saved to: {args.output}")
        print(f"{'='*60}\n")
    else:
        print("\nNo data generated. Check the logs for errors.")


if __name__ == "__main__":
    main()
