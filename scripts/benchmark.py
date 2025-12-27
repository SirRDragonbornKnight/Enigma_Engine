#!/usr/bin/env python3
"""
Benchmark Enigma model performance.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --size medium --batch-size 8
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Enigma model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--size", "-s", type=str, default="small",
        choices=["tiny", "small", "medium", "large", "xl"],
        help="Model size to benchmark"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length"
    )
    parser.add_argument(
        "--gen-len", type=int, default=50,
        help="Generation length"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=10,
        help="Benchmark iterations"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    import torch
    from enigma.core.model import Enigma
    from enigma.core.model_config import get_model_config
    from enigma.core.tokenizer import load_tokenizer
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Benchmarking Enigma {args.size} on {device}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}, Gen len: {args.gen_len}")
    print("-" * 60)
    
    # Create model
    config = get_model_config(args.size)
    tokenizer = load_tokenizer()
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    
    model = Enigma(vocab_size=vocab_size, **config)
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {model.num_parameters:,}")
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    
    # Benchmark forward pass
    print("\n1. Forward Pass (no cache)")
    
    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_ids, use_cache=False)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(args.iterations):
            start = time.perf_counter()
            _ = model(input_ids, use_cache=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = (args.batch_size * args.seq_len) / avg_time
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    
    # Benchmark generation
    print(f"\n2. Generation ({args.gen_len} tokens)")
    
    gen_input = torch.randint(0, vocab_size, (1, 10), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.generate(gen_input.clone(), max_new_tokens=args.gen_len, use_cache=True)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(args.iterations):
            start = time.perf_counter()
            _ = model.generate(gen_input.clone(), max_new_tokens=args.gen_len, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    tokens_per_sec = args.gen_len / avg_time
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Time per token: {(avg_time/args.gen_len)*1000:.2f} ms")
    
    # Memory usage
    print("\n3. Memory Usage")
    if device.type == "cuda":
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    else:
        import psutil
        process = psutil.Process()
        print(f"  RAM usage: {process.memory_info().rss/1024**2:.1f} MB")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
