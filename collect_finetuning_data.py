"""
Fine-Tuning Data Collector for Enigma Engine
=============================================
Downloads and formats instruction-following datasets for fine-tuning.

Sources (all require `pip install datasets`):
  - OASST1 (--oasst)        - Open Assistant conversations (~80K turns)
  - Dolly 15k (--dolly)     - Databricks instruction pairs (~15K)
  - SlimOrca (--slimorca N)  - Instruction-following from Open-Orca (N = max samples)
  - All sources (--all)      - Download everything

Output format:
  JSONL with {"prompt": "...", "completion": "..."} per line.
  Ready for FORGE → Basic training mode.

Usage:
  python collect_finetuning_data.py --oasst
  python collect_finetuning_data.py --dolly
  python collect_finetuning_data.py --slimorca 50000
  python collect_finetuning_data.py --all
  python collect_finetuning_data.py --stats
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/finetune")


def _ensure_datasets():
    """Check that the `datasets` library is installed."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        logger.error(
            "The `datasets` library is required.\n"
            "Install with: pip install datasets")
        return False


def _dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Remove exact-duplicate prompt+completion pairs."""
    seen = set()
    unique = []
    for item in pairs:
        key = hashlib.sha256(
            (item["prompt"] + item["completion"]).encode()
        ).digest()[:16]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _write_jsonl(pairs: list[dict], path: Path) -> int:
    """Write pairs to JSONL file, return count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(pairs)


def _clean_text(text: str) -> str:
    """Basic text cleaning for instruction data."""
    if not text:
        return ""
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


# ── OASST1 ─────────────────────────────────────────────────────────

def collect_oasst() -> list[dict]:
    """Download and format Open Assistant Conversations (OASST1).

    Extracts instruction/response pairs from the conversation tree.
    Uses only English, top-ranked messages.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info("Downloading OASST1 dataset...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    logger.info(f"OASST1: {len(ds)} messages loaded")

    # Build conversation trees
    messages_by_id = {}
    children = {}
    for msg in ds:
        msg_id = msg["message_id"]
        messages_by_id[msg_id] = msg
        parent = msg.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(msg_id)

    # Extract prompt-response pairs: root message (prompter) →
    # best child (assistant, highest rank)
    pairs = []
    for msg in ds:
        if msg["role"] != "prompter" or msg["parent_id"] is not None:
            continue
        if msg.get("lang", "en") != "en":
            continue

        prompt = _clean_text(msg["text"])
        if not prompt or len(prompt) < 10:
            continue

        # Find best assistant reply
        kid_ids = children.get(msg["message_id"], [])
        assistant_replies = []
        for kid_id in kid_ids:
            kid = messages_by_id.get(kid_id)
            if kid and kid["role"] == "assistant":
                rank = kid.get("rank", 999)
                if rank is None:
                    rank = 999
                assistant_replies.append((rank, kid))

        if not assistant_replies:
            continue

        # Pick highest-ranked (lowest rank number)
        assistant_replies.sort(key=lambda x: x[0])
        best = assistant_replies[0][1]
        completion = _clean_text(best["text"])
        if not completion or len(completion) < 10:
            continue

        pairs.append({
            "prompt": prompt,
            "completion": completion,
        })

    pairs = _dedup_pairs(pairs)
    logger.info(f"OASST1: {len(pairs)} instruction pairs extracted")
    return pairs


# ── Dolly 15k ──────────────────────────────────────────────────────

def collect_dolly() -> list[dict]:
    """Download and format Databricks Dolly 15k.

    Simple instruction/response pairs. Includes optional context
    which gets prepended to the prompt when present.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info("Downloading Dolly 15k dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    logger.info(f"Dolly: {len(ds)} examples loaded")

    pairs = []
    for item in ds:
        instruction = _clean_text(item.get("instruction", ""))
        context = _clean_text(item.get("context", ""))
        response = _clean_text(item.get("response", ""))

        if not instruction or not response:
            continue
        if len(instruction) < 5 or len(response) < 10:
            continue

        # Prepend context to prompt when available
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction

        pairs.append({
            "prompt": prompt,
            "completion": response,
        })

    pairs = _dedup_pairs(pairs)
    logger.info(f"Dolly: {len(pairs)} instruction pairs extracted")
    return pairs


# ── SlimOrca ───────────────────────────────────────────────────────

def collect_slimorca(max_samples: int = 100000) -> list[dict]:
    """Download and format Open-Orca SlimOrca.

    Large instruction-following dataset. Streams to avoid loading
    everything into memory at once.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info(
        f"Downloading SlimOrca (max {max_samples:,} samples)...")
    ds = load_dataset(
        "Open-Orca/SlimOrca", split="train", streaming=True)

    pairs = []
    count = 0
    for item in ds:
        if count >= max_samples:
            break

        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            count += 1
            continue

        # Find system + user + assistant turns
        system = ""
        prompt = ""
        completion = ""
        for turn in conversations:
            role = turn.get("from", "")
            value = _clean_text(turn.get("value", ""))
            if role == "system" and value:
                system = value
            elif role == "human" and value:
                prompt = value
            elif role == "gpt" and value:
                completion = value

        if not prompt or not completion:
            count += 1
            continue
        if len(prompt) < 5 or len(completion) < 10:
            count += 1
            continue

        # Include system prompt if non-generic
        if system and system.lower() not in (
            "you are an ai assistant.",
            "you are a helpful assistant.",
            "",
        ):
            prompt = f"{system}\n\n{prompt}"

        pairs.append({
            "prompt": prompt,
            "completion": completion,
        })
        count += 1

        if count % 10000 == 0:
            logger.info(f"  SlimOrca: {count:,} processed, "
                        f"{len(pairs):,} kept...")

    pairs = _dedup_pairs(pairs)
    logger.info(f"SlimOrca: {len(pairs)} instruction pairs extracted")
    return pairs


# ── Combine & Stats ───────────────────────────────────────────────

def combine_all(output_dir: Path) -> Path:
    """Merge all source JSONL files into one combined file."""
    combined_path = output_dir / "combined_finetune.jsonl"
    all_pairs = []

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        if jsonl_file.name == "combined_finetune.jsonl":
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    all_pairs = _dedup_pairs(all_pairs)
    _write_jsonl(all_pairs, combined_path)

    size_mb = combined_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Combined: {len(all_pairs):,} pairs, {size_mb:.1f} MB "
        f"→ {combined_path}")
    return combined_path


def show_stats(output_dir: Path):
    """Show statistics for collected fine-tuning data."""
    if not output_dir.exists():
        print("No fine-tuning data collected yet.")
        print("  Run: python collect_finetuning_data.py --all")
        return

    print(f"\n{'Source':<25} {'Pairs':>10} {'Size':>10}")
    print("-" * 47)

    total_pairs = 0
    total_bytes = 0

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        n_lines = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_lines += 1
        size = jsonl_file.stat().st_size
        total_pairs += n_lines
        total_bytes += size

        size_str = (
            f"{size / (1024*1024):.1f} MB" if size > 1024 * 1024
            else f"{size / 1024:.0f} KB")
        print(f"  {jsonl_file.name:<23} {n_lines:>10,} {size_str:>10}")

    print("-" * 47)
    total_str = (
        f"{total_bytes / (1024*1024):.1f} MB"
        if total_bytes > 1024 * 1024
        else f"{total_bytes / 1024:.0f} KB")
    print(f"  {'TOTAL':<23} {total_pairs:>10,} {total_str:>10}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect fine-tuning data for Enigma Engine")
    parser.add_argument(
        "--oasst", action="store_true",
        help="Download OASST1 conversation dataset")
    parser.add_argument(
        "--dolly", action="store_true",
        help="Download Databricks Dolly 15k")
    parser.add_argument(
        "--slimorca", type=int, nargs="?", const=100000,
        help="Download SlimOrca (default: 100K samples)")
    parser.add_argument(
        "--all", action="store_true",
        help="Download all sources")
    parser.add_argument(
        "--combine-only", action="store_true",
        help="Re-combine existing source files")
    parser.add_argument(
        "--stats", action="store_true",
        help="Show collected data statistics")
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        show_stats(output_dir)
        return

    if args.combine_only:
        combine_all(output_dir)
        return

    any_source = args.oasst or args.dolly or args.slimorca or args.all
    if not any_source:
        parser.print_help()
        return

    start = time.monotonic()
    collected = []

    if args.oasst or args.all:
        pairs = collect_oasst()
        if pairs:
            path = output_dir / "oasst1.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("OASST1", len(pairs)))

    if args.dolly or args.all:
        pairs = collect_dolly()
        if pairs:
            path = output_dir / "dolly_15k.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("Dolly 15k", len(pairs)))

    if args.slimorca is not None or args.all:
        max_n = args.slimorca if args.slimorca is not None else 100000
        pairs = collect_slimorca(max_samples=max_n)
        if pairs:
            path = output_dir / "slimorca.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("SlimOrca", len(pairs)))

    if collected:
        combine_all(output_dir)

    elapsed = time.monotonic() - start
    m, s = int(elapsed // 60), int(elapsed % 60)
    print(f"\nDone in {m}m {s:02d}s")
    for name, count in collected:
        print(f"  {name}: {count:,} pairs")
    print()
    show_stats(output_dir)


if __name__ == "__main__":
    main()
