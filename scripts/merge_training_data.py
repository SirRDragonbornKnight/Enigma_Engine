#!/usr/bin/env python3
"""Merge all training data files into a single comprehensive training file.

Combines data from:
- data/base_knowledge.txt
- data/self_awareness_training.txt
- data/combined_action_training.txt
- data/tool_training_data.txt
- data/specialized/*.txt
- data/training_generated.txt (if exists, from generate_training_data.py)

Normalizes all formats to Q:/A: pairs and deduplicates.

Usage:
    python scripts/merge_training_data.py
    python scripts/merge_training_data.py --output data/training.txt
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# All data files to merge (relative to project root)
DATA_FILES = [
    "data/base_knowledge.txt",
    "data/self_awareness_training.txt",
    "data/combined_action_training.txt",
    "data/tool_training_data.txt",
    "data/specialized/avatar_commands_training.txt",
    "data/specialized/avatar_control_training.txt",
    "data/specialized/avatar_control_training_tool_based.txt",
    "data/specialized/avatar_expression_training.txt",
    "data/specialized/avatar_training.txt",
    "data/specialized/code_training.txt",
    "data/specialized/router_training.txt",
    "data/specialized/trainer_training.txt",
    "data/specialized/vision_training.txt",
    "data/specialized/wants_and_learned_design_training.txt",
    # Generated data (may not exist yet)
    "data/training_generated.txt",
    "data/generated_training.txt",
]


def normalize_qa_format(text: str) -> list[tuple[str, str]]:
    """Parse text into (question, answer) pairs, handling multiple formats.

    Supported formats:
    - Q: ... / A: ...
    - Question: ... / Answer: ...
    - User: ... / Assistant: ...
    - Human: ... / AI: ...
    - ### Instruction / ### Response
    - Plain paragraphs (skipped)
    """
    pairs: list[tuple[str, str]] = []

    # Try Q:/A: format first (most common)
    qa_pattern = re.compile(
        r'(?:^|\n)\s*(?:Q|Question|User|Human|Input|Instruction)\s*:\s*(.+?)'
        r'\n\s*(?:A|Answer|Assistant|AI|Output|Response)\s*:\s*(.+?)(?=\n\s*(?:Q|Question|User|Human|Input|Instruction)\s*:|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    for match in qa_pattern.finditer(text):
        q = match.group(1).strip()
        a = match.group(2).strip()
        if q and a and len(q) >= 3 and len(a) >= 3:
            # Clean up multi-line answers
            q = " ".join(q.split())
            # Keep code blocks intact but clean regular text
            if "```" not in a and "\n" in a:
                # Check if it looks like structured output (numbered lists, etc.)
                lines = a.split("\n")
                if any(line.strip().startswith(("1.", "2.", "3.", "-", "*")) for line in lines):
                    a = a.strip()  # Keep structure
                else:
                    a = " ".join(a.split())
            pairs.append((q, a))

    # Try ### Instruction / ### Response format
    if not pairs:
        inst_pattern = re.compile(
            r'###\s*Instruction[:\s]*\n(.+?)\n###\s*Response[:\s]*\n(.+?)(?=###\s*Instruction|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        for match in inst_pattern.finditer(text):
            q = match.group(1).strip()
            a = match.group(2).strip()
            if q and a:
                pairs.append((q, a))

    # Try pipe-delimited format: message|response (used in avatar training)
    if not pairs:
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) == 2:
                    q = parts[0].strip()
                    a = parts[1].strip()
                    if q and a and len(q) >= 2 and len(a) >= 5:
                        pairs.append((q, a))

    return pairs


def deduplicate(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate Q&A pairs by normalizing questions."""
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []

    for q, a in pairs:
        key = re.sub(r'[^a-z0-9\s]', '', q.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append((q, a))

    return unique


def main():
    parser = argparse.ArgumentParser(description="Merge all training data into one file")
    parser.add_argument(
        "--output", "-o",
        default="data/training.txt",
        help="Output merged training file (default: data/training.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    all_pairs: list[tuple[str, str]] = []
    files_processed = 0

    print(f"\n{'=' * 60}")
    print("MERGE TRAINING DATA")
    print(f"{'=' * 60}")

    for rel_path in DATA_FILES:
        path = root / rel_path
        if not path.exists():
            continue

        try:
            text = path.read_text(encoding="utf-8")
            pairs = normalize_qa_format(text)
            if pairs:
                all_pairs.extend(pairs)
                files_processed += 1
                print(f"  {rel_path}: {len(pairs)} pairs")
            else:
                print(f"  {rel_path}: (no Q&A pairs found, skipped)")
        except Exception as e:
            logger.warning(f"  Failed to read {rel_path}: {e}")

    # Deduplicate
    before = len(all_pairs)
    all_pairs = deduplicate(all_pairs)
    dupes = before - len(all_pairs)

    print(f"\n  Files processed: {files_processed}")
    print(f"  Total pairs: {len(all_pairs)}")
    if dupes:
        print(f"  Duplicates removed: {dupes}")

    if args.dry_run:
        print("\n(Dry run - no files written)")
        return

    # Write output
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Enigma AI Engine - Merged Training Data\n")
        f.write(f"# {len(all_pairs)} Q&A pairs from {files_processed} source files\n")
        f.write(f"# Generated by scripts/merge_training_data.py\n\n")

        for q, a in all_pairs:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n  Saved to: {output_path} ({size_kb:.1f} KB)")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  1. Generate more data: python scripts/generate_training_data.py --comprehensive")
    print(f"  2. Re-merge: python scripts/merge_training_data.py")
    print(f"  3. Train: python run.py --train --data {args.output} --epochs 20")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
