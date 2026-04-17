"""
Tokenizer Metrics & Analysis

Provides vocabulary analysis, coverage evaluation, and compression
ratio computation for BPE and character tokenizers.
"""
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def analyze_vocabulary(tokenizer: Any) -> dict[str, Any]:
    """Analyze a tokenizer's vocabulary.

    Returns a dict with:
      - vocab_size: total tokens
      - num_merges: number of BPE merges (0 for non-BPE)
      - num_special: count of special tokens
      - token_lengths: {min, max, mean, median} of token string lengths
      - top_tokens: 20 shortest merged tokens (most common subwords)
      - single_char_tokens: count of single-character (base) tokens
      - use_utf8_bytes: whether byte-level mode is active
    """
    vocab = getattr(tokenizer, 'token_to_id', None) or {}
    specials = getattr(tokenizer, 'special_tokens', None) or {}
    merges = getattr(tokenizer, 'merges', None) or []
    use_bytes = getattr(tokenizer, 'use_utf8_bytes', False)

    # Token string lengths (excluding specials)
    lengths = []
    single_char = 0
    for tok in vocab:
        if tok in specials:
            continue
        n = len(tok)
        lengths.append(n)
        if n == 1:
            single_char += 1

    lengths.sort()
    mean_len = sum(lengths) / max(len(lengths), 1)
    median_len = lengths[len(lengths) // 2] if lengths else 0.0

    # Top merged tokens: shortest non-special tokens that came from merges
    merged_tokens = []
    if merges:
        merge_set = {a + b for a, b in merges}
        merged_tokens = sorted(
            (tok for tok in vocab if tok in merge_set),
            key=len,
        )[:20]

    return {
        'vocab_size': len(vocab),
        'num_merges': len(merges),
        'num_special': len(specials),
        'token_lengths': {
            'min': lengths[0] if lengths else 0,
            'max': lengths[-1] if lengths else 0,
            'mean': round(mean_len, 2),
            'median': median_len,
        },
        'top_tokens': merged_tokens,
        'single_char_tokens': single_char,
        'use_utf8_bytes': use_bytes,
    }


def evaluate_coverage(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    """Evaluate how well a tokenizer covers a corpus.

    Returns:
      - total_tokens: number of tokens produced
      - unique_tokens: distinct token IDs used
      - unk_count: tokens mapped to <unk>
      - unk_rate: fraction of tokens that are <unk>
      - coverage: 1 - unk_rate (higher is better)
      - token_freq: Counter of the top 20 most common token IDs
    """
    unk_id = getattr(tokenizer, 'unk_token_id', 3)
    total = 0
    unk_count = 0
    freq: Counter = Counter()

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for tid in ids:
            total += 1
            freq[tid] += 1
            if tid == unk_id:
                unk_count += 1

    unk_rate = unk_count / max(total, 1)

    # Top 20 most frequent
    top20 = dict(freq.most_common(20))

    return {
        'total_tokens': total,
        'unique_tokens': len(freq),
        'unk_count': unk_count,
        'unk_rate': round(unk_rate, 6),
        'coverage': round(1.0 - unk_rate, 6),
        'token_freq_top20': top20,
    }


def compute_compression_ratio(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    """Compute how efficiently the tokenizer compresses text.

    Returns:
      - total_chars: raw character count
      - total_tokens: token count
      - chars_per_token: average characters per token (higher = better compression)
      - compression_ratio: chars / tokens
    """
    total_chars = 0
    total_tokens = 0

    for text in texts:
        total_chars += len(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(ids)

    cpt = total_chars / max(total_tokens, 1)

    return {
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'chars_per_token': round(cpt, 2),
        'compression_ratio': round(cpt, 2),
    }


def detect_issues(tokenizer: Any, texts: list[str]) -> list[str]:
    """Run basic health checks on a tokenizer against a corpus.

    Returns a list of human-readable warning strings (empty = healthy).
    """
    warnings: list[str] = []

    coverage = evaluate_coverage(tokenizer, texts)
    compression = compute_compression_ratio(tokenizer, texts)

    if coverage['unk_rate'] > 0.05:
        warnings.append(
            f"High UNK rate: {coverage['unk_rate']:.1%} of tokens are <unk>. "
            f"Consider retraining with more data or a larger vocab."
        )

    if coverage['unk_rate'] > 0.0:
        warnings.append(
            f"UNK tokens present: {coverage['unk_count']} tokens "
            f"({coverage['unk_rate']:.2%}) mapped to <unk>."
        )

    vocab_size = getattr(tokenizer, 'vocab_size', 0)
    if coverage['unique_tokens'] < vocab_size * 0.1 and vocab_size > 100:
        warnings.append(
            f"Low vocabulary utilization: only {coverage['unique_tokens']} "
            f"of {vocab_size} tokens used ({coverage['unique_tokens'] / vocab_size:.0%}). "
            f"Vocab may be oversized for this data."
        )

    if compression['chars_per_token'] < 2.0:
        warnings.append(
            f"Low compression: {compression['chars_per_token']:.1f} chars/token. "
            f"Tokenizer may not have learned useful merges."
        )

    return warnings


def format_report(tokenizer: Any, texts: list[str]) -> str:
    """Generate a human-readable tokenizer analysis report."""
    vocab = analyze_vocabulary(tokenizer)
    coverage = evaluate_coverage(tokenizer, texts)
    compression = compute_compression_ratio(tokenizer, texts)
    issues = detect_issues(tokenizer, texts)

    lines = [
        "=" * 50,
        "  Tokenizer Analysis Report",
        "=" * 50,
        "",
        "Vocabulary:",
        f"  Vocab size:        {vocab['vocab_size']:,}",
        f"  BPE merges:        {vocab['num_merges']:,}",
        f"  Special tokens:    {vocab['num_special']}",
        f"  Single-char tokens: {vocab['single_char_tokens']}",
        f"  UTF-8 byte mode:   {vocab['use_utf8_bytes']}",
        "",
        "Token lengths:",
        f"  Min: {vocab['token_lengths']['min']}  "
        f"Max: {vocab['token_lengths']['max']}  "
        f"Mean: {vocab['token_lengths']['mean']}  "
        f"Median: {vocab['token_lengths']['median']}",
        "",
        "Coverage:",
        f"  Total tokens:      {coverage['total_tokens']:,}",
        f"  Unique tokens:     {coverage['unique_tokens']:,}",
        f"  UNK tokens:        {coverage['unk_count']:,}",
        f"  Coverage:          {coverage['coverage']:.2%}",
        "",
        "Compression:",
        f"  Total chars:       {compression['total_chars']:,}",
        f"  Total tokens:      {compression['total_tokens']:,}",
        f"  Chars per token:   {compression['chars_per_token']}",
    ]

    if vocab['top_tokens']:
        lines.append("")
        lines.append("Top merged tokens:")
        for tok in vocab['top_tokens']:
            lines.append(f"  '{tok}'")

    if issues:
        lines.append("")
        lines.append("Issues:")
        for w in issues:
            lines.append(f"  [!] {w}")
    else:
        lines.append("")
        lines.append("No issues detected.")

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
