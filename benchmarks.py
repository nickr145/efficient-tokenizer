"""
benchmarks.py — Tokenizer benchmarking suite for the Efficient Tokenizer project.

Compares three encoding approaches on the same corpus:
  1. CharacterLevel      — baseline matching shakespeare_gpt.py (stoi/itos mapping)
  2. StandardBPE         — vanilla frequency-based BPE
  3. SignificanceAwareBPE — entropy-weighted BPE (novel)

Metrics
-------
  compression_ratio          original bytes / token count
  merge_speed (train_time_s) seconds to build the vocabulary
  entropy_per_token          Shannon entropy H of the encoded token distribution (bits)
  vocabulary_coverage        % of unique characters in the text that the tokenizer handles
  sequence_length_reduction  % shorter than the character-level baseline

Usage
-----
  python benchmarks.py                   # downloads input.txt, runs on first 50K chars
  python benchmarks.py --full            # runs on the entire corpus (~1MB, slower)
  python benchmarks.py --vocab 1024      # custom BPE vocab size
"""

import csv
import math
import os
import time
from collections import Counter
from typing import Dict, List

from bpe_tokenizer import StandardBPE, SignificanceAwareBPE


# ---------------------------------------------------------------------------
# Character-level baseline  (mirrors shakespeare_gpt.py)
# ---------------------------------------------------------------------------

class _CharacterLevel:
    """
    Replicates the stoi/itos encoding from shakespeare_gpt.py.
    Vocabulary = sorted unique characters seen during construction.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size: int = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def coverage(self, text: str) -> float:
        unique = set(text)
        return len(unique & set(self.stoi)) / len(unique) * 100 if unique else 100.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(token_ids: List[int]) -> float:
    """H = −Σ p_i · log₂(p_i) for the token frequency distribution."""
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    N = len(token_ids)
    return -sum((c / N) * math.log2(c / N) for c in counts.values())


def _vocab_utilization(token_ids: List[int], vocab_size: int) -> float:
    """% of vocab slots that appear at least once in the encoded text."""
    return len(set(token_ids)) / vocab_size * 100 if vocab_size else 0.0


# ---------------------------------------------------------------------------
# Per-tokenizer benchmark runners
# ---------------------------------------------------------------------------

def _run_char_level(text: str) -> dict:
    tok = _CharacterLevel(text)
    tokens = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    seq_len = len(tokens)

    return {
        "name": "CharacterLevel",
        "vocab_size": tok.vocab_size,
        "original_bytes": n_bytes,
        "sequence_length": seq_len,
        "compression_ratio": n_bytes / seq_len if seq_len else 0.0,
        "train_time_s": 0.0,
        "entropy_per_token": _shannon_entropy(tokens),
        "vocabulary_coverage": tok.coverage(text),
        "vocabulary_utilization": _vocab_utilization(tokens, tok.vocab_size),
        "sequence_length_reduction_pct": 0.0,  # this IS the baseline
    }


def _run_bpe(tokenizer, text: str, vocab_size: int, baseline_seq_len: int) -> dict:
    n_bytes = len(text.encode("utf-8"))

    t0 = time.perf_counter()
    tokenizer.train(text, vocab_size)
    train_time = time.perf_counter() - t0

    tokens = tokenizer.encode(text)
    seq_len = len(tokens)
    reduction = (
        (baseline_seq_len - seq_len) / baseline_seq_len * 100
        if baseline_seq_len > 0
        else 0.0
    )

    return {
        "name": tokenizer.__class__.__name__,
        "vocab_size": vocab_size,
        "original_bytes": n_bytes,
        "sequence_length": seq_len,
        "compression_ratio": n_bytes / seq_len if seq_len else 0.0,
        "train_time_s": train_time,
        "entropy_per_token": _shannon_entropy(tokens),
        "vocabulary_coverage": 100.0,  # BPE handles all bytes by construction
        "vocabulary_utilization": _vocab_utilization(tokens, vocab_size),
        "sequence_length_reduction_pct": reduction,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmarks(
    text: str,
    vocab_size: int = 512,
    save_csv: bool = True,
    output_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Run all three tokenizers and collect metrics.

    Args:
        text:       Training and evaluation corpus (same text for fair comparison).
        vocab_size: BPE target vocabulary size (must be >= 256).
        save_csv:   Write results/compression_comparison.csv when True.
        output_dir: Directory for CSV output.
        verbose:    Print progress lines during training.

    Returns:
        Dict mapping tokenizer name → metrics dict.
    """
    n_bytes = len(text.encode("utf-8"))
    if verbose:
        print(
            f"Benchmarking on {len(text):,} chars / {n_bytes:,} bytes "
            f"| BPE vocab_size={vocab_size}\n"
        )

    results: Dict[str, dict] = {}

    # 1 — character-level baseline
    if verbose:
        print("  [1/3] CharacterLevel ...", end=" ", flush=True)
    r = _run_char_level(text)
    results["CharacterLevel"] = r
    if verbose:
        print(f"done  ({r['sequence_length']:,} tokens, {r['compression_ratio']:.3f}x)")

    baseline_len = r["sequence_length"]

    # 2 — Standard BPE
    if verbose:
        print("  [2/3] StandardBPE ...", end=" ", flush=True)
    r = _run_bpe(StandardBPE(), text, vocab_size, baseline_len)
    results["StandardBPE"] = r
    if verbose:
        print(
            f"done  ({r['sequence_length']:,} tokens, {r['compression_ratio']:.3f}x, "
            f"{r['train_time_s']:.2f}s)"
        )

    # 3 — Significance-Aware BPE
    if verbose:
        print("  [3/3] SignificanceAwareBPE ...", end=" ", flush=True)
    r = _run_bpe(SignificanceAwareBPE(), text, vocab_size, baseline_len)
    results["SignificanceAwareBPE"] = r
    if verbose:
        print(
            f"done  ({r['sequence_length']:,} tokens, {r['compression_ratio']:.3f}x, "
            f"{r['train_time_s']:.2f}s)\n"
        )

    print_table(results)

    if save_csv:
        save_results_csv(results, output_dir)

    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

_ROWS = [
    ("Vocab size",              "vocab_size",                    lambda v: f"{int(v):,}"),
    ("Original bytes",          "original_bytes",                lambda v: f"{int(v):,}"),
    ("Sequence length",         "sequence_length",               lambda v: f"{int(v):,}"),
    ("Compression ratio",       "compression_ratio",             lambda v: f"{v:.4f}x"),
    ("Train time (s)",          "train_time_s",                  lambda v: f"{v:.4f}"),
    ("Entropy / token (bits)",  "entropy_per_token",             lambda v: f"{v:.4f}"),
    ("Vocab coverage (%)",      "vocabulary_coverage",           lambda v: f"{v:.1f}%"),
    ("Vocab utilization (%)",   "vocabulary_utilization",        lambda v: f"{v:.1f}%"),
    ("Seq len reduction (%)",   "sequence_length_reduction_pct", lambda v: f"{v:.2f}%"),
]


def print_table(results: Dict[str, dict]) -> None:
    """Print a formatted side-by-side comparison table."""
    names = list(results.keys())
    label_w = 26
    val_w = 22
    total_w = label_w + val_w * len(names)
    sep = "-" * total_w

    print(sep)
    header = f"{'Metric':<{label_w}}" + "".join(f"{n:>{val_w}}" for n in names)
    print(header)
    print(sep)

    for label, key, fmt in _ROWS:
        row = f"{label:<{label_w}}"
        for name in names:
            val = results[name].get(key, "N/A")
            cell = fmt(val) if isinstance(val, (int, float)) else str(val)
            row += f"{cell:>{val_w}}"
        print(row)

    print(sep)
    _print_summary(results, names)


def _print_summary(results: Dict[str, dict], names: list) -> None:
    """Print one-line winner callouts for the key metrics."""
    bpe_names = [n for n in names if n != "CharacterLevel"]
    if not bpe_names:
        return

    print("\nBPE comparison (vs StandardBPE baseline):")
    std = results.get("StandardBPE", {})
    sig = results.get("SignificanceAwareBPE", {})

    if std and sig:
        cr_diff = sig["compression_ratio"] - std["compression_ratio"]
        er_diff = sig["entropy_per_token"] - std["entropy_per_token"]
        sl_diff = sig["sequence_length_reduction_pct"] - std["sequence_length_reduction_pct"]
        speed_ratio = sig["train_time_s"] / std["train_time_s"] if std["train_time_s"] > 0 else float("inf")

        sign = lambda x: "+" if x >= 0 else ""
        print(f"  Compression ratio     {sign(cr_diff)}{cr_diff:+.4f}x  "
              f"({'better' if cr_diff > 0 else 'worse'})")
        print(f"  Entropy / token       {er_diff:+.4f} bits")
        print(f"  Seq len reduction     {sl_diff:+.2f} pct pts")
        print(f"  Training overhead     {speed_ratio:.2f}x slower")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "name", "vocab_size", "original_bytes", "sequence_length",
    "compression_ratio", "train_time_s", "entropy_per_token",
    "vocabulary_coverage", "vocabulary_utilization",
    "sequence_length_reduction_pct",
]


def save_results_csv(results: Dict[str, dict], output_dir: str = "results") -> None:
    """Save benchmark results to results/compression_comparison.csv."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "compression_comparison.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)

    print(f"\nSaved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import urllib.request

    parser = argparse.ArgumentParser(description="Run tokenizer benchmarks")
    parser.add_argument(
        "--full", action="store_true",
        help="Benchmark on the entire corpus instead of the first 50K chars"
    )
    parser.add_argument(
        "--vocab", type=int, default=512,
        help="BPE vocabulary size (default: 512)"
    )
    parser.add_argument(
        "--sample", type=int, default=50_000,
        help="Number of characters to sample when --full is not set (default: 50000)"
    )
    args = parser.parse_args()

    data_path = os.path.join("data", "input.txt")
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        print("Downloading Tiny Shakespeare dataset ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            data_path,
        )
        print(f"Saved to {data_path}\n")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    corpus = text if args.full else text[: args.sample]
    run_benchmarks(corpus, vocab_size=args.vocab)
