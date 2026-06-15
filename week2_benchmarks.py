"""
week2_benchmarks.py — Week 2 model comparison benchmarks.

Evaluates all three trained GPT models across six metrics:
  1. Sequence length reduction  (vs char-level baseline)
  2. Training speed             (sec / 1000 steps, tokens / sec)
  3. Final validation loss      (cross-entropy, bits-per-token)
  4. Test-set perplexity        (exp(val_loss) on held-out test split)
  5. Peak memory usage          (MB — RSS process memory delta during inference)
  6. Text generation samples    (200-token samples from each model)

Outputs
-------
  results/week2_model_comparison.csv   — one row per model, all metrics
  results/week2_samples.txt            — 200-token generation sample per model

Models are loaded from checkpoints in models/; missing checkpoints are
skipped with a warning rather than aborting the whole run.

Usage
-----
  python week2_benchmarks.py                          # benchmark all 3 models
  python week2_benchmarks.py --model char             # single model
  python week2_benchmarks.py --vocab 512 --gen-len 200
"""

import argparse
import csv
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe_tokenizer import StandardBPE, SignificanceAwareBPE
from shakespeare_gpt_v2 import (
    CharLevelTokenizer, BPETokenizerWrapper,
    GPTLanguageModel, ModelConfig,
    build_splits,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _get_device()

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def _load_corpus() -> str:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "input.txt")
    if not os.path.exists(path):
        print("Downloading Tiny Shakespeare ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Current process RSS in megabytes (cross-platform)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        import resource
        # macOS / Linux: getrusage returns KB on Linux, bytes on macOS
        ru = resource.getrusage(resource.RUSAGE_SELF)
        import platform
        factor = 1024 if platform.system() == "Darwin" else 1024 * 1024
        return ru.ru_maxrss / factor

# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------

def _build_tokenizer(model_key: str, text: str, vocab_size: int, bpe_train_chars: int):
    """Return (tokenizer, name, ckpt_path).  Raises FileNotFoundError if missing."""
    name_map = {
        "char":    ("CharacterLevel",       "characterlevel_best.pt",       None),
        "bpe_std": ("StandardBPE",          "standardbpe_best.pt",          StandardBPE),
        "bpe_sig": ("SignificanceAwareBPE", "significanceawarebpe_best.pt", SignificanceAwareBPE),
    }
    display_name, ckpt_file, bpe_cls = name_map[model_key]
    ckpt_path = os.path.join("models", ckpt_file)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_key == "char":
        tok = CharLevelTokenizer(text)
    else:
        tok = BPETokenizerWrapper(bpe_cls(), text[:bpe_train_chars], vocab_size)

    return tok, display_name, ckpt_path


def _load_model(ckpt_path: str, vocab_size: int, block_size: int = 128) -> GPTLanguageModel:
    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=128,
        n_head=8,
        n_layer=8,
        dropout=0.0,
    )
    model = GPTLanguageModel(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(DEVICE)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_test_loss(
    model: GPTLanguageModel,
    token_ids: List[int],
    block_size: int,
    batch_size: int,
    n_batches: int,
) -> float:
    """Average cross-entropy loss on a held-out test split."""
    from torch.utils.data import DataLoader
    from shakespeare_gpt_v2 import TokenDataset

    ds     = TokenDataset(token_ids, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    losses: List[float] = []

    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        _, loss = model(x.to(DEVICE), y.to(DEVICE))
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("nan")


def _tokens_per_second(
    model: GPTLanguageModel,
    block_size: int,
    batch_size: int = 8,
    n_warmup: int = 3,
    n_measure: int = 20,
) -> float:
    """Tokens/sec throughput during training-style forward+backward pass."""
    dummy_x = torch.randint(0, model.cfg.vocab_size, (batch_size, block_size), device=DEVICE)
    dummy_y = torch.randint(0, model.cfg.vocab_size, (batch_size, block_size), device=DEVICE)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(n_warmup):
        _, loss = model(dummy_x, dummy_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Sync before timing
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_measure):
        _, loss = model(dummy_x, dummy_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()

    elapsed  = time.perf_counter() - t0
    n_tokens = n_measure * batch_size * block_size
    model.eval()
    return n_tokens / elapsed


def _peak_memory_mb(
    model: GPTLanguageModel,
    block_size: int,
    batch_size: int = 8,
) -> float:
    """
    Peak memory estimate (MB) for inference.

    - CUDA: torch.cuda.max_memory_allocated() — exact device peak.
    - MPS/CPU: model parameters + rough activation estimate.
      (MPS driver pre-allocates aggressively, making delta measurements
      unreliable after the model is already resident; parameter + activation
      accounting is more reproducible and meaningful for model comparison.)
    """
    dummy_x = torch.randint(0, model.cfg.vocab_size, (batch_size, block_size), device=DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(dummy_x)
        return torch.cuda.max_memory_allocated() / 1024 ** 2

    # Parameter memory (exact)
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2

    # Activation memory estimate: float32 × B × T × d_model × n_layer
    # (key, query, value, ff hidden — ~4× multiplier is a reasonable mid-point)
    cfg = model.cfg
    act_mb = (batch_size * block_size * cfg.n_embd * cfg.n_layer * 4 * 4) / 1024 ** 2

    return param_mb + act_mb


@torch.no_grad()
def _generate_sample(
    model: GPTLanguageModel,
    tokenizer,
    prompt: str,
    gen_len: int,
    temperature: float = 0.8,
) -> str:
    """Generate gen_len new tokens starting from prompt and decode."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=gen_len, temperature=temperature)
    return tokenizer.decode(out[0].tolist())

# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_key: str,
    text: str,
    char_total_tokens: int,
    vocab_size: int,
    bpe_train_chars: int,
    block_size: int,
    test_frac: float,
    n_eval_batches: int,
    gen_len: int,
    gen_prompt: str,
) -> Optional[dict]:
    """
    Run all six benchmark metrics for one model.

    Returns a metrics dict, or None if the checkpoint is missing.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model_key}")
    print(f"{'='*60}")

    # ── Load tokenizer + model ───────────────────────────────────────
    try:
        tokenizer, display_name, ckpt_path = _build_tokenizer(
            model_key, text, vocab_size, bpe_train_chars
        )
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    model = _load_model(ckpt_path, tokenizer.vocab_size, block_size)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Vocab size: {tokenizer.vocab_size}  |  Params: {n_params/1e6:.3f}M")

    # ── 1. Sequence length ───────────────────────────────────────────
    print("  [1/6] Encoding corpus ...")
    all_ids   = tokenizer.encode(text)
    total_tok = len(all_ids)
    seq_red   = 1.0 - (total_tok / char_total_tokens) if char_total_tokens else 0.0
    print(f"        {total_tok:,} tokens  ({seq_red*100:+.1f}% vs char)")

    # Train / val / test split  (80 / 10 / 10)
    n_train = int(total_tok * 0.80)
    n_val   = int(total_tok * 0.90)
    test_ids = all_ids[n_val:]

    # ── 2. Training speed ────────────────────────────────────────────
    print("  [2/6] Measuring tokens/sec ...")
    tok_per_sec = _tokens_per_second(model, block_size)
    print(f"        {tok_per_sec:,.0f} tokens/sec (fwd+bwd)")

    # ── 3. Validation loss (from checkpoint) ─────────────────────────
    val_loss_ckpt = float(ckpt.get("val_loss", float("nan")))
    print(f"  [3/6] Val loss (checkpoint):  {val_loss_ckpt:.4f}")

    # ── 4. Test-set perplexity ────────────────────────────────────────
    print("  [4/6] Test perplexity ...")
    if len(test_ids) < block_size + 1:
        test_loss   = float("nan")
        test_ppl    = float("nan")
        print("        Not enough test tokens — skipped")
    else:
        test_loss = _compute_test_loss(
            model, test_ids, block_size,
            batch_size=32, n_batches=n_eval_batches,
        )
        test_ppl  = math.exp(min(test_loss, 20))
        print(f"        test_loss={test_loss:.4f}  perplexity={test_ppl:.3f}")

    # ── 5. Memory usage ──────────────────────────────────────────────
    print("  [5/6] Peak memory ...")
    mem_mb = _peak_memory_mb(model, block_size)
    print(f"        {mem_mb:.1f} MB")

    # ── 6. Generation sample ─────────────────────────────────────────
    print("  [6/6] Generating sample ...")
    sample = _generate_sample(model, tokenizer, gen_prompt, gen_len)
    print(f"        {repr(sample[:80])} ...")

    return {
        "model":               model_key,
        "tokenizer":           display_name,
        "vocab_size":          tokenizer.vocab_size,
        "n_params":            n_params,
        # Metric 1
        "total_tokens":        total_tok,
        "seq_len_reduction":   round(seq_red, 4),
        # Metric 2
        "tokens_per_sec":      round(tok_per_sec, 1),
        # Metric 3
        "val_loss":            round(val_loss_ckpt, 4),
        "val_bits_per_token":  round(val_loss_ckpt / math.log(2), 4),
        # Metric 4
        "test_loss":           round(test_loss, 4) if not math.isnan(test_loss) else "nan",
        "test_perplexity":     round(test_ppl, 3)  if not math.isnan(test_ppl)  else "nan",
        # Metric 5
        "peak_memory_mb":      round(mem_mb, 2),
        # Metric 6
        "sample_text":         sample,
    }

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "model", "tokenizer", "vocab_size", "n_params",
    "total_tokens", "seq_len_reduction",
    "tokens_per_sec",
    "val_loss", "val_bits_per_token",
    "test_loss", "test_perplexity",
    "peak_memory_mb",
]

_TABLE_ROWS = [
    ("Vocab size",             "vocab_size",          lambda v: f"{int(v):,}"),
    ("Parameters",             "n_params",            lambda v: f"{int(v)/1e6:.3f}M"),
    ("Total tokens",           "total_tokens",        lambda v: f"{int(v):,}"),
    ("Seq len reduction",      "seq_len_reduction",   lambda v: f"{float(v)*100:.1f}%"),
    ("Tokens / sec",           "tokens_per_sec",      lambda v: f"{float(v):,.0f}"),
    ("Val loss",               "val_loss",            lambda v: f"{float(v):.4f}"),
    ("Val bits / token",       "val_bits_per_token",  lambda v: f"{float(v):.4f}"),
    ("Test loss",              "test_loss",           lambda v: f"{float(v):.4f}" if v != "nan" else "n/a"),
    ("Test perplexity",        "test_perplexity",     lambda v: f"{float(v):.3f}" if v != "nan" else "n/a"),
    ("Peak memory (MB)",       "peak_memory_mb",      lambda v: f"{float(v):.1f}"),
]


def _print_table(results: List[dict]) -> None:
    lw  = 24
    cw  = 20
    sep = "-" * (lw + cw * len(results))
    print(f"\n{sep}")
    print(f"{'Metric':<{lw}}" + "".join(f"{r['tokenizer']:>{cw}}" for r in results))
    print(sep)
    for label, key, fmt in _TABLE_ROWS:
        row = f"{label:<{lw}}"
        for r in results:
            val = r.get(key, "—")
            try:
                cell = fmt(val)
            except (ValueError, TypeError):
                cell = str(val)
            row += f"{cell:>{cw}}"
        print(row)
    print(sep)


def _save_csv(results: List[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "week2_model_comparison.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved → {path}")


def _save_samples(results: List[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "week2_samples.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{'='*60}\n")
            f.write(f"Model: {r['tokenizer']}  (vocab={r['vocab_size']})\n")
            f.write(f"{'='*60}\n")
            f.write(r.get("sample_text", "") + "\n\n")
    print(f"Saved → {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all 3 Shakespeare GPT models — 6 metrics"
    )
    parser.add_argument(
        "--model", choices=["all", "char", "bpe_std", "bpe_sig"],
        default="all", help="Which model(s) to benchmark (default: all)",
    )
    parser.add_argument("--vocab",          type=int, default=512,
                        help="BPE vocab size used at training time (default: 512)")
    parser.add_argument("--bpe-train-chars", dest="bpe_chars", type=int, default=100_000,
                        help="Chars used to retrain BPE tokenizer (default: 100000)")
    parser.add_argument("--block-size",     type=int, default=128,
                        help="Token context window (default: 128)")
    parser.add_argument("--eval-batches",   type=int, default=50,
                        help="Batches for test-loss estimation (default: 50)")
    parser.add_argument("--gen-len",        type=int, default=200,
                        help="Generation sample length in tokens (default: 200)")
    parser.add_argument("--gen-prompt",     type=str,
                        default="ROMEO:\nWhat light through yonder window breaks?\n",
                        help="Prompt for text generation")
    parser.add_argument("--output-dir",     default="results",
                        help="Output directory for CSV and samples (default: results)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    text = _load_corpus()
    print(f"Corpus: {len(text):,} chars\n")

    char_tok = CharLevelTokenizer(text)
    char_total = len(char_tok.encode(text))
    print(f"Char-level token count (baseline): {char_total:,}")

    keys = (
        ["char", "bpe_std", "bpe_sig"]
        if args.model == "all"
        else [args.model]
    )

    results: List[dict] = []
    for key in keys:
        r = benchmark_model(
            model_key       = key,
            text            = text,
            char_total_tokens = char_total,
            vocab_size      = args.vocab,
            bpe_train_chars = args.bpe_chars,
            block_size      = args.block_size,
            test_frac       = 0.10,
            n_eval_batches  = args.eval_batches,
            gen_len         = args.gen_len,
            gen_prompt      = args.gen_prompt,
        )
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo checkpoints found.  Train models first:")
        print("  python shakespeare_gpt_v2.py")
        return

    _print_table(results)
    _save_csv(results, args.output_dir)
    _save_samples(results, args.output_dir)


if __name__ == "__main__":
    torch.manual_seed(1337)
    main()
