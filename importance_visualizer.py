"""
importance_visualizer.py — 5 token importance plots for Week 2.

Loads a trained checkpoint, runs TokenImportanceAnalyzer, and saves 5
publication-ready plots to analysis/week2/.

Plots
-----
  1. plot1_top20_bar.png/pdf      — top-20 tokens grouped bar (freq / attn / grad)
  2. plot2_layer_heatmap.png/pdf  — attention heatmap: top tokens × transformer layers
  3. plot3_distribution.png/pdf   — importance score distribution (overlapping histograms)
  4. plot4_model_comparison.png/pdf — model comparison from week2_model_comparison.csv
  5. plot5_method_agreement.png/pdf — parallel-coord profile of top-10 tokens

Usage
-----
  python importance_visualizer.py                    # char-level model (default)
  python importance_visualizer.py --model bpe_std    # Standard BPE
  python importance_visualizer.py --model bpe_sig    # Significance-Aware BPE
  python importance_visualizer.py --vocab 512 --top-k 20 --sample 30000
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from token_importance import TokenImportanceAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("analysis", "week2")

COLORS = {
    "frequency": "#2196F3",
    "attention":  "#FF5722",
    "gradient":   "#4CAF50",
    "aggregate":  "#9C27B0",
}
METHOD_ORDER = ["frequency", "attention", "gradient"]
LABELS = {
    "frequency": "Frequency",
    "attention":  "Attention",
    "gradient":   "Gradient",
    "aggregate":  "Aggregate",
}
MODEL_DISPLAY = {
    "CharacterLevel":      "Char-Level",
    "StandardBPE":         "Standard BPE",
    "SignificanceAwareBPE": "Sig-Aware BPE",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})

# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def _load_corpus() -> str:
    import urllib.request
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


def _load_model_and_tokenizer(model_key: str, text: str, vocab_size: int, bpe_train_chars: int):
    """
    Reconstruct tokenizer + model from a saved checkpoint.

    Returns (model, tokenizer, device).
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    from shakespeare_gpt_v2 import (
        CharLevelTokenizer, BPETokenizerWrapper,
        GPTLanguageModel, ModelConfig,
    )
    from bpe_tokenizer import StandardBPE, SignificanceAwareBPE

    name_map = {
        "char":    ("characterlevel",      None),
        "bpe_std": ("standardbpe",         StandardBPE),
        "bpe_sig": ("significanceawarebpe", SignificanceAwareBPE),
    }
    file_stem, bpe_cls = name_map[model_key]
    ckpt_path = os.path.join("models", f"{file_stem}_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train it first:  python shakespeare_gpt_v2.py --model {model_key}"
        )

    if model_key == "char":
        tokenizer = CharLevelTokenizer(text)
    else:
        tokenizer = BPETokenizerWrapper(bpe_cls(), text[:bpe_train_chars], vocab_size)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_embd=128,
        n_head=8,
        n_layer=8,
        dropout=0.0,
    )
    model = GPTLanguageModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cpu")    # CPU for analysis — deterministic, no OOM risk
    return model.to(device), tokenizer


# ---------------------------------------------------------------------------
# Token label helpers
# ---------------------------------------------------------------------------

def _token_label(token_id: int, tokenizer) -> str:
    """Return a short printable label for a token (max 6 chars)."""
    try:
        if hasattr(tokenizer, "itos"):                        # CharLevelTokenizer
            raw = tokenizer.itos.get(token_id, f"#{token_id}")
            raw = raw.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "·")
            return raw[:6]
        else:                                                 # BPETokenizerWrapper
            vocab = tokenizer._tok.get_vocab()
            b = vocab.get(token_id, b"")
            try:
                s = b.decode("utf-8")
            except UnicodeDecodeError:
                s = f"<{token_id}>"
            s = s.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "·")
            return s[:6] or f"<{token_id}>"
    except Exception:
        return f"#{token_id}"


# ---------------------------------------------------------------------------
# Per-layer attention helper
# ---------------------------------------------------------------------------

def _layerwise_attention(
    model: nn.Module,
    data: List[int],
    vocab_size: int,
    batch_size: int = 32,
    n_batches: int = 10,
) -> Dict[int, Dict[int, float]]:
    """
    Compute per-layer attention importance.

    Returns {layer_idx: {token_id: normalised_score}} where scores are [0, 1]
    within each layer.  Patches each Head in order (block0-head0, …) and
    groups captures by floor-dividing capture index by n_heads_per_layer.
    """
    block_size = model.cfg.block_size
    n_layers   = model.cfg.n_layer
    n_heads    = model.cfg.n_head
    device     = next(model.parameters()).device

    windows = TokenImportanceAnalyzer._input_windows(data, block_size)
    if not windows:
        return {}

    layer_sums = [torch.zeros(vocab_size) for _ in range(n_layers)]
    layer_cnts = [torch.zeros(vocab_size, dtype=torch.long) for _ in range(n_layers)]

    heads, original_fwds, captured = [], [], []
    for block in model.blocks:
        for head in block.sa.heads:
            heads.append(head)
            original_fwds.append(head.forward)

    def _make_patched(h):
        def patched(x):
            B, T, _ = x.shape
            k = h.key(x)
            q = h.query(x)
            wei = q @ k.transpose(-2, -1) * (h.head_size ** -0.5)
            wei = wei.masked_fill(h.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            captured.append(wei.detach().cpu())
            wei = h.dropout(wei)
            return wei @ h.value(x)
        return patched

    for head in heads:
        head.forward = _make_patched(head)

    try:
        for batch_start in range(0, min(n_batches * batch_size, len(windows)), batch_size):
            batch  = windows[batch_start : batch_start + batch_size]
            x      = torch.stack([torch.tensor(w, dtype=torch.long) for w in batch]).to(device)
            tok_ids = x.cpu()

            captured.clear()
            with torch.no_grad():
                model(x)

            for cap_idx, wei in enumerate(captured):
                layer_idx = cap_idx // n_heads          # groups: n_layers groups of n_heads each
                received  = wei.sum(dim=1)              # (B, T)
                flat_ids  = tok_ids.view(-1)
                flat_recv = received.view(-1)
                layer_sums[layer_idx].scatter_add_(0, flat_ids, flat_recv)
                layer_cnts[layer_idx].scatter_add_(0, flat_ids, torch.ones_like(flat_ids))
    finally:
        for head, orig in zip(heads, original_fwds):
            head.forward = orig

    result: Dict[int, Dict[int, float]] = {}
    for l in range(n_layers):
        mask  = layer_cnts[l] > 0
        avg   = torch.where(mask, layer_sums[l] / layer_cnts[l].float().clamp(min=1),
                            torch.zeros(vocab_size))
        mx    = avg.max().item()
        result[l] = (
            {i: avg[i].item() / mx for i in range(vocab_size) if layer_cnts[l][i] > 0}
            if mx > 0 else {}
        )
    return result


# ---------------------------------------------------------------------------
# Plot 1 — Top-20 tokens grouped bar chart
# ---------------------------------------------------------------------------

def plot1_top20_bar(
    analyzer: TokenImportanceAnalyzer,
    tokenizer,
    top_k: int = 20,
    output_dir: str = OUTPUT_DIR,
) -> None:
    top = analyzer.get_top_k_tokens(k=top_k)
    if not top:
        print("  [plot1] No tokens to plot.")
        return

    labels = [_token_label(t["token_id"], tokenizer) for t in top]
    n = len(top)
    y = np.arange(n)
    h = 0.25

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.35)))

    for i, method in enumerate(METHOD_ORDER):
        vals = [t[method] for t in top]
        bars = ax.barh(
            y + (i - 1) * h, vals, h,
            label=LABELS[method],
            color=COLORS[method],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
        )

    # Aggregate score as a step line
    agg_vals = [t["aggregate"] for t in top]
    ax.step(agg_vals, y, color=COLORS["aggregate"], linewidth=1.5,
            linestyle="--", label="Aggregate", where="mid", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score (normalised to [0, 1])")
    ax.set_title(f"Top-{n} Tokens by Aggregate Importance — all three methods")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "plot1_top20_bar", output_dir)


# ---------------------------------------------------------------------------
# Plot 2 — Layer-wise attention heatmap
# ---------------------------------------------------------------------------

def plot2_layer_heatmap(
    model: nn.Module,
    analyzer: TokenImportanceAnalyzer,
    tokenizer,
    data: List[int],
    top_k: int = 20,
    output_dir: str = OUTPUT_DIR,
) -> None:
    top = analyzer.get_top_k_tokens(k=top_k)
    if not top:
        print("  [plot2] No tokens to plot.")
        return

    top_ids = [t["token_id"] for t in top]
    n_layers = model.cfg.n_layer

    layer_attn = _layerwise_attention(
        model, data, analyzer.vocab_size, batch_size=32, n_batches=10
    )

    # Build matrix: rows = tokens (top_k), cols = layers
    matrix = np.zeros((len(top_ids), n_layers))
    for col, l in enumerate(range(n_layers)):
        layer_scores = layer_attn.get(l, {})
        for row, tid in enumerate(top_ids):
            matrix[row, col] = layer_scores.get(tid, 0.0)

    labels = [_token_label(tid, tokenizer) for tid in top_ids]

    fig, ax = plt.subplots(figsize=(11, max(5, len(top_ids) * 0.38)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{l}" for l in range(n_layers)])
    ax.set_yticks(range(len(top_ids)))
    ax.set_yticklabels(labels, fontfamily="monospace")
    ax.set_xlabel("Transformer Layer")
    ax.set_title(f"Layer-wise Attention Importance — top-{len(top_ids)} tokens")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalised attention received", fontsize=9)

    # Annotate cells
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            color = "white" if val > 0.6 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    _save(fig, "plot2_layer_heatmap", output_dir)


# ---------------------------------------------------------------------------
# Plot 3 — Importance distribution histogram
# ---------------------------------------------------------------------------

def plot3_distribution(
    analyzer: TokenImportanceAnalyzer,
    output_dir: str = OUTPUT_DIR,
) -> None:
    score_dicts = {
        "frequency": analyzer._freq_scores,
        "attention":  analyzer._attn_scores,
        "gradient":   analyzer._grad_scores,
    }
    all_empty = all(len(d) == 0 for d in score_dicts.values())
    if all_empty:
        print("  [plot3] No scores to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, method in zip(axes, METHOD_ORDER):
        scores = list(score_dicts[method].values())
        if not scores:
            ax.set_visible(False)
            continue

        arr = np.array(scores)
        n_bins = min(30, max(5, len(arr) // 3))

        ax.hist(arr, bins=n_bins, color=COLORS[method], alpha=0.7,
                edgecolor="white", linewidth=0.5, density=True)

        # KDE overlay
        if len(arr) > 3:
            from scipy.stats import gaussian_kde  # soft-dep; falls back gracefully
            try:
                kde = gaussian_kde(arr, bw_method="scott")
                xs  = np.linspace(0, 1, 200)
                ax.plot(xs, kde(xs), color=COLORS[method], linewidth=2)
            except Exception:
                pass

        mean_val  = np.mean(arr)
        ax.axvline(mean_val, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(mean_val + 0.02, ax.get_ylim()[1] * 0.95,
                f"μ={mean_val:.2f}", fontsize=8, va="top")

        ax.set_title(LABELS[method])
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Density" if ax == axes[0] else "")
        ax.set_xlim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    fig.suptitle("Importance Score Distributions by Method", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "plot3_distribution", output_dir)


# ---------------------------------------------------------------------------
# Plot 4 — Model comparison (from CSV)
# ---------------------------------------------------------------------------

def plot4_model_comparison(
    csv_path: str = os.path.join("results", "week2_model_comparison.csv"),
    output_dir: str = OUTPUT_DIR,
) -> None:
    if not os.path.exists(csv_path):
        print(f"  [plot4] CSV not found: {csv_path}")
        return

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("  [plot4] Empty CSV.")
        return

    names     = [MODEL_DISPLAY.get(r["tokenizer"], r["tokenizer"]) for r in rows]
    metrics   = {
        "Seq Len Reduction (%)": [float(r["seq_len_reduction"]) * 100 for r in rows],
        "Best Val Loss":         [float(r["best_val_loss"]) for r in rows],
        "Perplexity":            [float(r["perplexity"]) for r in rows],
    }
    palette = ["#2196F3", "#FF5722", "#4CAF50"][:len(rows)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (title, vals) in zip(axes, metrics.items()):
        bars = ax.bar(names, vals, color=palette, width=0.55,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_ylim(bottom=0)

    fig.suptitle("Model Comparison across Tokenization Strategies", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "plot4_model_comparison", output_dir)


# ---------------------------------------------------------------------------
# Plot 5 — Method agreement for top-10 tokens (parallel coordinates)
# ---------------------------------------------------------------------------

def plot5_method_agreement(
    analyzer: TokenImportanceAnalyzer,
    tokenizer,
    top_k: int = 10,
    output_dir: str = OUTPUT_DIR,
) -> None:
    top = analyzer.get_top_k_tokens(k=top_k)
    if not top:
        print("  [plot5] No tokens to plot.")
        return

    # Colour ramp from purple (rank 1) to teal (rank top_k)
    cmap   = plt.cm.plasma
    n      = len(top)
    x_pos  = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(8, 5))

    for rank, entry in enumerate(top):
        color  = cmap(rank / max(n - 1, 1))
        label  = _token_label(entry["token_id"], tokenizer)
        scores = [entry[m] for m in METHOD_ORDER]
        ax.plot(x_pos, scores, "-o", color=color, linewidth=1.6,
                markersize=5, alpha=0.85, label=f"#{rank+1} '{label}'")

    # Mark the aggregate centroid
    agg_scores = [
        np.mean([t[m] for t in top]) for m in METHOD_ORDER
    ]
    ax.plot(x_pos, agg_scores, "k--", linewidth=2, label="Mean (top tokens)", zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Importance Score (normalised [0, 1])")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Importance Method Agreement — top-{n} tokens")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7.5,
              framealpha=0.85, borderpad=0.7)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "plot5_method_agreement", output_dir)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, stem: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5 token importance plots for Week 2"
    )
    parser.add_argument(
        "--model", choices=["char", "bpe_std", "bpe_sig"], default="char",
        help="Which trained model to analyse (default: char)"
    )
    parser.add_argument("--vocab",   type=int, default=512,
                        help="BPE vocab size used during training (default: 512)")
    parser.add_argument("--top-k",   type=int, default=20, dest="top_k",
                        help="Number of top tokens for plots 1, 2, 5 (default: 20)")
    parser.add_argument("--sample",  type=int, default=30_000,
                        help="Chars of corpus to use for importance analysis (default: 30000)")
    parser.add_argument("--bpe-train-chars", dest="bpe_chars", type=int, default=100_000,
                        help="Chars used to re-train BPE tokenizer (default: 100000)")
    parser.add_argument("--n-batches", type=int, default=20, dest="n_batches",
                        help="Importance estimation batches (default: 20)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    print(f"\nLoading corpus ...")
    text = _load_corpus()
    analysis_text = text[:args.sample]
    print(f"  Corpus: {len(text):,} chars total, using first {len(analysis_text):,} for analysis")

    print(f"\nLoading model '{args.model}' ...")
    model, tokenizer = _load_model_and_tokenizer(
        args.model, text, args.vocab, args.bpe_chars
    )
    print(f"  Vocab size: {tokenizer.vocab_size}")

    ids = tokenizer.encode(analysis_text)
    print(f"  Encoded {len(ids):,} tokens for analysis")

    print("\nComputing importance scores ...")
    analyzer = TokenImportanceAnalyzer(tokenizer.vocab_size)

    print("  [1/3] Frequency ...")
    analyzer.compute_frequency_importance(ids)

    print("  [2/3] Attention ...")
    analyzer.compute_attention_importance(
        model, ids, batch_size=32, n_batches=args.n_batches
    )

    print("  [3/3] Gradient ...")
    analyzer.compute_gradient_importance(
        model, ids, epsilon=0.01, batch_size=16, n_batches=args.n_batches
    )

    top5 = analyzer.get_top_k_tokens(k=5)
    print(f"\n  Top-5 tokens (aggregate): "
          f"{[_token_label(t['token_id'], tokenizer) for t in top5]}")

    print(f"\nGenerating plots → {args.output_dir}")

    print("  [1/5] Top-20 grouped bar ...")
    plot1_top20_bar(analyzer, tokenizer, top_k=args.top_k, output_dir=args.output_dir)

    print("  [2/5] Layer-wise heatmap ...")
    plot2_layer_heatmap(
        model, analyzer, tokenizer, ids,
        top_k=min(args.top_k, 20), output_dir=args.output_dir
    )

    print("  [3/5] Distribution histogram ...")
    plot3_distribution(analyzer, output_dir=args.output_dir)

    print("  [4/5] Model comparison ...")
    plot4_model_comparison(output_dir=args.output_dir)

    print("  [5/5] Method agreement ...")
    plot5_method_agreement(
        analyzer, tokenizer, top_k=min(args.top_k, 10), output_dir=args.output_dir
    )

    print(f"\nDone. All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
