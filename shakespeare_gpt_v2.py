"""
shakespeare_gpt_v2.py — Week 2: BPE tokenizer integration with Shakespeare GPT.

Trains the same Transformer architecture under three tokenization strategies:
  1. CharacterLevel      — stoi/itos baseline (65 vocab, matches shakespeare_gpt.ipynb)
  2. StandardBPE         — frequency-based BPE merges
  3. SignificanceAwareBPE — entropy-weighted BPE merges (novel, from Week 1)

Model architecture (identical to shakespeare_gpt.ipynb):
  8 layers · 128 embed dim · 8 attention heads · block_size = 128

Supports CUDA, MPS (Apple Silicon), and CPU automatically.

Usage:
  python shakespeare_gpt_v2.py                     # train all 3 models
  python shakespeare_gpt_v2.py --model char         # only char-level
  python shakespeare_gpt_v2.py --model bpe_std      # only Standard BPE
  python shakespeare_gpt_v2.py --model bpe_sig      # only Significance-Aware BPE
  python shakespeare_gpt_v2.py --iters 3000         # override max training steps
  python shakespeare_gpt_v2.py --vocab 512          # BPE vocabulary size
  python shakespeare_gpt_v2.py --bpe-train-chars 100000  # chars used to train BPE
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
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from bpe_tokenizer import StandardBPE, SignificanceAwareBPE

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
# Tokenizer wrappers — unified interface
# ---------------------------------------------------------------------------

class CharLevelTokenizer:
    """
    Character-level tokenizer matching shakespeare_gpt.ipynb exactly.
    Vocabulary = sorted unique characters in the training text.
    """
    name = "CharacterLevel"

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)


class BPETokenizerWrapper:
    """Thin wrapper around StandardBPE / SignificanceAwareBPE for a uniform API."""

    def __init__(self, bpe_instance, train_text: str, vocab_size: int) -> None:
        self._tok = bpe_instance
        self.name = type(bpe_instance).__name__
        print(f"  Training {self.name} (vocab={vocab_size}) ...", end=" ", flush=True)
        t0 = time.perf_counter()
        self._tok.train(train_text, vocab_size)
        print(f"done ({time.perf_counter() - t0:.1f}s)")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """Sliding-window dataset over a 1-D token id sequence."""

    def __init__(self, token_ids: List[int], block_size: int) -> None:
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def build_splits(
    tokenizer, text: str, block_size: int, val_frac: float = 0.1
) -> Tuple[TokenDataset, TokenDataset, int]:
    """Tokenise text, split train/val, return datasets and char-level baseline length."""
    ids = tokenizer.encode(text)
    n = int(len(ids) * (1 - val_frac))
    train_ds = TokenDataset(ids[:n], block_size)
    val_ds   = TokenDataset(ids[n:], block_size)
    return train_ds, val_ds, len(ids)

# ---------------------------------------------------------------------------
# Model  (identical to shakespeare_gpt.ipynb)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_embd:     int = 128
    n_head:     int = 8
    n_layer:    int = 8
    dropout:  float = 0.1


class Head(nn.Module):
    def __init__(self, cfg: ModelConfig, head_size: int) -> None:
        super().__init__()
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj  = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.sa   = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.ln2  = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table    = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f   = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(self.ln_f(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) \
               if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    batch_size:    int   = 32
    block_size:    int   = 128
    max_iters:     int   = 5_000
    eval_interval: int   = 250
    eval_iters:    int   = 100
    learning_rate: float = 1e-3
    grad_clip:     float = 1.0
    patience:      int   = 10
    # model shape (kept in sync with ModelConfig at runtime)
    n_embd:  int = 128
    n_head:  int = 8
    n_layer: int = 8
    dropout: float = 0.1

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _estimate_loss(
    model: GPTLanguageModel,
    loader: DataLoader,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= eval_iters:
            break
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_model(
    tokenizer,
    text: str,
    cfg: TrainConfig,
    checkpoint_dir: str,
    char_total_tokens: int,
) -> dict:
    """
    Train one GPT model with the given tokenizer.

    Returns a metrics dict with loss curves and final results.
    """
    tok_name = tokenizer.name
    print(f"\n{'='*60}")
    print(f"  Training: {tok_name}  (vocab={tokenizer.vocab_size})")
    print(f"{'='*60}")

    # ── Data ────────────────────────────────────────────────────────
    train_ds, val_ds, total_tokens = build_splits(tokenizer, text, cfg.block_size)
    seq_len_reduction = 1.0 - (total_tokens / char_total_tokens) if char_total_tokens else 0.0

    print(f"  Tokens    : {total_tokens:,}  ({seq_len_reduction*100:+.1f}% vs char-level)")
    print(f"  Train     : {len(train_ds):,} windows  |  Val: {len(val_ds):,} windows")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    # ── Model ───────────────────────────────────────────────────────
    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
    )
    model = GPTLanguageModel(mcfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_iters)

    # ── Training state ──────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    train_curve: List[Tuple[int, float]] = []
    val_curve:   List[Tuple[int, float]] = []
    best_ckpt_path = os.path.join(checkpoint_dir, f"{tok_name.lower()}_best.pt")

    train_iter = iter(train_loader)
    t_start = time.perf_counter()

    for step in range(cfg.max_iters):
        # ── Eval ────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            t_loss = _estimate_loss(model, train_loader, cfg.eval_iters, DEVICE)
            v_loss = _estimate_loss(model, val_loader,   cfg.eval_iters, DEVICE)
            elapsed = time.perf_counter() - t_start
            train_curve.append((step, t_loss))
            val_curve.append((step, v_loss))

            improved = v_loss < best_val_loss
            if improved:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save({"step": step, "model_state": model.state_dict(),
                            "val_loss": v_loss, "tokenizer_name": tok_name},
                           best_ckpt_path)
            else:
                patience_counter += 1

            mark = " ✓" if improved else f" (patience {patience_counter}/{cfg.patience})"
            print(f"  step {step:5d} | train {t_loss:.4f} | val {v_loss:.4f} |"
                  f" {elapsed:6.1f}s{mark}")

            if patience_counter >= cfg.patience:
                print(f"  Early stopping at step {step}.")
                break

        # ── Forward / backward ──────────────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        _, loss = model(x.to(DEVICE), y.to(DEVICE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

    total_time = time.perf_counter() - t_start
    final_val_loss = val_curve[-1][1] if val_curve else float("nan")
    perplexity = math.exp(min(best_val_loss, 20))   # cap to avoid overflow

    print(f"\n  Done in {total_time:.1f}s | best_val_loss={best_val_loss:.4f}"
          f" | perplexity={perplexity:.3f}")
    print(f"  Checkpoint → {best_ckpt_path}")

    return {
        "tokenizer":          tok_name,
        "vocab_size":         tokenizer.vocab_size,
        "total_tokens":       total_tokens,
        "seq_len_reduction":  round(seq_len_reduction, 4),
        "train_time_s":       round(total_time, 2),
        "best_val_loss":      round(best_val_loss, 4),
        "final_val_loss":     round(final_val_loss, 4),
        "perplexity":         round(perplexity, 4),
        "n_params":           n_params,
        "checkpoint":         best_ckpt_path,
        "train_curve":        train_curve,
        "val_curve":          val_curve,
    }

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "tokenizer", "vocab_size", "total_tokens", "seq_len_reduction",
    "train_time_s", "best_val_loss", "final_val_loss", "perplexity", "n_params",
]


def save_comparison_csv(results: List[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "week2_model_comparison.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def print_comparison_table(results: List[dict]) -> None:
    w = 24
    sep = "-" * (w + 12 * (len(results)))
    header = f"{'Metric':<{w}}" + "".join(f"{r['tokenizer']:>12}" for r in results)
    print(f"\n{sep}\n{header}\n{sep}")
    rows = [
        ("Vocab size",          "vocab_size",         lambda v: f"{int(v):,}"),
        ("Total tokens",        "total_tokens",        lambda v: f"{int(v):,}"),
        ("Seq len reduction",   "seq_len_reduction",   lambda v: f"{v*100:.1f}%"),
        ("Train time (s)",      "train_time_s",        lambda v: f"{v:.1f}"),
        ("Best val loss",       "best_val_loss",       lambda v: f"{v:.4f}"),
        ("Perplexity",          "perplexity",          lambda v: f"{v:.3f}"),
    ]
    for label, key, fmt in rows:
        row = f"{label:<{w}}" + "".join(f"{fmt(r[key]):>12}" for r in results)
        print(row)
    print(sep)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Shakespeare GPT with 3 tokenizers")
    parser.add_argument("--model", choices=["all", "char", "bpe_std", "bpe_sig"],
                        default="all", help="Which model(s) to train (default: all)")
    parser.add_argument("--iters",   type=int, default=5_000,   help="Max training steps (default: 5000)")
    parser.add_argument("--vocab",   type=int, default=512,     help="BPE vocab size (default: 512)")
    parser.add_argument("--bpe-train-chars", dest="bpe_chars", type=int, default=100_000,
                        help="Chars used to train BPE tokenizers (default: 100000)")
    parser.add_argument("--batch",   type=int, default=32,      help="Batch size (default: 32)")
    parser.add_argument("--patience",type=int, default=10,      help="Early-stopping patience (default: 10)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # ── Corpus ──────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "input.txt")
    if not os.path.exists(data_path):
        print("Downloading Tiny Shakespeare ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            data_path,
        )
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} chars")

    os.makedirs("models", exist_ok=True)

    cfg = TrainConfig(
        max_iters=args.iters,
        batch_size=args.batch,
        patience=args.patience,
    )

    # ── Build tokenizers ────────────────────────────────────────────────────
    print("\nInitialising tokenizers ...")
    bpe_train_text = text[: args.bpe_chars]

    tokenizers = []
    if args.model in ("all", "char"):
        tokenizers.append(CharLevelTokenizer(text))
    if args.model in ("all", "bpe_std"):
        tokenizers.append(BPETokenizerWrapper(StandardBPE(), bpe_train_text, args.vocab))
    if args.model in ("all", "bpe_sig"):
        tokenizers.append(BPETokenizerWrapper(SignificanceAwareBPE(), bpe_train_text, args.vocab))

    # ── Char-level token count (baseline for seq_len_reduction) ─────────────
    char_tok = CharLevelTokenizer(text)
    char_total = len(char_tok.encode(text))

    # ── Train ───────────────────────────────────────────────────────────────
    all_results = []
    for tok in tokenizers:
        result = train_model(tok, text, cfg, "models", char_total)
        all_results.append(result)

    # ── Compare & save ──────────────────────────────────────────────────────
    if len(all_results) > 1:
        print_comparison_table(all_results)
    save_comparison_csv(all_results, "results")


if __name__ == "__main__":
    torch.manual_seed(1337)
    main()
