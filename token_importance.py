"""
token_importance.py — Token importance analysis for trained GPT models.

Three complementary importance signals:
  1. Frequency  — normalized occurrence rate in the corpus
  2. Attention  — average attention weight received across all heads and layers
  3. Gradient   — gradient norm of the token embedding w.r.t. model loss

All three compute_* methods return {token_id: score} normalized to [0, 1]
so scores are directly comparable across methods.

Usage
-----
  from token_importance import TokenImportanceAnalyzer
  from shakespeare_gpt_v2 import GPTLanguageModel, ModelConfig, CharLevelTokenizer

  tokenizer = CharLevelTokenizer(text)
  ids = tokenizer.encode(text)
  # ... build model, load checkpoint ...

  analyzer = TokenImportanceAnalyzer(tokenizer.vocab_size)
  freq_scores = analyzer.compute_frequency_importance(ids)
  attn_scores = analyzer.compute_attention_importance(model, ids)
  grad_scores = analyzer.compute_gradient_importance(model, ids)

  top20 = analyzer.get_top_k_tokens(k=20)
  info  = analyzer.analyze_token(42)
"""

from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class TokenImportanceAnalyzer:
    """
    Analyses token importance for a trained GPT model under three lenses.

    Typical workflow:
      1. Call each compute_* method once.
      2. Use analyze_token / get_top_k_tokens / aggregate_importance.

    The analyzer does NOT modify the model permanently.  Attention heads are
    monkey-patched for the duration of compute_attention_importance and then
    restored to their original forward methods.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialise the analyzer for a vocabulary of the given size.

        Args:
            vocab_size: Number of tokens in the model vocabulary.  Must be > 0.

        Raises:
            ValueError: If vocab_size is not a positive integer.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0; got {vocab_size}")
        self.vocab_size = vocab_size
        self._freq_scores: Dict[int, float] = {}
        self._attn_scores: Dict[int, float] = {}
        self._grad_scores: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # 1. Frequency importance
    # ------------------------------------------------------------------

    def compute_frequency_importance(self, data: List[int]) -> Dict[int, float]:
        """Score each token by normalized occurrence frequency.

        score(tok) = count(tok) / max_count_in_corpus, normalized to [0, 1].
        Scores are also stored internally for use by analyze_token /
        get_top_k_tokens.

        Args:
            data: Flat list of encoded token ids representing the corpus.

        Returns:
            Dict mapping token_id → frequency score in [0, 1].
            Returns an empty dict if data is empty.
        """
        if not data:
            self._freq_scores = {}
            return {}

        counts = Counter(data)
        max_count = max(counts.values())
        self._freq_scores = {tok: cnt / max_count for tok, cnt in counts.items()}
        return dict(self._freq_scores)

    # ------------------------------------------------------------------
    # 2. Attention importance
    # ------------------------------------------------------------------

    def compute_attention_importance(
        self,
        model: nn.Module,
        data: List[int],
        batch_size: int = 32,
        n_batches: int = 20,
    ) -> Dict[int, float]:
        """
        Score each token by the average attention weight it receives.

        For every attention head in every layer, the (T × T) attention
        probability matrix is captured.  Column sums give "total attention
        received" per position; results are accumulated by token ID and then
        averaged and normalised to [0, 1].

        Attention heads are temporarily monkey-patched and restored afterward.

        Args:
            model:      Trained GPTLanguageModel (must expose .cfg.block_size
                        and .blocks[*].sa.heads[*] with a Head-like forward).
            data:       Flat list of encoded token IDs.
            batch_size: Sequences per forward pass.
            n_batches:  Maximum number of batches to process.

        Returns:
            Dict mapping token_id → attention score in [0, 1].
            Returns an empty dict if data is too short for even one window.

        Raises:
            ValueError: If model does not expose .cfg.block_size.
        """
        block_size = self._get_block_size(model)
        device = next(model.parameters()).device
        windows = self._input_windows(data, block_size)
        if not windows:
            self._attn_scores = {}
            return {}

        attn_sum = torch.zeros(self.vocab_size)
        attn_cnt = torch.zeros(self.vocab_size, dtype=torch.long)

        # Collect all Head instances and save their original forward methods
        heads, original_fwds = [], []
        for block in model.blocks:
            for head in block.sa.heads:
                heads.append(head)
                original_fwds.append(head.forward)

        captured: List[torch.Tensor] = []

        def _make_patched(h):
            def patched(x):
                B, T, _ = x.shape
                k_proj = h.key(x)
                q_proj = h.query(x)
                wei = q_proj @ k_proj.transpose(-2, -1) * (h.head_size ** -0.5)
                wei = wei.masked_fill(h.tril[:T, :T] == 0, float("-inf"))
                wei = F.softmax(wei, dim=-1)        # (B, T, T)
                captured.append(wei.detach().cpu())
                wei = h.dropout(wei)
                return wei @ h.value(x)
            return patched

        for head in heads:
            head.forward = _make_patched(head)

        model.eval()
        try:
            for batch_start in range(0, min(n_batches * batch_size, len(windows)), batch_size):
                batch = windows[batch_start : batch_start + batch_size]
                x = torch.stack(
                    [torch.tensor(w, dtype=torch.long) for w in batch]
                ).to(device)                             # (B, T)

                captured.clear()
                with torch.no_grad():
                    model(x)

                tok_ids = x.cpu()                        # (B, T) long
                for wei in captured:                     # (B, T, T)
                    # column sum: total attention received by each key position
                    received = wei.sum(dim=1)            # (B, T)
                    flat_ids  = tok_ids.view(-1)         # (B*T,)
                    flat_recv = received.view(-1)        # (B*T,)
                    attn_sum.scatter_add_(0, flat_ids, flat_recv)
                    attn_cnt.scatter_add_(0, flat_ids, torch.ones_like(flat_ids))
        finally:
            for head, orig in zip(heads, original_fwds):
                head.forward = orig

        mask = attn_cnt > 0
        avg = torch.where(
            mask,
            attn_sum / attn_cnt.float().clamp(min=1),
            torch.zeros(self.vocab_size),
        )
        max_val = avg.max().item()
        if max_val == 0:
            self._attn_scores = {}
            return {}

        self._attn_scores = {
            i: avg[i].item() / max_val
            for i in range(self.vocab_size)
            if attn_cnt[i] > 0
        }
        return dict(self._attn_scores)

    # ------------------------------------------------------------------
    # 3. Gradient importance
    # ------------------------------------------------------------------

    def compute_gradient_importance(
        self,
        model: nn.Module,
        data: List[int],
        epsilon: float = 0.01,
        batch_size: int = 16,
        n_batches: int = 20,
    ) -> Dict[int, float]:
        """
        Score each token by the L2 norm of ∂Loss/∂embedding accumulated
        across batches.

        For each mini-batch a forward + backward pass is run.  The gradient
        of the cross-entropy loss w.r.t. the token embedding table is read
        and accumulated.  The final per-token norm is normalised to [0, 1].

        epsilon is added to every score before normalisation so near-zero
        gradients (rare but present tokens) are not completely masked out.

        Args:
            model:      Trained GPTLanguageModel.
            data:       Flat list of encoded token IDs.
            epsilon:    Smoothing constant added to all raw scores before
                        normalisation.  Prevents zero-gradient tokens from
                        being invisible relative to high-gradient tokens.
            batch_size: Sequences per gradient accumulation step.
            n_batches:  Maximum number of batches to process.

        Returns:
            Dict mapping token_id → gradient importance score in [0, 1].
            Only tokens that appear in data are included.

        Raises:
            ValueError: If model does not expose .cfg.block_size.
        """
        block_size = self._get_block_size(model)
        device = next(model.parameters()).device
        windows = self._io_windows(data, block_size)
        if not windows:
            self._grad_scores = {}
            return {}

        emb = model.token_embedding_table
        grad_accum = torch.zeros(self.vocab_size, emb.embedding_dim)

        model.eval()          # dropout off for deterministic results
        model.zero_grad()

        for batch_start in range(0, min(n_batches * batch_size, len(windows)), batch_size):
            batch = windows[batch_start : batch_start + batch_size]
            x = torch.stack(
                [torch.tensor(w[:-1], dtype=torch.long) for w in batch]
            ).to(device)
            y = torch.stack(
                [torch.tensor(w[1:],  dtype=torch.long) for w in batch]
            ).to(device)

            model.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            if emb.weight.grad is not None:
                grad_accum += emb.weight.grad.detach().cpu()

        model.zero_grad()

        # Per-token L2 norm of accumulated embedding gradient
        grad_norms = grad_accum.norm(dim=1)   # (vocab_size,)

        seen = set(data)
        raw = {
            tok: grad_norms[tok].item() + epsilon
            for tok in range(self.vocab_size)
            if tok in seen
        }
        if not raw:
            self._grad_scores = {}
            return {}

        max_val = max(raw.values())
        self._grad_scores = {tok: v / max_val for tok, v in raw.items()}
        return dict(self._grad_scores)

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze_token(self, token_id: int) -> Dict[str, float]:
        """Return all three importance scores for a single token.

        Args:
            token_id: Integer token id to look up.

        Returns:
            Dict with keys ``"frequency"``, ``"attention"``, ``"gradient"``,
            each mapped to a float in [0, 1].  Returns 0.0 for any method
            whose compute_* has not yet been called.
        """
        return {
            "frequency": self._freq_scores.get(token_id, 0.0),
            "attention": self._attn_scores.get(token_id, 0.0),
            "gradient":  self._grad_scores.get(token_id, 0.0),
        }

    def get_top_k_tokens(self, k: int = 20) -> List[Dict]:
        """Return the top-k tokens ranked by aggregate importance (descending).

        Aggregates the three stored score dicts using equal weights (1/3 each).
        Call compute_*_importance methods first to populate the scores.

        Args:
            k: Number of top tokens to return.  Must be >= 1.

        Returns:
            List of dicts sorted by aggregate score (descending), each with
            keys: ``token_id``, ``frequency``, ``attention``, ``gradient``,
            ``aggregate``.  Returns an empty list if no scores have been
            computed yet.
        """
        combined = self.aggregate_importance(
            self._freq_scores, self._attn_scores, self._grad_scores
        )
        if not combined:
            return []

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            {
                "token_id":  tid,
                "frequency": self._freq_scores.get(tid, 0.0),
                "attention": self._attn_scores.get(tid, 0.0),
                "gradient":  self._grad_scores.get(tid, 0.0),
                "aggregate": score,
            }
            for tid, score in ranked
        ]

    def aggregate_importance(
        self,
        freq: Dict[int, float],
        attn: Dict[int, float],
        grad: Dict[int, float],
        weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ) -> Dict[int, float]:
        """
        Weighted linear combination of three importance score dicts.

        Args:
            freq, attn, grad: {token_id: score} from compute_*_importance.
            weights:          (w_freq, w_attn, w_grad) — should sum to 1.

        Returns:
            {token_id: aggregate_score} for the union of all three dicts.
        """
        w_f, w_a, w_g = weights
        all_ids = set(freq) | set(attn) | set(grad)
        return {
            tid: w_f * freq.get(tid, 0.0)
               + w_a * attn.get(tid, 0.0)
               + w_g * grad.get(tid, 0.0)
            for tid in all_ids
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_block_size(model: nn.Module) -> int:
        try:
            return model.cfg.block_size
        except AttributeError:
            raise ValueError("model must expose .cfg.block_size")

    @staticmethod
    def _input_windows(data: List[int], block_size: int) -> List[List[int]]:
        """Sliding windows of exactly block_size tokens, stride = block_size // 2."""
        stride = max(1, block_size // 2)
        return [
            data[i : i + block_size]
            for i in range(0, len(data) - block_size + 1, stride)
        ]

    @staticmethod
    def _io_windows(data: List[int], block_size: int) -> List[List[int]]:
        """Sliding windows of block_size + 1 tokens for (input, target) pairs."""
        win = block_size + 1
        stride = max(1, block_size // 2)
        return [
            data[i : i + win]
            for i in range(0, len(data) - win + 1, stride)
        ]
