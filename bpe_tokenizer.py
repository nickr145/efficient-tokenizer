"""
bpe_tokenizer.py — BPE tokenizer variants for the Efficient Tokenizer project.

Two variants:
  StandardBPE          — vanilla frequency-based BPE (baseline)
  SignificanceAwareBPE — entropy-weighted BPE; prefers merges that reduce
                         bits-per-token most (novel, from "Not All Tokens Matter")

Both start from a 256-token byte vocabulary and operate on UTF-8 byte sequences.

Entropy math (used in SignificanceAwareBPE):
  H = log2(N) − L/N,  where L = Σ c_i · log2(c_i)
  After merging pair (a,b) with frequency f, L and N update in O(1) by adjusting
  only the three affected token-count terms — no full re-scan of the sequence.
"""

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Tokenizer(ABC):
    """Shared interface for all BPE tokenizer variants."""

    @abstractmethod
    def train(self, text: str, vocab_size: int) -> None:
        """Build vocabulary from text up to vocab_size tokens."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token ids."""

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Reconstruct text from a list of token ids."""

    @abstractmethod
    def get_vocab(self) -> Dict[int, bytes]:
        """Return {token_id: bytes} mapping for the full vocabulary."""

    @abstractmethod
    def get_merge_history(self) -> List[dict]:
        """Return metadata for every merge performed during training."""

    def __repr__(self) -> str:
        v = getattr(self, "vocab", {})
        return f"{self.__class__.__name__}(vocab_size={len(v)}, trained={len(v) > 0})"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _get_pairs(ids: List[int]) -> Counter:
    """Count every consecutive pair in ids."""
    return Counter(zip(ids, ids[1:]))


def _merge_ids(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Replace all non-overlapping left-to-right occurrences of pair with new_id.
    """
    result: List[int] = []
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    if i < len(ids):
        result.append(ids[i])
    return result


# ---------------------------------------------------------------------------
# Standard BPE
# ---------------------------------------------------------------------------

class StandardBPE(Tokenizer):
    """
    Vanilla byte-pair encoding.

    At each training step, the most-frequent consecutive pair is merged.
    Serves as the quantitative baseline against SignificanceAwareBPE.
    """

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self._merge_history: List[dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train on text until the vocabulary reaches vocab_size tokens.

        Args:
            text: Raw training corpus.
            vocab_size: Target vocabulary size (must be >= 256).
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte vocab floor)"

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self._merge_history = []

        ids: List[int] = list(text.encode("utf-8"))

        for step in range(vocab_size - 256):
            pairs = _get_pairs(ids)
            if not pairs:
                break

            best = max(pairs, key=lambda p: (pairs[p], p))
            freq = pairs[best]
            new_id = 256 + step

            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self._merge_history.append(
                {
                    "step": step,
                    "pair": best,
                    "frequency": freq,
                    "new_token_id": new_id,
                    "new_token": self.vocab[new_id],
                }
            )
            ids = _merge_ids(ids, best, new_id)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = _merge_ids(ids, pair, new_id)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        raw = b"".join(self.vocab[t] for t in token_ids)
        return raw.decode("utf-8", errors="replace")

    def get_vocab(self) -> Dict[int, bytes]:
        return dict(self.vocab)

    def get_merge_history(self) -> List[dict]:
        return list(self._merge_history)


# ---------------------------------------------------------------------------
# Significance-Aware BPE
# ---------------------------------------------------------------------------

class SignificanceAwareBPE(Tokenizer):
    """
    Entropy-weighted BPE.

    Selects each merge by significance_score = entropy_reduction × frequency,
    where entropy_reduction = H_before − H_after (bits per token saved).

    The incremental entropy formula lets us score every candidate pair in O(1)
    without re-scanning the sequence, so per-step cost is the same order as
    StandardBPE (dominated by the O(n) pair-counting and merge passes).

    Args:
        entropy_weight: Blend factor in [0.0, 1.0].
            1.0 → pure significance score  (default — the novel mode)
            0.0 → pure frequency           (degenerates to StandardBPE)
            0 < w < 1 → normalised blend: both terms are scaled to [0, 1]
                        within each step, so the weight is a true proportion.
    """

    def __init__(self, entropy_weight: float = 1.0) -> None:
        assert 0.0 <= entropy_weight <= 1.0, "entropy_weight must be in [0, 1]"
        self.entropy_weight = entropy_weight
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self._merge_history: List[dict] = []

    # ------------------------------------------------------------------
    # Entropy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_L(counts: Counter) -> float:
        """L = Σ c_i · log2(c_i).  Building block for O(1) entropy updates."""
        return sum(c * math.log2(c) for c in counts.values() if c > 0)

    @staticmethod
    def _H_from_L(L: float, N: int) -> float:
        """Shannon entropy H = log2(N) − L/N  (bits per token)."""
        return math.log2(N) - L / N if N > 0 else 0.0

    def _H_after_merge(
        self,
        pair: Tuple[int, int],
        freq: int,
        counts: Counter,
        N: int,
        L: float,
    ) -> float:
        """
        Compute bits-per-token entropy after merging pair, in O(1).

        Updates L by removing the old contribution of each affected token and
        adding the updated contribution, then derives H from the new L and N.

        For the edge-case pair (a, a): each non-overlapping merge consumes two
        copies of a, so c_a decreases by 2·freq rather than freq.
        Note: pair-counter may overcount overlapping self-pairs (e.g., 'aaa' → 2
        pairs but only 1 merge), so entropy reduction for self-pairs is
        a slight over-estimate — acceptable for relative scoring.
        """
        a, b = pair
        f = freq
        N_new = N - f  # each merge collapses 2 tokens → 1, net −f tokens
        if N_new <= 0:
            return 0.0

        L_new = L

        if a == b:
            c_a = counts[a]
            if c_a > 0:
                L_new -= c_a * math.log2(c_a)
            c_a_new = c_a - 2 * f
            if c_a_new > 0:
                L_new += c_a_new * math.log2(c_a_new)
        else:
            c_a = counts[a]
            if c_a > 0:
                L_new -= c_a * math.log2(c_a)
            c_a_new = c_a - f
            if c_a_new > 0:
                L_new += c_a_new * math.log2(c_a_new)

            c_b = counts[b]
            if c_b > 0:
                L_new -= c_b * math.log2(c_b)
            c_b_new = c_b - f
            if c_b_new > 0:
                L_new += c_b_new * math.log2(c_b_new)

        # New merged token enters with count f
        L_new += f * math.log2(f)

        return self._H_from_L(L_new, N_new)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train on text until the vocabulary reaches vocab_size tokens.

        Each step:
          1. Count all consecutive pairs in the current token sequence.
          2. For each candidate pair, compute significance_score in O(1).
          3. Select the highest-scoring pair (ties broken by frequency).
          4. Apply the merge, extend vocab, record history.

        Args:
            text: Raw training corpus.
            vocab_size: Target vocabulary size (must be >= 256).
        """
        assert vocab_size >= 256, "vocab_size must be >= 256"

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self._merge_history = []

        ids: List[int] = list(text.encode("utf-8"))

        for step in range(vocab_size - 256):
            pairs = _get_pairs(ids)
            if not pairs:
                break

            counts = Counter(ids)
            N = len(ids)
            L = self._compute_L(counts)
            H_now = self._H_from_L(L, N)

            # Score every candidate pair
            entropy_reductions: Dict[Tuple[int, int], float] = {}
            sig_scores: Dict[Tuple[int, int], float] = {}

            for pair, freq in pairs.items():
                H_new = self._H_after_merge(pair, freq, counts, N, L)
                er = max(0.0, H_now - H_new)
                entropy_reductions[pair] = er
                sig_scores[pair] = er * freq  # significance_score

            # Build final selection score according to entropy_weight
            if self.entropy_weight >= 1.0:
                final: Dict[Tuple[int, int], float] = sig_scores

            elif self.entropy_weight <= 0.0:
                final = {p: float(f) for p, f in pairs.items()}

            else:
                # Normalised blend — scale each dimension to [0, 1] so that
                # entropy_weight is a true proportion of each component.
                max_freq = float(max(pairs.values()))
                max_sig = max(sig_scores.values()) or 1e-10
                w = self.entropy_weight
                final = {
                    p: (1 - w) * pairs[p] / max_freq + w * sig_scores[p] / max_sig
                    for p in pairs
                }

            # Tiebreak by raw frequency for determinism
            best = max(final, key=lambda p: (final[p], pairs[p]))
            freq = pairs[best]
            er = entropy_reductions[best]
            sig = sig_scores[best]

            new_id = 256 + step
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self._merge_history.append(
                {
                    "step": step,
                    "pair": best,
                    "frequency": freq,
                    "entropy_before": H_now,
                    "entropy_after": H_now - er,
                    "entropy_reduction": er,
                    "significance_score": sig,
                    "new_token_id": new_id,
                    "new_token": self.vocab[new_id],
                }
            )

            ids = _merge_ids(ids, best, new_id)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        ids: List[int] = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = _merge_ids(ids, pair, new_id)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        raw = b"".join(self.vocab[t] for t in token_ids)
        return raw.decode("utf-8", errors="replace")

    def get_vocab(self) -> Dict[int, bytes]:
        return dict(self.vocab)

    def get_merge_history(self) -> List[dict]:
        return list(self._merge_history)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "To be, or not to be, that is the question." * 20

    for cls in (StandardBPE, SignificanceAwareBPE):
        tok = cls()
        tok.train(sample, vocab_size=280)

        tokens = tok.encode(sample)
        decoded = tok.decode(tokens)
        ratio = len(sample.encode("utf-8")) / len(tokens)

        print(f"\n{cls.__name__}")
        print(f"  vocab_size  : {len(tok.get_vocab())}")
        print(f"  tokens      : {len(tokens)}  (from {len(sample.encode('utf-8'))} bytes)")
        print(f"  compression : {ratio:.3f}x")
        print(f"  round-trip  : {'OK' if decoded == sample else 'FAIL'}")

        hist = tok.get_merge_history()
        print(f"  merges      : {len(hist)}")
        if hist and "significance_score" in hist[0]:
            top = sorted(hist, key=lambda m: m["significance_score"], reverse=True)[:3]
            print("  top-3 by significance:")
            for m in top:
                print(
                    f"    step {m['step']:3d}  {m['new_token']!r:20s}"
                    f"  freq={m['frequency']:5d}"
                    f"  ΔH={m['entropy_reduction']:.6f}"
                    f"  sig={m['significance_score']:.4f}"
                )
