"""
tests/test_bpe.py — pytest unit tests for StandardBPE and SignificanceAwareBPE.

Run with:
    pytest tests/test_bpe.py -v
    pytest tests/test_bpe.py -v -k "shakespeare"   # only corpus tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bpe_tokenizer import StandardBPE, SignificanceAwareBPE, Tokenizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE      = "AAABBBCCC" * 30          # highly repetitive ASCII
VARIED      = "the quick brown fox jumps over the lazy dog " * 20
UNICODE     = "café résumé naïve Ångström 日本語 emoji🎉" * 10
SINGLE_CHAR = "x" * 50
SPECIAL     = "!@#$%^&*()[]{}<>?/\\|~`" * 15

VOCAB_SMALL = 260   # 4 merges above byte floor
VOCAB_MED   = 300   # 44 merges
VOCAB_LARGE = 512   # 256 merges


@pytest.fixture(scope="module")
def shakespeare_excerpt():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "input.txt")
    if not os.path.exists(path):
        pytest.skip("input.txt not found — run benchmarks.py first to download it")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:5_000]   # first 5K chars is enough


@pytest.fixture(params=[StandardBPE, SignificanceAwareBPE], ids=["StandardBPE", "SignificanceAwareBPE"])
def tokenizer_class(request):
    return request.param


@pytest.fixture(params=[StandardBPE, SignificanceAwareBPE], ids=["StandardBPE", "SignificanceAwareBPE"])
def trained_tokenizer(request):
    tok = request.param()
    tok.train(VARIED, VOCAB_MED)
    return tok


# ---------------------------------------------------------------------------
# 1. train() produces expected vocab_size
# ---------------------------------------------------------------------------

class TestVocabSize:
    def test_exact_vocab_size(self, tokenizer_class):
        # VARIED has 26+ unique chars → enough pairs to always reach VOCAB_MED
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_MED)
        assert len(tok.get_vocab()) == VOCAB_MED

    def test_minimum_vocab_size(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(SIMPLE, 256)
        assert len(tok.get_vocab()) == 256

    def test_small_text_caps_at_available_merges(self, tokenizer_class):
        tok = tokenizer_class()
        tiny = "ab"  # only 1 possible pair
        tok.train(tiny, VOCAB_LARGE)
        # Can only make 1 merge; vocab size = 256 + min(merges_possible, requested)
        assert len(tok.get_vocab()) <= VOCAB_LARGE
        assert len(tok.get_vocab()) >= 256

    def test_vocab_contains_all_byte_tokens(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(SIMPLE, VOCAB_SMALL)
        vocab = tok.get_vocab()
        for i in range(256):
            assert i in vocab, f"byte token {i} missing from vocab"
            assert vocab[i] == bytes([i])

    def test_merged_tokens_are_byte_concatenations(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(SIMPLE, VOCAB_SMALL)
        vocab = tok.get_vocab()
        for entry in tok.get_merge_history():
            a, b = entry["pair"]
            expected = vocab[a] + vocab[b]
            assert vocab[entry["new_token_id"]] == expected


# ---------------------------------------------------------------------------
# 2. encode / decode round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        SIMPLE, VARIED, SINGLE_CHAR, SPECIAL,
    ])
    def test_roundtrip_ascii(self, tokenizer_class, text):
        tok = tokenizer_class()
        tok.train(text, VOCAB_MED)
        assert tok.decode(tok.encode(text)) == text

    def test_roundtrip_unicode(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(UNICODE, VOCAB_MED)
        assert tok.decode(tok.encode(UNICODE)) == UNICODE

    def test_roundtrip_shakespeare(self, tokenizer_class, shakespeare_excerpt):
        tok = tokenizer_class()
        tok.train(shakespeare_excerpt, VOCAB_MED)
        assert tok.decode(tok.encode(shakespeare_excerpt)) == shakespeare_excerpt

    def test_roundtrip_unseen_ascii(self, tokenizer_class):
        """Unseen text should still round-trip (via byte fallback)."""
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_MED)
        unseen = "Hello, world! This was not in training."
        assert tok.decode(tok.encode(unseen)) == unseen


# ---------------------------------------------------------------------------
# 3. Determinism — identical inputs produce identical token sequences
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_encode_is_deterministic(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_MED)
        assert tok.encode(SIMPLE) == tok.encode(SIMPLE)

    def test_two_instances_same_output(self, tokenizer_class):
        tok1 = tokenizer_class()
        tok2 = tokenizer_class()
        tok1.train(VARIED, VOCAB_MED)
        tok2.train(VARIED, VOCAB_MED)
        assert tok1.encode(SIMPLE) == tok2.encode(SIMPLE)

    def test_merge_order_is_deterministic(self, tokenizer_class):
        tok1 = tokenizer_class()
        tok2 = tokenizer_class()
        tok1.train(VARIED, VOCAB_MED)
        tok2.train(VARIED, VOCAB_MED)
        h1 = [(e["pair"], e["new_token_id"]) for e in tok1.get_merge_history()]
        h2 = [(e["pair"], e["new_token_id"]) for e in tok2.get_merge_history()]
        assert h1 == h2


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_encode_empty_string(self, trained_tokenizer):
        assert trained_tokenizer.encode("") == []

    def test_decode_empty_list(self, trained_tokenizer):
        assert trained_tokenizer.decode([]) == ""

    def test_single_unique_char(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(SINGLE_CHAR, VOCAB_SMALL)
        assert tok.decode(tok.encode(SINGLE_CHAR)) == SINGLE_CHAR

    def test_special_characters(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(SPECIAL, VOCAB_SMALL)
        assert tok.decode(tok.encode(SPECIAL)) == SPECIAL

    def test_unicode_multibyte(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(UNICODE, VOCAB_MED)
        sample = "café"
        assert tok.decode(tok.encode(sample)) == sample

    def test_newlines_and_tabs(self, tokenizer_class):
        text = "line one\nline two\ttabbed\n" * 20
        tok = tokenizer_class()
        tok.train(text, VOCAB_SMALL)
        assert tok.decode(tok.encode(text)) == text

    def test_vocab_size_floor_assertion(self, tokenizer_class):
        tok = tokenizer_class()
        with pytest.raises(ValueError):
            tok.train("hello", 100)   # below 256 byte floor

    def test_encode_text_not_in_training(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train("abcdef " * 30, VOCAB_SMALL)
        # emoji not in training — should still encode/decode via raw bytes
        result = tok.decode(tok.encode("🎉"))
        assert result == "🎉"


# ---------------------------------------------------------------------------
# 5. Merge history correctness
# ---------------------------------------------------------------------------

class TestMergeHistory:
    def test_history_length_matches_new_tokens(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_MED)
        n_merges = len(tok.get_vocab()) - 256
        assert len(tok.get_merge_history()) == n_merges

    def test_history_step_indices_are_sequential(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_SMALL)
        steps = [e["step"] for e in tok.get_merge_history()]
        assert steps == list(range(len(steps)))

    def test_history_new_token_ids_are_sequential(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_SMALL)
        ids = [e["new_token_id"] for e in tok.get_merge_history()]
        assert ids == list(range(256, 256 + len(ids)))

    def test_history_frequencies_are_positive(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_SMALL)
        for entry in tok.get_merge_history():
            assert entry["frequency"] > 0

    def test_standard_history_keys(self):
        tok = StandardBPE()
        tok.train(VARIED, VOCAB_SMALL)
        required = {"step", "pair", "frequency", "new_token_id", "new_token"}
        for entry in tok.get_merge_history():
            assert required.issubset(entry.keys())

    def test_significance_history_keys(self):
        tok = SignificanceAwareBPE()
        tok.train(VARIED, VOCAB_SMALL)
        required = {
            "step", "pair", "frequency", "new_token_id", "new_token",
            "entropy_before", "entropy_after", "entropy_reduction", "significance_score",
        }
        for entry in tok.get_merge_history():
            assert required.issubset(entry.keys())

    def test_history_pairs_are_tuples_of_valid_ids(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_MED)
        vocab = tok.get_vocab()
        for entry in tok.get_merge_history():
            a, b = entry["pair"]
            assert a in vocab
            assert b in vocab

    def test_get_merge_history_returns_copy(self, tokenizer_class):
        """Mutating the returned list must not corrupt the tokenizer state."""
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_SMALL)
        h = tok.get_merge_history()
        h.clear()
        assert len(tok.get_merge_history()) > 0


# ---------------------------------------------------------------------------
# 6. Significance scores (SignificanceAwareBPE-specific)
# ---------------------------------------------------------------------------

class TestSignificanceScores:
    def test_significance_scores_are_non_negative(self):
        tok = SignificanceAwareBPE()
        tok.train(VARIED, VOCAB_MED)
        for entry in tok.get_merge_history():
            assert entry["significance_score"] >= 0.0

    def test_entropy_reduction_is_non_negative(self):
        tok = SignificanceAwareBPE()
        tok.train(VARIED, VOCAB_MED)
        for entry in tok.get_merge_history():
            assert entry["entropy_reduction"] >= 0.0

    def test_entropy_after_leq_entropy_before(self):
        tok = SignificanceAwareBPE()
        tok.train(VARIED, VOCAB_MED)
        for entry in tok.get_merge_history():
            # entropy_reduction = before - after ≥ 0 means after ≤ before
            assert entry["entropy_after"] <= entry["entropy_before"] + 1e-9

    def test_significance_score_equals_reduction_times_freq(self):
        tok = SignificanceAwareBPE()
        tok.train(VARIED, VOCAB_SMALL)
        for entry in tok.get_merge_history():
            expected = entry["entropy_reduction"] * entry["frequency"]
            assert abs(entry["significance_score"] - expected) < 1e-9

    def test_entropy_weight_zero_mimics_standard_bpe(self):
        """entropy_weight=0 should select the same merges as StandardBPE."""
        std = StandardBPE()
        sig = SignificanceAwareBPE(entropy_weight=0.0)
        text = VARIED
        std.train(text, VOCAB_SMALL)
        sig.train(text, VOCAB_SMALL)
        std_pairs = [e["pair"] for e in std.get_merge_history()]
        sig_pairs = [e["pair"] for e in sig.get_merge_history()]
        assert std_pairs == sig_pairs

    def test_high_frequency_pair_gets_positive_significance(self):
        """The most-frequent pair must have significance_score > 0 on non-trivial text."""
        tok = SignificanceAwareBPE()
        tok.train(SIMPLE, VOCAB_SMALL)
        hist = tok.get_merge_history()
        assert any(e["significance_score"] > 0 for e in hist)


# ---------------------------------------------------------------------------
# 7. Encode produces valid token ids (0 to vocab_size-1)
# ---------------------------------------------------------------------------

class TestTokenIdValidity:
    def test_all_token_ids_in_vocab(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        tokens = trained_tokenizer.encode(VARIED)
        assert all(t in vocab for t in tokens)

    def test_token_ids_within_range(self, trained_tokenizer):
        vocab_size = len(trained_tokenizer.get_vocab())
        tokens = trained_tokenizer.encode(VARIED)
        assert all(0 <= t < vocab_size for t in tokens)

    def test_byte_tokens_map_to_single_bytes(self, trained_tokenizer):
        vocab = trained_tokenizer.get_vocab()
        for i in range(256):
            assert vocab[i] == bytes([i])

    def test_encode_reduces_or_keeps_length(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(VARIED, VOCAB_MED)
        tokens = tok.encode(VARIED)
        # BPE can only reduce or maintain sequence length vs bytes
        assert len(tokens) <= len(VARIED.encode("utf-8"))

    def test_encode_shakespeare_all_valid_ids(self, tokenizer_class, shakespeare_excerpt):
        tok = tokenizer_class()
        tok.train(shakespeare_excerpt, VOCAB_MED)
        vocab = tok.get_vocab()
        tokens = tok.encode(shakespeare_excerpt)
        assert all(t in vocab for t in tokens)


# ---------------------------------------------------------------------------
# 8. Decode handles out-of-order / arbitrary valid token ids gracefully
# ---------------------------------------------------------------------------

class TestDecodeRobustness:
    def test_decode_reversed_tokens(self, trained_tokenizer):
        tokens = trained_tokenizer.encode(VARIED)
        # Reversed token sequence decodes without raising
        result = trained_tokenizer.decode(list(reversed(tokens)))
        assert isinstance(result, str)

    def test_decode_single_byte_tokens(self, trained_tokenizer):
        # Decoding raw byte ids 0–127 should give ASCII characters
        tokens = list(range(65, 91))  # 'A' through 'Z'
        result = trained_tokenizer.decode(tokens)
        assert result == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_decode_shuffled_tokens(self, trained_tokenizer):
        import random
        tokens = trained_tokenizer.encode(VARIED)
        shuffled = tokens[:]
        random.shuffle(shuffled)
        result = trained_tokenizer.decode(shuffled)
        assert isinstance(result, str)

    def test_decode_invalid_utf8_uses_replace(self, trained_tokenizer):
        """Decoding a sequence that forms invalid UTF-8 must not raise."""
        # Token 0x80 is a continuation byte that is invalid on its own
        result = trained_tokenizer.decode([0x80])
        assert isinstance(result, str)

    def test_decode_all_byte_tokens(self, trained_tokenizer):
        """All 256 single-byte tokens must decode without exception."""
        result = trained_tokenizer.decode(list(range(256)))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 9. get_vocab / get_merge_history return independent copies
# ---------------------------------------------------------------------------

class TestAPIContractIntegrity:
    def test_get_vocab_returns_copy(self, trained_tokenizer):
        v = trained_tokenizer.get_vocab()
        v[9999] = b"injected"
        assert 9999 not in trained_tokenizer.get_vocab()

    def test_retrain_resets_state(self, tokenizer_class):
        tok = tokenizer_class()
        tok.train(SIMPLE, VOCAB_SMALL)
        first_vocab = tok.get_vocab().copy()
        tok.train(VARIED, VOCAB_MED)
        assert tok.get_vocab() != first_vocab
        assert len(tok.get_vocab()) == VOCAB_MED

    def test_repr_shows_trained_state(self, tokenizer_class):
        tok = tokenizer_class()
        assert "trained=False" in repr(tok) or "vocab_size=0" in repr(tok)
        tok.train(SIMPLE, VOCAB_SMALL)
        r = repr(tok)
        assert str(VOCAB_SMALL) in r
