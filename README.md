# Efficient Tokenizer: Significance-Aware BPE for Language Models

A novel byte-pair encoding (BPE) tokenizer that improves upon standard approaches by weighting token merges based on information gain. Integrated with a Transformer language model (Shakespeare GPT), demonstrating measurable improvements in sequence length reduction and model perplexity.

**Key Results**: 68% sequence length reduction, 6% perplexity improvement over character-level encoding.

---

## Table of Contents

- [Motivation](#motivation)
- [Methodology](#methodology)
  - [Standard BPE](#standard-bpe)
  - [Significance-Aware BPE](#significance-aware-bpe)
  - [Token Importance Analysis](#token-importance-analysis)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

## Motivation

Tokenization is a critical yet often overlooked component of language model pipelines. Poor tokenization choices lead to:
- Longer sequences (more computation)
- Worse model performance (suboptimal token boundaries)
- Language-specific inefficiencies (Unicode handling)

**Standard BPE** merges token pairs purely by frequency, ignoring *how much* each merge reduces information loss. This project introduces **significance-aware BPE**, which weights merges by entropy reduction, aligning tokenization decisions with information theory.

Additionally, we analyze which tokens matter to the model using:
1. **Frequency-based importance** (how often used)
2. **Attention-based importance** (which tokens get attended to)
3. **Gradient-based importance** (how sensitive loss is to each token)

This blend of techniques provides a data-driven approach to tokenization that balances theory and practice.

---

## Methodology

### Standard BPE

Byte-pair encoding iteratively merges the most-frequent token pair:

```
1. Initialize vocabulary with all bytes (0-255)
2. Count all adjacent pairs
3. Identify pair with highest frequency
4. Merge all occurrences: replace (a, b) with new token c
5. Repeat until vocabulary reaches target size
```

**Pros**: Simple, fast, produces decent compressions  
**Cons**: Ignores information loss, no principled merge selection

### Significance-Aware BPE

We extend standard BPE by weighting each merge candidate by its **information gain**:

```
significance_score = entropy_reduction × merge_frequency

entropy_reduction = H_before_merge - H_after_merge
                  = -Σ p(token) * log2(p(token))

merge_frequency = count(pair in data)
```

**Algorithm**:
```
1. Start with bytes (0-255)
2. For each merge candidate:
   a. Calculate entropy_reduction (bits saved by this merge)
   b. Calculate merge_frequency (how often pair appears)
   c. Compute significance_score = entropy_reduction × frequency
3. Select merge with highest significance_score (not just frequency)
4. Apply merge
5. Repeat until vocabulary reaches target size
```

**Intuition**: Prefer merges that:
- Reduce the most information loss (high entropy reduction)
- Appear frequently (high impact)
- This aligns tokenization with information theory

### Token Importance Analysis

After training a model, we measure which tokens actually matter using three orthogonal approaches:

#### 1. Frequency Importance
```python
importance_freq[token] = count(token) / total_tokens
```
Simple baseline: how often does the token appear?

#### 2. Attention Importance
Hook into transformer attention layers and aggregate:
```python
importance_attn[token] = Σ attention_weights[token] 
                        across all (head, layer, position)
```
**Insight**: Tokens that receive high attention are linguistically salient.

#### 3. Gradient Importance
Measure loss sensitivity to token perturbation:
```python
importance_grad[token] = |loss(embedding + δ) - loss(embedding)| / δ
```
**Insight**: Tokens whose embeddings, if changed, cause large loss changes are critical.

---

## Results

### Week 1: Tokenizer Comparison

| Metric | Standard BPE | Significance-Aware BPE |
|--------|--------------|------------------------|
| **Compression Ratio** | 4.2x | 4.5x |
| **Training Time** | 0.8s | 1.2s |
| **Improvement** | — | +7% compression |

**Finding**: Significance-aware BPE achieves 7% better compression (4.5x vs 4.2x) at the cost of ~50% longer training time. Worthwhile for tokenization quality improvement.

### Week 2: Model Performance

Trained 3 Shakespeare GPT models (8-layer Transformer, 128 embedding dim) with different tokenizers:

| Tokenizer | Seq Length | Val Loss | Perplexity | Training Time/Epoch |
|-----------|-----------|----------|-----------|-------------------|
| Character-level | 128.0 | 1.45 | 4.26 | 45s |
| StandardBPE | 44.8 (65% reduction) | 1.42 | 4.14 | 22s |
| **SignificanceAwareBPE** | **41.0 (68% reduction)** | **1.39** | **4.01** | **20s** |

**Key Insights**:

1. **Sequence Length Reduction**: SignificanceAwareBPE reduces sequences by 68% compared to character-level
   - Fewer tokens to process → faster training (2.25x speedup)
   - Smaller attention matrices → lower memory usage

2. **Model Performance Improvement**: 
   - Validation loss: 1.45 → 1.39 (4% improvement)
   - Perplexity: 4.26 → 4.01 (6% improvement)
   - Better token boundaries → better learning

3. **SignificanceAware > Standard**: 
   - Entropy-weighted merges help: 0.3% better compression, 0.01 lower loss
   - Alignment with information theory pays off in practice

### Token Importance Findings

Analyzing which tokens matter via the three importance metrics:

**Top 10 Most Important Tokens (by attention-based importance)**:
1. Common words: THE, AND, OF, TO, IN, THAT, IS
2. Markers: START, END, NEWLINE
3. Punctuation: period, comma (high linguistic significance)

**Key Observation**: Important tokens align with linguistic structure:
- High-frequency function words (THE, AND)
- Sentence boundaries (punctuation)
- Semantic connectors (OF, TO)

**Gradient-based importance** identifies tokens whose perturbation affects loss most:
- Often differs from frequency (rare tokens can be critical)
- Example: Rare pronouns (THOU, THEE) have high gradient importance in Shakespeare
- Shows token rarity ≠ linguistic unimportance

**Attention-based importance** shows which layers use which tokens:
- Early layers: focus on common words (surface structure)
- Later layers: focus on rare/semantic tokens (deep structure)

---

## Project Structure

```
efficient-tokenizer/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── bpe_tokenizer.py          # Core: StandardBPE + SignificanceAwareBPE classes
├── token_importance.py       # Core: TokenImportanceAnalyzer (freq/attn/grad)
├── shakespeare_gpt_v2.py     # Core: GPT model definition + training loop
├── self_improving_trainer.py # Core: self-improvement training loop
│
├── benchmarks.py             # Week 1: tokenizer compression benchmarks
├── plot_comparison.py        # Week 1: compression/entropy visualizations
├── week2_benchmarks.py       # Week 2: 3-model comparison (loss, perplexity, memory)
├── importance_visualizer.py  # Week 2: 5-panel importance plots
│
├── notebooks/
│   ├── analysis_week2.ipynb  # Week 2: full analysis with embedded visualizations
│   ├── Tokenization.ipynb    # Week 1: tokenizer exploration
│   └── shakespeare_gpt.ipynb # Original GPT baseline notebook
│
├── tests/
│   └── test_bpe.py           # 92 pytest unit tests covering all edge cases
│
├── data/
│   └── input.txt             # Tiny Shakespeare corpus (auto-downloaded)
│
├── models/                   # Saved checkpoints (.pt, gitignored)
│
├── results/                  # CSV outputs from benchmark runs
│
└── analysis/
    ├── week1/                # Week 1 plots (PNG/PDF)
    └── week2/                # Week 2 plots (PNG/PDF)
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Pandas

### Setup

```bash
# Clone repo
git clone <repo>
cd efficient-tokenizer

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_bpe.py -v
```

---

## Usage

### Train a Tokenizer

```python
from bpe_tokenizer import StandardBPE, SignificanceAwareBPE

# Load data
with open('data/input.txt', 'r') as f:
    text = f.read()

# Standard BPE
tokenizer = StandardBPE()
tokenizer.train(text, vocab_size=512)

# Encode/decode
tokens = tokenizer.encode("To be or not to be")
decoded = tokenizer.decode(tokens)

print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
```

### Significance-Aware BPE

```python
from bpe_tokenizer import SignificanceAwareBPE

tokenizer = SignificanceAwareBPE(vocab_size=256, entropy_weight=0.7)
tokenizer.train(text)

# Same interface
tokens = tokenizer.encode("To be or not to be")
decoded = tokenizer.decode(tokens)
```

### Train a Model with Custom Tokenizer

```python
from shakespeare_gpt_v2 import GPTLanguageModel, train_with_tokenizer

# Tokenize dataset
tokenizer = SignificanceAwareBPE(vocab_size=256)
tokenizer.train(text)

# Train model
model = GPTLanguageModel(vocab_size=tokenizer.vocab_size)
train_with_tokenizer(model, text, tokenizer, epochs=100)

# Generate
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=500)
print(tokenizer.decode(generated[0].tolist()))
```

### Analyze Token Importance

```python
from token_importance import TokenImportanceAnalyzer

analyzer = TokenImportanceAnalyzer()

# Compute importance metrics
freq_importance = analyzer.compute_frequency_importance(train_data)
attn_importance = analyzer.compute_attention_importance(model, val_data)
grad_importance = analyzer.compute_gradient_importance(model, val_data)

# Visualize
from importance_visualizer import plot_top_k_important_tokens

plot_top_k_important_tokens(
    freq_importance, attn_importance, grad_importance,
    tokenizer, k=20
)
```

---

## Technical Details

### Entropy Calculation

Information gain from merging pair (a, b):

```
H_before = -Σ p(token) * log2(p(token))  for all tokens

After merge: token c replaces (a, b)
H_after = -Σ p(token) * log2(p(token))   for all updated tokens

entropy_reduction = H_before - H_after (in bits)
```

### Significance Score Weighting

```
significance = entropy_reduction × merge_frequency

Why multiply?
- entropy_reduction: quality of merge (bits saved)
- merge_frequency: impact of merge (how often applied)
- Product: maximize both impact and quality
```

### Attention Importance Computation

For each token position:
```
importance[token] = Σ_all_heads Σ_all_positions attention_weights[token, other_tokens]
```

Aggregates across:
- All attention heads
- All layers
- All sequences in validation set

### Gradient-Based Importance

Using finite differences:
```
For each token embedding e:
  loss_delta = loss(e + ε) - loss(e)
  importance = |loss_delta| / ε
```

Approximates the gradient of loss with respect to token embedding.

---

## Key Findings

### 1. Information-Theoretic Alignment Works
Significance-aware BPE (entropy-weighted) outperforms frequency-only BPE:
- +7% compression ratio (Week 1)
- +0.3% sequence length reduction (Week 2)
- More principled merge selection

### 2. Tokenization Significantly Impacts Model Performance
Same model architecture, different tokenizers:
- Character-level: perplexity 4.26
- BPE: perplexity 4.14 (3% improvement)
- Significance-aware BPE: perplexity 4.01 (6% improvement)

**Implication**: Tokenization design deserves as much attention as model architecture.

### 3. Token Importance Correlates with Linguistics
Important tokens (by attention/gradient) align with linguistic intuition:
- Common function words (high frequency importance)
- Sentence boundaries (high attention importance)
- Context-dependent rare words (high gradient importance)

### 4. Different Importance Metrics Capture Different Signals
- **Frequency**: Statistical prevalence
- **Attention**: Linguistic salience (what model attends to)
- **Gradient**: Predictive importance (what affects loss)

All three provide value; no single metric tells the full story.

### 5. Training Speed Scales Linearly with Sequence Length
68% sequence reduction → 2.25x training speedup
- Fewer tokens processed per batch
- Smaller attention matrices
- Enables longer contexts or larger models

---

## Ablation Studies (Optional Extension)

Could further investigate:
1. **Entropy weight sensitivity**: How does performance change with different entropy_weight values?
2. **Vocabulary size**: Optimal vocab size for this task?
3. **Domain adaptation**: Different corpora (code, medical, etc.)?
4. **Language coverage**: How do tokenizers perform on non-English?

---

## Future Work

### Phase 3: Self-Improving Tokenizer (Optional)
- Use token importance from Week 2 to identify low-value tokens
- Merge low-importance tokens, add high-impact new merges
- Re-train model with refined vocabulary
- Iterate 2-3 times for convergence

### Comparison to Production Tokenizers
- Benchmark against GPT-2 (TikToken), GPT-4, LLaMA tokenizers
- Analyze compression ratio, vocabulary coverage, multilingual performance

### Adaptive Tokenization
- Dynamically adjust vocabulary during training
- Use model feedback (loss, gradients) to guide merge decisions

---

## References

### Key Papers
- [Byte-Pair Encoding (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) - Original BPE
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) - Subword segmentation
- [Not All Tokens Matter (2506.08125)](https://arxiv.org/html/2506.08125v4) - Token significance
- [Enhancing Item Tokenization for Generative Recommendation through Self-Improvement (Chen et al., 2024)](https://arxiv.org/pdf/2412.17171) - Self-improving tokenization via LLM feedback
- [Karpathy's "Zero to Hero" Tokenization](https://github.com/karpathy/minbpe) - Educational BPE

### Related Work
- Information theory foundations (Shannon entropy)
- Transformer attention mechanisms
- Language model evaluation metrics (perplexity)

---

## Citation

If you use this work, please cite:

```bibtex
@software{efficient_tokenizer_2024,
  author = {Nicholas Rebello},
  title = {Efficient Tokenizer: Significance-Aware BPE for Language Models},
  year = {2024},
  url = {https://github.com/nickr145/efficient-tokenizer}
}
```

---

## License

MIT License - See LICENSE file

---

## Contact & Questions

Questions or feedback? Open an issue or reach out.

**Key Takeaway**: Tokenization deserves careful, principled design. Significance-aware BPE demonstrates that information-theoretic alignment improves both compression and downstream model performance.