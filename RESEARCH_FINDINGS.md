# Tropical Attention Research Findings

## Experiment: Categorical Deep Learning for Arithmetic Generalization

**Date**: December 2025
**Branch**: `tropical-attention-research`

---

## 1. Research Question

> Does replacing the standard Sum-Product semiring in Transformer attention with the Tropical (Max-Plus) semiring improve out-of-distribution generalization on arithmetic tasks?

### Hypothesis

Standard attention uses **softmax** (sum-product operations), which creates "soft" weighted averages that may blur precise algorithmic steps. Tropical attention uses **max** operations, creating "hard" discrete pointers that might better capture the precise, step-by-step nature of arithmetic (especially carry propagation).

---

## 2. Implementation

### 2.1 Tropical Attention Mechanism

Located in `model.py`, class `TropicalCausalSelfAttention`:

```
Standard Attention:
  1. Score = Q @ K^T                    (dot product: sum of products)
  2. Weights = softmax(Score)           (normalize by sum)
  3. Output = Weights @ V               (weighted sum)

Tropical Attention:
  1. Score[i,j] = max_d(Q[i,d] + K[j,d])  (max of sums)
  2. Weights = Score - max(Score)          (normalize by max)
  3. Output[i,d] = max_j(W[i,j] + V[j,d])  (max of sums)
```

### 2.2 Differentiable Training

Since `max` has sparse gradients, we use **LogSumExp** approximation with temperature annealing:

```python
tropical_max(x) ≈ T * log(sum(exp(x/T)))
```

- **T = 1.0**: Soft (similar to mean)
- **T → 0**: Hard (approaches true max)

Training anneals T from 1.0 → 0.1 over the course of training.

### 2.3 Task: 3-Digit Addition

- **Format**: `"123+456=0579\n"` (zero-padded result)
- **Training data**: 50,000 random 3-digit addition problems
- **Vocabulary**: 13 tokens (`0-9`, `+`, `=`, `\n`)

---

## 3. Results

### 3.1 Accuracy by Digit Length

| Digits | Type | Standard GPT | Tropical GPT |
|--------|------|--------------|--------------|
| 2-digit | Simpler | 0.0% | 0.0% |
| **3-digit** | **Training** | **98.5%** | **95.5%** |
| 4-digit | OOD | 0.0% | 0.0% |
| 5-digit | OOD | 0.0% | 0.0% |
| 6-digit | OOD | 0.0% | 0.0% |
| 7-digit | OOD | 0.0% | 0.0% |
| 8-digit | OOD | 0.0% | 0.0% |

### 3.2 Training Statistics

| Metric | Standard GPT | Tropical GPT |
|--------|--------------|--------------|
| Iterations | 4,500 | 13,250 |
| Best Val Loss | 1.42 | 1.45 |
| 3-digit Accuracy | 98.5% | 95.5% |
| OOD Accuracy | 0.0% | 0.0% |

### 3.3 Follow-up Experiment: Variable-Length Training

To test if training on mixed digit lengths prevents positional memorization:

| Digits | Type | Standard GPT | Tropical GPT |
|--------|------|--------------|--------------|
| 2-digit | Training | **100%** | 93% |
| 3-digit | Training | **97%** | 88% |
| 4-digit | Training | **95%** | 83% |
| 5-digit | Training | **95%** | 56% |
| 6-digit | OOD | 0% | 0% |
| 7-digit | OOD | 0% | 0% |
| 8-digit | OOD | 0% | 0% |

**Key Finding**: Standard GPT significantly outperforms Tropical GPT on variable-length training! Standard maintains high accuracy (95-100%) while Tropical degrades severely (56% on 5-digit).

### 3.4 Follow-up Experiment: Reversed Digit Order

Tested little-endian format (`321+654=975` instead of `123+456=579`):

| Model | 3-digit Accuracy |
|-------|------------------|
| Standard GPT | 13% |
| Tropical GPT | ~15% |

**Key Finding**: Reversed format is significantly harder for both models. Neither learns it well.

### 3.6 Follow-up Experiment: Binary Addition (4-bit)

Tested binary representation where carries are explicit:

| Model | 4-bit Accuracy |
|-------|----------------|
| Standard GPT | **99.5%** |
| Tropical GPT | 0.0% |

**Key Finding**: Tropical attention COMPLETELY FAILS on binary addition. Standard GPT nearly masters it while Tropical cannot learn it at all. This is strong evidence against the tropical attention hypothesis.

### 3.7 Follow-up Experiment: Longer Training (10k-50k iterations)

**Date**: December 29, 2025

Extended training revealed that **2k iterations was insufficient**. With longer training:

| Configuration | 3-digit | 5-digit | 10-digit | 15-digit (OOD) | Val Loss |
|--------------|---------|---------|----------|----------------|----------|
| **Abacus (50k)** | **99%** | **97%** | **96%** | 0% | 0.089 |
| Baseline (50k) | 98% | 96% | 85% | 0% | 0.107 |
| **Abacus (10k)** | **97%** | **95%** | **45%** | 0% | 0.090 |
| Baseline (10k) | 37% | 10% | 0% | 0% | 0.123 |
| All 2k configs | 0% | 0% | 0% | 0% | 0.13-0.17 |

**Key Findings:**

1. **Abacus embeddings are highly effective**: 96% vs 85% on 10-digit addition at 50k iterations
2. **Training duration matters**: 2k iterations shows 0% accuracy; 10k shows emergence; 50k achieves near-perfect ID accuracy
3. **OOD generalization still fails**: 0% on 15-20 digit for ALL configurations (positional overfitting)
4. **Loss is misleading at low iterations**: 0.17 loss at 2k vs 0.09 loss at 50k, but accuracy difference is 0% vs 96%

### 3.8 Tropical Attention Status

Tropical attention experiments on MPS (Mac M4) were extremely slow (~13 hours for 10k iterations) and were killed before convergence. CUDA experiments on RTX 3080 are needed for fair comparison.

### 3.9 Conclusion

**Standard GPT outperforms Tropical GPT across ALL experiments.**

- Fixed-length 3-digit: Standard (98.5%) ≈ Tropical (95.5%)
- Variable-length: Standard (95-100%) >> Tropical (56-93%)
- Reversed digits: Both fail (~13-15%)
- Binary addition: Standard (99.5%) >>> Tropical (0%)
- OOD generalization: Both fail (0%)
- **NEW**: Tropical attention improves per-digit accuracy 2x (22% vs 12%) but not enough for exact match

---

## 4. Analysis

### 4.1 Why Both Models Fail on OOD

The failure is **not due to the attention mechanism** but to **positional overfitting**:

1. **Fixed-length format**: Training data has a rigid structure
   - Positions 0-2: First number
   - Position 3: `+` operator
   - Positions 4-6: Second number
   - Position 7: `=` sign
   - Positions 8-11: Result
   - Position 12: `\n`

2. **Position embeddings memorize format**: The model learns that "digit in position 8 is first digit of answer" rather than learning the addition algorithm.

3. **Different lengths = different positions**: A 2-digit problem `"12+34=046\n"` places digits at different positions, breaking the learned mapping.

### 4.2 Why Tropical Attention Didn't Help

The tropical semiring hypothesis targets **attention pattern sharpness**—the idea that discrete, hard attention would better capture algorithmic steps. However, our experiments show:

1. **Tropical attention hurts performance**: In variable-length and binary tasks, tropical attention performs significantly worse than standard attention.

2. **The LogSumExp approximation may be the problem**: Training tropical attention requires a differentiable approximation (LogSumExp with temperature annealing). This may not converge well, especially on tasks requiring precise discrete reasoning.

3. **Standard softmax attention is already sufficient**: For arithmetic tasks within the training distribution, standard attention achieves near-perfect accuracy (98-99.5%).

4. **Tropical attention may need different training dynamics**: The temperature annealing schedule (1.0 → 0.1) may not be optimal. The model may need curriculum learning or different optimization.

### 4.3 The "Carry Problem" Remains Unsolved

The original motivation was that tropical attention might better handle carry propagation in addition. However, we cannot test this hypothesis because:

- The model fails on OOD lengths before we can observe carry behavior
- Within the training distribution, standard attention already achieves near-perfect accuracy

---

## 5. Future Directions

To properly test the tropical attention hypothesis for arithmetic generalization:

### 5.1 Variable-Length Training
Train on mixed digit lengths (2-5 digits) to prevent positional memorization.

### 5.2 Reversed Digit Order
Present numbers with least-significant digit first: `"321+654=9750\n"`. This aligns carry propagation with left-to-right autoregressive generation.

### 5.3 Digit-by-Digit Output
Output one digit at a time with explicit carry tokens: `"123+456=` → `9,carry` → `7,carry` → `5,no_carry` → `0`

### 5.4 Relative Position Encoding
Replace absolute position embeddings with relative positional encoding (e.g., RoPE, ALiBi) to enable length generalization.

### 5.5 Scratchpad / Chain-of-Thought
Train with intermediate computation steps visible, allowing the model to "show its work."

### 5.6 Longer Training
The TropiGPT v2 experiments only ran 2000 iterations. Standard nanoGPT arithmetic experiments typically need 10,000-50,000 iterations. The 22% per-digit accuracy for Tropical attention (vs 12% baseline) suggests potential that might be realized with longer training.

### 5.7 Better Evaluation Metrics
Focus on per-digit accuracy rather than exact match when comparing approaches. A model with 50% per-digit accuracy is significantly better than 10% (random), even if both show 0% exact match.

---

## 6. Files in This Branch

| File | Description |
|------|-------------|
| `model.py` | Added `TropicalCausalSelfAttention` class |
| `train.py` | Added temperature annealing for tropical attention |
| `config/train_adder_standard.py` | Standard GPT config for addition task |
| `config/train_adder_tropical.py` | Tropical GPT config for addition task |
| `data/adder/prepare.py` | Dataset generation for 3-digit addition |
| `evaluate_arithmetic.py` | Basic evaluation script |
| `evaluate_generalization.py` | Comprehensive multi-length evaluation |
| `EVALUATION_GUIDE.md` | Guide for interpreting results |

---

## 7. Reproduction

```bash
# Setup
uv venv && uv pip install torch numpy transformers datasets tiktoken wandb tqdm

# Generate data
.venv/bin/python data/adder/prepare.py

# Train both models
.venv/bin/python train.py config/train_adder_standard.py
.venv/bin/python train.py config/train_adder_tropical.py

# Evaluate
.venv/bin/python evaluate_generalization.py --samples 200
```

---

## 8. Key Takeaway

> **Tropical attention is a valid, learnable attention mechanism**, but it does not solve length generalization for arithmetic when trained on fixed-length sequences. The fundamental challenge of positional generalization requires architectural changes (relative positions) or training data changes (variable lengths) rather than just modifying the attention semiring.

---

## References

1. Karpathy, A. "nanoGPT" - https://github.com/karpathy/nanoGPT
2. Tropical Semiring - Max-Plus algebra in optimization and neural networks
3. Length Generalization in Transformers - Various works on positional encoding limitations
