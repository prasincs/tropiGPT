# Tropical Attention Research Findings

## Experiment: Categorical Deep Learning for Arithmetic Generalization

**Date**: December 2024
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

### 3.3 Conclusion

**Neither model generalizes to out-of-distribution digit lengths.**

Both models achieve high accuracy (~95-98%) on the training distribution but **completely fail** (0%) on all other digit lengths—including 2-digit problems which are strictly simpler.

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

The tropical semiring hypothesis targets **attention pattern sharpness**—the idea that discrete, hard attention would better capture algorithmic steps. However:

1. **The bottleneck isn't attention sharpness**: Both models learn to attend correctly within the training format.

2. **The bottleneck is positional generalization**: Neither model understands that digit positions are relative, not absolute.

3. **Tropical attention works**: It learns the training task (95.5% accuracy), proving the implementation is correct.

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
