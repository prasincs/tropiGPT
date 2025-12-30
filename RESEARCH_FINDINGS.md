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

### 2.3 Abacus Embeddings

Located in `model.py`, significance-based positional encoding:

```python
# Each digit gets an embedding based on its place value (10^0, 10^1, etc.)
# "123" → significance [3, 2, 1] (hundreds, tens, ones)
# Operators and spaces get significance 0
```

This helps the model learn that digits at the same significance level (ones, tens, etc.) should interact similarly, regardless of absolute position.

### 2.4 Muon Optimizer

Newton-Schulz orthogonalization for weight matrices:

```python
# Only for 2D weight matrices (not embeddings, biases, layernorms)
# X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)
```

### 2.5 ArithmeticDataset

Located in `data/arithmetic_dataset.py`:
- Infinite IterableDataset for addition problems
- Character-level tokenization (vocabulary size 14)
- Returns `(input_ids, significance_ids, target_ids)`
- Configurable digit range (1-10 digits for training)

### 2.6 Task: Variable-Length Addition

- **Format**: `"123 + 456 = 579\n"` (natural format with spaces)
- **Training data**: Infinite stream, 1-10 digit operands
- **Vocabulary**: 14 tokens (`0-9`, `+`, `=`, ` `, `\n`)

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

### Core Implementation
| File | Description |
|------|-------------|
| `model.py` | GPT with Tropical Attention, Abacus Embeddings, Muon optimizer |
| `train_arithmetic.py` | Training script with temperature annealing & evaluation |
| `data/arithmetic_dataset.py` | ArithmeticDataset with character-level tokenization |
| `compare_experiments.py` | Checkpoint evaluation and comparison script |

### Experiment Configs
| File | Description |
|------|-------------|
| `config/train_adder_standard.py` | Standard GPT for 3-digit addition |
| `config/train_adder_tropical.py` | Tropical GPT for 3-digit addition |
| `config/train_variable_*.py` | Variable-length training configs |
| `config/train_reversed_*.py` | Reversed digit order configs |
| `config/train_binary_*.py` | Binary addition configs |

### Experiment Runners
| File | Description |
|------|-------------|
| `run_cuda_experiments.sh` | CUDA experiment runner (RTX 3080) |
| `run_experiment.py` | Generic experiment runner |

### Evaluation Scripts
| File | Description |
|------|-------------|
| `eval_variable.py` | Variable-length evaluation |
| `eval_reversed.py` | Reversed digit evaluation |
| `evaluate_binary.py` | Binary addition evaluation |

---

## 7. Reproduction

### Quick Start (MPS / Mac)

```bash
# Setup
uv venv && uv pip install torch numpy transformers datasets tiktoken tqdm

# Run experiments (non-tropical are fast)
PYTHONUNBUFFERED=1 .venv/bin/python train_arithmetic.py \
    --max_iters=50000 --eval_interval=5000 \
    --tropical_attention=False --use_abacus=True \
    --out_dir=out-abacus-50k

# Compare all checkpoints
.venv/bin/python compare_experiments.py
```

### CUDA (RTX 3080 or similar)

```bash
# Setup and run all 4 configurations
./run_cuda_experiments.sh

# Or individually with larger batch size
PYTHONUNBUFFERED=1 .venv/bin/python train_arithmetic.py \
    --max_iters=10000 --batch_size=64 \
    --tropical_attention=True --use_abacus=True \
    --out_dir=out-tropigpt-10k-cuda
```

### Experiment Configurations

| Flag | Description |
|------|-------------|
| `--tropical_attention=True` | Enable Tropical (Max-Plus) attention |
| `--use_abacus=True` | Enable Abacus (significance) embeddings |
| `--use_muon=True` | Enable Muon optimizer |
| `--reverse_digits=True` | LSB-first digit order |
| `--batch_size=N` | Batch size (64 for CUDA, 16-64 for MPS) |
| `--max_iters=N` | Training iterations (need 10k+ for learning) |

---

## 8. Current Status (December 29, 2025)

### Completed Experiments (MPS - Mac M4)

| Experiment | Iterations | Status | Best Result |
|------------|------------|--------|-------------|
| Baseline | 2k, 10k, 50k | ✅ Done | 98%/96%/85% (3/5/10-digit) |
| Abacus | 2k, 10k, 50k | ✅ Done | **99%/97%/96%** (3/5/10-digit) |
| Tropical | 2k | ✅ Done | 0% (but 2x per-digit accuracy) |
| TropiGPT | 2k | ✅ Done | 0% |

### Pending Experiments (CUDA - RTX 3080)

| Experiment | Iterations | Status | Notes |
|------------|------------|--------|-------|
| Tropical | 10k | ⏳ Pending | Too slow on MPS (~13h) |
| TropiGPT | 10k | ⏳ Pending | Too slow on MPS |
| All configs | 10k | ⏳ Pending | `./run_cuda_experiments.sh` |

### Key Finding So Far

**Abacus embeddings provide significant improvement:**
- 96% vs 85% accuracy on 10-digit addition (11% absolute improvement)
- Achieves near-perfect accuracy (99%) on 3-digit with 50k iterations
- Faster convergence than baseline

**OOD generalization remains unsolved:**
- 0% accuracy on 15-20 digit for ALL configurations
- Positional overfitting is the core problem

---

## 9. Key Takeaway

> **Abacus embeddings (significance-based positional encoding) significantly improve arithmetic learning**, achieving 96% accuracy on 10-digit addition vs 85% for baseline. However, **OOD length generalization remains unsolved** - all models fail at 15+ digits due to positional overfitting.
>
> **Tropical attention needs further evaluation on CUDA** - MPS experiments were too slow to complete. Early results (2k iterations) showed 2x improvement in per-digit accuracy but 0% exact match.

---

## References

1. Karpathy, A. "nanoGPT" - https://github.com/karpathy/nanoGPT
2. Tropical Semiring - Max-Plus algebra in optimization and neural networks
3. Length Generalization in Transformers - Various works on positional encoding limitations
