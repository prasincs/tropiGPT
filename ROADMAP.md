# Tropical GPT Research Roadmap

## Current Status

We have implemented a comprehensive research framework for testing the Tropical Attention hypothesis on arithmetic tasks:

### ✅ Completed Implementations

| Component | File | Description |
|-----------|------|-------------|
| **ArithmeticDataset** | `data/arithmetic_dataset.py` | Infinite IterableDataset with character-level tokenization |
| **Abacus Embeddings** | `model.py` | Significance-based positional encoding (10^0, 10^1, etc.) |
| **Tropical Attention** | `model.py` | Max-Plus semiring with stable LSE trick and temperature annealing |
| **Muon Optimizer** | `model.py` | Newton-Schulz orthogonalization for weight matrices |
| **Training Script** | `train_arithmetic.py` | Full training loop with curriculum and evaluation |

### Key Insight: Evaluation Methodology

We implemented two evaluation approaches:

1. **Teacher-Forcing (Primary)**: Feed full problem to model, extract predictions after "=" token
   - Faster and more reliable
   - Directly measures learned mapping quality
   - Avoids compounding autoregressive errors

2. **Autoregressive (Secondary)**: Generate tokens one by one
   - Tests true generation capability
   - Slower, more susceptible to error propagation

---

## Priority 1: Fix Positional Overfitting

These experiments address the root cause of the current failure.

### 1.1 Variable-Length Training Data
**Difficulty**: Easy | **Impact**: High

Train on mixed digit lengths to prevent position memorization.

```python
# In data/adder/prepare.py, modify:
def generate_addition_problem():
    num_digits = random.choice([2, 3, 4, 5])  # Variable lengths
    # ... rest of generation
```

**Hypothesis**: Both models should improve on OOD, but tropical may show steeper improvement curve.

---

### 1.2 Reversed Digit Order (Little-Endian)
**Difficulty**: Easy | **Impact**: High

Present numbers least-significant digit first:
- Standard: `123+456=579`
- Reversed: `321+654=975`

This aligns carry propagation with left-to-right autoregressive generation.

```python
def reverse_digits(n):
    return str(n)[::-1]

# "321+654=975\n" instead of "123+456=579\n"
```

**Hypothesis**: Tropical attention should excel here since carries propagate in generation order.

---

### 1.3 Relative Positional Encoding (RoPE/ALiBi)
**Difficulty**: Medium | **Impact**: High

Replace absolute position embeddings with relative positional encoding.

```python
# In model.py, replace wpe with RoPE
def apply_rotary_pos_emb(q, k, cos, sin):
    # Rotary position embedding implementation
    ...
```

**Hypothesis**: Should enable both models to generalize to longer sequences. Tropical may still show advantage.

---

## Priority 2: Better Arithmetic Tasks

Test tropical attention on tasks that more directly require discrete/algorithmic reasoning.

### 2.1 Binary Addition
**Difficulty**: Easy | **Impact**: Medium

Use binary representation where carry is explicit:
- `1011+0110=10001` (11+6=17 in decimal)

**Hypothesis**: Tropical attention should show clearer advantage with explicit binary carries.

---

### 2.2 Multiplication
**Difficulty**: Medium | **Impact**: High

Multiplication requires tracking multiple partial products:
- `12*34=408`

**Hypothesis**: More complex attention patterns may benefit more from tropical sharpness.

---

### 2.3 Modular Arithmetic
**Difficulty**: Easy | **Impact**: Medium

Compute `(a + b) mod p` for small primes p.

**Hypothesis**: Discrete modular structure may align well with tropical max operations.

---

### 2.4 Sorting / Comparison
**Difficulty**: Medium | **Impact**: High

Sort a sequence of numbers:
- Input: `[5,2,8,1,9]`
- Output: `[1,2,5,8,9]`

**Hypothesis**: Sorting requires discrete comparisons (max/min) - perfect for tropical semiring.

---

## Priority 3: Architectural Variations

Explore different ways to incorporate tropical operations.

### 3.1 Hybrid Attention (Tropical + Softmax)
**Difficulty**: Medium | **Impact**: Medium

Use tropical attention in early layers, softmax in later layers (or vice versa).

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        if layer_idx < config.n_layer // 2:
            self.attn = TropicalCausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
```

**Hypothesis**: Different layers may benefit from different inductive biases.

---

### 3.2 Tropical MLP
**Difficulty**: Hard | **Impact**: High

Extend tropical operations to the MLP blocks:
- Standard: `GELU(xW1) @ W2`
- Tropical: `max(x + W1) + W2` (in log space)

**Hypothesis**: Full tropical network may unlock more compositional reasoning.

---

### 3.3 Tropical Embeddings
**Difficulty**: Medium | **Impact**: Medium

Use additive (tropical) token embeddings instead of lookup:
- Each token has a "tropical embedding" that combines additively

**Hypothesis**: May provide more compositional token representations.

---

## Priority 4: Training Dynamics

Improve how tropical models are trained.

### 4.1 Curriculum Temperature Annealing
**Difficulty**: Easy | **Impact**: Medium

Anneal temperature in stages tied to data complexity:
1. Train on 2-digit at T=1.0
2. Add 3-digit, anneal to T=0.5
3. Add 4-digit, anneal to T=0.1

**Hypothesis**: Gradual curriculum may improve convergence.

---

### 4.2 Straight-Through Estimator
**Difficulty**: Medium | **Impact**: Medium

Use hard max in forward pass, but soft gradients in backward:

```python
def tropical_max_ste(x, dim):
    hard = x.max(dim=dim).values
    soft = temperature * torch.logsumexp(x / temperature, dim=dim)
    return hard + (soft - soft.detach())  # Straight-through
```

**Hypothesis**: May enable training with truly discrete attention while maintaining gradient flow.

---

### 4.3 Gumbel-Max Tropical Attention
**Difficulty**: Medium | **Impact**: Medium

Add Gumbel noise for stochastic hard attention:

```python
def gumbel_tropical_max(x, dim, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(x)))
    return ((x + gumbel_noise) / temperature).max(dim=dim)
```

**Hypothesis**: Stochastic exploration may improve optimization landscape.

---

## Priority 5: Analysis & Interpretability

Understand what tropical attention learns differently.

### 5.1 Attention Pattern Visualization
**Difficulty**: Easy | **Impact**: Medium

Compare attention heatmaps between standard and tropical models.

```python
def visualize_attention(model, input_text):
    # Extract and plot attention weights
    ...
```

**Hypothesis**: Tropical attention should show sparser, more discrete patterns.

---

### 5.2 Probing for Carry Detection
**Difficulty**: Medium | **Impact**: Medium

Train linear probes on hidden states to detect carry signals.

**Hypothesis**: Tropical models may have more explicit carry representations.

---

### 5.3 Loss Landscape Analysis
**Difficulty**: Hard | **Impact**: Medium

Visualize the loss landscape of tropical vs standard models.

**Hypothesis**: Tropical may have sharper minima corresponding to discrete solutions.

---

## Quick Experiments (< 1 hour each)

| Experiment | Priority | Difficulty |
|------------|----------|------------|
| Variable-length training data | 1.1 | Easy |
| Reversed digit order | 1.2 | Easy |
| Binary addition | 2.1 | Easy |
| Attention visualization | 5.1 | Easy |
| Curriculum annealing | 4.1 | Easy |

---

## Medium Experiments (1-4 hours each)

| Experiment | Priority | Difficulty |
|------------|----------|------------|
| RoPE positional encoding | 1.3 | Medium |
| Multiplication task | 2.2 | Medium |
| Sorting task | 2.4 | Medium |
| Hybrid attention layers | 3.1 | Medium |
| Straight-through estimator | 4.2 | Medium |

---

## Research Experiments (1+ days)

| Experiment | Priority | Difficulty |
|------------|----------|------------|
| Tropical MLP | 3.2 | Hard |
| Loss landscape analysis | 5.3 | Hard |
| Full tropical transformer | 3.2 + 3.3 | Hard |

---

## Recommended Next Steps

### Immediate (Infrastructure Ready)

```bash
# 1. Baseline: Standard attention without Abacus
python train_arithmetic.py --max_iters=5000 --tropical_attention=False --use_abacus=False

# 2. Abacus only: Test if significance embeddings help
python train_arithmetic.py --max_iters=5000 --tropical_attention=False --use_abacus=True

# 3. Tropical only: Test tropical attention alone
python train_arithmetic.py --max_iters=5000 --tropical_attention=True --use_abacus=False

# 4. Full TropiGPT: Tropical + Abacus + Muon
python train_arithmetic.py --max_iters=5000 --tropical_attention=True --use_abacus=True --use_muon=True

# 5. Reversed digits (LSB-first): Better carry alignment
python train_arithmetic.py --max_iters=5000 --tropical_attention=True --use_abacus=True --reverse_digits=True
```

### Key Experiments

| Experiment | Command | Hypothesis |
|------------|---------|------------|
| Baseline | `--tropical_attention=False` | Standard transformer baseline |
| Abacus Embeddings | `--use_abacus=True` | Significance alignment improves generalization |
| Tropical Attention | `--tropical_attention=True` | Discrete attention helps carry propagation |
| LSB-First | `--reverse_digits=True` | Carry aligns with generation order |
| Full Stack | All flags | Combined benefits |

### Success Criteria

- **ID Accuracy (10-digit)**: Should reach >90% with good configuration
- **OOD Accuracy (20-digit)**: >50% indicates generalization
- **OOD Accuracy (100-digit)**: Any non-zero indicates strong generalization

---

## Success Metrics

For any experiment, success is measured by:

1. **OOD Generalization Gap**: `(ID accuracy) - (OOD accuracy)`
   - Smaller gap = better generalization
   - Tropical should have smaller gap than standard

2. **Length Extrapolation**: Accuracy on lengths 2x training length
   - Any non-zero accuracy = success

3. **Sample Efficiency**: Iterations to reach 90% training accuracy
   - Tropical may need more iterations due to harder optimization

4. **Per-Digit Accuracy**: Fraction of correct digits (even if full answer wrong)
   - Use `evaluate_per_digit_accuracy()` for fine-grained analysis
   - Reveals which positions are hardest (usually carry positions)

---

## Evaluation Methodology

### Teacher-Forcing Evaluation (Recommended)

```python
# From train_arithmetic.py
def evaluate_exact_match(num_digits, num_samples=100, reverse=False):
    """
    1. Feed FULL problem (including answer) to model
    2. Find "=" token position
    3. Extract predictions for tokens AFTER "="
    4. Compare to ground truth
    """
```

**Why this approach?**
- Single forward pass per problem (fast)
- No error accumulation from autoregressive sampling
- Directly measures if model learned the mapping
- Consistent with how we compute training loss

### Per-Position Analysis

```python
results = evaluate_per_digit_accuracy(num_digits=20, num_samples=100)
# Returns:
# - exact_match: Full answer accuracy
# - per_digit: Average digit accuracy
# - per_position: {0: 0.95, 1: 0.87, 2: 0.72, ...}  # Position 2 is hardest (carry!)
```

This reveals:
- Which digit positions the model struggles with
- Whether errors cluster at carry positions
- If Abacus embeddings help align carries

---

## References for Further Reading

1. **Tropical Geometry & Neural Networks**
   - Zhang et al., "Tropical Geometry of Deep Neural Networks"

2. **Length Generalization in Transformers**
   - Anil et al., "Exploring Length Generalization in Large Language Models"
   - Press et al., "Train Short, Test Long: Attention with Linear Biases"

3. **Algorithmic Reasoning**
   - Nye et al., "Show Your Work: Scratchpads for Intermediate Computation"
   - Zhou et al., "Teaching Algorithmic Reasoning via In-context Learning"

4. **Discrete Attention Mechanisms**
   - Martins & Astudillo, "From Softmax to Sparsemax"
   - Correia et al., "Adaptively Sparse Transformers"
