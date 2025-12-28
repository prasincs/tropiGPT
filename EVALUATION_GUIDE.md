# Tropical GPT Evaluation Guide

## Quick Start

```bash
# Run the comprehensive evaluation
.venv/bin/python evaluate_generalization.py

# With more samples for statistical significance
.venv/bin/python evaluate_generalization.py --samples 500

# With verbose error analysis
.venv/bin/python evaluate_generalization.py --samples 200 --verbose
```

## What the Evaluation Tests

### Digit Length Progression

| Digits | Type | Description |
|--------|------|-------------|
| 2-digit | Easier than training | 10-99 + 10-99 |
| **3-digit** | **Training distribution** | **100-999 + 100-999** |
| 4-digit | Mild OOD | 1000-9999 + 1000-9999 |
| 5-digit | Moderate OOD | 10000-99999 + ... |
| 6-digit | Strong OOD | 100000-999999 + ... |
| 7-digit | Extreme OOD | 1000000-9999999 + ... |
| 8-digit | Extreme OOD | 10000000-99999999 + ... |

### The Hypothesis

**Standard GPT (Sum-Product Semiring):**
- Uses softmax → distributes attention across many positions
- "Soft" weighted averaging may blur precise algorithmic steps
- Expected: Good on training distribution, degrades on OOD

**Tropical GPT (Max-Plus Semiring):**
- Uses max → sharp, discrete attention patterns
- Creates "hard pointers" to specific positions
- Expected: Better length generalization for algorithmic tasks

## Interpreting Results

### Comparison Table

```
Digits   Standard    Tropical   Difference     Winner
----------------------------------------------------------------------
2-digit     95.0%       93.5%        -1.5%        Tie
3-digit *   92.0%       91.5%        -0.5%        Tie      ← Training
4-digit     45.0%       62.0%       +17.0%    Tropical
5-digit     12.0%       38.0%       +26.0%    Tropical
6-digit      3.5%       21.0%       +17.5%    Tropical
```

**How to read:**
- `*` marks the training distribution (3-digit)
- `Difference` = Tropical - Standard (positive = Tropical wins)
- `Winner` = which model is better (>2% difference)

### Generalization Gap

```
Standard GPT:
  In-distribution (3-digit):     92.0%
  Out-of-distribution (>3-digit): 15.1% (avg)
  Generalization gap:            76.9%   ← BAD: huge drop

Tropical GPT:
  In-distribution (3-digit):     91.5%
  Out-of-distribution (>3-digit): 40.3% (avg)
  Generalization gap:            51.2%   ← BETTER: smaller drop
```

**Smaller gap = better generalization**

### Hypothesis Test Interpretation

| Avg OOD Improvement | Conclusion |
|---------------------|------------|
| > +5% | **STRONG EVIDENCE** for Tropical advantage |
| +2% to +5% | Weak evidence for Tropical advantage |
| -2% to +2% | No significant difference |
| < -2% | Standard actually better (hypothesis rejected) |

## Error Types

The evaluation classifies errors into categories:

| Error Type | Description | Example |
|------------|-------------|---------|
| `correct` | Prediction matches | 123+456=579 ✓ |
| `carry_error` | Off by power of 10 | Expected 1579, got 579 |
| `small_error` | Off by < 10 | Expected 579, got 578 |
| `length_error` | Wrong number of digits | Expected 1579, got 15790 |
| `invalid_format` | Non-numeric output | Expected 579, got "5?9" |
| `other_error` | Other mistakes | Expected 579, got 123 |

**Carry errors** are particularly interesting because:
- Addition requires propagating carries through digits
- Standard attention may "blur" the carry signal
- Tropical attention should maintain sharp carry propagation

## Running Additional Tests

### Test with Different Seeds
```bash
# Run with different random seeds to check consistency
.venv/bin/python evaluate_generalization.py --seed 42
.venv/bin/python evaluate_generalization.py --seed 123
.venv/bin/python evaluate_generalization.py --seed 456
```

### Detailed Error Analysis
```bash
# Show 5 error examples per category
.venv/bin/python evaluate_generalization.py --verbose --errors 5
```

### Quick Sanity Check
```bash
# Fast check with fewer samples
.venv/bin/python evaluate_generalization.py --samples 50
```

## What Success Looks Like

### Scenario 1: Hypothesis Confirmed
```
HYPOTHESIS TEST
H1: Tropical attention DOES improve OOD generalization

Average OOD improvement: +15.3%

Conclusion: STRONG EVIDENCE for H1
Tropical attention shows meaningful OOD improvement.
```

This means the Tropical semiring successfully creates sharper attention patterns that generalize better to longer sequences.

### Scenario 2: No Difference
```
Average OOD improvement: +0.8%

Conclusion: NO SIGNIFICANT DIFFERENCE
Both models perform similarly on OOD tasks.
```

This could mean:
- The task doesn't require the "sharpness" of tropical attention
- The model is too small to see the difference
- More training or different hyperparameters needed

### Scenario 3: Hypothesis Rejected
```
Average OOD improvement: -5.2%

Conclusion: EVIDENCE AGAINST H1
Standard attention actually performs better on OOD tasks.
```

This could mean:
- The smooth interpolation of softmax is actually beneficial
- Tropical attention needs different training strategy
- The temperature annealing schedule needs tuning

## Troubleshooting

### Both models have ~0% OOD accuracy
- Models may be undertrained
- Try training longer: increase `max_iters`
- Check training loss convergence

### Tropical model performs worse on training distribution
- Temperature annealing may be too aggressive
- Try higher `tropical_min_temperature` (e.g., 0.1 instead of 0.01)
- The LogSumExp approximation may need tuning

### High variance between runs
- Use more samples: `--samples 500` or `--samples 1000`
- Average results across multiple seeds

## Next Steps After Evaluation

1. **If Tropical wins:** Consider testing on other algorithmic tasks (multiplication, sorting, parity)

2. **If results are mixed:** Try tuning:
   - Model size (n_layer, n_embd)
   - Temperature annealing schedule
   - Training duration

3. **For publication:** Run with `--samples 1000` across 5 different seeds and report mean ± std
