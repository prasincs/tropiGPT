#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Tropical GPT Experiment

Tests generalization across multiple digit lengths and provides detailed analysis.

Usage:
    python evaluate_generalization.py [--samples N] [--seed S] [--verbose]

Example:
    python evaluate_generalization.py --samples 200 --verbose
"""

import os
import sys
import pickle
import random
import argparse
from collections import defaultdict
import torch

from model import GPTConfig, GPT

# ============================================================================
# Configuration
# ============================================================================

STANDARD_CHECKPOINT = 'out-adder-standard/ckpt.pt'
TROPICAL_CHECKPOINT = 'out-adder-tropical/ckpt.pt'

# Digit lengths to test (training was on 3-digit)
DIGIT_LENGTHS = [2, 3, 4, 5, 6, 7, 8]
TRAINING_DIGITS = 3

# Device setup
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


# ============================================================================
# Utility Functions
# ============================================================================

def load_vocab():
    """Load vocabulary from meta.pkl"""
    meta_path = 'data/adder/meta.pkl'
    if not os.path.exists(meta_path):
        print("Error: data/adder/meta.pkl not found. Run data/adder/prepare.py first.")
        sys.exit(1)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']


def load_model(checkpoint_path, name="Model"):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"  {name}: checkpoint not found at {checkpoint_path}")
        return None

    print(f"  {name}: loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint['model_args']

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()

    # For tropical models, set temperature to near-zero for hard attention
    is_tropical = getattr(gptconf, 'tropical_attention', False)
    if is_tropical:
        for block in model.transformer.h:
            if hasattr(block.attn, 'temperature'):
                block.attn.temperature = 0.01

    iter_num = checkpoint.get('iter_num', 'unknown')
    val_loss = checkpoint.get('best_val_loss', 'unknown')
    print(f"           iterations: {iter_num}, best val loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"           iterations: {iter_num}")

    return model, is_tropical


def generate_problem(num_digits):
    """Generate a single addition problem."""
    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    result = a + b

    # Zero-pad result to match training format (num_digits + 1 for potential carry)
    answer = str(result).zfill(num_digits + 1)
    prompt = f"{a}+{b}="

    return a, b, result, prompt, answer


def encode(text, stoi):
    """Encode text to token ids."""
    return [stoi[ch] for ch in text if ch in stoi]


def decode(ids, itos):
    """Decode token ids to text."""
    return ''.join([itos[i] for i in ids])


@torch.no_grad()
def generate_answer(model, prompt, stoi, itos, max_new_tokens=15):
    """Generate answer using greedy decoding."""
    input_ids = encode(prompt, stoi)
    if not input_ids:
        return ""

    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    generated = []
    for _ in range(max_new_tokens):
        if x.size(1) > model.config.block_size:
            break

        logits, _ = model(x)
        logits = logits[:, -1, :]
        next_token = logits.argmax(dim=-1)
        next_token_item = next_token.item()
        generated.append(next_token_item)

        # Stop at newline
        if itos[next_token_item] == '\n':
            break

        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    result = decode(generated, itos).strip()
    return result


def analyze_error(a, b, expected, predicted):
    """Analyze what type of error occurred."""
    if predicted == expected:
        return "correct"

    # Check if it's a numeric prediction
    try:
        pred_num = int(predicted)
    except ValueError:
        return "invalid_format"

    exp_num = int(expected)
    diff = pred_num - exp_num

    # Off by powers of 10 often indicates carry errors
    if abs(diff) in [10, 100, 1000, 10000, 100000, 1000000]:
        return "carry_error"

    # Off by 1 in any digit
    if abs(diff) < 10:
        return "small_error"

    # Check if digits are swapped or truncated
    if len(predicted) != len(expected):
        return "length_error"

    return "other_error"


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model_on_length(model, num_digits, stoi, itos, num_samples, verbose=False):
    """Evaluate model on problems of a specific digit length."""
    if model is None:
        return None

    correct = 0
    errors = defaultdict(list)

    for i in range(num_samples):
        a, b, result, prompt, expected = generate_problem(num_digits)
        predicted = generate_answer(model, prompt, stoi, itos)

        if predicted == expected:
            correct += 1
        else:
            error_type = analyze_error(a, b, expected, predicted)
            errors[error_type].append({
                'a': a, 'b': b,
                'expected': expected,
                'predicted': predicted,
                'prompt': prompt
            })

    accuracy = correct / num_samples * 100
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': num_samples,
        'errors': dict(errors)
    }


def run_full_evaluation(model, model_name, stoi, itos, num_samples, verbose=False):
    """Run evaluation across all digit lengths."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    if model is None:
        print("  Model not available")
        return None

    results = {}
    for num_digits in DIGIT_LENGTHS:
        result = evaluate_model_on_length(model, num_digits, stoi, itos, num_samples, verbose)
        results[num_digits] = result

        marker = " (training)" if num_digits == TRAINING_DIGITS else ""
        print(f"  {num_digits}-digit{marker}: {result['accuracy']:5.1f}% ({result['correct']}/{result['total']})")

    return results


def print_error_analysis(results, model_name, num_examples=3):
    """Print detailed error analysis."""
    print(f"\n{'='*60}")
    print(f"Error Analysis: {model_name}")
    print(f"{'='*60}")

    for num_digits in DIGIT_LENGTHS:
        if results is None or num_digits not in results:
            continue

        errors = results[num_digits]['errors']
        if not errors:
            continue

        print(f"\n  {num_digits}-digit errors:")
        for error_type, examples in errors.items():
            print(f"    {error_type}: {len(examples)} cases")
            for ex in examples[:num_examples]:
                print(f"      {ex['prompt']}{ex['expected']} (got: {ex['predicted']})")


def print_comparison_table(standard_results, tropical_results, num_samples):
    """Print side-by-side comparison table."""
    print(f"\n{'='*70}")
    print("COMPARISON TABLE: Standard vs Tropical GPT")
    print(f"{'='*70}")
    print(f"{'Digits':<8} {'Standard':>12} {'Tropical':>12} {'Difference':>12} {'Winner':>10}")
    print("-" * 70)

    for num_digits in DIGIT_LENGTHS:
        std_acc = standard_results[num_digits]['accuracy'] if standard_results else None
        trop_acc = tropical_results[num_digits]['accuracy'] if tropical_results else None

        std_str = f"{std_acc:.1f}%" if std_acc is not None else "N/A"
        trop_str = f"{trop_acc:.1f}%" if trop_acc is not None else "N/A"

        if std_acc is not None and trop_acc is not None:
            diff = trop_acc - std_acc
            diff_str = f"{diff:+.1f}%"
            if diff > 2:
                winner = "Tropical"
            elif diff < -2:
                winner = "Standard"
            else:
                winner = "Tie"
        else:
            diff_str = "N/A"
            winner = "N/A"

        marker = " *" if num_digits == TRAINING_DIGITS else ""
        print(f"{num_digits}-digit{marker:<3} {std_str:>12} {trop_str:>12} {diff_str:>12} {winner:>10}")

    print("-" * 70)
    print("* = training distribution")


def print_generalization_summary(standard_results, tropical_results):
    """Print summary of generalization performance."""
    print(f"\n{'='*70}")
    print("GENERALIZATION SUMMARY")
    print(f"{'='*70}")

    if standard_results:
        id_acc = standard_results[TRAINING_DIGITS]['accuracy']
        ood_accs = [standard_results[d]['accuracy'] for d in DIGIT_LENGTHS if d > TRAINING_DIGITS]
        avg_ood = sum(ood_accs) / len(ood_accs) if ood_accs else 0
        gap = id_acc - avg_ood
        print(f"\nStandard GPT:")
        print(f"  In-distribution ({TRAINING_DIGITS}-digit):     {id_acc:.1f}%")
        print(f"  Out-of-distribution (>{TRAINING_DIGITS}-digit): {avg_ood:.1f}% (avg)")
        print(f"  Generalization gap:            {gap:.1f}%")

    if tropical_results:
        id_acc = tropical_results[TRAINING_DIGITS]['accuracy']
        ood_accs = [tropical_results[d]['accuracy'] for d in DIGIT_LENGTHS if d > TRAINING_DIGITS]
        avg_ood = sum(ood_accs) / len(ood_accs) if ood_accs else 0
        gap = id_acc - avg_ood
        print(f"\nTropical GPT:")
        print(f"  In-distribution ({TRAINING_DIGITS}-digit):     {id_acc:.1f}%")
        print(f"  Out-of-distribution (>{TRAINING_DIGITS}-digit): {avg_ood:.1f}% (avg)")
        print(f"  Generalization gap:            {gap:.1f}%")

    # Hypothesis test
    if standard_results and tropical_results:
        print(f"\n{'='*70}")
        print("HYPOTHESIS TEST")
        print(f"{'='*70}")
        print("\nH0: Tropical attention does NOT improve OOD generalization")
        print("H1: Tropical attention DOES improve OOD generalization")

        std_ood = sum(standard_results[d]['accuracy'] for d in DIGIT_LENGTHS if d > TRAINING_DIGITS)
        trop_ood = sum(tropical_results[d]['accuracy'] for d in DIGIT_LENGTHS if d > TRAINING_DIGITS)
        n_ood = len([d for d in DIGIT_LENGTHS if d > TRAINING_DIGITS])

        improvement = (trop_ood - std_ood) / n_ood

        print(f"\nAverage OOD improvement: {improvement:+.1f}%")

        if improvement > 5:
            print("\nConclusion: STRONG EVIDENCE for H1")
            print("Tropical attention shows meaningful OOD improvement.")
        elif improvement > 2:
            print("\nConclusion: WEAK EVIDENCE for H1")
            print("Tropical attention shows some OOD improvement.")
        elif improvement > -2:
            print("\nConclusion: NO SIGNIFICANT DIFFERENCE")
            print("Both models perform similarly on OOD tasks.")
        else:
            print("\nConclusion: EVIDENCE AGAINST H1")
            print("Standard attention actually performs better on OOD tasks.")


def generate_ascii_chart(standard_results, tropical_results):
    """Generate ASCII bar chart of results."""
    print(f"\n{'='*70}")
    print("ACCURACY BY DIGIT LENGTH (ASCII Chart)")
    print(f"{'='*70}")

    max_width = 40

    for num_digits in DIGIT_LENGTHS:
        std_acc = standard_results[num_digits]['accuracy'] if standard_results else 0
        trop_acc = tropical_results[num_digits]['accuracy'] if tropical_results else 0

        std_bar = int(std_acc / 100 * max_width)
        trop_bar = int(trop_acc / 100 * max_width)

        marker = "*" if num_digits == TRAINING_DIGITS else " "
        print(f"\n{num_digits}-digit{marker}")
        print(f"  Std: {'#' * std_bar}{'.' * (max_width - std_bar)} {std_acc:.1f}%")
        print(f"  Tro: {'#' * trop_bar}{'.' * (max_width - trop_bar)} {trop_acc:.1f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Tropical GPT generalization')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of test samples per digit length (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed error examples')
    parser.add_argument('--errors', type=int, default=3,
                        help='Number of error examples to show (default: 3)')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("TROPICAL GPT GENERALIZATION EVALUATION")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Samples per digit length: {args.samples}")
    print(f"Training digit length: {TRAINING_DIGITS}")
    print(f"Test digit lengths: {DIGIT_LENGTHS}")

    # Load vocabulary
    stoi, itos = load_vocab()
    print(f"Vocabulary size: {len(stoi)}")

    # Load models
    print("\nLoading models...")
    standard_model, _ = load_model(STANDARD_CHECKPOINT, "Standard GPT") if os.path.exists(STANDARD_CHECKPOINT) else (None, False)
    tropical_model, _ = load_model(TROPICAL_CHECKPOINT, "Tropical GPT") if os.path.exists(TROPICAL_CHECKPOINT) else (None, False)

    if standard_model is None and tropical_model is None:
        print("\nNo trained models found! Please train first:")
        print("  python train.py config/train_adder_standard.py")
        print("  python train.py config/train_adder_tropical.py")
        return

    # Run evaluations
    standard_results = run_full_evaluation(
        standard_model, "Standard GPT", stoi, itos, args.samples, args.verbose
    )
    tropical_results = run_full_evaluation(
        tropical_model, "Tropical GPT", stoi, itos, args.samples, args.verbose
    )

    # Print comparison
    if standard_results and tropical_results:
        print_comparison_table(standard_results, tropical_results, args.samples)
        generate_ascii_chart(standard_results, tropical_results)
        print_generalization_summary(standard_results, tropical_results)

    # Print error analysis if verbose
    if args.verbose:
        if standard_results:
            print_error_analysis(standard_results, "Standard GPT", args.errors)
        if tropical_results:
            print_error_analysis(tropical_results, "Tropical GPT", args.errors)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
