"""
Compare all experiment checkpoints by evaluating them on the same test sets.
"""
import torch
from model import GPTConfig, GPT
from data.arithmetic_dataset import VOCAB_SIZE, CHAR_TO_ID, ID_TO_CHAR


def compute_significance(text, reverse=False):
    """Compute significance IDs for a text string."""
    sig = []
    current_num = []

    for ch in text:
        if ch.isdigit():
            current_num.append(ch)
        else:
            if current_num:
                n = len(current_num)
                if reverse:
                    sig.extend([i + 1 for i in range(n)])
                else:
                    sig.extend([n - i for i in range(n)])
                current_num = []
            sig.append(0)

    if current_num:
        n = len(current_num)
        if reverse:
            sig.extend([i + 1 for i in range(n)])
        else:
            sig.extend([n - i for i in range(n)])

    return sig

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def generate_test_problem(num_digits, reverse=False):
    """Generate a test problem with specific digit count."""
    import random
    min_val = 10 ** (num_digits - 1) if num_digits > 1 else 0
    max_val = 10 ** num_digits - 1
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    c = a + b

    if reverse:
        a_str = str(a)[::-1]
        b_str = str(b)[::-1]
        c_str = str(c)[::-1]
    else:
        a_str = str(a)
        b_str = str(b)
        c_str = str(c)

    prompt = f"{a_str} + {b_str} = "
    answer = c_str
    return prompt, answer, (a, b, c)


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, name, num_samples=100):
    """Evaluate a checkpoint on various digit lengths."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {ckpt_path}")
    print('='*60)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    config = checkpoint.get('config', {})

    print(f"Model config: n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, n_embd={model_args['n_embd']}")
    print(f"Tropical: {model_args.get('tropical_attention', False)}, Abacus: {model_args.get('use_abacus', False)}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    # Create model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    use_abacus = model_args.get('use_abacus', False)
    reverse = config.get('reverse_digits', False)
    seq_length = model_args.get('block_size', 256)

    # Evaluate on different digit lengths
    print(f"\nExact Match Accuracy (n={num_samples} per digit length):")
    print("-" * 40)

    results = {}
    for num_digits in [3, 5, 10, 15, 20]:
        correct = 0
        total = 0

        for _ in range(num_samples):
            prompt, answer, _ = generate_test_problem(num_digits, reverse)
            full_problem = prompt + answer + "\n"

            if len(full_problem) > seq_length:
                continue

            tokens = [CHAR_TO_ID[ch] for ch in full_problem]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

            # Create targets (same as input shifted by 1, or just use input for getting all logits)
            # We pass targets to get logits for ALL positions, not just the last one
            targets = input_ids.clone()

            if use_abacus:
                sig = compute_significance(full_problem, reverse)
                significance_ids = torch.tensor([sig], dtype=torch.long, device=device)
                logits, _ = model(input_ids, targets=targets, significance_ids=significance_ids)
            else:
                logits, _ = model(input_ids, targets=targets)

            # Find "=" position
            equals_pos = None
            for i, tok in enumerate(tokens):
                if tok == CHAR_TO_ID['=']:
                    equals_pos = i
                    break

            if equals_pos is None:
                continue

            answer_start = equals_pos + 2
            predicted_tokens = []
            for i in range(len(answer)):
                pred_pos = answer_start + i - 1
                if pred_pos >= 0 and pred_pos < logits.shape[1]:
                    pred_token = logits[0, pred_pos, :].argmax().item()
                    predicted_tokens.append(pred_token)

            predicted_str = ''.join(ID_TO_CHAR.get(t, '?') for t in predicted_tokens)

            if predicted_str == answer:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[num_digits] = acc
        label = "ID" if num_digits <= 10 else "OOD"
        print(f"  {num_digits:2d}-digit ({label}): {acc*100:5.1f}%  ({correct}/{total})")

    return results


def main():
    import os

    # Auto-discover checkpoints
    experiments = []
    checkpoint_dirs = [
        # 2k iteration experiments (MPS)
        ("out-baseline/ckpt.pt", "Baseline (2k)"),
        ("out-abacus/ckpt.pt", "Abacus (2k)"),
        ("out-tropical/ckpt.pt", "Tropical (2k)"),
        ("out-tropigpt/ckpt.pt", "TropiGPT (2k)"),
        # 10k iteration experiments (MPS)
        ("out-baseline-10k/ckpt.pt", "Baseline (10k)"),
        ("out-abacus-10k/ckpt.pt", "Abacus (10k)"),
        ("out-tropical-10k/ckpt.pt", "Tropical (10k)"),
        ("out-tropigpt-10k/ckpt.pt", "TropiGPT (10k)"),
        # 10k iteration experiments (CUDA)
        ("out-baseline-10k-cuda/ckpt.pt", "Baseline (10k CUDA)"),
        ("out-abacus-10k-cuda/ckpt.pt", "Abacus (10k CUDA)"),
        ("out-tropical-10k-cuda/ckpt.pt", "Tropical (10k CUDA)"),
        ("out-tropigpt-10k-cuda/ckpt.pt", "TropiGPT (10k CUDA)"),
        # 50k iteration experiments
        ("out-tropigpt-50k/ckpt.pt", "TropiGPT (50k)"),
        ("out-tropigpt-50k-cuda/ckpt.pt", "TropiGPT (50k CUDA)"),
    ]

    for ckpt_path, name in checkpoint_dirs:
        if os.path.exists(ckpt_path):
            experiments.append((ckpt_path, name))

    all_results = {}
    for ckpt_path, name in experiments:
        try:
            results = evaluate_checkpoint(ckpt_path, name, num_samples=100)
            all_results[name] = results
        except Exception as e:
            print(f"\nError evaluating {name}: {e}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Exact Match Accuracy (%)")
    print("="*70)
    print(f"{'Configuration':<35} {'3-dig':>7} {'5-dig':>7} {'10-dig':>7} {'15-dig':>7} {'20-dig':>7}")
    print("-"*70)

    for name, results in all_results.items():
        row = f"{name:<35}"
        for d in [3, 5, 10, 15, 20]:
            acc = results.get(d, 0) * 100
            row += f" {acc:6.1f}%"
        print(row)

    print("-"*70)
    print("ID = In-Distribution (≤10 digits), OOD = Out-of-Distribution (>10 digits)")


if __name__ == "__main__":
    main()
