"""
Binary Addition Evaluation Script

Evaluates trained models on binary addition at various bit widths.
Tests in-distribution (8-bit) and out-of-distribution (4-bit, 12-bit, 16-bit).
"""

import os
import pickle
import argparse
import random
import torch
from model import GPTConfig, GPT

def int_to_binary(n, bits):
    """Convert integer to fixed-width binary string."""
    return format(n, f'0{bits}b')

def generate_binary_problem(num_bits):
    """Generate a binary addition problem."""
    max_val = (1 << num_bits) - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    result = a + b
    result_bits = num_bits + 1

    a_bin = int_to_binary(a, num_bits)
    b_bin = int_to_binary(b, num_bits)
    result_bin = int_to_binary(result, result_bits)

    prompt = f"{a_bin}+{b_bin}="
    answer = result_bin

    return prompt, answer, (a, b, result)

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_args = checkpoint['model_args']
    config = GPTConfig(**config_args)
    model = GPT(config)

    state_dict = checkpoint['model']
    # Remove _orig_mod prefix if present (from torch.compile)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config

def evaluate_model(model, device, stoi, itos, num_bits, num_samples, block_size):
    """Evaluate model on binary addition problems."""
    correct = 0
    total = 0

    # Expected answer length
    answer_len = num_bits + 1  # Result has one more bit

    for _ in range(num_samples):
        prompt, expected_answer, (a, b, result) = generate_binary_problem(num_bits)

        # Check if prompt + answer fits in block_size
        full_len = len(prompt) + answer_len + 1  # +1 for newline
        if full_len > block_size:
            continue

        # Encode prompt
        prompt_ids = [stoi[c] for c in prompt]
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Generate answer
        with torch.no_grad():
            for _ in range(answer_len):
                logits, _ = model(x)
                next_token = logits[0, -1, :].argmax().item()
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

        # Decode generated answer
        generated_ids = x[0, len(prompt_ids):].tolist()
        generated = ''.join([itos[i] for i in generated_ids])

        if generated == expected_answer:
            correct += 1
        total += 1

    return correct, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100, help='Samples per bit width')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/mps/cpu)')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load vocabulary
    meta_path = 'data/binary_adder/meta.pkl'
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Run data/binary_adder/prepare.py first.")
        return

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi = meta['stoi']
    itos = meta['itos']
    training_bits = meta['num_bits']  # What the model was trained on

    print(f"Vocabulary: {list(stoi.keys())}")
    print(f"Training bit width: {training_bits}")

    # Models to evaluate
    models = [
        ('Standard GPT', 'out-binary-standard/ckpt.pt'),
        ('Tropical GPT', 'out-binary-tropical/ckpt.pt'),
    ]

    # Bit widths to test: smaller (OOD), training (ID), larger (OOD)
    bit_widths = [4, 6, 8, 10, 12]

    print(f"\n{'='*60}")
    print(f"Binary Addition Evaluation - {args.samples} samples per bit width")
    print(f"{'='*60}")

    results = {}

    for model_name, ckpt_path in models:
        if not os.path.exists(ckpt_path):
            print(f"\n{model_name}: Checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{model_name}:")
        print("-" * 40)

        model, config = load_model(ckpt_path, device)
        block_size = config.block_size

        results[model_name] = {}

        for bits in bit_widths:
            # Check if this bit width fits in block_size
            # Format: bits + 1 ('+') + bits + 1 ('=') + (bits+1) + 1 ('\n') = 3*bits + 4
            required_len = 3 * bits + 4
            if required_len > block_size:
                print(f"  {bits}-bit: SKIPPED (requires {required_len} > block_size {block_size})")
                results[model_name][bits] = (0, 0, 'skipped')
                continue

            correct, total = evaluate_model(
                model, device, stoi, itos, bits, args.samples, block_size
            )

            if total > 0:
                accuracy = 100.0 * correct / total
                ood_marker = "" if bits == training_bits else " (OOD)"
                print(f"  {bits}-bit: {correct}/{total} = {accuracy:.1f}%{ood_marker}")
                results[model_name][bits] = (correct, total, accuracy)
            else:
                print(f"  {bits}-bit: No valid samples")
                results[model_name][bits] = (0, 0, 0.0)

    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison:")
    print(f"{'='*60}")
    print(f"{'Bit Width':<12}", end="")
    for model_name in results.keys():
        print(f"{model_name:<20}", end="")
    print()
    print("-" * 52)

    for bits in bit_widths:
        print(f"{bits}-bit{' (ID)' if bits == training_bits else ' (OOD)':<12}", end="")
        for model_name in results.keys():
            if bits in results[model_name]:
                val = results[model_name][bits]
                if val[2] == 'skipped':
                    print(f"{'SKIP':<20}", end="")
                else:
                    print(f"{val[2]:.1f}%{'':<16}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

if __name__ == '__main__':
    main()
