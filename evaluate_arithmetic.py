"""
Evaluate trained models on arithmetic tasks.
Tests In-Distribution (3-digit) vs Out-of-Distribution (6-digit) generalization.
"""

import os
import pickle
import random
import torch
from model import GPTConfig, GPT

# Configuration
STANDARD_CHECKPOINT = 'out-adder-standard/ckpt.pt'
TROPICAL_CHECKPOINT = 'out-adder-tropical/ckpt.pt'

# Device setup (prioritize MPS for Apple Silicon)
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

# Number of test samples per condition
NUM_TEST_SAMPLES = 500


def load_vocab():
    """Load vocabulary from meta.pkl"""
    meta_path = 'data/adder/meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']


def generate_addition_problem(num_digits):
    """Generate a single addition problem."""
    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    result = a + b

    # Zero-pad result to fixed length
    result_str = str(result).zfill(num_digits + 1)

    prompt = f"{a}+{b}="
    answer = f"{result_str}\n"
    full = prompt + answer

    return prompt, answer, full


def load_model(checkpoint_path):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']

    # Create model with same config
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # For tropical models, set temperature to near-zero for hard attention
    if getattr(gptconf, 'tropical_attention', False):
        print(f"  Tropical model detected, setting temperature to 0.01 for inference")
        for block in model.transformer.h:
            if hasattr(block.attn, 'temperature'):
                block.attn.temperature = 0.01

    return model


def encode(text, stoi):
    """Encode text to token ids."""
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    """Decode token ids to text."""
    return ''.join([itos[i] for i in ids])


@torch.no_grad()
def generate_answer(model, prompt, stoi, itos, max_new_tokens=10):
    """Generate answer using greedy decoding."""
    model.eval()

    # Encode prompt
    input_ids = encode(prompt, stoi)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate tokens one at a time (greedy)
    generated = []
    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = model(x)
        # Get logits for the last position
        logits = logits[:, -1, :]
        # Greedy: pick token with highest probability
        next_token = logits.argmax(dim=-1)
        next_token_item = next_token.item()
        generated.append(next_token_item)

        # Check for end token (newline)
        if itos[next_token_item] == '\n':
            break

        # Append and continue
        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return decode(generated, itos)


def evaluate_model(model, num_digits, stoi, itos, num_samples):
    """Evaluate model on addition problems of given digit length."""
    if model is None:
        return None

    correct = 0
    total = 0

    for _ in range(num_samples):
        prompt, expected_answer, _ = generate_addition_problem(num_digits)
        generated_answer = generate_answer(model, prompt, stoi, itos)

        # Check if answer matches (strip whitespace for comparison)
        expected = expected_answer.strip()
        generated = generated_answer.strip()

        if generated == expected:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    return accuracy


def main():
    random.seed(42)

    # Load vocabulary
    stoi, itos = load_vocab()
    print(f"Vocabulary size: {len(stoi)}")

    # Load models
    print("\n" + "="*60)
    standard_model = load_model(STANDARD_CHECKPOINT)
    tropical_model = load_model(TROPICAL_CHECKPOINT)
    print("="*60)

    if standard_model is None and tropical_model is None:
        print("\nNo trained models found!")
        print("Please train the models first:")
        print("  python train.py config/train_adder_standard.py")
        print("  python train.py config/train_adder_tropical.py")
        return

    # Evaluate on In-Distribution (3-digit)
    print(f"\n{'='*60}")
    print("IN-DISTRIBUTION EVALUATION (3-digit additions)")
    print(f"{'='*60}")

    if standard_model is not None:
        acc_standard_3d = evaluate_model(standard_model, 3, stoi, itos, NUM_TEST_SAMPLES)
        print(f"Standard GPT:  {acc_standard_3d:.1f}% ({int(acc_standard_3d * NUM_TEST_SAMPLES / 100)}/{NUM_TEST_SAMPLES})")
    else:
        acc_standard_3d = None
        print("Standard GPT:  [not trained]")

    if tropical_model is not None:
        acc_tropical_3d = evaluate_model(tropical_model, 3, stoi, itos, NUM_TEST_SAMPLES)
        print(f"Tropical GPT:  {acc_tropical_3d:.1f}% ({int(acc_tropical_3d * NUM_TEST_SAMPLES / 100)}/{NUM_TEST_SAMPLES})")
    else:
        acc_tropical_3d = None
        print("Tropical GPT:  [not trained]")

    # Evaluate on Out-of-Distribution (6-digit)
    print(f"\n{'='*60}")
    print("OUT-OF-DISTRIBUTION EVALUATION (6-digit additions)")
    print(f"{'='*60}")

    if standard_model is not None:
        acc_standard_6d = evaluate_model(standard_model, 6, stoi, itos, NUM_TEST_SAMPLES)
        print(f"Standard GPT:  {acc_standard_6d:.1f}% ({int(acc_standard_6d * NUM_TEST_SAMPLES / 100)}/{NUM_TEST_SAMPLES})")
    else:
        acc_standard_6d = None
        print("Standard GPT:  [not trained]")

    if tropical_model is not None:
        acc_tropical_6d = evaluate_model(tropical_model, 6, stoi, itos, NUM_TEST_SAMPLES)
        print(f"Tropical GPT:  {acc_tropical_6d:.1f}% ({int(acc_tropical_6d * NUM_TEST_SAMPLES / 100)}/{NUM_TEST_SAMPLES})")
    else:
        acc_tropical_6d = None
        print("Tropical GPT:  [not trained]")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Generalization Gap (3-digit -> 6-digit)")
    print(f"{'='*60}")

    if acc_standard_3d is not None and acc_standard_6d is not None:
        gap_standard = acc_standard_3d - acc_standard_6d
        print(f"Standard GPT:  {acc_standard_3d:.1f}% -> {acc_standard_6d:.1f}% (gap: {gap_standard:.1f}%)")

    if acc_tropical_3d is not None and acc_tropical_6d is not None:
        gap_tropical = acc_tropical_3d - acc_tropical_6d
        print(f"Tropical GPT:  {acc_tropical_3d:.1f}% -> {acc_tropical_6d:.1f}% (gap: {gap_tropical:.1f}%)")

    if (acc_standard_6d is not None and acc_tropical_6d is not None):
        print(f"\nOOD Improvement (Tropical - Standard): {acc_tropical_6d - acc_standard_6d:+.1f}%")


if __name__ == '__main__':
    main()
