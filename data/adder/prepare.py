"""
Prepare the addition dataset for character-level modeling.
Generates 3-digit addition problems: "123+456=0579\n"
"""

import os
import pickle
import random
import numpy as np

# Configuration
NUM_EXAMPLES = 50000  # Total number of addition problems
VAL_RATIO = 0.1  # 10% for validation
SEED = 42

# Digit configuration for training data
MIN_DIGITS = 3  # 3-digit numbers (100-999)
MAX_DIGITS = 3

def generate_addition_problem(num_digits):
    """Generate a single addition problem with specified digit count."""
    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    result = a + b

    # Zero-pad result to fixed length (num_digits + 1 for carry)
    result_str = str(result).zfill(num_digits + 1)

    # Format: "123+456=0579\n"
    problem = f"{a}+{b}={result_str}\n"
    return problem

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define vocabulary: digits 0-9, '+', '=', '\n'
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '\n']
    vocab_size = len(chars)

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {chars}")

    # Generate all problems
    print(f"\nGenerating {NUM_EXAMPLES} addition problems...")
    problems = []
    for _ in range(NUM_EXAMPLES):
        num_digits = random.randint(MIN_DIGITS, MAX_DIGITS)
        problem = generate_addition_problem(num_digits)
        problems.append(problem)

    # Shuffle and split
    random.shuffle(problems)
    n_val = int(NUM_EXAMPLES * VAL_RATIO)
    val_problems = problems[:n_val]
    train_problems = problems[n_val:]

    print(f"Training examples: {len(train_problems)}")
    print(f"Validation examples: {len(val_problems)}")

    # Show a few examples
    print("\nSample training problems:")
    for i in range(5):
        print(f"  {repr(train_problems[i])}")

    # Encode to integers
    def encode(text):
        return [stoi[ch] for ch in text]

    # Concatenate all problems into single strings
    train_text = ''.join(train_problems)
    val_text = ''.join(val_problems)

    train_ids = encode(train_text)
    val_ids = encode(val_text)

    print(f"\nTrain has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # Save meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nSaved train.bin, val.bin, and meta.pkl to {os.path.dirname(__file__)}")

    # Compute expected sequence length for block_size guidance
    sample_problem = generate_addition_problem(MAX_DIGITS)
    print(f"\nSample problem length: {len(sample_problem)} chars")
    print(f"Recommended block_size: {len(sample_problem) + 3} (with padding)")

if __name__ == '__main__':
    main()
