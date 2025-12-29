"""
Binary Addition Dataset Generator

Generates binary addition problems for training GPT models.
Binary format makes carry propagation explicit:
- 1011 + 0110 = 10001  (11 + 6 = 17 in decimal)

The tropical attention hypothesis: max operations may better
capture the discrete binary logic of carry propagation.
"""

import os
import pickle
import random
import numpy as np

# Configuration
NUM_BITS = 4  # 4-bit binary numbers (0-15) - easier task
# Note: 4-bit has only 16*16=256 unique problems, so we use all of them
# and repeat to get enough data
NUM_TRAIN = 50000
NUM_VAL = 5000

# Characters: 0, 1, +, =, newline
chars = ['0', '1', '+', '=', '\n']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def int_to_binary(n, bits):
    """Convert integer to fixed-width binary string."""
    return format(n, f'0{bits}b')

def generate_binary_addition_problem(num_bits):
    """Generate a binary addition problem string."""
    max_val = (1 << num_bits) - 1  # 2^bits - 1

    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    result = a + b

    # Result can be up to num_bits + 1 bits
    result_bits = num_bits + 1

    a_bin = int_to_binary(a, num_bits)
    b_bin = int_to_binary(b, num_bits)
    result_bin = int_to_binary(result, result_bits)

    return f"{a_bin}+{b_bin}={result_bin}\n"

def encode(s):
    """Convert string to list of integers."""
    return [stoi[c] for c in s]

def decode(l):
    """Convert list of integers to string."""
    return ''.join([itos[i] for i in l])

def main():
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    print(f"Generating {NUM_BITS}-bit binary addition dataset...")
    print(f"Vocabulary: {chars}")
    print(f"Vocab size: {len(chars)}")

    # Generate all possible problems for this bit width
    max_val = (1 << NUM_BITS) - 1
    all_problems = []
    for a in range(max_val + 1):
        for b in range(max_val + 1):
            result = a + b
            result_bits = NUM_BITS + 1
            a_bin = int_to_binary(a, NUM_BITS)
            b_bin = int_to_binary(b, NUM_BITS)
            result_bin = int_to_binary(result, result_bits)
            all_problems.append(f"{a_bin}+{b_bin}={result_bin}\n")

    print(f"Total unique problems: {len(all_problems)}")

    # Shuffle and split into train/val
    random.shuffle(all_problems)
    num_unique = len(all_problems)

    # Use 80% for train, 20% for val (or repeat if we need more)
    train_unique = all_problems[:int(num_unique * 0.8)]
    val_unique = all_problems[int(num_unique * 0.8):]

    # Repeat to get desired dataset size
    train_problems = []
    while len(train_problems) < NUM_TRAIN:
        train_problems.extend(train_unique)
    train_problems = train_problems[:NUM_TRAIN]

    val_problems = []
    while len(val_problems) < NUM_VAL:
        val_problems.extend(val_unique)
    val_problems = val_problems[:NUM_VAL]

    random.shuffle(train_problems)
    random.shuffle(val_problems)

    # Convert to lists
    train_problems = list(train_problems)
    val_problems = list(val_problems)

    # Join all problems
    train_data = ''.join(train_problems)
    val_data = ''.join(val_problems)

    # Show examples
    print(f"\nExample problems:")
    for i in range(5):
        prob = train_problems[i].strip()
        parts = prob.replace('+', ' + ').replace('=', ' = ')
        a_bin, b_bin, r_bin = prob.split('+')[0], prob.split('+')[1].split('=')[0], prob.split('=')[1]
        a_dec = int(a_bin, 2)
        b_dec = int(b_bin, 2)
        r_dec = int(r_bin, 2)
        print(f"  {prob}  ({a_dec} + {b_dec} = {r_dec})")

    # Calculate sequence length
    # Format: XXXXXXXX+XXXXXXXX=XXXXXXXXX\n
    # = num_bits + 1 + num_bits + 1 + (num_bits+1) + 1 = 3*num_bits + 4
    seq_len = 3 * NUM_BITS + 4
    print(f"\nSequence length per problem: {seq_len}")
    print(f"Block size needed: {seq_len}")

    # Encode
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    print(f"\nTrain set: {len(train_ids):,} tokens ({len(train_problems):,} problems)")
    print(f"Val set: {len(val_ids):,} tokens ({len(val_problems):,} problems)")

    # Export
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # Save metadata
    meta = {
        'vocab_size': len(chars),
        'itos': itos,
        'stoi': stoi,
        'num_bits': NUM_BITS,
        'seq_len': seq_len,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {os.path.dirname(os.path.abspath(__file__))}/")
    print("  train.bin")
    print("  val.bin")
    print("  meta.pkl")

if __name__ == '__main__':
    main()
