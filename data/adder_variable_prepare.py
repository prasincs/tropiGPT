
"""Generate variable-length addition dataset"""
import os, pickle, random
import numpy as np

random.seed(42)
np.random.seed(42)

NUM_EXAMPLES = 100000  # More examples for variable lengths
VAL_RATIO = 0.1

chars = ['0','1','2','3','4','5','6','7','8','9','+','=','\n',' ']  # Added space for padding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def generate_problem():
    num_digits = random.choice([2, 3, 4, 5])
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    result = a + b
    return f"{a}+{b}={result}\n"

problems = [generate_problem() for _ in range(NUM_EXAMPLES)]
random.shuffle(problems)

n_val = int(NUM_EXAMPLES * VAL_RATIO)
val_data = ''.join(problems[:n_val])
train_data = ''.join(problems[n_val:])

train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)

os.makedirs('data/adder_variable', exist_ok=True)
train_ids.tofile('data/adder_variable/train.bin')
val_ids.tofile('data/adder_variable/val.bin')

with open('data/adder_variable/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': len(chars), 'stoi': stoi, 'itos': itos}, f)

print(f"Created variable dataset: {len(train_ids)} train, {len(val_ids)} val tokens")
for i in range(5):
    print(f"  Sample {i+1}: {problems[i]!r}")
