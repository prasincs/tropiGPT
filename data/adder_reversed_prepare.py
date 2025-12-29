
"""Generate reversed-digit addition dataset"""
import os, pickle, random
import numpy as np

random.seed(42)
np.random.seed(42)

NUM_EXAMPLES = 50000
VAL_RATIO = 0.1

chars = ['0','1','2','3','4','5','6','7','8','9','+','=','\n']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def generate_problem(num_digits):
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    result = a + b
    # Reverse all numbers
    a_rev = str(a)[::-1]
    b_rev = str(b)[::-1]
    r_rev = str(result).zfill(num_digits + 1)[::-1]
    return f"{a_rev}+{b_rev}={r_rev}\n"

problems = [generate_problem(3) for _ in range(NUM_EXAMPLES)]
random.shuffle(problems)

n_val = int(NUM_EXAMPLES * VAL_RATIO)
val_data = ''.join(problems[:n_val])
train_data = ''.join(problems[n_val:])

train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)

os.makedirs('data/adder_reversed', exist_ok=True)
train_ids.tofile('data/adder_reversed/train.bin')
val_ids.tofile('data/adder_reversed/val.bin')

with open('data/adder_reversed/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': len(chars), 'stoi': stoi, 'itos': itos}, f)

print(f"Created reversed dataset: {len(train_ids)} train, {len(val_ids)} val tokens")
print(f"Sample: {problems[0]!r}")
