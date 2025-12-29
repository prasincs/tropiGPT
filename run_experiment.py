#!/usr/bin/env python3
"""
Autonomous Experiment Runner for Tropical GPT Research

Runs experiments from the ROADMAP and sends notifications via ntfy.sh

Usage:
    python run_experiment.py --experiment reversed_digits
    python run_experiment.py --experiment variable_length
    python run_experiment.py --experiment binary_addition
    python run_experiment.py --list
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime

NTFY_TOPIC = "https://ntfy.sh/8c549cf5-bd9d-48e1-b595-6cba439d3faa"

def notify(message, title="TropiGPT", priority="default", tags=""):
    """Send notification via ntfy.sh"""
    cmd = [
        "curl", "-s",
        "-H", f"Title: {title}",
        "-H", f"Priority: {priority}",
        "-H", f"Tags: {tags}",
        "-d", message,
        NTFY_TOPIC
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
    except Exception as e:
        print(f"Notification failed: {e}")

def run_command(cmd, description):
    """Run a command and return success/failure"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)

    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    return result.returncode == 0, elapsed, result.stdout

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

EXPERIMENTS = {}

def register_experiment(name, description):
    def decorator(func):
        EXPERIMENTS[name] = {"func": func, "description": description}
        return func
    return decorator


@register_experiment("reversed_digits", "Train on reversed digit order (little-endian)")
def experiment_reversed_digits():
    """
    Priority 1.2: Reversed Digit Order
    Present numbers least-significant digit first to align carry with generation.
    """
    notify("Starting experiment: Reversed Digits", tags="rocket")

    # Step 1: Create reversed data generator
    notify("Step 1/4: Creating reversed data generator", priority="low")

    reversed_prepare = '''
"""Generate reversed-digit addition dataset"""
import os, pickle, random
import numpy as np

random.seed(42)
np.random.seed(42)

NUM_EXAMPLES = 50000
VAL_RATIO = 0.1

chars = ['0','1','2','3','4','5','6','7','8','9','+','=','\\n']
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
    return f"{a_rev}+{b_rev}={r_rev}\\n"

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
'''

    with open('data/adder_reversed_prepare.py', 'w') as f:
        f.write(reversed_prepare)

    success, elapsed, output = run_command(
        ".venv/bin/python data/adder_reversed_prepare.py",
        "Generate reversed dataset"
    )
    if not success:
        notify("FAILED: Could not generate reversed dataset", priority="high", tags="x")
        return False

    # Step 2: Create configs for reversed experiments
    notify("Step 2/4: Creating training configs", priority="low")

    std_config = '''
import torch
out_dir = 'out-reversed-standard'
eval_interval = 250
eval_iters = 100
log_interval = 50
always_save_checkpoint = False
wandb_log = False
dataset = 'adder_reversed'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 16
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
tropical_attention = False
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

if torch.cuda.is_available():
    device = 'cuda'
    compile = True
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif torch.backends.mps.is_available():
    device = 'mps'
    compile = False
    dtype = 'float32'
else:
    device = 'cpu'
    compile = False
    dtype = 'float32'
'''

    trop_config = std_config.replace(
        "out_dir = 'out-reversed-standard'",
        "out_dir = 'out-reversed-tropical'"
    ).replace(
        "tropical_attention = False",
        "tropical_attention = True\ntropical_temperature = 1.0\ntropical_min_temperature = 0.1"
    ).replace(
        "max_iters = 5000",
        "max_iters = 10000"
    ).replace(
        "lr_decay_iters = 5000",
        "lr_decay_iters = 10000"
    )

    with open('config/train_reversed_standard.py', 'w') as f:
        f.write(std_config)
    with open('config/train_reversed_tropical.py', 'w') as f:
        f.write(trop_config)

    # Step 3: Train both models
    notify("Step 3/4: Training Standard GPT on reversed digits...", priority="low")
    success, elapsed, _ = run_command(
        ".venv/bin/python train.py config/train_reversed_standard.py",
        "Train Standard GPT (reversed)"
    )
    if not success:
        notify("FAILED: Standard training failed", priority="high", tags="x")
        return False
    notify(f"Standard GPT trained in {elapsed:.1f}s", priority="low")

    notify("Step 3/4: Training Tropical GPT on reversed digits...", priority="low")
    success, elapsed, _ = run_command(
        ".venv/bin/python train.py config/train_reversed_tropical.py",
        "Train Tropical GPT (reversed)"
    )
    if not success:
        notify("FAILED: Tropical training failed", priority="high", tags="x")
        return False
    notify(f"Tropical GPT trained in {elapsed:.1f}s", priority="low")

    # Step 4: Evaluate
    notify("Step 4/4: Evaluating models...", priority="low")

    eval_script = '''
import torch, pickle, random
from model import GPTConfig, GPT

random.seed(42)
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

with open('data/adder_reversed/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    if getattr(config, 'tropical_attention', False):
        for block in model.transformer.h:
            if hasattr(block.attn, 'temperature'):
                block.attn.temperature = 0.1
    return model

def test_model(model, num_digits, n=100):
    correct = 0
    for _ in range(n):
        a = random.randint(10**(num_digits-1), 10**num_digits - 1)
        b = random.randint(10**(num_digits-1), 10**num_digits - 1)
        result = str(a + b).zfill(num_digits + 1)[::-1]
        prompt = f"{str(a)[::-1]}+{str(b)[::-1]}="

        x = torch.tensor([[stoi[c] for c in prompt]], device=device)
        gen = []
        for _ in range(num_digits + 2):
            logits, _ = model(x)
            next_tok = logits[:, -1, :].argmax(dim=-1)
            ch = itos[next_tok.item()]
            if ch == "\\n": break
            gen.append(ch)
            x = torch.cat([x, next_tok.unsqueeze(0)], dim=1)
        if ''.join(gen) == result:
            correct += 1
    return correct / n * 100

import os
results = {}
for name, path in [('Standard', 'out-reversed-standard/ckpt.pt'),
                   ('Tropical', 'out-reversed-tropical/ckpt.pt')]:
    if os.path.exists(path):
        model = load_model(path)
        results[name] = {}
        for digits in [2, 3, 4, 5, 6]:
            acc = test_model(model, digits, n=200)
            results[name][digits] = acc
            print(f"{name} {digits}-digit: {acc:.1f}%")

print("\\n=== RESULTS ===")
for name in results:
    print(f"{name}: 3-digit={results[name].get(3,0):.1f}%, 6-digit={results[name].get(6,0):.1f}%")
'''

    with open('eval_reversed.py', 'w') as f:
        f.write(eval_script)

    success, elapsed, output = run_command(
        ".venv/bin/python eval_reversed.py",
        "Evaluate reversed models"
    )

    # Parse results and notify
    notify(
        f"COMPLETED: Reversed Digits Experiment\n\n{output}",
        title="Experiment Complete",
        priority="high",
        tags="white_check_mark"
    )

    return True


@register_experiment("variable_length", "Train on variable digit lengths (2-5)")
def experiment_variable_length():
    """
    Priority 1.1: Variable-Length Training Data
    Train on mixed digit lengths to prevent positional memorization.
    """
    notify("Starting experiment: Variable Length Training", tags="rocket")

    # Step 1: Create variable-length data generator
    notify("Step 1/4: Creating variable-length dataset", priority="low")

    var_prepare = '''
"""Generate variable-length addition dataset"""
import os, pickle, random
import numpy as np

random.seed(42)
np.random.seed(42)

NUM_EXAMPLES = 100000  # More examples for variable lengths
VAL_RATIO = 0.1

chars = ['0','1','2','3','4','5','6','7','8','9','+','=','\\n',' ']  # Added space for padding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def generate_problem():
    num_digits = random.choice([2, 3, 4, 5])
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    result = a + b
    return f"{a}+{b}={result}\\n"

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
'''

    with open('data/adder_variable_prepare.py', 'w') as f:
        f.write(var_prepare)

    success, _, _ = run_command(
        ".venv/bin/python data/adder_variable_prepare.py",
        "Generate variable-length dataset"
    )
    if not success:
        notify("FAILED: Could not generate variable dataset", priority="high", tags="x")
        return False

    # Create configs
    notify("Step 2/4: Creating training configs", priority="low")

    std_config = '''
import torch
out_dir = 'out-variable-standard'
eval_interval = 500
eval_iters = 100
log_interval = 100
always_save_checkpoint = False
wandb_log = False
dataset = 'adder_variable'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 24  # Longer for variable lengths
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
tropical_attention = False
learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

if torch.cuda.is_available():
    device = 'cuda'
    compile = True
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif torch.backends.mps.is_available():
    device = 'mps'
    compile = False
    dtype = 'float32'
else:
    device = 'cpu'
    compile = False
    dtype = 'float32'
'''

    trop_config = std_config.replace(
        "out_dir = 'out-variable-standard'",
        "out_dir = 'out-variable-tropical'"
    ).replace(
        "tropical_attention = False",
        "tropical_attention = True\ntropical_temperature = 1.0\ntropical_min_temperature = 0.1"
    ).replace(
        "max_iters = 10000",
        "max_iters = 15000"
    ).replace(
        "lr_decay_iters = 10000",
        "lr_decay_iters = 15000"
    )

    with open('config/train_variable_standard.py', 'w') as f:
        f.write(std_config)
    with open('config/train_variable_tropical.py', 'w') as f:
        f.write(trop_config)

    # Train models
    notify("Step 3/4: Training models (this may take a while)...", priority="low")

    run_command(".venv/bin/python train.py config/train_variable_standard.py", "Train Standard")
    run_command(".venv/bin/python train.py config/train_variable_tropical.py", "Train Tropical")

    # Evaluate
    notify("Step 4/4: Evaluating...", priority="low")

    eval_script = '''
import torch, pickle, random
from model import GPTConfig, GPT

random.seed(42)
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

with open('data/adder_variable/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt['model_args'])
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    if getattr(config, 'tropical_attention', False):
        for block in model.transformer.h:
            if hasattr(block.attn, 'temperature'):
                block.attn.temperature = 0.1
    return model

def test_model(model, num_digits, n=100):
    correct = 0
    for _ in range(n):
        a = random.randint(10**(num_digits-1), 10**num_digits - 1)
        b = random.randint(10**(num_digits-1), 10**num_digits - 1)
        result = str(a + b)
        prompt = f"{a}+{b}="

        x = torch.tensor([[stoi[c] for c in prompt]], device=device)
        gen = []
        for _ in range(15):
            logits, _ = model(x)
            next_tok = logits[:, -1, :].argmax(dim=-1)
            ch = itos[next_tok.item()]
            if ch == "\\n": break
            gen.append(ch)
            x = torch.cat([x, next_tok.unsqueeze(0)], dim=1)
        if ''.join(gen) == result:
            correct += 1
    return correct / n * 100

import os
print("Variable-Length Training Results")
print("="*50)
print("Training included: 2, 3, 4, 5 digit numbers")
print("="*50)

for name, path in [('Standard', 'out-variable-standard/ckpt.pt'),
                   ('Tropical', 'out-variable-tropical/ckpt.pt')]:
    if os.path.exists(path):
        model = load_model(path)
        print(f"\\n{name} GPT:")
        for digits in [2, 3, 4, 5, 6, 7, 8]:
            acc = test_model(model, digits, n=200)
            marker = "(train)" if digits <= 5 else "(OOD)"
            print(f"  {digits}-digit {marker}: {acc:.1f}%")
'''

    with open('eval_variable.py', 'w') as f:
        f.write(eval_script)

    success, _, output = run_command(".venv/bin/python eval_variable.py", "Evaluate")

    notify(
        f"COMPLETED: Variable Length Experiment\n\n{output}",
        title="Experiment Complete",
        priority="high",
        tags="white_check_mark"
    )

    return True


@register_experiment("list", "List all available experiments")
def list_experiments():
    print("\nAvailable experiments:")
    print("="*60)
    for name, info in EXPERIMENTS.items():
        if name != "list":
            print(f"  {name:20s} - {info['description']}")
    print("="*60)
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Tropical GPT experiments')
    parser.add_argument('--experiment', '-e', type=str, required=True,
                        help='Experiment to run (use --experiment list to see all)')
    parser.add_argument('--notify-start', action='store_true',
                        help='Send notification when starting')
    args = parser.parse_args()

    if args.experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {args.experiment}")
        print("Use --experiment list to see available experiments")
        sys.exit(1)

    if args.notify_start:
        notify(f"Starting experiment: {args.experiment}", tags="rocket")

    start_time = time.time()

    try:
        success = EXPERIMENTS[args.experiment]["func"]()
        elapsed = time.time() - start_time

        if success:
            print(f"\n{'='*60}")
            print(f"Experiment '{args.experiment}' completed in {elapsed:.1f}s")
            print('='*60)
        else:
            print(f"\nExperiment '{args.experiment}' FAILED")
            sys.exit(1)

    except Exception as e:
        notify(f"CRASHED: {args.experiment}\n\n{str(e)}", priority="urgent", tags="x")
        raise


if __name__ == '__main__':
    main()
