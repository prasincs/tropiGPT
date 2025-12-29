"""
Training script for TropiGPT on arithmetic addition tasks.

Simplified for single GPU (Mac Mini M4 / MPS).

Features:
- Uses ArithmeticDataset with Abacus Embeddings
- Temperature annealing for Tropical Attention (1.0 → 0.01)
- Exact match evaluation on ID (20-digit) and OOD (100-digit) sums
- Optional Muon optimizer for internal weights

Usage:
    python train_arithmetic.py
    python train_arithmetic.py --tropical_attention=True --use_abacus=True
    python train_arithmetic.py --use_muon=True
"""

import os
import time
import math
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from model import GPTConfig, GPT
from data.arithmetic_dataset import (
    ArithmeticDataset, VOCAB_SIZE, CHAR_TO_ID, ID_TO_CHAR, decode_tokens
)

# =============================================================================
# Configuration
# =============================================================================

# I/O
out_dir = 'out-arithmetic'
eval_interval = 500
log_interval = 10
eval_iters = 50
save_checkpoint = True

# Data
min_digits = 1          # Min digits for training operands
max_digits = 10         # Max digits for training operands (curriculum)
reverse_digits = False  # LSB-first format
seq_length = 256        # Sequence length (must fit max problem size for eval)

# Model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Tropical Attention
tropical_attention = False
tropical_temperature = 1.0
tropical_min_temperature = 0.01
temperature_anneal_steps = 2000  # Linear anneal over this many steps

# Abacus Embeddings
use_abacus = False
max_significance = 32

# Optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 200

# Muon optimizer
use_muon = False
muon_lr = 0.02
muon_momentum = 0.95

# Training
batch_size = 64
gradient_accumulation_steps = 1

# System
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
dtype = 'float32'  # MPS works best with float32
compile_model = False  # torch.compile not fully supported on MPS

# =============================================================================
# Override config from command line
# =============================================================================
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) if os.path.exists('configurator.py') else None
config = {k: globals()[k] for k in config_keys}

# =============================================================================
# Setup
# =============================================================================

print(f"Device: {device}")
print(f"Tropical Attention: {tropical_attention}")
print(f"Abacus Embeddings: {use_abacus}")
print(f"Muon Optimizer: {use_muon}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# Device type for autocast
if 'cuda' in device:
    device_type = 'cuda'
elif 'mps' in device:
    device_type = 'mps'
else:
    device_type = 'cpu'

# Context manager for autocast (MPS doesn't benefit)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
if device_type == 'cuda':
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
else:
    ctx = nullcontext()

# =============================================================================
# Data Loading
# =============================================================================

print(f"\nCreating ArithmeticDataset (digits: {min_digits}-{max_digits}, reversed: {reverse_digits})")

train_dataset = ArithmeticDataset(
    min_digits=min_digits,
    max_digits=max_digits,
    reverse_digits=reverse_digits,
    seed=42,
    seq_length=seq_length,
)

# For validation, use same config but different seed
val_dataset = ArithmeticDataset(
    min_digits=min_digits,
    max_digits=max_digits,
    reverse_digits=reverse_digits,
    seed=12345,
    seq_length=seq_length,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
train_iter = iter(train_loader)
val_iter = iter(val_loader)


def get_batch(split):
    """Get a batch from the appropriate dataset."""
    global train_iter, val_iter
    if split == 'train':
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
    else:
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)

    input_ids, significance_ids, target_ids = batch
    input_ids = input_ids.to(device)
    significance_ids = significance_ids.to(device)
    target_ids = target_ids.to(device)

    return input_ids, significance_ids, target_ids


# =============================================================================
# Model Initialization
# =============================================================================

print(f"\nInitializing model...")

model_args = dict(
    vocab_size=VOCAB_SIZE,
    block_size=seq_length,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    tropical_attention=tropical_attention,
    tropical_temperature=tropical_temperature,
    use_abacus=use_abacus,
    max_significance=max_significance,
)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

if compile_model and device_type == 'cuda':
    print("Compiling model...")
    model = torch.compile(model)

# =============================================================================
# Optimizer
# =============================================================================

if use_muon:
    optimizers = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type,
        use_muon=True, muon_lr=muon_lr, muon_momentum=muon_momentum
    )
else:
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type,
        use_muon=False
    )
    optimizers = [optimizer]

# =============================================================================
# Learning Rate Schedule
# =============================================================================

def get_lr(it):
    """Cosine learning rate with warmup."""
    min_lr = learning_rate / 10
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_temperature(it):
    """Linear temperature annealing for tropical attention."""
    if it >= temperature_anneal_steps:
        return tropical_min_temperature
    # Linear interpolation from tropical_temperature to tropical_min_temperature
    progress = it / temperature_anneal_steps
    return tropical_temperature + progress * (tropical_min_temperature - tropical_temperature)


# =============================================================================
# Evaluation Functions
# =============================================================================

def generate_test_problem(num_digits, reverse=False):
    """Generate a single test problem with specified digit count."""
    import random
    min_val = 10 ** (num_digits - 1) if num_digits > 1 else 0
    max_val = 10 ** num_digits - 1

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    c = a + b

    a_str, b_str, c_str = str(a), str(b), str(c)

    if reverse:
        a_str, b_str, c_str = a_str[::-1], b_str[::-1], c_str[::-1]

    # Create prompt (without the answer)
    prompt = f"{a_str} + {b_str} = "
    answer = c_str

    return prompt, answer, (a, b, c)


def tokenize_prompt(prompt):
    """Tokenize a prompt string."""
    return [CHAR_TO_ID[ch] for ch in prompt]


def compute_significance(text, reverse=False):
    """Compute significance IDs for a text string."""
    sig = []
    current_num = []

    for ch in text:
        if ch.isdigit():
            current_num.append(ch)
        else:
            # Flush current number
            if current_num:
                n = len(current_num)
                if reverse:
                    sig.extend([i + 1 for i in range(n)])
                else:
                    sig.extend([n - i for i in range(n)])
                current_num = []
            sig.append(0)  # Non-digit

    # Flush final number
    if current_num:
        n = len(current_num)
        if reverse:
            sig.extend([i + 1 for i in range(n)])
        else:
            sig.extend([n - i for i in range(n)])

    return sig


@torch.no_grad()
def evaluate_exact_match(num_digits, num_samples=100, reverse=False):
    """
    Evaluate exact match accuracy on addition problems.

    Args:
        num_digits: Number of digits for operands
        num_samples: Number of problems to test
        reverse: Whether to use LSB-first format

    Returns:
        Accuracy (0.0 to 1.0)
    """
    model.eval()
    correct = 0
    max_seq = seq_length - 10  # Leave room for generation

    for _ in range(num_samples):
        prompt, answer, (a, b, c) = generate_test_problem(num_digits, reverse)

        # Skip if prompt is too long for model
        if len(prompt) + len(answer) + 5 > seq_length:
            continue

        # Tokenize prompt
        tokens = tokenize_prompt(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Compute significance for prompt
        sig = compute_significance(prompt, reverse)
        significance_ids = torch.tensor([sig], dtype=torch.long, device=device)

        # Generate answer tokens one by one
        generated = []
        max_answer_len = min(len(answer) + 2, seq_length - len(tokens))  # Don't exceed block size

        for _ in range(max_answer_len):
            # Check sequence length limit
            if input_ids.shape[1] >= seq_length:
                break

            # Pad significance if needed
            if significance_ids.shape[1] < input_ids.shape[1]:
                pad = torch.zeros(1, input_ids.shape[1] - significance_ids.shape[1],
                                  dtype=torch.long, device=device)
                significance_ids = torch.cat([significance_ids, pad], dim=1)

            # Forward pass
            if use_abacus:
                logits, _ = model(input_ids, significance_ids=significance_ids)
            else:
                logits, _ = model(input_ids)

            # Get next token (greedy)
            next_token = logits[0, -1, :].argmax().item()

            # Check for newline (end of answer)
            if next_token == CHAR_TO_ID['\n']:
                break

            generated.append(ID_TO_CHAR[next_token])

            # Append to sequence
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ], dim=1)

            # Update significance (new digit gets appropriate significance)
            new_sig = len(generated) if reverse else 1
            significance_ids = torch.cat([
                significance_ids,
                torch.tensor([[new_sig]], dtype=torch.long, device=device)
            ], dim=1)

        # Check exact match
        generated_str = ''.join(generated)
        if generated_str == answer:
            correct += 1

    model.train()
    return correct / num_samples


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, significance_ids, target_ids = get_batch(split)
            with ctx:
                if use_abacus:
                    logits, loss = model(input_ids, targets=target_ids, significance_ids=significance_ids)
                else:
                    logits, loss = model(input_ids, targets=target_ids)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


# =============================================================================
# Training Loop
# =============================================================================

print(f"\nStarting training...")
print(f"Max iterations: {max_iters}")
print(f"Batch size: {batch_size}")
print(f"Gradient accumulation: {gradient_accumulation_steps}")
print("-" * 60)

iter_num = 0
best_val_loss = float('inf')
t0 = time.time()

while iter_num < max_iters:

    # === Learning Rate ===
    lr = get_lr(iter_num)
    for opt in optimizers:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    # === Temperature Annealing ===
    if tropical_attention:
        temp = get_temperature(iter_num)
        model.set_tropical_temperature(temp)

    # === Evaluation ===
    if iter_num % eval_interval == 0:
        losses = estimate_loss()

        # Exact match evaluation
        acc_id = evaluate_exact_match(20, num_samples=50, reverse=reverse_digits)   # ID: 20-digit
        acc_ood = evaluate_exact_match(100, num_samples=50, reverse=reverse_digits)  # OOD: 100-digit

        temp_str = f", temp={temp:.4f}" if tropical_attention else ""
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{temp_str}")
        print(f"           exact match: ID(20-digit)={acc_id*100:.1f}%, OOD(100-digit)={acc_ood*100:.1f}%")

        # Save checkpoint
        if save_checkpoint and losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            print(f"           saved checkpoint to {out_dir}/ckpt.pt")

    # === Forward/Backward ===
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    for micro_step in range(gradient_accumulation_steps):
        input_ids, significance_ids, target_ids = get_batch('train')

        with ctx:
            if use_abacus:
                logits, loss = model(input_ids, targets=target_ids, significance_ids=significance_ids)
            else:
                logits, loss = model(input_ids, targets=target_ids)
            loss = loss / gradient_accumulation_steps

        loss.backward()

    # Gradient clipping
    if grad_clip != 0.0:
        for opt in optimizers:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    for opt in optimizers:
        opt.step()

    # === Logging ===
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, time {dt*1000:.0f}ms")

    iter_num += 1

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")

# Final evaluation
print("\nFinal Evaluation:")
print("-" * 40)
for digits in [5, 10, 20, 50, 100]:
    acc = evaluate_exact_match(digits, num_samples=100, reverse=reverse_digits)
    label = "ID" if digits <= max_digits else "OOD"
    print(f"  {digits:3d}-digit ({label}): {acc*100:.1f}%")
