# Train a standard GPT on 8-bit binary addition task
# Auto-detects best device (CUDA > MPS > CPU)

import torch

out_dir = 'out-binary-standard'
eval_interval = 250
eval_iters = 100
log_interval = 10

# Only save when validation improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'binary-adder'
wandb_run_name = 'standard-gpt'

# Data
dataset = 'binary_adder'
gradient_accumulation_steps = 1
batch_size = 64
# 4-bit binary: XXXX+XXXX=XXXXX\n = 16 chars
block_size = 16

# Small model for arithmetic task
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Standard attention (not tropical)
tropical_attention = False

# Learning rate settings - 4-bit is simpler, fewer iters needed
learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Auto-detect best device and settings
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
