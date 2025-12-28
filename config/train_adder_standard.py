# Train a standard GPT on the addition task
# Auto-detects best device (CUDA > MPS > CPU)

import torch

out_dir = 'out-adder-standard'
eval_interval = 250
eval_iters = 100
log_interval = 10

# Only save when validation improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'adder'
wandb_run_name = 'standard-gpt'

# Data
dataset = 'adder'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 16  # "123+456=0579\n" = 13 chars, padded

# Small model for arithmetic task
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# Standard attention (not tropical)
tropical_attention = False

# Learning rate settings
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Auto-detect best device and settings
if torch.cuda.is_available():
    device = 'cuda'
    compile = True  # torch.compile works well on CUDA
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif torch.backends.mps.is_available():
    device = 'mps'
    compile = False  # torch.compile not stable on MPS
    dtype = 'float32'  # float32 is more stable on MPS
else:
    device = 'cpu'
    compile = False
    dtype = 'float32'
