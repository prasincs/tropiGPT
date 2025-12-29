
import torch
out_dir = 'out-variable-tropical'
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
tropical_attention = True
tropical_temperature = 1.0
tropical_min_temperature = 0.1
learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
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
