
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
            if ch == "\n": break
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
        print(f"\n{name} GPT:")
        for digits in [2, 3, 4, 5, 6, 7, 8]:
            acc = test_model(model, digits, n=200)
            marker = "(train)" if digits <= 5 else "(OOD)"
            print(f"  {digits}-digit {marker}: {acc:.1f}%")
