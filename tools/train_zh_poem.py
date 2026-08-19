"""
Train Classical Chinese Poetry LLM
Supports CPU (with Intel Ultra optimizations), CUDA, and XPU backends.
Exports trained model to data/poem_model.bin.
"""

import os
import time
import math
import random
import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs
from build_zh_tokenizer import build_tokenizer
from dataset import get_training_text_samples
from export_zh_model import export_model

# Hyperparameters
BATCH_SIZE = 32
MAX_STEPS = 1200
LEARNING_RATE = 3e-3
MIN_LR = 1e-4
WARMUP_STEPS = 60
WEIGHT_DECAY = 0.01

def setup_hardware_acceleration():
    """Configures device and Intel hardware optimizations."""
    if torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA CUDA GPU acceleration.")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
        print("Using Intel Arc / XPU GPU acceleration.")
    else:
        device = "cpu"
        # Intel CPU & OpenMP affinity optimizations
        num_cores = os.cpu_count() or 4
        p_cores = min(8, max(2, num_cores // 2))
        torch.set_num_threads(p_cores)
        os.environ["OMP_NUM_THREADS"] = str(p_cores)
        os.environ["MKL_NUM_THREADS"] = str(p_cores)
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["KMP_BLOCKTIME"] = "1"
        print(f"Using Intel Core Ultra hardware acceleration with {p_cores} P-Core threads.")
    return device

def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def train():
    # 1. Build Tokenizer and load vocab
    vocab = build_tokenizer()
    vocab_map = {t: i for i, t in enumerate(vocab)}
    vocab_size = len(vocab)
    unk_id = vocab_map.get("<unk>", 0)
    bos_id = vocab_map.get("<s>", 1)
    eos_id = vocab_map.get("</s>", 2)
    
    # 2. Encode samples
    samples = get_training_text_samples()
    encoded_samples = []
    for s in samples:
        tokens = [bos_id] + [vocab_map.get(c, unk_id) for c in s] + [eos_id]
        encoded_samples.append(tokens)
        
    print(f"Total encoded training samples: {len(encoded_samples)}")
    
    # 3. Model Configuration
    args = ModelArgs()
    args.dim = 96
    args.hidden_dim = 256
    args.n_layers = 5
    args.n_heads = 6
    args.n_kv_heads = 6
    args.vocab_size = vocab_size
    args.seq_len = 128
    
    device = setup_hardware_acceleration()
    model = Transformer(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized Chinese Poetry Transformer ({total_params:,} parameters, {total_params * 4 / 1024 / 1024:.2f} MB)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    
    # Check and apply Intel Extension for PyTorch (IPEX) optimizations if available
    try:
        import intel_extension_for_pytorch as ipex
        print("Applying Intel Extension for PyTorch (IPEX) optimizations...")
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16 if device == "cpu" else torch.float32)
    except (ImportError, Exception) as e:
        print("Standard Intel oneDNN native vector pipeline active.")
    
    def get_batch():
        batch_tokens = []
        for _ in range(BATCH_SIZE):
            chosen = random.choice(encoded_samples)
            if len(chosen) < args.seq_len:
                padded = chosen + [0] * (args.seq_len - len(chosen))
            else:
                start = random.randint(0, len(chosen) - args.seq_len)
                padded = chosen[start : start + args.seq_len]
            batch_tokens.append(padded)
        t = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        x = t[:, :-1]
        y = t[:, 1:].clone()
        y[y == 0] = -1  # ignore padding
        return x, y

    print("Starting training...")
    start_time = time.time()
    
    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        x, y = get_batch()
        
        # Mixed precision on CPU/GPU
        use_amp = (device == "cpu") or (device == "cuda")
        amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16
        
        if use_amp:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0 or step == MAX_STEPS:
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{MAX_STEPS} | Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {elapsed:.1f}s")
            
    print("Training finished successfully!")
    
    # 4. Generate sample poem on PC
    model.eval()
    with torch.no_grad():
        prompt_text = "床前明月光"
        prompt_ids = [bos_id] + [vocab_map.get(c, unk_id) for c in prompt_text]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        generated = list(prompt_ids)
        for _ in range(35):
            curr_input = torch.tensor([generated[-args.seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(curr_input)
            next_token_logits = logits[0, -1, :] / 0.8
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == eos_id:
                break
            generated.append(next_token)
            
        generated_poem = "".join([vocab[idx] if idx < len(vocab) else "" for idx in generated[1:]])
        print("\n--- Model Generation Sample ---")
        print(generated_poem)
        print("-------------------------------\n")
        
    # 5. Export binary model
    export_model(model, "data/poem_model.bin")
    print("Model and Tokenizer exported to data/ ready for ESP32-S3 deployment!")

if __name__ == "__main__":
    train()
