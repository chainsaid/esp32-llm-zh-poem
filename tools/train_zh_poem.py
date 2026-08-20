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

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)

# Hyperparameters
SEED = 1337
BATCH_SIZE = 32
# Validation loss bottoms out near step 2500 and climbs after, but do NOT
# shorten this to ~3000 to "stop at the minimum": MAX_STEPS also sets the
# cosine decay horizon, so a shorter run decays the LR faster and lands on a
# worse optimum (3000-step schedule bottomed at val 4.4259, the 6000-step one
# at 4.3865, both at step ~2500). The best-checkpoint restore below already
# discards the overfit tail, so the longer, slower-decaying schedule is
# strictly better here -- it only costs wall time (~7 min on an Arc 130T).
MAX_STEPS = 6000
LEARNING_RATE = 3e-3
MIN_LR = 1e-4
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 500
EVAL_ITERS = 40

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
    decay_ratio = min(1.0, (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def build_packed_stream(samples, vocab_map, unk_id, bos_id, eos_id, seed=SEED):
    """Concatenates every <s>...</s>-wrapped sample into one long token stream.
    Random seq_len windows are then cut from this stream at train time, so a
    ~63-token average sample no longer needs padding out to seq_len=128 —
    padding was wasting roughly half of every training batch."""
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    stream = []
    for i in order:
        s = samples[i]
        stream.append(bos_id)
        stream.extend(vocab_map.get(c, unk_id) for c in s)
        stream.append(eos_id)
    return stream

def make_batch_fn(stream, seq_len, batch_size, device):
    """Keeps the whole token stream resident on the target device and gathers
    each batch there. Building batches as Python lists and copying them over
    every step stalls a GPU on host->device transfers; at ~2.6M tokens the
    stream is only ~21MB as int64, so it just lives on the device.

    Note the batch offsets come from the device RNG (seeded by
    torch.manual_seed), so a run is reproducible on a given backend but the
    exact batch sequence differs between CPU and XPU."""
    data = torch.tensor(stream, dtype=torch.long, device=device)
    max_start = data.numel() - seq_len - 1
    window = torch.arange(seq_len + 1, device=device)

    def get_batch():
        starts = torch.randint(0, max_start, (batch_size,), device=device)
        t = data[starts[:, None] + window[None, :]]
        return t[:, :-1], t[:, 1:]
    return get_batch

def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Build Tokenizer and load vocab (built from the full corpus; char
    # frequency statistics don't leak prediction labels across the split)
    vocab = build_tokenizer()
    vocab_map = {t: i for i, t in enumerate(vocab)}
    vocab_size = len(vocab)
    unk_id = vocab_map.get("<unk>", 0)
    bos_id = vocab_map.get("<s>", 1)
    eos_id = vocab_map.get("</s>", 2)

    # 2. Load samples, split by poem (not by sample) so a poem's several
    # conditioned variants never end up split across train and val
    train_samples = get_training_text_samples(split="train")
    val_samples = get_training_text_samples(split="val")

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

    train_stream = build_packed_stream(train_samples, vocab_map, unk_id, bos_id, eos_id, seed=SEED)
    val_stream = build_packed_stream(val_samples, vocab_map, unk_id, bos_id, eos_id, seed=SEED)
    print(f"Packed token stream: {len(train_stream):,} train tokens, {len(val_stream):,} val tokens.")

    get_train_batch = make_batch_fn(train_stream, args.seq_len, BATCH_SIZE, device)
    get_val_batch = make_batch_fn(val_stream, args.seq_len, BATCH_SIZE, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    # IPEX is only useful on the CPU path here. Intel GPU support has been
    # upstream in PyTorch since 2.5 (the torch.xpu backend), so on XPU the
    # stock build already handles the device and ipex.optimize adds nothing.
    if device == "cpu":
        try:
            import intel_extension_for_pytorch as ipex
            print("Applying Intel Extension for PyTorch (IPEX) CPU optimizations...")
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
        except ImportError:
            print("IPEX not installed; using stock oneDNN CPU pipeline.")
        except Exception as e:
            print(f"IPEX optimization failed, falling back to stock pipeline: {e}")

    # bfloat16 autocast on CPU, CUDA and XPU alike. fp16 autocast on CUDA
    # needs a GradScaler (grads silently underflow to zero without one), and
    # bf16 has fp32's dynamic range so no scaler is required on any backend.
    # Intel Arc (Xe-LPG/Xe2) has native bf16 matrix engines, so leaving XPU
    # at fp32 would give up most of the GPU's throughput.
    use_amp = device in ("cpu", "cuda", "xpu")
    amp_dtype = torch.bfloat16

    @torch.no_grad()
    def estimate_val_loss():
        model.eval()
        losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_val_batch()
            if use_amp:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    print("Starting training...")
    start_time = time.time()

    # This model overfits well before MAX_STEPS (each poem is seen many times:
    # the corpus is ~10.8k poems expanded into ~43k conditioned samples), so
    # the last step is not the best step. Keep the lowest-val-loss weights and
    # export those instead of whatever the final step happened to land on.
    best_val = float("inf")
    best_state = None
    best_step = 0

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_train_batch()

        if use_amp:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_loss = estimate_val_loss()
            elapsed = time.time() - start_time
            marker = ""
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                marker = "  <- best"
            print(f"Step {step:4d}/{MAX_STEPS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f} | Elapsed: {elapsed:.1f}s{marker}")
        elif step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{MAX_STEPS} | Train Loss: {loss.item():.4f} | LR: {lr:.6f} | Elapsed: {elapsed:.1f}s")

    print("Training finished successfully!")

    # Roll back to the best-validation weights before sampling and exporting.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from step {best_step} (val loss {best_val:.4f}); "
              f"final step was {val_loss:.4f}.")

    # 4. Generate sample poem on PC
    model.eval()
    with torch.no_grad():
        prompt_text = "主题：明月 体裁：五绝\n"
        prompt_ids = [bos_id] + [vocab_map.get(c, unk_id) for c in prompt_text]

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

    # 5. Save raw PyTorch checkpoint
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(ckpt_dir, "poem_model.pt")
    torch.save(model.state_dict(), ckpt_file)
    print(f"Saved PyTorch checkpoint to {ckpt_file}")

    # 6. Export binary model
    export_bin = os.path.join(REPO_ROOT, "data", "poem_model.bin")
    export_model(model, export_bin)
    print("Model and Tokenizer exported to data/ ready for ESP32-S3 deployment!")

if __name__ == "__main__":
    train()
