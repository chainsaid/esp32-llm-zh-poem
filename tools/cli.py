"""
Unified CLI Toolkit for ESP32 Chinese Poetry LLM
Provides streamlined commands for corpus preparation, tokenizer building,
model training, font atlas generation, binary export, and PC-side inference testing.
"""

import os
import sys
import argparse
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

def cmd_download(args):
    """Download and clean poetry corpus."""
    from download_corpus import download_tang_poetry
    print(f"[*] Downloading Tang poetry corpus (max_volumes={args.volumes})...")
    download_tang_poetry(max_volumes=args.volumes)

def cmd_tokenizer(args):
    """Build vocabulary and tokenizer."""
    from build_zh_tokenizer import build_tokenizer
    print(f"[*] Building tokenizer (vocab_size={args.vocab_size})...")
    build_tokenizer(vocab_size=args.vocab_size)

def cmd_train(args):
    """Train transformer model and export weights."""
    from train_zh_poem import train
    print("[*] Starting LLM training pipeline...")
    train()

def cmd_font(args):
    """Generate 16x16 glyph font atlas."""
    from build_zh_font import build_font_atlas, DEFAULT_VOCAB, DEFAULT_OUT_C, DEFAULT_OUT_H, pick_font_path
    print("[*] Generating 16x16 Chinese font atlas for ESP32 LCD...")
    font_path = pick_font_path(args.font)
    build_font_atlas(
        vocab_path=DEFAULT_VOCAB,
        out_c_path=DEFAULT_OUT_C,
        out_h_path=DEFAULT_OUT_H,
        font_path=font_path,
        verbose=args.verbose
    )

def cmd_export(args):
    """Export checkpoint to binary model."""
    import torch
    from model import Transformer, ModelArgs
    from export_zh_model import export_model

    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "poem_model.pt")
    out_bin = os.path.join(REPO_ROOT, "data", "poem_model.bin")
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        return

    print(f"[*] Loading PyTorch weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    vocab_size = state_dict["tok_embeddings.weight"].shape[0]

    margs = ModelArgs()
    margs.vocab_size = vocab_size
    model = Transformer(margs)
    model.load_state_dict(state_dict)
    export_model(model, out_bin)

def cmd_generate(args):
    """PC-side poem generation."""
    from generate_zh_poem import (
        load_pc_model_and_vocab,
        generate_rhymed_poem
    )

    model, vocab_map, idx_to_token, margs, rhyme_map, rhyme_group_index = load_pc_model_and_vocab()
    
    prompts = [args.prompt] if args.prompt else [
        "主题：明月 体裁：五绝\n",
        "主题：春风 体裁：七绝\n",
        "主题：边塞 体裁：七律\n",
        "主题：江南 体裁：七绝\n",
        "主题：孤舟 体裁：五绝\n"
    ]
    
    count = max(1, args.count)
    print(f"[*] Generating {count} poem(s) (Temperature: {args.temperature}, Top-p: {args.top_p})...\n")

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        if not prompt.endswith("\n") and prompt:
            prompt += "\n"
        
        start_t = time.time()
        poem, first_rhyme, r_group = generate_rhymed_poem(
            model, vocab_map, idx_to_token, margs, rhyme_map, rhyme_group_index,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_rhyme_constraint=not args.no_rhyme,
            enforce_length=not args.no_enforce_length
        )
        elapsed = (time.time() - start_t) * 1000
        prompt_label = prompt.replace("\n", "")
        print(f"--- [Sample {i + 1}/{count}] {prompt_label} ({elapsed:.1f} ms) ---")
        print(poem)
        print()

def cmd_infinite(args):
    """PC-side simulation of the infinite line-by-line poem stream."""
    from generate_zh_poem import load_pc_model_and_vocab, generate_infinite_poem

    model, vocab_map, idx_to_token, margs, rhyme_map, rhyme_group_index = load_pc_model_and_vocab()

    meter = 7 if args.meter == 7 else 5
    print(f"[*] Streaming an unbounded poem (theme={args.theme!r}, start meter={meter}言, "
          f"switching every {args.switch_every} lines) -- Ctrl+C to stop.\n")

    gen = generate_infinite_poem(
        model, vocab_map, idx_to_token, margs, rhyme_map, rhyme_group_index,
        theme=args.theme, meter_chars=meter, temperature=args.temperature, top_p=args.top_p,
        enable_rhyme_constraint=not args.no_rhyme)

    send_value = None
    try:
        for i in range(1, args.lines + 1):
            line = gen.send(send_value)
            print(line)
            send_value = None
            if args.switch_every > 0 and i % args.switch_every == 0:
                meter = 7 if meter == 5 else 5
                send_value = meter
                print(f"    [-- switching to {meter}言 --]")
    except KeyboardInterrupt:
        print("\n[*] Stopped.")

def cmd_clean(args):
    """Clean Python cache files and temporary artifacts."""
    import shutil
    cleaned = 0
    ignored_dirs = {".venv", "venv", ".git", ".cache", "build", "managed_components"}
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for d in list(dirs):
            if d == "__pycache__":
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"Removed directory: {full_path}")
                    cleaned += 1
                except Exception as e:
                    print(f"Failed to remove {full_path}: {e}")
        for f in files:
            if f.endswith(".pyc") or f.endswith(".pyo"):
                full_path = os.path.join(root, f)
                try:
                    os.remove(full_path)
                    print(f"Removed file: {full_path}")
                    cleaned += 1
                except Exception as e:
                    print(f"Failed to remove {full_path}: {e}")
    print(f"[*] Clean completed. Removed {cleaned} cache item(s).")

def main():
    parser = argparse.ArgumentParser(
        description="ESP32 Chinese Poetry LLM - Unified Tools & Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # download
    p_dl = subparsers.add_parser("download", help="Download and clean poetry corpus from chinese-poetry")
    p_dl.add_argument("--volumes", type=int, default=12, help="Number of Tang poetry volumes to fetch (default: 12)")
    p_dl.set_defaults(func=cmd_download)

    # tokenizer
    p_tok = subparsers.add_parser("tokenizer", help="Build character vocabulary and tokenizer binary")
    p_tok.add_argument("--vocab-size", type=int, default=2048, help="Vocabulary size (default: 2048)")
    p_tok.set_defaults(func=cmd_tokenizer)

    # train
    p_train = subparsers.add_parser("train", help="Train transformer model and export binary weights")
    p_train.set_defaults(func=cmd_train)

    # font
    p_font = subparsers.add_parser("font", help="Generate 16x16 Chinese dot-matrix font atlas (C code)")
    p_font.add_argument("--font", default=None, help="Explicit TTF/TTC font path")
    p_font.add_argument("--verbose", action="store_true", help="Print detailed font building logs")
    p_font.set_defaults(func=cmd_font)

    # export
    p_exp = subparsers.add_parser("export", help="Export PyTorch checkpoint to data/poem_model.bin")
    p_exp.set_defaults(func=cmd_export)

    # generate
    p_gen = subparsers.add_parser("generate", help="Run PC-side inference simulation")
    p_gen.add_argument("--prompt", default="", help="Poetry prompt, e.g. '主题：明月 体裁：五绝'")
    p_gen.add_argument("--count", type=int, default=3, help="Number of poems to generate")
    p_gen.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature (default: 0.75)")
    p_gen.add_argument("--top-p", type=float, default=0.9, help="Nucleus top-p sampling (default: 0.9)")
    p_gen.add_argument("--no-rhyme", action="store_true", help="Disable rhyme constraint masking")
    p_gen.add_argument("--no-enforce-length", action="store_true", help="Disable hard length enforcement")
    p_gen.set_defaults(func=cmd_generate)

    # infinite
    p_inf = subparsers.add_parser("infinite", help="Stream an unbounded line-by-line poem (previous line -> next prompt)")
    p_inf.add_argument("--theme", default="明月", help="Theme to seed the first line only (default: 明月)")
    p_inf.add_argument("--meter", type=int, default=5, choices=[5, 7], help="Starting meter: 5 (五言) or 7 (七言)")
    p_inf.add_argument("--lines", type=int, default=20, help="Number of lines to stream (default: 20)")
    p_inf.add_argument("--switch-every", type=int, default=8, help="Toggle meter every N lines, 0 to disable (default: 8)")
    p_inf.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature (default: 0.75)")
    p_inf.add_argument("--top-p", type=float, default=0.9, help="Nucleus top-p sampling (default: 0.9)")
    p_inf.add_argument("--no-rhyme", action="store_true", help="Disable rhyme constraint masking")
    p_inf.set_defaults(func=cmd_infinite)

    # clean
    p_clean = subparsers.add_parser("clean", help="Clean Python pycache and temporary files")
    p_clean.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)

if __name__ == "__main__":
    main()
