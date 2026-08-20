"""
Build Chinese Classical Poetry Tokenizer & Vocabulary
Extracts character vocabulary purely from tools/poetry_corpus.json dataset.
Exports tools/vocab.json and data/poem_tok.bin in standard llama2.c binary format.
"""

import os
import struct
import json
from collections import Counter
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
VOCAB_SIZE = 2048  # Balances ESP32-S3 memory budget against corpus char coverage (~96%)

def build_tokenizer(output_bin=None, output_json=None, vocab_size=VOCAB_SIZE):
    if output_bin is None:
        output_bin = os.path.join(REPO_ROOT, "data", "poem_tok.bin")
    if output_json is None:
        output_json = os.path.join(TOOLS_DIR, "vocab.json")
    # Special tokens
    special_tokens = ["<unk>", "<s>", "</s>", "\n"]
    punctuations = ["，", "。", "！", "？", "、", "；", "：", "《", "》", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    # Extract unique characters and count frequencies purely from dataset
    samples = get_training_text_samples()
    char_counts = Counter()
    for s in samples:
        for char in s:
            if char not in special_tokens and char not in punctuations:
                char_counts[char] += 1

    vocab = list(special_tokens) + [p for p in punctuations if p not in special_tokens]
    
    # Add top characters by statistical frequency in the corpus
    for char, _ in char_counts.most_common():
        if char not in vocab:
            vocab.append(char)
        if len(vocab) >= vocab_size:
            break
            
    # Pad if needed
    dummy_idx = 0
    while len(vocab) < vocab_size:
        vocab.append(f"<extra_{dummy_idx}>")
        dummy_idx += 1
        
    vocab = vocab[:vocab_size]
    print(f"Constructed vocabulary of size {len(vocab)} from corpus statistics.")
    
    # Save vocab.json
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({t: i for i, t in enumerate(vocab)}, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON vocabulary to {output_json}")

    # Export llama2.c binary tokenizer
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)
    encoded_tokens = [t.encode('utf-8') for t in vocab]
    max_token_len = max(len(t) for t in encoded_tokens)
    
    with open(output_bin, "wb") as f:
        f.write(struct.pack("i", max_token_len))
        for i, token_bytes in enumerate(encoded_tokens):
            score = -float(i)
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)
            
    print(f"Exported binary tokenizer to {output_bin} (max_token_length={max_token_len})")
    return vocab

if __name__ == "__main__":
    build_tokenizer()
