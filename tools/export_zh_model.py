"""
Export PyTorch Llama-2 model to standard llama2.c binary checkpoint format (data/poem_model.bin).
"""

import os
import struct
import torch
import numpy as np
from model import Transformer, ModelArgs

def serialize_fp32(file, tensor):
    """Writes a PyTorch tensor as continuous float32 binary."""
    data = tensor.detach().cpu().to(torch.float32).numpy().flatten()
    data.tofile(file)

def export_model(model: Transformer, output_path: str = "data/poem_model.bin"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    args: ModelArgs = model.args
    
    with open(output_path, "wb") as f:
        # 1. Header (7 * 4 bytes int32 = 28 bytes)
        # dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        # positive vocab_size denotes shared weights (token_embeddings tied with output)
        shared_weights = 1
        vocab_val = args.vocab_size if shared_weights else -args.vocab_size
        header = struct.pack(
            "7i",
            args.dim,
            args.hidden_dim,
            args.n_layers,
            args.n_heads,
            args.n_kv_heads,
            vocab_val,
            args.seq_len
        )
        f.write(header)
        
        # 2. Token Embedding Table
        serialize_fp32(f, model.tok_embeddings.weight)
        
        # 3. Layer weights
        # rms_att_weight
        for layer in model.layers:
            serialize_fp32(f, layer.attention_norm.weight)
            
        # wq
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wq.weight)
            
        # wk
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wk.weight)
            
        # wv
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wv.weight)
            
        # wo
        for layer in model.layers:
            serialize_fp32(f, layer.attention.wo.weight)
            
        # rms_ffn_weight
        for layer in model.layers:
            serialize_fp32(f, layer.ffn_norm.weight)
            
        # w1
        for layer in model.layers:
            serialize_fp32(f, layer.feed_forward.w1.weight)
            
        # w2
        for layer in model.layers:
            serialize_fp32(f, layer.feed_forward.w2.weight)
            
        # w3
        for layer in model.layers:
            serialize_fp32(f, layer.feed_forward.w3.weight)
            
        # 4. Final RMS Norm weight
        serialize_fp32(f, model.norm.weight)
        
        # 5. RoPE freq_cis real & imag padding (p->seq_len * head_size)
        head_size = args.dim // args.n_heads
        freq_padding = np.zeros(args.seq_len * head_size, dtype=np.float32)
        freq_padding.tofile(f)
        
    file_size = os.path.getsize(output_path)
    print(f"Successfully exported model checkpoint to {output_path} ({file_size} bytes, {file_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    args = ModelArgs()
    model = Transformer(args)
    export_model(model)
