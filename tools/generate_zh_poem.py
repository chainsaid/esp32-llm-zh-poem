"""
PC-side Chinese Classical Poetry LLM Inference Simulator
Supports Theme + Form Control (五绝 / 七绝) & Constrained Rhyme Masking
(中华通韵, modern Mandarin finals via pypinyin -- NOT 平水韵, which needs
historical tone/rime data pypinyin doesn't provide)
"""

import os
import sys
import json
import time
import random
import torch
import struct
import numpy as np
import torch.nn.functional as F
from model import Transformer, ModelArgs
from pypinyin import pinyin, Style

def build_rhyme_table(vocab_map):
    def get_rhyme_group(char):
        if len(char) != 1 or char in '，。？！、；：《》 0123456789\n' or char.startswith('<'):
            return -1
        res = pinyin(char, style=Style.FINALS, heteronym=False)
        if not res or not res[0]:
            return -1
        final = res[0][0]
        if final in ['a', 'ia', 'ua']: return 1 # 麻
        if final in ['o', 'e', 'uo', 'ie', 've', 'üe']: return 2 # 歌皆
        if final in ['ai', 'uai']: return 3 # 齐微
        if final in ['ei', 'ui', 'uei', 'i']: return 4 # 支微
        if final in ['ao', 'iao']: return 5 # 萧豪
        if final in ['ou', 'iu', 'iou']: return 6 # 尤侯
        if final in ['an', 'ian', 'uan', 'üan', 'van']: return 7 # 寒先
        if final in ['en', 'in', 'un', 'ün', 'vn']: return 8 # 真文
        if final in ['ang', 'iang', 'uang']: return 9 # 江阳
        if final in ['eng', 'ing', 'ong', 'iong', 'ueng']: return 10 # 东庚
        if final in ['u']: return 11 # 鱼模
        if final in ['v', 'ü', 'er']: return 12 # 虞鱼
        return 0

    rhyme_map = {}
    for word, idx in vocab_map.items():
        rhyme_map[idx] = get_rhyme_group(word)
    return rhyme_map

def build_rhyme_group_index(rhyme_map, min_members=3):
    """Groups -> member token ids, restricted to classified groups (1-12) with
    enough members that forcing one doesn't collapse to a near-arbitrary pick."""
    groups = {}
    for token_idx, r_group in rhyme_map.items():
        if r_group > 0:
            groups.setdefault(r_group, []).append(token_idx)
    return {g: ids for g, ids in groups.items() if len(ids) >= min_members}

def load_pc_model_and_vocab(vocab_path="tools/vocab.json", checkpoint_bin="data/poem_model.bin"):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_map = json.load(f)
    idx_to_token = {i: t for t, i in vocab_map.items()}
    
    args = ModelArgs()
    args.dim = 96
    args.hidden_dim = 256
    args.n_layers = 5
    args.n_heads = 6
    args.n_kv_heads = 6
    args.vocab_size = len(vocab_map)
    args.seq_len = 128
    
    model = Transformer(args)
    
    with open(checkpoint_bin, "rb") as f:
        header = struct.unpack("7i", f.read(28))
        raw_weights = np.fromfile(f, dtype=np.float32)
        
    ptr = 0
    def take_tensor(shape):
        nonlocal ptr
        num = 1
        for s in shape: num *= s
        chunk = raw_weights[ptr : ptr + num]
        ptr += num
        return torch.from_numpy(chunk.reshape(shape)).float()

    state_dict = {}
    state_dict["tok_embeddings.weight"] = take_tensor((args.vocab_size, args.dim))
    for i in range(args.n_layers): state_dict[f"layers.{i}.attention_norm.weight"] = take_tensor((args.dim,))
    for i in range(args.n_layers): state_dict[f"layers.{i}.attention.wq.weight"] = take_tensor((args.n_heads * (args.dim // args.n_heads), args.dim))
    for i in range(args.n_layers): state_dict[f"layers.{i}.attention.wk.weight"] = take_tensor((args.n_kv_heads * (args.dim // args.n_heads), args.dim))
    for i in range(args.n_layers): state_dict[f"layers.{i}.attention.wv.weight"] = take_tensor((args.n_kv_heads * (args.dim // args.n_heads), args.dim))
    for i in range(args.n_layers): state_dict[f"layers.{i}.attention.wo.weight"] = take_tensor((args.dim, args.n_heads * (args.dim // args.n_heads)))
    for i in range(args.n_layers): state_dict[f"layers.{i}.ffn_norm.weight"] = take_tensor((args.dim,))
    for i in range(args.n_layers): state_dict[f"layers.{i}.feed_forward.w1.weight"] = take_tensor((args.hidden_dim, args.dim))
    for i in range(args.n_layers): state_dict[f"layers.{i}.feed_forward.w2.weight"] = take_tensor((args.dim, args.hidden_dim))
    for i in range(args.n_layers): state_dict[f"layers.{i}.feed_forward.w3.weight"] = take_tensor((args.hidden_dim, args.dim))
    state_dict["norm.weight"] = take_tensor((args.dim,))
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    rhyme_map = build_rhyme_table(vocab_map)
    rhyme_group_index = build_rhyme_group_index(rhyme_map)
    return model, vocab_map, idx_to_token, args, rhyme_map, rhyme_group_index

FORM_SPECS = {
    # label -> (chars per line, total lines). Matches tools/dataset.py's
    # get_poem_meter(): odd lines end in '，', even lines end in '。', which
    # holds across the whole corpus (verified against 五律/七律 samples too,
    # not just quatrains).
    "五绝": (5, 4), "七绝": (7, 4),
    "五律": (5, 8), "七律": (7, 8),
}

def parse_form(prompt):
    """Reads a 体裁 tag out of the prompt. Returns (line_len, total_lines,
    hard) where hard=True means both are known exactly (五绝/七绝/五律/七律)
    and can be enforced with logit masking; hard=False means line_len is only
    a guess (五言/七言, or no 体裁 tag at all -- default to 5) used solely to
    position the rhyme check, with no forced line breaks."""
    for label, (line_len, total_lines) in FORM_SPECS.items():
        if label in prompt:
            return line_len, total_lines, True
    return (7 if "七" in prompt else 5), None, False

def generate_rhymed_poem(model, vocab_map, idx_to_token, args, rhyme_map, rhyme_group_index,
                          prompt="主题：明月 体裁：七绝\n", temperature=0.75, top_p=0.9, max_new_tokens=65,
                          enable_rhyme_constraint=True, target_rhyme=None, enforce_length=True):
    """Generates a poem, optionally with two independent decode-time constraints:

    1. Rhyme: forces the last character of every even line into one
       pre-selected rhyme group (classical 押韵 applies to all even lines in a
       poem, not just lines 2 and 4 -- a 五律/七律 rhymes on lines 2,4,6,8).
       The group is chosen *before* generation starts and every rhyme
       position is masked to it, rather than capturing whatever line 2
       happens to produce and only constraining later lines to match: that
       earlier approach left line 2 itself free, and did nothing at all if
       the model landed on an unclassified character there.

    2. Length (enforce_length, default on): for 五绝/七绝/五律/七律, punctuation
       is masked out until a line reaches its required character count, then
       forced at exactly that position. Measured against 80 unconstrained
       samples (10 themes x 5言/7言 x 4), only 46-60% actually matched their
       requested line count/length -- a 750K-parameter char-level model isn't
       reliable at counting on its own. This constraint makes compliance
       exact for these four forms. 五言/七言 (no fixed line count) and
       theme-only prompts fall back to natural, unconstrained line breaks.

    First-line rhyme (首句入韵, an optional classical variant) is not
    attempted: it would require deciding upfront whether this poem uses it,
    which isn't knowable from the prompt.
    """
    bos_id = vocab_map.get("<s>", 1)
    eos_id = vocab_map.get("</s>", 2)
    unk_id = vocab_map.get("<unk>", 0)
    comma_id = vocab_map.get("，", -1)
    period_id = vocab_map.get("。", -1)

    if enable_rhyme_constraint and target_rhyme is None:
        target_rhyme = random.choice(list(rhyme_group_index.keys()))
    elif not enable_rhyme_constraint:
        target_rhyme = -1

    line_len, total_lines, hard_length = parse_form(prompt)
    hard_length = hard_length and enforce_length

    # <unk>/<s>/</s>/<extra_N> are filtered out of the *displayed* text at the
    # end of this function (they're not real characters), but nothing stops
    # the model from sampling <unk> as an ordinary "content" character mid-line.
    # It would still count toward char_in_line -- so the enforced length holds
    # for tokens, but the rendered line comes out short once the <unk> is
    # stripped. Ban special tokens from content positions so token count and
    # displayed character count can't diverge.
    special_ids = [i for i, t in idx_to_token.items() if t.startswith('<')] if hard_length else []

    tokens = [bos_id] + [vocab_map.get(c, unk_id) for c in prompt]
    generated = list(tokens)

    line_idx = 1  # 1-based; odd lines end '，', even lines end '。' and rhyme
    char_in_line = 0
    first_rhyme_char = ""

    for step in range(max_new_tokens):
        context = generated[-args.seq_len:]
        x = torch.tensor([context], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)

        next_logits = logits[0, -1, :].clone()
        at_line_end = char_in_line == line_len

        # Force the last content character of every even line into the rhyme group
        if (enable_rhyme_constraint and target_rhyme > 0
                and line_idx % 2 == 0 and char_in_line == line_len - 1):
            allowed = set(rhyme_group_index[target_rhyme])
            for token_idx in rhyme_map:
                if token_idx not in allowed:
                    next_logits[token_idx] = -float('inf')

        if hard_length:
            if at_line_end:
                # Deterministically force the correct punctuation, overriding
                # whatever the model's raw logits say (0.0 for the forced
                # token beats copying its real logit, which could in
                # principle be -inf and produce a NaN softmax).
                forced_id = period_id if line_idx % 2 == 0 else comma_id
                if forced_id >= 0:
                    next_logits = torch.full_like(next_logits, -float('inf'))
                    next_logits[forced_id] = 0.0
            else:
                if comma_id >= 0:
                    next_logits[comma_id] = -float('inf')
                if period_id >= 0:
                    next_logits[period_id] = -float('inf')
                for sid in special_ids:
                    next_logits[sid] = -float('inf')

        # Apply temperature and top-p sampling
        if temperature > 0:
            next_logits = next_logits / temperature
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(next_logits, dim=-1).item()

        if next_token == eos_id or next_token == bos_id:
            break

        generated.append(next_token)
        char_in_line += 1

        if next_token in (comma_id, period_id):
            if next_token == period_id and line_idx % 2 == 0 and not first_rhyme_char:
                first_rhyme_char = idx_to_token.get(generated[-2], "")
            line_idx += 1
            char_in_line = 0
            if next_token == period_id:
                if hard_length and total_lines is not None and line_idx > total_lines:
                    break
                if not hard_length and line_idx > 4:
                    # Soft/unconstrained-form path: stop after a natural
                    # quatrain-shaped close, same default as before this
                    # rewrite. 五言/七言 samples with more lines will simply
                    # run until max_new_tokens or EOS instead.
                    break

    full_text = "".join([idx_to_token.get(idx, "") for idx in generated[1:] if not idx_to_token.get(idx, "").startswith("<")])
    return full_text, first_rhyme_char, target_rhyme

def run_pc_demonstration():
    print("=" * 68)
    print("  ESP32-S3 中文诗词大模型 - 主题+体裁 中华通韵+硬约束句长解码模拟器")
    print("=" * 68)

    model, vocab_map, idx_to_token, args, rhyme_map, rhyme_group_index = load_pc_model_and_vocab()
    print("模型权重、词表与通韵表加载完毕，开始创作测试：\n")

    test_cases = [
        ("七言绝句", "主题：明月 体裁：七绝\n"),
        ("七言绝句", "主题：边塞 体裁：七绝\n"),
        ("七言绝句", "主题：春风 体裁：七绝\n"),
        ("七言绝句", "主题：江南 体裁：七绝\n"),
        ("五言绝句", "主题：孤舟 体裁：五绝\n"),
        ("五言绝句", "主题：相思 体裁：五绝\n"),
        ("五言绝句", "主题：山水 体裁：五绝\n"),
        ("五言绝句", "主题：夕阳 体裁：五绝\n"),
        ("七言律诗", "主题：故乡 体裁：七律\n"),
        ("五言律诗", "主题：落花 体裁：五律\n"),
    ]

    for i, (genre, prompt) in enumerate(test_cases, 1):
        prompt_label = prompt.replace("\n", "")
        print(f"【测试 {i}】[{genre}] {prompt_label}")
        start_t = time.time()
        poem, first_rhyme_char, r_group = generate_rhymed_poem(
            model, vocab_map, idx_to_token, args, rhyme_map, rhyme_group_index,
            prompt=prompt, temperature=0.75, top_p=0.9,
            enable_rhyme_constraint=True, enforce_length=True)
        elapsed = (time.time() - start_t) * 1000
        print(f"创作诗词:\n{poem}")
        print(f"韵脚追踪: 每个偶数句末字均强制约束在同一通韵韵部（首个韵脚字【{first_rhyme_char}】）；句长/句数已硬约束。推理耗时: {elapsed:.2f} ms")
        print("-" * 55)

if __name__ == "__main__":
    run_pc_demonstration()
