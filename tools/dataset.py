"""
Classical Chinese Poetry Dataset with Theme and Meter/Form Conditioning
Supports:
  主题：{关键词} 体裁：{五绝/七绝/五律/七律/五言/七言}
  {诗句正文}
"""

import os
import json
import random

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "poetry_corpus.json")

COMMON_THEMES = [
    "明月", "春风", "秋风", "冬雪", "山水", "江河", "孤舟", "白云",
    "落花", "离别", "相思", "边塞", "关山", "夕阳", "故乡", "松竹",
    "黄昏", "青云", "高楼", "美酒", "夜雨", "江南", "清风", "寒江",
    "春晓", "白发", "归人", "青山", "红叶", "幽居", "行路", "送别"
]

SINGLE_CHAR_THEMES = ["月", "风", "花", "雪", "山", "水", "云", "雨", "酒", "春", "秋", "夜", "舟", "归", "梦"]

def get_poem_meter(content):
    """Identifies the verse form: quatrain (4 lines), regulated verse (8 lines), or generic."""
    sentences = [s.strip() for s in content.replace('。', '，').split('，') if s.strip()]
    if not sentences:
        return ""
    first_len = len(sentences[0])
    num_sentences = len(sentences)
    if not all(len(s) == first_len for s in sentences):
        # Irregular line lengths: not a clean 5/7-char form, skip the meter tag entirely
        # rather than mislabeling it, since the tag would give the model no real signal.
        return ""

    if first_len == 5:
        if num_sentences == 4:
            return "五绝"
        if num_sentences == 8:
            return "五律"
        return "五言"
    elif first_len == 7:
        if num_sentences == 4:
            return "七绝"
        if num_sentences == 8:
            return "七律"
        return "七言"
    return ""

def extract_themes_for_poem(title, content):
    """Returns a list of theme tags. Common multi-char themes and a single-char
    fallback are both collected independently (no title-slice fallback: 57% of
    those turned out to contain out-of-vocab characters and taught the model
    that the theme slot is unrelated to the content)."""
    matched = []
    text = title + content
    for theme in COMMON_THEMES:
        if theme in text:
            matched.append(theme)
    for single in SINGLE_CHAR_THEMES:
        if single in text:
            matched.append(single)
            break
    return matched

def _load_poems():
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_FILE}. "
            "Please run 'python tools/download_corpus.py' first."
        )
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _valid_poems(poems):
    for p in poems:
        title = p.get("title", "").strip()
        content = p.get("content", "").strip()
        if not content or ("，" not in content and "。" not in content) or len(content) < 10:
            continue
        yield title, content

def _samples_for_poem(title, content):
    samples = []
    meter = get_poem_meter(content)
    themes = extract_themes_for_poem(title, content)

    if meter:
        samples.append(f"体裁：{meter}\n{content}")
    for theme in themes:
        samples.append(f"主题：{theme}\n{content}")
        if meter:
            samples.append(f"主题：{theme} 体裁：{meter}\n{content}")

    # Unconditional sample
    samples.append(content)
    return samples

def get_poem_split_ids(val_fraction=0.05, seed=1337):
    """Deterministically assigns each poem (by index in the corpus) to train/val,
    so samples derived from the same poem never leak across the split."""
    poems = _load_poems()
    n = len(poems)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_ids = set(indices[:n_val])
    return n, val_ids

def get_training_text_samples(split=None, val_fraction=0.05, seed=1337):
    """Generates thematic and meter-conditioned training samples.

    split: None (all poems), "train", or "val". Splitting is done per-poem
    (not per-sample) since a single poem expands into several samples.
    """
    poems = _load_poems()
    val_ids = None
    if split is not None:
        _, val_ids = get_poem_split_ids(val_fraction=val_fraction, seed=seed)

    samples = []
    for idx, p in enumerate(poems):
        title = p.get("title", "").strip()
        content = p.get("content", "").strip()
        if not content or ("，" not in content and "。" not in content) or len(content) < 10:
            continue

        if split == "train" and idx in val_ids:
            continue
        if split == "val" and idx not in val_ids:
            continue

        samples.extend(_samples_for_poem(title, content))

    label = split or "all"
    print(f"Loaded {len(samples)} thematic/meter-conditioned poetry samples ({label} split) from {CORPUS_FILE}.")
    return samples

if __name__ == "__main__":
    samples = get_training_text_samples()
    print("Example 1:", samples[0])
    print("Example 2:", samples[1])
    print("Example 3:", samples[2])
