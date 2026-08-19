"""
Classical Chinese Poetry Dataset Collection & Preprocessing
Loads clean pure poetry verse corpus from tools/poetry_corpus.json.
Focuses 100% on poetic content, rhyming, and structure without author/title metadata.
"""

import os
import json

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "poetry_corpus.json")

def get_training_text_samples():
    """Generates pure poetry verse samples directly from tools/poetry_corpus.json."""
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_FILE}. "
            "Please run 'python tools/download_corpus.py' first."
        )

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        poems = json.load(f)

    samples = []
    for p in poems:
        content = p.get("content", "").strip()
        # Keep clean classical poems with valid punctuation
        if content and ("，" in content or "。" in content) and len(content) >= 10:
            samples.append(content)

    print(f"Loaded {len(samples)} pure classical poetry samples from {CORPUS_FILE}.")
    return samples

if __name__ == "__main__":
    samples = get_training_text_samples()
    print(f"Sample 0: {samples[0]}")
    print(f"Sample 1: {samples[1]}")
