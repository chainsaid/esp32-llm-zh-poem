"""
Download and Clean Classical Tang & Song Poetry Dataset
Sources from the open-source chinese-poetry repository.
Converts Traditional Chinese to Simplified Chinese and filters clean quatrains & verses.
"""

import os
import json
import urllib.request
import re
import zhconv

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_OUTPUT = os.path.join(TOOLS_DIR, "poetry_corpus.json")

# Source-level content filter: a poem whose title/content contains one of
# these terms is dropped entirely before it ever reaches dataset.py, so the
# model has zero training signal for it (as opposed to filtering generated
# output after the fact, which only catches what a blocklist happens to
# enumerate). Deliberately narrow to whole modern-specific names/events, not
# ordinary classical imperial vocabulary (天子/皇帝/龙/朝廷/君臣...) -- those
# are normal Tang-poetry register describing ancient emperors and appear
# hundreds of times per corpus scan; blacklisting them would gut the corpus
# for a false-positive concern. This list is a manually curated starting
# point, not sourced from any official registry -- extend it as needed.
SENSITIVE_TERMS = [
    # Modern political figures (full names only, to avoid matching common
    # single surname/given-name characters that are everyday vocabulary)
    "毛泽东", "邓小平", "江泽民", "胡锦涛", "习近平",
    "李克强", "温家宝", "周恩来", "李鹏",
    # Modern political events / movements. Full names only -- short 2-char
    # forms (港独/藏独/疆独) were tried and dropped: "独" is common enough in
    # classical Chinese (独倚/独坐/独酌/独立...) that they false-positived on
    # real Tang poems, e.g. 杜甫《江上》's "行藏独倚楼" (a classical idiom
    # meaning "whether to serve office or retire" + "alone") matched "藏独".
    "六四事件", "天安门事件", "法轮功", "反送中", "颜色革命", "和平演变",
    "新疆独立", "西藏独立", "台湾独立",
]

def contains_sensitive_term(*texts):
    """Returns the first matching term found in any of the given strings, or None."""
    for text in texts:
        for term in SENSITIVE_TERMS:
            if term in text:
                return term
    return None

def filter_sensitive_poems(poems):
    """Drops any poem whose title/author/content matches SENSITIVE_TERMS.
    Returns (kept, removed) where removed is a list of (poem, matched_term)."""
    kept, removed = [], []
    for p in poems:
        hit = contains_sensitive_term(p.get("title", ""), p.get("author", ""), p.get("content", ""))
        if hit:
            removed.append((p, hit))
        else:
            kept.append(p)
    return kept, removed

def download_tang_poetry(max_volumes=15):
    """
    Downloads Tang poetry volumes from chinese-poetry repo.
    Each volume contains 1000 poems. 15 volumes = 15,000 poems.
    """
    base_url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.{}.json"
    
    all_poems = []
    print(f"Downloading Tang Poetry dataset (up to {max_volumes} volumes, ~{max_volumes * 1000} poems)...")
    
    for i in range(0, max_volumes * 1000, 1000):
        url = base_url.format(i)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                print(f"  Downloaded volume poet.tang.{i}.json ({len(data)} poems)")
                
                for item in data:
                    title = item.get("title", "")
                    author = item.get("author", "")
                    paragraphs = item.get("paragraphs", [])
                    
                    if not paragraphs or len(paragraphs) < 2:
                        continue
                        
                    content = "".join(paragraphs)
                    # Skip corrupted texts with placeholder symbols
                    if "□" in content or "{" in content or "*" in content or len(content) < 10:
                        continue
                    if len(content) > 120:  # Focus on quatrains & regulated verse
                        continue
                        
                    # Convert Traditional to Simplified Chinese
                    title_cn = zhconv.convert(title, 'zh-cn').strip()
                    author_cn = zhconv.convert(author, 'zh-cn').strip()
                    content_cn = zhconv.convert(content, 'zh-cn').strip()
                    
                    all_poems.append({
                        "title": title_cn,
                        "author": author_cn,
                        "content": content_cn
                    })
        except Exception as e:
            print(f"  Warning: Failed to download {url}: {e}")
            
    print(f"Total clean Chinese poems collected: {len(all_poems)}")

    all_poems, removed = filter_sensitive_poems(all_poems)
    if removed:
        print(f"Filtered out {len(removed)} poem(s) matching SENSITIVE_TERMS:")
        for p, term in removed:
            print(f"  [{term}] {p.get('title', '(untitled)')} -- {p.get('author', '')}")
    print(f"Total poems after sensitive-term filtering: {len(all_poems)}")

    os.makedirs(os.path.dirname(CORPUS_OUTPUT), exist_ok=True)
    with open(CORPUS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_poems, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved poetry dataset to {CORPUS_OUTPUT}")
    return all_poems

if __name__ == "__main__":
    download_tang_poetry(max_volumes=12)
