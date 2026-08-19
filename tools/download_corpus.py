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

CORPUS_OUTPUT = "tools/poetry_corpus.json"

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
    
    os.makedirs(os.path.dirname(CORPUS_OUTPUT), exist_ok=True)
    with open(CORPUS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_poems, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved poetry dataset to {CORPUS_OUTPUT}")
    return all_poems

if __name__ == "__main__":
    download_tang_poetry(max_volumes=12)
