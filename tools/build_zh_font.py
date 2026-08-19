"""
Generate the 16x16 Chinese glyph atlas consumed by main/display_driver.c.

The atlas is indexed *directly by token id*, so it is only valid for the exact
tools/vocab.json it was generated from. Any change to the corpus, to
dataset.py, or to VOCAB_SIZE reshuffles the vocabulary and invalidates it --
rerun this script every time build_zh_tokenizer.py runs, or the display will
render the wrong characters (silently, since it is just an array lookup).

Emits a .c/.h pair rather than a header-only static array: linker.lf maps all
of libmain.a to (noflash), and a 2048-entry atlas is 64KB that has no business
sitting in DRAM. The .c gets an explicit (default) mapping back to flash.

Bit layout, matching display_driver_append_token():
  32 bytes per glyph, 16 rows x 2 bytes, MSB = leftmost pixel.
"""

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont

DEFAULT_VOCAB = os.path.join(os.path.dirname(__file__), "vocab.json")
DEFAULT_OUT_C = os.path.join(os.path.dirname(__file__), "..", "main", "zh_font_16x16.c")
DEFAULT_OUT_H = os.path.join(os.path.dirname(__file__), "..", "main", "zh_font_16x16.h")

# SimHei (黑体) has uniform stroke weight and stays legible at 16px far better
# than Song/Ming faces, whose hairline horizontals disappear at this size.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

GLYPH_W = 16
GLYPH_H = 16


def pick_font_path(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Font not found: {explicit}")
        return explicit
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No CJK font found. Pass --font with a path to a .ttf/.ttc file."
    )


def is_blank_token(token):
    """Special tokens and whitespace get an empty glyph. display_driver.c
    already skips ids < 4 and handles \\n as a line break, but padding slots
    (<extra_N>) can appear anywhere in the vocab and must not render."""
    if token.startswith("<") and token.endswith(">"):
        return True
    if not token.strip():
        return True
    return False


def compute_em_origin(font):
    """Derives one shared drawing origin from a dense full-width reference
    glyph. Every CJK token then uses it, which preserves the type designer's
    intent: 月 fills the square while ， stays low-left. Centering each glyph
    on its own ink box instead would float the punctuation to mid-height."""
    img = Image.new("1", (GLYPH_W * 3, GLYPH_H * 3), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), "國", font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return (GLYPH_W - w) // 2 - bbox[0], (GLYPH_H - h) // 2 - bbox[1]


def render_glyph(token, font, origin):
    """Renders one token into a 16x16 1-bit bitmap -> 32 bytes."""
    img = Image.new("1", (GLYPH_W, GLYPH_H), 0)
    draw = ImageDraw.Draw(img)

    ox, oy = origin
    if token.isascii():
        # Half-width digits would otherwise hug the left edge; center these
        # horizontally while keeping the shared baseline.
        bbox = draw.textbbox((0, 0), token, font=font)
        ox = (GLYPH_W - (bbox[2] - bbox[0])) // 2 - bbox[0]

    draw.text((ox, oy), token, font=font, fill=1)

    data = bytearray(GLYPH_H * 2)
    px = img.load()
    for y in range(GLYPH_H):
        hi = lo = 0
        for x in range(8):
            if px[x, y]:
                hi |= 1 << (7 - x)
            if px[x + 8, y]:
                lo |= 1 << (7 - x)
        data[y * 2] = hi
        data[y * 2 + 1] = lo
    return bytes(data)


def preview(glyph, token):
    lines = [f"  '{token}'"]
    for y in range(GLYPH_H):
        hi, lo = glyph[y * 2], glyph[y * 2 + 1]
        row = "".join("#" if (hi >> (7 - x)) & 1 else "." for x in range(8))
        row += "".join("#" if (lo >> (7 - x)) & 1 else "." for x in range(8))
        lines.append("  " + row)
    return "\n".join(lines)


def build_font(vocab_path=DEFAULT_VOCAB, out_c=DEFAULT_OUT_C, out_h=DEFAULT_OUT_H,
               font_path=None, font_size=16, show_preview=False):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_map = json.load(f)

    count = len(vocab_map)
    idx_to_token = {i: t for t, i in vocab_map.items()}
    missing = [i for i in range(count) if i not in idx_to_token]
    if missing:
        raise ValueError(f"vocab.json is not a contiguous 0..N-1 id map; missing {missing[:5]}")

    font_path = pick_font_path(font_path)
    font = ImageFont.truetype(font_path, font_size)
    origin = compute_em_origin(font)
    print(f"Rendering {count} glyphs at {font_size}px from {font_path} (origin {origin})")

    glyphs = []
    blank_count = 0
    for i in range(count):
        token = idx_to_token[i]
        if is_blank_token(token):
            glyphs.append(bytes(GLYPH_H * 2))
            blank_count += 1
        else:
            glyphs.append(render_glyph(token, font, origin))

    empty_ink = sum(1 for i, g in enumerate(glyphs)
                    if not any(g) and not is_blank_token(idx_to_token[i]))
    if empty_ink:
        print(f"  WARNING: {empty_ink} non-special tokens rendered blank "
              f"(font is missing those glyphs)")
    print(f"  {blank_count} intentionally blank (special/whitespace tokens)")

    if show_preview:
        for probe in ["月", "，", "。", "国"]:
            if probe in vocab_map:
                print(preview(glyphs[vocab_map[probe]], probe))

    os.makedirs(os.path.dirname(os.path.abspath(out_h)), exist_ok=True)

    with open(out_h, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"// Auto-generated by tools/build_zh_font.py -- DO NOT EDIT.\n")
        f.write(f"// 16x16 Chinese poetry glyph atlas, indexed by token id.\n")
        f.write(f"// Regenerate whenever tools/vocab.json changes.\n")
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write(f"#define ZH_FONT_COUNT {count}\n\n")
        f.write("extern const uint8_t zh_font_16x16[ZH_FONT_COUNT][32];\n")
    print(f"Wrote {out_h}")

    with open(out_c, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"// Auto-generated by tools/build_zh_font.py -- DO NOT EDIT.\n")
        f.write(f"// Source vocabulary: {os.path.basename(vocab_path)} ({count} tokens)\n")
        f.write(f"// Font: {os.path.basename(font_path)} @ {font_size}px\n")
        f.write('#include "zh_font_16x16.h"\n\n')
        f.write("const uint8_t zh_font_16x16[ZH_FONT_COUNT][32] = {\n")
        for i, g in enumerate(glyphs):
            body = ",".join(f"0x{b:02X}" for b in g)
            token = idx_to_token[i]
            label = token if not is_blank_token(token) else repr(token)
            f.write(f"    {{{body}}}, // {i} {label}\n")
        f.write("};\n")
    size_kb = os.path.getsize(out_c) / 1024
    print(f"Wrote {out_c} ({size_kb:.0f} KB source, {count * 32 / 1024:.0f} KB in flash)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--out-c", default=DEFAULT_OUT_C)
    ap.add_argument("--out-h", default=DEFAULT_OUT_H)
    ap.add_argument("--font", default=None, help="Path to a CJK .ttf/.ttc")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--preview", action="store_true", help="Dump ASCII art for a few glyphs")
    a = ap.parse_args()
    build_font(a.vocab, a.out_c, a.out_h, a.font, a.size, a.preview)
