#!/usr/bin/env python3
import os
import json
import random
import re

JSON_DIR = "generated_data/json"
STATIC_CAPTION_DIR = "generated_data/static_caption"
os.makedirs(STATIC_CAPTION_DIR, exist_ok=True)

CAPTION_TEMPLATES = [
    "A {color_word} {shape_word} of {size_word} size placed {position_word} of the canvas.",
    "A {size_word} {style_word} {shape_word} positioned {position_word}, rendered in {color_word}.",
    "An object shaped like a {shape_word}, {size_word} in size, {style_word} in style, located {position_word}.",
    "A {size_word} {color_word} {shape_word} with {style_word} styling, initially appearing {position_word}.",
    "At {position_word} lies a {size_word} {style_word} {shape_word} in {color_word}.",
    "A {shape_word}, {size_word} and {style_word}, placed {position_word} against a neutral background.",
    "A {size_word} {shape_word} with a {style_word} finish, sitting {position_word} of the frame.",
    "A centrally aligned {size_word} {shape_word} rendered in {color_word}, {style_word} in style, found {position_word}.",
]

# ------------------------------------------------------------
# Helper: extract position from motion
# ------------------------------------------------------------
def infer_position_from_motion(motion: str) -> str:
    m = motion.lower()

    if m == "left-to-right":
        return "at the left"
    if m == "right-to-left":
        return "at the right"
    if m == "up-to-down":
        return "at the top"
    if m == "down-to-up":
        return "at the bottom"

    # Rotation → starts at dead center
    if m.startswith("clockwise") or m.startswith("anticlockwise"):
        return "at the center"

    return "near the center"

# ------------------------------------------------------------
# Helper: extract words from a filename
# ------------------------------------------------------------
# filename format:
#   shape__motion__easing__size__color__style__scale-X__sample-Y.json
FNAME_RE = re.compile(
    r"^(?P<shape>[^_]+)__"
    r"(?P<motion>[^_]+)__"
    r"(?P<easing>[^_]+)__"
    r"(?P<size>[^_]+)__"
    r"(?P<color>[^_]+)__"
    r"(?P<style>(?:dotted-gap-\d+|outline|fill))__"
    r"scale-(?P<scale>[\d\.]+)__sample-(?P<sample>\d+)$"
)

def parse_filename(stem: str):
    m = FNAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()

    # Normalize words for caption
    shape_word = d["shape"].replace("-", " ")
    size_word = d["size"].replace("-", " ")
    color_word = d["color"].replace("-", " ")

    style = d["style"]
    if style.startswith("dotted"):
        style_word = "dotted"
    elif style == "outline":
        style_word = "outlined"
    else:
        style_word = "filled"

    motion = d["motion"].replace("-", " ")

    return {
        "shape_word": shape_word,
        "size_word": size_word,
        "color_word": color_word,
        "style_word": style_word,
        "motion": motion,
    }

# ------------------------------------------------------------
# Main generation
# ------------------------------------------------------------
files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
print(f"Found {len(files)} JSON files")

for fname in files:
    stem = fname[:-5]

    parsed = parse_filename(stem)
    if not parsed:
        print(f"[WARN] Could not parse filename: {fname}")
        continue

    position_word = infer_position_from_motion(parsed["motion"])
    template = random.choice(CAPTION_TEMPLATES)

    caption = template.format(
        color_word=parsed["color_word"],
        shape_word=parsed["shape_word"],
        size_word=parsed["size_word"],
        style_word=parsed["style_word"],
        position_word=position_word,
    )

    out_path = os.path.join(STATIC_CAPTION_DIR, stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

print(f"✅ Generated static captions for {len(files)} animations")
print(f"📁 Output: {STATIC_CAPTION_DIR}")
