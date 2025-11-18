#!/usr/bin/env python3
import os
import json
import random

JSON_DIR = "dataset_for_masked/generated_data/json"
STATIC_CAPTION_DIR = "dataset_for_masked/generated_data/static_caption"
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
    "A {size_word} {color_word} {shape_word}, styled in a {style_word} manner, positioned {position_word}.",
    "A {shape_word} featuring a {style_word} appearance, {size_word} in scale and located {position_word}.",
    "Placed {position_word}, a {size_word} {color_word} {shape_word} appears with a {style_word} look.",
    "A {style_word} {shape_word} of {size_word} dimensions, shown {position_word} and colored in {color_word}.",
    "A {color_word} {shape_word}, designed in a {style_word} style and sized {size_word}, resting {position_word}.",
    "A {shape_word} with {style_word} styling and {size_word} proportions positioned clearly {position_word}.",
    "Situated {position_word}, a {size_word} {shape_word} rendered in {color_word} displays a {style_word} finish.",
    "A visually simple {color_word} {shape_word}, {size_word} in size and {style_word} in design, located {position_word}.",
]


# ------------------------------------------------------------
#_POSITION INFERENCE
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

    # Rotation
    if m.startswith("clockwise") or m.startswith("anticlockwise"):
        return "at the center"

    return "near the center"


# ------------------------------------------------------------
#  FILENAME PARSER  (no regex, robust)
# ------------------------------------------------------------
# Expected structure:
# shape__motion__easing__size__color__style__scale-X__sample-Y.json
def parse_filename(stem: str):
    # Collapse triple underscores into double
    while "___" in stem:
        stem = stem.replace("___", "__")

    parts = stem.split("__")
    if len(parts) != 8:
        return None

    shape      = parts[0]
    motion     = parts[1].rstrip("_")  # drop trailing artifacts
    easing     = parts[2]
    size       = parts[3]
    color      = parts[4]
    style      = parts[5]
    scale_part = parts[6]
    sample_part= parts[7]

    if not scale_part.startswith("scale-"):
        return None
    if not sample_part.startswith("sample-"):
        return None

    # Normalize for caption
    shape_word = shape.replace("-", " ").replace("_", " ")
    size_word  = size.replace("_", " ").replace("-", " ")
    color_word = color.replace("_", " ").replace("-", " ")
    motion_word= motion.replace("_", " ").replace("-", " ")

    if style.startswith("dotted-gap"):
        style_word = "dotted"
    elif style == "outline":
        style_word = "outlined"
    else:
        style_word = "filled"

    return {
        "shape_word": shape_word,
        "size_word": size_word,
        "color_word": color_word,
        "style_word": style_word,
        "motion": motion_word,
    }


# ------------------------------------------------------------
# MAIN
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
