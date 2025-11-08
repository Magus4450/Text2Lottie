import os, json, re, random

# ===== CONFIG =====
JSON_DIR = "generated_data/json"
CAPTION_DIR = "generated_data/static_caption"
os.makedirs(CAPTION_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

base_color = "#000000"
size_labels = {100: "small", 160: "medium", 220: "large"}

# ===== HELPERS =====
def restore_shape_from_slug(shape_slug: str) -> str:
    """
    Reverse filename-safe encoding of shape parameter.
    Supports both underscore and dash variants, e.g.:
      polygon_5  -> polygon:5
      polygon-5  -> polygon:5
      star_8     -> star:8
      rounded-square_16 -> rounded-square:16
    """
    m = re.match(r"^(polygon|star|rounded-square)[_-](\d+)$", shape_slug)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return shape_slug

def slug_to_shape(filename: str) -> str:
    """Extract shape slug from filename (first segment before '__')."""
    m = re.search(r"^(.*?)__", filename)
    if not m:
        return "unknown"
    raw = m.group(1)
    # restore potential numeric suffix (e.g. polygon_5 → polygon:5)
    return restore_shape_from_slug(raw)

def shape_noun(shape: str) -> str:
    s = shape.lower()
    if ":" in s:
        kind, arg = s.split(":", 1)
    else:
        kind, arg = s, None
    if kind == "circle": return "circle"
    if kind == "square": return "square"
    if kind == "rounded-square": return "rounded square"
    if kind == "triangle": return "triangle"
    if kind == "polygon":
        try:
            n = max(3, int(arg)) if arg else 5
        except ValueError:
            n = 5
        return f"{n}-sided polygon"
    if kind == "star":
        try:
            p = max(3, int(arg)) if arg else 5
        except ValueError:
            p = 5
        return f"{p}-point star"
    return s

def parse_size_from_name(fname: str):
    m = re.search(r"size-(\d+)px", fname)
    return int(m.group(1)) if m else None

def infer_initial_position(fname: str) -> str:
    """Infer initial position from animation type."""
    if "left-to-right" in fname:
        return "on the left side"
    if "right-to-left" in fname:
        return "on the right side"
    if "up-to-down" in fname:
        return "near the top"
    if "down-to-up" in fname:
        return "near the bottom"
    if "clockwise" in fname or "anticlockwise" in fname:
        return "at the center"
    return "at the center"

# ===== CAPTION TEMPLATES =====
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

def choose_template():
    return random.choice(CAPTION_TEMPLATES)

# ===== MAIN LOOP =====
count = 0
for file in os.listdir(JSON_DIR):
    if not file.endswith(".json"):
        continue

    fname = file[:-5]  # remove .json

    # Extract and restore shape
    shape_slug = slug_to_shape(fname)
    shape_word = shape_noun(shape_slug)

    size_px = parse_size_from_name(fname)
    size_word = size_labels.get(size_px, f"{size_px}px")

    style_word = "filled"
    if "dotted" in fname:
        style_word = "dotted"
    elif "outline" in fname:
        style_word = "outlined"

    color_word = "black"
    position_word = infer_initial_position(fname)

    template = choose_template()
    caption = template.format(
        size_word=size_word,
        shape_word=shape_word,
        style_word=style_word,
        color_word=color_word,
        position_word=position_word,
    )

    caption_path = os.path.join(CAPTION_DIR, fname + ".txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

    count += 1

print(f"✅ Generated {count} static captions with proper shape restoration in {CAPTION_DIR}")
print(f"🎲 Random seed: {RANDOM_SEED} ensures reproducibility.")
