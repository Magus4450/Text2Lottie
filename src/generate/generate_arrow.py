#!/usr/bin/env python3
import os, json, math, random, argparse, re
import itertools

# =========================
# CONFIGURATION
# =========================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---- 20 visible colors on white ----
VISIBLE_COLORS = [
    "#000000", "#1E90FF", "#FF4500", "#32CD32", "#FFD700", "#8A2BE2",
    "#DC143C", "#FF1493", "#20B2AA", "#FF8C00", "#4682B4", "#B22222",
    "#FF6347", "#006400", "#00CED1", "#FF69B4", "#A52A2A", "#008B8B",
    "#C71585", "#2E8B57"
]

# ---- readable names for captions ----
COLOR_NAMES = {
    "#000000": "black",
    "#1E90FF": "blue",
    "#FF4500": "orange-red",
    "#32CD32": "green",
    "#FFD700": "gold",
    "#8A2BE2": "purple",
    "#DC143C": "crimson",
    "#FF1493": "pink",
    "#20B2AA": "teal",
    "#FF8C00": "orange",
    "#4682B4": "steel blue",
    "#B22222": "firebrick red",
    "#FF6347": "tomato red",
    "#006400": "dark green",
    "#00CED1": "turquoise",
    "#FF69B4": "hot pink",
    "#A52A2A": "brown",
    "#008B8B": "dark cyan",
    "#C71585": "magenta",
    "#2E8B57": "sea green",
}

SIZE = 512
FR = 30


# =========================
# LOTTIE ARROW GENERATOR
# =========================
def make_arrow(
    color="#FF6600",
    size=512,
    arrow_length=200,
    arrow_height=40,
    body_width=12,
    translation_dir="right",  # "right" or "left"
    translation_distance=150,  # pixels to move
    spin_rotations=0,  # number of full rotations (can be 0, 1, 2, etc.)
    scale_start=100,  # starting scale percentage
    scale_end=100,  # ending scale percentage
    duration=2.0,
    out_dir="arrows"
):
    """
    Generate a Lottie arrow animation with configurable parameters:
    - translation: moves from one side to another
    - spin: rotates the arrow
    - scale: changes size during animation
    """
    
    os.makedirs(out_dir, exist_ok=True)
    fr = 30
    total_frames = int(fr * duration)
    
    # Determine arrow direction based on translation
    pointing_left = (translation_dir == "right")
    
    color_rgba = [
        int(color[1:3], 16) / 255,
        int(color[3:5], 16) / 255,
        int(color[5:7], 16) / 255,
        1
    ]
    
    # Calculate arrow geometry
    head_width = arrow_height * 1.5
    
    if pointing_left:
        # Arrow pointing left: head on left, body on right
        body_x_start = head_width / 2
        body_x_end = body_x_start + arrow_length - head_width
        body_center_x = (body_x_start + body_x_end) / 2
        
        head_vertices = [
            [head_width / 2, -arrow_height / 2],
            [-head_width / 2, 0],
            [head_width / 2, arrow_height / 2]
        ]
        arrow_anchor_x = body_center_x
    else:
        # Arrow pointing right: body on left, head on right
        body_x_start = -arrow_length / 2 + head_width / 2
        body_x_end = arrow_length / 2 - head_width
        body_center_x = (body_x_start + body_x_end) / 2
        
        head_vertices = [
            [body_x_end - head_width / 2, -arrow_height / 2],
            [body_x_end + head_width / 2, 0],
            [body_x_end - head_width / 2, arrow_height / 2]
        ]
        arrow_anchor_x = 0
    
    # Arrow body (rectangle)
    body_shape = {
        "ty": "gr",
        "it": [
            {
                "ty": "rc",
                "d": 1,
                "s": {"a": 0, "k": [arrow_length - head_width, body_width]},
                "p": {"a": 0, "k": [body_center_x, 0]},
                "r": {"a": 0, "k": 0}
            },
            {
                "ty": "fl",
                "c": {"a": 0, "k": color_rgba},
                "o": {"a": 0, "k": 100},
                "r": 1,
                "bm": 0
            },
            {
                "ty": "tr",
                "p": {"a": 0, "k": [0, 0]},
                "a": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [100, 100]},
                "r": {"a": 0, "k": 0},
                "o": {"a": 0, "k": 100},
                "sk": {"a": 0, "k": 0},
                "sa": {"a": 0, "k": 0}
            }
        ],
        "nm": "BodyGroup",
        "bm": 0
    }
    
    # Arrow head (triangle)
    head_shape = {
        "ty": "gr",
        "it": [
            {
                "ty": "sh",
                "ks": {
                    "a": 0,
                    "k": {
                        "i": [[0, 0], [0, 0], [0, 0]],
                        "o": [[0, 0], [0, 0], [0, 0]],
                        "v": head_vertices,
                        "c": True
                    }
                }
            },
            {
                "ty": "fl",
                "c": {"a": 0, "k": color_rgba},
                "o": {"a": 0, "k": 100},
                "r": 1,
                "bm": 0
            },
            {
                "ty": "tr",
                "p": {"a": 0, "k": [body_x_end if not pointing_left else 0, 0]},
                "a": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [100, 100]},
                "r": {"a": 0, "k": 0},
                "o": {"a": 0, "k": 100},
                "sk": {"a": 0, "k": 0},
                "sa": {"a": 0, "k": 0}
            }
        ],
        "nm": "HeadGroup",
        "bm": 0
    }
    
    # Calculate position animation
    start_x = size / 2
    end_x = size / 2
    
    if translation_distance > 0:
        if translation_dir == "right":
            # Moving from left to right
            start_x = size / 2 - translation_distance / 2
            end_x = size / 2 + translation_distance / 2
        else:  # left
            # Moving from right to left
            start_x = size / 2 + translation_distance / 2
            end_x = size / 2 - translation_distance / 2
    
    # Position keyframes
    position_keyframes = {
        "a": 1 if translation_distance > 0 else 0,
        "k": [
            {
                "t": 0,
                "s": [start_x, size / 2, 0],
                "i": {"x": [0.667], "y": [1]},
                "o": {"x": [0.333], "y": [0]}
            },
            {
                "t": total_frames,
                "s": [end_x, size / 2, 0],
                "i": {"x": [0.667], "y": [1]},
                "o": {"x": [0.333], "y": [0]}
            }
        ] if translation_distance > 0 else [start_x, size / 2, 0]
    }
    
    # Rotation keyframes
    rotation_keyframes = {
        "a": 1 if spin_rotations != 0 else 0,
        "k": [
            {
                "t": 0,
                "s": [0],
                "i": {"x": [0.667], "y": [1]},
                "o": {"x": [0.333], "y": [0]}
            },
            {
                "t": total_frames,
                "s": [360 * spin_rotations],
                "i": {"x": [0.667], "y": [1]},
                "o": {"x": [0.333], "y": [0]}
            }
        ] if spin_rotations != 0 else [0]
    }
    
    # Scale keyframes
    scale_keyframes = {
        "a": 1 if scale_start != scale_end else 0,
        "k": [
            {
                "t": 0,
                "s": [scale_start, scale_start, 100],
                "i": {"x": [0.667, 0.667, 0.667], "y": [1, 1, 1]},
                "o": {"x": [0.333, 0.333, 0.333], "y": [0, 0, 0]}
            },
            {
                "t": total_frames,
                "s": [scale_end, scale_end, 100],
                "i": {"x": [0.667, 0.667, 0.667], "y": [1, 1, 1]},
                "o": {"x": [0.333, 0.333, 0.333], "y": [0, 0, 0]}
            }
        ] if scale_start != scale_end else [scale_start, scale_start, 100]
    }
    
    arrow_layer = {
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Arrow",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": rotation_keyframes,
            "p": position_keyframes,
            "a": {"a": 0, "k": [arrow_anchor_x, 0, 0]},
            "s": scale_keyframes
        },
        "ao": 0,
        "shapes": [body_shape, head_shape],
        "ip": 0,
        "op": total_frames,
        "st": 0,
        "bm": 0
    }
    
    lottie = {
        "v": "5.7.8",
        "fr": fr,
        "w": size,
        "h": size,
        "ip": 0,
        "op": total_frames,
        "layers": [arrow_layer],
        "markers": []
    }
    
    fname = (
        f"arrow_color-{color.strip('#')}_"
        f"dir-{translation_dir}_dist-{translation_distance}_"
        f"spin-{spin_rotations}_scale-{scale_start}-{scale_end}.json"
    )
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lottie, f, ensure_ascii=False, separators=(",", ":"))
    
    print(f"✅ Arrow ({translation_dir}, spin={spin_rotations}, scale={scale_start}-{scale_end}) saved → {path}")
    return fname


# =========================
# CAPTION GENERATION
# =========================
STATIC_TEMPLATES = [
    "A {color_word} arrow with a triangular head and rectangular body, pointing {pointing_word}, positioned {position_word} on the canvas.",
    "A horizontal {color_word} arrow consisting of a {body_word} body and a pointed triangular head facing {pointing_word}, located {position_word}.",
    "An arrow shape in {color_word} with its triangular arrowhead pointing {pointing_word} and a straight body extending behind it, placed {position_word}.",
    "A {color_word} directional arrow with a sharp triangular point facing {pointing_word} and a rectangular shaft, {position_word} on the 512x512 canvas.",
    "A simple {color_word} arrow icon pointing {pointing_word}, composed of a triangular head and a rectangular body, positioned {position_word}.",
    "A clean-lined {color_word} arrow with a triangular head and elongated {body_word} body pointing {pointing_word}, situated {position_word} on the canvas.",
    "A {color_word} arrow featuring a pointed triangular tip oriented {pointing_word} and a solid {body_word} shaft, placed {position_word}.",
    "A stylized {color_word} arrow with its head angled {pointing_word} and a rectangular {body_word} section, positioned precisely {position_word}.",
    "A bold {color_word} arrow composed of a triangular head pointing {pointing_word} and a uniform {body_word} body, located {position_word}.",
    "A centered {color_word} arrow with a triangular arrowhead facing {pointing_word} and a rectangular body behind it, positioned {position_word} on the canvas.",
]

ANIMATION_TEMPLATES = [
    "A {color_word} arrow pointing {pointing_word} {translation_phrase}{spin_phrase}{scale_phrase}.",
    "An animated {color_word} arrow with its head facing {pointing_word} that {translation_phrase}{spin_phrase}{scale_phrase}.",
    "A {color_word} directional arrow pointing {pointing_word} {translation_phrase}{spin_phrase}{scale_phrase} over {duration_word} seconds.",
    "An arrow in {color_word} pointing {pointing_word} undergoes animation where it {translation_phrase}{spin_phrase}{scale_phrase}.",
    "A {color_word} arrow with a triangular head facing {pointing_word} {translation_phrase}{spin_phrase}{scale_phrase} in a smooth motion.",
    "A dynamic {color_word} arrow pointing {pointing_word} that {translation_phrase}{spin_phrase}{scale_phrase} throughout the {duration_word}-second animation.",
    "A {color_word} arrow oriented {pointing_word} performs an animation in which it {translation_phrase}{spin_phrase}{scale_phrase} over {duration_word} seconds.",
    "An animated sequence features a {color_word} arrow pointing {pointing_word} as it {translation_phrase}{spin_phrase}{scale_phrase} continuously.",
    "A transforming {color_word} arrow facing {pointing_word} {translation_phrase}{spin_phrase}{scale_phrase} during a {duration_word}-second motion cycle.",
    "A {color_word} arrow with its head aimed {pointing_word} is animated so that it {translation_phrase}{spin_phrase}{scale_phrase}, completing the action in {duration_word} seconds.",
]


def color_name_from_hex(hex_color: str) -> str:
    """Return readable color name for a given hex code."""
    hex_color = hex_color.upper()
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color
    return COLOR_NAMES.get(hex_color, "unknown color")


def describe_translation(direction: str, distance: float) -> tuple:
    """Return description of translation animation."""
    if distance == 0:
        return "", "centered"
    
    if direction == "right":
        phrase = "moves from left to right"
        position = "starting from the left side"
    else:
        phrase = "moves from right to left"
        position = "starting from the right side"
    
    if distance < 100:
        phrase = phrase.replace("to", "slightly towards the")
    elif distance > 200:
        phrase = phrase.replace("to", "across to the")
    
    return phrase, position


def describe_spin(rotations: float) -> str:
    """Return description of spin animation."""
    if rotations == 0:
        return ""
    
    abs_rot = abs(rotations)
    direction = "clockwise" if rotations > 0 else "counterclockwise"
    
    if abs_rot == 1:
        return f"rotates {direction} once"
    elif abs_rot == 3:
        return f"spins {direction} thrice"
    elif abs_rot < 1:
        return f"rotates partially {direction}"
    else:
        return f"spins {direction} {abs_rot} times"


def describe_scale(start: float, end: float) -> str:
    """Return description of scale animation."""
    if start == end:
        return ""
    
    if end > start:
        ratio = end / start
        if ratio > 1.5:
            return "grows significantly in size"
        else:
            return "grows larger"
    else:
        ratio = start / end
        if ratio > 1.5:
            return "shrinks significantly"
        else:
            return "shrinks smaller"


def extract_params(base_name: str):
    """Extract parameters from filename."""
    m = re.search(
        r"color-([0-9A-Fa-f]+).*?dir-(right|left).*?dist-(\d+).*?spin-([-+]?[0-9]*\.?[0-9]+).*?scale-(\d+)-(\d+)",
        base_name
    )
    
    color_hex = "#" + m.group(1).upper() if m else "#FF6600"
    translation_dir = m.group(2) if m else "right"
    translation_distance = int(m.group(3)) if m else 0
    spin_rotations = float(m.group(4)) if m else 0
    scale_start = int(m.group(5)) if m else 100
    scale_end = int(m.group(6)) if m else 100
    
    return color_hex, translation_dir, translation_distance, spin_rotations, scale_start, scale_end


def generate_captions(base_name: str, static_dir: str, anim_dir: str, duration: float):
    """Generate static and animated captions for an arrow."""
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)
    
    color_hex, translation_dir, translation_distance, spin_rotations, scale_start, scale_end = extract_params(base_name)
    
    color_word = color_name_from_hex(color_hex)
    pointing_word = "left" if translation_dir == "right" else "right"
    body_word = "rectangular" if random.random() > 0.5 else "straight"
    
    translation_phrase, position_word = describe_translation(translation_dir, translation_distance)
    spin_phrase = describe_spin(spin_rotations)
    scale_phrase = describe_scale(scale_start, scale_end)
    duration_word = f"{duration:.1f}"
    
    # Build animation phrase with proper conjunctions
    anim_parts = []
    if translation_phrase:
        anim_parts.append(translation_phrase)
    if spin_phrase:
        anim_parts.append(spin_phrase)
    if scale_phrase:
        anim_parts.append(scale_phrase)
    
    # Join parts with "while" or "and"
    if len(anim_parts) == 0:
        combined_animation = "remains static"
    elif len(anim_parts) == 1:
        combined_animation = anim_parts[0]
    elif len(anim_parts) == 2:
        combined_animation = f"{anim_parts[0]} while it {anim_parts[1]}"
    else:
        combined_animation = f"{anim_parts[0]}, {anim_parts[1]}, and {anim_parts[2]}"
    
    # Static caption
    static_caption = random.choice(STATIC_TEMPLATES).format(
        color_word=color_word,
        pointing_word=pointing_word,
        position_word=position_word,
        body_word=body_word
    )
    
    # Animation caption
    anim_caption = random.choice(ANIMATION_TEMPLATES).format(
        color_word=color_word,
        pointing_word=pointing_word,
        translation_phrase=combined_animation,
        spin_phrase="",
        scale_phrase="",
        duration_word=duration_word
    )
    
    with open(os.path.join(static_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(static_caption + "\n")
    
    with open(os.path.join(anim_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(anim_caption + "\n")


# =========================
# MAIN DRIVER
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate Lottie arrows with captions.")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    
    json_dir = os.path.join(args.outdir, "json")
    static_dir = os.path.join(args.outdir, "static_caption")
    anim_dir = os.path.join(args.outdir, "animation_caption")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)
    
    # Number of total samples
    N_SAMPLES = 10
    
    colors = VISIBLE_COLORS
    translation_dirs = ["right", "left"]
    translation_distances = [150, 220]
    spin_rotations = [1, 3, -1, -3, 0.5]
    scale_ranges = [(100, 100), (100, 150), (150, 100), (80, 120), (120, 80)]
    
    # Build all combinations
    all_combinations = list(itertools.product(
        colors, translation_dirs, translation_distances, spin_rotations, scale_ranges
    ))
    
    # Sample random unique combinations
    sampled_combos = random.sample(all_combinations, min(N_SAMPLES, len(all_combinations)))
    
    for color, trans_dir, trans_dist, spin, (scale_start, scale_end) in sampled_combos:
        duration = 2.0
        
        base_name = make_arrow(
            color=color,
            translation_dir=trans_dir,
            translation_distance=trans_dist,
            spin_rotations=spin,
            scale_start=scale_start,
            scale_end=scale_end,
            duration=duration,
            out_dir=json_dir
        )
        
        generate_captions(base_name, static_dir, anim_dir, duration)
    
    print(f"✅ Generated {len(sampled_combos)} arrow JSONs and captions in '{args.outdir}'")
    print(f"🎲 Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    main()