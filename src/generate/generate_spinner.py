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
# LOTTIE SPINNER GENERATOR
# =========================
def make_spinner(
    color="#FF6600",
    size=300,
    ring_radius=70,
    entity_radius=10,
    entity_count=8,  # between 5 and 15
    rotation_speed=1.0,
    duration=2.0,
    out_dir="spinners",
    direction="clockwise"  # <-- new parameter
):
    """
    Generate a visible, smooth Lottie spinner animation
    with a configurable number of entities (5–15) and rotation direction.
    """

    if not (5 <= entity_count <= 15):
        raise ValueError("entity_count must be between 5 and 15")

    if direction not in ("clockwise", "anticlockwise"):
        raise ValueError("direction must be 'clockwise' or 'anticlockwise'")

    os.makedirs(out_dir, exist_ok=True)
    fr = 30
    total_frames = int(fr * duration)
    angle_step = 360 / entity_count

    # Flip direction of placement
    direction_sign = 1 if direction == "clockwise" else -1

    color_rgba = [
        int(color[1:3], 16) / 255,
        int(color[3:5], 16) / 255,
        int(color[5:7], 16) / 255,
        1
    ]

    dot_groups = []
    for i in range(entity_count):
        angle_deg = direction_sign * i * angle_step  # <-- direction applied here
        rad = math.radians(angle_deg)
        x = ring_radius * math.cos(rad)
        y = ring_radius * math.sin(rad)

        offset = int((i / entity_count) * total_frames)
        k_opacity = [
            {"t": offset % total_frames, "s": [30], "i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}},
            {"t": (offset + total_frames / 4) % total_frames, "s": [100], "i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}},
            {"t": (offset + total_frames / 2) % total_frames, "s": [30], "i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}},
            {"t": total_frames, "s": [30]}
        ]

        dot_groups.append({
            "ty": "gr",
            "it": [
                {
                    "ty": "el",
                    "p": {"a": 0, "k": [x, y]},
                    "s": {"a": 0, "k": [entity_radius * 2, entity_radius * 2]},
                    "nm": f"Ellipse_{i}"
                },
                {
                    "ty": "fl",
                    "c": {"a": 0, "k": color_rgba},
                    "o": {"a": 1, "k": k_opacity},
                    "nm": "Fill 1"
                },
                {
                    "ty": "tr",
                    "p": {"a": 0, "k": [0, 0]},
                    "a": {"a": 0, "k": [0, 0]},
                    "s": {"a": 0, "k": [100, 100]},
                    "r": {"a": 0, "k": 0},
                    "o": {"a": 0, "k": 100},
                    "nm": "Transform"
                }
            ],
            "nm": f"DotGroup_{i}"
        })

    spinner_layer = {
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Spinner",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {
                "a": 1,
                "k": [
                    {
                        "t": 0,
                        "s": [0],
                        "i": {"x": [0.667], "y": [1]},
                        "o": {"x": [0.333], "y": [0]}
                    },
                    {
                        "t": total_frames,
                        "s": [360 * rotation_speed * direction_sign],  # <-- rotation matches direction
                        "i": {"x": [0.667], "y": [1]},
                        "o": {"x": [0.333], "y": [0]}
                    }
                ]
            },
            "p": {"a": 0, "k": [size / 2, size / 2, 0]},
            "a": {"a": 0, "k": [0, 0, 0]},
            "s": {"a": 0, "k": [100, 100, 100]}
        },
        "shapes": dot_groups,
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
        "layers": [spinner_layer]
    }

    fname = (
        f"spinner_color-{color.strip('#')}_"
        f"entities-{entity_count}_r-{ring_radius}_speed-{rotation_speed}_dir-{direction}.json"
    )
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lottie, f, ensure_ascii=False, separators=(",", ":"))

    print(f"✅ Spinner ({entity_count} dots, {direction}) saved → {path}")
    return fname


# =========================
# CAPTION GENERATION
# =========================
STATIC_TEMPLATES = [
    "A {color_word} spinner consisting of {entity_word} small circular dots arranged evenly in a ring of {radius_word} radius.",
    "A circular arrangement of {entity_word} {color_word} dots forming a ring with a radius of {radius_word}.",
    "An even ring of {entity_word} {color_word} filled circles positioned around the center of the {canvas_word} canvas.",
    "A centered circular structure made up of {entity_word} {color_word} dots, placed at a radius of {radius_word}.",
]

ANIMATION_TEMPLATES = [
    "A {color_word} spinner with {entity_word} dots arranged in a {radius_word}-pixel ring rotates {direction_word} at a {speed_word} speed, while each dot fades in and out sequentially.",
    "An animation showing {entity_word} {color_word} circles forming a {radius_word}-pixel ring that spins {direction_word} with a {speed_word} rotation and rhythmic opacity change.",
    "A {color_word} circular spinner of {entity_word} dots, evenly spaced in a ring of radius {radius_word}, rotates {direction_word} around the center at a {speed_word} pace while pulsing in brightness.",
    "A spinner made up of {entity_word} small {color_word} dots arranged in a {radius_word}-pixel ring, rotating {direction_word} with a {speed_word} motion, each dot alternately appearing and fading.",
    "A {direction_word}-rotating {color_word} spinner composed of {entity_word} circular dots in a ring of {radius_word} pixels, smoothly animating with a {speed_word} rhythm.",
    "An animation of {entity_word} {color_word} dots arranged in a circular ring of {radius_word} pixels that rotate {direction_word} at a {speed_word} rate while their brightness cycles in sequence."
]


def color_name_from_hex(hex_color: str) -> str:
    """Return readable color name for a given hex code."""
    hex_color = hex_color.upper()
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color
    return COLOR_NAMES.get(hex_color, "unknown color")


def describe_speed(speed: float) -> str:
    abs_speed = abs(speed)
    if abs_speed < 0.9: return "slow"
    if abs_speed < 1.1: return "steady"
    return "fast"

def describe_direction(speed: float) -> str:
    return "clockwise" if speed >= 0 else "counterclockwise"

def extract_params(base_name: str):
    m = re.search(
        r"color-([0-9A-Fa-f]+).*?entities-(\d+).*?r-(\d+).*?speed-([-+]?[0-9]*\.?[0-9]+).*?_dir-(clockwise|anticlockwise)",
        base_name
    )
    color_hex = m.group(1) if m else "000000"
    entity_count = int(m.group(2)) if m else 6
    radius = int(m.group(3)) if m else 70
    speed = float(m.group(4)) if m else 1.0
    direction = m.group(5) if m else "clockwise"
    return "#" + color_hex.upper(), entity_count, radius, speed, direction


def generate_captions(base_name: str, static_dir: str, anim_dir: str):
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    color_hex, entity_count, radius, speed, direction = extract_params(base_name)
    color_word = color_name_from_hex(color_hex)
    entity_word = str(entity_count)
    radius_word = str(radius)
    direction_word = direction
    speed_word = describe_speed(speed)
    canvas_word = "512x512"

    static_caption = random.choice(STATIC_TEMPLATES).format(
        color_word=color_word, entity_word=entity_word,
        radius_word=radius_word, canvas_word=canvas_word
    )

    anim_caption = random.choice(ANIMATION_TEMPLATES).format(
        color_word=color_word,
        entity_word=entity_word,
        radius_word=radius_word,
        direction_word=direction_word,
        speed_word=speed_word
    )

    with open(os.path.join(static_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(static_caption + "\n")

    with open(os.path.join(anim_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(anim_caption + "\n")



# =========================
# MAIN DRIVER
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate Lottie spinners with color-name captions.")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    json_dir = os.path.join(args.outdir, "json")
    static_dir = os.path.join(args.outdir, "static_caption")
    anim_dir = os.path.join(args.outdir, "animation_caption")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    # number of total samples you want
    N_SAMPLES = 500  # change as desired

    colors = VISIBLE_COLORS
    entity_counts = [8, 10, 15]
    ring_radii = [60, 80, 100]
    entity_radii = [8, 10, 12]
    rotation_speeds = [-1.2, -1.0, -0.8, 0.8, 1.0, 1.2]
    directions = ["clockwise", "anticlockwise"]

    # Build all possible combinations
    all_combinations = list(itertools.product(
        colors, entity_counts, ring_radii, entity_radii, rotation_speeds, directions
    ))

    # Sample n random unique combinations
    sampled_combos = random.sample(all_combinations, min(N_SAMPLES, len(all_combinations)))

    for color, entity_count, ring_radius, entity_radius, rotation_speed, direction in sampled_combos:
        duration = 2.0

        base_name = make_spinner(
            color=color,
            entity_count=entity_count,
            ring_radius=ring_radius,
            entity_radius=entity_radius,
            rotation_speed=rotation_speed,
            duration=duration,
            direction=direction,
            out_dir=json_dir,
        )

        generate_captions(base_name, static_dir, anim_dir)

    print(f"✅ Generated spinner JSONs and readable captions in '{args.outdir}'")
    print(f"🎲 Random seed: {RANDOM_SEED}")

if __name__ == "__main__":
    main()
