#!/usr/bin/env python3
import os, json, math, random, argparse, re
import itertools
import numpy as np
import copy

# =========================
# CONFIGURATION
# =========================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    direction="clockwise",
    add_noise=True  # <-- new parameter
):
    """
    Generate a visible, smooth Lottie spinner animation
    with a configurable number of entities and rotation direction.
    Now includes variability/noise across multiple parameters.
    """
    # Store original values for filename
    ring_radius_orig = ring_radius
    entity_radius_orig = entity_radius
    entity_count_orig = entity_count
    size_orig = size
    
    if add_noise:
        # Canvas size variability (±5%)
        size = int(size * random.uniform(0.95, 1.05))
        
        # Ring radius variability (±8%)
        ring_radius = ring_radius * random.uniform(0.92, 1.08)
        
        # Entity radius variability (±10%)
        entity_radius = entity_radius * random.uniform(0.90, 1.10)
        
        # Entity count stays the same (already an integer)
        # but we can add slight positioning noise later
        
        # Rotation speed variability (±5%)
        rotation_speed = rotation_speed * random.uniform(0.95, 1.05)
        
        # Frame rate variability
        fr = random.randint(25, 35)
    else:
        fr = 30

    if not (5 <= entity_count <= 15):
        raise ValueError("entity_count must be between 5 and 15")

    if direction not in ("clockwise", "anticlockwise"):
        raise ValueError("direction must be 'clockwise' or 'anticlockwise'")

    os.makedirs(out_dir, exist_ok=True)
    total_frames = int(fr * duration)
    angle_step = 360 / entity_count

    # Flip direction of placement
    direction_sign = 1 if direction == "clockwise" else -1

    # Color with variability
    def hex_to_rgba01(hexstr):
        s = hexstr.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return [r, g, b, 1]
    
    color_rgba = hex_to_rgba01(color)
    
    if add_noise:
        # Add slight color jitter (±0.08 in RGB space, clamped)
        color_rgba = [
            max(0, min(1, color_rgba[0] + random.uniform(-0.08, 0.08))),
            max(0, min(1, color_rgba[1] + random.uniform(-0.08, 0.08))),
            max(0, min(1, color_rgba[2] + random.uniform(-0.08, 0.08))),
            1
        ]

    dot_groups = []
    for i in range(entity_count):
        angle_deg = direction_sign * i * angle_step
        
        # Add angular noise (±2 degrees)
        if add_noise:
            angle_deg += random.uniform(-2, 2)
        
        rad = math.radians(angle_deg)
        
        # Calculate position with optional radial noise
        current_ring_radius = ring_radius
        if add_noise:
            # Add radial position noise (±3%)
            current_ring_radius *= random.uniform(0.97, 1.03)
        
        x = current_ring_radius * math.cos(rad)
        y = current_ring_radius * math.sin(rad)

        # Calculate opacity animation offset
        offset = int((i / entity_count) * total_frames)
        
        # Opacity range with variability
        min_opacity = 30
        max_opacity = 100
        if add_noise:
            min_opacity = max(10, min_opacity + random.uniform(-10, 10))
            max_opacity = min(100, max_opacity + random.uniform(-5, 5))
        
        # Opacity timing with slight variability
        quarter_frame = total_frames / 4
        half_frame = total_frames / 2
        if add_noise:
            quarter_frame *= random.uniform(0.95, 1.05)
            half_frame *= random.uniform(0.95, 1.05)
        
        k_opacity = [
            {"t": offset % total_frames, "s": [min_opacity], "i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}},
            {"t": int((offset + quarter_frame) % total_frames), "s": [max_opacity], "i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}},
            {"t": int((offset + half_frame) % total_frames), "s": [min_opacity], "i": {"x": [0.667], "y": [1]}, "o": {"x": [0.333], "y": [0]}},
            {"t": total_frames, "s": [min_opacity]}
        ]

        # Entity size with individual variability
        current_entity_radius = entity_radius
        if add_noise:
            current_entity_radius *= random.uniform(0.90, 1.10)

        dot_groups.append({
            "ty": "gr",
            "it": [
                {
                    "ty": "el",
                    "p": {"a": 0, "k": [x, y]},
                    "s": {"a": 0, "k": [current_entity_radius * 2, current_entity_radius * 2]},
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

    # Final rotation angle with variability
    final_rotation = 360 * rotation_speed * direction_sign
    if add_noise:
        final_rotation += random.uniform(-5, 5)

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
                        "s": [final_rotation],
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

    # Use original values for filename for consistency
    fname = (
        f"spinner_color-{color.strip('#')}_"
        f"entities-{entity_count_orig}_r-{ring_radius_orig}_speed-{rotation_speed}_dir-{direction}.json"
    )
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lottie, f, ensure_ascii=False, separators=(",", ":"))

    print(f"✅ Spinner ({entity_count_orig} dots, {direction}) saved → {path}")
    return fname


# =========================
# CAPTION GENERATION
# =========================
STATIC_TEMPLATES = [
    "A {color_word} spinner consisting of {entity_word} small circular dots arranged evenly in a ring of {radius_word} radius.",
    "A circular arrangement of {entity_word} {color_word} dots forming a ring with a radius of {radius_word}.",
    "An even ring of {entity_word} {color_word} filled circles positioned around the center of the canvas.",
    "A centered circular structure made up of {entity_word} {color_word} dots, placed at a radius of {radius_word}.",
    "A symmetrical ring of {entity_word} {color_word} circles arranged evenly at a distance of {radius_word} from the center.",
    "A radial formation of {entity_word} {color_word} dots placed in a perfect circle of radius {radius_word}.",
    "A ring-shaped pattern composed of {entity_word} evenly spaced {color_word} circular dots centered on the canvas.",
    "A {color_word} circular dot pattern consisting of {entity_word} elements placed uniformly along a ring of {radius_word}.",
    "A ring of {entity_word} {color_word} small filled circles, each positioned at the same {radius_word} distance from the center.",
    "A structured circular grid of {entity_word} {color_word} dots forming a ring with a radius of {radius_word}, centered on the canvas.",
]

ANIMATION_TEMPLATES = [
    "A {color_word} spinner with {entity_word} dots arranged in a {radius_word} ring rotates {direction_word} at a {speed_word} speed, while each dot fades in and out sequentially.",
    "An animation showing {entity_word} {color_word} circles forming a {radius_word} ring that spins {direction_word} with a {speed_word} rotation and rhythmic opacity change.",
    "A {color_word} circular spinner of {entity_word} dots, evenly spaced in a ring of radius {radius_word}, rotates {direction_word} around the center at a {speed_word} pace while pulsing in brightness.",
    "A spinner made up of {entity_word} small {color_word} dots arranged in a {radius_word} ring, rotating {direction_word} with a {speed_word} motion, each dot alternately appearing and fading.",
    "A {direction_word}-rotating {color_word} spinner composed of {entity_word} circular dots in a ring of {radius_word}, smoothly animating with a {speed_word} rhythm.",
    "An animation of {entity_word} {color_word} dots arranged in a circular ring of {radius_word} that rotate {direction_word} at a {speed_word} rate while their brightness cycles in sequence.",
    "A {color_word} dot-based spinner of {entity_word} elements placed along a ring of radius {radius_word}, rotating {direction_word} with a {speed_word} speed and sequential opacity pulses.",
    "A circular spinner formed by {entity_word} {color_word} dots at {radius_word} radius rotates {direction_word} at a {speed_word} pace, each dot brightening and dimming in turn.",
    "A ring of {entity_word} small {color_word} circles at {radius_word} distance rotates {direction_word} at a {speed_word} rate while the dots fade in a looping sequence.",
    "A {color_word} spinner of {entity_word} evenly spaced dots arranged in a {radius_word} ring, rotating {direction_word}, with dot brightness oscillating at a {speed_word} tempo.",
    "A dynamic animation of {entity_word} {color_word} dots positioned around a {radius_word} ring that spin {direction_word} at a {speed_word} speed while cycling through opacity levels.",
    "A smoothly rotating {color_word} ring of {entity_word} dots spaced at {radius_word} radius, spinning {direction_word} with a {speed_word} rotation and synchronized fade effects.",
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

def describe_direction(direction: str) -> str:
    return direction

def describe_radii(radii: int) -> str:
    if radii > 100:
        return "big"
    elif radii > 50:
        return "medium"
    return "small"

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
    radius_word = describe_radii(radius)
    direction_word = describe_direction(direction)
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
    parser = argparse.ArgumentParser(description="Generate Lottie spinners with color-name captions and variability.")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    json_dir = os.path.join(args.outdir, "json")
    static_dir = os.path.join(args.outdir, "static_caption")
    anim_dir = os.path.join(args.outdir, "animation_caption")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    # number of total samples you want
    N_SAMPLES = 30  # change as desired

    colors = VISIBLE_COLORS
    entity_counts = [5, 10, 15]
    ring_radii = [30, 70, 120]
    entity_radii = [5, 15, 30]
    rotation_speeds = [-2, -1.0, -0.8, 0.8, 1.0, 2]
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
            add_noise=True  # <-- enable variability
        )

        generate_captions(base_name, static_dir, anim_dir)

    print(f"✅ Generated {len(sampled_combos)} spinner JSONs with variability and readable captions in '{args.outdir}'")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    print(f"🎨 Variability added to: canvas size, ring radius, entity radius, positions, colors, rotation speed, opacity, and timing")

if __name__ == "__main__":
    main()