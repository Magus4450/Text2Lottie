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
# LOTTIE SQUARE TRACE GENERATOR
# =========================
def make_square_trace(
    color="#FF6600",
    size=512,
    square_size=200,
    stroke_width=8,
    position=(256, 256),
    rotation=0,
    trace_speed=1.0,
    trace_direction="clockwise",
    duration=3.0,
    out_dir="square_traces"
):
    """
    Generate a Lottie square path tracing animation with configurable parameters:
    - color: hex color code
    - size: canvas size
    - square_size: size of the square
    - stroke_width: thickness of the traced line
    - position: (x, y) position of square center
    - rotation: rotation angle in degrees
    - trace_speed: how fast the tracing happens
    - trace_direction: "clockwise" or "counterclockwise"
    """
    
    os.makedirs(out_dir, exist_ok=True)
    fr = FR
    total_frames = int(fr * duration)
    
    color_rgba = [
        int(color[1:3], 16) / 255,
        int(color[3:5], 16) / 255,
        int(color[5:7], 16) / 255,
        1
    ]
    
    # Define square vertices (centered at origin)
    half_size = square_size / 2
    
    if trace_direction == "clockwise":
        vertices = [
            [-half_size, -half_size],  # top-left
            [half_size, -half_size],   # top-right
            [half_size, half_size],    # bottom-right
            [-half_size, half_size],   # bottom-left
            [-half_size, -half_size]   # back to start
        ]
    else:  # counterclockwise
        vertices = [
            [-half_size, -half_size],  # top-left
            [-half_size, half_size],   # bottom-left
            [half_size, half_size],    # bottom-right
            [half_size, -half_size],   # top-right
            [-half_size, -half_size]   # back to start
        ]
    
    # Trim path animation
    trim_start = 0
    trim_end_start = 0
    trim_end_end = 100
    
    # Adjust duration based on speed
    actual_frames = int(total_frames / trace_speed)
    
    square_layer = {
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "SquareTrace",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {"a": 0, "k": rotation},
            "p": {"a": 0, "k": [position[0], position[1], 0]},
            "a": {"a": 0, "k": [0, 0, 0]},
            "s": {"a": 0, "k": [100, 100, 100]}
        },
        "ao": 0,
        "shapes": [
            {
                "ty": "gr",
                "it": [
                    {
                        "ty": "sh",
                        "ks": {
                            "a": 0,
                            "k": {
                                "i": [[0, 0]] * len(vertices),
                                "o": [[0, 0]] * len(vertices),
                                "v": vertices,
                                "c": False
                            }
                        }
                    },
                    {
                        "ty": "st",
                        "c": {"a": 0, "k": color_rgba},
                        "o": {"a": 0, "k": 100},
                        "w": {"a": 0, "k": stroke_width},
                        "lc": 2,
                        "lj": 2,
                        "ml": 4,
                        "bm": 0
                    },
                    {
                        "ty": "tm",
                        "s": {"a": 0, "k": trim_start},
                        "e": {
                            "a": 1,
                            "k": [
                                {
                                    "t": 0,
                                    "s": [trim_end_start],
                                    "i": {"x": [0.667], "y": [1]},
                                    "o": {"x": [0.333], "y": [0]}
                                },
                                {
                                    "t": actual_frames,
                                    "s": [trim_end_end],
                                    "i": {"x": [0.667], "y": [1]},
                                    "o": {"x": [0.333], "y": [0]}
                                }
                            ]
                        },
                        "o": {"a": 0, "k": 0},
                        "m": 1
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
                "nm": "SquarePath",
                "bm": 0
            }
        ],
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
        "layers": [square_layer],
        "markers": []
    }
    
    fname = (
        f"square_trace_color-{color.strip('#')}_"
        f"size-{square_size}_stroke-{stroke_width}_"
        f"pos-{position[0]}-{position[1]}_rot-{rotation}_"
        f"speed-{trace_speed}_dir-{trace_direction}.json"
    )
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lottie, f, ensure_ascii=False, separators=(",", ":"))
    
    print(f"✅ Square trace (size={square_size}, rot={rotation}°, pos={position}) saved → {path}")
    return fname


# =========================
# CAPTION GENERATION
# =========================
STATIC_TEMPLATES = [
    "A {color_word} square outline with {size_word} dimensions, drawn with a {stroke_word} stroke{position_phrase}{rotation_phrase}.",
    "An outlined {color_word} square of {size_word} size, rendered with a {stroke_word} line width{position_phrase}{rotation_phrase}.",
    "A {color_word} square outline measuring {square_size} pixels, with {stroke_word} stroke thickness{position_phrase}{rotation_phrase}.",
    "A simple {color_word} outlined square of {size_word} size, drawn with a {stroke_word} stroke{position_phrase}{rotation_phrase}.",
    "An outlined square in {color_word} with {size_word} dimensions and a {stroke_word} line{position_phrase}{rotation_phrase}.",
]

ANIMATION_TEMPLATES = [
    "A {color_word} square outline of {size_word} size is traced {trace_phrase} at a {speed_word} pace with a {stroke_word} stroke{position_phrase}{rotation_phrase}.",
    "An animated path-tracing effect draws a {color_word} square {trace_phrase}, moving at a {speed_word} speed with {stroke_word} line thickness{position_phrase}{rotation_phrase}.",
    "A {color_word} outlined square with {size_word} dimensions appears through a {trace_phrase} drawing animation at {speed_word} speed, using a {stroke_word} stroke{position_phrase}{rotation_phrase}.",
    "Path tracing animation reveals a {color_word} square by drawing it {trace_phrase} at a {speed_word} pace with a {stroke_word} line{position_phrase}{rotation_phrase}.",
    "A {color_word} square outline measuring {square_size} pixels is progressively drawn {trace_phrase} in a {speed_word} tracing motion with {stroke_word} stroke width{position_phrase}{rotation_phrase}.",
    "An outlined square in {color_word} of {size_word} size gradually appears through a {trace_phrase} path animation at {speed_word} speed, rendered with a {stroke_word} stroke{position_phrase}{rotation_phrase}.",
]


def color_name_from_hex(hex_color: str) -> str:
    """Return readable color name for a given hex code."""
    hex_color = hex_color.upper()
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color
    return COLOR_NAMES.get(hex_color, "unknown color")


def describe_stroke(width: float) -> str:
    """Return description of stroke width."""
    if width <= 4:
        return "thin"
    elif width <= 6:
        return "medium-thin"
    elif width <= 8:
        return "medium"
    elif width <= 10:
        return "medium-thick"
    else:
        return "thick"


def describe_speed(speed: float) -> str:
    """Return description of tracing speed."""
    if speed <= 0.7:
        return "slow"
    elif speed <= 1.3:
        return "steady"
    else:
        return "fast"


def describe_trace_direction(trace_dir: str) -> str:
    """Return description of trace direction."""
    if trace_dir == "clockwise":
        return "clockwise"
    else:
        return "counterclockwise"


def describe_size(size: int) -> str:
    """Return description of square size."""
    if size <= 120:
        return "small"
    elif size <= 180:
        return "medium"
    else:
        return "large"


def describe_position(pos: tuple, canvas_size: int) -> str:
    """Return description of position."""
    x, y = pos
    center = canvas_size / 2
    
    if abs(x - center) < 10 and abs(y - center) < 10:
        return ", centered on the canvas"
    
    h_pos = "left" if x < center - 50 else "right" if x > center + 50 else "horizontally centered"
    v_pos = "top" if y < center - 50 else "bottom" if y > center + 50 else "vertically centered"
    
    if "centered" in h_pos and "centered" in v_pos:
        return ", centered on the canvas"
    elif "centered" in h_pos:
        return f", positioned at the {v_pos}"
    elif "centered" in v_pos:
        return f", positioned at the {h_pos}"
    else:
        return f", positioned at the {v_pos}-{h_pos}"


def describe_rotation(rot: float) -> str:
    """Return description of rotation."""
    if abs(rot) < 5:
        return ""
    return f", rotated {int(rot)} degrees"


def extract_params(base_name: str):
    """Extract parameters from filename."""
    m = re.search(
        r"color-([0-9A-Fa-f]+).*?size-(\d+).*?stroke-(\d+).*?"
        r"pos-(\d+)-(\d+).*?rot-([-+]?\d+).*?"
        r"speed-([-+]?[0-9]*\.?[0-9]+).*?dir-(clockwise|counterclockwise)",
        base_name
    )
    
    color_hex = "#" + m.group(1).upper() if m else "#FF6600"
    square_size = int(m.group(2)) if m else 200
    stroke_width = int(m.group(3)) if m else 8
    pos_x = int(m.group(4)) if m else 256
    pos_y = int(m.group(5)) if m else 256
    rotation = int(m.group(6)) if m else 0
    trace_speed = float(m.group(7)) if m else 1.0
    trace_direction = m.group(8) if m else "clockwise"
    
    return color_hex, square_size, stroke_width, (pos_x, pos_y), rotation, trace_speed, trace_direction


def generate_captions(base_name: str, static_dir: str, anim_dir: str):
    """Generate static and animated captions for a square trace."""
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)
    
    color_hex, square_size, stroke_width, position, rotation, trace_speed, trace_direction = extract_params(base_name)
    
    color_word = color_name_from_hex(color_hex)
    stroke_word = describe_stroke(stroke_width)
    speed_word = describe_speed(trace_speed)
    trace_phrase = describe_trace_direction(trace_direction)
    size_word = describe_size(square_size)
    position_phrase = describe_position(position, SIZE)
    rotation_phrase = describe_rotation(rotation)
    
    # Static caption (no animation)
    static_caption = random.choice(STATIC_TEMPLATES).format(
        color_word=color_word,
        size_word=size_word,
        square_size=square_size,
        stroke_word=stroke_word,
        position_phrase=position_phrase,
        rotation_phrase=rotation_phrase
    )
    
    # Animation caption
    anim_caption = random.choice(ANIMATION_TEMPLATES).format(
        color_word=color_word,
        size_word=size_word,
        square_size=square_size,
        trace_phrase=trace_phrase,
        speed_word=speed_word,
        stroke_word=stroke_word,
        position_phrase=position_phrase,
        rotation_phrase=rotation_phrase
    )
    
    with open(os.path.join(static_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(static_caption + "\n")
    
    with open(os.path.join(anim_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(anim_caption + "\n")


# =========================
# MAIN DRIVER
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate Lottie square path tracing animations with captions.")
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    
    json_dir = os.path.join(args.outdir, "json")
    static_dir = os.path.join(args.outdir, "static_caption")
    anim_dir = os.path.join(args.outdir, "animation_caption")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)
    
    # Number of total samples
    N_SAMPLES = 100
    
    colors = VISIBLE_COLORS
    square_sizes = [100, 120, 150, 180, 200, 220, 250]
    stroke_widths = [4, 6, 8, 10, 12]
    trace_speeds = [0.5, 0.8, 1.0, 1.5, 2.0]
    trace_directions = ["clockwise", "counterclockwise"]
    
    # Position variations (avoiding edges)
    positions = [
        (256, 256),  # center
        (180, 180),  # top-left quadrant
        (332, 180),  # top-right quadrant
        (180, 332),  # bottom-left quadrant
        (332, 332),  # bottom-right quadrant
        (256, 180),  # top center
        (256, 332),  # bottom center
        (180, 256),  # left center
        (332, 256),  # right center
    ]
    
    # Rotation variations (in degrees)
    rotations = [0, 15, 30, 45, 60, 90, 120, 135]
    
    # Build all combinations
    all_combinations = list(itertools.product(
        colors, square_sizes, stroke_widths, trace_speeds, 
        trace_directions, positions, rotations
    ))
    
    # Sample random unique combinations
    sampled_combos = random.sample(all_combinations, min(N_SAMPLES, len(all_combinations)))
    
    for color, square_size, stroke_width, trace_speed, trace_dir, position, rotation in sampled_combos:
        duration = 3.0
        
        base_name = make_square_trace(
            color=color,
            size=SIZE,
            square_size=square_size,
            stroke_width=stroke_width,
            position=position,
            rotation=rotation,
            trace_speed=trace_speed,
            trace_direction=trace_dir,
            duration=duration,
            out_dir=json_dir
        )
        
        generate_captions(base_name, static_dir, anim_dir)
    
    print(f"\n✅ Generated {len(sampled_combos)} square trace JSONs and captions in '{args.outdir}'")
    print(f"🎲 Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    main()