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
# LOTTIE ARROW TRACE GENERATOR
# =========================
def make_arrow_trace(
    color="#FF6600",
    size=512,
    direction="right",  # "right", "left", "up", "down"
    stroke_width=8,  # thickness of the line
    trace_speed=1.0,  # speed multiplier (0.5 = slow, 2.0 = fast)
    arrow_size=160,  # overall size of arrow
    trace_direction="forward",  # "forward" or "reverse"
    duration=3.0,
    out_dir="arrow_traces"
):
    """
    Generate a Lottie arrow path tracing animation with configurable parameters:
    - direction: which way the arrow points (right, left, up, down)
    - stroke_width: thickness of the traced line
    - trace_speed: how fast the tracing happens
    - arrow_size: overall size of the arrow
    - trace_direction: whether to trace from start to end or end to start
    """
    
    os.makedirs(out_dir, exist_ok=True)
    fr = 30
    total_frames = int(fr * duration)
    
    color_rgba = [
        int(color[1:3], 16) / 255,
        int(color[3:5], 16) / 255,
        int(color[5:7], 16) / 255,
        1
    ]
    
    # Calculate arrow geometry based on direction
    body_length = arrow_size * 0.75
    head_size = arrow_size * 0.5
    
    # Define arrow path vertices based on direction
    if direction == "right":
        vertices = [
            [-body_length / 2, 0],
            [body_length / 4, 0],
            [body_length / 4, -head_size / 2],
            [body_length / 2, 0],
            [body_length / 4, head_size / 2],
            [body_length / 4, 0],
            [-body_length / 2, 0]
        ]
    elif direction == "left":
        vertices = [
            [body_length / 2, 0],
            [-body_length / 4, 0],
            [-body_length / 4, -head_size / 2],
            [-body_length / 2, 0],
            [-body_length / 4, head_size / 2],
            [-body_length / 4, 0],
            [body_length / 2, 0]
        ]
    elif direction == "up":
        vertices = [
            [0, body_length / 2],
            [0, -body_length / 4],
            [-head_size / 2, -body_length / 4],
            [0, -body_length / 2],
            [head_size / 2, -body_length / 4],
            [0, -body_length / 4],
            [0, body_length / 2]
        ]
    else:  # down
        vertices = [
            [0, -body_length / 2],
            [0, body_length / 4],
            [-head_size / 2, body_length / 4],
            [0, body_length / 2],
            [head_size / 2, body_length / 4],
            [0, body_length / 4],
            [0, -body_length / 2]
        ]
    
    # Trim path animation based on trace direction
    if trace_direction == "forward":
        trim_start = 0
        trim_end_start = 0
        trim_end_end = 100
    else:  # reverse
        trim_start = 0
        trim_end_start = 100
        trim_end_end = 0
    
    # Adjust duration based on speed
    actual_frames = int(total_frames / trace_speed)
    
    arrow_layer = {
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "ArrowTrace",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {"a": 0, "k": 0},
            "p": {"a": 0, "k": [size / 2, size / 2, 0]},
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
                "nm": "ArrowPath",
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
        "layers": [arrow_layer],
        "markers": []
    }
    
    fname = (
        f"arrow_trace_color-{color.strip('#')}_"
        f"dir-{direction}_stroke-{stroke_width}_"
        f"speed-{trace_speed}_size-{arrow_size}_trace-{trace_direction}.json"
    )
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lottie, f, ensure_ascii=False, separators=(",", ":"))
    
    print(f"✅ Arrow trace ({direction}, {trace_direction}, speed={trace_speed}) saved → {path}")
    return fname


# =========================
# CAPTION GENERATION
# =========================
STATIC_TEMPLATES = [ 
    "A {color_word} arrow outline pointing {direction_word}, drawn with a {stroke_word} stroke, positioned at the center of the canvas.", 
    "An outlined {color_word} arrow with a triangular head facing {direction_word}, rendered with a {stroke_word} line width, centered on a 512x512 canvas.", 
    "A {color_word} arrow icon pointing {direction_word}, consisting of an outlined body and arrowhead with {stroke_word} stroke thickness, placed in the middle of the canvas.", 
    "A simple {color_word} outlined arrow pointing {direction_word}, drawn with a {stroke_word} stroke and centered within the frame.", 
    "An arrow outline in {color_word} with its head facing {direction_word} and a {stroke_word} line, positioned in the center of a plain canvas.", 
    "A {color_word} directional arrow pointing {direction_word}, shown as an outlined shape with a uniform {stroke_word} stroke thickness at the canvas center.", 
    "A centered {color_word} arrow outline with its tip aimed {direction_word}, using a consistent {stroke_word} stroke.", 
    "A geometric {color_word} arrow pointing {direction_word}, drawn with a clean {stroke_word} outline and placed centrally.", 
    "A {color_word} arrow outline oriented {direction_word}, displayed with a smooth {stroke_word} stroke at the canvas center.", 
    "A crisp {color_word} outlined arrow facing {direction_word}, rendered using a {stroke_word} stroke and centered on a 512x512 canvas."
]

ANIMATION_TEMPLATES = [
    "A {color_word} arrow outline pointing {direction_word} is traced {trace_phrase} at a {speed_word} pace with a {stroke_word} stroke.",
    "An animated path-tracing effect draws a {color_word} arrow pointing {direction_word} {trace_phrase}, moving at a {speed_word} speed with a {stroke_word} line thickness.",
    "A {color_word} outlined arrow facing {direction_word} appears through a {trace_phrase} drawing animation at {speed_word} speed, using a {stroke_word} stroke.",
    "Path tracing animation reveals a {color_word} arrow pointing {direction_word} by drawing it {trace_phrase} at a {speed_word} pace with a {stroke_word} line.",
    "A {color_word} arrow outline pointing {direction_word} is progressively drawn {trace_phrase} in a {speed_word} tracing motion with {stroke_word} stroke width.",
    "An arrow icon in {color_word} facing {direction_word} gradually appears through a {trace_phrase} path animation at {speed_word} speed, rendered with a {stroke_word} stroke.",
    "A dynamic tracing animation draws a {color_word} arrow pointing {direction_word} {trace_phrase}, unfolding at a {speed_word} pace with a uniform {stroke_word} stroke.",
    "A {color_word} directional arrow aimed {direction_word} emerges as its outline is drawn {trace_phrase} at a {speed_word} speed using a {stroke_word} line.",
    "A {color_word} arrow oriented {direction_word} is sketched into view {trace_phrase} with a {stroke_word} stroke while animating at a {speed_word} rate.",
    "A continuous path animation forms a {color_word} arrow pointing {direction_word} by tracing it {trace_phrase} at a {speed_word} pace with a {stroke_word} outline.",
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
    elif width <= 13:
        return "medium"
    elif width <= 40:
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
    if trace_dir == "forward":
        return "from start to finish"
    else:
        return "in reverse from end to start"


def describe_size(size: int) -> str:
    """Return description of arrow size."""
    if size <= 100:
        return "small"
    elif size <= 300:
        return "medium"
    else:
        return "large"


def extract_params(base_name: str):
    """Extract parameters from filename."""
    m = re.search(
        r"color-([0-9A-Fa-f]+).*?dir-(right|left|up|down).*?stroke-(\d+).*?"
        r"speed-([-+]?[0-9]*\.?[0-9]+).*?size-(\d+).*?trace-(forward|reverse)",
        base_name
    )
    
    color_hex = "#" + m.group(1).upper() if m else "#FF6600"
    direction = m.group(2) if m else "right"
    stroke_width = int(m.group(3)) if m else 8
    trace_speed = float(m.group(4)) if m else 1.0
    arrow_size = int(m.group(5)) if m else 160
    trace_direction = m.group(6) if m else "forward"
    
    return color_hex, direction, stroke_width, trace_speed, arrow_size, trace_direction


def generate_captions(base_name: str, static_dir: str, anim_dir: str):
    """Generate static and animated captions for an arrow trace."""
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)
    
    color_hex, direction, stroke_width, trace_speed, arrow_size, trace_direction = extract_params(base_name)
    
    color_word = color_name_from_hex(color_hex)
    direction_word = direction
    stroke_word = describe_stroke(stroke_width)
    speed_word = describe_speed(trace_speed)
    trace_phrase = describe_trace_direction(trace_direction)
    size_word = describe_size(arrow_size)
    
    # Static caption (no animation)
    static_caption = random.choice(STATIC_TEMPLATES).format(
        color_word=color_word,
        direction_word=direction_word,
        stroke_word=stroke_word
    )
    
    # Animation caption
    anim_caption = random.choice(ANIMATION_TEMPLATES).format(
        color_word=color_word,
        direction_word=direction_word,
        trace_phrase=trace_phrase,
        speed_word=speed_word,
        stroke_word=stroke_word
    )
    
    with open(os.path.join(static_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(static_caption + "\n")
    
    with open(os.path.join(anim_dir, base_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(anim_caption + "\n")


# =========================
# MAIN DRIVER
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate Lottie arrow path tracing animations with captions.")
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
    directions = ["right", "left", "up", "down"]
    stroke_widths = [4, 12, 36]
    trace_speeds = [0.5, 1.0, 2.0]
    arrow_sizes = [50, 200, 400]
    trace_directions = ["forward", "reverse"]
    
    # Build all combinations
    all_combinations = list(itertools.product(
        colors, directions, stroke_widths, trace_speeds, arrow_sizes, trace_directions
    ))
    
    # Sample random unique combinations
    sampled_combos = random.sample(all_combinations, min(N_SAMPLES, len(all_combinations)))
    
    for color, direction, stroke_width, trace_speed, arrow_size, trace_dir in sampled_combos:
        duration = 3.0
        
        base_name = make_arrow_trace(
            color=color,
            direction=direction,
            stroke_width=stroke_width,
            trace_speed=trace_speed,
            arrow_size=arrow_size,
            trace_direction=trace_dir,
            duration=duration,
            out_dir=json_dir
        )
        
        generate_captions(base_name, static_dir, anim_dir)
    
    print(f"✅ Generated {len(sampled_combos)} arrow trace JSONs and captions in '{args.outdir}'")
    print(f"🎲 Random seed: {RANDOM_SEED}")


if __name__ == "__main__":
    main()