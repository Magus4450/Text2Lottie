import json
import random
import os
import itertools
import re

# Color palette with semantic names
COLOR_PALETTE = {
    "red": "#E53E3E",
    "blue": "#3B82F6",
    "green": "#10B981",
    "yellow": "#F59E0B",
    "purple": "#8B5CF6",
    "pink": "#EC4899",
    "orange": "#F97316",
    "teal": "#14B8A6",
    "indigo": "#6366F1",
    "cyan": "#06B6D4",
    "lime": "#84CC16",
    "emerald": "#059669",
    "violet": "#7C3AED",
    "fuchsia": "#D946EF",
    "rose": "#F43F5E",
    "amber": "#F59E0B",
    "sky": "#0EA5E9",
    "mint": "#6EE7B7",
    "coral": "#FF6B6B",
    "lavender": "#A78BFA",
}

# Size definitions (as percentage of min canvas dimension)
SIZE_DEFINITIONS = {
    "very small": 0.10,    # 10% of canvas
    "small": 0.16,         # 16% of canvas (original default)
    "medium": 0.25,        # 25% of canvas
    "large": 0.35,         # 35% of canvas
    "very large": 0.45,    # 45% of canvas
}

def generate_lottie_shape(
    outline: bool,
    motion: str,
    time: float,
    color: str,
    color_name: str,  # NEW: semantic color name
    easing_type: str,
    dotted: bool = False,
    dot_spacing: float = 4.0,
    shape: str = "circle",
    size: float = None,
    size_name: str = "small",  # NEW: semantic size name
    scaling: float = 1.0,
    add_noise: bool = True,
    seed: int = None
) -> dict:
    if seed is not None:
        random.seed(seed)
    
    # --- canvas & timing with variability ---
    w, h = 512, 512
    if add_noise:
        w = int(w * random.uniform(0.95, 1.05))
        h = int(h * random.uniform(0.95, 1.05))
    
    fr = 60
    if add_noise:
        fr = random.randint(50, 75)
    
    ip = 0
    op = max(1, int(round(fr * max(0.001, time))))
    
    # --- color with variability ---
    def hex_to_rgba01(hexstr):
        s = hexstr.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
        return [r, g, b, 1]
    
    rgba = hex_to_rgba01(color)
    if add_noise:
        # Add slight color jitter (±0.08 in RGB space, clamped)
        rgba = [
            max(0, min(1, rgba[0] + random.uniform(-0.08, 0.08))),
            max(0, min(1, rgba[1] + random.uniform(-0.08, 0.08))),
            max(0, min(1, rgba[2] + random.uniform(-0.08, 0.08))),
            1
        ]
    
    # --- geometry with variability ---
    base_size = min(w, h) * 0.16
    shape_size = size if size is not None else base_size
    
    if add_noise:
        # Size jitter (±8%)
        shape_size *= random.uniform(0.92, 1.08)
    
    radius = shape_size / 2
    cx, cy = w / 2, h / 2
    
    if add_noise:
        cx += random.uniform(-w * 0.03, w * 0.03)
        cy += random.uniform(-h * 0.03, h * 0.03)
    
    x0, x1 = w * 0.2, w * 0.8
    y0, y1 = h * 0.2, h * 0.8
    
    if add_noise:
        margin_jitter = 0.05
        x0 += random.uniform(-w * margin_jitter, w * margin_jitter)
        x1 += random.uniform(-w * margin_jitter, w * margin_jitter)
        y0 += random.uniform(-h * margin_jitter, h * margin_jitter)
        y1 += random.uniform(-h * margin_jitter, h * margin_jitter)
    
    # --- easing with variability ---
    e = (easing_type or "").strip().lower()
    
    def add_easing_jitter(easing_dict):
        if not add_noise:
            return easing_dict
        result = {}
        for key in easing_dict:
            if isinstance(easing_dict[key], list):
                result[key] = [
                    max(0, min(1, v + random.uniform(-0.1, 0.1)))
                    for v in easing_dict[key]
                ]
            else:
                result[key] = easing_dict[key]
        return result
    
    if e == "ease-in":
        pos_in = add_easing_jitter({"x": [0.67, 0.67], "y": [1.0, 1.0]})
        pos_out = add_easing_jitter({"x": [0.33, 0.33], "y": [0.0, 0.0]})
        rot_in = add_easing_jitter({"x": [0.67], "y": [1.0]})
        rot_out = add_easing_jitter({"x": [0.33], "y": [0.0]})
    elif e == "ease-out":
        pos_in = add_easing_jitter({"x": [0.33, 0.33], "y": [1.0, 1.0]})
        pos_out = add_easing_jitter({"x": [0.67, 0.67], "y": [0.0, 0.0]})
        rot_in = add_easing_jitter({"x": [0.33], "y": [1.0]})
        rot_out = add_easing_jitter({"x": [0.67], "y": [0.0]})
    else:
        pos_in = pos_out = {"x": [0.5, 0.5], "y": [0.5, 0.5]}
        rot_in = rot_out = {"x": [0.5], "y": [0.5]}
    
    # --- keyframe builders ---
    def pos_kfs(start_xy, end_xy):
        return [{"t": ip, "s": start_xy, "e": end_xy, "i": pos_in, "o": pos_out}, {"t": op}]
    
    def rot_kfs(deg0, deg1):
        if add_noise:
            deg0 += random.uniform(-5, 5)
            deg1 += random.uniform(-5, 5)
        return [{"t": ip, "s": [deg0], "e": [deg1], "i": rot_in, "o": rot_out}, {"t": op}]
    
    # --- motion setup ---
    m = (motion or "").strip().lower()
    if m == "left-to-right":
        p_anim, spin_anim = pos_kfs([x0, cy], [x1, cy]), None
    elif m == "right-to-left":
        p_anim, spin_anim = pos_kfs([x1, cy], [x0, cy]), None
    elif m == "up-to-down":
        p_anim, spin_anim = pos_kfs([cx, y0], [cx, y1]), None
    elif m == "down-to-up":
        p_anim, spin_anim = pos_kfs([cx, y1], [cx, y0]), None
    elif m == "clockwise":
        p_anim, spin_anim = pos_kfs([cx, cy], [cx, cy]), rot_kfs(0, 360)
    elif m == "anticlockwise":
        p_anim, spin_anim = pos_kfs([cx, cy], [cx, cy]), rot_kfs(0, -360)
    else:
        motion_angle_re = re.compile(r"^(clockwise|anticlockwise)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*\)$")
        mo = motion_angle_re.match(m)
        if mo:
            direction = mo.group(1)
            ang = float(mo.group(2))
            final_angle = ang if direction == "clockwise" else -ang
            p_anim, spin_anim = pos_kfs([cx, cy], [cx, cy]), rot_kfs(0, final_angle)
        else:
            p_anim, spin_anim = pos_kfs([cx, cy], [cx, cy]), None
    
    # --- shape factory ---
    def parse_shape(shape_str: str):
        base = (shape_str or "circle").strip().lower()
        kind, arg = base, None
        if ":" in base:
            kind, arg = base.split(":", 1)
            arg = arg.strip()
        return kind, arg
    
    def make_shape_items(shape_str: str):
        kind, arg = parse_shape(shape_str)
        
        if kind == "circle":
            return [{
                "ty": "el",
                "p": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [shape_size, shape_size]},
                "nm": "Ellipse"
            }]
        
        if kind == "square":
            return [{
                "ty": "rc",
                "p": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [shape_size, shape_size]},
                "r": {"a": 0, "k": 0},
                "nm": "Square"
            }]
        
        if kind == "rounded-square":
            try:
                cr = float(arg) if arg is not None else shape_size * 0.12
            except ValueError:
                cr = shape_size * 0.12
            
            if add_noise:
                cr *= random.uniform(0.8, 1.2)
            
            return [{
                "ty": "rc",
                "p": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [shape_size, shape_size]},
                "r": {"a": 0, "k": cr},
                "nm": "Rounded Square"
            }]
        
        if kind == "triangle":
            return [{
                "ty": "sr",
                "sy": 2,
                "pt": {"a": 0, "k": 3},
                "p": {"a": 0, "k": [0, 0]},
                "r": {"a": 0, "k": -90},
                "or": {"a": 0, "k": radius},
                "os": {"a": 0, "k": 0},
                "nm": "Triangle"
            }]
        
        if kind == "polygon":
            try:
                n = max(3, int(arg)) if arg is not None else 5
            except ValueError:
                n = 5
            return [{
                "ty": "sr",
                "sy": 2,
                "pt": {"a": 0, "k": n},
                "p": {"a": 0, "k": [0, 0]},
                "r": {"a": 0, "k": -90},
                "or": {"a": 0, "k": radius},
                "os": {"a": 0, "k": 0},
                "nm": f"Polygon {n}-gon"
            }]
        
        if kind == "star":
            try:
                p = max(3, int(arg)) if arg is not None else 5
            except ValueError:
                p = 5
            
            inner_ratio = 0.5
            if add_noise:
                inner_ratio = random.uniform(0.4, 0.6)
            
            return [{
                "ty": "sr",
                "sy": 1,
                "pt": {"a": 0, "k": p},
                "p": {"a": 0, "k": [0, 0]},
                "r": {"a": 0, "k": -90},
                "or": {"a": 0, "k": radius},
                "ir": {"a": 0, "k": radius * inner_ratio},
                "os": {"a": 0, "k": 0},
                "is": {"a": 0, "k": 0},
                "nm": f"Star {p}-pt"
            }]
        
        return [{
            "ty": "el",
            "p": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [shape_size, shape_size]},
            "nm": "Ellipse"
        }]
    
    shape_items = make_shape_items(shape)
    
    # --- paint styles with variability ---
    stroke_width = 8
    dotted_stroke_width = 6
    
    if add_noise:
        stroke_width *= random.uniform(0.75, 1.25)
        dotted_stroke_width *= random.uniform(0.75, 1.25)
    
    stroke = {
        "ty": "st",
        "c": {"a": 0, "k": rgba},
        "o": {"a": 0, "k": 100},
        "w": {"a": 0, "k": stroke_width},
        "lc": 2,
        "lj": 2,
        "ml": 4,
        "nm": "Stroke"
    }
    
    actual_dot_spacing = dot_spacing
    if add_noise and dotted:
        actual_dot_spacing *= random.uniform(0.8, 1.2)
    
    dotted_stroke = {
        "ty": "st",
        "c": {"a": 0, "k": rgba},
        "o": {"a": 0, "k": 100},
        "w": {"a": 0, "k": dotted_stroke_width},
        "lc": 2,
        "lj": 2,
        "ml": 4,
        "nm": "Dotted Stroke",
        "d": [
            {"n": "d", "nm": "dash", "v": {"a": 0, "k": 2}},
            {"n": "g", "nm": "gap", "v": {"a": 0, "k": float(actual_dot_spacing)}}
        ],
    }
    
    fill = {
        "ty": "fl",
        "c": {"a": 0, "k": rgba},
        "o": {"a": 0, "k": 100},
        "nm": "Fill"
    }
    
    # Add scaling animation if requested
    scale_anim = None
    if scaling != 1.0:
        start_scale = 100
        end_scale = 100 * scaling
        
        if add_noise:
            start_scale *= random.uniform(0.95, 1.05)
            end_scale *= random.uniform(0.95, 1.05)
        
        scale_anim = [
            {"t": ip, "s": [start_scale, start_scale], "e": [end_scale, end_scale], "i": pos_in, "o": pos_out},
            {"t": op}
        ]
    
    group_tr = {
        "ty": "tr",
        "p": {"a": 0, "k": [0, 0]},
        "a": {"a": 0, "k": [0, 0]},
        "s": ({"a": 1, "k": scale_anim} if scale_anim else {"a": 0, "k": [100, 100]}),
        "r": ({"a": 1, "k": spin_anim} if spin_anim else {"a": 0, "k": 0}),
        "o": {"a": 0, "k": 100},
        "nm": "Transform"
    }
    
    # Combine
    if dotted:
        contents = shape_items + [dotted_stroke, group_tr]
    elif outline:
        contents = shape_items + [stroke, group_tr]
    else:
        contents = shape_items + [fill, group_tr]
    
    # --- layer ---
    layer_rot = [{"t": ip, "s": [0], "e": [0], "i": rot_in, "o": rot_out}, {"t": op}]
    layer = {
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Animated Shape",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {"a": 1, "k": layer_rot},
            "p": {"a": 1, "k": p_anim},
            "a": {"a": 0, "k": [0, 0, 0]},
            "s": {"a": 0, "k": [100, 100, 100]},
        },
        "ao": 0,
        "shapes": [{"ty": "gr", "nm": "Shape Group", "it": contents}],
        "ip": ip,
        "op": op,
        "st": ip,
        "bm": 0
    }
    
    return {
        "v": "5.7.6",
        "fr": fr,
        "ip": ip,
        "op": op,
        "w": w,
        "h": h,
        "nm": "Shape Motion",
        "ddd": 0,
        "assets": [],
        "layers": [layer],
    }

# === UNIFORM SAMPLING GENERATION (replace the old GENERATE block) ===

motions_linear = ["left-to-right", "right-to-left", "up-to-down", "down-to-up"]
angle_set = [90, 180, 270]
motions_rotate = (
    ["clockwise", "anticlockwise"]
    + [f"clockwise({a})" for a in angle_set]
    + [f"anticlockwise({a})" for a in angle_set]
)
motions = motions_linear + motions_rotate

easings = ["ease-in", "ease-out", "linear"]

shapes = [
    "circle", "square", "rounded-square:16", "triangle",
    "polygon:5", "polygon:6", "star:5", "star:8",
]

# Use semantic size names instead of pixel values
size_names = list(SIZE_DEFINITIONS.keys())

dotted_options = [False, True]
outline_options = [False, True]
dot_spacings = [20]

scaling_options = [0.5, 1.0, 1.5]

# Use semantic color names instead of hex values
color_names = list(COLOR_PALETTE.keys())

duration_seconds = 2.0

OUT_DIR = "dataset_variations/generated_data"
JSON_DIR = os.path.join(OUT_DIR, "json")
CAPTION_DIR = os.path.join(OUT_DIR, "caption")
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)

# # Helper functions
# motion_angle_re = re.compile(r"^(clockwise|anticlockwise)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*\)$")

# def parse_motion(m: str):
#     m = (m or "").strip().lower()
#     if m in {"left-to-right", "right-to-left", "up-to-down", "down-to-up"}:
#         return ("translate", None)
#     if m in {"clockwise", "anticlockwise"}:
#         return ("rotate", 360 if m == "clockwise" else -360)
#     mo = motion_angle_re.match(m)
#     if mo:
#         direction = mo.group(1)
#         ang = float(mo.group(2))
#         return ("rotate", ang if direction == "clockwise" else -ang)
#     return ("unknown", None)

# def is_visibly_animated(shape: str, motion: str, dotted: bool, outline: bool, scaling: float) -> bool:
#     if scaling is not None and abs(float(scaling) - 1.0) > 1e-9:
#         return True
#     kind, angle = parse_motion(motion)
#     if kind == "translate":
#         return True
#     if kind == "rotate":
#         if shape.startswith("circle"):
#             return False
#         if angle is None or abs(angle) < 1e-9:
#             return False
#         return True
#     return False

# def slug(s: str) -> str:
#     return re.sub(r"[^a-z0-9\-_.]+", "_", (s or "").lower())

# def shape_noun(shape: str) -> str:
#     s = (shape or "circle").lower()
#     if ":" in s:
#         kind, arg = s.split(":", 1)
#     else:
#         kind, arg = s, None
#     if kind == "circle":
#         return "circle"
#     if kind == "square":
#         return "square"
#     if kind == "rounded-square":
#         return "rounded square"
#     if kind == "triangle":
#         return "triangle"
#     if kind == "polygon":
#         try:
#             n = max(3, int(arg)) if arg else 5
#         except ValueError:
#             n = 5
#         return f"{n}-sided polygon"
#     if kind == "star":
#         try:
#             p = max(3, int(arg)) if arg else 5
#         except ValueError:
#             p = 5
#         return f"{p}-point star"
#     return s

# def describe_motion(m: str) -> str:
#     m = (m or "").strip().lower()
#     if m == "left-to-right":
#         return "moves from left to right"
#     if m == "right-to-left":
#         return "moves from right to left"
#     if m == "up-to-down":
#         return "moves from up to down"
#     if m == "down-to-up":
#         return "moves from down to up"
#     if m == "clockwise":
#         return "spins clockwise"
#     if m == "anticlockwise":
#         return "spins anticlockwise"
#     mo = motion_angle_re.match(m)
#     if mo:
#         direction = mo.group(1)
#         ang = float(mo.group(2))
#         return (f"rotates clockwise to {ang:g}°" if direction == "clockwise"
#                 else f"rotates anticlockwise to {ang:g}°")
#     return "moves"

# def build_caption(shape: str, dotted: bool, outline: bool, dot_spacing: float,
#                   motion: str, easing: str, scaling: float, duration_s: float,
#                   size_name: str, color_name: str) -> str:
#     style = "dotted" if dotted else ("outlined" if outline else "filled")
#     noun = shape_noun(shape)
#     motion_phrase = describe_motion(motion)
#     if abs(float(scaling) - 1.0) > 1e-9:
#         scale_phrase = f" and scales to {scaling:g}× its size"
#     else:
#         scale_phrase = " and keeps its size"
#     spacing_detail = f" with {dot_spacing:g}px spacing" if dotted else ""
#     easing_phrase = ("with ease-in easing" if easing == "ease-in"
#                      else ("with ease-out easing" if easing == "ease-out" else "with linear easing"))
#     dur = f"{duration_s:.1f}".rstrip("0").rstrip(".")
#     return f"A {size_name} {color_name} {style} {noun}{spacing_detail} that {motion_phrase}{scale_phrase}, {easing_phrase}, over {dur} seconds."

import random
import re

motion_angle_re = re.compile(r"^(clockwise|anticlockwise)\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*\)$")

# -----------------------------
# VARIATION HELPERS
# -----------------------------

def choose(options):
    return random.choice(options)

# For style variation
STYLE_PHRASES = {
    "dotted": [
        "dotted",
        "dot-pattern",
        "dotted-line"
    ],
    "outlined": [
        "outlined",
        "stroke-only",
        "line-art style"
    ],
    "filled": [
        "filled",
        "solid",
        "fully filled-in"
    ]
}

EASING_PHRASES = {
    "ease-in": [
        "using ease-in easing",
        "starting slowly with ease-in easing",
        "animated with an ease-in acceleration"
    ],
    "ease-out": [
        "using ease-out easing",
        "slowing down with ease-out easing",
        "animated with an ease-out deceleration"
    ],
    "linear": [
        "with linear easing",
        "moving at a steady linear rate",
        "animated at constant linear speed"
    ]
}

SCALE_TEMPLATES = [
    "and scales to {scaling:g}× its size",
    "scaling up to {scaling:g}× its original size",
    "changing size to {scaling:g}×",
]

NO_SCALE_TEMPLATES = [
    "and keeps its size",
    "without changing its size",
    "while maintaining its original size"
]

MOTION_TEMPLATES = [
    "that {motion_phrase}{scale_phrase}",
    "which {motion_phrase}{scale_phrase}",
    "as it {motion_phrase}{scale_phrase}",
]

DURATION_TEMPLATES = [
    "over {duration_s} seconds",
    "throughout a {duration_s}-second animation",
    "within {duration_s} seconds",
    "completed in {duration_s} seconds",
]

CAPTION_TEMPLATES = [
    "A {size_name} {color_name} {style} {noun}{spacing_detail} {motion_clause}, {easing_clause}, {duration_clause}.",
    "A {size_name} {color_name} {style} {noun}{spacing_detail} {motion_clause} {easing_clause} {duration_clause}.",
    "This {size_name} {color_name} {style} {noun}{spacing_detail} {motion_clause}, {easing_clause}, lasting {duration_clause}.",
]

# -----------------------------
# ORIGINAL PARSE FUNCTIONS
# -----------------------------

def parse_motion(m: str):
    m = (m or "").strip().lower()
    if m in {"left-to-right", "right-to-left", "up-to-down", "down-to-up"}:
        return ("translate", None)
    if m in {"clockwise", "anticlockwise"}:
        return ("rotate", 360 if m == "clockwise" else -360)
    mo = motion_angle_re.match(m)
    if mo:
        direction = mo.group(1)
        ang = float(mo.group(2))
        return ("rotate", ang if direction == "clockwise" else -ang)
    return ("unknown", None)


def is_visibly_animated(shape: str, motion: str, dotted: bool, outline: bool, scaling: float) -> bool:
    if scaling is not None and abs(float(scaling) - 1.0) > 1e-9:
        return True
    kind, angle = parse_motion(motion)
    if kind == "translate":
        return True
    if kind == "rotate":
        if shape.startswith("circle"):
            return False
        if angle is None or abs(angle) < 1e-9:
            return False
        return True
    return False


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9\-_.]+", "_", (s or "").lower())


# -----------------------------
# SMALL VARIATION FOR NOUN
# -----------------------------

def shape_noun(shape: str) -> str:
    s = (shape or "circle").lower()
    if ":" in s:
        kind, arg = s.split(":", 1)
    else:
        kind, arg = s, None

    # small variation options
    if kind == "circle":
        return choose(["circle", "circular shape"])
    if kind == "square":
        return choose(["square", "four-sided square"])
    if kind == "rounded-square":
        return choose(["rounded square", "soft-cornered square"])
    if kind == "triangle":
        return choose(["triangle", "three-sided triangle"])
    if kind == "polygon":
        try:
            n = max(3, int(arg)) if arg else 5
        except ValueError:
            n = 5
        return choose([f"{n}-sided polygon", f"{n}-gon"])
    if kind == "star":
        try:
            p = max(3, int(arg)) if arg else 5
        except ValueError:
            p = 5
        return choose([f"{p}-point star", f"{p}-pointed star"])
    return s


# -----------------------------
# VARIED MOTION DESCRIPTIONS
# -----------------------------

def describe_motion(m: str) -> str:
    m = (m or "").strip().lower()

    table = {
        "left-to-right": [
            "moves from left to right",
            "slides horizontally from left to right",
            "shifts left to right"
        ],
        "right-to-left": [
            "moves from right to left",
            "slides right to left",
            "shifts horizontally toward the left"
        ],
        "up-to-down": [
            "moves downward",
            "drops from top to bottom",
            "shifts from up to down"
        ],
        "down-to-up": [
            "moves upward",
            "rises from bottom to top",
            "shifts from down to up"
        ],
        "clockwise": [
            "spins clockwise",
            "rotates in a clockwise direction",
            "turns clockwise"
        ],
        "anticlockwise": [
            "spins anticlockwise",
            "rotates in an anticlockwise direction",
            "turns anticlockwise"
        ]
    }

    if m in table:
        return choose(table[m])

    mo = motion_angle_re.match(m)
    if mo:
        direction = mo.group(1)
        ang = float(mo.group(2))
        if direction == "clockwise":
            return choose([
                f"rotates clockwise to {ang:g}°",
                f"turns clockwise up to {ang:g}°",
                f"spins clockwise reaching {ang:g}°"
            ])
        else:
            return choose([
                f"rotates anticlockwise to {ang:g}°",
                f"turns anticlockwise up to {ang:g}°",
                f"spins anticlockwise reaching {ang:g}°"
            ])

    return choose(["moves", "shifts", "animates"])


# -----------------------------
# MAIN CAPTION GENERATOR WITH VARIATION
# -----------------------------

def build_caption(shape: str, dotted: bool, outline: bool, dot_spacing: float,
                  motion: str, easing: str, scaling: float, duration_s: float,
                  size_name: str, color_name: str) -> str:

    # varied style
    base_style = "dotted" if dotted else ("outlined" if outline else "filled")
    style = choose(STYLE_PHRASES[base_style])

    noun = shape_noun(shape)
    motion_phrase = describe_motion(motion)

    # scaling variation
    if abs(float(scaling) - 1.0) > 1e-9:
        scale_phrase = choose(SCALE_TEMPLATES).format(scaling=scaling)
    else:
        scale_phrase = choose(NO_SCALE_TEMPLATES)

    spacing_detail = f" with {dot_spacing:g}px spacing" if dotted else ""

    # easing variation
    easing_key = easing if easing in EASING_PHRASES else "linear"
    easing_clause = choose(EASING_PHRASES[easing_key])

    # unify motion clause
    motion_clause = choose(MOTION_TEMPLATES).format(
        motion_phrase=motion_phrase,
        scale_phrase=scale_phrase,
    )

    duration_clean = f"{duration_s:.1f}".rstrip("0").rstrip(".")
    duration_clause = choose(DURATION_TEMPLATES).format(duration_s=duration_clean)

    template = choose(CAPTION_TEMPLATES)

    return template.format(
        size_name=size_name,
        color_name=color_name,
        style=style,
        noun=noun,
        spacing_detail=spacing_detail,
        motion_clause=motion_clause,
        easing_clause=easing_clause,
        duration_clause=duration_clause,
    )

# === UNIFORM SAMPLER ===

def sample_params():
    motion = random.choice(motions)
    easing = random.choice(easings)
    shape = random.choice(shapes)
    size_name = random.choice(size_names)
    scaling = random.choice(scaling_options)
    color_name = random.choice(color_names)
    dotted = random.choice(dotted_options)
    if dotted:
        outline = False
        spacing = random.choice(dot_spacings)
    else:
        outline = random.choice(outline_options)
        spacing = 0
    return {
        "motion": motion,
        "easing": easing,
        "shape": shape,
        "size_name": size_name,
        "scaling": scaling,
        "color_name": color_name,
        "dotted": dotted,
        "outline": outline,
        "spacing": spacing,
    }

# Set your target number of samples here
N_SAMPLES = 300  # <--- change as needed
RANDOM_SEED = 42  # e.g., set to 42 for reproducible sampling

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

count = 0
skipped = 0
seen = set()  # to avoid duplicate parameter combos

MAX_ATTEMPTS = N_SAMPLES * 20  # generous cap to avoid infinite loops
attempts = 0

while count < N_SAMPLES and attempts < MAX_ATTEMPTS:
    attempts += 1
    params = sample_params()

    # Skip non-animated combos
    if not is_visibly_animated(
        params["shape"], params["motion"],
        dotted=params["dotted"], outline=params["outline"],
        scaling=params["scaling"]
    ):
        skipped += 1
        continue

    # Deduplicate exact param tuples (no noise fields included)
    key = (
        params["motion"], params["easing"], params["shape"], params["size_name"],
        params["scaling"], params["color_name"], params["dotted"], params["outline"], params["spacing"]
    )
    if key in seen:
        continue
    seen.add(key)

    # Compute concrete size and color
    canvas_min = 512
    size_pixels = canvas_min * SIZE_DEFINITIONS[params["size_name"]]
    base_color = COLOR_PALETTE[params["color_name"]]

    # Build animation
    data = generate_lottie_shape(
        outline=params["outline"],
        motion=params["motion"],
        time=duration_seconds,
        color=base_color,
        color_name=params["color_name"],
        easing_type=params["easing"],
        dotted=params["dotted"],
        dot_spacing=params["spacing"] if params["dotted"] else 20,
        shape=params["shape"],
        size=size_pixels,
        size_name=params["size_name"],
        scaling=params["scaling"],
        add_noise=True,
        seed=None
    )

    # Filename
    style_tag = "dotted-gap-{}".format(params["spacing"]) if params["dotted"] else ("outline" if params["outline"] else "fill")
    base = (
        f"{slug(params['shape'])}__{slug(params['motion'])}__{slug(params['easing'])}"
        f"__{slug(params['size_name'])}__{slug(params['color_name'])}__{style_tag}"
        f"__scale-{params['scaling']}__sample-{count}"
    )
    json_path = os.path.join(JSON_DIR, base + ".json")
    caption_path = os.path.join(CAPTION_DIR, base + ".txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    caption = build_caption(
        shape=params["shape"], dotted=params["dotted"], outline=params["outline"],
        dot_spacing=params["spacing"], motion=params["motion"], easing=params["easing"],
        scaling=params["scaling"], duration_s=duration_seconds,
        size_name=params["size_name"], color_name=params["color_name"]
    )
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

    count += 1

print(f"✅ Generated {count} uniformly sampled animations")
print(f"🎯 Target samples: {N_SAMPLES} (attempted {attempts}, dedup+skipped handled)")
print(f"🎨 Colors: {len(color_names)} semantic colors with noise")
print(f"📏 Sizes: {len(size_names)} semantic sizes (very small to very large)")
print(f"📁 Output: {OUT_DIR}/json and {OUT_DIR}/caption")
print(f"🚫 Skipped {skipped} non-animated or duplicate combinations")