#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, random
from typing import List, Dict, Any

# ---------------------------
# Folders
# ---------------------------
JSON_SUBDIR = "json"
CAP_SUBDIR = "animation_caption"
STATIC_CAP_SUBDIR = "static_caption"

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def hex_to_rgb01(s: str):
    s = s.strip()
    if s.startswith("#"): s = s[1:]
    if len(s) == 3: s = "".join(ch*2 for ch in s)
    return [int(s[0:2],16)/255.0, int(s[2:4],16)/255.0, int(s[4:6],16)/255.0]

# ---------------------------
# Captions (descriptive, human)
# ---------------------------
CAPTION_TEMPLATES = [
  "Soft rings radiate from the center, expanding and fading in a gentle staggered rhythm.",
  "Concentric outlines bloom outward one after another, growing larger as they dissolve.",
  "A calm ripple effect: thin circles expand from the middle and fade away in sequence.",
  "Waves of delicate rings rise from the center, each swelling outward before vanishing.",
  "A pulsing halo spreads in repeating rings, expanding smoothly and drifting into transparency."
]

STATIC_CAP_TEMPLATES = [
  "A centered set of thin circular outlines arranged for a clean ripple look.",
  "A neat, minimal circle outline at the center, ready to echo outward.",
  "A simple, centered ring with a light stroke, poised for a ripple effect.",
  "A quiet, balanced circle outline sits in the middle.",
  "A single, delicate ring rests at the center of the canvas."
]

def render_caption(_params, rng):   return rng.choice(CAPTION_TEMPLATES)
def render_static_caption(_params, rng): return rng.choice(STATIC_CAP_TEMPLATES)

# ---------------------------
# Core: ripple / pulsing rings
# ---------------------------
def make_ripples(
    w=512, h=512,
    duration=2.0, fps=60,
    ring_count=3,              # number of rings
    max_diam_ratio=0.85,       # final diameter as fraction of min(w,h)
    stroke_px=8,               # stroke width (kept constant)
    color_hex="#00AEEF",       # uniform color for all rings
    stagger_ratio=0.15,        # how much the starts are offset (0..1)
    life_ratio=0.65,           # fraction of duration each ring lives (expands+fades)
    add_noise=True             # <-- new parameter for variability
) -> Dict[str, Any]:
    """
    Each ring is its own layer (ty:4) centered on the canvas.
    - Ellipse path size (el.s) animates from tiny -> max
    - Layer opacity (ks.o) animates 100 -> 0 over its life window
    Offsets (stagger) create the ripple.
    Now includes variability/noise across multiple parameters.
    """
    # Apply canvas size variability (±5%)
    if add_noise:
        w = int(w * random.uniform(0.95, 1.05))
        h = int(h * random.uniform(0.95, 1.05))
    
    # Apply FPS variability
    if add_noise:
        fps = int(fps * random.uniform(0.85, 1.15))
    
    fr = int(fps); ip = 0; op = int(round(duration * fr))
    cx, cy = w/2, h/2
    min_side = min(w, h)

    # Apply max diameter variability (±8%)
    if add_noise:
        max_diam_ratio *= random.uniform(0.92, 1.08)
    
    max_diam = min_side * max(0.2, min(0.98, max_diam_ratio))
    start_diam = max(2.0, max_diam * 0.05)  # tiny but nonzero start

    # Apply start diameter variability (±20%)
    if add_noise:
        start_diam *= random.uniform(0.80, 1.20)

    # Apply stroke width variability (±15%)
    stroke_px_actual = stroke_px
    if add_noise:
        stroke_px_actual *= random.uniform(0.85, 1.15)

    # timings with variability
    life_frames = max(4, int(op * max(0.2, min(0.95, life_ratio))))
    
    # Apply life ratio variability (±10%)
    if add_noise:
        life_frames = int(life_frames * random.uniform(0.90, 1.10))
    
    # stagger time between ring starts
    if ring_count > 1:
        stagger_frames = int(op * max(0.02, min(0.5, stagger_ratio)) / (ring_count - 1))
        # Apply stagger variability (±15%)
        if add_noise:
            stagger_frames = int(stagger_frames * random.uniform(0.85, 1.15))
    else:
        stagger_frames = 0

    # Color with variability
    col = hex_to_rgb01(color_hex)
    if add_noise:
        # Add slight color jitter (±0.08 in RGB space, clamped)
        col = [
            max(0, min(1, col[0] + random.uniform(-0.08, 0.08))),
            max(0, min(1, col[1] + random.uniform(-0.08, 0.08))),
            max(0, min(1, col[2] + random.uniform(-0.08, 0.08)))
        ]
    
    # Apply center position variability (±3%)
    if add_noise:
        cx += random.uniform(-w * 0.03, w * 0.03)
        cy += random.uniform(-h * 0.03, h * 0.03)

    def ring_layer(idx: int):
        # start and end for this ring
        t0 = ip + idx * stagger_frames
        
        # Add timing jitter per ring (±5%)
        if add_noise:
            t0 = int(t0 * random.uniform(0.95, 1.05))
        
        t1 = min(op, t0 + life_frames)

        # Individual ring max diameter variability (±5%)
        ring_max_diam = max_diam
        if add_noise:
            ring_max_diam *= random.uniform(0.95, 1.05)
        
        # Individual ring start diameter variability (±15%)
        ring_start_diam = start_diam
        if add_noise:
            ring_start_diam *= random.uniform(0.85, 1.15)

        # Easing variability
        ease_in_x = [0.67, 0.67]
        ease_in_y = [1.0, 1.0]
        ease_out_x = [0.33, 0.33]
        ease_out_y = [0.0, 0.0]
        
        if add_noise:
            ease_in_x = [max(0, min(1, x + random.uniform(-0.1, 0.1))) for x in ease_in_x]
            ease_in_y = [max(0, min(1, y + random.uniform(-0.1, 0.1))) for y in ease_in_y]
            ease_out_x = [max(0, min(1, x + random.uniform(-0.1, 0.1))) for x in ease_out_x]
            ease_out_y = [max(0, min(1, y + random.uniform(-0.1, 0.1))) for y in ease_out_y]

        # Size keys (ellipse path size)
        size_kf = {
            "a": 1,
            "k": [
                {"t": t0, "s": [ring_start_diam, ring_start_diam], "e": [ring_max_diam, ring_max_diam],
                 "i": {"x": ease_in_x, "y": ease_in_y}, "o": {"x": ease_out_x, "y": ease_out_y}},
                {"t": t1}
            ]
        }

        # Opacity transition point with variability (default 60% into life)
        opacity_fade_point = 0.6
        if add_noise:
            opacity_fade_point = random.uniform(0.5, 0.7)
        
        # Starting opacity with variability
        start_opacity = 100
        if add_noise:
            start_opacity = random.uniform(90, 100)

        # Opacity easing variability
        opa_ease_in_x = [0.67]
        opa_ease_in_y = [1.0]
        opa_ease_out_x = [0.33]
        opa_ease_out_y = [0.0]
        
        if add_noise:
            opa_ease_in_x = [max(0, min(1, opa_ease_in_x[0] + random.uniform(-0.1, 0.1)))]
            opa_ease_in_y = [max(0, min(1, opa_ease_in_y[0] + random.uniform(-0.1, 0.1)))]
            opa_ease_out_x = [max(0, min(1, opa_ease_out_x[0] + random.uniform(-0.1, 0.1)))]
            opa_ease_out_y = [max(0, min(1, opa_ease_out_y[0] + random.uniform(-0.1, 0.1)))]

        # Opacity keys (layer opacity)
        opa_kf = {
            "a": 1,
            "k": [
                {"t": t0, "s": [start_opacity], "e": [start_opacity]},
                {"t": int(t0 + opacity_fade_point*(t1 - t0)), "s": [start_opacity], "e": [0],
                 "i": {"x": opa_ease_in_x, "y": opa_ease_in_y}, "o": {"x": opa_ease_out_x, "y": opa_ease_out_y}},
                {"t": t1, "s": [0]}
            ]
        }

        # Individual ring stroke width variability (±10%)
        ring_stroke = stroke_px_actual
        if add_noise:
            ring_stroke *= random.uniform(0.90, 1.10)

        layer = {
            "ddd": 0, "ind": idx+1, "ty": 4, "nm": f"Ring_{idx}", "sr": 1,
            "ks": {
                "o": opa_kf,                 # fade on layer opacity
                "r": {"a":0,"k":0},
                "p": {"a":0,"k":[cx, cy, 0]},
                "a": {"a":0,"k":[0,0,0]},
                "s": {"a":0,"k":[100,100,100]}
            },
            "shapes": [{
                "ty": "gr", "nm": f"RingShape_{idx}", "it": [
                    {"ty":"el","p":{"a":0,"k":[0,0]}, "s": size_kf, "d":1, "nm":"Ellipse"},
                    {"ty":"st","c":{"a":0,"k":[*col,1]}, "o":{"a":0,"k":100}, "w":{"a":0,"k":ring_stroke}, "lc":2, "lj":2, "ml":4, "nm":"Stroke"},
                    {"ty":"tr","p":{"a":0,"k":[0,0]}, "a":{"a":0,"k":[0,0]},
                     "s":{"a":0,"k":[100,100]}, "r":{"a":0,"k":0}, "o":{"a":0,"k":100}}
                ]
            }],
            "ip": ip, "op": op, "st": ip, "bm": 0
        }
        return layer

    layers = [ring_layer(i) for i in range(ring_count)]

    return {
        "v":"5.7.6",
        "fr": fr, "ip": ip, "op": op,
        "w": w, "h": h,
        "nm": "ripples_with_variability",
        "ddd": 0, "assets": [],
        "layers": layers
    }

# ---------------------------
# Dataset generator (JSON + captions + static captions)
# ---------------------------
def generate_dataset(
    n: int,
    output_folder="out_ripples",
    canvas=(512, 512),
    seed: int = 123,
    param_space: Dict[str, List[Any]] = None
):
    if param_space is None:
        param_space = {
            "ring_count": [4, 8],
            "max_diam_ratio": [0.3, 0.7, 0.9],
            "stroke_px": [6, 12],
            "color_hex": ["#00AEEF", "#22C55E", "#8B5CF6", "#FF3366", "#F59E0B", "#111827"],
            "stagger_ratio": [0.10, 0.30],
            "life_ratio": [0.55, 0.65, 0.75],
            "duration": [2],
            "fps": [30],
        }

    rng = random.Random(seed)
    # Set Python's random seed for noise generation
    random.seed(seed)
    
    ensure_dir(output_folder)
    json_dir = os.path.join(output_folder, JSON_SUBDIR); ensure_dir(json_dir)
    cap_dir = os.path.join(output_folder, CAP_SUBDIR); ensure_dir(cap_dir)
    static_dir = os.path.join(output_folder, STATIC_CAP_SUBDIR); ensure_dir(static_dir)
    meta_fp = os.path.join(output_folder, "metadata.jsonl")

    w, h = canvas

    with open(meta_fp, "w", encoding="utf-8") as meta_out:
        for i in range(n):
            pick = lambda k: rng.choice(param_space[k])
            params = {
                "ring_count": pick("ring_count"),
                "max_diam_ratio": pick("max_diam_ratio"),
                "stroke_px": pick("stroke_px"),
                "color_hex": pick("color_hex"),
                "stagger_ratio": pick("stagger_ratio"),
                "life_ratio": pick("life_ratio"),
                "duration": pick("duration"),
                "fps": pick("fps"),
            }

            comp = make_ripples(
                w=w, h=h,
                duration=params["duration"], fps=params["fps"],
                ring_count=params["ring_count"],
                max_diam_ratio=params["max_diam_ratio"],
                stroke_px=params["stroke_px"],
                color_hex=params["color_hex"],
                stagger_ratio=params["stagger_ratio"],
                life_ratio=params["life_ratio"],
                add_noise=True  # <-- enable variability
            )

            stem = (
                f"ripple_{i:04d}"
                f"_r{params['ring_count']}"
                f"_mx{params['max_diam_ratio']}"
                f"_sw{params['stroke_px']}"
                f"_{params['color_hex'][1:].lower()}"
                f"_st{params['stagger_ratio']}"
                f"_life{params['life_ratio']}"
                f"_dur{params['duration']}"
                f"_fps{params['fps']}"
            )

            # JSON
            json_path = os.path.join(json_dir, stem + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, ensure_ascii=False, separators=(",", ":"))

            # Captions (descriptive, human)
            tmpl_rng = random.Random(seed + i)
            caption = render_caption(params, tmpl_rng)
            with open(os.path.join(cap_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption.strip() + "\n")

            # Static captions (human, no px/hex/coords)
            static_caption = render_static_caption(params, tmpl_rng)
            with open(os.path.join(static_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(static_caption.strip() + "\n")

            # Metadata
            meta_out.write(json.dumps({
                "file_stem": stem,
                "json": os.path.relpath(json_path, output_folder),
                "caption": os.path.relpath(os.path.join(cap_dir, stem + ".txt"), output_folder),
                "static_caption": os.path.relpath(os.path.join(static_dir, stem + ".txt"), output_folder),
                "params": params
            }) + "\n")

    print(f"✅ Wrote {n} samples with variability to:\n  {json_dir}\n  {cap_dir}\n  {static_dir}\n  and {meta_fp}")
    print(f"🎨 Variability added to: canvas size, FPS, diameters, stroke width, colors, timing, easing, and opacity")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    generate_dataset(n=5, output_folder="dataset_for_masked/ripples")