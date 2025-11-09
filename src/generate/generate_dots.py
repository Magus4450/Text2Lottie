#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, random
from typing import List, Dict, Any, Optional, Union
from numbers import Number

# ---------------------------
# Folders
# ---------------------------
JSON_SUBDIR = "jsons"
CAP_SUBDIR = "captions"
STATIC_CAP_SUBDIR = "static_captions"

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def hex_to_rgb01(s: str):
    s = s.strip()
    if s.startswith("#"): s = s[1:]
    if len(s) == 3: s = "".join(ch*2 for ch in s)
    if len(s) != 6: raise ValueError(f"Invalid hex color: #{s}")
    return [int(s[0:2],16)/255.0, int(s[2:4],16)/255.0, int(s[4:6],16)/255.0]

def color_to_rgb01(c: Union[str, List[float], tuple]):
    if isinstance(c, str): return hex_to_rgb01(c)
    if isinstance(c, (list, tuple)):
        if len(c) == 1 and isinstance(c[0], str): return hex_to_rgb01(c[0])
        if len(c) == 3 and all(isinstance(v, Number) for v in c):
            r,g,b = [float(v) for v in c]
            return [r/255.0, g/255.0, b/255.0] if any(v>1 for v in (r,g,b)) else [r,g,b]
    raise ValueError(f"Unsupported color: {c!r}")

# Easing like your working sample
EASE_I = {"x":[0.67], "y":[1.0]}
EASE_O = {"x":[0.33], "y":[0.0]}

# ---------------------------
# Keyframe helpers
# ---------------------------
def kf_rot(ip: int, op: int, end_angle: float):
    return {
        "a": 1,
        "k": [
            {"t": ip, "s":[0.0], "e":[end_angle], "i":EASE_I, "o":EASE_O},
            {"t": op}
        ]
    }

def kf_scale_pulse(ip: int, op: int, s0=100, s1=120):
    mid = ip + (op - ip)//2
    return {
        "a": 1,
        "k": [
            {"t": ip, "s":[s0, s0], "e":[s1, s1], "i":{"x":[0.67,0.67], "y":[1.0,1.0]}, "o":{"x":[0.33,0.33], "y":[0.0,0.0]}},
            {"t": mid, "s":[s1, s1], "e":[s0, s0]},
            {"t": op}
        ]
    }

def kf_color_cycle(ip: int, op: int, palette_rgba: List[List[float]]):
    steps = max(2, len(palette_rgba))
    kfs = []
    for i in range(steps):
        t = ip + int(round((op - ip) * (i / steps)))
        kfs.append({"t": t, "s": palette_rgba[i]})
    kfs.append({"t": op, "s": palette_rgba[0]})
    return {"a": 1, "k": kfs}

# ---------------------------
# Caption templates (descriptive, human, no numbers/px/hex)
# ---------------------------
CAPTION_TEMPLATES = [
  "Several small dots glide counter-clockwise around the center on a {shape_word} path, staying evenly spaced with a {pace_word} rhythm{style_suffix}.",
  "A band of colored dots orbits the center in a {shape_word} loop, moving together with {pace_word} motion{style_suffix}.",
  "Tiny dots circle a central focus along a {shape_word} track, creating a calm, continuous loop{style_suffix}.",
  "Multiple bright dots revolve around the middle on a {shape_word} orbit, keeping tidy spacing and a {pace_word} tempo{style_suffix}.",
  "A clean ring of dots sweeps counter-clockwise along a {shape_word} path, forming a smooth, repeating motion{style_suffix}."
]

STATIC_CAP_TEMPLATES = [
  "A {shape_word} arrangement of {count_word} {palette_word} dots rests evenly around the center, forming a neat ring of {dot_word} points.",
  "{count_word} {palette_word} dots sit evenly spaced on a {shape_word} orbit around the center, each one a {dot_word} accent.",
  "A tidy, centered ring of {count_word} {palette_word} dots forms a {shape_word} layout with {dot_word} marks.",
  "An orderly {shape_word} circle of {count_word} {palette_word} dots surrounds the center, each dot appearing {dot_word}.",
  "A minimal {shape_word} band of {count_word} {palette_word} dots encircles the middle in a balanced arrangement of {dot_word} elements."
]

# ---------------------------
# Wording helpers (map params -> human words)
# ---------------------------
def _shape_word(ellipse_ratio: float) -> str:
    # ≤1.0 in your generator; vary wording for feel
    if ellipse_ratio >= 0.95:
        return "circular"
    if ellipse_ratio >= 0.8:
        return "slightly elliptical"
    return "softly squashed ellipse"

def _count_word(n: int) -> str:
    return {3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight"}.get(int(n), "several")

def _dot_word(dot_size: float, radius: float) -> str:
    # relative to radius; keep simple buckets
    r = max(1.0, radius)
    ratio = float(dot_size) / r
    if ratio <= 0.06:  # very small vs orbit
        return "tiny"
    if ratio <= 0.1:
        return "small"
    return "compact"

def _pace_word(revs: float, duration: float) -> str:
    # loops per second ≈ revs/duration
    lps = 0 if duration == 0 else revs / duration
    if lps >= 0.8:
        return "quick"
    if lps >= 0.35:
        return "steady"
    return "slow"

def _palette_word(palette: List[Union[str, List[float]]]) -> str:
    # Map typical hues to human words; collapse to up to 3 unique tokens
    def name_one(c):
        r,g,b = color_to_rgb01(c)
        # gray check
        if abs(r-g) < 0.05 and abs(g-b) < 0.05:
            return "neutral"
        # hue-ish buckets
        if r >= g and r >= b:
            if g > b: return "orange"
            return "red"
        if g >= r and g >= b:
            if r > b: return "yellow"
            return "green"
        # b dominant
        if g > r: return "teal" if r < 0.2 else "cyan"
        return "blue"
    words = []
    for c in palette:
        w = name_one(c)
        if w not in words:
            words.append(w)
        if len(words) == 3: break
    # friendly combos
    combos = {
        ("red","orange","yellow"): "warm",
        ("blue","teal","cyan"): "cool",
        ("neutral",): "neutral",
    }
    for k,v in combos.items():
        if all(x in words for x in k):
            return v + " tones"
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return f"{words[0]}, {words[1]}, and {words[2]}"

def _style_suffix(enable_pulse: bool, enable_color_cycle: bool, minicrot_deg: float, trails: int, phase_jitter_deg: float) -> str:
    bits = []
    if enable_color_cycle: bits.append("gentle color shifts")
    if enable_pulse: bits.append("subtle pulsing")
    if trails > 0: bits.append("soft motion trails")
    # minicrot & jitter are very minor; mention only one for brevity
    if not bits and (abs(minicrot_deg) > 0.01 or phase_jitter_deg > 0.0):
        bits.append("a light organic feel")
    return "" if not bits else " with " + ", ".join(bits)

# ---------------------------
# Renderers (fill templates with human words only)
# ---------------------------
def render_caption(params: Dict[str, Any], tmpl_rng: random.Random) -> str:
    tpl = tmpl_rng.choice(CAPTION_TEMPLATES)
    return tpl.format(
        shape_word=_shape_word(params["ellipse_ratio"]),
        pace_word=_pace_word(params["revs"], params["duration"]),
        style_suffix=_style_suffix(
            params["enable_pulse"], params["enable_color_cycle"],
            params["minicrot_deg"], params["trails"], params["phase_jitter_deg"]
        )
    )

def render_static_caption(params: Dict[str, Any], tmpl_rng: random.Random) -> str:
    tpl = tmpl_rng.choice(STATIC_CAP_TEMPLATES)
    return tpl.format(
        shape_word=_shape_word(params["ellipse_ratio"]),
        count_word=_count_word(params["count"]),
        palette_word=_palette_word(params["palette"]),
        dot_word=_dot_word(params["dot_size"], params["radius"])
    )

# ---------------------------
# Core generator (stays robust for your player)
# ---------------------------
def make_orbiting_dots_enhanced(
    w=512, h=512,
    duration=2.0, fps=60,
    count=5,
    radius=140,
    ellipse_ratio=0.8,
    dot_size=12,
    palette=("#111827", "#00AEEF", "#22C55E", "#FF3366", "#F59E0B", "#8B5CF6"),
    revs=1.0,
    # Enhancements:
    enable_pulse=False,
    pulse_pct=20,
    enable_color_cycle=False,
    color_cycle_overrides: Optional[List[str]] = None,
    minicrot_deg=0.0,
    phase_jitter_deg=0.0,
    trails=0,
    trails_scales=(90, 80),
    trails_opac=(60, 30),
    bg: Optional[Union[str, List[float]]] = None
) -> Dict[str, Any]:
    # sanitize to match the robust constraints
    count = max(3, min(8, int(count)))
    fr = int(fps); ip = 0; op = int(round(duration * fr))
    cx, cy = w/2.0, h/2.0
    radius = float(max(110.0, radius))
    ellipse_ratio = float(max(0.6, min(1.0, ellipse_ratio)))
    max_dot = min(18.0, 0.2*radius)
    dot_size = float(max(10.0, min(max_dot, dot_size)))
    revs = float(max(1.0, min(2.0, revs)))
    rot_end = -360.0 * revs  # CCW only
    minicrot_deg = float(max(-15.0, min(15.0, minicrot_deg)))
    phase_jitter_deg = float(max(0.0, min(12.0, phase_jitter_deg)))
    trails = int(max(0, min(2, trails)))

    palette_rgb = [color_to_rgb01(c) for c in palette]
    bg_col = color_to_rgb01(bg) if bg is not None else None

    # Optional color cycle list (as RGBA)
    if color_cycle_overrides:
        cycle_rgba = [color_to_rgb01(c) + [1] for c in color_cycle_overrides]
    else:
        cycle_src = [palette_rgb[i % len(palette_rgb)] for i in range(3)]
        cycle_rgba = [c + [1] for c in cycle_src]

    layers: List[Dict[str, Any]] = []

    if bg_col is not None:
        layers.append({
            "ddd":0, "ind":1, "ty":4, "nm":"BG", "sr":1,
            "ks":{"o":{"a":0,"k":100},"r":{"a":0,"k":0},
                  "p":{"a":0,"k":[cx,cy,0]},
                  "a":{"a":0,"k":[0,0,0]},
                  "s":{"a":0,"k":[100,100,100]}},
            "shapes":[{
                "ty":"gr","nm":"BG Group","it":[
                    {"ty":"rc","p":{"a":0,"k":[0,0]},
                     "s":{"a":0,"k":[w,h]},
                     "r":{"a":0,"k":0},"d":1,"nm":"Rect"},
                    {"ty":"fl","c":{"a":0,"k":[*bg_col,1]},
                     "o":{"a":0,"k":100},"nm":"Fill"},
                    {"ty":"tr","p":{"a":0,"k":[0,0]},
                     "a":{"a":0,"k":[0,0]},
                     "s":{"a":0,"k":[100,100]},
                     "r":{"a":0,"k":0},"o":{"a":0,"k":100}}
                ]
            }],
            "ip": ip, "op": op, "st": ip, "bm": 0
        })

    main_layer = {
        "ddd":0, "ind":2, "ty":4, "nm":"Orbiting Dots (enhanced)", "sr":1,
        "ks":{"o":{"a":0,"k":100},"r":{"a":0,"k":0},
              "p":{"a":0,"k":[cx,cy,0]},
              "a":{"a":0,"k":[0,0,0]},
              "s":{"a":0,"k":[100,100,100]}},
        "ao":0, "shapes": [], "ip": ip, "op": op, "st": ip, "bm": 0
    }

    dot_items: List[Dict[str, Any]] = []
    rng = random.Random(1337)

    for i in range(count):
        base_col = palette_rgb[i % len(palette_rgb)]
        phase = (360.0 * i) / count
        if phase_jitter_deg > 0:
            phase += rng.uniform(-phase_jitter_deg, phase_jitter_deg)

        inner_tr = {
            "ty":"tr",
            "p":{"a":0,"k":[0,0]},
            "a":{"a":0,"k":[0,0]},
            "s": kf_scale_pulse(ip, op, 100, 100 + int(pulse_pct)) if enable_pulse else {"a":0,"k":[100,100]},
            "r":{"a":0,"k":0},
            "o":{"a":0,"k":100}
        }

        dot_fill = {
            "ty":"fl",
            "c": kf_color_cycle(ip, op, cycle_rgba) if enable_color_cycle else {"a":0,"k":[*base_col,1]},
            "o":{"a":0,"k":100},
            "nm":"Fill"
        }

        def make_dot_group(name: str, size_scale=100, opacity=100):
            return {
                "ty":"gr","nm":name,"it":[
                    {"ty":"el","p":{"a":0,"k":[0,0]},
                     "s":{"a":0,"k":[dot_size*size_scale/100.0, dot_size*size_scale/100.0]},
                     "d":1,"nm":"Ellipse"},
                    {**dot_fill},
                    {**inner_tr, "o": {"a":0,"k":opacity}}
                ]
            }

        dot_group_main = make_dot_group(f"Dot_{i}", size_scale=100, opacity=100)

        trail_groups: List[Dict[str, Any]] = []
        for t in range(trails):
            t_scale = (90, 80)[t] if t < 2 else 80
            t_opac  = (60, 30)[t] if t < 2 else 40
            trail_groups.append(make_dot_group(f"DotTrail{t}_{i}", size_scale=t_scale, opacity=t_opac))

        offset_group = {
            "ty":"gr","nm":f"Offset_{i}",
            "it": trail_groups + [dot_group_main] + [
                {"ty":"tr","p":{"a":0,"k":[radius,0]},
                 "a":{"a":0,"k":[0,0]},
                 "s":{"a":0,"k":[100,100]},
                 "r":{"a":0,"k":0},"o":{"a":0,"k":100}}
            ]
        }

        phase_group = {
            "ty":"gr","nm":f"Phase_{i}",
            "it":[
                offset_group,
                {"ty":"tr",
                 "p":{"a":0,"k":[0,0]},
                 "a":{"a":0,"k":[0,0]},
                 "s":{"a":0,"k":[100, 100*ellipse_ratio]},
                 "r":{"a":0,"k":phase},
                 "o":{"a":0,"k":100}}
            ]
        }

        if abs(minicrot_deg) > 0.01:
            orbit_i = {
                "ty":"gr","nm":f"OrbitMicro_{i}",
                "it":[
                    phase_group,
                    {"ty":"tr",
                     "p":{"a":0,"k":[0,0]},
                     "a":{"a":0,"k":[0,0]},
                     "s":{"a":0,"k":[100,100]},
                     "r": kf_rot(ip, op, float(minicrot_deg)),
                     "o":{"a":0,"k":100}}
                ]
            }
            dot_items.append(orbit_i)
        else:
            dot_items.append(phase_group)

    orbit_container = {
        "ty":"gr","nm":"OrbitContainer",
        "it": dot_items + [
            {"ty":"tr",
             "p":{"a":0,"k":[0,0]},
             "a":{"a":0,"k":[0,0]},
             "s":{"a":0,"k":[100,100]},
             "r": kf_rot(ip, op, rot_end),   # CCW rotation
             "o":{"a":0,"k":100}}
        ]
    }

    main_layer["shapes"].append(orbit_container)
    layers = [main_layer] if bg_col is None else [layers[0], main_layer]

    return {
        "v":"5.7.6",
        "fr": fr, "ip": ip, "op": op,
        "w": w, "h": h,
        "nm": "orbiting_dots_enhanced",
        "ddd": 0, "assets": [], "layers": layers
    }

# ---------------------------
# Dataset generator with JSON + captions + static captions
# ---------------------------
def generate_dataset(
    n: int,
    output_folder="out_orbits_plus",
    canvas=(512, 512),
    param_space: Optional[Dict[str, List[Any]]] = None,
    seed: int = 123
):
    if param_space is None:
        param_space = {
            "count": [3, 4, 5, 6, 7, 8],
            "radius": [110, 140, 170, 200],
            "ellipse_ratio": [0.6, 0.7, 0.8, 0.9, 1.0],  # ≤1.0 only
            "dot_size": [10, 12, 14, 16, 18],
            "palette": [
                ["#111827", "#00AEEF", "#22C55E"],
                ["#FF3366", "#F59E0B", "#8B5CF6"],
                ["#111827", "#E5E7EB", "#A3A3A3"]
            ],
            "revs": [1.0, 2.0],
            "duration": [1.2, 2.0, 2.5, 3.0],
            "fps": [30, 60],
            "bg": [None],
            # Enhancements
            "enable_pulse": [False, True],
            "pulse_pct": [15, 20, 25],
            "enable_color_cycle": [False, True],
            "minicrot_deg": [0.0, 5.0, -5.0, 8.0, -8.0],
            "phase_jitter_deg": [0.0, 5.0, 8.0, 10.0],
            "trails": [0, 1, 2],
        }

    rng = random.Random(seed)
    # Make subfolders
    json_dir = os.path.join(output_folder, JSON_SUBDIR); ensure_dir(json_dir)
    cap_dir = os.path.join(output_folder, CAP_SUBDIR); ensure_dir(cap_dir)
    static_dir = os.path.join(output_folder, STATIC_CAP_SUBDIR); ensure_dir(static_dir)

    meta_fp = os.path.join(output_folder, "metadata.jsonl")
    ensure_dir(output_folder)

    w, h = canvas

    with open(meta_fp, "w", encoding="utf-8") as meta_out:
        for i in range(n):
            pick = lambda k: rng.choice(param_space[k])

            params = {
                "count": pick("count"),
                "radius": pick("radius"),
                "ellipse_ratio": pick("ellipse_ratio"),
                "dot_size": pick("dot_size"),
                "palette": pick("palette"),
                "revs": pick("revs"),
                "duration": pick("duration"),
                "fps": pick("fps"),
                "bg": pick("bg"),
                "enable_pulse": pick("enable_pulse"),
                "pulse_pct": pick("pulse_pct"),
                "enable_color_cycle": pick("enable_color_cycle"),
                "minicrot_deg": pick("minicrot_deg"),
                "phase_jitter_deg": pick("phase_jitter_deg"),
                "trails": pick("trails"),
            }

            comp = make_orbiting_dots_enhanced(
                w=w, h=h,
                duration=params["duration"], fps=params["fps"],
                count=params["count"], radius=params["radius"],
                ellipse_ratio=params["ellipse_ratio"],
                dot_size=params["dot_size"], palette=params["palette"],
                revs=params["revs"], bg=params["bg"],
                enable_pulse=params["enable_pulse"],
                pulse_pct=params["pulse_pct"],
                enable_color_cycle=params["enable_color_cycle"],
                color_cycle_overrides=None,
                minicrot_deg=params["minicrot_deg"],
                phase_jitter_deg=params["phase_jitter_deg"],
                trails=params["trails"],
            )

            stem = (
                f"orbit_plus_{i:04d}"
                f"_n{params['count']}"
                f"_r{params['radius']}"
                f"_er{params['ellipse_ratio']}"
                f"_dot{params['dot_size']}"
                f"_rev{params['revs']}"
                f"_dur{params['duration']}"
                f"_fps{params['fps']}"
                f"_pulse{int(params['enable_pulse'])}"
                f"_cycle{int(params['enable_color_cycle'])}"
                f"_mr{int(params['minicrot_deg'])}"
                f"_j{int(params['phase_jitter_deg'])}"
                f"_tr{params['trails']}"
                f"_ccw"
            )

            # Write JSON
            json_path = os.path.join(json_dir, stem + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, ensure_ascii=False, separators=(",", ":"))

            # Write captions (use uniform sampling across templates)
            tmpl_rng = random.Random(seed + i)  # deterministic per file
            caption = render_caption(params, tmpl_rng)
            with open(os.path.join(cap_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption.strip() + "\n")

            static_caption = render_static_caption(params, tmpl_rng)
            with open(os.path.join(static_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(static_caption.strip() + "\n")

            # Metadata record
            meta_out.write(json.dumps({
                "file_stem": stem,
                "json": os.path.relpath(json_path, output_folder),
                "caption": os.path.relpath(os.path.join(cap_dir, stem + ".txt"), output_folder),
                "static_caption": os.path.relpath(os.path.join(static_dir, stem + ".txt"), output_folder),
                "params": params
            }) + "\n")

    print(f"✅ Wrote {n} samples to:\n  {json_dir}\n  {cap_dir}\n  {static_dir}\n  and {meta_fp}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    # Example run
    generate_dataset(n=8, output_folder="out_orbits_plus")
