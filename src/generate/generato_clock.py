#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, random
from typing import List, Dict, Any, Optional

# ---------------------------
# Subfolders
# ---------------------------
JSON_SUBDIR = "jsons"
CAP_SUBDIR = "captions"
STATIC_CAP_SUBDIR = "static_captions"

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
  "A clean timer dial with a single hand sweeping smoothly around the center, guided by simple tick marks.",
  "A minimal clock face: a lone hand rotates in a steady loop while slim ticks mark the circle.",
  "A quiet circular timer where the hand glides around the dial and the ticks define the rhythm.",
  "A simple rotating hand orbits the center, with tidy tick marks framing the motion.",
  "A crisp timer look: one hand turns around a ring of evenly spaced ticks."
]

STATIC_CAP_TEMPLATES = [
  "A centered circular dial with a single hand and evenly spaced tick marks.",
  "A minimal clock face at the center, one hand and neat ticks around it.",
  "A tidy round layout: a lone hand at the center, ticks arranged evenly on the perimeter.",
  "A clean circular timer with one hand and simple, even tick marks.",
  "A calm, centered dial ringed by uniform ticks and a single central hand."
]

def render_caption(_p, rng): return rng.choice(CAPTION_TEMPLATES)
def render_static_caption(_p, rng): return rng.choice(STATIC_CAP_TEMPLATES)

# ---------------------------
# Core generator
# ---------------------------
def make_clock_timer(
    w=512, h=512,
    duration=2.0, fps=60,
    radius=180,               # dial radius
    tick_count=60,            # number of ticks
    major_every=5,            # every Nth tick slightly longer
    hand_color="#111827",
    tick_color="#111827",
    bg_color=None,            # None or hex
    revs=1.0,                 # how many full rotations
    clockwise=True,
    ease="linear",            # "linear" or "ease"
    hand_width=10,            # px
    hand_len_ratio=0.9,       # relative to radius
    tick_len_small=10,        # px
    tick_len_major=18,        # px
) -> Dict[str, Any]:
    """
    - Static ticks layer (groups with rc+fl+tr).
    - One hand layer; rotate via layer ks.r from start to end angle.
    - Anchor: layer ks.a = [0,0,0]; layer ks.p = [cx,cy,0];
      hand rectangle is positioned so its base sits at (0,0), pointing upward.
    """
    fr = int(fps); ip = 0; op = int(round(duration * fr))
    cx, cy = w//2, h//2
    r = int(radius)
    hand_len = int(round(r * hand_len_ratio))

    hand_col = hex_to_rgb01(hand_color)
    tick_col = hex_to_rgb01(tick_color)
    bg_col = hex_to_rgb01(bg_color) if bg_color else None

    # ----- Optional background layer -----
    layers: List[Dict[str, Any]] = []
    if bg_col is not None:
        layers.append({
            "ddd":0,"ind":1,"ty":4,"nm":"BG","sr":1,
            "ks":{"o":{"a":0,"k":100},"r":{"a":0,"k":0},
                  "p":{"a":0,"k":[cx,cy,0]},
                  "a":{"a":0,"k":[0,0,0]},
                  "s":{"a":0,"k":[100,100,100]}},
            "shapes":[{
                "ty":"gr","nm":"BGGroup","it":[
                    {"ty":"rc","p":{"a":0,"k":[0,0]},"s":{"a":0,"k":[w,h]},
                     "r":{"a":0,"k":0},"d":1,"nm":"Rect"},
                    {"ty":"fl","c":{"a":0,"k":[*bg_col,1]},"o":{"a":0,"k":100},"nm":"Fill"},
                    {"ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},
                     "s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":100}}
                ]
            }],
            "ip":ip,"op":op,"st":ip,"bm":0
        })
        base_ind = 2
    else:
        base_ind = 1

    # ----- Tick layer (static) -----
    tick_groups = []
    for i in range(max(1, int(tick_count))):
        ang = (360.0 * i) / max(1, tick_count)
        is_major = (i % max(1, int(major_every))) == 0
        tlen = tick_len_major if is_major else tick_len_small
        # Build a tick as a small vertical rectangle whose base is at radius
        # We place it so center of the rect is at (0, -(r - tlen/2))
        tick_groups.append({
            "ty":"gr","nm":f"Tick_{i}","it":[
                {"ty":"rc","p":{"a":0,"k":[0,0]},
                 "s":{"a":0,"k":[max(2, hand_width//3), tlen]},
                 "r":{"a":0,"k":max(0, hand_width//6)},
                 "d":1,"nm":"Rect"},
                {"ty":"fl","c":{"a":0,"k":[*tick_col,1]},"o":{"a":0,"k":100},"nm":"Fill"},
                {"ty":"tr",
                 "p":{"a":0,"k":[0, -(r - tlen/2)]},
                 "a":{"a":0,"k":[0,0]},
                 "s":{"a":0,"k":[100,100]},
                 "r":{"a":0,"k":ang},
                 "o":{"a":0,"k":100}}
            ]
        })

    tick_layer = {
        "ddd":0,"ind":base_ind,"ty":4,"nm":"Ticks","sr":1,
        "ks":{"o":{"a":0,"k":100},"r":{"a":0,"k":0},
              "p":{"a":0,"k":[cx,cy,0]},
              "a":{"a":0,"k":[0,0,0]},
              "s":{"a":0,"k":[100,100,100]}},
        "shapes": tick_groups + [{
            "ty":"tr","p":{"a":0,"k":[0,0]},"a":{"a":0,"k":[0,0]},
            "s":{"a":0,"k":[100,100]},"r":{"a":0,"k":0},"o":{"a":0,"k":100}
        }],
        "ip":ip,"op":op,"st":ip,"bm":0
    }
    layers.append(tick_layer)

    # ----- Hand layer (animated rotation) -----
    # Hand is a vertical rectangle, base at origin, pointing up.
    hand_shape = {
        "ty":"gr","nm":"Hand","it":[
            {"ty":"rc","p":{"a":0,"k":[0,0]},
             "s":{"a":0,"k":[hand_width, hand_len]},
             "r":{"a":0,"k":max(0, hand_width/3)},
             "d":1,"nm":"Rect"},
            {"ty":"fl","c":{"a":0,"k":[*hand_col,1]},"o":{"a":0,"k":100},"nm":"Fill"},
            {"ty":"tr",
             "p":{"a":0,"k":[0, -(hand_len/2)]},  # move so base sits at layer origin
             "a":{"a":0,"k":[0,0]},
             "s":{"a":0,"k":[100,100]},
             "r":{"a":0,"k":0},
             "o":{"a":0,"k":100}}
        ]
    }

    # Rotation keys on the LAYER (about its anchor at [0,0,0])
    start = 0.0
    direction = -1.0 if clockwise else 1.0
    end = start + direction * 360.0 * float(revs)
    t0, t1 = ip, op

    if ease == "ease":
        kf_r = {
            "a":1,"k":[
                {"t": t0, "s":[start], "e":[end],
                 "i":{"x":[0.67], "y":[1.0]}, "o":{"x":[0.33], "y":[0.0]}},
                {"t": t1}
            ]
        }
    else:  # linear
        kf_r = {"a":1,"k":[{"t": t0, "s":[start], "e":[end]}, {"t": t1}]}

    hand_layer = {
        "ddd":0,"ind":base_ind+1,"ty":4,"nm":"Hand","sr":1,
        "ks":{
            "o":{"a":0,"k":100},
            "r": kf_r,                              # animate rotation
            "p":{"a":0,"k":[cx,cy,0]},              # center of dial
            "a":{"a":0,"k":[0,0,0]},                # rotate around base of hand (origin)
            "s":{"a":0,"k":[100,100,100]}
        },
        "shapes":[hand_shape],
        "ip":ip,"op":op,"st":ip,"bm":0
    }
    layers.append(hand_layer)

    return {
        "v":"5.7.6","fr":fr,"ip":ip,"op":op,
        "w":w,"h":h,"nm":"clock_timer",
        "ddd":0,"assets":[],
        "layers": layers
    }

# ---------------------------
# Dataset generator
# ---------------------------
def generate_dataset(
    n: int,
    output_folder="out_clock_timer",
    canvas=(512, 512),
    seed: int = 123,
    param_space: Optional[Dict[str, List[Any]]] = None
):
    if param_space is None:
        param_space = {
            "radius": [160, 180, 200],
            "tick_count": [12, 24, 30, 60],
            "hand_color": ["#111827","#00AEEF","#22C55E","#FF3366","#F59E0B","#8B5CF6"],
            "tick_color": ["#111827","#A3A3A3","#E5E7EB"],
            "revs": [0.25, 0.5, 1.0, 2.0],   # quarter-turn to full loops
            "clockwise": [True, False],
            "ease": ["linear","ease"],
            "duration": [1.2, 1.6, 2.0, 3.0],
            "fps": [30, 60],
            "hand_width": [8, 10, 12],
            "hand_len_ratio": [0.8, 0.9, 0.95],
            "tick_len_small": [8, 10, 12],
            "tick_len_major": [14, 18, 22],
            "bg_color": [None]  # keep transparent by default for reliability
        }

    rng = random.Random(seed)
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
                "radius": pick("radius"),
                "tick_count": pick("tick_count"),
                "hand_color": pick("hand_color"),
                "tick_color": pick("tick_color"),
                "revs": pick("revs"),
                "clockwise": pick("clockwise"),
                "ease": pick("ease"),
                "duration": pick("duration"),
                "fps": pick("fps"),
                "hand_width": pick("hand_width"),
                "hand_len_ratio": pick("hand_len_ratio"),
                "tick_len_small": pick("tick_len_small"),
                "tick_len_major": pick("tick_len_major"),
                "bg_color": pick("bg_color"),
            }

            comp = make_clock_timer(
                w=w, h=h,
                duration=params["duration"], fps=params["fps"],
                radius=params["radius"], tick_count=params["tick_count"],
                hand_color=params["hand_color"], tick_color=params["tick_color"],
                bg_color=params["bg_color"],
                revs=params["revs"], clockwise=params["clockwise"], ease=params["ease"],
                hand_width=params["hand_width"], hand_len_ratio=params["hand_len_ratio"],
                tick_len_small=params["tick_len_small"], tick_len_major=params["tick_len_major"]
            )

            stem = (
                f"clock_{i:04d}"
                f"_r{params['radius']}"
                f"_t{params['tick_count']}"
                f"_{'cw' if params['clockwise'] else 'ccw'}"
                f"_rev{params['revs']}"
                f"_{params['ease']}"
                f"_dur{params['duration']}"
                f"_fps{params['fps']}"
            )

            # JSON
            json_path = os.path.join(json_dir, stem + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, ensure_ascii=False, separators=(",", ":"))

            # Captions
            tmpl_rng = random.Random(seed + i)
            caption = render_caption(params, tmpl_rng)
            with open(os.path.join(cap_dir, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption.strip() + "\n")

            # Static captions
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

    print(f"✅ Wrote {n} samples to:\n  {json_dir}\n  {cap_dir}\n  {static_dir}\n  and {meta_fp}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    # Example: 12 samples
    generate_dataset(n=12, output_folder="out_clock_timer")
