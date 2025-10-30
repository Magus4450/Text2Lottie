#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from PIL import Image
from rlottie_python import LottieAnimation

def parse_hex_rgba(s: str):
    """Parse #RRGGBB or #RRGGBBAA into an (R,G,B,A) tuple."""
    s = s.lstrip("#")
    if len(s) == 6:
        s += "ff"
    if len(s) != 8:
        raise argparse.ArgumentTypeError("bg must be #RRGGBB or #RRGGBBAA")
    r, g, b, a = (int(s[i:i + 2], 16) for i in (0, 2, 4, 6))
    return (r, g, b, a)

def render_first_frame(input_file: Path, output_file: Path, bg_rgba):
    """Render the first frame of a Lottie JSON to a PNG."""
    with LottieAnimation.from_file(str(input_file)) as anim:
        src_w, src_h = anim.lottie_animation_get_size()
        frame = anim.render_pillow_frame(frame_num=0)
        bg = Image.new("RGBA", (src_w, src_h), bg_rgba)
        composed = Image.alpha_composite(bg, frame)
        composed.save(output_file, format="PNG")
        print(f"✅ Saved first frame: {output_file.name} ({src_w}x{src_h})")

def main():
    ap = argparse.ArgumentParser(description="Render first frame of all Lottie JSONs in a folder to PNGs.")
    ap.add_argument("input_folder", type=Path, help="Folder containing .json Lottie files")
    ap.add_argument("output_folder", type=Path, help="Folder to save .png files")
    ap.add_argument("--bg", type=parse_hex_rgba, default="#00000000", help="Background color RGBA hex")
    args = ap.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_folder.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {input_folder}")
        return

    print(f"Found {len(json_files)} Lottie files. Rendering first frames...\n")

    for json_file in json_files:
        output_file = output_folder / f"{json_file.stem}.png"
        try:
            render_first_frame(json_file, output_file, args.bg)
        except Exception as e:
            print(f"Error rendering {json_file.name}: {e}")

    print("\n✨ Rendering complete!")

if __name__ == "__main__":
    main()
