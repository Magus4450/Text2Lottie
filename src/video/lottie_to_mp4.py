#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import imageio
from rlottie_python import LottieAnimation
import subprocess, os

def make_even(x: int) -> int:
    """Ensure dimension is even (required for yuv420p)."""
    return (int(x) // 2) * 2


def parse_hex_rgba(s: str):
    """Parse #RRGGBB or #RRGGBBAA into an (R,G,B,A) tuple."""
    s = s.lstrip("#")
    if len(s) == 6:
        s += "ff"
    if len(s) != 8:
        raise argparse.ArgumentTypeError("bg must be #RRGGBB or #RRGGBBAA")
    r, g, b, a = (int(s[i:i + 2], 16) for i in (0, 2, 4, 6))
    return (r, g, b, a)


def convert_lottie_to_mp4(input_file: Path, output_file: Path, args):
    """Convert a single Lottie JSON file to MP4."""
    with LottieAnimation.from_file(str(input_file)) as anim:
        src_w, src_h = anim.lottie_animation_get_size()
        lottie_fps = anim.lottie_animation_get_framerate()
        total_frames = anim.lottie_animation_get_totalframe()

        out_fps = args.fps if args.fps is not None else float(lottie_fps or 12.0)
        out_fps = max(1.0, min(out_fps, 30.0))

        # Output size
        if args.width and args.height:
            W, H = args.width, args.height
        elif args.width:
            W = args.width
            H = round(src_h * (W / src_w)) if src_w else args.width
        elif args.height:
            H = args.height
            W = round(src_w * (H / src_h)) if src_h else args.height
        else:
            W, H = src_w, src_h

        W, H = make_even(W), make_even(H)
        if W <= 0 or H <= 0:
            raise SystemExit(f"Invalid output size computed ({W}x{H})")

        n_frames = total_frames if args.max_frames is None else min(args.max_frames, total_frames)

        bg_rgba = args.bg
        solid_bg = Image.new("RGBA", (W, H), bg_rgba)

        ffmpeg_params = ["-movflags", "+faststart", "-preset", "veryfast", "-crf", "23"]
        with imageio.get_writer(
            str(output_file),
            fps=out_fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=ffmpeg_params,
        ) as writer:
            for f in range(n_frames):
                im = anim.render_pillow_frame(frame_num=f)
                if (im.width, im.height) != (W, H):
                    im = im.resize((W, H), resample=Image.BICUBIC)
                rgb = Image.alpha_composite(solid_bg, im).convert("RGB")
                writer.append_data(np.asarray(rgb))

    print(f"✅ {input_file.name} → {output_file.name} ({W}x{H}, {out_fps:.1f}fps, {n_frames} frames)")


def main():
    ap = argparse.ArgumentParser(description="Batch convert all Lottie JSONs in a folder to MP4s.")
    ap.add_argument("input_folder", type=Path, help="Folder containing .json Lottie files")
    ap.add_argument("output_folder", type=Path, help="Folder to save .mp4 files")
    ap.add_argument("--fps", type=float, default=None, help="Output FPS (default: Lottie FPS, capped at 60)")
    ap.add_argument("--width", type=int, default=None, help="Force output width")
    ap.add_argument("--height", type=int, default=None, help="Force output height")
    ap.add_argument("--bg", type=parse_hex_rgba, default="#00000000", help="Background color RGBA hex")
    ap.add_argument("--max-frames", type=int, default=None, help="Limit frames for testing")
    args = ap.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_folder.glob("*.json"))
    if not json_files:
        print(f"⚠️ No .json files found in {input_folder}")
        return

    print(f"🎬 Found {len(json_files)} Lottie files. Converting...\n")

    for json_file in json_files:
        output_file = output_folder / f"{json_file.stem}.mp4"
        try:
            convert_lottie_to_mp4(json_file, output_file, args)
        except Exception as e:
            print(f"❌ Error converting {json_file.name}: {e}")

        target_size=384
        output_str = str(output_file)
        subprocess.run([
            "ffmpeg", "-y", "-i", output_str,
            "-vf", f"scale={target_size}:{target_size}",
            "-c:a", "copy", output_str.replace(".mp4", "_small.mp4")
        ])
        os.replace(output_str.replace(".mp4", "_small.mp4"), output_str)

    print("\n✨ Conversion complete!")


if __name__ == "__main__":
    main()