#!/usr/bin/env python3
import os
import csv
import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
import imageio
from rlottie_python import LottieAnimation

TARGET_COLOR = np.array([255, 255, 255, 255], dtype=np.uint8)
TOL = 3  # small tolerance for compression artifacts

def is_single_color_white_video(video_path, tolerance=TOL):
    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return False

    # Define the target color (White) in BGR format
    # OpenCV reads color images as BGR by default
    target_bgr = np.array([255, 255, 255], dtype=np.uint8)

    # Flag to track if all frames are the target color
    is_white_video = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame has 3 channels (BGR)
        # Note: If the frame has an alpha channel (4 channels), this check will not work directly.
        if frame.ndim != 3 or frame.shape[2] != 3:
            print("Warning: Frame is not a standard 3-channel BGR image. Assuming not white.")
            is_white_video = False
            break
        
        diff = np.abs(frame.astype(np.int16) - target_bgr.astype(np.int16))
        
        if not np.all(diff <= tolerance):
            is_white_video = False
            break 

    # Release the video capture object
    cap.release()

    print("Video is all white" if is_white_video else "Not all white")
    return is_white_video

# def is_frame_flat(frame, target_color=TARGET_COLOR, tol=TOL):
#     diff = np.abs(frame.astype(np.int16) - target_color)
#     return np.all(diff <= tol)


# def is_video_flat_color(video_path, num_samples=12):
#     try:
#         reader = imageio.get_reader(video_path, "ffmpeg")
#         total_frames = reader.count_frames()
#     except Exception:
#         return False  # corrupted video

#     if total_frames == 0:
#         return True

#     indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

#     for idx in indices:
#         try:
#             frame = reader.get_data(idx)
#         except Exception:
#             return True  # unreadable frame → treat as invalid
#         if not is_frame_flat(frame):
#             return False

#     return True


# ============================================================
# LOTTIE VALIDATION (JSON STRUCTURE)
# ============================================================

def inspect_lottie(path: str):
    try:
        anim = LottieAnimation.from_file(path)
        width, height = anim.lottie_animation_get_size()
        duration = anim.lottie_animation_get_duration()
        total_frames = anim.lottie_animation_get_totalframe()
        framerate = anim.lottie_animation_get_framerate()

        if width == 0 or height == 0 or duration == 0 or total_frames == 0 or framerate == 0:
            status = "invalid-json"
        else:
            status = "valid-json"

        return {
            "file": path,
            "width": width,
            "height": height,
            "duration": duration,
            "frames": total_frames,
            "fps": framerate,
            "status": status,
        }

    except Exception as e:
        return {
            "file": path,
            "width": 0,
            "height": 0,
            "duration": 0,
            "frames": 0,
            "fps": 0,
            "status": f"invalid-json ({type(e).__name__}: {e})",
        }


# ============================================================
# MAIN VALIDATOR WITH VIDEO CHECK
# ============================================================

def validate_folder(folder: str,
                    video_dir: str,
                    export_csv: bool = True,
                    delete_invalid: bool = False,
                    force: bool = False):

    folder = Path(folder)
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = 0
    valid = 0
    invalid = 0
    flat_videos = 0

    invalid_files = []

    # --------------------------------------------------------
    # SCAN ALL JSON FILES
    # --------------------------------------------------------
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.lower().endswith(".json"):
                continue

            total += 1
            json_path = Path(root) / file

            # STEP 1 — JSON structure validation
            meta = inspect_lottie(str(json_path))
            print(f"{meta['status']} {json_path}")

            # If JSON invalid → record & continue
            if not meta["status"].startswith("valid-json"):
                invalid += 1
                invalid_files.append(str(json_path))
                results.append(meta)
                continue

            # STEP 2 — Export to MP4 (temporary)
            mp4_name = json_path.stem + ".mp4"
            mp4_path = video_dir / mp4_name

            cmd = [
                "lottie_convert.py",
                str(json_path),
                str(mp4_path)
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                meta["status"] = "invalid-video-export"
                invalid += 1
                invalid_files.append(str(json_path))
                results.append(meta)
                try:
                    # Attempt to delete the file, even if partially written
                    os.remove(mp4_path)
                except Exception:
                    pass
                continue

            # STEP 3 — Video flatness test
            # if is_video_flat_color(str(mp4_path)):
            if is_single_color_white_video(str(mp4_path)):
                meta["status"] = "invalid-flat-video"
                flat_videos += 1
                invalid += 1
                invalid_files.append(str(json_path))

                # delete empty/flat mp4
                try:
                    os.remove(mp4_path)
                except Exception:
                    pass

            else:
                meta["status"] = "valid"
                valid += 1

            results.append(meta)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n========== SUMMARY ==========")
    print(f"Total JSONs:            {total}")
    print(f"Valid (non-flat):       {valid}")
    print(f"Invalid (JSON):         {sum(1 for r in results if 'invalid-json' in r['status'])}")
    print(f"Invalid (export err):   {sum(1 for r in results if r['status']=='invalid-video-export')}")
    print(f"Flat videos:            {flat_videos}")
    print(f"TOTAL invalid:          {invalid}")

    # ============================================================
    # CSV OUTPUT
    # ============================================================
    if export_csv:
        csv_path = "lottie_validation_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "width", "height", "duration", "frames", "fps", "status"]
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\n📄 CSV summary saved to: {csv_path}")

    # ============================================================
    # DELETE INVALID JSONs
    # ============================================================
    if delete_invalid and invalid_files:
        print(f"\n{len(invalid_files)} invalid JSONs found.")

        if not force:
            confirm = input("Delete all invalid JSONs? [y/N]: ").strip().lower()
            if confirm != "y":
                print("Deletion cancelled.")
                return

        for p in invalid_files:
            try:
                os.remove(p)
                print(f"Deleted: {p}")
            except Exception as e:
                print(f"Failed to delete {p}: {e}")

        print(f"\nDeleted {len(invalid_files)} invalid JSON files.")


# ============================================================
# CLI ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Lottie JSONs + detect empty videos.")
    parser.add_argument("folder", help="Path to folder containing JSON files")
    parser.add_argument("--video_dir", required=True,
                        help="Where temporary MP4s should be created")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--delete-invalid", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    validate_folder(
        args.folder,
        video_dir=args.video_dir,
        export_csv=not args.no_csv,
        delete_invalid=args.delete_invalid,
        force=args.force,
    )
