#!/usr/bin/env python3
import os
import csv
from rlottie_python import LottieAnimation

def inspect_lottie(path: str):
    """Return animation metadata and validity."""
    try:
        anim = LottieAnimation.from_file(path)
        width, height = anim.lottie_animation_get_size()
        duration = anim.lottie_animation_get_duration()
        total_frames = anim.lottie_animation_get_totalframe()
        framerate = anim.lottie_animation_get_framerate()

        # Classify validity
        if width == 0 or height == 0 or duration == 0 or total_frames == 0 or framerate == 0:
            status = "invalid"
        else:
            status = "valid"

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
            "status": f"invalid ({type(e).__name__}: {e})",
        }

def validate_folder(folder: str, export_csv: bool = True, delete_invalid: bool = False, force: bool = False):
    results = []
    total, valid, invalid = 0, 0, 0
    invalid_files = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".json"):
                total += 1
                path = os.path.join(root, file)
                result = inspect_lottie(path)
                results.append(result)

                print(
                    f"{result['status']} {path}\n"
                    f"  size: {result['width']}x{result['height']}\n"
                    f"  duration: {result['duration']:.2f}s\n"
                    f"  frames: {result['frames']}\n"
                    f"  fps: {result['fps']}\n"
                )

                if result["status"].startswith("valid"):
                    valid += 1
                else:
                    invalid += 1
                    invalid_files.append(path)

    print("====== SUMMARY ======")
    print(f"Total:   {total}")
    print(f"Valid:   {valid}")
    print(f"Invalid: {invalid}")

    # Export CSV summary
    if export_csv:
        csv_path = os.path.join("lottie_validation_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "width", "height", "duration", "frames", "fps", "status"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📄 CSV summary saved to: {csv_path}")

    # Delete invalid files if requested
    if delete_invalid and invalid_files:
        print(f"\n  {len(invalid_files)} invalid files detected.")
        if not force:
            confirm = input("Do you want to delete all invalid JSONs? [y/N]: ").strip().lower()
            if confirm != "y":
                print("Deletion cancelled.")
                return

        for path in invalid_files:
            try:
                os.remove(path)
                print(f"Deleted: {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")

        print(f"\nDeleted {len(invalid_files)} invalid JSON files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Lottie JSONs and optionally delete invalid ones.")
    parser.add_argument("folder", help="Path to folder containing JSON files")
    parser.add_argument("--no-csv", action="store_true", help="Do not export CSV summary")
    parser.add_argument("--delete-invalid", action="store_true", help="Delete invalid JSONs after validation")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt when deleting invalid files")
    args = parser.parse_args()

    validate_folder(
        args.folder,
        export_csv=not args.no_csv,
        delete_invalid=args.delete_invalid,
        force=args.force,
    )
