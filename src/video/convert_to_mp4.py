#!/usr/bin/env python3

import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch convert Lottie JSONs to MP4.")
    parser.add_argument("--json_dir", required=True, help="Folder containing Lottie JSON files")
    parser.add_argument("--out_dir", required=True, help="Folder where MP4 outputs are saved")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found.")
        return

    for json_path in json_files:
        mp4_name = json_path.stem + ".mp4"
        mp4_path = out_dir / mp4_name

        cmd = [
            "lottie_convert.py",
            str(json_path),
            str(mp4_path)
        ]

        print(f"Converting: {json_path.name} → {mp4_name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed: {json_path}")

if __name__ == "__main__":
    main()
