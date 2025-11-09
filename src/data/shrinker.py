#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

THRESHOLD = 10 * 1024  # 10 KB in bytes

def remove_large_files(root: Path, recursive: bool, ext: Optional[str], dry_run: bool):
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Error: '{root}' is not a directory.")

    files = root.rglob("*") if recursive else root.glob("*")
    removed, kept = 0, 0

    for p in files:
        if not p.is_file():
            continue
        if ext and p.suffix.lower() != ext.lower():
            kept += 1
            continue

        try:
            size = p.stat().st_size
        except OSError as e:
            print(f"Skipping {p} (stat error: {e})")
            continue

        # Delete strictly greater than 10 KB; keep 10 KB and smaller
        if size > THRESHOLD:
            print(f"{'[DRY-RUN] ' if dry_run else ''}Removing: {p} ({size} bytes)")
            if not dry_run:
                try:
                    p.unlink()
                except OSError as e:
                    print(f"  Failed to remove {p}: {e}")
                    continue
            removed += 1
        else:
            kept += 1

    print(f"\nSummary: removed={removed}, kept={kept}, threshold={THRESHOLD} bytes")

def main():
    ap = argparse.ArgumentParser(
        description="Remove files strictly larger than 10 KB (keeps 10 KB and smaller)."
    )
    ap.add_argument("folder", type=Path, help="Folder to scan")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("-e", "--ext", help="Only consider files with this extension (e.g., .json)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    args = ap.parse_args()

    remove_large_files(args.folder, args.recursive, args.ext, args.dry_run)

if __name__ == "__main__":
    main()

