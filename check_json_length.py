#!/usr/bin/env python3
import os
import json
import argparse

def count_chars_in_json_file(path):
    """Return character count of JSON file content ignoring spaces/tabs/newlines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None

    # Remove spaces, tabs, newlines
    cleaned = raw.replace(" ", "").replace("\t", "").replace("\n", "")

    return len(cleaned)


def process_folder(folder):
    if not os.path.isdir(folder):
        raise ValueError(f"Not a folder: {folder}")

    lengths = []
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue

        count = count_chars_in_json_file(fpath)
        if count is not None:
            lengths.append((fname, count))

    if not lengths:
        print("[INFO] No JSON files found or no readable files.")
        return

    # Compute statistics
    only_vals = [v for _, v in lengths]

    print("=== Character Statistics (ignoring whitespace) ===")
    print(f"Files processed : {len(only_vals)}")
    print(f"Min length      : {min(only_vals)}")
    print(f"Max length      : {max(only_vals)}")
    print(f"Average length  : {sum(only_vals)/len(only_vals):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute JSON character stats in a folder.")
    parser.add_argument("folder", help="Path to folder containing JSON files")
    args = parser.parse_args()

    process_folder(args.folder)
