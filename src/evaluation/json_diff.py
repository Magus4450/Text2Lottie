#!/usr/bin/env python3
import os
import csv
import json
from typing import Any, List, Dict, Tuple
import numpy as np


# ------------------------------------------------------------
# Load JSON
# ------------------------------------------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Recursively collect FULL KEY PATHS (shape)
# ------------------------------------------------------------
def collect_key_paths(obj: Any, prefix="") -> List[str]:
    paths = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}/{k}"
            paths.append(p)
            paths.extend(collect_key_paths(v, p))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            p = f"{prefix}[{i}]"
            paths.append(p)
            paths.extend(collect_key_paths(item, p))

    return paths


# ------------------------------------------------------------
# Recursively collect LEAF VALUES ONLY
# Returns: list of (path, value)
# ------------------------------------------------------------
def collect_leaf_values(obj: Any, prefix="") -> List[Tuple[str, Any]]:
    leaves = []

    if isinstance(obj, dict):
        # dive deeper
        for k, v in obj.items():
            leaves.extend(collect_leaf_values(v, f"{prefix}/{k}"))
        return leaves

    if isinstance(obj, list):
        # dive deeper
        for i, item in enumerate(obj):
            leaves.extend(collect_leaf_values(item, f"{prefix}[{i}]"))
        return leaves

    # If here, primitive → leaf node
    leaves.append((prefix, obj))
    return leaves


# ------------------------------------------------------------
# Main Eval Logic
# ------------------------------------------------------------
def evaluate(json_gen, json_gold):
    # ---------------------------------------------------------
    # 1. KEY STRUCTURE DIFFERENCE
    # ---------------------------------------------------------
    keys_gen = set(collect_key_paths(json_gen))
    keys_gold = set(collect_key_paths(json_gold))

    missing_keys = keys_gold - keys_gen       # expected but not in generated
    extra_keys   = keys_gen - keys_gold       # generated but gold doesn't have

    raw_edit_distance = len(missing_keys) + len(extra_keys)
    # normalized_edit = raw_edit_distance / max(1, len(keys_gold))

    # ---------------------------------------------------------
    # 2. LEAF VALUE DIFFERENCE (ONLY FOR MATCHING KEYS)
    # ---------------------------------------------------------
    leaves_gen  = dict(collect_leaf_values(json_gen))
    leaves_gold = dict(collect_leaf_values(json_gold))

    numeric_diffs = []
    categorical_diffs = []

    # Only compute diffs for paths that exist in both JSONs
    matching_paths = leaves_gold.keys() & leaves_gen.keys()

    for path in matching_paths:
        gold_val = leaves_gold[path]
        gen_val  = leaves_gen[path]

        # Equal → no mismatch
        if gold_val == gen_val:
            continue

        # Numeric mismatch
        if isinstance(gen_val, (int, float)) and isinstance(gold_val, (int, float)):
            
            # Avoid inf issues — handle gold == 0 safely
            if gold_val == 0:
                if gen_val == 0:
                    pct = 0.0
                else:
                    pct = abs(gen_val - gold_val) / abs(gen_val)  # absolute error
            else:
                pct = abs(gen_val - gold_val) / abs(gold_val)
            
            pct = np.log(1 + pct)  # log-scale

            numeric_diffs.append((path, gen_val, gold_val, pct))

        else:
            # Categorical mismatch
            categorical_diffs.append((path, gen_val, gold_val))


    # Compute aggregated metrics
    avg_numeric_pct = (
        sum(p[3] for p in numeric_diffs) / len(numeric_diffs)
        if numeric_diffs else 0.0
    )

    categorical_error_rate = (
        len(categorical_diffs) / len(leaves_gold)
        if leaves_gold else 0.0
    )

    return {
        "key_mismatch": {
            "missing_keys": sorted(list(missing_keys)),
            "extra_keys": sorted(list(extra_keys)),
            "raw_edit_distance": raw_edit_distance,
            # "normalized_edit_distance": normalized_edit,
            "total_keys_gold": len(keys_gold),
        },

        "value_mismatch": {
            "numeric_mismatches": numeric_diffs,
            "avg_numeric_pct_diff": avg_numeric_pct,
            "categorical_mismatches": categorical_diffs,
            "categorical_error_rate": categorical_error_rate,
            "total_leaf_values_gold": len(leaves_gold)
        }
    }


def main(gold_dir="gold", gen_dir="gen", output_csv="lottie_diff_results.csv"):
    rows = []

    gold_files = {os.path.splitext(f)[0]: f for f in os.listdir(gold_dir) if f.endswith(".json")}
    gen_files  = {os.path.splitext(f)[0]: f for f in os.listdir(gen_dir)  if f.endswith(".json")}

    shared = sorted(set(gold_files.keys()) & set(gen_files.keys()))

    if not shared:
        print("No shared JSON filenames between gold/ and gen/.")
        return

    # Accumulators for global stats
    key_errors = []
    value_errors = []
    category_errors = []

    for name in shared:
        gold_path = os.path.join(gold_dir, gold_files[name])
        gen_path  = os.path.join(gen_dir,  gen_files[name])

        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        with open(gen_path, "r", encoding="utf-8") as f:
            gen = json.load(f)

        result = evaluate(gen, gold)

        # key_error       = result["key_mismatch"]["normalized_edit_distance"]
        key_error       = result["key_mismatch"]["raw_edit_distance"]
        value_error     = result["value_mismatch"]["avg_numeric_pct_diff"]
        category_error  = result["value_mismatch"]["categorical_error_rate"]

        rows.append([name, key_error, value_error, category_error])

        # add to accumulators
        key_errors.append(key_error)
        value_errors.append(value_error)
        category_errors.append(category_error)

    # -----------------------------
    # GLOBAL AGGREGATE STATISTICS
    # -----------------------------
    global_stats = {
        "avg_key_error": float(np.mean(key_errors)),
        "med_key_error": float(np.median(key_errors)),
        "avg_value_error": float(np.mean(value_errors)),
        "med_value_error": float(np.median(value_errors)),
        "avg_category_error": float(np.mean(category_errors)),
        "med_category_error": float(np.median(category_errors)),
        "std_key_error": float(np.std(key_errors)),
        "std_value_error": float(np.std(value_errors)),
        "std_category_error": float(np.std(category_errors)),
        "min_key_error": float(np.min(key_errors)),
        "max_key_error": float(np.max(key_errors)),
        "min_value_error": float(np.min(value_errors)),
        "max_value_error": float(np.max(value_errors)),
        "min_category_error": float(np.min(category_errors)),
        "max_category_error": float(np.max(category_errors)),
        "num_files": len(shared)
    }

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "key_error", "value_error", "category_error"])
        writer.writerows(rows)

    print(f"Saved results to {output_csv}")
    
    # -----------------------------
    # PRINT AGGREGATE STATISTICS
    # -----------------------------
    print("\n=== GLOBAL STATISTICS ===")
    for k, v in global_stats.items():
        print(f"{k}: {v}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Lottie JSON diffs in gold/ and gen/ folders.")
    parser.add_argument("--gold", type=str, default="gold", help="Path to gold JSON folder")
    parser.add_argument("--gen", type=str, default="gen", help="Path to generated JSON folder")
    parser.add_argument("--out", type=str, default="lottie_diff_results.csv", help="Output CSV file name")

    args = parser.parse_args()

    main(gold_dir=args.gold, gen_dir=args.gen, output_csv=args.out)

