#!/usr/bin/env python3
"""
Build an instruction-finetuning dataset (JSONL) from a Lottie dataset folder structure.

Assumptions
-----------
datasets/
  <dataset_name>/
    animation_caption/*.txt        # normal captions
    json/*.json                    # normal lottie JSONs
    static_caption/*.txt           # (optional) static captions
    static_json/*.json             # (optional) static lottie JSONs

Pairing is done by matching the *stem* (filename without extension).
Only pairs that exist on both sides are used to create supervised examples.

For datasets listed in WITH_STATIC, we create:
  - static forward (caption -> json) and static reverse (json -> caption)
  - normal forward (caption -> json) and normal reverse (json -> caption)
  - **NEW** static+description augment: given a static JSON and a (normal) description,
    produce an animated JSON. Requires a triple match on the same stem:
    static_json[k] + animation_caption[k] -> json[k]

For all others:
  - normal forward and normal reverse only

Output
------
- instruction_dataset.jsonl : one example per line in chat format
- stats.json                 : counts per dataset/category/direction
"""

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CONFIG
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ROOT_DIR = "raw_datasets"
OUTPUT_JSONL = "instruction_dataset.jsonl"
STATS_JSON = "stats.json"

# Datasets in this list are treated as having "static" subfolders.
WITH_STATIC = {"generated_data"}  # update as needed


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# UTILITIES
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def read_text_files(folder: Path) -> Dict[str, str]:
    """Return {stem: content} for all .txt in folder. Missing folder -> {}."""
    data = {}
    if folder.exists():
        for p in folder.glob("*.txt"):
            try:
                data[p.stem.replace(".json", "")] = p.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")
    return data


def read_json_files_as_string(folder: Path) -> Dict[str, str]:
    """
    Return {stem: cleaned_json_string} for all .json in folder.
    Removes unnecessary Lottie keys that don't affect animation.
    Missing folder -> {}.
    """
    UNNECESSARY_KEYS = {"nm", "mn", "v", "ddd", "sr", "cl", "ln", "bm", "hd"}

    def _clean(obj):
        if isinstance(obj, dict):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if k not in UNNECESSARY_KEYS
            }
        elif isinstance(obj, list):
            return [_clean(item) for item in obj]
        else:
            return obj

    data = {}
    if folder.exists():
        for p in folder.glob("*.json"):
            try:
                s = p.read_text(encoding="utf-8").strip()
                obj = json.loads(s)

                # Clean the Lottie JSON recursively
                cleaned = _clean(obj)

                # Serialize back to compact string
                data[p.stem] = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

            except Exception as e:
                print(f"[WARN] Failed to read/parse JSON {p}: {e}")

    return data

def mk_messages(user_prompt: str, assistant_answer: str) -> List[Dict[str, str]]:
    """Create a chat-format example (system optional; keep simple & universal)."""
    return [
        {"role": "user", "content": user_prompt.strip()},
        {"role": "assistant", "content": assistant_answer.strip()},
    ]


@dataclass
class Example:
    id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, str]


@dataclass
class Counter:
    static_fwd: int = 0
    static_rev: int = 0
    static_aug_anim: int = 0  # NEW: static JSON + description -> animated JSON
    normal_fwd: int = 0
    normal_rev: int = 0


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# BUILDERS
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def build_forward_prompt(caption: str, static: bool) -> str:
    if static:
        return f"Generate a static lottie JSON animation given the following description:\n{caption}"
    return f"Generate a lottie JSON animation given the following description:\n{caption}"


def build_reverse_prompt(json_str: str, static: bool) -> str:
    if static:
        return "What does the following static lottie JSON animation represent?\n\n```json\n" + json_str + "\n```"
    return "What does the following lottie JSON animation represent?\n\n```json\n" + json_str + "\n```"


def build_static_augment_prompt(static_json: str, animated_caption: str) -> str:
    return (
        "Given a static lottie JSON animation, add animation with the given description.\n\n"
        "Static:\n```json\n" + static_json + "\n```\n\n"
        "Description:\n" + animated_caption
    )


def make_pairs(captions: Dict[str, str], jsons: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Return list of (key, caption, json) only where both sides exist.
    """
    keys = sorted(set(captions.keys()) & set(jsons.keys()))
    return [(k, captions[k], jsons[k]) for k in keys]


def build_examples_for_dataset(ds_name: str, ds_path: Path, with_static: bool) -> Tuple[List[Example], Counter]:
    out: List[Example] = []
    ctr = Counter()

    # normal
    anim_caps = read_text_files(ds_path / "animation_caption")
    anim_json = read_json_files_as_string(ds_path / "json")
    normal_pairs = make_pairs(anim_caps, anim_json)

    # forward (caption -> json)
    for k, cap, jstr in normal_pairs:
        ex_id = f"{ds_name}::normal::fwd::{k}"
        user_prompt = build_forward_prompt(cap, static=False)
        messages = mk_messages(user_prompt, jstr)
        out.append(Example(
            id=ex_id,
            messages=messages,
            metadata={"dataset": ds_name, "category": "normal", "direction": "forward", "key": k}
        ))
        ctr.normal_fwd += 1

    # reverse (json -> caption)
    for k, cap, jstr in normal_pairs:
        ex_id = f"{ds_name}::normal::rev::{k}"
        user_prompt = build_reverse_prompt(jstr, static=False)
        messages = mk_messages(user_prompt, cap)
        out.append(Example(
            id=ex_id,
            messages=messages,
            metadata={"dataset": ds_name, "category": "normal", "direction": "reverse", "key": k}
        ))
        ctr.normal_rev += 1

    # static optional
    if with_static:
        st_caps = read_text_files(ds_path / "static_caption")
        st_json = read_json_files_as_string(ds_path / "static_json")
        static_pairs = make_pairs(st_caps, st_json)

        # forward (caption -> json)
        for k, cap, jstr in static_pairs:
            ex_id = f"{ds_name}::static::fwd::{k}"
            user_prompt = build_forward_prompt(cap, static=True)
            messages = mk_messages(user_prompt, jstr)
            out.append(Example(
                id=ex_id,
                messages=messages,
                metadata={"dataset": ds_name, "category": "static", "direction": "forward", "key": k}
            ))
            ctr.static_fwd += 1

        # reverse (json -> caption)
        for k, cap, jstr in static_pairs:
            ex_id = f"{ds_name}::static::rev::{k}"
            user_prompt = build_reverse_prompt(jstr, static=True)
            messages = mk_messages(user_prompt, cap)
            out.append(Example(
                id=ex_id,
                messages=messages,
                metadata={"dataset": ds_name, "category": "static", "direction": "reverse", "key": k}
            ))
            ctr.static_rev += 1

        # NEW: static augment — given static JSON + (normal) description -> animated JSON
        # Requires triple match across: static_json, animation_caption, animated json
        triple_keys = sorted(set(st_json.keys()) & set(anim_caps.keys()) & set(anim_json.keys()))
        for k in triple_keys:
            ex_id = f"{ds_name}::static::augment_anim::{k}"
            user_prompt = build_static_augment_prompt(st_json[k], anim_caps[k])
            messages = mk_messages(user_prompt, anim_json[k])
            out.append(Example(
                id=ex_id,
                messages=messages,
                metadata={
                    "dataset": ds_name,
                    "category": "static",
                    "direction": "augment_anim",
                    "key": k
                }
            ))
            ctr.static_aug_anim += 1

    return out, ctr


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MAIN
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def main():
    root = Path(ROOT_DIR)
    assert root.exists(), f"Root folder not found: {ROOT_DIR}"

    all_ds_dirs = [p for p in root.iterdir() if p.is_dir()]
    totals = {}
    all_examples: List[Example] = []

    for ds_path in sorted(all_ds_dirs):
        ds_name = ds_path.name
        print(f"[INFO] Processing: {ds_name}")
        exs, ctr = build_examples_for_dataset(ds_name, ds_path, ds_name in WITH_STATIC)
        all_examples.extend(exs)
        totals[ds_name] = asdict(ctr)
        print(f"       normal_fwd={ctr.normal_fwd} normal_rev={ctr.normal_rev} "
              f"static_fwd={ctr.static_fwd} static_rev={ctr.static_rev} static_aug_anim={ctr.static_aug_anim}")

    # Write JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ex in all_examples:
            obj = {"id": ex.id, "messages": ex.messages, "metadata": ex.metadata}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(all_examples)} examples -> {OUTPUT_JSONL}")

    # Write stats
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(totals, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote stats -> {STATS_JSON}")


if __name__ == "__main__":
    main()
