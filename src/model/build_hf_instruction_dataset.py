#!/usr/bin/env python3
"""
GLOBAL STRATIFIED DATASET BUILDER FOR LOTTIE INSTRUCTION FINETUNING

Rules:
------
1) Train contains:
   - normal_fwd (minus those used by val/test)
   - normal_rev
   - static_fwd
   - static_rev
   - static_augment

2) Val + Test contain ONLY:
   - normal_fwd

3) Stratification:
   - Only normal_fwd is used for splitting
   - Val & test maintain proportional distribution across dataset folders

4) Train/val/test JSONLs are generated separately.
"""

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
ROOT_DIR = "dataset_new"
WITH_STATIC = {"generated_data"}  # dataset folders that have static_json & static_caption

random.seed(42)


# -------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------
@dataclass
class Example:
    id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, str]


def mk_messages(up: str, ans: str):
    return [
        {"role": "user", "content": up},
        {"role": "assistant", "content": ans},
    ]


# -------------------------------------------------------------
# FILE READERS
# -------------------------------------------------------------
def read_text_files(folder: Path) -> Dict[str, str]:
    data = {}
    if folder.exists():
        for p in folder.glob("*.txt"):
            try:
                data[p.stem.replace(".json", "")] = p.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")
    return data


def read_json_files_as_string(folder: Path) -> Dict[str, str]:
    UNNECESSARY_KEYS = set()

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in UNNECESSARY_KEYS}
        elif isinstance(obj, list):
            return [_clean(x) for x in obj]
        else:
            return obj

    data = {}
    if folder.exists():
        for p in folder.glob("*.json"):
            try:
                obj = json.loads(p.read_text().strip())
                cleaned = _clean(obj)
                s = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
                s = "".join(s.split())  # remove all whitespace
                data[p.stem] = s
            except Exception as e:
                print(f"[WARN] Failed JSON {p}: {e}")
    return data


# -------------------------------------------------------------
# PROMPT BUILDERS
# -------------------------------------------------------------
def build_forward_prompt(caption: str, static: bool):
    if static:
        return f"Generate a static lottie JSON animation given the following description:\n{caption}"
    return f"Generate a lottie JSON animation given the following description:\n{caption}"


def build_reverse_prompt(json_str: str, static: bool):
    if static:
        return (
            f"What does the following static lottie JSON animation represent?\n\n"
            f"```json\n{json_str}\n```"
        )
    return (
        f"What does the following lottie JSON animation represent?\n\n"
        f"```json\n{json_str}\n```"
    )


def build_static_augment_prompt(static_json: str, desc: str):
    return (
        "Given a static lottie JSON animation, add animation with the given description.\n\n"
        f"Static:\n```json\n{static_json}\n```\n\nDescription:\n{desc}"
    )


# -------------------------------------------------------------
# PROCESS A SINGLE DATASET FOLDER
# -------------------------------------------------------------
def process_dataset(ds_name: str, ds_path: Path, with_static: bool):
    out = {
        "normal_fwd": [],
        "normal_rev": [],
        "static_fwd": [],
        "static_rev": [],
        "static_augment": [],
    }

    # load all required components
    anim_caps = read_text_files(ds_path / "animation_caption")
    anim_json = read_json_files_as_string(ds_path / "json")
    normal_keys = sorted(set(anim_caps) & set(anim_json))

    st_caps = read_text_files(ds_path / "static_caption") if with_static else {}
    st_json = read_json_files_as_string(ds_path / "static_json") if with_static else {}
    static_keys = sorted(set(st_caps) & set(st_json))

    triple_keys = (
        sorted(set(st_json) & set(anim_caps) & set(anim_json))
        if with_static else []
    )

    # ---- normal ----
    for k in normal_keys:
        out["normal_fwd"].append(Example(
            id=f"{ds_name}::normal::fwd::{k}",
            messages=mk_messages(build_forward_prompt(anim_caps[k], False), anim_json[k]),
            metadata={"dataset": ds_name, "type": "normal_fwd", "key": k}
        ))
        out["normal_rev"].append(Example(
            id=f"{ds_name}::normal::rev::{k}",
            messages=mk_messages(build_reverse_prompt(anim_json[k], False), anim_caps[k]),
            metadata={"dataset": ds_name, "type": "normal_rev", "key": k}
        ))

    # ---- static ----
    if with_static:
        for k in static_keys:
            out["static_fwd"].append(Example(
                id=f"{ds_name}::static::fwd::{k}",
                messages=mk_messages(build_forward_prompt(st_caps[k], True), st_json[k]),
                metadata={"dataset": ds_name, "type": "static_fwd", "key": k}
            ))
            out["static_rev"].append(Example(
                id=f"{ds_name}::static::rev::{k}",
                messages=mk_messages(build_reverse_prompt(st_json[k], True), st_caps[k]),
                metadata={"dataset": ds_name, "type": "static_rev", "key": k}
            ))

        for k in triple_keys:
            out["static_augment"].append(Example(
                id=f"{ds_name}::static::augment::{k}",
                messages=mk_messages(
                    build_static_augment_prompt(st_json[k], anim_caps[k]),
                    anim_json[k]
                ),
                metadata={"dataset": ds_name, "type": "static_augment", "key": k}
            ))

    for k, v in out.items():
        print(k, len(v))
    return out


# -------------------------------------------------------------
# GLOBAL STRATIFIED SPLIT FOR NORMAL_FWD ONLY
# -------------------------------------------------------------
def split_normal_fwd(dataset_examples, train_ratio=0.9, val_ratio=0.05):
    train, val, test = [], [], []

    for ds, parts in dataset_examples.items():
        nfwd = parts["normal_fwd"]
        n = len(nfwd)
        if n == 0:
            continue

        random.shuffle(nfwd)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(nfwd[:n_train])
        val.extend(nfwd[n_train:n_train + n_val])
        test.extend(nfwd[n_train + n_val:])

    return train, val, test


# -------------------------------------------------------------
# WRITE JSONL
# -------------------------------------------------------------
def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "id": ex.id,
                "messages": ex.messages,
                "metadata": ex.metadata
            }, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(examples)} → {path}")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    root = Path(ROOT_DIR)
    assert root.exists(), f"Root folder not found: {ROOT_DIR}"

    # Load all dataset folders
    dataset_examples = {}
    for ds_path in sorted(p for p in root.iterdir() if p.is_dir()):
        ds_name = ds_path.name
        print(f"[INFO] Processing {ds_name}")
        dataset_examples[ds_name] = process_dataset(ds_name, ds_path, ds_name in WITH_STATIC)

    # ---------- STRATIFIED SPLIT: ONLY NORMAL_FWD ----------
    nfwd_train, val, test = split_normal_fwd(dataset_examples)

    for ex in val: ex.metadata["split"] = "val"
    for ex in test: ex.metadata["split"] = "test"

    # ---------- BUILD TRAIN ----------
    train = []

    # include normal_fwd parts destined for train
    nfwd_train_ids = {e.id for e in nfwd_train}

    for ds, parts in dataset_examples.items():
        # normal_fwd: keep only those not in val/test
        for ex in parts["normal_fwd"]:
            if ex.id in nfwd_train_ids:
                ex.metadata["split"] = "train"
                train.append(ex)

        # add all other training-only types
        for typ in ["normal_rev", "static_fwd", "static_rev", "static_augment"]:
            for ex in parts[typ]:
                ex.metadata["split"] = "train"
                train.append(ex)

    # ---------- WRITE FILES ----------
    write_jsonl("train.jsonl", train)
    write_jsonl("val.jsonl", val)
    write_jsonl("test.jsonl", test)

    print("[DONE]")


if __name__ == "__main__":
    main()
