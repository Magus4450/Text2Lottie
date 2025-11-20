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
import numpy as np

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
ROOT_DIR = "dataset_for_masked"
WITH_STATIC = {"SVG2Lottie","generated_data"}  # dataset folders that have static_json & static_caption

random.seed(42)

def shuffle_sibling_kv(obj):
    """
    Recursively shuffle only sibling key-value pairs in dictionaries.
    Lists are kept as-is but their elements are recursively processed.
    """

    # If list → process children, do NOT shuffle list order
    if isinstance(obj, list):
        return [shuffle_sibling_kv(x) for x in obj]

    # If dict → shuffle only *this level's* keys
    elif isinstance(obj, dict):
        # First process the children so internal order stays valid
        items = [(k, shuffle_sibling_kv(v)) for k, v in obj.items()]

        # Now shuffle ONLY these siblings
        random.shuffle(items)

        # Rebuild the dict preserving shuffled order
        return {k: v for k, v in items}

    # Primitive → return directly
    else:
        return obj


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
    UNNECESSARY_KEYS = set(["meta"])

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
            # try:
            obj = json.loads(p.read_text().strip())
            cleaned = _clean(obj)
            s = shuffle_sibling_kv(cleaned)
            s = json.dumps(s, ensure_ascii=False, separators=(",", ":"))

            s = "".join(s.split())  # remove all whitespace
            data[p.stem] = s
            # except Exception as e:
            #     print(f"[WARN] Failed JSON {p}: {e}")
    return data


# -------------------------------------------------------------
# PROMPT BUILDERS
# -------------------------------------------------------------
def build_forward_prompt(caption: str, static: bool):
    if static:
        return f"Generate lottie JSON with only the object following instruction:\n{caption}. Only return a JSON."
    return f"Generate a lottie JSON animation given the following description:\n{caption}. Only return a JSON."


def build_reverse_prompt(json_str: str, static: bool):
    if static:
        return (
            f"What is the object that is represented by this Lottie JSON?\n\n"
            f"```json\n{json_str}\n```"
        )
    return (
        f"What does the following lottie JSON animation represent?\n\n"
        f"```json\n{json_str}\n```"
    )


def build_static_augment_prompt(static_json: str, desc: str):
    return (
        "Given a lottie JSON of a static object, add animation to it following the given description.\n\n"
        f"Static:\n```json\n{static_json}\n```\n\nDescription:\n{desc}"
    )

def remove_random_layers_from_string(json_str):
    data = json.loads(json_str)

    layers = data.get("layers", [])
    total_layers = len(layers)

    if total_layers <= 1:
        # Nothing removed
        return json.dumps(data, indent=2), []

    # Choose random number of layers to remove (1–3 but not removing all)
    k = random.choice([1, 2])
    num_to_remove = min(total_layers - 1, k)

    indices = list(range(total_layers))
    remove_indices = set(random.sample(indices, num_to_remove))

    removed_layers = [layers[idx] for idx in sorted(remove_indices)]

    new_layers = [
        layer for idx, layer in enumerate(layers)
        if idx not in remove_indices
    ]

    data["layers"] = new_layers

    # j_res = "".join(str(json.dumps(data, indent=0)).split())
    # l_res = ["".join(str(k).split()) for k in removed_layers]

    # return j_res, l_res
    return "".join(str(json.dumps(data, indent=0)).split()), ["".join(str(k).split()) for k in removed_layers]

def build_lottie_layer_masked_prompt(masked_json: str, desc: str):
    masked_json = "".join(masked_json.split())
    return (
        "Given the description of a lottie JSON animation and its corresponding JSON with some layers removed, complete the JSON to represent the description.\n\n"
        f"Masked JSON:\n```json\n{masked_json}\n```\n\nDescription:\n{desc}\n. Only return a JSON."
    )

# -------------------------------------------------------------
# PROCESS A SINGLE DATASET FOLDER
# -------------------------------------------------------------
def process_dataset(ds_name: str, ds_path: Path, with_static: bool):
    train_out = {
        "normal_fwd": [],
        "normal_rev": [],
        "static_fwd": [],
        "static_rev": [],
        "static_augment": [],
        "normal_masked": [],
        "static_masked": [],
    }
    val_out = {
        "normal_fwd": [],
        "static_fwd": [],
    }
    test_out = {
        "normal_fwd": [],
        "static_fwd": [],
    }

    # load all required components
    if "SVG2Lottie" not in ds_name:
        anim_caps = read_text_files(ds_path / "animation_caption")
        anim_json = read_json_files_as_string(ds_path / "json")
    else:
        anim_caps = read_text_files(ds_path / "static_caption")
        anim_json = read_json_files_as_string(ds_path / "static_json")

    normal_keys = sorted(set(anim_caps) & set(anim_json))

    n = len(normal_keys)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)
    np.random.shuffle(normal_keys)

    train_keys = normal_keys[:n_train]
    val_keys = normal_keys[n_train:n_train + n_val]
    test_keys = normal_keys[n_train + n_val:]

    # ---------------------------------------------------------
    # PREFIX GROUPING ENFORCEMENT
    # ---------------------------------------------------------
    def get_prefix(name):
        parts = name.split("-")
        return "-".join(parts[:4]) if len(parts) >= 4 else name

    # Build mapping prefix → list of keys
    prefix_map = {}
    for k in normal_keys:
        px = get_prefix(k)
        prefix_map.setdefault(px, []).append(k)

    # Collect prefixes chosen for train
    train_prefixes = {get_prefix(k) for k in train_keys}

    # Expand train keys to include *all* items from those prefixes
    expanded_train = set(train_keys)
    for px in train_prefixes:
        expanded_train.update(prefix_map.get(px, []))

    # Remove entries from val/test
    expanded_train = sorted(expanded_train)
    expanded_train = set(expanded_train)

    val_keys = [k for k in val_keys if k not in expanded_train]
    test_keys = [k for k in test_keys if k not in expanded_train]

    train_keys = sorted(expanded_train)

    # -------------------

    st_caps = read_text_files(ds_path / "static_caption") if with_static else {}
    st_json = read_json_files_as_string(ds_path / "static_json") if with_static else {}
    static_keys = sorted(set(st_caps) & set(st_json) & set(train_keys))
    # if "SVG2Lottie" in ds_name:
    #     static_keys = sorted(set(st_caps) & set(st_json))

    #     # control the number of static json
    #     np.random.shuffle(static_keys)
    #     static_keys = static_keys[:5_000]

    triple_keys = (
        sorted(set(st_json) & set(anim_caps) & set(anim_json) & set(train_keys))
        if with_static else []
    )

    # ---- normal ----
    if "SVG2Lottie" not in ds_name:
        for k in train_keys:
            choice = np.random.random()
            if choice <= 0.5:
                train_out["normal_fwd"].append(Example(
                    id=f"{ds_name}::normal::fwd::{k}",
                    messages=mk_messages(build_forward_prompt(anim_caps[k], False), anim_json[k]),
                    metadata={"dataset": ds_name, "type": "normal_fwd", "key": k}
                ))
            if choice <= 0.8 and choice > 0.45:
                train_out["normal_rev"].append(Example(
                    id=f"{ds_name}::normal::rev::{k}",
                    messages=mk_messages(build_reverse_prompt(anim_json[k], False), anim_caps[k]),
                    metadata={"dataset": ds_name, "type": "normal_rev", "key": k}
                ))
            if choice >= 0.75:
                if random.random() <= 1:
                    masked_json, removed_layers = remove_random_layers_from_string(anim_json[k])
                    if not removed_layers:
                        train_out["normal_fwd"].append(Example(
                            id=f"{ds_name}::normal::fwd::{k}",
                            messages=mk_messages(build_forward_prompt(anim_caps[k], False), anim_json[k]),
                            metadata={"dataset": ds_name, "type": "normal_fwd", "key": k}
                        ))
                        continue
                    train_out["normal_masked"].append(Example(
                        id=f"{ds_name}::normal_masked::fwd::{k}",
                        messages=mk_messages(build_lottie_layer_masked_prompt(masked_json, anim_caps[k]), "\n".join([f"Layer {i}: {l}" for i, l in enumerate(removed_layers)])),
                        metadata={"dataset": ds_name, "type": "normal_masked", "key": k}
                    ))

    # ---- static ----
    if with_static:
        for k in static_keys:
            choice = np.random.random()
            if choice <= 0.5:
                train_out["static_fwd"].append(Example(
                    id=f"{ds_name}::static::fwd::{k}",
                    messages=mk_messages(build_forward_prompt(st_caps[k], True), st_json[k]),
                    metadata={"dataset": ds_name, "type": "static_fwd", "key": k}
                ))
            if choice <= 0.8 and choice > 0.45:
                train_out["static_rev"].append(Example(
                    id=f"{ds_name}::static::rev::{k}",
                    messages=mk_messages(build_reverse_prompt(st_json[k], True), st_caps[k]),
                    metadata={"dataset": ds_name, "type": "static_rev", "key": k}
                ))
            if choice >= 0.75:
                if random.random() <= 1:
                    masked_json, removed_layers = remove_random_layers_from_string(st_json[k])
                    if not removed_layers:
                        train_out["static_fwd"].append(Example(
                            id=f"{ds_name}::static::fwd::{k}",
                            messages=mk_messages(build_forward_prompt(st_caps[k], True), st_json[k]),
                            metadata={"dataset": ds_name, "type": "static_fwd", "key": k}
                        ))
                        continue
                    train_out["static_masked"].append(Example(
                        id=f"{ds_name}::static_masked::fwd::{k}",
                        messages=mk_messages(build_lottie_layer_masked_prompt(masked_json, st_caps[k]), "\n".join([f"Layer {i}: {l}" for i, l in enumerate(removed_layers)])),
                        metadata={"dataset": ds_name, "type": "static_masked", "key": k}
                    ))
        # if "SVG2Lottie" not in ds_name:
        #     for k in triple_keys:
        #         if np.random.random() < 0.3:
        #             train_out["static_augment"].append(Example(
        #                 id=f"{ds_name}::static::augment::{k}",
        #                 messages=mk_messages(
        #                     build_static_augment_prompt(st_json[k], anim_caps[k]),
        #                     anim_json[k]
        #                 ),
        #                 metadata={"dataset": ds_name, "type": "static_augment", "key": k}
        #             ))

    for k in val_keys:
        if "SVG2Lottie" not in ds_name:
            val_out["normal_fwd"].append(Example(
                id=f"{ds_name}::normal::fwd::{k}",
                messages=mk_messages(build_forward_prompt(anim_caps[k], False), anim_json[k]),
                metadata={"dataset": ds_name, "type": "normal_fwd", "key": k}
            ))
    if with_static:
        for k in val_keys:
            val_out["static_fwd"].append(Example(
                id=f"{ds_name}::static::fwd::{k}",
                messages=mk_messages(build_forward_prompt(st_caps[k], True), st_json[k]),
                metadata={"dataset": ds_name, "type": "static_fwd", "key": k}
            ))

    for k in test_keys:
        if "SVG2Lottie" not in ds_name:
            test_out["normal_fwd"].append(Example(
                id=f"{ds_name}::normal::fwd::{k}",
                messages=mk_messages(build_forward_prompt(anim_caps[k], False), anim_json[k]),
                metadata={"dataset": ds_name, "type": "normal_fwd", "key": k}
            ))
        if with_static:
            test_out["static_fwd"].append(Example(
                id=f"{ds_name}::static::fwd::{k}",
                messages=mk_messages(build_forward_prompt(st_caps[k], True), st_json[k]),
                metadata={"dataset": ds_name, "type": "static_fwd", "key": k}
            ))
    for k, v in train_out.items():
        print(k, len(v))
    for k, v in val_out.items():
        print(k, len(v))
    for k, v in test_out.items():
        print(k, len(v))
    
    # only return some sample copy of out
    # if "scraped_data" in ds_name:
    #     return out
    # small_out = {}
    # sampling_prop = 0.3 if not "generated_data" in ds_name else 0.1
    # for k, v in out.items():
    #     temp = np.random.choice(v, size=int(len(v) * sampling_prop), replace=False)
    #     small_out[k] = temp
    #     print("new:", k, len(temp))
    return train_out, val_out, test_out


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
                "messages": ex.messages
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

        # DOING THIS WILL ONLY CREATE DATASET WITH SCRAPED_DATA
        # if "scraped_data" not in ds_name:
        #     print("Skipping: ", ds_name)
        #     continue

        print(f"[INFO] Processing {ds_name}")
        dataset_examples[ds_name] = process_dataset(ds_name, ds_path, ds_name in WITH_STATIC)

    # # ---------- STRATIFIED SPLIT: ONLY NORMAL_FWD ----------
    # nfwd_train, val, test = split_normal_fwd(dataset_examples)

    # dataset_examples 
    # keys -> gen_data, scraped_data
    # val -> [train, val, test]
    # for ex in val: ex.metadata["split"] = "val"
    # for ex in test: ex.metadata["split"] = "test"

    # ---------- BUILD TRAIN ----------
    train = []
    val = []
    test = []

    for _, (batch_train, batch_val, batch_test) in dataset_examples.items():
        for _, v in batch_train.items():
            train.extend(v)
        for _, v in batch_val.items():
            val.extend(v)
        for _, v in batch_test.items():
            test.extend(v)

    # # include normal_fwd parts destined for train
    # nfwd_train_ids = {e.id for e in nfwd_train}

    # for ds, parts in dataset_examples.items():
    #     # normal_fwd: keep only those not in val/test
    #     for ex in parts["normal_fwd"]:
    #         if ex.id in nfwd_train_ids:
    #             ex.metadata["split"] = "train"
    #             train.append(ex)

    #     # add all other training-only types
    #     for typ in ["normal_rev", "static_fwd", "static_rev", "static_augment", "masked"]:
    #         for ex in parts[typ]:
    #             ex.metadata["split"] = "train"
    #             train.append(ex)

    # ---------- WRITE FILES ----------
    write_jsonl("train.jsonl", train)
    write_jsonl("val.jsonl", val)
    write_jsonl("test.jsonl", test)

    print("[DONE]")


if __name__ == "__main__":
    main()
