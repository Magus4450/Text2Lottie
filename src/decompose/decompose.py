import os
import json
from typing import Any, Dict, List, Union

# If you have your own load_json/save_json utilities, keep using them.
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

Number = Union[int, float]

def _first_static_value(k: Any) -> Any:
    """
    Given a Lottie 'k' value (which may be a number, list of numbers,
    list of keyframe dicts, or a dict), return a static value.

    Prefers the first keyframe's 's' when keyframes are present.
    """
    # Keyframes: list of dicts -> use the first 's'
    if isinstance(k, list) and k and isinstance(k[0], dict):
        return k[0].get("s", 0)

    # Already a numeric or list-of-numbers -> keep as-is
    if isinstance(k, (int, float)):
        return k

    if isinstance(k, list):
        # Could be [x, y] or [x, y, z]; or sometimes [[x,y],[...]] in odd exports.
        # If it's a list of lists and you intended the first vector, take it.
        if k and isinstance(k[0], list):
            return k[0]
        return k

    # Some exporters wrap again as dicts; try common patterns.
    if isinstance(k, dict):
        # AE expressions sometimes leave 'k' nested.
        if "k" in k:
            return _first_static_value(k["k"])
        if "s" in k:
            return k["s"]
        # Fallback: nothing better to do
        return 0

    # Unknown -> zero
    return 0

def _strip_anim_block(block: Dict[str, Any]) -> None:
    """
    Mutate a transform block (e.g., layer.ks or shape.tr):
    - set every animatable prop to static (a:0) and k := first static value
    """
    if not isinstance(block, dict):
        return
    for key, val in list(block.items()):
        # Skip non-animatable markers/names
        if key in ("ty", "nm"):
            continue
        if isinstance(val, dict) and "k" in val:
            # Ensure 'a' exists and is 0
            val["a"] = 0
            val["k"] = _first_static_value(val.get("k"))

def _normalize_types_for_layer_ks(ks: Dict[str, Any]) -> None:
    """
    Ensure correct types for layer transforms:
      - r: number
      - p: [x, y] or [x, y, z]
      - s: [sx, sy, sz]
    """
    if not isinstance(ks, dict):
        return

    # Rotation: must be number
    if "r" in ks and isinstance(ks["r"], dict):
        v = ks["r"].get("k")
        if isinstance(v, list):
            v = v[0] if v else 0
        ks["r"]["k"] = float(v) if isinstance(v, (int, float)) else 0

    # Position: must be list [x,y] or [x,y,z]
    if "p" in ks and isinstance(ks["p"], dict):
        v = ks["p"].get("k")
        if isinstance(v, (int, float)):
            v = [v, v]
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            pass  # good
        elif isinstance(v, list) and v and isinstance(v[0], list):
            v = v[0]
        else:
            v = [0, 0]
        ks["p"]["k"] = v

    # Scale: must be [sx, sy, sz] for layers
    if "s" in ks and isinstance(ks["s"], dict):
        v = ks["s"].get("k")
        if isinstance(v, (int, float)):
            v = [v, v, v]
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            # If only 2 components provided, pad to 3
            if len(v) == 2:
                v = [v[0], v[1], 100]
        else:
            v = [100, 100, 100]
        ks["s"]["k"] = v

def _normalize_types_for_shape_tr(tr: Dict[str, Any]) -> None:
    """
    Ensure correct types for shape group transforms:
      - r: number
      - p: [x, y] (shape transforms are 2D)
      - s: [sx, sy] (two components)
    """
    if not isinstance(tr, dict):
        return

    # Rotation: must be number
    if "r" in tr and isinstance(tr["r"], dict):
        v = tr["r"].get("k")
        if isinstance(v, list):
            v = v[0] if v else 0
        tr["r"]["k"] = float(v) if isinstance(v, (int, float)) else 0

    # Position: must be [x, y]
    if "p" in tr and isinstance(tr["p"], dict):
        v = tr["p"].get("k")
        if isinstance(v, (int, float)):
            v = [v, v]
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            v = v[:2] if len(v) >= 2 else [v[0], 0]
        elif isinstance(v, list) and v and isinstance(v[0], list):
            v = v[0][:2] if v[0] else [0, 0]
        else:
            v = [0, 0]
        tr["p"]["k"] = v

    # Scale: must be [sx, sy]
    if "s" in tr and isinstance(tr["s"], dict):
        v = tr["s"].get("k")
        if isinstance(v, (int, float)):
            v = [v, v]
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            v = v[:2] if len(v) >= 2 else [v[0], v[0]]
        else:
            v = [100, 100]
        tr["s"]["k"] = v

def _process_shape_items(items: List[Dict[str, Any]]) -> None:
    """
    Walk a shape 'it' array, strip animation and normalize transform types.
    Recurse into nested groups.
    """
    for it in items:
        ty = it.get("ty")
        if ty == "gr":
            # Nested group
            inner = it.get("it", [])
            _process_shape_items(inner)
        elif ty == "tr":
            # Strip animation for transform block
            _strip_anim_block(it)
            _normalize_types_for_shape_tr(it)
        else:
            # Other shape items may have their own animatable subprops (e.g., ellipse size/pos)
            # Normalize any anim blocks found at top level of the item.
            for k, v in list(it.items()):
                if isinstance(v, dict) and "k" in v:
                    _strip_anim_block({k: v})  # wrap to reuse
            # No special type normalization needed here.

def remove_animation(lottie: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove animations from a Lottie JSON and fix type mismatches
    (layer vs shape transforms), so players won't choke on arrays vs scalars.
    """
    layers = lottie.get("layers", [])
    for layer in layers:
        ks = layer.get("ks")
        if isinstance(ks, dict):
            _strip_anim_block(ks)
            _normalize_types_for_layer_ks(ks)

        shapes = layer.get("shapes", [])
        _process_shape_items(shapes)

    return lottie


if __name__ == "__main__":
    input_folder = "generated_data/json"
    output_folder = "generated_data/static_json"
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.endswith(".json"):
            continue

        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        try:
            obj = load_json(in_path)
            obj = remove_animation(obj)
            save_json(out_path, obj)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Failed {fname}: {e}")

    print(f"All processed files saved to: {output_folder}")

