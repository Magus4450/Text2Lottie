# lottie_semantic_tokenizer.py
import json, re
from typing import Any, Dict, List, Optional

# =============================================================================
# 0) COMMON VALUE/PATTERN TOKENS (for vocab only; we don't rewrite values)
# =============================================================================
# These strings are added to the tokenizer's vocab so they tend to become
# single tokens during training/inference. We do not perform replacements for
# them here—this module only replaces *keys* inside fenced JSON blocks.
COMMON_VALUES: Dict[str, str] = {
    # Property animation flags (recommended)
    '"a":0': '<prop_not_animated>',
    '"a":1': '<prop_animated>',
    '"k":': '<prop_k>',
    '"ix":': '<prop_ix>',

    # Shape type patterns (recommended)
    '"ty":"gr"': '<shape_ty_gr>',
    '"ty":"sh"': '<shape_ty_sh>',
    '"ty":"fl"': '<shape_ty_fl>',
    '"ty":"st"': '<shape_ty_st>',
    '"ty":"tr"': '<shape_ty_tr>',
    '"ty":"tm"': '<shape_ty_tm>',

    # Frequent layer/type literals
    '"ty":4': '<layer_ty_shape>',

    # A few very common scalars/booleans
    '0': '<0>',
    '1': '<1>',
    '100': '<100>',
    'true': '<true>',
    'false': '<false>',
}

# =============================================================================
# 1) TAG TABLES (Lottie keys → semantic tags) per your 48-key reference
# =============================================================================

# --- ROOT LEVEL KEYS (11) ---
ROOT_KEYS = {
    "v": "<version>",            # e.g., "5.6.5"
    "fr": "<frame_rate>",        # frames per second
    "ip": "<in_point>",          # start frame
    "op": "<out_point>",         # end frame
    "w": "<width>",              # canvas width
    "h": "<height>",             # canvas height
    "nm": "<name>",              # animation name
    "ddd": "<is_3d>",            # 0/1
    "assets": "<assets>",        # asset array
    "layers": "<layers>",        # main layer array
    "markers": "<markers>",      # timeline markers
}

# --- LAYER KEYS (subset from your list) ---
LAYER_KEYS = {
    "ind": "<layer_index>",
    "ty": "<layer_type>",          # 0..13 (see LAYER_TYPES)
    "nm": "<layer_name>",
    "sr": "<time_stretch>",
    "ks": "<transform>",
    "ao": "<auto_orient>",         # 0/1
    "ip": "<layer_in_point>",
    "op": "<layer_out_point>",
    "st": "<layer_start_time>",
    "bm": "<blend_mode>",
    "ddd": "<layer_is_3d>",
    "parent": "<parent_layer>",
    "shapes": "<shapes>",
    "ct": "<collapse_transform>",  # 0/1
}

# --- LAYER TYPES (value legend; not used for tagging directly) ---
LAYER_TYPES = {
    0: "Precomp",
    1: "Solid",
    2: "Image",
    3: "Null",
    4: "Shape",
    5: "Text",
    6: "Audio",
    7: "Video Placeholder",
    8: "Image Sequence",
    9: "Video",
    10: "Image Placeholder",
    11: "Guide",
    12: "Adjustment",
    13: "Camera",
}

# --- TRANSFORM (ks) ---
TRANSFORM_KEYS = {
    "a": "<anchor>",
    "p": "<position>",
    "s": "<scale>",
    "r": "<rotation>",
    "o": "<opacity>",
    "sk": "<skew>",
    "sa": "<skew_axis>",
}

# --- ANIMATED PROPERTY WRAPPER (for a/p/s/r/o/sk/sa and some shape props) ---
PROPERTY_KEYS = {
    "a": "<animated_flag>",
    "k": "<key_or_value>",
    "ix": "<prop_index>",
}

# --- KEYFRAME OBJECT (when a=1 → k is an array of these) ---
KEYFRAME_KEYS = {
    "t": "<kf_time>",
    "s": "<kf_start>",
    "e": "<kf_end>",
    "i": "<kf_in_tangent>",
    "o": "<kf_out_tangent>",
    "ti": "<kf_to_tangent_in>",
    "to": "<kf_to_tangent_out>",
    "l": "<kf_spatial_length>",  # spatial length (when present)
    # Note: we intentionally omit "h" (hold) to match your 48-key list
}

# --- EASING SUBOBJECT (inside keyframe "i"/"o") ---
EASING_KEYS = {
    "x": "<ease_x>",
    "y": "<ease_y>",
}

# --- SHAPE / ITEM KEYS (group container + many modifiers, strokes, fills, etc.) ---
SHAPE_KEYS = {
    "ty": "<shape_type>",
    "nm": "<shape_name>",
    "mn": "<match_name>",
    "it": "<shape_items>",
    "np": "<shape_prop_count>",
    "cix": "<shape_class_index>",
    "bm": "<shape_blend_mode>",
    "ix": "<shape_prop_index>",
    "hd": "<hidden>",
    "ind": "<shape_index>",
    "m": "<merge_mode>",
    "e": "<end>",                 # often trim end
    "s": "<start>",               # often trim start
    "o": "<offset>",              # often trim offset
    "ks": "<shape_transform>",    # transform for 'tr' item
    "lc": "<line_cap>",
    "lj": "<line_join>",
    "ml": "<miter_limit>",
    "w": "<stroke_width>",
    "c": "<fs_color>",            # fill/stroke color
    "p": "<path_or_position>",    # path/position (contextual)
    "r": "<roundness>",           # or rotation in other contexts
    "sa": "<start_angle>",
    "sk": "<shape_skew>",
}

# --- SHAPE TYPES (value of "ty") ---
SHAPE_TYPES = {
    "gr": "<shape_group>",
    "rc": "<rect>",
    "el": "<ellipse>",
    "sr": "<star>",
    "sh": "<path>",
    "fl": "<fill>",
    "st": "<stroke>",
    "gf": "<grad_fill>",
    "gs": "<grad_stroke>",
    "tr": "<shape_tr>",   # shape transform item
    "tm": "<trim>",
    "rd": "<round_corners>",
    "pb": "<pucker_bloat>",
    "mm": "<merge>",
    "rp": "<repeater>",
}

# --- PATH KEYS (for "sh" type) ---
PATH_KEYS = {
    "i": "<path_in_tangent_pts>",
    "o": "<path_out_tangent_pts>",
    "v": "<path_vertices>",
    "c": "<path_closed>",
}

# =============================================================================
# 2) CONTEXT TRACKER
# =============================================================================

class Ctx:
    def __init__(
        self,
        where: str = "root",            # 'root' | 'layers' | 'assets' | 'asset_comp' | 'shape'
        layer: bool = False,            # True when iterating a concrete layer object
        in_ks: bool = False,            # inside transform ('ks')
        in_tm: bool = False,            # inside trim paths ('tm') if needed
        in_keyframe: bool = False,      # inside a keyframe obj (has 't' or 's'+'e', etc.)
        shape_type: Optional[str] = None,  # 'sh','fl','st','gf','gs','tm', etc.
        in_fillstroke: bool = False,    # inside fill/stroke item (fl/st)
        in_path: bool = False,          # inside a path object (ty='sh')
    ):
        self.where = where
        self.layer = layer
        self.in_ks = in_ks
        self.in_tm = in_tm
        self.in_keyframe = in_keyframe
        self.shape_type = shape_type
        self.in_fillstroke = in_fillstroke
        self.in_path = in_path

    def child(self, **updates) -> "Ctx":
        c = Ctx(**self.__dict__)
        for k, v in updates.items():
            setattr(c, k, v)
        return c

# =============================================================================
# 3) KEY → TAG (context-aware)
# =============================================================================

def key_to_tag(key: str, ctx: Ctx) -> Optional[str]:
    # Root-like keys (also used by comps inside assets)
    if ctx.where in ("root", "asset_comp"):
        if key in ROOT_KEYS:
            return ROOT_KEYS[key]

    # Layer object
    if ctx.layer and key in LAYER_KEYS:
        return LAYER_KEYS[key]

    # Transform (ks)
    if ctx.in_ks and key in TRANSFORM_KEYS:
        return TRANSFORM_KEYS[key]

    # Keyframe dict + easing/space helpers
    if ctx.in_keyframe:
        if key in KEYFRAME_KEYS:
            return KEYFRAME_KEYS[key]
        if key in EASING_KEYS:
            return EASING_KEYS[key]

    # Shape container
    if ctx.where == "shape":
        # Shape/Item keys
        if key in SHAPE_KEYS:
            return SHAPE_KEYS[key]
        # Path block
        if ctx.in_path and key in PATH_KEYS:
            return PATH_KEYS[key]

    # Property wrapper (appears widely as children of transform or shape props)
    if key in PROPERTY_KEYS:
        return PROPERTY_KEYS[key]

    # Fallback: leave key as-is (no tag)
    return None

# =============================================================================
# 4) ENCODER: JSON → tagged text
# =============================================================================

def _is_keyframe_obj(d: Dict[str, Any]) -> bool:
    # Heuristic: keyframe-like if has 't' or both 's' and 'e' (and often i/o/ti/to)
    return isinstance(d, dict) and ("t" in d or ("s" in d and "e" in d))

def _detect_shape_type(d: Dict[str, Any], parent_shape_type: Optional[str]) -> Optional[str]:
    # Prefer explicit 'ty' if recognized; else inherit
    ty = d.get("ty")
    if isinstance(ty, str) and ty in SHAPE_TYPES:
        return ty
    return parent_shape_type

def _serialize_val(v: Any) -> str:
    # Standard JSON for primitives/values
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

def encode_obj(obj: Any, ctx: Ctx) -> str:
    if isinstance(obj, dict):
        # Derive context flags for this object
        shape_type_here = _detect_shape_type(obj, ctx.shape_type)
        in_keyframe = _is_keyframe_obj(obj) or ctx.in_keyframe
        in_path = ctx.in_path or (shape_type_here == "sh")
        in_fillstroke = ctx.in_fillstroke or (shape_type_here in ("fl", "st"))

        ctx_here = ctx.child(
            shape_type=shape_type_here,
            in_keyframe=in_keyframe,
            in_path=in_path,
            in_fillstroke=in_fillstroke,
        )

        parts: List[str] = ["{"]
        first = True
        for k, v in obj.items():
            # Update nested container flags
            c2 = ctx_here
            if k == "ks":
                c2 = c2.child(in_ks=True)
            if k == "tm":
                c2 = c2.child(in_tm=True)
            if k == "it":
                c2 = c2.child(where="shape")   # shape item array/object
            if k == "layers":
                c2 = c2.child(where="layers")
            if k == "assets":
                c2 = c2.child(where="assets")

            # When we’re inside a layers array, elements are layer objects
            if ctx.where == "layers" and isinstance(v, (dict, list)):
                c2 = c2.child(layer=True)

            # When iterating shapes content, keep shape context
            if ctx.where == "shape" and isinstance(v, (dict, list)):
                c2 = c2.child(where="shape")

            # Tag mapping
            tag = key_to_tag(k, c2)
            key_txt = tag if tag else f'"{k}"'

            if not first:
                parts.append(",")
            parts.append(f"{key_txt}:{encode_obj(v, c2)}")
            first = False

        parts.append("}")
        return "".join(parts)

    elif isinstance(obj, list):
        def child_ctx_for_item(item, ctx_list: Ctx) -> Ctx:
            # Items of 'layers' → concrete layer objects
            if ctx_list.where == "layers" and isinstance(item, dict):
                return ctx_list.child(layer=True)
            # Items of shapes ('it') → remain in shape context
            if ctx_list.where == "shape" and isinstance(item, dict):
                return ctx_list.child(where="shape")
            # Items of 'assets' → comp/media objects with root-like keys
            if ctx_list.where == "assets" and isinstance(item, dict):
                return ctx_list.child(where="asset_comp")
            return ctx_list

        inner = ",".join(encode_obj(x, child_ctx_for_item(x, ctx)) for x in obj)
        return f"[{inner}]"

    # Primitives
    return _serialize_val(obj)

# =============================================================================
# 5) DECODER: tagged text → JSON string (reversible)
# =============================================================================

# Build reverse maps for keys; for ambiguous cases, tags encode context already
REVERSE_TABLE: Dict[str, str] = {}

def _add_reverse(d: Dict[str, str]):
    for k, v in d.items():
        REVERSE_TABLE[v] = k

for table in [
    ROOT_KEYS, LAYER_KEYS, TRANSFORM_KEYS, PROPERTY_KEYS, KEYFRAME_KEYS,
    EASING_KEYS, SHAPE_KEYS, PATH_KEYS
]:
    _add_reverse(table)

# Token pattern (tags) — replace tags with quoted JSON keys
_tag_keys = list(REVERSE_TABLE.keys())
TAG_KEY_RE = re.compile("|".join(map(re.escape, _tag_keys))) if _tag_keys else re.compile(r"$^")

def decode_tags_to_keys(s: str) -> str:
    """Replace <tag> with the corresponding JSON key (quoted)."""
    def _sub(m):
        tag = m.group(0)
        key = REVERSE_TABLE.get(tag)
        return f'"{key}"' if key else tag
    return TAG_KEY_RE.sub(_sub, s)

# =============================================================================
# 6) PUBLIC API (operate only inside fenced ```json / ```lottie code blocks)
# =============================================================================

FENCE_RE = re.compile(r"```(?:json|JSON|lottie)?\s*\n(.*?)```", re.DOTALL)

def _encode_block(block: str) -> str:
    try:
        obj = json.loads(block)
        tagged = encode_obj(obj, Ctx(where="root"))
        return tagged
    except Exception:
        # If it’s not valid JSON, just pass it through unchanged
        return block

def _decode_block(block: str) -> str:
    return decode_tags_to_keys(block)

def to_semantic(text: str) -> str:
    """Within fenced JSON/lottie code blocks, convert keys → semantic tags."""
    def _enc(m):
        inner = m.group(1)
        return "```\n" + _encode_block(inner) + "```"
    return FENCE_RE.sub(_enc, text)

def from_semantic(text: str) -> str:
    """Within fenced JSON/lottie code blocks, convert semantic tags → raw keys."""
    def _dec(m):
        inner = m.group(1)
        return "```\n" + _decode_block(inner) + "```"
    return FENCE_RE.sub(_dec, text)

# =============================================================================
# 7) HF WRAPPER
# =============================================================================

class LottieSemanticTokenizer:
    """
    Wraps a base HF tokenizer. Adds semantic tags (the <...> tokens used for keys)
    and some optional common value/pattern tokens to encourage single-token chunks
    for frequent constructs in Lottie JSON.

    Usage:
        base_tok = AutoTokenizer.from_pretrained(...)
        lottie_tok = LottieSemanticTokenizer(base_tok)
        model.resize_token_embeddings(len(base_tok))  # after adding tokens

        # In your dataset mapping (after apply_chat_template):
        text = lottie_tok.encode_preprocess(text)

        # After generation:
        text = lottie_tok.decode_postprocess(text)
    """
    def __init__(self, base_tokenizer, add_as_special_tokens: bool = False):
        self.base = base_tokenizer

        # Collect all key tags + recommended value/pattern tokens
        tag_tokens = set(REVERSE_TABLE.keys())
        value_tokens = set(COMMON_VALUES.values()) | set(COMMON_VALUES.keys())  # keep both literal & alias
        # We add both the *literal* strings (e.g., '"a":0') and their aliases to bias vocab coverage.
        new_tokens = sorted(tag_tokens | value_tokens)

        if add_as_special_tokens:
            self.base.add_special_tokens({"additional_special_tokens": new_tokens})
        else:
            self.base.add_tokens(list(new_tokens), special_tokens=False)

    def __len__(self):
        """Return the total vocab size of the underlying tokenizer."""
        return len(self.base)

    def encode_preprocess(self, text: str) -> str:
        return to_semantic(text)

    def decode_postprocess(self, text: str) -> str:
        return from_semantic(text)

# =============================================================================
# 8) CLI (demo)
# =============================================================================

if __name__ == "__main__":
    # 1) Load JSON from disk
    with open("samples/gen_anim.json", "r", encoding="utf-8") as f:
        raw_json = f.read()

    # 2) Wrap in a json code-fence (the converter operates inside fences)
    text = f"```json\n{raw_json}\n```"

    # 3) Convert keys to semantic tags
    semantic_text = to_semantic(text)
    print("=== SEMANTIC REPRESENTATION ===")
    print(semantic_text)

    # 4) Convert back to normal Lottie JSON
    restored_text = from_semantic(semantic_text)
    print("\n=== RESTORED JSON ===")
    print(restored_text)
