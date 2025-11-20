import os
import csv
import json
import argparse
import numpy as np
import torch
import open_clip
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------
# Frame Sampling
# -----------------------------------------------------------

def sample_frames(video_path, num_frames=8):
    frames = list(iio.imiter(video_path, plugin="pyav"))
    total = len(frames)
    if total == 0:
        raise ValueError(f"No frames found in {video_path}")

    idx = np.linspace(0, total - 1, num_frames, dtype=int)
    return [Image.fromarray(frames[i]) for i in idx]


# -----------------------------------------------------------
# CLIP Scoring
# -----------------------------------------------------------

def compute_clip_score(video_path, caption, model, preprocess, tokenizer, device, num_frames=8):
    frames = sample_frames(video_path, num_frames)
    images = torch.stack([preprocess(f) for f in frames]).to(device)

    with torch.no_grad():
        image_feat = model.encode_image(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        image_emb = image_feat.mean(0, keepdim=True)

        text_tok = tokenizer([caption]).to(device)
        text_feat = model.encode_text(text_tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        score = (image_emb @ text_feat.T).item()

    return score


# -----------------------------------------------------------
# JSONL Parsing Helpers
# -----------------------------------------------------------

def parse_id(id_str):
    parts = id_str.split("::")
    if len(parts) < 4:
        raise ValueError(f"Invalid id: {id_str}")

    mode = parts[-3]
    direction = parts[-2]
    body = parts[-1]
    return mode, direction, body


def extract_caption_from_user_msg(user_msg):
    if "\n" in user_msg:
        return user_msg.split("\n", 1)[1].strip()
    return user_msg.strip()


def load_caption_mapping(jsonl_path):
    mapping = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            mode, direction, body = parse_id(entry["id"])
            key = f"{mode}-{direction}-{body}"

            user_msg = next(m["content"] for m in entry["messages"] if m["role"] == "user")
            caption = extract_caption_from_user_msg(user_msg)

            mapping[key] = caption

    return mapping


def match_video_to_caption(video_filename, caption_map):
    base = os.path.splitext(video_filename)[0]

    if base in caption_map:
        return caption_map[base]

    # fallback fuzzy match
    for key in caption_map:
        if key.endswith(base):
            return caption_map[key]

    raise KeyError(f"No caption for video: {video_filename}")


# -----------------------------------------------------------
# CSV Saving
# -----------------------------------------------------------

def save_csv(out_path, rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "raw_score", "clipped_score", "caption"])
        writer.writerows(rows)

    print(f"\nSaved results to: {out_path}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main(videos_dir, jsonl_path, num_frames, csv_out):
    print(f"Loading captions from {jsonl_path} ...")
    caption_map = load_caption_mapping(jsonl_path)
    print(f"Loaded {len(caption_map)} caption entries.\n")

    # load CLIP model
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # list videos
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")]
    print(f"Found {len(video_files)} videos.\n")

    rows = []      # for CSV
    scores = []    # for summary
    matched = []   # track names

    for vf in tqdm(sorted(video_files), desc="Processing videos", ncols=90):
        full_path = os.path.join(videos_dir, vf)
        base = os.path.splitext(vf)[0]

        try:
            caption = match_video_to_caption(vf, caption_map)
        except KeyError:
            tqdm.write(f"[WARN] Missing caption → {vf}")
            continue

        try:
            raw = compute_clip_score(full_path, caption, model, preprocess, tokenizer, device, num_frames)
            clipped = max(raw, 0.0)
            tqdm.write(f"{base}: raw={raw:.4f}, clipped={clipped:.4f}")

            scores.append(clipped)
            matched.append(vf)

            rows.append([vf, f"{raw:.6f}", f"{clipped:.6f}", caption])

        except Exception as e:
            tqdm.write(f"[ERROR] {vf}: {e}")

    # summary
    print("\n=== Summary ===")
    for vf, s in zip(matched, scores):
        print(f"{os.path.splitext(vf)[0]}: {s:.4f}")

    avg_score = np.mean(scores) if scores else 0.0
    print(f"\nAverage clipped CLIP score: {avg_score:.4f}")

    # save CSV
    if csv_out is not None:
        save_csv(csv_out, rows)


# -----------------------------------------------------------
# Argparse Entry Point
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CLIP score for videos using captions from JSONL.")
    parser.add_argument("--videos", type=str, required=True, help="Folder containing .mp4 videos")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to captions JSONL file")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to sample per video")
    parser.add_argument("--out", type=str, default=None, help="Path to save CSV results")
    args = parser.parse_args()

    main(args.videos, args.jsonl, args.frames, args.out)
