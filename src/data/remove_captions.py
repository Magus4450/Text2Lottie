import os

json_dir = "scraped_data/json"
caption_dir = "scraped_data/caption"

# Collect all base names (without extension) from json folder
json_basenames = {os.path.splitext(f)[0] for f in os.listdir(json_dir) if f.endswith(".json")}

# Loop through caption files and remove unmatched ones
for fname in os.listdir(caption_dir):
    base, _ = os.path.splitext(fname)
    if base not in json_basenames:
        full_path = os.path.join(caption_dir, fname)
        print(f"Removing unmatched caption: {fname}")
        os.remove(full_path)
