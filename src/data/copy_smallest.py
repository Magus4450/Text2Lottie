import os
import shutil
import glob

def copy_smallest_files(n):
    # Define the directories and N
    SOURCE_DIR = "dataset_for_masked/SVG2Lottie/static_json_ori"
    DEST_DIR = "dataset_for_masked/SVG2Lottie/static_json_small"
    N_FILES = n

    # 1. Create the destination directory if it doesn't exist
    os.makedirs(DEST_DIR, exist_ok=True)

    # 2. Get all JSON files
    search_path = os.path.join(SOURCE_DIR, "*.json")
    all_files = glob.glob(search_path)

    if not all_files:
        print(f"⚠️ No JSON files found in '{SOURCE_DIR}'.")
        return

    # 3. Get the size of each file and store as (size, filepath)
    file_sizes = []
    for filepath in all_files:
        size = os.path.getsize(filepath)
        file_sizes.append((size, filepath))

    # 4. Sort the list by size (the first element of the tuple).
    # Python's sort is ascending by default, which puts the smallest files first.
    file_sizes.sort(key=lambda x: x[0])

    # 5. Select the smallest N files
    smallest_files = file_sizes[:N_FILES]

    # 6. Copy the selected files
    print(f"Attempting to copy the {len(smallest_files)} smallest files...")
    for size, src_path in smallest_files:
        # Get just the file name to create the destination path
        filename = os.path.basename(src_path)
        dest_path = os.path.join(DEST_DIR, filename)

        # Copy the file
        shutil.copy2(src_path, dest_path)
        print(f"  Copied: {filename} ({size} bytes)")

    print(f"\n✅ Finished copying {len(smallest_files)} files to '{DEST_DIR}'.")

# --- EXECUTION ---
# Change 10 to the desired number (n) of files you want to copy.
copy_smallest_files(n=2081)