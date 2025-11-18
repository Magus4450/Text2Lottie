import string
import numpy as np
import os
import json
from tqdm import tqdm

# --- Helper Function (from your prompt) ---

def get_str():
    """Generates a random 6-character string from shuffled letters."""
    all_str = string.ascii_letters
    all_list = list(all_str)
    
    np.random.shuffle(all_list)
    
    l = len(all_list)
    max_start_index = l - 6
    if max_start_index < 0:
        return "".join(all_list)
        
    i = np.random.choice(range(max_start_index + 1)) 
    
    str_ = "".join(all_list[i:i+6])

    return str_

# --- Recursive Function (from your prompt) ---

def change_nm_recursively(data):
    """Recursively changes the values of keys 'nm' and 'mn'."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("nm", "mn"):
                data[key] = get_str()
            # Note: We recurse on the value directly. Python handles the dict/list
            # update by reference, but re-assigning here is safer for complex structures.
            elif isinstance(value, (dict, list)):
                data[key] = change_nm_recursively(value)
        return data
        
    elif isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], (dict, list)):
                data[i] = change_nm_recursively(data[i])
        return data
    else:
        return data

# --- Main Processing Logic ---

def process_dataset(root_dir="dataset_for_masked"):
    """
    Traverses the directory structure, processes JSON files, and saves them in place.
    
    Expected Structure:
    root_dir/
    ├── folder_A/
    │   └── json/
    │       ├── file1.json
    │       └── file2.json
    ├── folder_B/
    │   └── json/
    │       └── file3.json
    └── ...
    """
    
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        return

    print(f"Starting to process JSON files in '{root_dir}'...")
    
    # We use os.walk to efficiently traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # We only care about files directly inside a folder named 'json'
        if os.path.basename(dirpath) == 'static_json':
            print(f"  -> Found JSON directory: {dirpath}")
            
            for filename in tqdm(filenames):
                if filename.endswith('.json'):
                    file_path = os.path.join(dirpath, filename)
                    
                    try:
                        # 1. Read the JSON file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # 2. Apply the recursive function
                        modified_data = change_nm_recursively(data)
                        
                        # 3. Write the modified data back to the same file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            # Use indent=4 for human-readable output, which is generally 
                            # good practice when modifying config/data files in place.
                            json.dump(modified_data, f, indent=4) 
                            
                        # print(f"    - Successfully updated: {filename}")
                        
                    except json.JSONDecodeError:
                        print(f"    - Skipping file: {filename} (Invalid JSON format)")
                    except Exception as e:
                        print(f"    - An error occurred processing {filename}: {e}")

    print("\n✅ Processing complete.")

# --- Run the script ---
if __name__ == "__main__":
    # Ensure you run this script from the parent directory of 'dataset_for_masked'
    process_dataset()