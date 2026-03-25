import os
import json
import random
from tqdm import tqdm

BASE_DIR = "/root/dreamNav/pairUAV/origin_test"
MAPPING_FILE = "/root/dreamNav/pairUAV/test_tour_mapping.json"
OUTPUT_DIR = "/root/dreamNav/pairUAV/gt_test"
GROUP_SIZE = 2000

# Load tour mapping
print("Loading test_tour_mapping.json...")
with open(MAPPING_FILE, "r") as f:
    tour_mapping_data = json.load(f)
image_mapping = tour_mapping_data["mapping"]

# Collect all JSON files from subdirectories (depth >= 2)
print("Collecting JSON files from subdirectories...")
json_files = []
for tour_id in sorted(os.listdir(BASE_DIR)):
    tour_path = os.path.join(BASE_DIR, tour_id)
    if not os.path.isdir(tour_path):
        continue
    for fname in os.listdir(tour_path):
        if fname.endswith(".json"):
            json_files.append(os.path.join(tour_path, fname))

total = len(json_files)
print(f"Found {total} JSON files")

# Determine zero-padding width for json IDs within each group
num_digits_id = len(str(GROUP_SIZE - 1))
# Determine zero-padding width for group IDs
num_groups = (total + GROUP_SIZE - 1) // GROUP_SIZE
num_digits_group = len(str(num_groups - 1))
print(f"Using {num_digits_group} digits for group IDs, {num_digits_id} digits for json IDs")
print(f"Total groups: {num_groups}")

# Create shuffled index list
indices = list(range(total))
random.shuffle(indices)

# Pre-create group directories
for g in range(num_groups):
    group_dir = os.path.join(OUTPUT_DIR, str(g).zfill(num_digits_group))
    os.makedirs(group_dir, exist_ok=True)

# Process each file
for i, src_path in enumerate(tqdm(json_files, desc="Processing")):
    with open(src_path, "r") as f:
        data = json.load(f)

    # Update image_a and image_b using the mapping (error if not found)
    for key in ("image_a", "image_b"):
        old_path = data[key]
        if old_path not in image_mapping:
            raise KeyError(f"Mapping not found for {key}='{old_path}' in file {src_path}")
        data[key] = image_mapping[old_path] + '.webp'

    # Compute group_id and in-group id from shuffled index
    shuffled_idx = indices[i]
    group_id = shuffled_idx // GROUP_SIZE
    in_group_id = shuffled_idx % GROUP_SIZE

    group_dir = os.path.join(OUTPUT_DIR, str(group_id).zfill(num_digits_group))
    new_id = str(in_group_id).zfill(num_digits_id)
    dst_path = os.path.join(group_dir, f"{new_id}.json")
    with open(dst_path, "w") as f:
        json.dump(data, f)

print(f"Done! Wrote {total} files into {num_groups} groups under {OUTPUT_DIR}/")
