import json
import os
import glob
from tqdm import tqdm 

SRC_DIR = "/root/dreamNav/pairUAV/gt_test"
DST_DIR = "/root/dreamNav/pairUAV/test"
TXT_PATH = "/root/dreamNav/pairUAV/test_gt.txt"

# Collect all json files with their (group_id, json_id) for sorting
entries = []
for json_path in sorted(glob.glob(os.path.join(SRC_DIR, "*", "*.json"))):
    group_id = os.path.basename(os.path.dirname(json_path))
    json_id = os.path.basename(json_path)
    entries.append((group_id, json_id, json_path))

# Sort by group_id then json_id (both are zero-padded strings, lexicographic == numeric)
entries.sort(key=lambda x: (x[0], x[1]))

txt_lines = []
count = 0

for group_id, json_id, json_path in tqdm(entries):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract heading_num and range_num for txt
    heading_num = data["heading_num"]
    range_num = data["range_num"]
    txt_lines.append(f"{heading_num}, {range_num}")

    # Keep only image_a and image_b
    filtered = {"image_a": data["image_a"], "image_b": data["image_b"]}

    # Write to destination
    dst_group_dir = os.path.join(DST_DIR, group_id)
    os.makedirs(dst_group_dir, exist_ok=True)
    dst_path = os.path.join(dst_group_dir, json_id)
    with open(dst_path, "w") as f:
        json.dump(filtered, f)

    count += 1

# Write txt file
with open(TXT_PATH, "w") as f:
    f.write("\n".join(txt_lines) + "\n")

print(f"Processed {count} JSON files.")
print(f"Filtered JSONs written to {DST_DIR}")
print(f"heading_num/range_num written to {TXT_PATH}")
