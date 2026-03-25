import os
import json
import glob

SRC_DIR = "/root/dreamNav/pairUAV/origin_train"
DST_DIR = "/root/dreamNav/pairUAV/train"

for json_path in glob.glob(os.path.join(SRC_DIR, "*", "*.json")):
    with open(json_path, "r") as f:
        data = json.load(f)

    for key in ("image_a", "image_b"):
        if key in data and data[key].endswith(".jpeg"):
            data[key] = data[key][:-5] + ".webp"

    rel = os.path.relpath(json_path, SRC_DIR)
    dst_path = os.path.join(DST_DIR, rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(dst_path, "w") as f:
        json.dump(data, f)

print("Done")
