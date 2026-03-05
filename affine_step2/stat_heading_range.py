import json
import os
from collections import Counter

dirs = [
    "/root/dreamNav/pairUAV/try_train",
    "/root/dreamNav/pairUAV/try_test",
]

heading_nums = Counter()
range_nums = Counter()

total_files = 0

for d in dirs:
    for scene in sorted(os.listdir(d)):
        scene_dir = os.path.join(d, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fname in sorted(os.listdir(scene_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(scene_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            heading_nums[round(data["heading_num"], 2)] += 1
            range_nums[round(data["range_num"], 2)] += 1
            total_files += 1

heading_path = "/root/dreamNav/pairUAV/heading_num.txt"
range_path = "/root/dreamNav/pairUAV/range_num.txt"

with open(heading_path, "w") as out:
    for val in sorted(heading_nums.keys()):
        out.write(f"{val}\n")

with open(range_path, "w") as out:
    for val in sorted(range_nums.keys()):
        out.write(f"{val}\n")

print(f"Results saved to {heading_path} and {range_path}")
