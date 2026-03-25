#!/usr/bin/env python3
"""
Copy images from origin_test_tour/{tour_id}/{image_id}.jpeg
to test_tour/{random_id}.webp (converted to WebP) with randomly generated unique IDs.

- Each file gets a random alphanumeric ID (letters + digits).
- The mapping (random_id -> original relative path) is saved to a JSON file.
- IDs are guaranteed unique via a set check.
"""

import json
import secrets
import string
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SRC_DIR = Path("/root/dreamNav/pairUAV/origin_test_tour")
DST_DIR = Path("/root/dreamNav/pairUAV/test_tour")
MAPPING_FILE = Path("/root/dreamNav/pairUAV/test_tour_mapping.json")
ID_LEN = 12  # 12 alphanumeric chars → 62^12 ≈ 3.2e21 possible IDs

CHARSET = string.ascii_lowercase + string.digits  # a-z 0-9


def generate_id(length, used_ids):
    while True:
        rid = ''.join(secrets.choice(CHARSET) for _ in range(length))
        if rid not in used_ids:
            return rid


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    jpeg_files = sorted(SRC_DIR.rglob("*.jpeg"))
    print(f"Found {len(jpeg_files)} jpeg files.")

    mapping = {}      # original relative path -> random_id
    used_ids = set()

    for src_path in tqdm(jpeg_files, desc="Converting", unit="img"):
        rel_path = str(src_path.relative_to(SRC_DIR))  # e.g. "0777/image-50.jpeg"

        rid = generate_id(ID_LEN, used_ids)
        used_ids.add(rid)
        mapping[rel_path] = rid

        dst_path = DST_DIR / f"{rid}.webp"
        with Image.open(src_path) as img:
            img.save(dst_path, format="WEBP", quality=80)

    # Save mapping
    output = {
        "total_files": len(mapping),
        "id_length": ID_LEN,
        "charset": CHARSET,
        "mapping": mapping,  # origin_path -> hashed_id
    }
    with open(MAPPING_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(mapping)} files copied to {DST_DIR}")
    print(f"Mapping saved to {MAPPING_FILE}")


if __name__ == "__main__":
    main()
