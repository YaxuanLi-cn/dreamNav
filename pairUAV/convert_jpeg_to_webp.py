import os
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_DIR = Path("/root/dreamNav/pairUAV/origin_train_tour")
DST_DIR = Path("/root/dreamNav/pairUAV/train_tour")


def convert_one(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img.save(dst_path, "webp", quality=80)
    return str(src_path)


def main():
    tasks = []
    for jpeg_file in sorted(SRC_DIR.rglob("*.jpeg")):
        rel = jpeg_file.relative_to(SRC_DIR)
        dst = DST_DIR / rel.with_suffix(".webp")
        if dst.exists():
            continue
        tasks.append((jpeg_file, dst))

    print(f"Total files to convert: {len(tasks)}")

    done = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(convert_one, s, d): s for s, d in tasks}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0 or done == len(tasks):
                print(f"Progress: {done}/{len(tasks)}")

    print("Done.")


if __name__ == "__main__":
    main()
