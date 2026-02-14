"""On-the-fly PairUAV dataset that reads image pairs directly from JSON + image files.

Eliminates the need for pre-conversion to LeRobot format. Each sample is a pair
(image_a, image_b) with heading and range labels.
"""

import json
import logging
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from openpi.shared.normalize import NormStats, RunningStats

MAX_HEADING = 180.0
MIN_HEADING = -180.0
MAX_RANGE = 132.0
MIN_RANGE = -132.0


class PairUAVDataset:
    """Dataset that generates (image_a, image_b) → (heading, range) samples on the fly.

    Instead of pre-converting all N² pairs to LeRobot format (slow, disk-heavy),
    this dataset reads directly from the raw JSON metadata and tour image files.

    Args:
        data_dir: Directory containing tour subdirectories, each with JSON pair files.
        tour_dir: Directory containing tour image subdirectories.
        action_horizon: Number of action steps per sample (should be 1 for this task).
    """

    def __init__(self, data_dir: str, tour_dir: str, action_horizon: int = 1):
        self.tour_dir = tour_dir
        self.action_horizon = action_horizon
        self._state = np.zeros(8, dtype=np.float32)

        # Build index: list of (json_path,) for all pair files
        self._samples: list[str] = []
        tour_ids = sorted(os.listdir(data_dir))
        for tour_id in tqdm(tour_ids, desc="Indexing PairUAV tours"):
            tour_path = os.path.join(data_dir, tour_id)
            if not os.path.isdir(tour_path):
                continue
            for json_file in sorted(os.listdir(tour_path)):
                if json_file.endswith(".json"):
                    self._samples.append(os.path.join(tour_path, json_file))

        logging.info(f"PairUAVDataset: indexed {len(self._samples)} samples from {len(tour_ids)} tours")

        # Image cache: rel_path -> PIL Image (stays in CPU memory)
        self._image_cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _get_image(self, rel_path: str) -> Image.Image:
        if rel_path not in self._image_cache:
            full_path = os.path.join(self.tour_dir, rel_path)
            with Image.open(full_path) as im:
                self._image_cache[rel_path] = im.convert("RGB")
        return self._image_cache[rel_path]

    def __getitem__(self, index) -> dict:
        with open(self._samples[index], "r") as f:
            data = json.load(f)

        heading = (data["heading_num"] - MIN_HEADING) / (MAX_HEADING - MIN_HEADING)
        range_val = (data["range_num"] - MIN_RANGE) / (MAX_RANGE - MIN_RANGE)

        image_a = self._get_image(data["image_a"])
        image_b = self._get_image(data["image_b"])

        # actions shape: (action_horizon, 2)
        actions = np.tile(
            np.array([heading, range_val], dtype=np.float32),
            (self.action_horizon, 1),
        )

        return {
            "image": image_a,
            "wrist_image": image_b,
            "state": self._state.copy(),
            "actions": actions,
            "prompt": "Move from image viewpoint to wrist_image viewpoint",
        }


def compute_pairuav_norm_stats(
    data_dir: str, max_samples: int | None = None
) -> dict[str, NormStats]:
    """Compute normalization stats from raw JSON metadata only (no image loading).

    This is extremely fast compared to the LeRobot-based approach since it only
    reads small JSON files.
    """
    state_stats = RunningStats()
    action_stats = RunningStats()
    dummy_state = np.zeros((1, 8), dtype=np.float32)

    count = 0
    tour_ids = sorted(os.listdir(data_dir))
    for tour_id in tqdm(tour_ids, desc="Computing norm stats"):
        tour_path = os.path.join(data_dir, tour_id)
        if not os.path.isdir(tour_path):
            continue
        for json_file in sorted(os.listdir(tour_path)):
            if not json_file.endswith(".json"):
                continue
            with open(os.path.join(tour_path, json_file), "r") as f:
                data = json.load(f)

            heading = (data["heading_num"] - MIN_HEADING) / (MAX_HEADING - MIN_HEADING)
            range_val = (data["range_num"] - MIN_RANGE) / (MAX_RANGE - MIN_RANGE)

            state_stats.update(dummy_state)
            action_stats.update(np.array([[heading, range_val]], dtype=np.float32))

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        if max_samples is not None and count >= max_samples:
            break

    logging.info(f"Computed norm stats from {count} samples")
    return {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
