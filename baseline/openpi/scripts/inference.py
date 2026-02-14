"""Inference script for PairUAV on the OpenPI pi05_ours model.

Loads a trained checkpoint, runs **batched** inference on the test set, and
computes:
  1. Heading MAE (degrees)
  2. Range MAE (meters)
  3. Success rate (predicted position within 10 m of ground-truth position)

Usage:
    # Auto-detect latest orbax checkpoint:
    uv run scripts/inference.py --checkpoint-dir checkpoints/pi05_ours

    # Specify a specific step checkpoint:
    uv run scripts/inference.py --checkpoint-dir checkpoints/pi05_ours/363

    # Custom test/tour dirs & batch size:
    uv run scripts/inference.py \
        --checkpoint-dir checkpoints/pi05_ours \
        --test-dir /root/dreamNav/pairUAV/try_test \
        --tour-dir /root/dreamNav/pairUAV/tours \
        --batch-size 64
"""

import argparse
import json
import logging
import math
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants matching pairuav_dataset.py
# ---------------------------------------------------------------------------
MAX_HEADING = 180.0
MIN_HEADING = -180.0
MAX_RANGE = 132.0
MIN_RANGE = -132.0

SUCCESS_THRESHOLD_M = 10.0  # metres


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def heading_range_to_xy(heading_deg: float, range_m: float):
    """Convert (heading_degrees, range_metres) to an (x, y) displacement."""
    heading_rad = math.radians(heading_deg)
    x = range_m * math.sin(heading_rad)
    y = range_m * math.cos(heading_rad)
    return x, y


def denormalize(norm_heading: float, norm_range: float):
    """Reverse the [0, 1] normalisation applied in PairUAVDataset."""
    heading_deg = norm_heading * (MAX_HEADING - MIN_HEADING) + MIN_HEADING
    range_m = norm_range * (MAX_RANGE - MIN_RANGE) + MIN_RANGE
    return heading_deg, range_m


def resolve_checkpoint_dir(checkpoint_dir: str) -> pathlib.Path:
    """If *checkpoint_dir* is the top-level orbax dir (contains numbered step
    sub-dirs), pick the latest step.  Otherwise return as-is."""
    p = pathlib.Path(checkpoint_dir).resolve()
    # Check if this dir itself contains a 'params' sub-dir (step-level dir).
    if (p / "params").exists():
        return p
    # Otherwise look for numbered sub-dirs and pick the latest.
    step_dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if step_dirs:
        latest = step_dirs[-1]
        logging.info(f"Auto-selected latest checkpoint step: {latest.name}")
        return latest
    # Fallback – return as-is and let downstream code report the error.
    return p


# ---------------------------------------------------------------------------
# Build policy — avoids the expensive PairUAVDataConfig.create() which
# recomputes norm stats from the full training set when the cache tag
# doesn't match.  Instead we load norm stats from the existing cache and
# construct the transform pipeline directly.
# ---------------------------------------------------------------------------

def build_policy(checkpoint_dir: pathlib.Path):
    """Build the inference policy from a trained checkpoint."""
    import openpi.models.model as _model
    import openpi.policies.libero_policy as libero_policy
    import openpi.policies.policy as _policy
    import openpi.shared.download as download
    import openpi.shared.normalize as _normalize
    import openpi.training.config as _config
    from openpi import transforms as _transforms

    train_config = _config.get_config("pi05_ours")
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))

    # ---- Load norm stats from existing cache (skip recomputation) ----------
    # Training was run with data_dir=try_train, so cache tag is "pairuav_try_train".
    # Fall back to the checkpoint's own assets dir if present.
    norm_stats = None
    assets_dir = train_config.assets_dirs  # e.g. ./assets/pi05_ours
    for candidate in [
        assets_dir / "pairuav_try_train",
        assets_dir / "pairuav_train",
        checkpoint_dir / "assets" / "pairuav",
    ]:
        ns_path = candidate / "norm_stats.json"
        if ns_path.exists():
            norm_stats = _normalize.load(candidate)
            logging.info(f"Loaded cached norm stats from {candidate}")
            break
    if norm_stats is None:
        raise FileNotFoundError(
            f"Could not find cached norm_stats.json under {assets_dir} or "
            f"{checkpoint_dir / 'assets'}. Run training first to generate them."
        )

    # ---- Load model --------------------------------------------------------
    weight_path = checkpoint_dir / "model.safetensors"
    is_pytorch = weight_path.exists()
    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(
            _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        )

    # ---- Build transforms inline (same as PairUAVDataConfig.create) --------
    model_config = train_config.model
    use_quantile_norm = model_config.model_type != _model.ModelType.PI0

    # Note: no "actions" key — not available during inference.
    repack_inputs = [
        _transforms.RepackTransform({
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "prompt": "prompt",
        })
    ]
    data_inputs = [libero_policy.LiberoInputs(model_type=model_config.model_type)]
    data_outputs = [libero_policy.LiberoOutputs()]
    model_transforms = _config.ModelTransformFactory()(model_config)

    # Determine device
    pytorch_device = None
    if is_pytorch:
        try:
            import torch
            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    policy = _policy.Policy(
        model,
        transforms=[
            *repack_inputs,
            _transforms.InjectDefaultPrompt("Move from image viewpoint to wrist_image viewpoint"),
            *data_inputs,
            _transforms.Normalize(norm_stats, use_quantiles=use_quantile_norm),
            *model_transforms.inputs,
        ],
        output_transforms=[
            *model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=use_quantile_norm),
            *data_outputs,
        ],
        sample_kwargs=None,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
    return policy


# ---------------------------------------------------------------------------
# Load one test sample
# ---------------------------------------------------------------------------

def load_sample(json_path: str, tour_dir: str, image_cache: dict):
    """Load a single test sample and return (obs_dict, gt_heading_deg, gt_range_m)."""
    with open(json_path, "r") as f:
        meta = json.load(f)

    gt_heading_deg = meta["heading_num"]
    gt_range_m = meta["range_num"]

    def _get_image(rel_path: str) -> np.ndarray:
        if rel_path not in image_cache:
            full_path = os.path.join(tour_dir, rel_path)
            with Image.open(full_path) as im:
                image_cache[rel_path] = np.array(im.convert("RGB"), dtype=np.uint8)
        return image_cache[rel_path]

    image_a = _get_image(meta["image_a"])
    image_b = _get_image(meta["image_b"])

    obs = {
        "image": image_a,
        "wrist_image": image_b,
        "state": np.zeros(8, dtype=np.float32),
        "prompt": "Move from image viewpoint to wrist_image viewpoint",
    }
    return obs, gt_heading_deg, gt_range_m


# ---------------------------------------------------------------------------
# Batched inference helpers
# ---------------------------------------------------------------------------

def _stack_leaves(list_of_dicts: list[dict]) -> dict:
    """Stack a list of identically-keyed dicts into a single dict with a
    leading batch dimension (numpy arrays)."""
    keys = list_of_dicts[0].keys()
    out = {}
    for k in keys:
        v0 = list_of_dicts[0][k]
        if isinstance(v0, dict):
            out[k] = _stack_leaves([d[k] for d in list_of_dicts])
        else:
            out[k] = np.stack([d[k] for d in list_of_dicts], axis=0)
    return out


def batched_infer(policy, obs_list: list[dict]) -> list[dict]:
    """Run a batch of observations through the policy at once.

    Steps:
      1. Apply per-sample input transforms (image resize, tokenise, normalise …).
      2. Stack transformed dicts into a single batched dict.
      3. Call model.sample_actions **once** for the whole batch.
      4. Split outputs, apply per-sample output transforms.
    """
    from openpi.models import model as _model

    # --- 1. Per-sample input transforms ---
    transformed: list[dict] = []
    for obs in obs_list:
        inp = jax.tree.map(lambda x: x, obs)
        inp = policy._input_transform(inp)
        transformed.append(inp)

    # --- 2. Stack into batch ---
    batched = _stack_leaves(transformed)
    is_pytorch = policy._is_pytorch_model
    if not is_pytorch:
        batched = jax.tree.map(lambda x: jnp.asarray(x), batched)
        policy._rng, sample_rng = jax.random.split(policy._rng)
    else:
        import torch
        batched = jax.tree.map(
            lambda x: torch.from_numpy(np.asarray(x)).to(policy._pytorch_device),
            batched,
        )
        sample_rng = policy._pytorch_device

    # --- 3. Batched model forward ---
    observation = _model.Observation.from_dict(batched)
    raw_actions = policy._sample_actions(sample_rng, observation, **policy._sample_kwargs)
    # raw_actions shape: (B, action_horizon, action_dim)

    # --- 4. Split per-sample & apply output transforms ---
    batch_size = len(obs_list)
    results: list[dict] = []
    for i in range(batch_size):
        if is_pytorch:
            actions_i = np.asarray(raw_actions[i].detach().cpu())
            state_i = np.asarray(batched["state"][i].detach().cpu())
        else:
            actions_i = np.asarray(raw_actions[i])
            state_i = np.asarray(batched["state"][i])
        out = {"state": state_i, "actions": actions_i}
        out = policy._output_transform(out)
        results.append(out)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="PairUAV OpenPI inference & evaluation")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pi05_ours",
        help="Path to the checkpoint directory (top-level or step-level).",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/root/dreamNav/pairUAV/try_test",
        help="Root directory of the test set (contains tour sub-dirs with JSONs).",
    )
    parser.add_argument(
        "--tour-dir",
        type=str,
        default="/root/dreamNav/pairUAV/tours",
        help="Root directory containing tour image sub-dirs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (higher = faster but more VRAM).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate (for quick debugging).",
    )
    args = parser.parse_args()

    # ---- Resolve checkpoint ------------------------------------------------
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    logging.info(f"Checkpoint dir: {ckpt_dir}")

    # ---- Build policy ------------------------------------------------------
    logging.info("Building policy …")
    policy = build_policy(ckpt_dir)
    logging.info("Policy ready.")

    # ---- Collect test samples ----------------------------------------------
    test_dir = args.test_dir
    sample_paths: list[str] = []
    for tour_id in sorted(os.listdir(test_dir)):
        tour_path = os.path.join(test_dir, tour_id)
        if not os.path.isdir(tour_path):
            continue
        for jf in sorted(os.listdir(tour_path)):
            if jf.endswith(".json"):
                sample_paths.append(os.path.join(tour_path, jf))

    if args.max_samples is not None:
        sample_paths = sample_paths[: args.max_samples]

    total = len(sample_paths)
    logging.info(f"Total test samples: {total}")

    # ---- Run batched inference ---------------------------------------------
    image_cache: dict[str, np.ndarray] = {}
    bs = args.batch_size

    heading_errors: list[float] = []
    range_errors: list[float] = []
    successes: list[bool] = []

    t0 = time.time()
    num_batches = (total + bs - 1) // bs

    for batch_idx in tqdm(range(num_batches), desc="Inference (batched)"):
        start = batch_idx * bs
        end = min(start + bs, total)
        batch_paths = sample_paths[start:end]

        obs_list: list[dict] = []
        gt_headings: list[float] = []
        gt_ranges: list[float] = []

        for jp in batch_paths:
            obs, gt_h, gt_r = load_sample(jp, args.tour_dir, image_cache)
            obs_list.append(obs)
            gt_headings.append(gt_h)
            gt_ranges.append(gt_r)

        results = batched_infer(policy, obs_list)

        for i, res in enumerate(results):
            pred_actions = res["actions"]  # shape (1, 2) — [heading_norm, range_norm]
            pred_heading_norm = float(pred_actions[0, 0])
            pred_range_norm = float(pred_actions[0, 1])
            pred_heading_deg, pred_range_m = denormalize(pred_heading_norm, pred_range_norm)

            gt_heading_deg = gt_headings[i]
            gt_range_m = gt_ranges[i]

            # --- Heading MAE (handle wrap-around at ±180°) ---
            h_err = abs(pred_heading_deg - gt_heading_deg)
            if h_err > 180.0:
                h_err = 360.0 - h_err
            heading_errors.append(h_err)

            # --- Range MAE ---
            range_errors.append(abs(pred_range_m - gt_range_m))

            # --- Success rate (position error < 10 m) ---
            gt_x, gt_y = heading_range_to_xy(gt_heading_deg, gt_range_m)
            pred_x, pred_y = heading_range_to_xy(pred_heading_deg, pred_range_m)
            pos_error = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            successes.append(pos_error <= SUCCESS_THRESHOLD_M)

    elapsed = time.time() - t0

    # ---- Report metrics ----------------------------------------------------
    n = len(heading_errors)
    heading_mae = np.mean(heading_errors)
    range_mae = np.mean(range_errors)
    success_rate = np.mean(successes) * 100.0

    print("\n" + "=" * 60)
    print(f"  PairUAV OpenPI Inference Results  ({n} samples)")
    print("=" * 60)
    print(f"  Heading MAE        : {heading_mae:8.2f}°")
    print(f"  Range MAE          : {range_mae:8.2f} m")
    print(f"  Success Rate (<10m): {success_rate:8.2f}%")
    print(f"  Total time         : {elapsed:8.1f} s  ({elapsed / max(n, 1):.3f} s/sample)")
    print(f"  Batch size         : {args.batch_size}")
    print("=" * 60)


if __name__ == "__main__":
    main()
