#!/usr/bin/env python3
"""Generate the Drift-Sense synthetic dataset (train + eval splits) to disk.

Writes, for each split: output/dataset/{split}/reference/NNNNN.png,
output/dataset/{split}/search/NNNNN.png and a per-split manifest.csv holding
paths, ground-truth centre (gt_x, gt_y in search px), scale, rotation and
generation metadata (architecture, noise level, per-pair parameters, seed).

Example:
    python generate_dataset.py --train-n 1000 --eval-n 60 --seed 20260815
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import generate_pair, sample_training_params, NOISE_LEVELS
from model.preprocess import SEARCH_IN  # noqa: F401  (import check)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-n", type=int, default=1000)
    p.add_argument("--eval-n", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--root", default="output/dataset")
    p.add_argument("--config", default="configs/default.json")
    return p.parse_args()


FIELD_ORDER = [
    "id", "reference_path", "search_path", "gt_x", "gt_y", "scale",
    "rotation_deg", "architecture", "noise_level", "seed",
    "beam_spot_size_nm", "collapse_threshold_nm", "dose_reference",
    "dose_search", "shear_amplitude_px", "drift_jitter_px",
    "detector_noise_sigma_ref", "detector_noise_sigma_search",
    "astigmatism_ratio", "vignette_strength", "gamma", "barrel_distortion_k",
    "charging_streak_prob", "charging_streak_intensity", "speckle_sigma",
    "salt_pepper_prob", "mat_size_nm", "strip_width_nm", "boundary_bias",
    "linewidth_bias_nm", "corner_rounding_px",
]


def main():
    args = parse_args()
    cfg = json.load(open(args.config))["dataset"]
    archs = cfg["architectures"]

    for split, n in (("train", args.train_n), ("eval", args.eval_n)):
        if n <= 0:
            continue
        seed = args.seed if split == "train" else args.seed + 12345
        rng = np.random.default_rng(seed)
        ref_dir = os.path.join(args.root, split, "reference")
        search_dir = os.path.join(args.root, split, "search")
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(search_dir, exist_ok=True)
        manifest = os.path.join(args.root, split, "manifest.csv")

        t0 = time.time()
        with open(manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_ORDER,
                                    extrasaction="ignore")
            writer.writeheader()
            for i in range(n):
                arch = archs[int(rng.integers(0, len(archs)))]
                level = list(NOISE_LEVELS.keys())[int(rng.integers(0, 4))]
                params = sample_training_params(rng, level)
                scale = cfg["scale_options"][int(rng.integers(0, len(cfg["scale_options"])))] \
                    if rng.random() < cfg["scale_prob"] else 10.0
                rot = cfg["rotation_options"][int(rng.integers(0, len(cfg["rotation_options"])))] \
                    if rng.random() < cfg["rotation_prob"] else 0.0
                sample = generate_pair(arch, rng, params, scale=scale,
                                       rotation_deg=rot)
                ref_path = os.path.join(ref_dir, f"{i:05d}.png")
                search_path = os.path.join(search_dir, f"{i:05d}.png")
                cv2.imwrite(ref_path, sample["reference_img"])
                cv2.imwrite(search_path, sample["search_img"])
                row = {
                    "id": i,
                    "reference_path": ref_path,
                    "search_path": search_path,
                    "gt_x": f"{sample['gt_x']:.4f}",
                    "gt_y": f"{sample['gt_y']:.4f}",
                    "scale": f"{scale:.3f}",
                    "rotation_deg": f"{rot:.3f}",
                    "architecture": arch,
                    "noise_level": level,
                    "seed": sample["seed"],
                }
                row.update({k: f"{v:.4f}" if isinstance(v, float) else v
                            for k, v in params.items()})
                writer.writerow(row)
                if (i + 1) % 100 == 0 or i == n - 1:
                    el = time.time() - t0
                    print(f"[{split}] {i+1}/{n}  gt=({sample['gt_x']:.0f},{sample['gt_y']:.0f}) "
                          f"s={scale:.1f} r={rot:+.1f}  {el:.0f}s", flush=True)
        print(f"[{split}] wrote {n} samples -> {os.path.join(args.root, split)}")


if __name__ == "__main__":
    main()
