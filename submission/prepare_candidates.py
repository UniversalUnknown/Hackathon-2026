"""Precompute the classical ZNCC candidates for every dataset sample.

Runs the expensive multi-scale/multi-rotation correlation once per image and
caches the top-K candidate boxes (plus per-scale scores and the label) to
``<root>/<split>/candidates/<id>.npz`` so training the re-ranker never has to
recompute correlations.

Usage:
    python prepare_candidates.py --config configs/default.json [--split train] [--max-samples N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_gen"))

from ml.zncc import correlation_maps, per_scale_scores, top_k_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--job-idx", type=int, default=0, help="parallel: slice index")
    ap.add_argument("--job-n", type=int, default=1, help="parallel: number of slices")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    d = cfg["dataset"]
    split_root = Path(d["root"]) / args.split
    with open(split_root / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = rows

    if args.max_samples:
        manifest = manifest[: args.max_samples]
    if args.job_n > 1:
        manifest = manifest[args.job_idx :: args.job_n]

    k = cfg["train"]["candidate_k"]
    min_dist = cfg["train"]["candidate_min_dist"]
    pos_margin = cfg["train"]["pos_margin_px"]
    scales = cfg["infer"]["scales"]
    rotations = cfg["infer"]["rotations"]
    med = int(cfg["infer"].get("prefilter_median", 0))

    out_dir = split_root / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    for row in manifest:
        sid = int(row["id"])
        npz_path = out_dir / f"{sid:05d}.npz"
        if npz_path.exists():
            done += 1
            continue
        search = cv2_imread(split_root / "search" / f"{sid:05d}.png")
        reference = cv2_imread(split_root / "reference" / f"{sid:05d}.png")
        if search is None or reference is None:
            print(f"skip {sid}: missing image", flush=True)
            continue
        if med > 1:
            import cv2 as _cv2
            search = _cv2.medianBlur(search, med)

        maps = correlation_maps(search, reference, scales, rotations)
        cands = top_k_candidates(maps, k=k, min_dist=min_dist)

        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        recs = []
        for c in cands:
            cx = c["x"] + c["template_w"] / 2.0
            cy = c["y"] + c["template_w"] / 2.0
            recs.append({
                "x": c["x"], "y": c["y"],
                "cx": cx, "cy": cy,
                "score": c["score"], "scale": c["scale"], "rot": c["rot"],
                "template_w": c["template_w"],
                "per_scale": per_scale_scores(maps, c["x"], c["y"]),
                "label": float(np.hypot(cx - gx, cy - gy) <= pos_margin),
            })
        np.savez_compressed(
            npz_path,
            n=len(recs),
            cands=recs,
            arch=row["architecture"],
            level=row["noise_level"],
        )
        done += 1
        if done % 20 == 0:
            print(f"[{args.split}] {done}/{len(manifest)}", flush=True)

    print(f"[{args.split}] done: {done} samples -> {out_dir}")


def cv2_imread(path):
    import cv2
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":
    main()
