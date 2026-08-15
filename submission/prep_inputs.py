"""Precompute the ranker input tensors (X, F, y) for a split to a single .npz.

The crop/resize/z-score input building is the slow part of the training loop;
materialising it once makes training epochs ~5x faster. Output:
``<root>/<split>/ranker_inputs.npz`` with X (N, k, 2, 64, 64) f32,
F (N, k, n_feat) f32, y (N, k) f32 (masked with -100 for padding).

Usage:
    python prep_inputs.py --config configs/default.json --split train
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml.ranker import build_rank_inputs, build_rank_inputs_global

BUILDERS = {"local": build_rank_inputs, "global": build_rank_inputs_global}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--variant", default="local", choices=list(BUILDERS))
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    dcfg, tcfg = cfg["dataset"], cfg["train"]
    k = tcfg["candidate_k"]
    split_root = Path(dcfg["root"]) / args.split
    npz_paths = sorted((split_root / "candidates").glob("*.npz"))
    if args.max_samples:
        npz_paths = npz_paths[: args.max_samples]

    n = len(npz_paths)
    n_feat = 1 + len(cfg["infer"]["scales"])
    n_ch = 3 if args.variant == "global" else 2
    X = np.zeros((n, k, n_ch, 64, 64), dtype=np.float32)
    F = np.full((n, k, n_feat), np.nan, dtype=np.float32)
    y = np.full((n, k), -100.0, dtype=np.float32)
    mask = np.zeros((n, k), dtype=bool)

    for i, p in enumerate(npz_paths):
        sid = int(p.stem)
        data = np.load(p, allow_pickle=True)
        cands = list(data["cands"])
        search = cv2.imread(str(split_root / "search" / f"{sid:05d}.png"),
                            cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(split_root / "reference" / f"{sid:05d}.png"),
                               cv2.IMREAD_GRAYSCALE)
        xi, fi = BUILDERS[args.variant](search, reference, cands)
        m = min(xi.shape[0], k)
        X[i, :m] = xi[:m]
        F[i, :m] = fi[:m]
        y[i, :m] = np.asarray([c["label"] for c in cands[:m]])
        mask[i, :m] = True
        if (i + 1) % 200 == 0:
            print(f"[{args.split}] {i+1}/{n}", flush=True)

    out = split_root / f"ranker_inputs_{args.variant}.npz"
    np.savez_compressed(out, X=X, F=F, y=y, mask=mask)
    print(f"saved {out} ({n} samples)")


if __name__ == "__main__":
    main()
