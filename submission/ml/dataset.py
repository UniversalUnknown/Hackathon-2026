"""
PyTorch dataset over precomputed ZNCC candidate caches.

Each item is one image pair's candidate set: the full (K, 2, 64, 64) ranker
input, (K, n_feat) per-candidate features, and (K,) binary labels
(centre within ``pos_margin_px`` of the ground truth = 1).
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.ranker import build_rank_inputs
from ml.zncc import refine_peak


class CandidateDataset(Dataset):
    def __init__(self, split_root: str, aug: bool = True,
                 pos_margin_px: float = 20.0, k: int = 5):
        self.split_root = Path(split_root)
        self.aug = aug
        self.pos_margin = pos_margin_px
        self.k = k
        self.npz_paths = sorted((self.split_root / "candidates").glob("*.npz"))

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, i):
        npz_path = self.npz_paths[i]
        sid = int(npz_path.stem)
        data = np.load(npz_path, allow_pickle=True)
        cands = list(data["cands"])

        search = cv2.imread(str(self.split_root / "search" / f"{sid:05d}.png"),
                            cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(self.split_root / "reference" / f"{sid:05d}.png"),
                               cv2.IMREAD_GRAYSCALE)

        X, F = build_rank_inputs(search, reference, cands)
        y = np.asarray([c["label"] for c in cands], dtype=np.float32)

        if self.aug:
            rnd = random.Random(i)
            # spatial flip (search+template both flip -> fine for the ranker)
            if rnd.random() < 0.5:
                X = X[:, :, ::-1].copy()
            if rnd.random() < 0.5:
                X = X[:, :, :, ::-1].copy()
            # small brightness/contrast on the search channel only
            ch0 = X[:, 0] * rnd.uniform(0.9, 1.1) + rnd.uniform(-0.03, 0.03)
            X[:, 0] = np.clip(ch0, -4, 4)

        return (
            torch.from_numpy(X.copy()),
            torch.from_numpy(F.copy()),
            torch.from_numpy(y.copy()),
        )


def collate_candidates(batch, k: int):
    """Pad candidate sets to k and produce a mask for masked-BCE.
    Handles any number of input channels / feature dims from the first item."""
    n = len(batch)
    x0 = batch[0][0]
    ch, h, w = x0.shape[1], x0.shape[2], x0.shape[3]
    nf = batch[0][1].shape[1]
    X = torch.zeros(n, k, ch, h, w)
    F = torch.zeros(n, k, nf)
    y = torch.full((n, k), -100.0)   # ignored via mask
    mask = torch.zeros(n, k, dtype=torch.bool)
    for i, (xi, fi, yi) in enumerate(batch):
        m = min(xi.shape[0], k)
        X[i, :m] = xi[:m]
        F[i, :m] = fi[:m]
        y[i, :m] = yi[:m]
        mask[i, :m] = True
    return X, F, y, mask
