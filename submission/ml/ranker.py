"""
Learned candidate re-ranker — improved architecture.

Given the classical ZNCC candidates, a deeper residual CNN with Squeeze-and-
Excitation attention scores each one: how likely is this candidate the TRUE
location of the reference pattern? It sees
  * a context crop of the search image around the candidate (2 template widths
    square, downsampled to 64x64, z-scored),
  * the reference template (32x32, z-scored, tiled 2x2 -> 64x64),
  * the per-scale ZNCC scores at the candidate location (numeric features).

Architecture improvements over baseline:
  - Residual blocks with skip connections for better gradient flow
  - Squeeze-and-Excitation (SE) attention for channel re-weighting
  - Wider feature maps: 48->96->192 channels
  - LayerNorm in MLP head for stable training
  - Deeper MLP: 192+n_feat -> 128 -> 64 -> 1
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.preprocess import zscore


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class _ResBlock(nn.Module):
    """Residual conv block with BN + ReLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class Ranker(nn.Module):
    """Enhanced residual CNN ranker with SE attention and improved capacity.

    Input : x (B, in_ch, 64, 64) image tensor
            feat (B, n_feat) per-candidate numeric features
    Output: (B,) raw logit — sigmoid = P(true site)
    """
    def __init__(self, n_feat: int = 6, in_ch: int = 2):
        super().__init__()
        # Stem: in_ch -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Layer 1: 64 -> 64, pool to 32x32, with two residual blocks
        self.layer1 = nn.Sequential(
            _ResBlock(64),
            _ResBlock(64),
            nn.MaxPool2d(2),
        )
        # Layer 2: 64 -> 128, pool to 16x16
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _ResBlock(128),
            _ResBlock(128),
            nn.MaxPool2d(2),
        )
        # Layer 3: 128 -> 256, pool to 8x8
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            _ResBlock(256),
            _ResBlock(256),
            nn.MaxPool2d(2),
        )
        # SE attention on 256-channel feature map
        self.se = _SEBlock(256, reduction=16)

        # MLP head: fuses visual features + numeric features with improved depth
        self.mlp = nn.Sequential(
            nn.Linear(256 + n_feat, 192),
            nn.LayerNorm(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x, feat):
        h = self.stem(x)       # B, 64, 64, 64
        h = self.layer1(h)     # B, 64, 32, 32
        h = self.layer2(h)     # B, 128, 16, 16
        h = self.layer3(h)     # B, 256, 8, 8
        h = self.se(h)         # channel attention
        h = h.mean(dim=(2, 3)) # global average pool -> B, 256
        return self.mlp(torch.cat([h, feat], dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Input-building helpers (used by both training pipeline and inference)
# ---------------------------------------------------------------------------

def context_crop(search: np.ndarray, cx: float, cy: float,
                 ctx_size: int, out: int = 64) -> np.ndarray:
    """Square crop of the search image centred on (cx, cy), resized to `out`."""
    h, w = search.shape
    half = ctx_size // 2
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(w, x0 + ctx_size)
    y1 = min(h, y0 + ctx_size)
    crop = search[y0:y1, x0:x1]
    if crop.shape[0] != ctx_size or crop.shape[1] != ctx_size:
        crop = np.pad(crop, ((0, ctx_size - crop.shape[0]),
                             (0, ctx_size - crop.shape[1])),
                      mode="edge")
    return cv2_resize(crop, out)


def cv2_resize(img, out):
    import cv2
    return cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA)


def template_channel(reference: np.ndarray, tile: int = 2,
                     t_size: int = 32, out: int = 64) -> np.ndarray:
    """Reference -> t_size px square, z-scored, tiled (tile x tile) -> out."""
    t = cv2_resize(reference, t_size)
    t = zscore(t)
    n = int(np.ceil(out / t_size))
    big = np.tile(t, (n, n))
    return big[:out, :out]


def build_rank_inputs(search: np.ndarray, reference: np.ndarray,
                      candidates: list, per_scale_scores=None) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, F) tensors for a list of candidate dicts.
    per_scale_scores: optional list of per-candidate arrays of length n_scale
    (falls back to the 'per_scale' field stored in each candidate dict).
    Returns X (B,2,64,64) float32, F (B, n_feat) float32."""
    X, F = [], []
    for i, c in enumerate(candidates):
        cx = c["x"] + c["template_w"] / 2.0
        cy = c["y"] + c["template_w"] / 2.0
        ctx_size = int(round(2.0 * c["template_w"]))
        crop = context_crop(search, cx, cy, ctx_size)
        ch0 = zscore(crop)
        ch1 = template_channel(reference)
        X.append(np.stack([ch0, ch1], axis=0))
        feats = [c["score"]]
        if per_scale_scores is not None:
            feats += list(per_scale_scores[i])
        elif "per_scale" in c and c["per_scale"] is not None:
            feats += list(np.asarray(c["per_scale"], dtype=np.float32))
        F.append(feats)
    return (np.stack(X).astype(np.float32),
            np.asarray(F, dtype=np.float32).reshape(len(candidates), -1))


def build_rank_inputs_global(search: np.ndarray, reference: np.ndarray,
                             candidates: list, per_scale_scores=None,
                             out: int = 64,
                             marker_sigma: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Whole-scene variant: 3 channels per candidate --
      * the full search image downsampled to (out x out),
      * same image plus a Gaussian marker at the candidate centre,
      * the tiled reference template.
    Gives the network access to global (zone-boundary, mat) structure.
    Returns X (B,3,64,64), F (B, n_feat)."""
    H, W = search.shape
    base = cv2_resize(search, out)
    base = zscore(base)
    t = template_channel(reference)
    X, F = [], []
    for i, c in enumerate(candidates):
        cx = c["x"] + c["template_w"] / 2.0
        cy = c["y"] + c["template_w"] / 2.0
        marker = np.zeros((out, out), dtype=np.float32)
        mx = int(round(cx / W * (out - 1)))
        my = int(round(cy / H * (out - 1)))
        yy, xx = np.mgrid[0:out, 0:out]
        marker = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * marker_sigma ** 2))
        marker = marker.astype(np.float32)
        X.append(np.stack([base, base + 0.35 * marker, t], axis=0))
        feats = [c["score"]]
        if per_scale_scores is not None:
            feats += list(per_scale_scores[i])
        elif "per_scale" in c and c["per_scale"] is not None:
            feats += list(np.asarray(c["per_scale"], dtype=np.float32))
        F.append(feats)
    return (np.stack(X).astype(np.float32),
            np.asarray(F, dtype=np.float32).reshape(len(candidates), -1))
