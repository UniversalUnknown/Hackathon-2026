"""
Input preprocessing shared by training and inference.

The model sees two 256x256 channels:
  ch0 = the search image downsampled to 256x256 (per-image z-scored)
  ch1 = the reference pattern resized to ~(256/scale) px, z-scored, and tiled
        to fill the frame. Tiling makes the reference context "stationary"
        so the network can compare it against every location in the search
        image (translation-equivalent to a learned template match).

Outputs are a coarse 64x64 heatmap; each heatmap cell covers 256/64 = 4 px
of the 256x256 input = 4 / 0.256 ~= 15.6 px of the 1000x1000 search image.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SEARCH_IN = 256          # model input side (search image downscaled)
HEAT = 64                # heatmap side
SEARCH_FACTOR = SEARCH_IN / 1000.0   # 0.256
CELL_PX = SEARCH_IN / HEAT           # 4 px of the 256 input per heatmap cell
FULL_RES_PER_CELL = 1000.0 / HEAT    # 15.625 px of the search image per cell


def zscore(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    m, s = float(img.mean()), float(img.std())
    return (img - m) / max(s, 1e-5)


def build_template_channel(reference: np.ndarray, scale: float = 10.0,
                           t_size: int = 32) -> np.ndarray:
    """Reference -> small square, z-scored, tiled over a 256x256 frame."""
    t = max(16, min(48, int(round(SEARCH_IN / scale))))
    tmpl = cv2.resize(reference, (t, t), interpolation=cv2.INTER_AREA)
    tmpl_n = zscore(tmpl)
    n = int(math.ceil(SEARCH_IN / t))
    big = np.tile(tmpl_n, (n, n))
    return big[:SEARCH_IN, :SEARCH_IN]


def make_input(reference: np.ndarray, search: np.ndarray,
               scale: float = 10.0) -> np.ndarray:
    """Return float32 array (2, 256, 256)."""
    search256 = cv2.resize(search, (SEARCH_IN, SEARCH_IN), interpolation=cv2.INTER_AREA)
    ch0 = zscore(search256)
    ch1 = build_template_channel(reference, scale)
    return np.stack([ch0, ch1], axis=0).astype(np.float32)


def gt_to_heat_cell(gt_x: float, gt_y: float) -> tuple[float, float]:
    """Ground-truth centre (search px) -> (x, y) in heatmap-cell units (0..64)."""
    return gt_x * SEARCH_FACTOR / CELL_PX, gt_y * SEARCH_FACTOR / CELL_PX


def heat_cell_to_search(cx: float, cy: float) -> tuple[float, float]:
    """Heatmap cell (x, y) -> search-image pixel centre."""
    return cx * FULL_RES_PER_CELL, cy * FULL_RES_PER_CELL


def gaussian_target(cell_x: float, cell_y: float, sigma: float = 2.0,
                    size: int = HEAT) -> np.ndarray:
    """Normalized (peak 1) Gaussian heatmap centred on (cell_x, cell_y)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    g = np.exp(-(((xx - cell_x) ** 2 + (yy - cell_y) ** 2) / (2.0 * sigma * sigma)))
    return g / float(g.max())
