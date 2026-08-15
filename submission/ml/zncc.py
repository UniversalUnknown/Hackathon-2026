"""
Classical multi-scale/multi-rotation ZNCC matching: the "search engine".

Provides the candidate generator used by both the learned re-ranker and the
pure classical baseline:
  * correlation_maps()    -- ZNCC (cv2.TM_CCOEFF_NORMED) at every (scale, rot)
  * top_k_candidates()    -- non-maximum-suppressed best peaks across scales
  * refine_peak()         -- sub-pixel (parabolic) refinement of a peak

scales ~ 9:1..11:1 covers the nominal 10:1 with the stated tolerance;
rotations cover the small (1-2 deg) search-image rotation.

Coordinates are the template-centre in search-image pixels.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import maximum_filter

DEFAULT_SCALES = (9.0, 9.5, 10.0, 10.5, 11.0)
DEFAULT_ROTATIONS = (-2.0, 0.0, 2.0)


def _build_template(reference: np.ndarray, scale: float, rot: float) -> np.ndarray:
    tw = max(int(round(reference.shape[1] / scale)), 4)
    th = max(int(round(reference.shape[0] / scale)), 4)
    t = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
    if abs(rot) > 1e-3:
        M = cv2.getRotationMatrix2D(((tw - 1) / 2.0, (th - 1) / 2.0), rot, 1.0)
        t = cv2.warpAffine(t, M, (tw, th), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return t


def correlation_maps(search: np.ndarray, reference: np.ndarray,
                     scales=DEFAULT_SCALES, rotations=DEFAULT_ROTATIONS) -> dict:
    """{(scale, rot): ZNCC result map}. Peak location = template top-left."""
    maps = {}
    for s in scales:
        for r in rotations:
            t = _build_template(reference, s, r)
            if t.shape[0] >= search.shape[0] or t.shape[1] >= search.shape[1]:
                continue
            maps[(s, r)] = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
    return maps


def best_over_maps(maps: dict) -> tuple[np.ndarray, dict]:
    """Max ZNCC score over all (scale, rot) at every location, plus the
    winning key per location. Maps are padded to a common size."""
    keys = list(maps.keys())
    H = max(m.shape[0] for m in maps.values())
    W = max(m.shape[1] for m in maps.values())
    best = np.full((H, W), -np.inf, dtype=np.float32)
    winner = np.zeros((H, W), dtype=np.int16)
    for i, k in enumerate(keys):
        m = maps[k]
        if m.shape != (H, W):
            padded = np.full((H, W), -np.inf, dtype=np.float32)
            padded[:m.shape[0], :m.shape[1]] = m
            m = padded
        take = m > best
        best[take] = m[take]
        winner[take] = i
    return best, winner, keys


def top_k_candidates(maps: dict, k: int = 5, min_dist: int = 30,
                     score_threshold: float = -0.5):
    """Return up to k non-maximum-suppressed candidate dicts, best first."""
    best, winner, keys = best_over_maps(maps)
    mx = maximum_filter(best, size=min_dist, mode="constant", cval=-np.inf)
    is_local = (best == mx) & (best > score_threshold)

    order = np.argsort(best[is_local])[::-1]
    ys, xs = np.where(is_local)
    picks = []
    used = []
    for idx in order:
        x, y = int(xs[idx]), int(ys[idx])
        if any(np.hypot(x - ux, y - uy) < min_dist for (ux, uy) in used):
            continue
        used.append((x, y))
        key = keys[int(winner[y, x])]
        picks.append({
            "x": x, "y": y,
            "score": float(best[y, x]),
            "scale": float(key[0]),
            "rot": float(key[1]),
            "template_w": _tw_for(maps[key], key),
        })
        if len(picks) >= k:
            break
    return picks


def _tw_for(map0, key) -> float:
    return 1000.0 / key[0]  # template width in search px for a 1000px reference


def per_scale_scores(maps: dict, x: int, y: int) -> np.ndarray:
    """Max-over-rotation ZNCC score at location (x, y) for each scale.
    Locations outside a given map's domain (smaller templates) are clamped
    to the map edge."""
    scales = sorted({k[0] for k in maps.keys()})
    out = []
    for s in scales:
        best = -np.inf
        for (ss, r), m in maps.items():
            if ss == s:
                cx = min(max(x, 0), m.shape[1] - 1)
                cy = min(max(y, 0), m.shape[0] - 1)
                best = max(best, float(m[cy, cx]))
        out.append(best)
    return np.asarray(out, dtype=np.float32)


def subpixel_parabola(peak_val: float, left: float, right: float) -> float:
    """Parabolic sub-pixel offset in [-0.5, 0.5] from 3 samples."""
    denom = 2.0 * peak_val - left - right
    if abs(denom) < 1e-9:
        return 0.0
    return 0.5 * (left - right) / denom


def refine_peak(search: np.ndarray, reference: np.ndarray, cxc: float, cyc: float,
                scale: float, rot: float, search_radius: int = 8) -> dict:
    """Refine a candidate (given its CENTRE coords) to sub-pixel by re-matching
    with its winning (scale, rot) template in a small window and applying
    parabolic fits."""
    t = _build_template(reference, scale, rot)
    tw, th = t.shape[1], t.shape[0]

    x0 = int(round(cxc - tw / 2.0)) - search_radius
    y0 = int(round(cyc - th / 2.0)) - search_radius
    x1 = x0 + tw + 2 * search_radius
    y1 = y0 + th + 2 * search_radius
    # clamp, keeping the window anchored near the centre
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > search.shape[1]:
        x0 -= x1 - search.shape[1]
        x1 = search.shape[1]
    if y1 > search.shape[0]:
        y0 -= y1 - search.shape[0]
        y1 = search.shape[0]
    window = search[y0:y1, x0:x1]

    res = cv2.matchTemplate(window, t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    px, py = loc
    cx_abs = x0 + px + tw / 2.0
    cy_abs = y0 + py + th / 2.0

    # sub-pixel offsets along x and y
    ox = 0.0
    oy = 0.0
    if 1 <= px < res.shape[1] - 1:
        ox = subpixel_parabola(float(res[py, px]), float(res[py, px - 1]),
                               float(res[py, px + 1]))
    if 1 <= py < res.shape[0] - 1:
        oy = subpixel_parabola(float(res[py, px]), float(res[py - 1, px]),
                               float(res[py + 1, px]))

    return {
        "x": cx_abs + ox,
        "y": cy_abs + oy,
        "score": float(score),
        "scale": float(scale),
        "rot": float(rot),
    }
