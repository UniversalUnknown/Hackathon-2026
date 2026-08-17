"""
Full Drift-Sense localization: denoise -> candidates -> re-rank -> refine.

Pipeline
--------
1. Dual pre-filter: bilateral filter (edge-preserving) + median blur to
   aggressively suppress SEM speckle noise while preserving edges.
2. Classical multi-scale / multi-rotation ZNCC over the (denoised) search
   image (ml.zncc) -> top-K non-maximum-suppressed candidate locations.
3. Every candidate gets a *defect-residue score* (ml.zncc): the FFT
   periodic (lattice) background is subtracted from the reference template and
   from the candidate's search window; the normalized cross-correlation of the
   two *non-periodic residues* is computed. Repeated patterns share the lattice
   but only the true site shares the reference's unique defects.
4. A CNN (ml.ranker.Ranker) scores each candidate (whole-scene context).
5. Selection: fuse_cnn combines ZNCC + defect residue + CNN probability with
   normalised score contributions for stable weighting.
6. Parabolic sub-pixel refinement on the winning template alignment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml.ranker import Ranker, build_rank_inputs, build_rank_inputs_global
from ml.zncc import (correlation_maps, defect_residue_ncc, per_scale_scores,
                     refine_peak, top_k_candidates, _build_template)


def _denoise(img: np.ndarray, median_k: int, bilateral: bool) -> np.ndarray:
    """Apply edge-preserving denoising pipeline to a search image.

    bilateral=True: runs a bilateral filter first (preserves edges/defects
    while blurring noise) then a median filter for salt-and-pepper.
    bilateral=False: median only (original behaviour).
    """
    import cv2
    out = img
    if bilateral:
        # sigmaColor / sigmaSpace tuned for SEM speckle at ~10x downscale
        out = cv2.bilateralFilter(out.astype(np.float32), d=7,
                                  sigmaColor=25, sigmaSpace=7).astype(np.uint8)
    if median_k > 1:
        out = cv2.medianBlur(out, median_k)
    return out


def _normalize_scores(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]. Returns zeros if all values are equal."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


class Localizer:
    def __init__(self, cfg: dict, weights: str | Path, device: str = "cpu"):
        self.cfg = cfg["infer"]
        n_feat = 1 + len(self.cfg["scales"])
        state = torch.load(weights, map_location="cpu")
        in_ch = int(state["model"]["stem.0.weight"].shape[1])
        self.model = Ranker(n_feat=n_feat, in_ch=in_ch)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.device = device
        self.in_ch = in_ch
        self.build_inputs = build_rank_inputs_global if in_ch == 3 else build_rank_inputs
        if device != "cpu":
            self.model.to(device)

    def localize(self, search: np.ndarray, reference: np.ndarray) -> dict:
        import cv2
        inf = self.cfg
        scales = inf["scales"]
        rotations = inf["rotations"]
        k = inf["n_candidates"]
        min_dist = inf["candidate_min_dist"]
        min_score = inf["min_score"]

        med = int(inf.get("prefilter_median", 0))
        use_bilateral = bool(inf.get("prefilter_bilateral", False))
        search_m = _denoise(search, med, use_bilateral)

        maps = correlation_maps(search_m, reference, scales, rotations)
        cands = top_k_candidates(maps, k=k, min_dist=min_dist,
                                 score_threshold=min_score)
        if not cands:
            return {"ok": False, "reason": "no candidates above min_score"}

        per_scale = [per_scale_scores(maps, int(c["x"]), int(c["y"]))
                     for c in cands]
        X, F = self.build_inputs(search, reference, cands, per_scale)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device),
                                torch.from_numpy(F).to(self.device))
            probs = torch.sigmoid(logits).numpy() if self.device == "cpu" \
                else torch.sigmoid(logits).cpu().numpy()
        for c, p in zip(cands, probs):
            c["prob"] = float(p)

        # Defect-residue score per candidate
        defect = []
        for c in cands:
            t = _build_template(reference, c["scale"], c["rot"])
            tw = int(round(c["template_w"]))
            x, y = int(c["x"]), int(c["y"])
            w = search_m[max(y, 0):min(y + tw, search_m.shape[0]),
                         max(x, 0):min(x + tw, search_m.shape[1])]
            w = cv2.resize(w, (tw, tw), interpolation=cv2.INTER_LINEAR)
            if abs(c["rot"]) > 1e-3:
                M = cv2.getRotationMatrix2D(((tw - 1) / 2.0, (tw - 1) / 2.0),
                                            c["rot"], 1.0)
                w = cv2.warpAffine(w, M, (tw, tw), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            d = defect_residue_ncc(t, w)
            defect.append(d)
            c["defect"] = d

        scores = np.array([c["score"] for c in cands])
        defects = np.array(defect)
        cnn_probs = np.array([c["prob"] for c in cands])
        order = np.argsort(-scores)
        gap = scores[order[0]] - scores[order[1]] if len(cands) > 1 else np.inf
        mode = inf.get("selection", "fuse")
        w_def = float(inf.get("defect_weight", 0.5))

        if mode == "zncc":
            pick = cands[order[0]]
        elif mode == "cnn":
            pick = max(cands, key=lambda c: c["prob"])
        elif mode == "cnn_tie":
            if gap < inf["tie_epsilon"]:
                pick = max(cands, key=lambda c: c["prob"])
            else:
                pick = cands[int((scores + w_def * defects).argmax())]
        elif mode == "fuse_cnn":
            w_cnn = float(inf.get("cnn_weight", 0.4))
            # Normalize each score component independently to [0,1] before
            # combining — prevents one component from dominating due to scale.
            s_norm = _normalize_scores(scores)
            d_norm = _normalize_scores(defects)
            p_norm = _normalize_scores(cnn_probs)
            # Improved fusion: emphasis on CNN when confidence is high
            conf_threshold = float(inf.get("conf_threshold", 0.25))
            high_conf_mask = cnn_probs >= conf_threshold
            if np.any(high_conf_mask):
                # Filter to high-confidence CNN predictions
                combined = np.where(high_conf_mask, 
                                   s_norm + w_def * d_norm + w_cnn * p_norm,
                                   -np.inf)
            else:
                combined = s_norm + w_def * d_norm + w_cnn * p_norm
            pick = cands[int(combined.argmax())]
        else:  # "fuse" (default fallback)
            pick = cands[int((scores + w_def * defects).argmax())]

        cx0 = pick["x"] + pick["template_w"] / 2.0
        cy0 = pick["y"] + pick["template_w"] / 2.0
        refined = refine_peak(search, reference, cx0, cy0, pick["scale"],
                              pick["rot"], inf["refine_search_radius"])

        return {
            "ok": True,
            "x": float(refined["x"]),
            "y": float(refined["y"]),
            "score": float(refined["score"]),
            "prob": float(pick["prob"]),
            "defect": float(pick["defect"]),
            "score_gap": float(gap),
            "scale": float(pick["scale"]),
            "rot": float(pick["rot"]),
            "n_candidates": len(cands),
            "refined": refined,
        }


def localize_pair(search_path: str, reference_path: str, cfg: dict,
                  weights: str) -> dict:
    import cv2
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    loc = Localizer(cfg, weights)
    return loc.localize(search, reference)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--weights", default="output/weights/ranker.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    result = localize_pair(args.search, args.reference, cfg, args.weights)
    if result.get("ok"):
        print(f"{result['x']:.3f} {result['y']:.3f}")
    else:
        print(f"FAILED: {result['reason']}", file=sys.stderr)
        sys.exit(1)
