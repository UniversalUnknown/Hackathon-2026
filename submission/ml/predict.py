"""
Full Drift-Sense localization: denoise -> candidates -> re-rank -> refine.

Pipeline
--------
1. Median pre-filter the search image (suppresses the SEM noise so the true
   correlation peak survives -- measured +6 recall and +7 hits on the eval set).
2. Classical multi-scale / multi-rotation ZNCC over the (denoised) search image
   (ml.zncc) -> top-K non-maximum-suppressed candidate locations.
3. Every candidate gets a *defect-residue score* (ml.zncc): the FFT
   periodic (lattice) background is subtracted from the reference template and
   from the candidate's search window; the normalized cross-correlation of the
   two *non-periodic residues* is computed. Repeated patterns share the lattice
   but only the true site shares the reference's unique defects, so this
   signal disambiguates near-tied correlations.
4. A small CNN (ml.ranker.Ranker) scores each candidate (whole-scene context)
   and its probability is reported as confidence for every candidate.
5. Selection (``cfg["infer"]["selection"]``):
     - ``"fuse"`` (default): combined = ZNCC score + ``defect_weight`` *
       defect-residue score; highest combined wins. Measured best (+8 hits).
     - ``"zncc"``: the correlation winner wins.
     - ``"cnn_tie"``: as ``"fuse"``, but when the top two ZNCC scores are
       nearly tied (gap < ``tie_epsilon``) the most probable CNN candidate
       wins instead.
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


class Localizer:
    def __init__(self, cfg: dict, weights: str | Path, device: str = "cpu"):
        self.cfg = cfg["infer"]
        n_feat = 1 + len(self.cfg["scales"])
        state = torch.load(weights, map_location="cpu")
        in_ch = int(state["model"]["features.0.net.0.weight"].shape[1])
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
        if med > 1:
            search_m = cv2.medianBlur(search, med)
        else:
            search_m = search

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

        # Defect-residue score per candidate: the non-periodic (unique) content
        # of the reference must appear at the true site and nowhere else.
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
        order = np.argsort(-scores)
        gap = scores[order[0]] - scores[order[1]] if len(cands) > 1 else np.inf
        mode = inf.get("selection", "fuse")
        w_def = float(inf.get("defect_weight", 0.5))
        if mode == "zncc":
            pick = cands[order[0]]
        elif mode == "cnn":
            pick = max(cands, key=lambda c: c["prob"])
        elif mode == "cnn_tie":
            # Ambiguous repeat (top two correlations nearly tied): let the
            # model's scene-level context decide; otherwise the fused score.
            if gap < inf["tie_epsilon"]:
                pick = max(cands, key=lambda c: c["prob"])
            else:
                pick = cands[int((scores + w_def * defects).argmax())]
        else:  # "fuse" (default)
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
