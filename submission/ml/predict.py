"""
Full Drift-Sense localization: candidates -> CNN re-rank -> refine.

Pipeline
--------
1. Classical multi-scale / multi-rotation ZNCC over the full search image
   (ml.zncc) -> top-K non-maximum-suppressed candidate locations.
2. A small CNN (ml.ranker.Ranker) scores each candidate: how likely is this
   the TRUE site of the reference pattern? It sees the local context crop, the
   reference template and the per-scale ZNCC scores. This resolves the repeated-
   pattern ambiguity that pure correlation cannot.
3. Selection: the candidate with the highest model probability wins; if several
   candidates are within ``tie_epsilon`` of the best probability (genuinely
   ambiguous repeats), the problem statement's tie-break applies -- pick the one
   closest to the search-image centre.
4. Parabolic sub-pixel refinement on the winning template alignment.
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
from ml.zncc import correlation_maps, per_scale_scores, refine_peak, top_k_candidates


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
        inf = self.cfg
        scales = inf["scales"]
        rotations = inf["rotations"]
        k = inf["n_candidates"]
        min_dist = inf["candidate_min_dist"]
        min_score = inf["min_score"]

        maps = correlation_maps(search, reference, scales, rotations)
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

        best = max(cands, key=lambda c: c["prob"])
        # tie-break (problem statement): among near-equal probabilities pick the
        # valid match closest to the search-image centre.
        tie_eps = inf["tie_epsilon"]
        top_group = [c for c in cands
                     if best["prob"] - c["prob"] <= tie_eps
                     and c["score"] >= min_score]
        if not top_group:
            top_group = [best]
        pick = min(top_group, key=lambda c: np.hypot(
            c["x"] + c["template_w"] / 2.0 - search.shape[1] / 2.0,
            c["y"] + c["template_w"] / 2.0 - search.shape[0] / 2.0))

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
            "scale": float(pick["scale"]),
            "rot": float(pick["rot"]),
            "n_candidates": len(cands),
            "n_top_group": len(top_group),
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
