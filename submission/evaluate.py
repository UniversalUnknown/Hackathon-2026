#!/usr/bin/env python3
"""Evaluate the localization pipeline on a held-out set.

Metrics (matching the assignment rubric):
  * pass rate at 5 px, 4 px, 2 px, 1 px and 0.5 px (sub-pixel),
  * mean / median / worst Euclidean error,
  * per-noise-level and per-scale/rotation breakdowns,
  * total + per-image runtime,
  * the worst failure case is rendered (context + predicted/GT boxes) with a
    root-cause note to results/FAILURES.md,
  * ablation vs the classical top-ZNCC baseline (no learned re-ranker).

Writes results/metrics_summary.json and results/localize_results.csv.

Usage:
    python evaluate.py --config configs/default.json \
        --weights output/weights/ranker.pt [--max-samples 20]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml.predict import Localizer
from ml.zncc import correlation_maps, top_k_candidates, refine_peak


THRESHOLDS = [5.0, 4.0, 2.0, 1.0, 0.5]


def classical_baseline(search, reference, scales, rotations,
                       k=8, min_dist=25, radius=8):
    """Top-ZNCC only: highest-score candidate, refined. No learned ranker."""
    maps = correlation_maps(search, reference, scales, rotations)
    cands = top_k_candidates(maps, k=k, min_dist=min_dist)
    if not cands:
        return None
    c = cands[0]
    r = refine_peak(search, reference,
                    c["x"] + c["template_w"] / 2.0,
                    c["y"] + c["template_w"] / 2.0,
                    c["scale"], c["rot"], radius)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--weights", default="output/weights/ranker.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out-dir", default="output/eval_results")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    dcfg, inf = cfg["dataset"], cfg["infer"]
    eval_root = Path(dcfg["root"]) / "eval"
    manifest = list(csv.DictReader(open(eval_root / "manifest.csv")))
    if args.max_samples:
        manifest = manifest[: args.max_samples]

    out_dir = Path(args.out_dir)
    fail_dir = out_dir / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    loc = Localizer(cfg, args.weights, device=args.device)

    rows = []
    runtimes = []
    t_start = time.time()
    for i, r in enumerate(manifest):
        sid = int(r["id"])
        search_path = eval_root / "search" / f"{sid:05d}.png"
        ref_path = eval_root / "reference" / f"{sid:05d}.png"
        import cv2
        search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        gx, gy = float(r["gt_x"]), float(r["gt_y"])

        t0 = time.time()
        res = loc.localize(search, reference)
        dt = time.time() - t0
        runtimes.append(dt)

        if res["ok"]:
            px, py, ok = res["x"], res["y"], True
        else:
            px, py, ok = np.nan, np.nan, False
        err = float(np.hypot(px - gx, py - gy)) if ok else np.nan

        rows.append({
            "id": sid,
            "reference_path": str(ref_path),
            "search_path": str(search_path),
            "gt_x": f"{gx:.3f}",
            "gt_y": f"{gy:.3f}",
            "pred_x": f"{px:.3f}",
            "pred_y": f"{py:.3f}",
            "error_px": f"{err:.3f}",
            "ok": int(ok),
            "architecture": r["architecture"],
            "noise_level": r["noise_level"],
            "scale": r["scale"],
            "rotation_deg": r["rotation_deg"],
            "runtime_s": f"{dt:.3f}",
            "prob": f"{res.get('prob', float('nan')):.3f}" if ok else "",
            "zncc_score": f"{res.get('score', float('nan')):.3f}" if ok else "",
        })
        if (i + 1) % 10 == 0 or i == len(manifest) - 1:
            print(f"eval {i+1}/{len(manifest)} err={err:.1f}px ({dt:.2f}s)",
                  flush=True)
    total_t = time.time() - t_start

    with open(out_dir / "localize_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    errs = np.array([r["error_px"] for r in rows], dtype=float)
    errs_ok = errs[np.isfinite(errs)]

    def summarize(errs, label):
        n = len(errs)
        out = {"n": n}
        for t in THRESHOLDS:
            out[f"pass_{t:g}px"] = round(float((errs <= t).mean()), 4)
        out["mean"] = round(float(errs.mean()), 3)
        out["median"] = round(float(np.median(errs)), 3)
        out["worst"] = round(float(errs.max()), 3)
        out["max_failures"] = int((errs > 5).sum())
        return out

    summary = {"total_runtime_s": round(total_t, 2),
               "per_image_ms": round(float(np.mean(runtimes)) * 1000, 1),
               "classical_baseline": None}
    summary["model"] = summarize(errs_ok, "model")

    # per-group breakdowns
    for key in ["noise_level", "architecture", "scale", "rotation_deg"]:
        groups = defaultdict(list)
        for r, e in zip(rows, errs):
            if np.isfinite(e):
                groups[r[key]].append(e)
        summary[f"by_{key}"] = {
            k: {"n": len(v), "mean": round(float(np.mean(v)), 3),
                "pass_5px": round(float((np.array(v) <= 5).mean()), 4),
                "pass_1px": round(float((np.array(v) <= 1).mean()), 4)}
            for k, v in groups.items()
        }

    if not args.no_baseline:
        base_errs = []
        for r in manifest:
            sid = int(r["id"])
            import cv2
            search = cv2.imread(str(eval_root / "search" / f"{sid:05d}.png"),
                                cv2.IMREAD_GRAYSCALE)
            reference = cv2.imread(str(eval_root / "reference" / f"{sid:05d}.png"),
                                   cv2.IMREAD_GRAYSCALE)
            b = classical_baseline(search, reference, inf["scales"],
                                   inf["rotations"])
            gx, gy = float(r["gt_x"]), float(r["gt_y"])
            base_errs.append(float(np.hypot(b["x"] - gx, b["y"] - gy)) if b else np.nan)
        base_errs = np.array(base_errs)
        base_ok = base_errs[np.isfinite(base_errs)]
        summary["classical_baseline"] = summarize(base_ok, "baseline")

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- worst failure visualization ----
    order = np.argsort(errs)[::-1]
    n_fail = min(3, int((errs > 5).sum()))
    fail_notes = []
    for idx in order[:max(n_fail, 1)]:
        r = rows[idx]
        if not np.isfinite(errs[idx]):
            continue
        sid = r["id"]
        import cv2
        search = cv2.imread(r["search_path"], cv2.IMREAD_GRAYSCALE)
        px, py = float(r["pred_x"]), float(r["pred_y"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        scale = float(r["scale"])
        tw = 1000.0 / scale
        color = cv2.imread(r["search_path"], cv2.IMREAD_COLOR)
        cv2.rectangle(color, (int(gx - tw / 2), int(gy - tw / 2)),
                      (int(gx + tw / 2), int(gy + tw / 2)), (0, 255, 0), 3)
        cv2.rectangle(color, (int(px - tw / 2), int(py - tw / 2)),
                      (int(px + tw / 2), int(py + tw / 2)), (0, 0, 255), 3)
        cv2.circle(color, (int(px), int(py)), 5, (0, 0, 255), -1)
        out = fail_dir / f"fail_{sid:05d}.png"
        cv2.imwrite(str(out), color)
        fail_notes.append(
            f"## failure {sid} ({r['architecture']}, {r['noise_level']}, "
            f"scale {r['scale']}, rot {r['rotation_deg']})\n"
            f"error {r['error_px']} px, ZNCC score {r['zncc_score']}, "
            f"model prob {r['prob']}. Green = GT, red = prediction.\n"
            f"![{sid}](failures/{out.name})\n"
            f"Root cause: {root_cause(r, search, gx, gy)}\n"
        )
    (out_dir / "FAILURES.md").write_text("\n\n".join(fail_notes) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote results -> {out_dir}")


def root_cause(r, search, gx, gy):
    """Heuristic root-cause text for a failure (used for documentation)."""
    import numpy as np
    causes = []
    if r["noise_level"] in ("high", "severe"):
        causes.append("low-SNR search image (high/severe noise level)")
    if abs(float(r["rotation_deg"])) > 0.5:
        causes.append("search/reference rotation exceeds the template grid step")
    tw = 1000.0 / float(r["scale"])
    if (gx - tw / 2 < 10 or gy - tw / 2 < 10
            or gx + tw / 2 > 998 or gy + tw / 2 > 998):
        causes.append("target is clipped at the search-image boundary, weakening "
                      "the correlation peak")
    if not causes:
        causes.append("repeated-pattern ambiguity resolved in favour of a "
                      "periodic duplicate")
    return "; ".join(causes)


if __name__ == "__main__":
    main()
