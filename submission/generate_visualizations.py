#!/usr/bin/env python3
"""Generate high-quality visualization images using matplotlib."""

import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EVAL_ROOT = "output/dataset/eval"
RESULTS_CSV = "output/eval_results/localize_results.csv"
SUCCESS_DIR = "results/success"
FAIL_DIR = "results/failures"
OVERLAY_DIR = "results/overlays"
IMG_DIR = "images"


def load_results():
    return list(csv.DictReader(open(RESULTS_CSV)))


def make_overlay(search, pred_x, pred_y, gt_x, gt_y, scale, error, label,
                 out_path, is_pass=True):
    color = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    tw = 1000.0 / scale

    cv2.rectangle(color,
                  (int(gt_x - tw / 2), int(gt_y - tw / 2)),
                  (int(gt_x + tw / 2), int(gt_y + tw / 2)),
                  (0, 255, 0), 2)
    cv2.circle(color, (int(gt_x), int(gt_y)), 4, (0, 255, 0), -1)

    pred_color = (255, 255, 0) if is_pass else (0, 0, 255)
    cv2.rectangle(color,
                  (int(pred_x - tw / 2), int(pred_y - tw / 2)),
                  (int(pred_x + tw / 2), int(pred_y + tw / 2)),
                  pred_color, 2)
    cv2.circle(color, (int(pred_x), int(pred_y)), 4, pred_color, -1)

    cv2.line(color, (int(gt_x), int(gt_y)), (int(pred_x), int(pred_y)),
             (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label} err={error:.1f}px"
    cv2.putText(color, text, (10, 25), font, 0.6, pred_color, 2)
    cv2.putText(color, f"Green=GT, {'Cyan' if is_pass else 'Red'}=Pred",
                (10, 50), font, 0.45, (200, 200, 200), 1)

    cv2.imwrite(out_path, color)


def make_combined_strip(ref, search, pred_x, pred_y, gt_x, gt_y, scale,
                        error, sid, label, out_path, is_pass=True):
    h, w = 1000, 1000
    tw = int(1000.0 / scale)

    ref_panel = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    search_color = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

    gt_x1, gt_y1 = max(0, int(gt_x - tw / 2)), max(0, int(gt_y - tw / 2))
    gt_x2, gt_y2 = min(w, int(gt_x + tw / 2)), min(h, int(gt_y + tw / 2))
    cv2.rectangle(search_color, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)

    pred_color = (255, 255, 0) if is_pass else (0, 0, 255)
    px1, py1 = max(0, int(pred_x - tw / 2)), max(0, int(pred_y - tw / 2))
    px2, py2 = min(w, int(pred_x + tw / 2)), min(h, int(pred_y + tw / 2))
    cv2.rectangle(search_color, (px1, py1), (px2, py2), pred_color, 2)
    cv2.circle(search_color, (int(pred_x), int(pred_y)), 5, pred_color, -1)

    crop_x1, crop_y1 = max(0, int(gt_x - 150)), max(0, int(gt_y - 150))
    crop_x2, crop_y2 = min(w, int(gt_x + 150)), min(h, int(gt_y + 150))
    zoom_crop = search[crop_y1:crop_y2, crop_x1:crop_x2]
    zoom_color = cv2.cvtColor(zoom_crop, cv2.COLOR_GRAY2BGR)
    scale_z = 300.0 / max(crop_x2 - crop_x1, crop_y2 - crop_y1)
    zgt_x, zgt_y = int((gt_x - crop_x1) * scale_z), int((gt_y - crop_y1) * scale_z)
    zpx, zpy = int((pred_x - crop_x1) * scale_z), int((pred_y - crop_y1) * scale_z)
    cv2.circle(zoom_color, (zgt_x, zgt_y), 6, (0, 255, 0), 2)
    cv2.circle(zoom_color, (zpx, zpy), 6, pred_color, 2)

    ref_p = cv2.resize(ref_panel, (300, 300))
    search_p = cv2.resize(search_color, (300, 300))
    zoom_p = cv2.resize(zoom_color, (300, 300))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, text in [(ref_p, "Reference (100x)"),
                        (search_p, "Search (10x) + Overlay"),
                        (zoom_p, "Zoomed Detail")]:
        cv2.putText(panel, text, (5, 22), font, 0.5, (0, 0, 0), 3)
        cv2.putText(panel, text, (5, 22), font, 0.5, (255, 255, 255), 1)

    footer = f"ID: {sid} | {label} | Error: {error:.1f}px"
    cv2.putText(search_p, footer, (5, 290), font, 0.38, (200, 200, 200), 1)

    strip = np.hstack([ref_p, search_p, zoom_p])
    cv2.imwrite(out_path, strip)


def generate_flowchart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7, 4.2, "Drift-Sense Localization Pipeline",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="#1a1a2e")

    stages = [
        ("Input", "Ref (100x) +\nSearch (10x)", "#e8f4fd", "#2196F3"),
        ("Adaptive\nDenoising", "MAD noise est.\nBilateral + CLAHE", "#e8f4fd", "#2196F3"),
        ("Multi-Scale\nZNCC", "9 scales × 5 rotations\n+ Phase Correlation", "#e8f4fd", "#2196F3"),
        ("Top-12\nCandidates", "NMS peaks\nscored & ranked", "#fff8e1", "#FF9800"),
        ("CNN\nRe-Ranker", "ResBlock + SE\n638K params", "#fce4ec", "#E91E63"),
        ("Defect\nResidue NCC", "FFT periodic removal\nUnique defect match", "#fce4ec", "#E91E63"),
        ("Noise-Aware\nFusion", "ZNCC + Defect + CNN\nweighted combo", "#f3e5f5", "#9C27B0"),
        ("Sub-pixel\nRefinement", "Two-pass parabolic\ninterpolation", "#f3e5f5", "#9C27B0"),
    ]

    n = len(stages)
    box_w, box_h = 1.35, 2.2
    gap = 0.33
    total_w = n * box_w + (n - 1) * gap
    x_start = (14 - total_w) / 2

    for i, (title, desc, bg, border) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        y = 0.8

        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.08",
                             facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(box)

        ax.text(x + box_w / 2, y + box_h * 0.68, title,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#1a1a2e")
        ax.text(x + box_w / 2, y + box_h * 0.28, desc,
                ha="center", va="center", fontsize=6.5, color="#555555",
                linespacing=1.4)

        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap * 0.9, y + box_h / 2),
                        xytext=(x + box_w + gap * 0.1, y + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#888888",
                                        lw=1.5, mutation_scale=14))

    ax.annotate("OUTPUT:  (x, y)", xy=(x_start + total_w + 0.25, 1.9),
                xytext=(x_start + total_w + 0.25, 1.9),
                fontsize=9, fontweight="bold", color="#2E7D32",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2E7D32", lw=1.5))

    cat_labels = [
        (x_start + 0.5, 3.3, "Classical CV", "#1565C0"),
        (x_start + 3.5, 3.3, "+ Phase Corr.", "#1565C0"),
        (x_start + 6.3, 3.3, "Deep Learning", "#AD1457"),
        (x_start + 9.3, 3.3, "Hybrid Fusion", "#6A1B9A"),
    ]
    for cx, cy, txt, c in cat_labels:
        ax.text(cx, cy, txt, fontsize=7, color=c, fontstyle="italic", ha="center")

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(IMG_DIR, "pipeline_flowchart.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {IMG_DIR}/pipeline_flowchart.png")


def generate_accuracy_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    thresholds = ["@ 5px", "@ 4px", "@ 2px", "@ 1px", "@ 0.5px"]
    model_rates = [66.7, 66.7, 60.0, 35.0, 11.7]
    base_rates = [46.7, 46.7, 41.7, 23.3, 8.3]

    x = np.arange(len(thresholds))
    w = 0.35

    bars1 = ax.bar(x - w / 2, model_rates, w, label="Drift-Sense (v2)",
                   color="#1976D2", edgecolor="#0D47A1", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + w / 2, base_rates, w, label="ZNCC Baseline",
                   color="#BDBDBD", edgecolor="#757575", linewidth=0.8, zorder=3)

    for bar, val in zip(bars1, model_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#0D47A1")
    for bar, val in zip(bars2, base_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                color="#616161")

    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, fontsize=11)
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_title("Localization Pass Rate by Error Threshold", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 82)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "accuracy_chart.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {IMG_DIR}/accuracy_chart.png")


def generate_noise_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    levels = ["Low\n(n=7)", "Medium\n(n=15)", "High\n(n=13)", "Severe\n(n=25)"]
    rates = [85.7, 80.0, 84.6, 44.0]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    bars = ax.bar(levels, rates, color=colors, edgecolor=["#2E7D32", "#1565C0", "#E65100", "#B71C1C"],
                  linewidth=1.0, width=0.55, zorder=3)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#212121")

    ax.axhline(y=80, color="#B71C1C", linestyle="--", linewidth=1, alpha=0.7, zorder=2)
    ax.text(3.35, 81, "80% target", fontsize=8, color="#B71C1C", ha="right")

    ax.set_ylabel("Pass Rate @ 5px (%)", fontsize=12)
    ax.set_title("Accuracy by SEM Noise Level", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "noise_chart.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {IMG_DIR}/noise_chart.png")


def generate_results_summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    passed, failed = [], []
    for r in load_results():
        err = float(r["error_px"])
        (passed if err <= 5.0 else failed).append(r)
    passed.sort(key=lambda r: float(r["error_px"]))
    failed.sort(key=lambda r: -float(r["error_px"]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=120)
    fig.suptitle("Results Summary  —  Top 4 Successes (top) / Top 4 Failures (bottom)",
                 fontsize=14, fontweight="bold", y=0.98)

    for col, r in enumerate(passed[:4]):
        sid = int(r["id"])
        path = os.path.join(OVERLAY_DIR, f"overlay_{sid:05d}.png")
        if os.path.exists(path):
            img = mpimg.imread(path)
            axes[0, col].imshow(img)
        err = float(r["error_px"])
        axes[0, col].set_title(f"#{sid}  {err:.1f}px  ✓", fontsize=9,
                               color="#1B5E20", fontweight="bold")
        axes[0, col].axis("off")

    for col, r in enumerate(failed[:4]):
        sid = int(r["id"])
        path = os.path.join(OVERLAY_DIR, f"overlay_fail_{sid:05d}.png")
        if os.path.exists(path):
            img = mpimg.imread(path)
            axes[1, col].imshow(img)
        err = float(r["error_px"])
        axes[1, col].set_title(f"#{sid}  {err:.1f}px  ✗", fontsize=9,
                               color="#B71C1C", fontweight="bold")
        axes[1, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(IMG_DIR, "results_summary.png"),
                dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {IMG_DIR}/results_summary.png")


def main():
    os.makedirs(SUCCESS_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    rows = load_results()

    passed, failed = [], []
    for r in rows:
        err = float(r["error_px"])
        (passed if err <= 5.0 else failed).append(r)
    passed.sort(key=lambda r: float(r["error_px"]))
    failed.sort(key=lambda r: -float(r["error_px"]))

    print(f"Generating success case overlays ({len(passed)} passed)...")
    for r in passed[:10]:
        sid = int(r["id"])
        search = cv2.imread(os.path.join(EVAL_ROOT, "search", f"{sid:05d}.png"), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(EVAL_ROOT, "reference", f"{sid:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if search is None or ref is None:
            continue
        px, py = float(r["pred_x"]), float(r["pred_y"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        scale = float(r["scale"])
        err = float(r["error_px"])
        make_overlay(search, px, py, gx, gy, scale, err,
                     f"PASS {r['architecture']} {r['noise_level']}",
                     os.path.join(SUCCESS_DIR, f"success_{sid:05d}.png"), is_pass=True)
        make_combined_strip(ref, search, px, py, gx, gy, scale, err,
                            sid, f"{r['architecture']} {r['noise_level']}",
                            os.path.join(OVERLAY_DIR, f"overlay_{sid:05d}.png"), is_pass=True)

    print(f"Generating failure case overlays ({len(failed)} failed)...")
    for r in failed[:10]:
        sid = int(r["id"])
        search = cv2.imread(os.path.join(EVAL_ROOT, "search", f"{sid:05d}.png"), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(EVAL_ROOT, "reference", f"{sid:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if search is None or ref is None:
            continue
        px, py = float(r["pred_x"]), float(r["pred_y"])
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        scale = float(r["scale"])
        err = float(r["error_px"])
        make_overlay(search, px, py, gx, gy, scale, err,
                     f"FAIL {r['architecture']} {r['noise_level']}",
                     os.path.join(FAIL_DIR, f"failure_{sid:05d}.png"), is_pass=False)
        make_combined_strip(ref, search, px, py, gx, gy, scale, err,
                            sid, f"{r['architecture']} {r['noise_level']}",
                            os.path.join(OVERLAY_DIR, f"overlay_fail_{sid:05d}.png"), is_pass=False)

    generate_accuracy_chart()
    generate_noise_chart()
    generate_flowchart()
    generate_results_summary()

    print(f"\nDone! Generated:")
    print(f"  {len(os.listdir(SUCCESS_DIR))} success images in {SUCCESS_DIR}/")
    print(f"  {len(os.listdir(FAIL_DIR))} failure images in {FAIL_DIR}/")
    print(f"  {len(os.listdir(OVERLAY_DIR))} overlay images in {OVERLAY_DIR}/")
    print(f"  4 chart images in {IMG_DIR}/")


if __name__ == "__main__":
    main()
