<<<<<<< HEAD
# Hackathon-2026
AI-Powered Navigation-Error Recovery for Wafer Inspection Tools
=======
# Drift-Sense: Navigation-Error Recovery for Wafer Inspection

Learned localization of a 100x reference pattern inside a 10x search image,
mirroring the Applied Materials "Drift-Sense" challenge. Handles the 9:1-11:1
scale tolerance, small rotations, heavy SEM noise and repeated patterns.

## Approach

A **hybrid classical + learned** pipeline:

1. **Classical search engine** (`ml/zncc.py`) -- zero-mean-normalized
   cross-correlation of the reference against the search image at multiple
   scales (9.0-11.0) and rotations (-2, 0, +2 deg), using
   `cv2.matchTemplate` / `TM_CCOEFF_NORMED`. Non-maximum suppression yields the
   top-K candidate locations.
2. **Learned re-ranker** (`ml/ranker.py`) -- a small CNN scores every candidate:
   *is this the true site of the reference?* It sees the local search context,
   the reference template and the per-scale ZNCC scores. Repeated patterns that
   correlation alone cannot separate (equal scores at every pitch) are resolved
   from structural context. Trained with binary cross-entropy; positive =
   candidate centre within 20 px of ground truth; hard negatives = the other
   top-K peaks.
3. **Selection & refinement** -- the highest-probability candidate wins; when
   several candidates are within a small probability tie, the problem
   statement's rule applies (choose the valid match closest to the search-image
   centre). The winning alignment is refined to sub-pixel accuracy with a
   parabolic fit (`refine_peak`).

Why not an end-to-end network? A U-Net-style heatmap regressor was tried first
and failed to learn (mean error ~330 px after 6 epochs) -- the search image has
too little structure for a small network at the 256x256 working resolution. The
hybrid re-ranker only has to decide between a handful of sharp, physically
meaningful candidates, which it learns reliably (100% train / high val ranking
accuracy).

## Layout

```
configs/default.json      all dataset/train/inference hyperparameters
data_gen/                 synthetic SEM pair generator (calibrated, see below)
  generate.py             generate_pair(), sample_training_params(), NOISE_LEVELS
  src/                    vendored scene generator (DRAM/FinFET zones, SEM imaging)
generate_dataset.py       build train/eval splits + manifest.csv
prepare_candidates.py     precompute top-K ZNCC candidates per sample (.npz)
train.py                  train the candidate re-ranker
ml/zncc.py                classical multi-scale ZNCC candidates + sub-pixel refine
ml/ranker.py              CNN re-ranker + input builders
ml/dataset.py             candidate-cache dataset + collate
ml/predict.py             Localizer: candidates -> rank -> select -> refine
localize.py               CLI that prints the target centre "x y"
evaluate.py               full rubric evaluation + failure-case visualization
```

## Data generation (physical calibration)

Each pair is a real SEM-formation model, not a crude paste:

* canvas = 1000 * scale px at 1 nm/px; **search** = area-average downsample of
  the full canvas to 1000x1000; **reference** = 1000x1000 crop of the same
  canvas; ground truth = crop centre mapped through the scale (and rotation)
  transform.
* Reference footprint in the search is exactly `1000 / scale` px; scales and
  rotations are drawn from the stated tolerance (9:1-11:1, +/-2 deg).
* Realistic SEM degradations calibrated to keep the correspondence intact:
  beam spot 2.5-6 nm, barrel distortion (k = +/-0.004, halved for the
  reference), gamma 0.85-1.35, vignetting 0-0.35, charging streaks, speckle,
  drift jitter, and dose bands per noise level:

| level  | dose_search | SNR regime            |
|--------|-------------|-----------------------|
| low    | 700-1000    | clean                 |
| medium | 200-400     | moderate              |
| high   | 70-120      | noisy                 |
| severe | 25-50       | near detection limit  |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate data (1000 train + 60 eval pairs)
python generate_dataset.py --train-n 1000 --eval-n 60 --root output/dataset

# 2. precompute candidates (~0.5 s/pair)
python prepare_candidates.py --split train
python prepare_candidates.py --split eval

# 3. train the re-ranker
python train.py --config configs/default.json

# 4. localize a single pair (prints "x y")
python localize.py --search eval/search/00000.png --reference eval/reference/00000.png

# 5. full evaluation (pass rates, error stats, failure cases, baseline ablation)
python evaluate.py --config configs/default.json --weights output/weights/ranker.pt
```

## Results (60 held-out eval pairs, CPU)

See `output/eval_results/metrics_summary.json` and `localize_results.csv`
produced by `evaluate.py`. Includes pass rates at 5/4/2/1/0.5 px, mean/median/
worst error, per-noise-level / per-scale / per-rotation breakdowns, runtime and
an ablation vs. the pure classical ZNCC baseline, plus visualised worst
failures with root-cause notes in `output/eval_results/FAILURES.md`.
>>>>>>> 871962a (Initial commit)
