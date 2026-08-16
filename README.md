# Drift-Sense: How To Run And Test Locally

This document explains everything you need to create the environment, install
the dependencies, generate a test dataset, train the model, run the
localization on the test pairs **in a loop**, and what the algorithm does with
its current measured efficiency.

---

## 1. What the project does (30-second version)

Given a **100x reference** image and a **10x search** image (both 1000x1000 px),
find the target centre `(x, y)` of the reference inside the search image. The
search can be scaled 9:1-11:1 vs. the reference, rotated by up to ~2 deg, and
degraded by realistic SEM noise.

---

## 2. Create a Python environment (venv)

Python 3.10+ is required. Open a terminal in the `submission/` folder:

```bash
cd submission

# create the virtual environment
python -m venv .venv

# activate it
# Linux / macOS:
source .venv/bin/activate
# Windows (cmd):
# .venv\Scripts\activate.bat
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# upgrade pip (optional but recommended)
python -m pip install --upgrade pip
```

> Note: on some Linux distros the system Python is "externally managed"
> (PEP 668). The venv above avoids that problem entirely.

### What to install

Everything is CPU-only (no GPU needed). From inside the activated venv:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```
numpy>=1.26
opencv-python-headless>=4.8
scipy>=1.10
torch>=2.0
```

Or install them directly:

```bash
pip install numpy opencv-python-headless scipy torch
```

Verify the install:

```bash
python -c "import numpy, cv2, scipy, torch; print('ok', torch.__version__)"
```

---

## 3. Files to run (in order)

Run all of these from inside the `submission/` folder with the venv active.

| Step | File | What it does | Typical time |
|------|------|--------------|--------------|
| 1 | `generate_dataset.py` | Builds synthetic SEM pairs (train + eval) + `manifest.csv` | ~1.5 s/pair |
| 2 | `prepare_candidates.py` | Classical ZNCC matching → top-8 candidates per pair (cached `.npz`) | ~0.5 s/pair |
| 3 | `prep_inputs.py` | Pre-bakes CNN input tensors (fast training) | ~2-3 min |
| 4 | `train.py` | Trains the CNN re-ranker → `output/weights/ranker.pt` | ~34 s/epoch |
| 5 | `localize.py` | Single-pair inference; prints `x y` | ~0.5 s |
| 6 | `evaluate.py` | Full rubric on the held-out eval set + failure figures | ~1-2 min |

### Step 1 — generate the test dataset

```bash
python generate_dataset.py --train-n 1000 --eval-n 60 --root output/dataset
```

- Writes images to `output/dataset/{train,eval}/{search,reference}/NNNNN.png`
- Writes the metadata manifest to `output/dataset/{split}/manifest.csv`
  (contains `gt_x, gt_y, scale, rotation_deg, architecture, noise_level`, all
  generator parameters, and the per-pair RNG `seed`)

For a *quick smoke test* (recommended first run) use small numbers:

```bash
python generate_dataset.py --train-n 50 --eval-n 10 --root output/dataset
```

### Step 2 — precompute candidates

```bash
python prepare_candidates.py --split train
python prepare_candidates.py --split eval
```

### Step 3 — materialize CNN inputs

```bash
python prep_inputs.py --split train --variant global
```

### Step 4 — train the re-ranker

```bash
python train.py --config configs/default.json --epochs 30 --variant global
```

Watch the `val_top1_hit_pos` line in the log (fraction of pairs with a
learnable candidate where the model ranks the true location #1). A checkpoint
is saved automatically whenever validation improves.

### Step 5 — test one pair

```bash
python localize.py \
  --search output/dataset/eval/search/00000.png \
  --reference output/dataset/eval/reference/00000.png \
  --weights output/weights/ranker.pt
```

Output (stdout): the predicted target centre in search-image pixels, e.g.:

```
243.300 583.700
```

Exit code `0` = success, `1` = failure (message on stderr).

---

## 4. Running the test in a loop

### Option A — the whole loop is `evaluate.py`

This is the intended way: it loops over **all** eval pairs, computes the error
against the ground truth, and writes results + metrics:

```bash
python evaluate.py --config configs/default.json --weights output/weights/ranker.pt
```

Results land in `output/eval_results/`:
- `localize_results.csv` — every pair, true vs. predicted, error, metadata
- `metrics_summary.json` — pass rates, error stats, breakdowns, runtime
- `failures/*.png` + `FAILURES.md` — worst cases with root-cause notes

### Option B — a manual bash loop over `localize.py`

If you want to run the CLI in a loop yourself (e.g. to test your own image
folder), put pairs in a directory and loop:

```bash
for f in output/dataset/eval/search/*.png; do
  id=$(basename "$f" .png)
  echo "--- sample $id ---"
  python localize.py \
    --search "output/dataset/eval/search/$id.png" \
    --reference "output/dataset/eval/reference/$id.png" \
    --weights output/weights/ranker.pt
done
```

To capture results into a CSV while looping:

```bash
out=results.csv
echo "id,pred_x,pred_y" > "$out"
for f in output/dataset/eval/search/*.png; do
  id=$(basename "$f" .png)
  xy=$(python localize.py --search "output/dataset/eval/search/$id.png" \
        --reference "output/dataset/eval/reference/$id.png" \
        --weights output/weights/ranker.pt)
  echo "$id,$xy" >> "$out"
done
```

### Option C — loop over arbitrary pairs (your own images)

Keep your test images in two folders `my_search/` and `my_reference/` with the
same base names and loop the same way. There is no need for a manifest — the
ground truth is only required for scoring, not for running.

---

## 5. How it works (plain English)

The task: you give the program a small **reference** photo of the sample and a
big **search** photo. It finds the one spot in the search photo that shows the
same thing as the reference, even when the search photo is scaled ~10x,
rotated a couple of degrees, and noisy. It returns the `(x, y)` of that spot.

The program does this in five steps:

1. **Clean the noise.** The search photo gets a gentle 3x3 median filter,
   which smooths away the speckle noise so the true pattern shows up more
   clearly. (Measured: this alone fixed 6 of the 60 test images.)

2. **Find promising spots (classical search).** It compares the reference
   against every position in the search photo. Because the scale and rotation
   are only known approximately, it tries a small grid: 5 scales (9.0-11.0)
   x 3 rotations (-2, 0, +2 degrees). It keeps the 8 best, well-separated
   matches ("candidates"), each with the scale/rotation that scored best.

3. **Check the defects (the clever part).** These patterns are periodic, so
   many spots look identical. But each spot also contains *unique* details --
   the little defects and irregularities. The program takes a fast Fourier
   transform of the reference and of each candidate's surroundings, subtracts
   the repeating background, and compares only what is left (the "residue").
   The true spot shares the reference's unique details; look-alike copies do
   not. This is what tells one box apart from all the identical boxes.

4. **Pick the winner.** The winning spot is the one with the best combination
   of the plain match score and the defect check (measured: this raised the
   pass rate from ~45% to ~63%). A CNN also scores each spot using the whole
   scene and its probability is reported as confidence.

5. **Zoom in to sub-pixel accuracy.** Around the winner it re-matches the
   template in a tiny window and fits a parabola to the score peak, giving a
   final `(x, y)` accurate to well under a pixel when the right spot was found.

**The ML part.** The CNN re-ranker (`ml/ranker.py`) is a small convolutional
network trained on our generated pairs to answer "is this candidate the true
site?" It looks at the whole scene (search image, a marker at the candidate,
and the reference pattern) because repeated patterns look identical up close.
Its probability is reported as confidence for every candidate and can break
near-ties (`cnn_tie` mode).

---

## 6. Current efficiency (measured on this machine, CPU)

Hardware note: results below are for a laptop-class CPU (single-threaded,
no GPU). Times scale with cores when you run multiple jobs.

**Speed**

| Operation | Time |
|-----------|------|
| Pair generation | ~1.5 s/pair |
| Candidate computation (ZNCC, 5 scales x 3 rotations) | ~0.5 s/pair |
| Full localization (candidates + ranker + refine) | ~0.5 s/pair |
| Ranker training | ~34 s/epoch (900 samples, batch 32) |

**Accuracy** (measured on the 60 held-out eval pairs, no training-set overlap)

| Metric | Value |
|--------|-------|
| True site among the top-8 candidates (recall, with noise filter) | ~83% (50/60) |
| Final answer within 5 px of truth (main pass rate) | **63.3%** (38/60) |
| Final answer within 1 px (sub-pixel hits) | 30% (18/60) |
| Average error across all 60 | ~67 px |
| Same pipeline **without** the defect check (previous version) | 56.7% (34/60) |
| Same pipeline without noise filter or defect check (original) | 45% (27/60) |
| Sub-pixel refinement on the chosen candidate | < 2 px error typical |

What the numbers mean: for 38 of the 60 test images the final answer lands
within 5 px of the true target centre, and for 18 of them within 1 px. Two
measured improvements got us here: the noise filter (+11.7 points) and the
FFT defect check (+6.6 points).

Important caveat: for the images that are still missed, the target is either
buried so deep in noise that no peak survives, or the pattern is perfectly
periodic so several spots are statistically identical. No method can tell
these apart from the images alone; they are the documented failure cases in
`output/eval_results/FAILURES.md`.

---

## 7. Config reference

Everything tunable lives in `configs/default.json`:

- `dataset`: number of train/eval pairs, seed, architectures, scale and
  rotation options and their probabilities.
- `train`: epochs, batch size, learning rate, positive margin (px), candidate
  count K, checkpoint path, validation split.
- `infer`: the scale/rotation grid, candidate count, noise-filter size
  (`prefilter_median`), the selection rule (`fuse`/`zncc`/`cnn`/`cnn_tie`)
  and its `defect_weight`, score thresholds, tie-break epsilon, output CSV
  path.

---

## 8. Troubleshooting

- `ModuleNotFoundError: numpy/cv2/torch` → the venv is not active, or you
  forgot `pip install -r requirements.txt`.
- `Missing output/dataset/.../manifest.csv` → run `generate_dataset.py` first.
- `No candidate caches found` → run `prepare_candidates.py --split train`
  (and `--split eval`) first.
- `cannot read search image` → wrong path; `localize.py` exits with code 1.
- PEP 668 "externally managed environment" error on `pip install` → you are
  not using the venv; activate it first.
