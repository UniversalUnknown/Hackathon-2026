# Drift-Sense: AI-Powered Navigation Error Recovery for Wafer Inspection Tools

Given a **1000×1000 100× reference** image and a **1000×1000 10× search** image
(~10:1 scale, ±2° rotation, SEM noise), find the precise centre `(x, y)` of the
reference pattern inside the search image.

---

## Pipeline Overview

**8 stages**: Input → Adaptive Denoising → Multi-Scale ZNCC (9 scales × 5 rotations = 45 variants)
→ Phase Correlation Seed → Top-12 Candidates → CNN Re-Ranker (ResBlock+SE, 638K params)
→ Defect Residue NCC → Noise-Aware Fusion + Sub-Pixel Refinement

---

## Results

| Metric | Model | Classical Baseline | Improvement |
|--------|-------|-------------------|-------------|
| **Pass Rate @ 5px** | **66.7%** (40/60) | 46.7% (28/60) | **+20.0pp** |
| Pass Rate @ 2px | 60.0% (36/60) | 41.7% | +18.3pp |
| Pass Rate @ 1px | 35.0% (21/60) | 23.3% | +11.7pp |
| Mean Error | 61.9 px | 95.5 px | −35% |
| Median Error | 1.5 px | 22.4 px | −93% |

### Performance by Noise Level

| Noise | Samples | Pass @ 5px | Mean Error |
|-------|---------|------------|------------|
| Low | 7 | **85.7%** | 2.0 px |
| Medium | 15 | **80.0%** | 44.9 px |
| High | 13 | **84.6%** | 5.5 px |
| Severe | 25 | **44.0%** | 118.2 px |

---

## Success Cases (green = ground truth, cyan = prediction)

<img src="submission/images/success_00039.png" width="300" alt="00039 — 0.35px"/>

<img src="submission/images/success_00043.png" width="300" alt="00043 — 0.38px"/>

---

## Failure Cases

<img src="submission/images/failure_00035.png" width="300" alt="00035"/>

<img src="submission/images/failure_00045.png" width="300" alt="00045"/>

<img src="submission/images/overlay_fail_00045.png" width="300" alt="overlay 00045"/>

<img src="submission/images/overlay_fail_00055.png" width="300" alt="overlay 00055"/>

**Root causes**: periodic pattern ambiguity under severe noise (44% of failures are severe-noise cases where all ZNCC peaks are random).

---

## How to Run

```bash
cd submission
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate dataset
python generate_dataset.py --train-n 500 --eval-n 60 --root output/dataset

# Precompute candidates
python prepare_candidates.py --split train
python prepare_candidates.py --split eval

# Materialize CNN inputs
python prep_inputs.py --split train --variant global

# Train the ranker
python train.py --config configs/default.json --epochs 50 --variant global

# Test one pair
python localize.py \
  --search output/dataset/eval/search/00000.png \
  --reference output/dataset/eval/reference/00000.png \
  --weights output/weights/ranker.pt

# Full evaluation
python evaluate.py --config configs/default.json --weights output/weights/ranker.pt
```

---

## Project Structure

```
submission/
├── README.md, requirements.txt, generate_dataset.py, localize.py
├── evaluate.py, train.py, prepare_candidates.py, prep_inputs.py
├── configs/default.json
├── src/                        ← synthetic data generation
│   ├── generate.py, sem_imaging.py, presets.py, structural_defects.py
│   └── patterns/{dram,finfet,zones}.py
├── model/                      ← ML localization
│   ├── predict.py, zncc.py, ranker.py, preprocess.py, dataset.py
├── results/                    ← eval outputs + overlay images
│   ├── success/, failures/, overlays/
│   ├── metrics_summary.json, localize_results.csv
├── images/                     ← result and overlay samples
│   ├── success_00039.png, success_00043.png
│   ├── failure_00035.png, failure_00045.png
│   └── overlay_fail_00045.png, overlay_fail_00055.png
└── references/references.md
```

---

## Dependencies

```
numpy>=1.26  |  opencv-python-headless>=4.8  |  scipy>=1.10  |  torch>=2.0
```

---

Due to hardware and time limitations we couldn't push this model further.