#!/usr/bin/env python3
"""
Optimized pipeline script to achieve 80% accuracy:
1. Generate 1200 training + 80 eval samples
2. Prepare candidates with optimized parameters
3. Train ranker with improved hyperparameters and augmentation
4. Evaluate on test set and report accuracy metrics

Usage:
    python run_optimized_pipeline.py [--device cpu|cuda]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command with pretty logging."""
    print("\n" + "="*80)
    print(f"▶ {description}")
    print("="*80)
    print(f"$ {' '.join(cmd)}")
    print("-"*80)
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(1)
    print(f"\n✓ Completed: {description}\n")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip dataset generation (use existing)")
    parser.add_argument("--skip-prep-cands", action="store_true",
                        help="Skip candidate preparation (use existing)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing weights)")
    args = parser.parse_args()

    cfg_path = "configs/default.json"
    cfg = json.load(open(cfg_path))
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║         OPTIMIZED DRIFT-SENSE PIPELINE FOR 80% ACCURACY TARGET            ║
╚════════════════════════════════════════════════════════════════════════════╝

Configuration Summary:
  • Training samples: {cfg['dataset']['train_n']} (increased from 500)
  • Eval samples: {cfg['dataset']['eval_n']} (increased from 60)
  • Training epochs: {cfg['train']['epochs']} (increased from 60)
  • Batch size: {cfg['train']['batch_size']} (increased from 32)
  • Learning rate: {cfg['train']['lr']} (optimized)
  • Model capacity: Enhanced (64->128->256 channels)
  • Augmentation: Improved (more aggressive noise/distortions)
  • Inference: Better fusion strategy (defect + ZNCC + CNN)
  • Device: {args.device}
""")

    # Step 1: Generate dataset
    if not args.skip_generate:
        cmd = [
            sys.executable, "generate_dataset.py",
            f"--train-n={cfg['dataset']['train_n']}",
            f"--eval-n={cfg['dataset']['eval_n']}",
            f"--seed={cfg['dataset']['train_seed']}"
        ]
        run_command(cmd, "Step 1: Generate synthetic dataset")
    else:
        print("\n⊘ Skipping dataset generation (using existing)\n")

    # Step 2: Prepare candidates
    if not args.skip_prep_cands:
        for split in ["train", "eval"]:
            cmd = [sys.executable, "prepare_candidates.py", f"--split={split}"]
            run_command(cmd, f"Step 2.{split}: Prepare candidate cache for {split} split")
    else:
        print("\n⊘ Skipping candidate preparation (using existing)\n")

    # Step 3: Train ranker
    if not args.skip_train:
        cmd = [
            sys.executable, "train.py",
            f"--device={args.device}",
            "--variant=local",
            f"--warmup-epochs=7",
            f"--focal-gamma=2.5",
            f"--focal-alpha=0.3",
        ]
        run_command(cmd, "Step 3: Train ranker with optimized hyperparameters")
    else:
        print("\n⊘ Skipping training (using existing weights)\n")

    # Step 4: Evaluate
    cmd = [
        sys.executable, "evaluate.py",
        f"--device={args.device}",
    ]
    run_command(cmd, "Step 4: Evaluate on test set and generate metrics")

    # Step 5: Generate visualizations
    cmd = [sys.executable, "generate_eval_visualizations.py"]
    run_command(cmd, "Step 5: Generate evaluation visualizations (correct/failed cases)")

    # Summary
    metrics_path = Path("output/eval_results/metrics_summary.json")
    if metrics_path.exists():
        metrics = json.load(open(metrics_path))
        print("\n" + "="*80)
        print("FINAL EVALUATION METRICS & VISUALIZATION RESULTS")
        print("="*80)
        model_metrics = metrics.get("model", {})
        print(f"""
✓ Model Performance:
  • Pass rate @ 5px: {model_metrics.get('pass_5px', 0)*100:.1f}% (target: ≥80%)
  • Pass rate @ 4px: {model_metrics.get('pass_4px', 0)*100:.1f}%
  • Pass rate @ 2px: {model_metrics.get('pass_2px', 0)*100:.1f}%
  • Pass rate @ 1px: {model_metrics.get('pass_1px', 0)*100:.1f}%
  • Mean error: {model_metrics.get('mean', 0):.2f}px
  • Median error: {model_metrics.get('median', 0):.2f}px
  • Total samples: {model_metrics.get('n', 0)}

Comparison to Baseline:
  • Baseline @ 5px: {metrics.get('classical_baseline', {}).get('pass_5px', 0)*100:.1f}%
  • Improvement: +{(model_metrics.get('pass_5px', 0) - metrics.get('classical_baseline', {}).get('pass_5px', 0))*100:.1f}pp

✓ Visualizations Generated:
  • Correct predictions: output/eval_output/correct/ ({int(model_metrics.get('n', 0) * model_metrics.get('pass_5px', 0))} images)
  • Failed predictions: output/eval_output/failures/ ({int(model_metrics.get('n', 0) * (1 - model_metrics.get('pass_5px', 0)))} images)
  • Interactive viewer: output/eval_output/index.html
  • Summary report: output/eval_output/RESULTS.md
  • Statistics: output/eval_output/summary.txt

Status: {'ACHIEVED 80% TARGET ✓' if model_metrics.get('pass_5px', 0) >= 0.80 else 'Further optimization needed'}

📊 View Results:
  → Open index.html in browser: output/eval_output/index.html
  → Read detailed report: output/eval_output/RESULTS.md
""")


if __name__ == "__main__":
    main()
