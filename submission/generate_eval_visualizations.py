#!/usr/bin/env python3
"""
Generate detailed evaluation visualizations showing correct/incorrect predictions.

Creates high-quality images showing:
  - Search image with ground truth (green) and predicted (blue/red) locations
  - Reference image context
  - Localization error in pixels
  - Pass/Fail status

Output structure:
  output/eval_output/
  ├── correct/          (64 passed cases)
  │   ├── 00000_PASS.png
  │   ├── 00001_PASS.png
  │   └── ...
  ├── failures/         (16 failed cases)
  │   ├── 00030_FAIL.png
  │   ├── 00031_FAIL.png
  │   └── ...
  ├── summary.txt       (Statistical summary)
  ├── index.html        (Interactive viewer)
  └── RESULTS.md        (Detailed report)

Usage:
    python generate_eval_visualizations.py [--max-samples 80]
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


def load_eval_data():
    """Load evaluation results and image data."""
    results_dir = Path("output/eval_results")
    eval_root = Path("output/dataset/eval")
    
    # Load metrics
    metrics = json.load(open(results_dir / "metrics_summary.json"))
    
    # Load per-image results
    manifest = list(csv.DictReader(open(eval_root / "manifest.csv")))
    localize_results = list(csv.DictReader(open(results_dir / "localize_results.csv")))
    
    # Create ID -> result mapping
    results_map = {}
    for row in localize_results:
        results_map[int(row["id"])] = row
    
    return metrics, manifest, results_map, eval_root


def create_visualization(sample_id, manifest_row, result_row, eval_root, output_dir):
    """Create a detailed visualization for a single sample."""
    
    # Load images
    search_path = eval_root / "search" / f"{sample_id:05d}.png"
    ref_path = eval_root / "reference" / f"{sample_id:05d}.png"
    
    search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    
    if search is None or reference is None:
        return None
    
    # Extract data
    gt_x = float(manifest_row["gt_x"])
    gt_y = float(manifest_row["gt_y"])
    pred_x = float(result_row["pred_x"])
    pred_y = float(result_row["pred_y"])
    error = float(result_row["error_px"])
    passed = float(result_row["error_px"]) <= 5.0
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8), dpi=100)
    
    # Left: Reference image
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(reference, cmap='gray')
    ax1.set_title("Reference Image (100x magnification)\nTarget template", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right: Search image with annotations
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(search, cmap='gray')
    
    # Draw ground truth (green circle)
    gt_circle = Circle((gt_x, gt_y), radius=20, fill=False, edgecolor='lime', linewidth=3, label='Ground Truth')
    ax2.add_patch(gt_circle)
    ax2.plot(gt_x, gt_y, 'g+', markersize=20, markeredgewidth=3)
    
    # Draw prediction
    if passed:
        # Correct prediction (blue)
        pred_circle = Circle((pred_x, pred_y), radius=15, fill=False, edgecolor='cyan', linewidth=2.5, linestyle='--', label='Predicted (Correct)')
        color = 'cyan'
    else:
        # Wrong prediction (red)
        pred_circle = Circle((pred_x, pred_y), radius=15, fill=False, edgecolor='red', linewidth=3, label='Predicted (Wrong)')
        color = 'red'
    
    ax2.add_patch(pred_circle)
    ax2.plot(pred_x, pred_y, marker='x', color=color, markersize=20, markeredgewidth=3)
    
    # Draw error line
    if not passed:
        ax2.plot([gt_x, pred_x], [gt_y, pred_y], 'r--', linewidth=2, alpha=0.6, label=f'Error: {error:.2f}px')
    
    # Title and status
    status = "> PASS" if passed else "X FAIL"
    status_color = 'green' if passed else 'red'
    title_text = f"Search Image (10x magnification) - {status}\nError: {error:.2f}px"
    ax2.set_title(title_text, fontsize=14, fontweight='bold', color=status_color)
    ax2.axis('off')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add metadata
    metadata_text = (
        f"Sample ID: {sample_id:05d}\n"
        f"Architecture: {manifest_row.get('architecture', 'N/A')}\n"
        f"Noise Level: {manifest_row.get('noise_level', 'N/A')}\n"
        f"Scale: {manifest_row.get('scale', 'N/A')}x\n"
        f"Rotation: {manifest_row.get('rotation_deg', 'N/A')}°\n"
        f"GT Location: ({gt_x:.1f}, {gt_y:.1f})\n"
        f"Pred Location: ({pred_x:.1f}, {pred_y:.1f})\n"
        f"Error: {error:.2f}px"
    )
    fig.text(0.02, 0.98, metadata_text, transform=fig.transFigure, 
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall status box
    status_box_text = f"STATUS: {status}\n{'Accuracy: +/-5.0px' if passed else f'Error exceeds 5.0px'}"
    fig.text(0.98, 0.02, status_box_text, transform=fig.transFigure,
             fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             fontweight='bold', family='monospace',
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    plt.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=80,
                       help="Maximum samples to visualize")
    parser.add_argument("--no-html", action="store_true",
                       help="Skip HTML index generation")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\n> Loading evaluation data...")
    metrics, manifest, results_map, eval_root = load_eval_data()
    
    model_metrics = metrics.get("model", {})
    n_samples = model_metrics.get("n", 60)
    pass_5px = model_metrics.get("pass_5px", 0.0)
    
    print(f"  - Total samples: {n_samples}")
    print(f"  - Pass rate @ 5px: {pass_5px*100:.1f}%")
    print(f"  - Passed cases: {int(n_samples * pass_5px)}")
    print(f"  - Failed cases: {int(n_samples * (1 - pass_5px))}")
    
    # Create output directories
    output_base = Path("output/eval_output")
    correct_dir = output_base / "correct"
    failures_dir = output_base / "failures"
    
    correct_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n> Creating visualizations in {output_base}/")
    
    # Generate visualizations
    passed_count = 0
    failed_count = 0
    passed_list = []
    failed_list = []
    
    for i, manifest_row in enumerate(manifest[:args.max_samples]):
        sample_id = int(manifest_row["id"])
        
        if sample_id not in results_map:
            print(f"  ⚠ Skipping sample {sample_id} (no results)")
            continue
        
        result_row = results_map[sample_id]
        error = float(result_row["error_px"])
        passed = error <= 5.0
        
        # Create visualization
        fig = create_visualization(sample_id, manifest_row, result_row, eval_root, output_base)
        
        if fig is None:
            print(f"  X Failed to create viz for sample {sample_id}")
            continue
        
        # Save visualization
        if passed:
            output_path = correct_dir / f"{sample_id:05d}_PASS.png"
            passed_count += 1
            passed_list.append({
                "id": sample_id,
                "error": error,
                "architecture": manifest_row.get("architecture"),
                "noise": manifest_row.get("noise_level")
            })
            status = "> PASS"
        else:
            output_path = failures_dir / f"{sample_id:05d}_FAIL.png"
            failed_count += 1
            failed_list.append({
                "id": sample_id,
                "error": error,
                "architecture": manifest_row.get("architecture"),
                "noise": manifest_row.get("noise_level")
            })
            status = "X FAIL"
        
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        progress = f"[{i+1}/{args.max_samples}]"
        print(f"  {progress} {status} - Sample {sample_id:05d}: error={error:6.2f}px -> {output_path.name}")
    
    print(f"\n> Visualization generation complete!")
    print(f"  - Correct predictions: {passed_count} -> {correct_dir}")
    print(f"  - Failed predictions: {failed_count} -> {failures_dir}")
    
    # Generate summary statistics
    print(f"\n> Generating summary report...")
    
    summary_text = _generate_summary(
        metrics, passed_count, failed_count, passed_list, failed_list
    )
    
    summary_path = output_base / "RESULTS.md"
    summary_path.write_text(summary_text)
    print(f"  > Summary: {summary_path}")
    
    # Generate text summary
    stats_text = _generate_stats(passed_list, failed_list, output_base)
    stats_path = output_base / "summary.txt"
    stats_path.write_text(stats_text)
    print(f"  > Statistics: {stats_path}")
    
    # Generate HTML index
    if not args.no_html:
        print(f"\n> Generating interactive HTML index...")
        html_content = _generate_html_index(output_base, passed_count, failed_count, metrics)
        html_path = output_base / "index.html"
        html_path.write_text(html_content)
        print(f"  > HTML index: {html_path}")
        print(f"\n  -> Open in browser: {html_path.absolute()}")
    
    print("\n" + "="*80)
    print(f"RESULTS SAVED TO: {output_base.absolute()}")
    print("="*80 + "\n")


def _generate_summary(metrics, passed_count, failed_count, passed_list, failed_list):
    """Generate markdown summary report."""
    
    model_metrics = metrics.get("model", {})
    baseline_metrics = metrics.get("classical_baseline", {})
    
    summary = f"""# Evaluation Visualization Results

## Overview

**Evaluation Date**: 2026-08-17  
**Total Samples**: {passed_count + failed_count}  
**Passed (±5px)**: {passed_count} ({passed_count/(passed_count+failed_count)*100:.1f}%)  
**Failed**: {failed_count} ({failed_count/(passed_count+failed_count)*100:.1f}%)  

---

## Performance Metrics

### Model Performance
- **Pass rate @ 5px**: {model_metrics.get('pass_5px', 0)*100:.1f}%
- **Pass rate @ 4px**: {model_metrics.get('pass_4px', 0)*100:.1f}%
- **Pass rate @ 2px**: {model_metrics.get('pass_2px', 0)*100:.1f}%
- **Pass rate @ 1px**: {model_metrics.get('pass_1px', 0)*100:.1f}%
- **Mean error**: {model_metrics.get('mean', 0):.2f}px
- **Median error**: {model_metrics.get('median', 0):.2f}px
- **Worst error**: {model_metrics.get('worst', 0):.2f}px

### Comparison to Baseline
- **Baseline @ 5px**: {baseline_metrics.get('pass_5px', 0)*100:.1f}%
- **Model improvement**: +{(model_metrics.get('pass_5px', 0) - baseline_metrics.get('pass_5px', 0))*100:.1f}pp

---

## Breakdown by Noise Level

"""
    
    by_noise = metrics.get("by_noise_level", {})
    for noise_level, noise_data in by_noise.items():
        summary += f"""### {noise_level.upper()}
- Samples: {noise_data.get('n', 0)}
- Pass rate: {noise_data.get('pass_5px', 0)*100:.1f}%
- Mean error: {noise_data.get('mean', 0):.2f}px

"""
    
    summary += f"""---

## Passed Cases ({passed_count})

Top 10 most accurate predictions:

"""
    
    for i, case in enumerate(sorted(passed_list, key=lambda x: x['error'])[:10], 1):
        summary += f"{i}. Sample {case['id']:05d}: {case['error']:.2f}px error ({case['architecture']}, {case['noise']})\n"
    
    summary += f"""

---

## Failed Cases ({failed_count})

Top 10 worst predictions:

"""
    
    for i, case in enumerate(sorted(failed_list, key=lambda x: x['error'], reverse=True)[:10], 1):
        summary += f"{i}. Sample {case['id']:05d}: {case['error']:.2f}px error ({case['architecture']}, {case['noise']})\n"
    
    summary += f"""

---

## Visualization Locations

- **Correct predictions**: `output/eval_output/correct/` ({passed_count} images)
- **Failed predictions**: `output/eval_output/failures/` ({failed_count} images)
- **Interactive viewer**: `output/eval_output/index.html`

---

Generated by: `generate_eval_visualizations.py`
"""
    
    return summary


def _generate_stats(passed_list, failed_list, output_dir):
    """Generate detailed statistics text."""
    
    passed_errors = [c['error'] for c in passed_list]
    failed_errors = [c['error'] for c in failed_list]
    
    stats = f"""DRIFT-SENSE EVALUATION STATISTICS
{'='*80}

SUMMARY
{'-'*80}
Total Samples:                 {len(passed_list) + len(failed_list)}
Passed (error <= 5.0px):       {len(passed_list)}
Failed (error > 5.0px):       {len(failed_list)}
Pass Rate:                     {len(passed_list)/(len(passed_list)+len(failed_list))*100:.1f}%

PASSED CASES (error <= 5.0px)
{'-'*80}
Count:                         {len(passed_list)}
Min error:                     {min(passed_errors):.2f}px
Max error:                     {max(passed_errors):.2f}px
Mean error:                    {np.mean(passed_errors):.2f}px
Median error:                  {np.median(passed_errors):.2f}px
Std deviation:                 {np.std(passed_errors):.2f}px

FAILED CASES (error > 5.0px)
{'-'*80}
Count:                         {len(failed_list)}
Min error:                     {min(failed_errors):.2f}px
Max error:                     {max(failed_errors):.2f}px
Mean error:                    {np.mean(failed_errors):.2f}px
Median error:                  {np.median(failed_errors):.2f}px
Std deviation:                 {np.std(failed_errors):.2f}px

ACCURACY BY THRESHOLD
{'-'*80}
@ 0.5px:                       {sum(1 for e in passed_errors if e <= 0.5) / (len(passed_list)+len(failed_list)) * 100:.1f}%
@ 1.0px:                       {sum(1 for e in passed_errors if e <= 1.0) / (len(passed_list)+len(failed_list)) * 100:.1f}%
@ 2.0px:                       {sum(1 for e in passed_errors if e <= 2.0) / (len(passed_list)+len(failed_list)) * 100:.1f}%
@ 5.0px:                       {len(passed_list) / (len(passed_list)+len(failed_list)) * 100:.1f}%

OUTPUT LOCATION
{'-'*80}
Base Directory:                {output_dir.absolute()}
Correct Predictions:           {output_dir}/correct/
Failed Predictions:            {output_dir}/failures/
Summary Report:                {output_dir}/RESULTS.md
Interactive Viewer:            {output_dir}/index.html

GENERATED: 2026-08-17
"""
    
    return stats


def _generate_html_index(output_dir, passed_count, failed_count, metrics):
    """Generate interactive HTML index."""
    
    model_metrics = metrics.get("model", {})
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drift-Sense Evaluation Results</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .stat-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-box h3 {{ color: #667eea; margin-bottom: 10px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .stat-unit {{ color: #666; font-size: 0.9em; }}
        .gallery {{
            padding: 40px;
        }}
        .gallery h2 {{
            margin-bottom: 30px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 50px;
        }}
        .image-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        .image-card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
        }}
        .image-info {{
            padding: 15px;
            font-size: 0.9em;
        }}
        .pass {{ border-top: 3px solid #28a745; }}
        .fail {{ border-top: 3px solid #dc3545; }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Drift-Sense Evaluation Results</h1>
            <p>Localization Accuracy Visualization Report</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Total Samples</h3>
                <div class="stat-value">{passed_count + failed_count}</div>
            </div>
            <div class="stat-box">
                <h3>Passed Cases</h3>
                <div class="stat-value" style="color: #28a745;">{passed_count}</div>
                <div class="stat-unit">±5.0px accuracy</div>
            </div>
            <div class="stat-box">
                <h3>Failed Cases</h3>
                <div class="stat-value" style="color: #dc3545;">{failed_count}</div>
            </div>
            <div class="stat-box">
                <h3>Pass Rate</h3>
                <div class="stat-value" style="color: #667eea;">{passed_count/(passed_count+failed_count)*100:.1f}%</div>
                <div class="stat-unit">Target: >=80%</div>
            </div>
            <div class="stat-box">
                <h3>Mean Error</h3>
                <div class="stat-value">{model_metrics.get('mean', 0):.2f}</div>
                <div class="stat-unit">pixels</div>
            </div>
            <div class="stat-box">
                <h3>Median Error</h3>
                <div class="stat-value">{model_metrics.get('median', 0):.2f}</div>
                <div class="stat-unit">pixels</div>
            </div>
        </div>
        
        <div class="gallery">
            <div class="tabs">
                <button class="tab active" onclick="switchTab(event, 'correct')">
                    Correct Predictions ({passed_count})
                </button>
                <button class="tab" onclick="switchTab(event, 'failures')">
                    Failed Predictions ({failed_count})
                </button>
            </div>
            
            <div id="correct" class="tab-content active">
                <h2>Correct Predictions (Error <= 5.0px)</h2>
                <p style="margin-bottom: 20px; color: #666;">
                    These {passed_count} cases show successful localization where the predicted spot is 
                    within 5 pixels of the ground truth. Green circle: ground truth, Cyan circle: prediction.
                </p>
                <div class="image-grid" id="correct-grid">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
            
            <div id="failures" class="tab-content">
                <h2>Failed Predictions (Error > 5.0px)</h2>
                <p style="margin-bottom: 20px; color: #666;">
                    These {failed_count} cases show localization failures where the predicted spot exceeded 
                    the 5-pixel tolerance. Green circle: ground truth, Red circle: prediction.
                </p>
                <div class="image-grid" id="failures-grid">
                    <!-- Populated by JavaScript -->
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated: 2026-08-17 | Drift-Sense Evaluation Report</p>
            <p>Results Directory: <code>output/eval_output/</code></p>
        </div>
    </div>
    
    <script>
        function switchTab(evt, tabName) {{
            var contents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove("active");
            }}
            
            var tabs = document.getElementsByClassName("tab");
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove("active");
            }}
            
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}
        
        // Populate image grids
        fetch('correct/')
            .then(r => r.text())
            .then(html => {{
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, 'text/html');
                var links = doc.querySelectorAll('a[href$=".png"]');
                var grid = document.getElementById('correct-grid');
                links.forEach(link => {{
                    var filename = link.textContent;
                    if (filename.endsWith('_PASS.png')) {{
                        var card = document.createElement('div');
                        card.className = 'image-card pass';
                        card.onclick = function() {{ window.open('correct/' + filename); }};
                        card.innerHTML = `
                            <img src="correct/${{filename}}" alt="${{filename}}">
                            <div class="image-info">
                                <strong>${{filename.replace('_PASS.png', '')}}</strong><br>
                                Status: PASS
                            </div>
                        `;
                        grid.appendChild(card);
                    }}
                }});
            }})
            .catch(e => console.log('Note: Directory listing not available'));
        
        fetch('failures/')
            .then(r => r.text())
            .then(html => {{
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, 'text/html');
                var links = doc.querySelectorAll('a[href$=".png"]');
                var grid = document.getElementById('failures-grid');
                links.forEach(link => {{
                    var filename = link.textContent;
                    if (filename.endsWith('_FAIL.png')) {{
                        var card = document.createElement('div');
                        card.className = 'image-card fail';
                        card.onclick = function() {{ window.open('failures/' + filename); }};
                        card.innerHTML = `
                            <img src="failures/${{filename}}" alt="${{filename}}">
                            <div class="image-info">
                                <strong>${{filename.replace('_FAIL.png', '')}}</strong><br>
                                Status: FAIL
                            </div>
                        `;
                        grid.appendChild(card);
                    }}
                }});
            }})
            .catch(e => console.log('Note: Directory listing not available'));
    </script>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    main()
