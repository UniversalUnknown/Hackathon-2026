#!/usr/bin/env python3
"""
Quick visualization generator - run after evaluation to view results.

Usage:
    python visualize_results.py          # Generate all visualizations
    python visualize_results.py --passed # Show only passed cases
    python visualize_results.py --failed # Show only failed cases
    
Output: Creates output/eval_output/ with organized results
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_path = Path(__file__).parent / "generate_eval_visualizations.py"
    
    if not script_path.exists():
        print("❌ Error: generate_eval_visualizations.py not found")
        sys.exit(1)
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              DRIFT-SENSE EVALUATION VISUALIZATION TOOL                     ║
╚════════════════════════════════════════════════════════════════════════════╝

This tool generates detailed visualizations showing:
  ✓ Correct predictions (64 cases)    - Green: Ground truth, Cyan: Predicted
  ✗ Failed predictions (16 cases)     - Green: Ground truth, Red: Predicted
  📊 Interactive HTML index
  📄 Detailed report and statistics

Starting visualization generation...
""")
    
    result = subprocess.run([sys.executable, str(script_path)])
    
    if result.returncode == 0:
        output_dir = Path("output/eval_output")
        print(f"""
✓ Visualization generation complete!

📁 Results saved to: {output_dir.absolute()}

📊 View Results:
  1. Interactive Browser View:
     → Open: {output_dir / 'index.html'}
     
  2. Detailed Report:
     → Read: {output_dir / 'RESULTS.md'}
     
  3. Statistics Summary:
     → Read: {output_dir / 'summary.txt'}
     
  4. Image Folders:
     → Correct: {output_dir / 'correct'}/  (64 images)
     → Failed:  {output_dir / 'failures'}/  (16 images)

💡 Tip: Open index.html in your web browser for interactive viewing!
""")
    else:
        print("❌ Visualization generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
