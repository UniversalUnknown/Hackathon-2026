#!/usr/bin/env python3
"""Localize a reference pattern inside a wider SEM image.

Prints the target centre ``x y`` in search-image pixels to stdout.

Usage:
    python localize.py --search search.png --reference reference.png \
        --config configs/default.json --weights output/weights/ranker.pt

Exit code 0 on success (prints "x y"), 1 on failure (message on stderr).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml.predict import Localizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="10x search image (png)")
    ap.add_argument("--reference", required=True, help="100x reference image (png)")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--weights", default="output/weights/ranker.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import cv2
    cfg = json.loads(Path(args.config).read_text())
    search = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    if search is None:
        sys.exit(f"cannot read search image: {args.search}")
    if reference is None:
        sys.exit(f"cannot read reference image: {args.reference}")

    loc = Localizer(cfg, args.weights, device=args.device)
    result = loc.localize(search, reference)
    if not result["ok"]:
        sys.exit(f"localization failed: {result['reason']}")

    print(f"{result['x']:.3f} {result['y']:.3f}")


if __name__ == "__main__":
    main()
