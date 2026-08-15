"""
Drift-Sense synthetic data generation (ML submission).

The physical scene generator is vendored under `data_gen/src` (imported as
`src.*`); `data_gen/generate.py` wraps it with the scale/rotation/perturbation
handling the challenge requires (nominal 10:1 but tested at ~9:1 to 11:1,
and small 1-2 deg rotations).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_gen.generate import (  # noqa: E402,F401
    generate_pair,
    generate_dataset,
    sample_training_params,
    NOISE_LEVELS,
)
