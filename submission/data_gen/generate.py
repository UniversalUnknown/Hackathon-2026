"""
Scale- and rotation-aware synthetic pair generator.

Builds on the vendored `src` (fial) physical scene generator but generalizes
it to the full robustness envelope the challenge asks for:

  * nominal 10:1 magnification difference, robustly handled at ~9:1 to 11:1
    (implemented physically: the fine canvas is 1000*scale px @ 1 nm/px and
    the search image is that canvas area-averaged back down to 1000x1000 px,
    so the reference's footprint in the search image is exactly 1000/scale px);
  * small rotation (0-2 deg) of the search capture relative to the reference
    (implemented as a rigid rotation of the downsampled canvas with an exact
    ground-truth coordinate transform);
  * SEM-style degradation controlled by GenerationParams (beam blur, shot/
    detector/speckle/salt-and-pepper noise, raster drift, astigmatism, gamma,
    vignette, charging streaks, structural pattern collapse).

Ground truth is always expressed in search-image pixels with the (0,0)
top-left convention: `gt_x, gt_y` = the target-pattern centre.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src import sem_imaging
from src.patterns.zones import generate_zone_canvas
from src.presets import get_preset

REFERENCE_SIZE_PX = 1000  # 1000 x 1000 @ 1 nm/px  (1 um FOV)
SEARCH_SIZE_PX = 1000     # 1000 x 1000 search image
PIXEL_SIZE_REF_NM = 1.0
PIXEL_SIZE_SEARCH_NM = 10.0


# --------------------------------------------------------------------------
# Acquisition-condition presets used to build a varied training/eval set.
# Fields are (lo, hi) uniform-sampling ranges unless noted.
# --------------------------------------------------------------------------
# Acquisition bands. Calibrated so the *same physical structure* remains
# clearly findable between reference and search (reference is cleaner/higher
# dose; the search degrades with dose/drift/astigmatism/etc.). Severe is
# deliberately pushing toward failure for the robustness study.
NOISE_LEVELS = {
    "low": {
        "dose_search": (700.0, 1000.0),
        "detector_noise_sigma_search": (1.5, 3.0),
        "shear_amplitude_px": (0.2, 0.8),
        "drift_jitter_px": (0.1, 0.4),
        "speckle_sigma": (0.0, 0.03),
        "salt_pepper_prob": (0.0, 0.001),
    },
    "medium": {
        "dose_search": (200.0, 400.0),
        "detector_noise_sigma_search": (3.0, 6.0),
        "shear_amplitude_px": (0.5, 1.5),
        "drift_jitter_px": (0.3, 0.7),
        "speckle_sigma": (0.0, 0.08),
        "salt_pepper_prob": (0.0, 0.003),
    },
    "high": {
        "dose_search": (70.0, 120.0),
        "detector_noise_sigma_search": (5.0, 9.0),
        "shear_amplitude_px": (1.2, 2.8),
        "drift_jitter_px": (0.6, 1.2),
        "speckle_sigma": (0.05, 0.18),
        "salt_pepper_prob": (0.0, 0.006),
    },
    "severe": {
        "dose_search": (25.0, 50.0),
        "detector_noise_sigma_search": (8.0, 13.0),
        "shear_amplitude_px": (2.0, 4.0),
        "drift_jitter_px": (1.0, 1.8),
        "speckle_sigma": (0.12, 0.3),
        "salt_pepper_prob": (0.002, 0.015),
    },
}


def sample_training_params(rng: np.random.Generator, level: str | None = None) -> dict:
    """Draw a random GenerationParams-style dict from a noise band plus the
    structural/imaging hyper-parameters that vary realistically between
    acquisitions. Used for training diversity and eval coverage.
    """
    level = level or list(NOISE_LEVELS.keys())[int(rng.integers(0, len(NOISE_LEVELS)))]
    band = NOISE_LEVELS[level]

    def uni(key):
        lo, hi = band[key]
        return float(rng.uniform(lo, hi))

    params = {
        "beam_spot_size_nm": float(rng.uniform(2.5, 6.0)),
        "collapse_threshold_nm": float(rng.uniform(8.0, 12.0)),
        "dose_reference": float(rng.uniform(1500.0, 4000.0)),
        "dose_search": uni("dose_search"),
        "shear_amplitude_px": uni("shear_amplitude_px"),
        "drift_jitter_px": uni("drift_jitter_px"),
        "detector_noise_sigma_ref": float(rng.uniform(1.0, 2.5)),
        "detector_noise_sigma_search": uni("detector_noise_sigma_search"),
        "astigmatism_ratio": float(rng.uniform(0.85, 1.3)),
        "vignette_strength": float(rng.uniform(0.0, 0.35)),
        "gamma": float(rng.uniform(0.85, 1.35)),
        "barrel_distortion_k": float(rng.uniform(-0.004, 0.004)),
        "charging_streak_prob": float(rng.uniform(0.0, 2.0)),
        "charging_streak_intensity": float(rng.uniform(0.0, 1.5)),
        "speckle_sigma": uni("speckle_sigma"),
        "salt_pepper_prob": uni("salt_pepper_prob"),
        "mat_size_nm": float(rng.uniform(2000.0, 3800.0)),
        "strip_width_nm": float(rng.uniform(250.0, 450.0)),
        "boundary_bias": float(rng.uniform(0.25, 0.5)),
        "linewidth_bias_nm": float(rng.uniform(-5.0, 5.0)),
        "corner_rounding_px": float(rng.uniform(0.0, 2.0)),
    }
    params["noise_level"] = level
    return params


def _rotate_search(img: np.ndarray, deg: float):
    """Rotate a square image by `deg` degrees about its centre, upscaling so
    no content is lost, then centre-crop back to the original size. Returns
    (rotated, affine_matrix, crop_offset). The affine matrix maps original
    pixels into the upscaled frame; subtract crop_offset for the final image.
    """
    h, w = img.shape
    th = math.radians(deg)
    out_side = int(math.ceil(w * (abs(math.cos(th)) + abs(math.sin(th))))) + 4
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg, 1.0)
    M[0, 2] += (out_side - w) / 2.0
    M[1, 2] += (out_side - h) / 2.0
    rotated = cv2.warpAffine(
        img, M, (out_side, out_side),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    off = (out_side - h) // 2
    return rotated[off:off + h, off:off + w], M, off


def generate_pair(
    architecture: str,
    rng: np.random.Generator,
    params: dict,
    scale: float = 10.0,
    rotation_deg: float = 0.0,
    preset_overrides: dict | None = None,
) -> dict:
    """Generate one Reference/Search pair with exact ground truth.

    Args:
        architecture: preset name, e.g. 'dram_1x', 'finfet_10nm'.
        rng: numpy Generator.
        params: dict matching GenerationParams fields (+ optional noise_level).
        scale: true magnification ratio (search px per reference px). Nominal 10.
        rotation_deg: rotation of the search capture relative to the reference.
    Returns:
        dict(reference_img, search_img, gt_x, gt_y, gt_box, architecture,
             scale, rotation_deg, seed, params)
    """
    preset = get_preset(architecture)
    kind = preset["kind"]

    # Physical scene: 1000*scale px @ 1 nm/px, i.e. the search image's FOV.
    canvas_size = int(round(SEARCH_SIZE_PX * scale))
    zone = generate_zone_canvas(
        canvas_size,
        kind,
        params.get("collapse_threshold_nm", 10.0),
        rng,
        mat_size_nm=params.get("mat_size_nm", 2600.0),
        strip_width_nm=params.get("strip_width_nm", 320.0),
        linewidth_bias_nm=params.get("linewidth_bias_nm", 0.0),
        corner_rounding_px=params.get("corner_rounding_px", 0.0),
    )
    fine = zone["canvas"]

    # Reference: random 1000x1000 crop of the fine canvas @ 1 nm/px.
    max_origin = canvas_size - REFERENCE_SIZE_PX
    x0 = int(rng.integers(0, max_origin + 1))
    y0 = int(rng.integers(0, max_origin + 1))
    crop = fine[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.get("beam_spot_size_nm", 5.0),
        dose=params.get("dose_reference", 2000.0),
        rng=rng,
        detector_noise_sigma=params.get("detector_noise_sigma_ref", 2.0),
        drift_jitter_px=params.get("drift_jitter_px", 0.5) * 0.2,
        astigmatism_ratio=params.get("astigmatism_ratio", 1.0),
        vignette_strength=params.get("vignette_strength", 0.0) * 0.5,
        gamma=params.get("gamma", 1.0),
        barrel_distortion_k=params.get("barrel_distortion_k", 0.0) * 0.5,
        charging_streak_prob=params.get("charging_streak_prob", 0.0),
        charging_streak_intensity=params.get("charging_streak_intensity", 0.0),
        speckle_sigma=params.get("speckle_sigma", 0.0),
        salt_pepper_prob=params.get("salt_pepper_prob", 0.0),
    )

    # Search: shared beam blur at 1 nm/px, area-average downsample to
    # 1000x1000 (factor = scale), optional rigid rotation, then capture
    # artifacts specific to the wide-area scan.
    blurred = sem_imaging.gaussian_psf_blur(
        fine, params.get("beam_spot_size_nm", 5.0), PIXEL_SIZE_REF_NM,
        params.get("astigmatism_ratio", 1.0),
    )
    down = cv2.resize(blurred, (SEARCH_SIZE_PX, SEARCH_SIZE_PX), interpolation=cv2.INTER_AREA)

    affine = None
    if abs(rotation_deg) > 1e-6:
        down, affine, off = _rotate_search(down, rotation_deg)

    search_img = sem_imaging.apply_raster_drift(
        down, params.get("shear_amplitude_px", 1.5), params.get("drift_jitter_px", 0.5), rng,
    )
    search_img = sem_imaging.apply_barrel_distortion(
        search_img, params.get("barrel_distortion_k", 0.0),
    )
    search_img = sem_imaging.add_shot_noise(search_img, params.get("dose_search", 200.0), rng)
    search_img = sem_imaging.add_detector_noise(
        search_img, params.get("detector_noise_sigma_search", 5.0), rng,
    )
    search_img = sem_imaging.add_speckle_noise(search_img, params.get("speckle_sigma", 0.0), rng)
    search_img = sem_imaging.add_salt_and_pepper_noise(
        search_img, params.get("salt_pepper_prob", 0.0), rng,
    )
    search_img = sem_imaging.apply_vignette(search_img, params.get("vignette_strength", 0.0))
    search_img = sem_imaging.apply_gamma(search_img, params.get("gamma", 1.0))
    search_img = sem_imaging.add_charging_streaks(
        search_img,
        params.get("charging_streak_prob", 0.0),
        params.get("charging_streak_intensity", 0.0),
        rng,
    )

    # Ground truth: crop origin -> search px, centre it.
    box_w = REFERENCE_SIZE_PX / scale  # footprint of the reference in search px
    gt_x0 = x0 / scale
    gt_y0 = y0 / scale
    gt_cx = gt_x0 + box_w / 2.0
    gt_cy = gt_y0 + box_w / 2.0

    if affine is not None:
        pt = affine @ np.array([gt_cx, gt_cy, 1.0])
        gt_cx = pt[0] - off
        gt_cy = pt[1] - off

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_w),
        "architecture": architecture,
        "scale": scale,
        "rotation_deg": rotation_deg,
        "seed": int(rng.integers(0, 2**31 - 1)),
        "params": dict(params),
    }


def generate_dataset(
    n_samples: int,
    rng: np.random.Generator,
    architectures,
    level_mix=None,
    scale_options=(9.0, 9.5, 10.0, 10.5, 11.0),
    rotation_options=(-2.0, -1.0, 0.0, 1.0, 2.0),
    scale_prob: float = 0.30,
    rotation_prob: float = 0.30,
    n_jobs: int = 1,
) -> list[dict]:
    """Generate `n_samples` diverse pairs. `level_mix` is a list of noise
    level names drawn from uniformly; default all four levels equally.
    Each sample randomly (with prob `scale_prob`/`rotation_prob`) deviates in
    scale / rotation. Deterministic for a given rng.
    """
    if level_mix is None:
        level_mix = list(NOISE_LEVELS.keys())
    arch_names = list(architectures)
    samples = []
    for i in range(n_samples):
        architecture = arch_names[int(rng.integers(0, len(arch_names)))]
        level = level_mix[int(rng.integers(0, len(level_mix)))]
        params = sample_training_params(rng, level)
        scale = scale_options[int(rng.integers(0, len(scale_options)))] if rng.random() < scale_prob else 10.0
        rot = rotation_options[int(rng.integers(0, len(rotation_options)))] if rng.random() < rotation_prob else 0.0
        sample = generate_pair(architecture, rng, params, scale=scale, rotation_deg=rot)
        sample["idx"] = i
        samples.append(sample)
    return samples
