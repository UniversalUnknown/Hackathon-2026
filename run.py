import os
import random

import cv2
import numpy as np
from scipy.signal import find_peaks


# ------------------------------------------------------------
# 1. Synthetic data generator (simple grid pattern)
# ------------------------------------------------------------
def generate_synthetic_pair(output_dir="./test_pairs", pair_id=0):
    """
    Creates a reference (1000x1000) and search (1000x1000) grayscale image pair.
    The search image is the large canvas downsampled to 10x scale.
    Returns (ref_path, search_path, true_x, true_y) in search-image pixels.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a big canvas (2000x2000) with a repeating grid pattern
    canvas_size = 2000
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Draw a grid of squares (simulating a simple periodic structure)
    cell_size = 40
    for i in range(0, canvas_size, cell_size):
        for j in range(0, canvas_size, cell_size):
            # Fill every other cell to create a pattern
            if (i // cell_size + j // cell_size) % 2 == 0:
                canvas[i : i + cell_size, j : j + cell_size] = 200
            else:
                canvas[i : i + cell_size, j : j + cell_size] = 100

    # Add some edge lines to make it more interesting
    cv2.rectangle(canvas, (0, 0), (canvas_size - 1, canvas_size - 1), 255, 2)

    # Randomly choose a target location in the canvas (ensure we have enough margin)
    margin = 100
    true_x_big = random.randint(margin, canvas_size - margin - 1)
    true_y_big = random.randint(margin, canvas_size - margin - 1)

    # --- Generate Reference (100x) ---
    # Crop a 1000x1000 patch around the target (that's the 100x view)
    half_ref = 500
    x1 = max(0, true_x_big - half_ref)
    y1 = max(0, true_y_big - half_ref)
    x2 = min(canvas_size, true_x_big + half_ref)
    y2 = min(canvas_size, true_y_big + half_ref)
    ref_patch = canvas[y1:y2, x1:x2]
    # Pad if near edges (should not happen with margin)
    if ref_patch.shape[0] < 1000 or ref_patch.shape[1] < 1000:
        ref_patch = cv2.copyMakeBorder(
            ref_patch,
            0,
            max(0, 1000 - ref_patch.shape[0]),
            0,
            max(0, 1000 - ref_patch.shape[1]),
            cv2.BORDER_CONSTANT,
            value=0,
        )
    ref_patch = cv2.resize(ref_patch, (1000, 1000))

    # --- Generate Search (10x) ---
    # Resize the whole canvas to 1000x1000 (simulating 10x magnification)
    search = cv2.resize(canvas, (1000, 1000), interpolation=cv2.INTER_AREA)

    # The target's true location in search coordinates:
    # Since canvas was scaled by factor (1000/2000)=0.5, we multiply.
    true_x = int(true_x_big * 0.5)
    true_y = int(true_y_big * 0.5)

    # Add some realistic degradation (noise, blur)
    # Shot noise (Poisson)
    search = np.random.poisson(search * 0.5)  # scale to reduce intensity
    search = np.clip(search * 0.5, 0, 255).astype(np.uint8)
    # Gaussian blur
    search = cv2.GaussianBlur(search, (3, 3), 0.5)
    # Salt & pepper noise
    salt_pepper = np.random.rand(*search.shape)
    search[salt_pepper < 0.002] = 0
    search[salt_pepper > 0.998] = 255

    # Save images
    ref_path = os.path.join(output_dir, f"ref_{pair_id:04d}.png")
    search_path = os.path.join(output_dir, f"search_{pair_id:04d}.png")
    cv2.imwrite(ref_path, ref_patch)
    cv2.imwrite(search_path, search)

    return ref_path, search_path, true_x, true_y


# ------------------------------------------------------------
# 2. Localization algorithm (scale-adaptive template matching)
# ------------------------------------------------------------
def localize(ref_path, search_path, scale_factor=10.0, threshold=0.7):
    """
    Finds the reference pattern in the search image.
    Args:
        ref_path: path to 1000x1000 reference image (100x magnification)
        search_path: path to 1000x1000 search image (10x magnification)
        scale_factor: nominal scale difference (10)
        threshold: confidence threshold for multiple matches
    Returns:
        (x, y, confidence) – predicted centre in search-image pixels
    """
    # Read images as grayscale
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or search is None:
        raise FileNotFoundError("Could not read images")

    # 1. Resize reference to match the scale of the search image
    # Reference is 100x, search is 10x, so pattern in search is 10x smaller.
    # We resize reference down by factor of 10 to ~100x100.
    new_w = int(ref.shape[1] / scale_factor)
    new_h = int(ref.shape[0] / scale_factor)
    ref_scaled = cv2.resize(ref, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Template matching (Normalized Cross-Correlation)
    # Result matrix of size (search_h - template_h + 1, search_w - template_w + 1)
    result = cv2.matchTemplate(search, ref_scaled, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # max_loc is the top-left of the best match

    # 3. Find all peaks above threshold
    # We'll find local maxima in the result matrix
    # Use scipy's find_peaks on flattened or use a sliding window approach.
    # Simpler: threshold and then use non-max suppression.
    # We'll implement a simple approach: find coordinates where result > threshold
    # and then pick the peak with highest value in a local neighborhood.
    h, w = result.shape
    # Pad result to handle borders
    result_pad = np.pad(result, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    peaks = []
    for y in range(h):
        for x in range(w):
            val = result[y, x]
            if val >= threshold:
                # Check if it's a local maximum in 3x3 neighborhood
                neighborhood = result_pad[y : y + 3, x : x + 3]
                if val == np.max(neighborhood):
                    peaks.append((x, y, val))

    if not peaks:
        # Fallback to global maximum if no peak meets threshold
        x, y = max_loc
        confidence = max_val
    else:
        # 4. Select the match closest to the centre of the search image (500,500)
        centre = (500, 500)
        best_peak = None
        best_dist = float("inf")
        for x, y, val in peaks:
            # The centre of the matched region is (x + new_w/2, y + new_h/2)
            cx = x + new_w / 2.0
            cy = y + new_h / 2.0
            dist = np.hypot(cx - centre[0], cy - centre[1])
            if dist < best_dist:
                best_dist = dist
                best_peak = (x, y, val, cx, cy)
        x, y, confidence, pred_x, pred_y = best_peak

    # Return the centre coordinates and confidence
    return int(round(pred_x)), int(round(pred_y)), confidence


# ------------------------------------------------------------
# 3. Demo: generate 5 pairs and test localization
# ------------------------------------------------------------
def run_demo():
    print("Generating 5 synthetic test pairs...")
    errors = []
    for i in range(5):
        ref_path, search_path, true_x, true_y = generate_synthetic_pair(pair_id=i)
        pred_x, pred_y, conf = localize(ref_path, search_path)
        error = np.hypot(pred_x - true_x, pred_y - true_y)
        errors.append(error)
        print(
            f"Pair {i}: true=({true_x:3d},{true_y:3d}) pred=({pred_x:3d},{pred_y:3d}) "
            f"error={error:5.2f} px, conf={conf:.3f}"
        )
    print(f"\nMean error: {np.mean(errors):.2f} px, Median: {np.median(errors):.2f} px")
    print("Done.")


if __name__ == "__main__":
    run_demo()
