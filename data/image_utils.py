"""Image preprocessing utilities for visual safety neuron detection.

Image specifications:
- Preserve aspect ratio, max long-side configurable (pipeline.yaml image.max_side)
- Blank: solid white
- Gaussian noise: per-pixel, mean 128, std 50, clipped to [0, 255]
"""

import numpy as np
from PIL import Image


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image so the longest side equals max_side, preserving aspect ratio.

    If both sides are already <= max_side, return a copy unchanged.
    Uses LANCZOS resampling for quality.
    """
    w, h = img.size
    if max(w, h) <= max_side:
        return img.copy()
    if w >= h:
        new_w = max_side
        new_h = int(h * max_side / w)
    else:
        new_h = max_side
        new_w = int(w * max_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def generate_blank(size: int = 1024) -> Image.Image:
    """Generate a solid white image of size×size."""
    return Image.new("RGB", (size, size), (255, 255, 255))


def generate_gaussian_noise(size: int = 1024, seed: int = None,
                            mean: float = 128, std: float = 50) -> Image.Image:
    """Generate a Gaussian noise image.

    Per-pixel values drawn from N(mean, std²), clipped to [0, 255].
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, std, (size, size, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noise)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def _test_image_utils():
    """Run basic unit tests for image preprocessing."""
    results = []

    # Test 1: Square image larger than max_side
    img = Image.new("RGB", (2048, 2048), (100, 100, 100))
    out = resize_max_side(img, 1024)
    assert out.size == (1024, 1024), f"Expected (1024,1024), got {out.size}"
    results.append(("square_large", "PASS"))

    # Test 2: 4:3 landscape
    img = Image.new("RGB", (4000, 3000), (100, 100, 100))
    out = resize_max_side(img, 1024)
    assert out.size[0] == 1024, f"Expected width 1024, got {out.size[0]}"
    assert abs(out.size[1] - 768) <= 1, f"Expected height ~768, got {out.size[1]}"
    results.append(("landscape_4_3", "PASS"))

    # Test 3: 2:1 wide
    img = Image.new("RGB", (2000, 1000), (100, 100, 100))
    out = resize_max_side(img, 1024)
    assert out.size[0] == 1024
    assert abs(out.size[1] - 512) <= 1
    results.append(("wide_2_1", "PASS"))

    # Test 4: Small image (no resize)
    img = Image.new("RGB", (500, 300), (100, 100, 100))
    out = resize_max_side(img, 1024)
    assert out.size == (500, 300), f"Small image should not resize, got {out.size}"
    results.append(("small_noresize", "PASS"))

    # Test 5: Portrait
    img = Image.new("RGB", (1000, 3000), (100, 100, 100))
    out = resize_max_side(img, 1024)
    assert out.size[1] == 1024
    assert abs(out.size[0] - 341) <= 1
    results.append(("portrait", "PASS"))

    # Test 6: Blank image
    blank = generate_blank(1024)
    assert blank.size == (1024, 1024)
    arr = np.array(blank)
    assert (arr == 255).all(), "Blank should be all white"
    results.append(("blank", "PASS"))

    # Test 7: Noise image — value range and seed reproducibility
    noise1 = generate_gaussian_noise(1024, seed=42)
    noise2 = generate_gaussian_noise(1024, seed=42)
    assert noise1.size == (1024, 1024)
    arr1 = np.array(noise1)
    arr2 = np.array(noise2)
    assert (arr1 == arr2).all(), "Same seed should produce identical noise"
    assert arr1.min() >= 0 and arr1.max() <= 255
    results.append(("noise_range_and_seed", "PASS"))

    # Test 8: Noise different seeds
    noise3 = generate_gaussian_noise(1024, seed=99)
    arr3 = np.array(noise3)
    assert not (arr1 == arr3).all(), "Different seeds should differ"
    results.append(("noise_different_seeds", "PASS"))

    for name, status in results:
        print(f"  {name}: {status}")

    return results


if __name__ == "__main__":
    print("Running image_utils unit tests...")
    _test_image_utils()
    print("All tests passed.")
