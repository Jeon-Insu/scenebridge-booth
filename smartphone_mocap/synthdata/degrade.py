"""Camera defects measured at the 2026-09-17..20 venue, applied to rendered images.

Renders are cleaner than the booth cameras were. Training on these defects means the
detector sees the kind of image it will actually get: blown-out white paper, white
bleeding into dark marker bits, flare that lifts the blacks, JPEG and soft focus.
"""
from __future__ import annotations

import cv2
import numpy as np


def overexpose(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * float(gain), 0, 255).astype(np.uint8)


def bloom_whites(image: np.ndarray, pixels: int = 1, threshold: int = 230) -> np.ndarray:
    """Bright regions spill into their neighbours, as paper did over the ChArUco bits."""
    kernel = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(image, kernel, iterations=int(pixels))
    bright = cv2.dilate((image.max(axis=2) >= threshold).astype(np.uint8), kernel,
                        iterations=int(pixels)).astype(bool)
    out = image.copy()
    out[bright] = grown[bright]
    return out


def lift_blacks(image: np.ndarray, floor: float) -> np.ndarray:
    """Lens flare adds light everywhere, so the darkest pixel no longer reaches zero."""
    floor = float(floor)
    lifted = floor + image.astype(np.float32) * (255.0 - floor) / 255.0
    return np.clip(np.round(lifted), 0, 255).astype(np.uint8)


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def defocus(image: np.ndarray, sigma: float) -> np.ndarray:
    return image if sigma <= 0 else cv2.GaussianBlur(image, (0, 0), float(sigma))


# name: (probability, (low, high)). Integer bounds draw integers.
PROFILE = {
    "overexpose": (0.35, (1.1, 1.6)),
    "bloom_whites": (0.30, (1, 2)),
    "lift_blacks": (0.35, (15, 60)),   # 9/19 views measured a 5th-percentile floor of 22-43
    "defocus": (0.25, (0.6, 1.6)),
    "jpeg": (0.80, (60, 92)),
}
_FUNCTIONS = {"overexpose": overexpose, "bloom_whites": bloom_whites, "lift_blacks": lift_blacks,
              "defocus": defocus, "jpeg": jpeg}


def degrade(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply a random subset of the venue defects; returns the image and what was applied."""
    applied: dict = {}
    out = image
    for name, (probability, (low, high)) in PROFILE.items():
        if rng.random() >= probability:
            continue
        value = (int(rng.integers(low, high + 1)) if isinstance(low, int)
                 else float(rng.uniform(low, high)))
        out = _FUNCTIONS[name](out, value)
        applied[name] = value
    return out, applied
