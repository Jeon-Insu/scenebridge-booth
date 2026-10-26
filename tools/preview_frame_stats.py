"""Brightness statistics of the frames the running venue preview saves every 2 seconds.

Measuring from those files lets camera controls be tuned while the preview keeps the
devices open (a second process cannot open a streaming V4L2 camera).
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / 'outputs/venue_finalize_20260918/preview'


def read_view(view, newer_than=0.0, timeout=8.0):
    """Latest complete frame for a view, optionally waiting for one written after a time."""
    path = PREVIEW / f'view_{view}.jpg'
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_mtime > newer_than:
            with tempfile.NamedTemporaryFile(suffix='.jpg') as copy:
                shutil.copy(path, copy.name)
                image = cv2.imread(copy.name)
            if image is not None and image.shape[0] == 720:
                return image
        time.sleep(0.3)
    return None


def stats(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {'mean': float(grey.mean()), 'std': float(grey.std()),
            'clip': float(np.mean(grey >= 250) * 100), 'p5': float(np.percentile(grey, 5)),
            'sharp': float(cv2.Laplacian(grey, cv2.CV_64F).var())}


if __name__ == '__main__':
    for view in (1, 2, 3, 4):
        image = read_view(view)
        if image is None:
            print(f'VIEW{view}: 프레임 없음')
            continue
        s = stats(image)
        print(f"VIEW{view}: 밝기 {s['mean']:5.1f}  대비 {s['std']:4.1f}  포화 {s['clip']:4.1f}%  "
              f"하위5% {s['p5']:5.1f}  선명도 {s['sharp']:5.0f}")
