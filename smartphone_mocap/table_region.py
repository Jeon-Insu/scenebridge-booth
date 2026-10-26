"""Drop object detections that are not on the table.

The table top (its footprint plus a margin) is projected into each calibrated camera,
and a detection is kept only when the point where it stands — the bottom of its mask,
or of its box without one — falls on it. People, monitors, chairs and tripods around the
booth are removed before reconstruction, whichever detector produced them.

The footprint, not the volume above it: projecting the space up to the workspace height
spread the region far past the table for cameras close to it (9/10 cam_C).
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import project_point


def contact_point(detection) -> tuple[float, float]:
    """Where the object stands: Object2D's mask_bottom_center, computed inside its box.

    Searching the full-frame mask cost ~1.7 ms per detection (~50 ms a frame).
    """
    mask = detection.mask
    if mask is None:
        return detection.anchor_xy("mask_bottom_center")
    height, width = mask.shape
    box = detection.box
    x0, y0 = max(0, int(box.x1) - 1), max(0, int(box.y1) - 1)
    x1, y1 = min(width, int(np.ceil(box.x2)) + 1), min(height, int(np.ceil(box.y2)) + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if not len(xs):
        return detection.anchor_xy("mask_bottom_center")
    cutoff = np.percentile(ys, 95.0)
    return float(np.median(xs[ys >= cutoff]) + x0), float(cutoff + y0)


class TableRegionFilter:
    def __init__(self, calibrations, table_size_xyz, *, margin_m: float = 0.05):
        size_x, size_y = (float(v) for v in table_size_xyz[:2])
        m = float(margin_m)
        corners = np.array([[-m, -m, 0.0], [size_x + m, -m, 0.0], [size_x + m, size_y + m, 0.0],
                            [-m, size_y + m, 0.0]])
        self._regions: dict[str, np.ndarray] = {}
        for camera_id, camera in calibrations.items():
            depth = (camera.rotation @ corners.T).T + camera.translation
            if np.any(depth[:, 2] <= 0):   # a corner behind the lens: no reliable outline
                continue
            points = np.array([project_point(camera, corner) for corner in corners], np.float32)
            self._regions[camera_id] = points

    def _on_table(self, outline, detection) -> bool:
        x, y = contact_point(detection)
        return cv2.pointPolygonTest(outline, (float(x), float(y)), False) >= 0

    def filter(self, detections_by_camera):
        out = {}
        for camera_id, items in detections_by_camera.items():
            outline = self._regions.get(camera_id)
            out[camera_id] = list(items) if outline is None else [d for d in items if self._on_table(outline, d)]
        return out


def build_table_region_filter(config, rig_config, calibrations):
    """The filter, when enabled and the rig says how big the table is; otherwise None."""
    size = rig_config.get("workspace_size_m")
    if not config.get("enabled", False) or not size:
        return None
    return TableRegionFilter(calibrations, size, margin_m=float(config.get("margin_m", 0.05)))
