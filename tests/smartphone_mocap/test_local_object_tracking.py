from dataclasses import replace

import numpy as np
import pytest

from smartphone_mocap.local_object_tracking import PerCameraObjectTracker
from smartphone_mocap.types import BoundingBox2D, Object2D


def _cup(detection_id: int, x: float) -> Object2D:
    return Object2D(
        camera_id="cam_a",
        local_detection_id=detection_id,
        label="cup",
        class_id=47,
        box=BoundingBox2D(x, 20.0, x + 30.0, 80.0),
        confidence=0.9,
    )


@pytest.mark.parametrize("backend", ["bytetrack", "tracktrack"])
def test_camera_local_ids_survive_detector_order_changes(backend):
    tracker = PerCameraObjectTracker(frame_rate=5.0, backend=backend)

    first = tracker.update({"cam_a": [_cup(0, 10.0), _cup(1, 100.0)]})["cam_a"]
    first_by_side = {
        "left" if item.box.x1 < 50.0 else "right": item.local_track_id
        for item in first
    }

    # The detector enumerates the right cup first on the next frame.
    second = tracker.update({"cam_a": [_cup(0, 104.0), _cup(1, 14.0)]})["cam_a"]
    second_by_side = {
        "left" if item.box.x1 < 50.0 else "right": item.local_track_id
        for item in second
    }

    assert all(value is not None for value in first_by_side.values())
    assert second_by_side == first_by_side


@pytest.mark.parametrize("backend", ["bytetrack", "tracktrack"])
def test_tracker_preserves_measurement_masks_and_handles_missing_classes(backend):
    tracker = PerCameraObjectTracker(backend=backend, frame_rate=30)
    mask = np.ones((100, 200), dtype=bool)
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    cup = replace(_cup(7, 10), mask=mask, representative_crop=crop, appearance=(1., 0.))
    first = tracker.update({"cam_a": [cup]})["cam_a"][0]
    tracker.update({"cam_a": []})
    # Another camera/class can legitimately have the same numeric local ID.
    tracker.update({"cam_b": [replace(cup, camera_id="cam_b", label="book")]})
    moved = replace(cup, box=BoundingBox2D(12, 20, 42, 80))
    result = tracker.update({"cam_a": [moved]})["cam_a"][0]
    assert result.local_track_id == first.local_track_id
    assert result.box is moved.box
    assert result.mask is mask
    assert result.representative_crop is crop
    assert result.appearance == (1., 0.)
    assert result.local_detection_id == 7


def test_unknown_tracker_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported local tracking"):
        PerCameraObjectTracker(backend="typo")
