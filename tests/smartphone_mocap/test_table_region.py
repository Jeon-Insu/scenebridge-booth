import pytest
import numpy as np

from smartphone_mocap.calibration import look_at_world_to_camera
from smartphone_mocap.geometry import project_point
from smartphone_mocap.types import BoundingBox2D, CameraCalibration, Object2D

TABLE = (1.25, 0.75, 0.35)


def _camera():
    w2c = look_at_world_to_camera(np.array([0.6, -0.9, 0.9]), np.array([0.62, 0.37, 0.0]))
    k = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    return CameraCalibration("cam_A", (1280, 720), k, np.zeros(5), w2c)


def _detection_at(camera, xyz, half=12, label="cup"):
    u, v = project_point(camera, np.asarray(xyz, float))
    mask = np.zeros((720, 1280), bool)
    y0, y1 = int(max(0, v - half)), int(min(720, v + half))
    x0, x1 = int(max(0, u - half)), int(min(1280, u + half))
    mask[y0:y1, x0:x1] = True
    return Object2D("cam_A", 0, label, 47, BoundingBox2D(x0, y0, x1, y1), .9, mask=mask)


def _filter(**kwargs):
    from smartphone_mocap.table_region import TableRegionFilter
    return TableRegionFilter({"cam_A": _camera()}, TABLE, **kwargs)


def test_object_on_the_table_is_kept():
    detection = _detection_at(_camera(), (0.6, 0.4, 0.05))
    assert _filter().filter({"cam_A": [detection]})["cam_A"] == [detection]


def test_object_beside_the_table_is_dropped():
    beside = _detection_at(_camera(), (1.6, 0.4, 0.2))      # a monitor next to the table
    behind = _detection_at(_camera(), (1.7, 0.6, 0.3))      # a person past the far corner
    assert _filter().filter({"cam_A": [beside, behind]})["cam_A"] == []


def test_detection_without_mask_uses_its_box_bottom():
    on_table = _detection_at(_camera(), (0.6, 0.4, 0.02))
    off_table = _detection_at(_camera(), (1.6, 0.4, 0.2))
    from dataclasses import replace
    kept = _filter().filter({"cam_A": [replace(on_table, mask=None), replace(off_table, mask=None)]})
    assert [d.box for d in kept["cam_A"]] == [on_table.box]


def test_camera_without_calibration_passes_detections_through():
    detection = _detection_at(_camera(), (1.9, 0.4, 0.3))
    assert _filter().filter({"cam_Z": [detection]}) == {"cam_Z": [detection]}


def test_build_returns_none_unless_enabled():
    from smartphone_mocap.table_region import build_table_region_filter
    rig = {"workspace_size_m": list(TABLE)}
    assert build_table_region_filter({"enabled": False}, rig, {"cam_A": _camera()}) is None
    assert build_table_region_filter({"enabled": True}, {}, {"cam_A": _camera()}) is None
    assert build_table_region_filter({"enabled": True, "margin_m": .02}, rig, {"cam_A": _camera()}) is not None


def test_tall_object_near_the_camera_is_judged_by_where_it_touches_the_table():
    # 2026-10-10: projecting the table volume (0.35 m high) spread the region far past the
    # table for near cameras; the decision is where the object stands, not its silhouette.
    camera = _camera()
    from smartphone_mocap.geometry import project_point
    from smartphone_mocap.types import BoundingBox2D, Object2D
    top = project_point(camera, np.array([1.6, 0.4, 0.3]))
    bottom = project_point(camera, np.array([1.6, 0.4, 0.0]))      # beside the table
    x0, x1 = int(top[0] - 10), int(top[0] + 10)
    mask = np.zeros((720, 1280), bool)
    mask[int(top[1]):int(bottom[1]), x0:x1] = True
    standing_beside = Object2D("cam_A", 0, "clock", 74, BoundingBox2D(x0, top[1], x1, bottom[1]), .9, mask=mask)
    assert _filter().filter({"cam_A": [standing_beside]})["cam_A"] == []


def test_contact_point_from_the_box_crop_matches_the_full_mask_anchor():
    from smartphone_mocap.table_region import contact_point
    from smartphone_mocap.types import BoundingBox2D, Object2D
    rng = np.random.default_rng(0)
    mask = np.zeros((720, 1280), bool)
    mask[300:360, 500:540] = rng.random((60, 40)) > 0.3
    ys, xs = np.nonzero(mask)
    detection = Object2D("cam_A", 0, "cup", 47, BoundingBox2D(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                         .9, mask=mask)
    assert contact_point(detection) == pytest.approx(detection.anchor_xy("mask_bottom_center"))
