# tests/smartphone_mocap/test_recognition_orientation.py
import numpy as np
import pytest

from smartphone_mocap.calibration import look_at_world_to_camera
from smartphone_mocap.recognition.orientation import (camera_azimuth_deg, circular_mean_deg, fuse,
                                                       relative_bin, table_yaw)
from smartphone_mocap.types import CameraCalibration


def _cal(position):
    return CameraCalibration("c", (1280, 720), np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1.]]),
                             np.zeros(5), look_at_world_to_camera(position, (0.6, 0.4, 0.0)))


def test_azimuth_points_from_camera_to_object():
    assert camera_azimuth_deg(_cal((0.6 - 1.0, 0.4, 0.8)), (0.6, 0.4, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert camera_azimuth_deg(_cal((0.6, 0.4 - 1.0, 0.8)), (0.6, 0.4, 0.0)) == pytest.approx(90.0, abs=1e-6)


def test_relative_bins_round_trip_through_table_yaw():
    for yaw in range(0, 360, 45):
        for azimuth in (0.0, 90.0, 225.0):      # bins are exact only on 45° steps
            assert table_yaw(relative_bin(yaw, azimuth), azimuth) == pytest.approx(yaw % 360)


def test_circular_mean_wraps_around_north():
    assert circular_mean_deg([350.0, 10.0], [1, 1]) == pytest.approx(0.0, abs=1e-6)
    assert circular_mean_deg([], []) is None


def test_fuse_combines_cameras_that_see_different_sides():
    # The object faces table yaw 90°. Camera 1 (azimuth 0) sees it at relative bin 2, camera 2 (azimuth 90) at bin 0.
    result = fuse([({"upright": .9, "lying": .1, "upside_down": 0}, {2: .8, 3: .2}, 0.0),
                   ({"upright": .7, "lying": .3, "upside_down": 0}, {0: .9, 7: .1}, 90.0)])
    assert result.pose == "upright"
    assert result.yaw_deg == pytest.approx(90.0, abs=12.0)




def test_silhouette_finds_the_turn_that_matches_the_masks():
    from smartphone_mocap.recognition.orientation import SilhouetteOrientation, _turn
    rng = np.random.default_rng(0)
    # A long, low box (like a lying animal): 20 cm along its own x, 6 cm wide, 8 cm tall.
    box = rng.uniform([-.10, -.03, 0], [.10, .03, .08], size=(1500, 3))
    cal = _cal((0.6, 0.4 - 0.9, 0.7))
    truth = _turn(box, ("x", 0.0), 60.0) + np.array([0.6, 0.4, 0.0])
    W = cal.world_to_camera
    import cv2
    px = cv2.projectPoints(truth, cv2.Rodrigues(W[:3, :3])[0], W[:3, 3], cal.camera_matrix, cal.distortion)[0]
    mask = np.zeros((720, 1280), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(px.reshape(-1, 2).astype(np.int32)), 1)
    result = SilhouetteOrientation({"box": box}).orient([], {"c": cal}, (0.6, 0.4, 0.0), "box", {"c": mask.astype(bool)})
    assert result.pose == "upright"
    assert min(abs(result.yaw_deg - 60.0), abs(result.yaw_deg - 240.0)) <= 15.0   # one view cannot tell front from back
