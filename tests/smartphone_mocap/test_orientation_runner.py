# tests/smartphone_mocap/test_orientation_runner.py
import numpy as np
import pytest

from smartphone_mocap.calibration import look_at_world_to_camera
from smartphone_mocap.recognition.orientation import OrientationRunner
from smartphone_mocap.types import BoundingBox2D, CameraCalibration, Object2D, SceneObject3D

CAL = {"a": CameraCalibration("a", (64, 48), np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1.]]), np.zeros(5),
                              look_at_world_to_camera((0.6 - 1.0, 0.4, 0.8), (0.6, 0.4, 0.0)))}


def _scene(label="giraffe"):
    det = Object2D("a", 0, label, 0, BoundingBox2D(10, 10, 20, 20), .9, mask=np.ones((48, 64), bool))
    obj = SceneObject3D("g1", label, (0.6, 0.4, 0.0), .9, 1., ("a",), state="active",
                        source_detection_keys=(("a", 0),))
    return [obj], {"a": [det]}, {"a": np.zeros((48, 64, 3), np.uint8)}


def _classify_to(bin_index):
    calls = []

    def classify(crops_by_object):
        calls.append(len(crops_by_object))
        return {oid: [({"upright": .9, "lying": .1, "upside_down": 0.}, {bin_index: 1.0})] for oid in crops_by_object}
    return classify, calls


def test_the_runner_turns_a_camera_relative_answer_into_table_yaw():
    classify, _ = _classify_to(2)                      # camera at azimuth 0 → table yaw 90
    objects, dets, images = _scene()
    out = OrientationRunner(classify, symmetric=set()).apply(objects, dets, images, CAL, 0)
    assert out[0].yaw_deg == pytest.approx(90.0, abs=1e-6) and out[0].pose == "upright"


def test_yaw_accumulates_across_the_north_wrap_without_jumping():
    objects, dets, images = _scene()
    runner = OrientationRunner(_classify_to(0)[0], symmetric=set(), interval_s=0.0, keep=0.5)
    runner.apply(objects, dets, images, CAL, 0)                          # 0°
    runner.classify = _classify_to(7)[0]                                 # 315°
    out = runner.apply(objects, dets, images, CAL, 10)
    assert min(abs(out[0].yaw_deg - 337.5), abs(out[0].yaw_deg + 22.5)) < 1.0


def test_symmetric_props_get_no_yaw_and_the_interval_skips_classifying():
    classify, calls = _classify_to(2)
    objects, dets, images = _scene("paper_cup")
    runner = OrientationRunner(classify, symmetric={"paper_cup"}, interval_s=0.5)
    out = runner.apply(objects, dets, images, CAL, 0)
    runner.apply(objects, dets, images, CAL, 100_000_000)               # 0.1 s later: no new call
    assert out[0].yaw_deg is None and calls == [0]


def test_missing_head_file_turns_orientation_off(tmp_path, capsys):
    assert OrientationRunner.from_files(tmp_path / "none.pt", symmetric=set()) is None
    assert "방향" in capsys.readouterr().out


def test_a_lying_object_gets_its_pose_but_no_yaw():
    def classify(crops_by_object):
        return {oid: [({"upright": .1, "lying": .9, "upside_down": 0.}, {2: 1.0})] for oid in crops_by_object}
    objects, dets, images = _scene()
    out = OrientationRunner(classify, symmetric=set()).apply(objects, dets, images, CAL, 0)
    assert out[0].pose == "lying" and out[0].yaw_deg is None
