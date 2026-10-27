# tests/tabletop_twin/test_flythrough.py
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "flythrough", Path(__file__).resolve().parents[2] / "exts/gist.tabletop_twin/gist/tabletop_twin/flythrough.py")
flythrough = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flythrough)

KEYS = [{"t": 0.0, "eye": [0, 0, 10], "target": [0, 10, 0]},
        {"t": 4.0, "eye": [4, 0, 2], "target": [4, 10, 0]}]


def test_the_camera_starts_and_ends_on_the_keyframes():
    assert flythrough.pose_at(KEYS, 0.0) == ([0, 0, 10], [0, 10, 0])
    assert flythrough.pose_at(KEYS, 4.0) == ([4, 0, 2], [4, 10, 0])
    assert flythrough.pose_at(KEYS, 99.0) == ([4, 0, 2], [4, 10, 0])


def test_the_camera_eases_between_keyframes():
    eye, _ = flythrough.pose_at(KEYS, 2.0)
    assert eye[0] == pytest.approx(2.0)                 # halfway in time = halfway in space (symmetric ease)
    early, _ = flythrough.pose_at(KEYS, 1.0)
    assert early[0] < 1.0                               # eases in: slower than linear at the start


def test_stages_fire_once_when_their_time_passes():
    stages = [{"at": 1.0, "label": "6F"}, {"at": 2.0, "label": "5F"}]
    assert [s["label"] for s in flythrough.stages_due(stages, 0.0, 1.5)] == ["6F"]
    assert [s["label"] for s in flythrough.stages_due(stages, 1.5, 3.0)] == ["5F"]
    assert flythrough.stages_due(stages, 3.0, 4.0) == []


def test_the_duration_is_the_last_keyframe():
    assert flythrough.duration(KEYS) == 4.0
