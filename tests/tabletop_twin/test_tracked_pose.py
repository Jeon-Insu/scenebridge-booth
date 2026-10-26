# tests/tabletop_twin/test_tracked_pose.py
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tracked_pose", Path(__file__).resolve().parents[2] / "exts/gist.tabletop_twin/gist/tabletop_twin/tracked_pose.py")
tracked_pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracked_pose)


def test_yaw_comes_from_the_message_and_defaults_to_zero():
    assert tracked_pose.tracked_yaw_deg({"yaw_deg": 90.0}) == 90.0
    assert tracked_pose.tracked_yaw_deg({}) == 0.0
    assert tracked_pose.tracked_yaw_deg({"yaw_deg": float("nan")}) == 0.0
    assert tracked_pose.tracked_yaw_deg({"yaw_deg": "x"}) == 0.0


def test_the_label_prefers_the_korean_display_name():
    assert tracked_pose.display_label({"display_name": "기린"}, "Giraffe") == "기린"
    assert tracked_pose.display_label({}, "Giraffe") == "Giraffe"


def test_an_agent_rotation_takes_over_from_the_tracked_heading():
    """Spec: when the agent has turned an object, its rotation wins over the camera-tracked yaw."""
    assert tracked_pose.shown_yaw_deg(90.0, (1.0, 0.0, 0.0, 0.0)) == 90.0
    assert tracked_pose.shown_yaw_deg(90.0, (0.7071, 0.0, 0.0, 0.7071)) == 0.0
    assert tracked_pose.shown_yaw_deg(90.0, [1, 0, 0, 0]) == 90.0
