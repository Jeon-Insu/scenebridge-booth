from __future__ import annotations

import json

import numpy as np

from smartphone_mocap.agent_control import (
    PerceptionAgentControl,
    read_json,
    write_control_request,
)


def test_snapshot_request_freezes_one_coherent_result(tmp_path) -> None:
    write_control_request(tmp_path, "snapshot", "request-001")
    control = PerceptionAgentControl(tmp_path)
    control.start_runtime("test")

    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    tracking = {
        "schema": "gist.smartphone_mocap.v1",
        "timestamp_ns": 123,
        "objects": [{"id": "cup/0", "label": "cup", "xyz": [0.1, 0.2, 0.0]}],
    }
    snapshot_path = control.capture_snapshot(
        source="test",
        frame_number=7,
        tracking=tracking,
        detections={"schema": "gist.tabletop_detections.v1"},
        raw_frames={"cam_a": frame},
        detection_frames={"cam_a": frame},
        segmentation_frames={"cam_a": frame},
    )

    assert snapshot_path is not None
    assert control.is_frozen()
    runtime = read_json(tmp_path / "runtime_state.json")
    assert runtime is not None
    assert runtime["mode"] == "snapshot"
    assert runtime["request_id"] == "request-001"
    assert read_json(snapshot_path)["tracking"] == tracking
    camera_files = read_json(snapshot_path)["camera_files"]["cam_a"]
    assert set(camera_files) == {"raw", "detection", "segmentation"}

    write_control_request(tmp_path, "live", "live-001")
    assert not control.is_frozen()


def test_invalid_control_mode_is_rejected(tmp_path) -> None:
    try:
        write_control_request(tmp_path, "paused", "bad")
    except ValueError as exc:
        assert "desired_mode" in str(exc)
    else:
        raise AssertionError("invalid control mode was accepted")
