import json
from pathlib import Path

import cv2
import numpy as np

from smartphone_mocap.recorded_session import RecordedSession


def _write_video(path: Path, values: list[int]) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 12)
    )
    assert writer.isOpened()
    for value in values:
        writer.write(np.full((12, 16, 3), value, dtype=np.uint8))
    writer.release()


def _make_session(root: Path) -> None:
    cameras = ["left", "right"]
    for offset, camera_id in enumerate(cameras):
        _write_video(root / f"{camera_id}.avi", [20 + offset, 40 + offset, 60 + offset])
    rows = []
    for index in range(3):
        rows.append(
            {
                "recording_index": index,
                "recorded_monotonic_ns": 1_000_000_000 + index * 200_000_000,
                "recorded_wall_time_ns": 2_000_000_000 + index * 200_000_000,
                "cameras": {
                    camera_id: {
                        "source_frame_number": index * 2 + camera_index,
                        "capture_time_ns": 3_000_000_000
                        + index * 200_000_000
                        + camera_index * 5_000_000,
                        "arrival_time_ns": 3_000_000_000
                        + index * 200_000_000
                        + camera_index * 5_000_000,
                    }
                    for camera_index, camera_id in enumerate(cameras)
                },
            }
        )
    (root / "camera_timestamps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "session.json").write_text(
        json.dumps(
            {
                "schema": "gist.tabletop_recording.v1",
                "completed": True,
                "sample_count": 3,
                "record_fps": 5.0,
                "camera_ids": cameras,
                "video_files": {camera_id: f"{camera_id}.avi" for camera_id in cameras},
                "timestamp_file": "camera_timestamps.jsonl",
                "tracking_file": "tracking.jsonl",
            }
        ),
        encoding="utf-8",
    )


def test_recorded_session_reads_aligned_frames_and_original_timestamps(tmp_path):
    _make_session(tmp_path)

    with RecordedSession(tmp_path) as session:
        integrity = session.validate()
        samples = list(session.iter_samples())

    assert integrity["valid"] is True
    assert integrity["timestamp_rows"] == 3
    assert integrity["video_frame_counts"] == {"left": 3, "right": 3}
    assert [sample.recording_index for sample in samples] == [0, 1, 2]
    assert samples[1].packets["left"].frame_number == 2
    assert samples[1].packets["right"].capture_time_ns == 3_205_000_000
    assert int(np.mean(samples[2].packets["left"].image)) == 60


def test_recorded_session_stride_preserves_recording_indices(tmp_path):
    _make_session(tmp_path)

    with RecordedSession(tmp_path) as session:
        samples = list(session.iter_samples(start=1, stop=3, stride=2))

    assert [sample.recording_index for sample in samples] == [1]

