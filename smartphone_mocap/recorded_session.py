from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .types import FramePacket


@dataclass(slots=True)
class RecordedSample:
    recording_index: int
    recorded_monotonic_ns: int
    recorded_wall_time_ns: int
    packets: dict[str, FramePacket]


class RecordedSession:
    """Read a recorded multi-camera session in deterministic lockstep."""

    def __init__(self, session_dir: str | Path) -> None:
        self.root = Path(session_dir).resolve()
        manifest_path = self.root / "session.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "gist.tabletop_recording.v1":
            raise ValueError(f"Unsupported recording schema in {manifest_path}")
        self.camera_ids = tuple(str(value) for value in self.manifest["camera_ids"])
        timestamp_path = self.root / str(
            self.manifest.get("timestamp_file", "camera_timestamps.jsonl")
        )
        self.timestamp_rows = [
            json.loads(line)
            for line in timestamp_path.open(encoding="utf-8")
            if line.strip()
        ]
        video_files = self.manifest.get("video_files", {})
        self.video_paths = {
            camera_id: self.root / str(video_files[camera_id])
            for camera_id in self.camera_ids
        }
        self._captures: dict[str, cv2.VideoCapture] = {}

    def __enter__(self) -> "RecordedSession":
        self._open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _open(self) -> None:
        if self._captures:
            return
        for camera_id, path in self.video_paths.items():
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                self.close()
                raise RuntimeError(f"Could not open recorded camera video: {path}")
            self._captures[camera_id] = capture

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def validate(self) -> dict:
        self._open()
        errors: list[str] = []
        warnings: list[str] = []
        expected_count = int(self.manifest.get("sample_count", len(self.timestamp_rows)))
        if not self.manifest.get("completed", False):
            warnings.append("Recording manifest is not marked completed")
        if expected_count != len(self.timestamp_rows):
            errors.append(
                f"Manifest sample_count={expected_count}, timestamp rows={len(self.timestamp_rows)}"
            )

        indices = [int(row.get("recording_index", -1)) for row in self.timestamp_rows]
        if indices != list(range(len(indices))):
            errors.append("Recording indices are not contiguous from zero")
        monotonic_times = [
            int(row.get("recorded_monotonic_ns", 0)) for row in self.timestamp_rows
        ]
        if any(second <= first for first, second in zip(monotonic_times, monotonic_times[1:])):
            errors.append("Recorded monotonic timestamps are not strictly increasing")
        for row_index, row in enumerate(self.timestamp_rows):
            missing = set(self.camera_ids) - set(row.get("cameras", {}))
            if missing:
                errors.append(
                    f"Timestamp row {row_index} is missing cameras: {sorted(missing)}"
                )
                break

        video_frame_counts: dict[str, int] = {}
        video_fps: dict[str, float] = {}
        video_resolutions: dict[str, list[int]] = {}
        for camera_id, capture in self._captures.items():
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            video_frame_counts[camera_id] = frame_count
            video_fps[camera_id] = float(capture.get(cv2.CAP_PROP_FPS))
            video_resolutions[camera_id] = [
                int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            ]
            if frame_count != len(self.timestamp_rows):
                errors.append(
                    f"{camera_id} has {frame_count} frames; expected {len(self.timestamp_rows)}"
                )
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "timestamp_rows": len(self.timestamp_rows),
            "video_frame_counts": video_frame_counts,
            "video_fps": video_fps,
            "video_resolutions": video_resolutions,
        }

    def iter_samples(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ) -> Iterator[RecordedSample]:
        if start < 0:
            raise ValueError("start must be non-negative")
        if stride < 1:
            raise ValueError("stride must be at least one")
        self._open()
        end = len(self.timestamp_rows) if stop is None else min(stop, len(self.timestamp_rows))
        if end < start:
            raise ValueError("stop must not be less than start")
        for capture in self._captures.values():
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                # MJPG AVI normally supports seeking, but sequential fallback is
                # safer than silently pairing frames with the wrong timestamps.
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for _ in range(start):
                    ok, _image = capture.read()
                    if not ok:
                        raise EOFError("Recorded video ended while seeking")

        for row_position in range(start, end):
            row = self.timestamp_rows[row_position]
            images: dict[str, np.ndarray] = {}
            for camera_id, capture in self._captures.items():
                ok, image = capture.read()
                if not ok or image is None:
                    raise EOFError(
                        f"{camera_id} ended before recorded sample {row_position}"
                    )
                images[camera_id] = image
            if (row_position - start) % stride:
                continue
            packets: dict[str, FramePacket] = {}
            for camera_id in self.camera_ids:
                camera_metadata = row["cameras"][camera_id]
                packets[camera_id] = FramePacket(
                    camera_id=camera_id,
                    frame_number=int(camera_metadata["source_frame_number"]),
                    capture_time_ns=int(camera_metadata["capture_time_ns"]),
                    image=images[camera_id],
                    metadata={
                        "arrival_time_ns": int(
                            camera_metadata.get(
                                "arrival_time_ns", camera_metadata["capture_time_ns"]
                            )
                        ),
                        "recording_index": int(row["recording_index"]),
                    },
                )
            yield RecordedSample(
                recording_index=int(row["recording_index"]),
                recorded_monotonic_ns=int(row["recorded_monotonic_ns"]),
                recorded_wall_time_ns=int(row["recorded_wall_time_ns"]),
                packets=packets,
            )

