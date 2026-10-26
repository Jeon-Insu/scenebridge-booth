from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from .source import OpenCVCaptureSource


class MultiCameraSessionRecorder:
    """Record aligned latest-frame samples from every active camera.

    Each AVI uses the same recording index.  The accompanying JSONL retains
    the original per-camera frame numbers and capture timestamps, so an
    offline replay can reason about the real arrival-time skew instead of
    assuming that USB cameras were hardware synchronized.
    """

    def __init__(
        self,
        sources: list[OpenCVCaptureSource],
        output_root: str | Path,
        *,
        session_label: str = "tabletop_session",
        fps: float = 5.0,
        codec: str = "MJPG",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(codec) != 4:
            raise ValueError("Recording codec must be a four-character code")
        self.sources = list(sources)
        self.output_root = Path(output_root)
        self.session_label = self._safe_label(session_label)
        self.fps = max(0.5, float(fps))
        self.codec = codec
        self.metadata = dict(metadata or {})
        self.session_dir = self._make_session_dir()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._sample_count = 0
        self.error: str | None = None
        self.started_at = datetime.now().astimezone()

    @staticmethod
    def _safe_label(value: str) -> str:
        cleaned = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(value).strip()
        ).strip("_")
        return cleaned or "tabletop_session"

    def _make_session_dir(self) -> Path:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        base = self.output_root / f"{stamp}_{self.session_label}"
        destination = base
        suffix = 1
        while destination.exists():
            destination = base.with_name(f"{base.name}_{suffix:02d}")
            suffix += 1
        destination.mkdir(parents=True, exist_ok=False)
        return destination

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._write_manifest(completed=False)
        self._thread = threading.Thread(
            target=self._run, name="tabletop-session-recorder", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self._write_manifest(completed=self.error is None)

    def _writer_for(self, camera_id: str, image) -> cv2.VideoWriter:
        writer = self._writers.get(camera_id)
        if writer is not None:
            return writer
        height, width = image.shape[:2]
        output_path = self.session_dir / f"{camera_id}.avi"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open video writer for {output_path}")
        self._writers[camera_id] = writer
        self._sizes[camera_id] = (width, height)
        return writer

    def _run(self) -> None:
        interval = 1.0 / self.fps
        timestamps_path = self.session_dir / "camera_timestamps.jsonl"
        try:
            with timestamps_path.open("w", encoding="utf-8") as timestamps:
                while not self._stop.is_set():
                    started = time.monotonic()
                    packets = {
                        source.camera_id: source.latest() for source in self.sources
                    }
                    if all(
                        packet is not None and packet.image is not None
                        for packet in packets.values()
                    ):
                        camera_metadata: dict[str, dict[str, Any]] = {}
                        for source in self.sources:
                            packet = packets[source.camera_id]
                            assert packet is not None and packet.image is not None
                            image = packet.image
                            writer = self._writer_for(source.camera_id, image)
                            expected_size = self._sizes[source.camera_id]
                            if image.shape[1::-1] != expected_size:
                                image = cv2.resize(image, expected_size)
                            writer.write(image)
                            camera_metadata[source.camera_id] = {
                                "source_frame_number": int(packet.frame_number),
                                "capture_time_ns": int(packet.capture_time_ns),
                                "arrival_time_ns": int(
                                    packet.metadata.get(
                                        "arrival_time_ns", packet.capture_time_ns
                                    )
                                ),
                            }
                        timestamps.write(
                            json.dumps(
                                {
                                    "recording_index": self._sample_count,
                                    "recorded_monotonic_ns": time.monotonic_ns(),
                                    "recorded_wall_time_ns": time.time_ns(),
                                    "cameras": camera_metadata,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        timestamps.flush()
                        self._sample_count += 1
                    remaining = interval - (time.monotonic() - started)
                    self._stop.wait(max(0.001, remaining))
        except Exception as exc:  # Keep perception alive if disk/codec recording fails.
            self.error = f"{type(exc).__name__}: {exc}"

    def _write_manifest(self, *, completed: bool) -> None:
        payload = {
            "schema": "gist.tabletop_recording.v1",
            "session_label": self.session_label,
            "session_dir": str(self.session_dir),
            "started_at": self.started_at.isoformat(),
            "ended_at": datetime.now().astimezone().isoformat() if completed else None,
            "completed": completed,
            "record_fps": self.fps,
            "codec": self.codec,
            "sample_count": self._sample_count,
            "camera_ids": [source.camera_id for source in self.sources],
            "video_files": {
                source.camera_id: f"{source.camera_id}.avi"
                for source in self.sources
            },
            "timestamp_file": "camera_timestamps.jsonl",
            "tracking_file": "tracking.jsonl",
            "error": self.error,
            **self.metadata,
        }
        temporary = self.session_dir / ".session.json.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.session_dir / "session.json")
