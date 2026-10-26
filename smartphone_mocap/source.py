from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2

from .types import FramePacket


def parse_capture_uri(value: str | int) -> str | int:
    """Turn a numeric JSON source into an OpenCV camera index."""
    if isinstance(value, int):
        return value
    stripped = str(value).strip()
    return int(stripped) if stripped.isdecimal() else stripped


class OpenCVCaptureSource:
    """Read a USB camera, phone stream URL, or video without blocking the pipeline."""

    def __init__(
        self,
        camera_id: str,
        uri: str | int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        fourcc: str | None = None,
        autofocus: bool | None = None,
        focus: float | None = None,
        queue_size: int = 8,
        reconnect_seconds: float = 1.0,
        loop_file: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.uri = parse_capture_uri(uri)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.autofocus = autofocus
        self.focus = focus
        self.reconnect_seconds = reconnect_seconds
        self.loop_file = loop_file
        self._is_file = isinstance(self.uri, str) and Path(self.uri).is_file()
        self._queue: deque[FramePacket] = deque(maxlen=max(1, queue_size))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._frame_number = 0
        self._latest: FramePacket | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, name=f"capture-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def poll(self) -> FramePacket | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def latest(self) -> FramePacket | None:
        """Return the newest captured packet without consuming the tracking queue."""
        with self._lock:
            return self._latest

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.uri)
        if self.fourcc:
            if len(self.fourcc) != 4:
                raise ValueError(f"fourcc must have four characters, got {self.fourcc!r}")
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        if self.autofocus is not None:
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if self.autofocus else 0.0)
        if self.focus is not None:
            capture.set(cv2.CAP_PROP_FOCUS, float(self.focus))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _reader_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        file_start_ns = time.monotonic_ns()
        try:
            while not self._stop.is_set():
                if capture is None or not capture.isOpened():
                    capture = self._open()
                    if not capture.isOpened():
                        self._error = f"cannot open source {self.uri!r}"
                        capture.release()
                        capture = None
                        self._stop.wait(self.reconnect_seconds)
                        continue
                    self._error = None
                    file_start_ns = time.monotonic_ns()

                ok, image = capture.read()
                arrival_ns = time.monotonic_ns()
                if not ok:
                    if self._is_file and self.loop_file:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        file_start_ns = time.monotonic_ns()
                        continue
                    if self._is_file:
                        break
                    capture.release()
                    capture = None
                    self._stop.wait(self.reconnect_seconds)
                    continue

                if self._is_file:
                    video_ms = max(0.0, capture.get(cv2.CAP_PROP_POS_MSEC))
                    capture_ns = file_start_ns + int(video_ms * 1_000_000)
                else:
                    # Phone streams usually do not expose a capture timestamp through OpenCV.
                    # Arrival time is sufficient for the first proof-of-concept; offsets are
                    # configurable and hardware/PTP timestamps can replace this later.
                    capture_ns = arrival_ns
                packet = FramePacket(
                    camera_id=self.camera_id,
                    frame_number=self._frame_number,
                    capture_time_ns=capture_ns,
                    image=image,
                    metadata={"uri": str(self.uri), "arrival_time_ns": arrival_ns},
                )
                self._frame_number += 1
                with self._lock:
                    self._queue.append(packet)
                    self._latest = packet
        finally:
            if capture is not None:
                capture.release()
