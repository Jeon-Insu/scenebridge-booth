from __future__ import annotations

import math
import hashlib
import json
import signal
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .io import load_json
from .source import OpenCVCaptureSource
from .camera_focus import make_camera_links


def write_preview_image(path: str | Path, image: np.ndarray, *, jpeg_quality: int = 78, links=None) -> None:
    """Atomically publish a BGR preview frame for another process to consume."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("OpenCV could not encode the embedded camera preview")
    temporary = destination.with_name(f".{destination.name}.tmp")
    data = encoded.tobytes()
    temporary.write_bytes(data)
    temporary.replace(destination)
    if links is not None:
        # Readers only use links whose digest matches the exact JPEG they loaded.
        sidecar = destination.with_suffix(".links.json")
        metadata = {**links, "image_sha256": hashlib.sha256(data).hexdigest()}
        pending = sidecar.with_name(f".{sidecar.name}.tmp")
        pending.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
        pending.replace(sidecar)


class MosaicFilePublisher:
    """Publish camera views independently from the slower inference loop."""

    def __init__(
        self,
        sources: list[OpenCVCaptureSource],
        output_path: str | Path,
        *,
        update_hz: float = 5.0,
        tile_width: int = 320,
        tile_height: int = 180,
        jpeg_quality: int = 78,
        layout: str = "grid",
    ) -> None:
        self.sources = list(sources)
        self.output_path = Path(output_path)
        self.update_interval = 1.0 / max(0.5, float(update_hz))
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.jpeg_quality = int(jpeg_quality)
        self.layout = str(layout)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._overlay_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._perception_links = None
        self._detections_by_camera = {}
        self._tracked_by_detection = {}
        self._perception_source_frames: dict[str, np.ndarray] = {}
        self._last_frame_numbers: dict[str, int] = {}
        self._arrival_times = {
            source.camera_id: deque(maxlen=30) for source in self.sources
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="tabletop-camera-mosaic", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
            # Drain any in-flight writer before the caller publishes a frozen bundle.
            with self._publish_lock:
                pass
        else:
            self._paused.clear()

    def update_overlays(
        self, detections_by_camera, tracked_by_detection, source_frames=None, *, frame_number=None, timestamp_ns=None
    ) -> None:
        """Replace one coherent detection/segmentation snapshot."""
        with self._overlay_lock:
            self._detections_by_camera = {
                camera_id: list(detections)
                for camera_id, detections in detections_by_camera.items()
            }
            self._tracked_by_detection = dict(tracked_by_detection)
            if source_frames is not None:
                self._perception_source_frames = {
                    camera_id: frame.copy()
                    for camera_id, frame in source_frames.items()
                    if frame is not None
                }
            self._perception_links = make_camera_links(
                [source.camera_id for source in self.sources], self._perception_source_frames,
                detections_by_camera, tracked_by_detection,
                tile_width=self.tile_width, tile_height=self.tile_height,
                frame_number=frame_number, timestamp_ns=timestamp_ns)

    def _run(self) -> None:
        camera_ids = [source.camera_id for source in self.sources]
        while not self._stop.is_set():
            started = time.monotonic()
            if self._paused.is_set():
                self._stop.wait(min(0.05, self.update_interval))
                continue
            frames: dict[str, np.ndarray] = {}
            now = time.monotonic()
            for source in self.sources:
                packet = source.latest()
                if packet is None or packet.image is None:
                    continue
                frames[source.camera_id] = packet.image
                if self._last_frame_numbers.get(source.camera_id) != packet.frame_number:
                    self._last_frame_numbers[source.camera_id] = packet.frame_number
                    self._arrival_times[source.camera_id].append(now)
            with self._overlay_lock:
                detections_by_camera = dict(self._detections_by_camera)
                tracked_by_detection = dict(self._tracked_by_detection)
                perception_source_frames = dict(self._perception_source_frames)
                links = self._perception_links
            detection_frames: dict[str, np.ndarray] = {}
            segmentation_frames: dict[str, np.ndarray] = {}
            if perception_source_frames:
                from .object_detection import draw_object_detections

                detection_frames = {
                    camera_id: draw_object_detections(
                        frame,
                        detections_by_camera.get(camera_id, []),
                        tracked_by_detection,
                        show_masks=False,
                    )
                    for camera_id, frame in perception_source_frames.items()
                }
                segmentation_frames = {
                    camera_id: draw_object_detections(
                        frame,
                        detections_by_camera.get(camera_id, []),
                        tracked_by_detection,
                        show_boxes=False,
                        show_labels=False,
                        show_anchors=False,
                        mask_alpha=0.62,
                    )
                    for camera_id, frame in perception_source_frames.items()
                }
            fps_by_camera: dict[str, float] = {}
            for camera_id, timestamps in self._arrival_times.items():
                while timestamps and now - timestamps[0] > 2.0:
                    timestamps.popleft()
                fps_by_camera[camera_id] = (
                    (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                    if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]
                    else 0.0
                )
            errors = {source.camera_id: source.error for source in self.sources}
            if self.layout == "raw_and_perception":
                mosaic = make_camera_analysis_mosaic(
                    camera_ids,
                    detection_frames,
                    segmentation_frames,
                    fps_by_camera,
                    errors,
                    tile_width=self.tile_width,
                    tile_height=self.tile_height,
                )
            else:
                mosaic = make_preview_mosaic(
                    camera_ids,
                    segmentation_frames or detection_frames or frames,
                    fps_by_camera,
                    errors,
                    tile_width=self.tile_width,
                    tile_height=self.tile_height,
                )
            try:
                with self._publish_lock:
                    if not self._paused.is_set():
                        write_preview_image(self.output_path, mosaic, jpeg_quality=self.jpeg_quality,
                                            links=links if self.layout == "raw_and_perception" else None)
            except OSError:
                pass
            remaining = self.update_interval - (time.monotonic() - started)
            self._stop.wait(max(0.001, remaining))


def _fit_frame(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image into a fixed tile without changing its aspect ratio."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _preview_tile(
    image: np.ndarray | None,
    *,
    label: str,
    error: str | None,
    width: int,
    height: int,
) -> np.ndarray:
    if image is None:
        tile = np.full((height, width, 3), 24, dtype=np.uint8)
        status = error or "waiting for video..."
        cv2.putText(
            tile,
            status,
            (12, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 180, 255),
            2,
        )
    else:
        tile = _fit_frame(image, width, height)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
    cv2.rectangle(tile, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (9, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 240, 80),
        1,
        cv2.LINE_AA,
    )
    return tile


def make_camera_analysis_mosaic(
    camera_ids: list[str],
    detection_frames: dict[str, np.ndarray],
    segmentation_frames: dict[str, np.ndarray],
    fps_by_camera: dict[str, float],
    errors: dict[str, str | None],
    *,
    tile_width: int = 320,
    tile_height: int = 180,
    detection_label: str = "RAW + OBJECT DETECTION",
    perception_label: str = "INSTANCE SEGMENTATION",
) -> np.ndarray:
    """Build four synchronized detection/segmentation row pairs."""
    rows: list[np.ndarray] = []
    for camera_id in camera_ids:
        fps = fps_by_camera.get(camera_id, 0.0)
        detection = _preview_tile(
            detection_frames.get(camera_id),
            label=f"{camera_id} | {detection_label}",
            error=errors.get(camera_id),
            width=tile_width,
            height=tile_height,
        )
        perception = _preview_tile(
            segmentation_frames.get(camera_id),
            label=f"{camera_id} | {perception_label}",
            error=errors.get(camera_id),
            width=tile_width,
            height=tile_height,
        )
        rows.append(np.hstack([detection, perception]))
    if not rows:
        rows.append(
            np.full((tile_height, tile_width * 2, 3), 24, dtype=np.uint8)
        )
    return np.vstack(rows)


def make_preview_mosaic(
    camera_ids: list[str],
    frames: dict[str, np.ndarray],
    fps_by_camera: dict[str, float],
    errors: dict[str, str | None],
    *,
    tile_width: int = 640,
    tile_height: int = 360,
) -> np.ndarray:
    """Build a labeled two-column preview grid from the latest camera frames."""
    columns = 1 if len(camera_ids) == 1 else 2
    rows = max(1, math.ceil(len(camera_ids) / columns))
    tile_count = rows * columns
    tiles: list[np.ndarray] = []

    for camera_id in camera_ids:
        image = frames.get(camera_id)
        if image is None:
            tile = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
            status = errors.get(camera_id) or "waiting for video..."
            cv2.putText(tile, status, (20, tile_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2)
        else:
            tile = _fit_frame(image, tile_width, tile_height)
        cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1), (80, 80, 80), 1)
        label = f"{camera_id}  {fps_by_camera.get(camera_id, 0.0):.1f} fps"
        cv2.rectangle(tile, (0, 0), (min(tile_width, 370), 42), (0, 0, 0), -1)
        cv2.putText(tile, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 240, 80), 2)
        tiles.append(tile)

    while len(tiles) < tile_count:
        tile = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
        cv2.putText(tile, "Q / ESC: close preview", (30, tile_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)
        tiles.append(tile)

    return np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )


def run_preview(
    config_path: str | Path,
    *,
    headless: bool = False,
    duration_seconds: float | None = None,
) -> dict[str, int]:
    """Display all configured sources in one window and return received frame counts."""
    config = load_json(config_path)
    camera_configs = config["cameras"]
    camera_ids = [str(item["id"]) for item in camera_configs]
    sources = [
        OpenCVCaptureSource(
            camera_id=str(item["id"]),
            uri=item["source"],
            width=item.get("width"),
            height=item.get("height"),
            fps=item.get("fps"),
            fourcc=item.get("fourcc"),
            autofocus=item.get("autofocus"),
            focus=item.get("focus"),
            queue_size=2,
        )
        for item in camera_configs
    ]
    tile_width = int(config.get("preview_tile_width", 640))
    tile_height = int(config.get("preview_tile_height", 360))
    title = str(config.get("preview_window_title", "Multi-camera preview"))

    latest: dict[str, np.ndarray] = {}
    arrival_times = {camera_id: deque(maxlen=120) for camera_id in camera_ids}
    counts = {camera_id: 0 for camera_id in camera_ids}
    stopping = False

    def request_stop(_signal, _frame) -> None:
        nonlocal stopping
        stopping = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    for source in sources:
        source.start()
    if not headless:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print(f"Previewing {len(sources)} cameras. Press Q, ESC, or Ctrl+C to stop.")
    try:
        while not stopping:
            now = time.monotonic()
            if duration_seconds is not None and now - started >= duration_seconds:
                break
            for source in sources:
                newest = None
                while (packet := source.poll()) is not None:
                    newest = packet
                if newest is not None and newest.image is not None:
                    latest[source.camera_id] = newest.image
                    arrival_times[source.camera_id].append(now)
                    counts[source.camera_id] += 1

            fps_by_camera: dict[str, float] = {}
            for camera_id, timestamps in arrival_times.items():
                while timestamps and now - timestamps[0] > 2.0:
                    timestamps.popleft()
                if len(timestamps) >= 2:
                    fps_by_camera[camera_id] = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                else:
                    fps_by_camera[camera_id] = 0.0

            if headless:
                time.sleep(0.005)
                continue
            mosaic = make_preview_mosaic(
                camera_ids,
                latest,
                fps_by_camera,
                {source.camera_id: source.error for source in sources},
                tile_width=tile_width,
                tile_height=tile_height,
            )
            cv2.imshow(title, mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        for source in sources:
            source.stop()
        if not headless:
            cv2.destroyWindow(title)
    print("Received frames: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return counts
