from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CONTROL_SCHEMA = "gist.tabletop_agent.control.v1"
RUNTIME_SCHEMA = "gist.tabletop_agent.runtime.v1"
SNAPSHOT_SCHEMA = "gist.tabletop_agent.snapshot.v1"


def camera_evidence_metadata(config, calibrations, packets):
    """Persist the actual selected packet times, never claim hardware sync."""
    cameras = {c["id"]: c for c in config.get("cameras", [])}
    result = {}
    for cid, packet in packets.items():
        calibration = calibrations[cid]
        center = -calibration.rotation.T @ calibration.translation
        forward = calibration.rotation.T @ np.array([0.0, 0.0, 1.0])
        result[cid] = {
            "corner": cameras.get(cid, {}).get("corner", "unknown"),
            "position_table_m": center.tolist(), "forward_table": forward.tolist(),
            "rough_calibration_only": config.get("rig", {}).get("rough_calibration_only", True),
            "capture_time_ns": int(packet.capture_time_ns),
            "adjusted_capture_time_ns": int(packet.capture_time_ns + calibration.time_offset_ns),
            "frame_number": int(packet.frame_number),
            "image_size_wh": [int(packet.image.shape[1]), int(packet.image.shape[0])] if packet.image is not None else None,
            "timing_basis": "software capture timestamp (+ calibration time offset), not measured exposure time",
        }
    return result


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class PerceptionAgentControl:
    """File-based LIVE/SNAPSHOT gate shared by Isaac Sim and perception.

    Camera capture threads remain alive while this gate is frozen. The live
    pipeline drains their small queues but skips detector/segmenter inference.
    File messages make the boundary independent of Isaac Sim's Python build.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.control_path = self.root / "control.json"
        self.runtime_path = self.root / "runtime_state.json"
        self.snapshots_dir = self.root / "snapshots"
        self._last_control: dict[str, Any] = {}
        self._last_control_mtime_ns = -1
        self._captured_request_id = ""

    def start_runtime(self, source: str) -> None:
        control = self.read_control()
        request_age_ns = time.time_ns() - int((control or {}).get("requested_at_ns", 0) or 0)
        if (
            control is None
            or str(control.get("desired_mode", "live")) != "snapshot"
            or request_age_ns > 10_000_000_000
        ):
            control = {
                "schema": CONTROL_SCHEMA,
                "desired_mode": "live",
                "request_id": "startup",
                "requested_at_ns": time.time_ns(),
            }
            atomic_write_json(self.control_path, control)
        runtime = {
            "schema": RUNTIME_SCHEMA,
            "mode": "live",
            "source": str(source),
            "request_id": str(control.get("request_id", "startup")),
            "updated_at_ns": time.time_ns(),
            "snapshot_path": None,
        }
        atomic_write_json(self.runtime_path, runtime)

    def read_control(self) -> dict[str, Any] | None:
        try:
            modified_ns = self.control_path.stat().st_mtime_ns
        except OSError:
            return self._last_control or None
        if modified_ns != self._last_control_mtime_ns:
            value = read_json(self.control_path)
            if value and value.get("schema") == CONTROL_SCHEMA:
                self._last_control = value
                self._last_control_mtime_ns = modified_ns
        return self._last_control or None

    def desired_mode(self) -> str:
        control = self.read_control() or {}
        return (
            "snapshot"
            if str(control.get("desired_mode", "live")) == "snapshot"
            else "live"
        )

    def pending_snapshot_request(self) -> str | None:
        control = self.read_control() or {}
        if str(control.get("desired_mode", "live")) != "snapshot":
            return None
        request_id = str(control.get("request_id", ""))
        if not request_id or request_id == self._captured_request_id:
            return None
        return request_id

    def is_frozen(self) -> bool:
        control = self.read_control() or {}
        return (
            str(control.get("desired_mode", "live")) == "snapshot"
            and str(control.get("request_id", "")) == self._captured_request_id
        )

    def mark_live(self, source: str) -> None:
        control = self.read_control() or {}
        atomic_write_json(
            self.runtime_path,
            {
                "schema": RUNTIME_SCHEMA,
                "mode": "live",
                "source": str(source),
                "request_id": str(control.get("request_id", "")),
                "updated_at_ns": time.time_ns(),
                "snapshot_path": None,
            },
        )

    def capture_snapshot(
        self,
        *,
        source: str,
        frame_number: int,
        tracking: dict[str, Any],
        detections: dict[str, Any],
        raw_frames: dict[str, Any],
        detection_frames: dict[str, Any],
        segmentation_frames: dict[str, Any],
        preview_path: str | Path | None = None,
        camera_metadata: dict | None = None,
        rig: dict | None = None,
    ) -> Path | None:
        request_id = self.pending_snapshot_request()
        if request_id is None:
            return None
        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in request_id
        )[:96] or str(time.time_ns())
        snapshot_dir = self.snapshots_dir / safe_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        cameras_dir = snapshot_dir / "cameras"
        cameras_dir.mkdir(parents=True, exist_ok=True)

        camera_files: dict[str, dict[str, str]] = {}
        camera_ids = sorted(
            set(raw_frames) | set(detection_frames) | set(segmentation_frames)
        )
        for camera_id in camera_ids:
            files: dict[str, str] = {}
            for kind, frames in (
                ("raw", raw_frames),
                ("detection", detection_frames),
                ("segmentation", segmentation_frames),
            ):
                image = frames.get(camera_id)
                if image is None:
                    continue
                path = cameras_dir / f"{camera_id}_{kind}.jpg"
                if cv2.imwrite(str(path), image):
                    files[kind] = str(path)
            camera_files[camera_id] = files

        copied_preview: str | None = None
        if preview_path is not None:
            source_preview = Path(preview_path)
            if source_preview.is_file():
                destination = snapshot_dir / "camera_mosaic.jpg"
                try:
                    shutil.copy2(source_preview, destination)
                    copied_preview = str(destination)
                except OSError:
                    copied_preview = None

        snapshot = {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_id": request_id,
            "captured_at_ns": time.time_ns(),
            "source": str(source),
            "frame_number": int(frame_number),
            "tracking": tracking,
            "detections": detections,
            "camera_files": camera_files,
            "camera_mosaic": copied_preview,
            "camera_metadata": camera_metadata or {},
            "rig": rig or {},
        }
        times = [m["adjusted_capture_time_ns"] for m in (camera_metadata or {}).values()
                 if m.get("adjusted_capture_time_ns") is not None]
        snapshot["timing"] = {"basis": "software timestamps; exposure synchronization not guaranteed",
                              "camera_skew_ms": (max(times) - min(times)) / 1e6 if times else None}
        snapshot_path = atomic_write_json(snapshot_dir / "snapshot.json", snapshot)
        self._captured_request_id = request_id
        atomic_write_json(
            self.runtime_path,
            {
                "schema": RUNTIME_SCHEMA,
                "mode": "snapshot",
                "source": str(source),
                "request_id": request_id,
                "frame_number": int(frame_number),
                "tracking_timestamp_ns": int(tracking.get("timestamp_ns", 0) or 0),
                "updated_at_ns": time.time_ns(),
                "snapshot_path": str(snapshot_path),
                "camera_ids": camera_ids,
            },
        )
        return snapshot_path


def write_control_request(
    root: str | Path, desired_mode: str, request_id: str
) -> Path:
    if desired_mode not in {"live", "snapshot"}:
        raise ValueError("desired_mode must be 'live' or 'snapshot'")
    return atomic_write_json(
        Path(root) / "control.json",
        {
            "schema": CONTROL_SCHEMA,
            "desired_mode": desired_mode,
            "request_id": str(request_id),
            "requested_at_ns": time.time_ns(),
        },
    )


def encode_detections(
    frame_number: int,
    timestamp_ns: int,
    detections: dict,
    tracked_by_detection: dict,
) -> dict[str, Any]:
    cameras: dict[str, list[dict[str, Any]]] = {}
    for camera_id, items in detections.items():
        encoded = []
        for item in items:
            tracked = tracked_by_detection.get((camera_id, item.local_detection_id))
            mask_area = int(np.count_nonzero(item.mask)) if item.mask is not None else 0
            mask_anchor = item.anchor_xy(
                "mask_centroid" if item.mask is not None else "center"
            )
            encoded.append(
                {
                    "local_detection_id": int(item.local_detection_id),
                    "local_track_id": (
                        int(item.local_track_id)
                        if item.local_track_id is not None
                        else None
                    ),
                    "label": item.label,
                    "class_id": int(item.class_id),
                    "confidence": float(item.confidence),
                    "box_xyxy": [
                        float(item.box.x1),
                        float(item.box.y1),
                        float(item.box.x2),
                        float(item.box.y2),
                    ],
                    "mask_area_px": mask_area,
                    "mask_anchor_xy": [float(mask_anchor[0]), float(mask_anchor[1])],
                    "tracked_id": tracked.global_id if tracked is not None else None,
                    "tracked_state": tracked.state if tracked is not None else None,
                    "xyz": list(tracked.xyz) if tracked is not None else None,
                }
            )
        cameras[camera_id] = encoded
    return {
        "schema": "gist.tabletop_detections.v1",
        "frame_number": int(frame_number),
        "timestamp_ns": int(timestamp_ns),
        "cameras": cameras,
    }
