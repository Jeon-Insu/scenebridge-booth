from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .types import Object2D


class _TrackTrackAdapter:
    """Optional CPU association experiment; no ReID downloads or image inference.

    Feed post-detection boxes, and return only IDs attached to ORIGINAL rows.
    Smoothed tracker boxes must not replace the masks/anchors used for geometry.
    Loose-NMS recovery is deliberately unavailable for this segmentation route.
    """

    def __init__(self, **settings) -> None:
        from types import SimpleNamespace

        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.track_tracker import TRACKTRACK
        from ultralytics.utils import ROOT, YAML

        options = YAML.load(ROOT / "cfg/trackers/tracktrack.yaml")
        activation = settings["track_activation_threshold"]
        options.update(
            gmc_method="none", with_reid=False,
            track_high_thresh=activation,
            track_low_thresh=min(0.1, activation),
            new_track_thresh=activation + 0.1 if activation <= 0.9 else activation,
            # Match Supervision's effective buffer, not its 30-FPS base value.
            track_buffer=int(settings["frame_rate"] / 30 * settings["lost_track_buffer"]),
            min_track_len=settings["minimum_consecutive_frames"],
        )
        # Keep TrackTrack's native match cost threshold: its cost is not
        # numerically interchangeable with ByteTrack's IoU cost threshold.
        self._tracker = TRACKTRACK(SimpleNamespace(**options))
        self._boxes_class = Boxes

    def update_with_detections(self, detections):
        data = np.empty((len(detections), 6), dtype=np.float32)
        if len(detections):
            data[:, :4] = detections.xyxy
            data[:, 4] = detections.confidence
            data[:, 5] = detections.class_id
        # orig_shape is metadata only; TRACKTRACK consumes pixel xywh directly.
        rows = self._tracker.update(self._boxes_class(data, orig_shape=(1, 1)))
        if rows.size == 0:
            output = detections[np.empty(0, dtype=np.int64)]
            output.tracker_id = np.empty(0, dtype=np.int64)
            return output
        if rows.ndim != 2 or rows.shape[1] != 8:
            raise RuntimeError("Unsupported TrackTrack output schema; expected Nx8")
        indices = rows[:, 7].astype(np.int64)
        if (np.any(indices < 0) or np.any(indices >= len(detections))
                or len(set(indices)) != len(indices)
                or not np.array_equal(indices, rows[:, 7])):
            raise RuntimeError("TrackTrack returned invalid detection-row indices")
        output = detections[indices]
        output.tracker_id = rows[:, 4].astype(np.int64)
        return output


class PerCameraObjectTracker:
    """Assign persistent camera-local IDs before multi-view reconstruction.

    One tracker instance is maintained per camera and semantic label. Keeping
    labels separate prevents a temporarily confused detector from linking a cup
    track to a book track.  Untracked first-frame detections are retained so
    cross-view reconstruction does not have to wait for the local tracker.
    """

    def __init__(
        self,
        *,
        backend: str = "bytetrack",
        frame_rate: float = 5.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        if backend not in {"bytetrack", "tracktrack"}:
            raise ValueError(f"Unsupported local tracking backend: {backend}")
        self.backend = backend
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "Local tracking dependencies are missing. "
                "Run tools/setup_smartphone_mocap.sh first."
            ) from exc
        # Supervision 0.30 exposes ByteTrack through a compatibility proxy.
        # Resolve its wrapped class to avoid one warning per camera/class.
        self._byte_track_class = getattr(sv.ByteTrack, "wrapped", sv.ByteTrack)
        self._detections_class = sv.Detections
        self._settings = {
            "frame_rate": float(frame_rate),
            "track_activation_threshold": float(track_activation_threshold),
            "lost_track_buffer": int(lost_track_buffer),
            "minimum_matching_threshold": float(minimum_matching_threshold),
            "minimum_consecutive_frames": int(minimum_consecutive_frames),
        }
        self._trackers: dict[tuple[str, str], object] = {}

    def _tracker(self, camera_id: str, label: str):
        key = (camera_id, label)
        tracker = self._trackers.get(key)
        if tracker is None:
            factory = self._byte_track_class if self.backend == "bytetrack" else _TrackTrackAdapter
            tracker = factory(**self._settings)
            self._trackers[key] = tracker
        return tracker

    def update(
        self, detections_by_camera: dict[str, list[Object2D]]
    ) -> dict[str, list[Object2D]]:
        tracked_by_camera: dict[str, list[Object2D]] = {}
        for camera_id, detections in detections_by_camera.items():
            grouped: dict[str, list[Object2D]] = defaultdict(list)
            for detection in detections:
                grouped[detection.label].append(detection)

            # Age label trackers even when their class is absent in this frame.
            known_labels = {
                label
                for tracked_camera_id, label in self._trackers
                if tracked_camera_id == camera_id
            }
            output_by_detection_id: dict[int, Object2D] = {
                item.local_detection_id: item for item in detections
            }
            for label in sorted(known_labels | set(grouped)):
                items = grouped.get(label, [])
                if items:
                    tracker_input = self._detections_class(
                        xyxy=np.asarray(
                            [
                                [item.box.x1, item.box.y1, item.box.x2, item.box.y2]
                                for item in items
                            ],
                            dtype=np.float32,
                        ),
                        confidence=np.asarray(
                            [item.confidence for item in items], dtype=np.float32
                        ),
                        class_id=np.asarray(
                            [item.class_id for item in items], dtype=np.int64
                        ),
                        data={
                            "local_detection_id": np.asarray(
                                [item.local_detection_id for item in items],
                                dtype=np.int64,
                            )
                        },
                    )
                else:
                    tracker_input = self._detections_class.empty()
                    tracker_input.tracker_id = np.asarray([], dtype=np.int64)
                tracked = self._tracker(camera_id, label).update_with_detections(
                    tracker_input
                )
                if not items or tracked.tracker_id is None:
                    continue
                detection_ids = tracked.data.get("local_detection_id", [])
                for detection_id, track_id in zip(
                    detection_ids, tracked.tracker_id, strict=True
                ):
                    key = int(detection_id)
                    original = output_by_detection_id.get(key)
                    if original is not None:
                        output_by_detection_id[key] = replace(
                            original, local_track_id=int(track_id)
                        )
            tracked_by_camera[camera_id] = [
                output_by_detection_id[item.local_detection_id] for item in detections
            ]
        return tracked_by_camera
