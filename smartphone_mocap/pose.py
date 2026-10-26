from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Keypoint2D, Person2D


COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class UltralyticsPoseEstimator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | int | None = None,
        image_size: int = 640,
        detection_confidence: float = 0.25,
        keypoint_confidence: float = 0.15,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Pose dependencies are missing. Run tools/setup_smartphone_mocap.sh first."
            ) from exc
        self.model = YOLO(str(model_path))
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.detection_confidence = float(detection_confidence)
        self.keypoint_confidence = float(keypoint_confidence)

    def infer(self, camera_id: str, image: np.ndarray) -> list[Person2D]:
        result = self.model.predict(
            source=image,
            device=self.device,
            imgsz=self.image_size,
            conf=self.detection_confidence,
            verbose=False,
        )[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return []
        xy = result.keypoints.xy.detach().cpu().numpy()
        if result.keypoints.conf is None:
            confidences = np.ones(xy.shape[:2], dtype=np.float32)
        else:
            confidences = result.keypoints.conf.detach().cpu().numpy()
        box_confidences = (
            result.boxes.conf.detach().cpu().numpy()
            if result.boxes is not None and result.boxes.conf is not None
            else np.ones(len(xy), dtype=np.float32)
        )
        people: list[Person2D] = []
        for person_index, (points, scores, box_confidence) in enumerate(
            zip(xy, confidences, box_confidences, strict=True)
        ):
            joints = {
                name: Keypoint2D(float(point[0]), float(point[1]), float(score))
                for name, point, score in zip(COCO_KEYPOINTS, points, scores, strict=True)
                if score >= self.keypoint_confidence
            }
            people.append(
                Person2D(
                    camera_id=camera_id,
                    local_track_id=person_index,
                    joints=joints,
                    confidence=float(box_confidence),
                )
            )
        return people

