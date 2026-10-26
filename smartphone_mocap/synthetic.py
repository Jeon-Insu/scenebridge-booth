from __future__ import annotations

import math

import cv2
import numpy as np

from .calibration import look_at_world_to_camera
from .geometry import project_point
from .pose import COCO_KEYPOINTS
from .types import CameraCalibration, Keypoint2D, Person2D


BASE_SKELETON = {
    "nose": (0.00, 0.00, 1.70),
    "left_eye": (-0.03, 0.00, 1.74),
    "right_eye": (0.03, 0.00, 1.74),
    "left_ear": (-0.08, 0.00, 1.72),
    "right_ear": (0.08, 0.00, 1.72),
    "left_shoulder": (-0.22, 0.00, 1.48),
    "right_shoulder": (0.22, 0.00, 1.48),
    "left_elbow": (-0.38, 0.00, 1.20),
    "right_elbow": (0.38, 0.00, 1.20),
    "left_wrist": (-0.43, 0.00, 0.94),
    "right_wrist": (0.43, 0.00, 0.94),
    "left_hip": (-0.14, 0.00, 1.00),
    "right_hip": (0.14, 0.00, 1.00),
    "left_knee": (-0.15, 0.00, 0.55),
    "right_knee": (0.15, 0.00, 0.55),
    "left_ankle": (-0.16, 0.00, 0.08),
    "right_ankle": (0.16, 0.00, 0.08),
}


def make_four_camera_rig(image_size: tuple[int, int] = (1280, 720)) -> dict[str, CameraCalibration]:
    width, height = image_size
    focal = 0.9 * width
    matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    positions = {
        "cam_0": (-1.45, -1.45, 1.65),
        "cam_1": (1.45, -1.45, 1.65),
        "cam_2": (1.45, 1.45, 1.65),
        "cam_3": (-1.45, 1.45, 1.65),
    }
    return {
        camera_id: CameraCalibration(
            camera_id=camera_id,
            image_size=image_size,
            camera_matrix=matrix.copy(),
            distortion=np.zeros(5),
            world_to_camera=look_at_world_to_camera(np.array(position), np.array([0.0, 0.0, 1.0])),
        )
        for camera_id, position in positions.items()
    }


def synthetic_joint_positions(frame_number: int, fps: float = 15.0) -> dict[str, np.ndarray]:
    t = frame_number / fps
    offset = np.array([0.45 * math.sin(0.6 * t), 0.30 * math.cos(0.5 * t), 0.0])
    return {name: np.asarray(BASE_SKELETON[name]) + offset for name in COCO_KEYPOINTS}


def project_synthetic_person(
    camera: CameraCalibration,
    positions: dict[str, np.ndarray],
    *,
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Person2D:
    rng = rng or np.random.default_rng(0)
    joints = {}
    for name, xyz in positions.items():
        point = project_point(camera, xyz)
        if noise_px:
            point += rng.normal(0.0, noise_px, 2)
        joints[name] = Keypoint2D(float(point[0]), float(point[1]), 0.98)
    return Person2D(camera_id=camera.camera_id, local_track_id=0, joints=joints, confidence=0.99)


def render_synthetic_frame(
    camera: CameraCalibration,
    positions: dict[str, np.ndarray],
) -> np.ndarray:
    width, height = camera.image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for xyz in positions.values():
        x, y = project_point(camera, xyz).round().astype(int)
        cv2.circle(image, (x, y), 4, (80, 230, 100), -1)
    return image
