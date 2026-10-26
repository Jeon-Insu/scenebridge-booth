from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from .types import CameraCalibration, Joint3D, Keypoint2D


@dataclass(slots=True, frozen=True)
class Observation:
    camera: CameraCalibration
    keypoint: Keypoint2D


def project_point(camera: CameraCalibration, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(1, 1, 3)
    rotation_vector, _ = cv2.Rodrigues(camera.rotation)
    image_points, _ = cv2.projectPoints(
        xyz,
        rotation_vector,
        camera.translation,
        camera.camera_matrix,
        camera.distortion,
    )
    return image_points.reshape(2)


def _normalized_xy(observation: Observation) -> np.ndarray:
    point = np.array([[[observation.keypoint.x, observation.keypoint.y]]], dtype=np.float64)
    normalized = cv2.undistortPoints(
        point,
        observation.camera.camera_matrix,
        observation.camera.distortion,
    )
    return normalized.reshape(2)


def _weighted_dlt(observations: list[Observation]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for observation in observations:
        x, y = _normalized_xy(observation)
        projection = observation.camera.normalized_projection
        weight = max(float(observation.keypoint.confidence), 1e-3) ** 0.5
        rows.append(weight * (x * projection[2] - projection[0]))
        rows.append(weight * (y * projection[2] - projection[1]))
    matrix = np.stack(rows)
    _, _, vt = np.linalg.svd(matrix)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-10:
        raise ValueError("Triangulation produced a point at infinity")
    return homogeneous[:3] / homogeneous[3]


def reprojection_errors(observations: list[Observation], xyz: np.ndarray) -> np.ndarray:
    errors = []
    for observation in observations:
        predicted = project_point(observation.camera, xyz)
        measured = np.array([observation.keypoint.x, observation.keypoint.y], dtype=np.float64)
        errors.append(float(np.linalg.norm(predicted - measured)))
    return np.asarray(errors, dtype=np.float64)


def triangulate_joint(
    observations: list[Observation],
    *,
    min_confidence: float = 0.25,
    max_reprojection_error_px: float = 20.0,
) -> Joint3D | None:
    active = [item for item in observations if item.keypoint.confidence >= min_confidence]
    if len(active) < 2:
        return None

    # Try all pairs first and retain the hypothesis supported by the most views.
    hypotheses: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for first, second in combinations(active, 2):
        try:
            xyz = _weighted_dlt([first, second])
        except ValueError:
            continue
        errors = reprojection_errors(active, xyz)
        inliers = errors <= max_reprojection_error_px
        hypotheses.append((int(inliers.sum()), float(errors[inliers].mean()) if inliers.any() else 1e9, xyz, inliers))

    if not hypotheses:
        return None
    _, _, _, best_inliers = max(hypotheses, key=lambda item: (item[0], -item[1]))
    inlier_observations = [item for item, keep in zip(active, best_inliers, strict=True) if keep]
    if len(inlier_observations) < 2:
        return None

    xyz = _weighted_dlt(inlier_observations)
    errors = reprojection_errors(inlier_observations, xyz)
    mean_error = float(errors.mean())
    confidence = float(
        np.mean([item.keypoint.confidence for item in inlier_observations])
        * np.exp(-mean_error / max(max_reprojection_error_px, 1.0))
    )
    return Joint3D(
        x=float(xyz[0]),
        y=float(xyz[1]),
        z=float(xyz[2]),
        confidence=max(0.0, min(1.0, confidence)),
        reprojection_error_px=mean_error,
        source_cameras=tuple(item.camera.camera_id for item in inlier_observations),
    )

