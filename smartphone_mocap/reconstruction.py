from __future__ import annotations

from .geometry import Observation, triangulate_joint
from .pose import COCO_KEYPOINTS
from .types import CameraCalibration, Person2D, Person3D


def select_primary_person(people: list[Person2D]) -> Person2D | None:
    """MVP association: pick the strongest/largest-confidence person per view."""
    return max(people, key=lambda person: (person.confidence, len(person.joints)), default=None)


def reconstruct_primary_person(
    people_by_camera: dict[str, list[Person2D]],
    calibrations: dict[str, CameraCalibration],
    *,
    global_id: str = "visitor_0",
    min_keypoint_confidence: float = 0.25,
    max_reprojection_error_px: float = 25.0,
) -> Person3D | None:
    selected = {
        camera_id: person
        for camera_id, people in people_by_camera.items()
        if (person := select_primary_person(people)) is not None
    }
    joints = {}
    for joint_name in COCO_KEYPOINTS:
        observations = [
            Observation(calibrations[camera_id], person.joints[joint_name])
            for camera_id, person in selected.items()
            if camera_id in calibrations and joint_name in person.joints
        ]
        joint = triangulate_joint(
            observations,
            min_confidence=min_keypoint_confidence,
            max_reprojection_error_px=max_reprojection_error_px,
        )
        if joint is not None:
            joints[joint_name] = joint
    return Person3D(global_id=global_id, joints=joints) if joints else None

