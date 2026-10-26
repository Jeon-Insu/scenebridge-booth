from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FramePacket:
    camera_id: str
    frame_number: int
    capture_time_ns: int
    image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Keypoint2D:
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class Person2D:
    camera_id: str
    local_track_id: int
    joints: dict[str, Keypoint2D]
    confidence: float = 1.0


@dataclass(slots=True, frozen=True)
class Joint3D:
    x: float
    y: float
    z: float
    confidence: float
    reprojection_error_px: float
    source_cameras: tuple[str, ...]


@dataclass(slots=True)
class Person3D:
    global_id: str
    joints: dict[str, Joint3D]
    state: str = "confirmed"

    def root_xyz(self) -> tuple[float, float, float] | None:
        left = self.joints.get("left_hip")
        right = self.joints.get("right_hip")
        if left and right:
            return (
                (left.x + right.x) * 0.5,
                (left.y + right.y) * 0.5,
                (left.z + right.z) * 0.5,
            )
        if self.joints:
            joint = next(iter(self.joints.values()))
            return (joint.x, joint.y, joint.z)
        return None


@dataclass(slots=True, frozen=True)
class BoundingBox2D:
    x1: float
    y1: float
    x2: float
    y2: float

    def anchor_xy(self, anchor: str = "bottom_center") -> tuple[float, float]:
        x = (self.x1 + self.x2) * 0.5
        if anchor == "center":
            return x, (self.y1 + self.y2) * 0.5
        if anchor == "bottom_center":
            return x, self.y2
        raise ValueError(f"Unsupported object anchor: {anchor}")


@dataclass(slots=True, frozen=True)
class Object2D:
    camera_id: str
    local_detection_id: int
    label: str
    class_id: int
    box: BoundingBox2D
    confidence: float
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    appearance: tuple[float, ...] | None = field(default=None, compare=False, repr=False)
    representative_crop: np.ndarray | None = field(default=None, compare=False, repr=False)
    local_track_id: int | None = None
    # Registered prop this detection names, when a prop-level detector produced it.
    asset_id: str | None = None

    def anchor_xy(self, anchor: str = "bottom_center") -> tuple[float, float]:
        """Return a box or instance-mask anchor in image pixels."""
        if self.mask is not None and anchor in {"mask_centroid", "mask_bottom_center"}:
            ys, xs = np.nonzero(self.mask)
            if len(xs):
                if anchor == "mask_centroid":
                    return float(np.mean(xs)), float(np.mean(ys))
                bottom_cutoff = np.percentile(ys, 95.0)
                bottom_xs = xs[ys >= bottom_cutoff]
                return float(np.median(bottom_xs)), float(bottom_cutoff)
        fallback = "center" if anchor == "mask_centroid" else "bottom_center"
        return self.box.anchor_xy(fallback if anchor.startswith("mask_") else anchor)


@dataclass(slots=True, frozen=True)
class SceneObject3D:
    global_id: str
    label: str
    xyz: tuple[float, float, float]
    confidence: float
    reprojection_error_px: float
    source_cameras: tuple[str, ...]
    asset_uri: str | None = None
    asset_key: str | None = None
    state: str = "confirmed"
    appearance: tuple[float, ...] | None = field(default=None, compare=False, repr=False)
    representative_crops: tuple[np.ndarray, ...] = field(
        default_factory=tuple, compare=False, repr=False
    )
    position_mode: str = "triangulated"
    source_detection_keys: tuple[tuple[str, int], ...] = field(
        default_factory=tuple, compare=False, repr=False
    )
    source_track_keys: tuple[tuple[str, int], ...] = field(
        default_factory=tuple, compare=False, repr=False
    )
    yaw_deg: float | None = None        # table-frame heading (v1.2); None for symmetric props
    pose: str | None = None             # upright / lying / upside_down, recorded only (v1.2)
    display_name: str | None = None     # Korean name from the booth list (v1.2)


@dataclass(slots=True)
class CameraCalibration:
    camera_id: str
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    world_to_camera: np.ndarray
    time_offset_ns: int = 0

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        self.world_to_camera = np.asarray(self.world_to_camera, dtype=np.float64).reshape(4, 4)

    @property
    def rotation(self) -> np.ndarray:
        return self.world_to_camera[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.world_to_camera[:3, 3]

    @property
    def normalized_projection(self) -> np.ndarray:
        return self.world_to_camera[:3, :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "camera_id": self.camera_id,
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.tolist(),
            "world_to_camera": self.world_to_camera.tolist(),
            "time_offset_ns": self.time_offset_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            camera_id=str(data["camera_id"]),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion=np.asarray(data.get("distortion", []), dtype=np.float64),
            world_to_camera=np.asarray(data["world_to_camera"], dtype=np.float64),
            time_offset_ns=int(data.get("time_offset_ns", 0)),
        )
