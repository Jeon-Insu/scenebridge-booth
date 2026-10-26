# smartphone_mocap/recognition/orientation.py
"""방향: 자세(서 있음/누움/뒤집힘)와 테이블 기준 yaw. 카메라별 '카메라 기준 8방향'을 테이블 yaw로 바꿔 합친다."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import POSES, Orientation

BINS = 8
STEP = 360.0 / BINS


def camera_azimuth_deg(calibration, xyz) -> float:
    W = np.asarray(calibration.world_to_camera)
    centre = -W[:3, :3].T @ W[:3, 3]
    d = np.asarray(xyz[:2], float) - centre[:2]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def relative_bin(yaw_deg: float, azimuth_deg: float) -> int:
    return int(round(((yaw_deg - azimuth_deg) % 360.0) / STEP)) % BINS


def table_yaw(bin_index: int, azimuth_deg: float) -> float:
    return float((azimuth_deg + bin_index * STEP) % 360.0)


def circular_mean_deg(values, weights) -> float | None:
    if not len(values) or not np.sum(weights):
        return None
    r = np.radians(np.asarray(values, float))
    w = np.asarray(weights, float)
    value = float(np.degrees(np.arctan2((w * np.sin(r)).sum(), (w * np.cos(r)).sum())) % 360.0)
    return 0.0 if value >= 360.0 - 1e-9 else value      # a hair below 0° wraps to 360.0


def fuse(per_camera) -> Orientation:
    if not per_camera:
        return Orientation(None, None, {})
    pose_scores = {p: float(np.sum([np.log(max(c[0].get(p, 0.0), 1e-6)) for c in per_camera])) for p in POSES}
    pose = max(pose_scores, key=pose_scores.get)
    yaws, weights = [], []
    for _, bins, azimuth in per_camera:
        for index, prob in bins.items():
            yaws.append(table_yaw(int(index), azimuth))
            weights.append(prob)
    yaw = circular_mean_deg(yaws, weights)
    return Orientation(pose, None if yaw is None else round(yaw, 1), pose_scores)


def _turn(points, pose_turn, yaw_deg):
    """Rest the model on the table in a pose ((axis, degrees) roll) and turn it by yaw about Z."""
    axis, angle = pose_turn
    a = np.radians(angle)
    c, s = np.cos(a), np.sin(a)
    R = (np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x"
         else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    p = points @ R.T
    p[:, 2] -= p[:, 2].min()
    y = np.radians(yaw_deg)
    Z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return p @ Z.T


SILHOUETTE_POSES = (("upright", ("x", 0.0)), ("lying", ("x", 90.0)), ("lying", ("x", -90.0)),
                    ("lying", ("y", 90.0)), ("lying", ("y", -90.0)), ("upside_down", ("x", 180.0)))


class SilhouetteOrientation:
    """Try every pose × yaw of the prop's 3D points; keep the one whose outlines best fit the masks."""

    def __init__(self, points: dict, yaw_step: float = 15.0):
        self.points, self.yaw_step = points, yaw_step

    def orient(self, crops, calibrations, xyz, asset_id, full_masks) -> Orientation:
        if asset_id not in self.points or not full_masks:
            return Orientation(None, None, {})
        best = (-1.0, None, None)
        for pose, turn in SILHOUETTE_POSES:
            for yaw in np.arange(0.0, 360.0, self.yaw_step):
                world = _turn(self.points[asset_id], turn, yaw) + np.array([xyz[0], xyz[1], 0.0])
                score = 0.0
                for cam, mask in full_masks.items():
                    cal = calibrations[cam]
                    W = np.asarray(cal.world_to_camera)
                    px = cv2.projectPoints(world, cv2.Rodrigues(W[:3, :3])[0], W[:3, 3],
                                           cal.camera_matrix, cal.distortion)[0].reshape(-1, 2)
                    drawn = np.zeros(mask.shape, np.uint8)
                    cv2.fillConvexPoly(drawn, cv2.convexHull(px.astype(np.int32)), 1)
                    union = np.logical_or(drawn, mask).sum()
                    score += np.logical_and(drawn, mask).sum() / union if union else 0.0
                if score > best[0]:
                    best = (float(score), pose, float(yaw))
        return Orientation(best[1], best[2], {"iou_sum": best[0]})


class ClassifierOrientation:
    """DINOv2 feature → linear head over (pose 3 × camera-relative bin 8), trained on synthetic crops."""

    def __init__(self, encoder, head_path: Path):
        import torch
        state = torch.load(head_path, map_location="cpu")
        self.head = torch.nn.Linear(state["weight"].shape[1], 24)
        self.head.load_state_dict(state)
        self.encoder, self.torch = encoder, torch

    def orient(self, crops, calibrations, xyz) -> Orientation:
        from PIL import Image
        per_camera = []
        for crop in crops:
            feature = self.encoder.encode([Image.fromarray(cv2.cvtColor(crop.image, cv2.COLOR_BGR2RGB))])
            with self.torch.no_grad():
                logits = self.head(self.torch.tensor(np.asarray(feature), dtype=self.torch.float32))
                grid = self.torch.softmax(logits, -1)[0].numpy().reshape(3, 8)
            per_camera.append(({p: float(grid[i].sum()) for i, p in enumerate(POSES)},
                               {b: float(grid[:, b].sum()) for b in range(8)},
                               camera_azimuth_deg(calibrations[crop.camera_id], xyz)))
        return fuse(per_camera)


from dataclasses import replace as _replace


def batch_classifier(encoder, head):
    """All crops of all objects in one encoder call → per object, per camera (pose probs, bin probs)."""
    import torch
    from PIL import Image

    def classify(crops_by_object):
        flat = [(oid, crop) for oid, crops in crops_by_object.items() for crop in crops]
        if not flat:
            return {}
        features = encoder.encode([Image.fromarray(cv2.cvtColor(c.image, cv2.COLOR_BGR2RGB)) for _, c in flat])
        with torch.no_grad():
            grids = torch.softmax(head(torch.tensor(np.asarray(features), dtype=torch.float32)), -1).numpy()
        out: dict = {}
        for (oid, _), probs in zip(flat, grids):
            grid = probs.reshape(3, 8)
            out.setdefault(oid, []).append(({p: float(grid[i].sum()) for i, p in enumerate(POSES)},
                                            {b: float(grid[:, b].sum()) for b in range(8)}))
        return out
    return classify


class OrientationRunner:
    def __init__(self, classify, *, symmetric, interval_s: float = 0.5, keep: float = 0.3):
        self.classify, self.symmetric = classify, set(symmetric)
        self.interval_ns, self.keep = int(interval_s * 1e9), float(keep)
        self._last_ns = None
        self._state: dict[str, tuple[float | None, str | None]] = {}

    @classmethod
    def from_files(cls, head_path, *, symmetric, interval_s: float = 0.5):
        if not Path(head_path).is_file():
            print(f"방향 분류기 파일이 없어 방향 표시를 끕니다: {head_path}")
            return None
        import torch
        from ..prop_encoder import PropEncoder
        state = torch.load(head_path, map_location="cpu")
        head = torch.nn.Linear(state["weight"].shape[1], 24)
        head.load_state_dict(state)
        return cls(batch_classifier(PropEncoder(), head), symmetric=symmetric, interval_s=interval_s)

    def apply(self, objects, detections_by_camera, images, calibrations, timestamp_ns):
        from .objects import crops_for
        due = self._last_ns is None or timestamp_ns - self._last_ns >= self.interval_ns
        if due:
            self._last_ns = timestamp_ns
            by_key = {(d.camera_id, d.local_detection_id): d for ds in detections_by_camera.values() for d in ds}
            wanted = {o.global_id: crops_for([by_key[k] for k in o.source_detection_keys if k in by_key], images)
                      for o in objects if o.state in ("active", "uncertain") and o.label not in self.symmetric}
            answers = self.classify({k: v for k, v in wanted.items() if v})
            for o in objects:
                rows = answers.get(o.global_id)
                if not rows:
                    continue
                crops = wanted[o.global_id]
                fused = fuse([(pose, bins, camera_azimuth_deg(calibrations[c.camera_id], o.xyz))
                              for (pose, bins), c in zip(rows, crops)])
                previous = self._state.get(o.global_id, (None, None))[0]
                yaw = fused.yaw_deg if previous is None or fused.yaw_deg is None else circular_mean_deg(
                    [previous, fused.yaw_deg], [1.0 - self.keep, self.keep])
                self._state[o.global_id] = (yaw, fused.pose)
        out = []
        for o in objects:
            yaw, pose = self._state.get(o.global_id, (None, None))
            # Only a standing object has a meaningful heading in the twin (v1.2 shows yaw only).
            shown = None if o.label in self.symmetric or pose not in (None, "upright") else yaw
            out.append(_replace(o, yaw_deg=shown, pose=pose))
        return out
