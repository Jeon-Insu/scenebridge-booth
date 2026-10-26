# smartphone_mocap/recognition/evalset.py
"""합성 렌더 + specs → 장면별 이미지·보정·정답(소품과 목록 밖 물건)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..synthdata.degrade import degrade
from ..synthdata.scene_sampler import read_specs
from ..types import CameraCalibration


MIN_TRUTH_HEIGHT_M = 0.02


@dataclass(frozen=True)
class GroundTruth:
    asset_id: str | None
    xyz: tuple[float, float, float]
    pose: str | None
    yaw_deg: float | None
    masks: dict


@dataclass(frozen=True)
class EvalScene:
    scene_id: int
    images: dict
    calibrations: dict
    truth: list
    table_size: tuple


def _nearest_masks(xyz, masks_by_camera, calibrations, max_px: float = 60.0) -> dict:
    """Per camera, the instance mask whose centroid is nearest the object's projected footprint."""
    out = {}
    for cam, masks in masks_by_camera.items():
        cal = calibrations[cam]
        W = np.asarray(cal.world_to_camera)
        px = cv2.projectPoints(np.array([[xyz[0], xyz[1], 0.02]]), cv2.Rodrigues(W[:3, :3])[0], W[:3, 3],
                               cal.camera_matrix, cal.distortion)[0].ravel()
        best = None
        for mask in masks:
            ys, xs = np.nonzero(mask)
            d = float(np.hypot(xs.mean() - px[0], ys.mean() - px[1]))
            if d <= max_px and (best is None or d < best[0]):
                best = (d, mask)
        if best is not None:
            out[cam] = best[1]
    return out


def _products(scene_dir: Path) -> list[Path]:
    return sorted((p for p in scene_dir.glob("Replicator*") if p.is_dir()),
                  key=lambda p: 0 if p.name == "Replicator" else int(p.name.split("_")[-1]))


def load_scenes(specs_path, raw_dir, *, degrade_seed: int | None = 0) -> Iterator[EvalScene]:
    specs, _ = read_specs(specs_path)
    for spec in specs:
        scene_dir = Path(raw_dir) / f"scene_{spec.scene_id:05d}"
        products = _products(scene_dir)
        if len(products) != len(spec.cameras):
            continue
        rng = np.random.default_rng([degrade_seed or 0, spec.scene_id])
        images, calibrations = {}, {}
        masks_by_class: dict[str, dict[str, np.ndarray]] = {}
        unlisted_masks: dict[str, list[np.ndarray]] = {}
        for view, product in zip(spec.cameras, products):
            image = cv2.imread(str(product / "rgb/rgb_0000.png"))
            images[view.camera_id] = image if degrade_seed is None else degrade(image, rng)[0]
            K = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]], float)
            calibrations[view.camera_id] = CameraCalibration(view.camera_id, (view.width, view.height), K,
                                                             np.zeros(5), np.asarray(view.world_to_camera, float))
            seg = product / "instance_segmentation"
            ids = cv2.imread(str(seg / "instance_segmentation_0000.png"), cv2.IMREAD_UNCHANGED)
            mapping = json.loads((seg / "instance_segmentation_semantics_mapping_0000.json").read_text())
            for raw_id, info in mapping.items():
                label = str(info.get("class", ""))
                mask = ids == int(raw_id)
                if label in ("BACKGROUND", "UNLABELLED") or not mask.any():
                    continue
                if label == "unlisted":
                    # Every unlisted object shares this label: keep each instance, assign below.
                    unlisted_masks.setdefault(view.camera_id, []).append(mask)
                else:
                    masks_by_class.setdefault(label, {})[view.camera_id] = mask
        # The renderer records where each object really rests (world bounding-box centre): a lying
        # prop's footprint moves off its spec x,y by about half its height.
        placed_path = scene_dir / "placed.json"
        placed = json.loads(placed_path.read_text()) if placed_path.is_file() else {}
        prop_centres = [tuple(row["centre"][:2]) for row in placed.get("props", [])]
        other_centres = [tuple(row["centre"][:2]) for row in placed.get("distractors", [])]
        # A flat unlisted object (a 7 mm sheet) is invisible from the booth cameras: not truth.
        flat = {i for i, row in enumerate(placed.get("distractors", []))
                if row.get("centre") and len(row["centre"]) > 3 and row["centre"][3] < MIN_TRUTH_HEIGHT_M}

        def where(centres, index, x, y):
            return (*centres[index], 0.0) if index < len(centres) else (x, y, 0.0)

        truth = [GroundTruth(p.asset_id, where(prop_centres, i, p.x_m, p.y_m), p.pose, float(p.yaw_deg),
                             masks_by_class.get(p.train_class.replace(" ", "_"), {}))
                 for i, p in enumerate(spec.props)]
        for i, d in enumerate(spec.distractors):
            if d.kind == "unlisted" and i not in flat:
                xyz = where(other_centres, i, d.x_m, d.y_m)
                truth.append(GroundTruth(None, xyz, None, None,
                                         _nearest_masks(xyz, unlisted_masks, calibrations)))
        yield EvalScene(spec.scene_id, images, calibrations, truth, tuple(spec.table_size_m))
