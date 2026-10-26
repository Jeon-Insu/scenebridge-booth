# smartphone_mocap/recognition/objects.py
"""종류 구분 없이 카메라 간 묶고, 물체별 잘라낸 사진을 만들고, 개수 전제를 건다."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..object_centric_association import reconstruct_object_centric
from ..types import Object2D, SceneObject3D
from .types import Crop, Identity, Registry

OBJECT = "object"


def associate(found, calibrations, *, workspace):
    relabelled = {cam: [replace(d, label=OBJECT) for d in dets] for cam, dets in found.items()}
    objects = reconstruct_object_centric(
        relabelled, calibrations, position_mode="hybrid_mask", min_confidence=0.0,
        class_height_m={OBJECT: 0.1}, default_object_height_m=0.1,
        workspace_min_xyz=workspace[0], workspace_max_xyz=workspace[1],
        max_instances_per_class=20, appearance_policy="continuity")
    by_key = {(d.camera_id, d.local_detection_id): d for dets in found.values() for d in dets}
    return [(obj, [by_key[k] for k in obj.source_detection_keys if k in by_key]) for obj in objects]


def crops_for(detections: list[Object2D], images, pad: int = 8) -> list[Crop]:
    crops = []
    for det in detections:
        image = images[det.camera_id]
        h, w = image.shape[:2]
        b = det.box
        x0, y0 = max(int(b.x1) - pad, 0), max(int(b.y1) - pad, 0)
        x1, y1 = min(int(b.x2) + pad, w), min(int(b.y2) + pad, h)
        piece = image[y0:y1, x0:x1].copy()
        mask = None if det.mask is None else det.mask[y0:y1, x0:x1]
        if mask is not None:
            piece[~mask] = 114
        crops.append(Crop(det.camera_id, piece, mask, det.asset_id, float(det.confidence)))
    return crops


def apply_counts(named, registry: Registry):
    order = sorted(range(len(named)), key=lambda i: (-len(named[i][0].source_cameras),
                                                     -max(named[i][1].probs.values(), default=0.0)))
    used: dict[str, int] = {}
    out = list(named)
    for i in order:
        obj, identity = named[i]
        if identity.asset_id is None:
            continue
        try:
            limit = registry.get(identity.asset_id).count
        except KeyError:
            limit = 0
        used[identity.asset_id] = used.get(identity.asset_id, 0) + 1
        if used[identity.asset_id] > limit:
            out[i] = (obj, Identity(None, identity.probs))
    return out


def keep_counted(observations, counts):
    """One booth premise: each listed prop appears at most `count` times; anything else is dropped."""
    kept = []
    for label, limit in counts.items():
        same = [o for o in observations if o.label == label]
        same.sort(key=lambda o: (-len(o.source_cameras), -o.confidence))
        kept.extend(same[:limit])
    return kept
