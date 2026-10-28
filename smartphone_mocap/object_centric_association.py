"""Experimental object-first cross-camera association.

Existing camera-local ownership seeds groups. Only residual observations and
group boundaries are compared; no Cartesian products of 2/3/4-camera sets.
This bounded greedy graph construction is NOT a globally optimal solver.
All caches are scoped to one synchronized input set.
"""
from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from .geometry import Observation, reprojection_errors
from .object_reconstruction import (
    _appearance_dispersion, _mean_appearance, _ray_plane_intersection, _source_track_map,
)
from .types import CameraCalibration, Keypoint2D, Object2D, SceneObject3D


def reconstruct_object_centric(
    detections_by_camera: dict[str, list[Object2D]],
    calibrations: dict[str, CameraCalibration],
    *,
    asset_map: dict[str, str] | None = None,
    fallback_asset_uri: str | None = None,
    anchor: str = "bottom_center",
    min_confidence: float = 0.25,
    max_reprojection_error_px: float = 50.0,
    max_instances_per_class: int = 10,
    workspace_min_xyz=None,
    workspace_max_xyz=None,
    position_mode: str = "triangulate",
    support_plane_z_m: float = 0.0,
    class_height_m: dict[str, float] | None = None,
    default_object_height_m: float = 0.10,
    snap_to_support_plane_m: float = 0.08,
    cross_view_appearance_weight_px: float = 12.0,
    max_cross_view_appearance_distance: float = 0.80,
    previous_track_groups: dict[str, list[dict[str, int]]] | None = None,
    diagnostics: dict | None = None,
    appearance_policy: str = "strict",
    single_view_min_confidence: float | None = None,
    single_view_merge_m: float = 0.15,
) -> list[SceneObject3D]:
    """single_view_min_confidence: when set, a new detection seen by one camera only (not part of any
    multi-view group, not owned by a known object) becomes an object on the table plane if it is at least
    this confident and no object of the same label lies within single_view_merge_m."""
    if appearance_policy not in {"strict", "continuity"}:
        raise ValueError(f"Unsupported appearance policy: {appearance_policy}")
    if position_mode not in {"triangulate", "hybrid_mask"}:
        raise ValueError(f"Unsupported object position mode: {position_mode}")
    stats = diagnostics if diagnostics is not None else {}
    stats.clear()
    stats.update(method="object_centric", appearance_policy=appearance_policy,
                 pair_checks=0, group_fits=0,
                 seeded_groups=0, seed_conflicts=0, accepted_groups=0,
                 pending_detections=0, ownership_conflicts=0)
    previous_track_groups = previous_track_groups or {}
    output = []
    labels = sorted({d.label for ds in detections_by_camera.values() for d in ds
                     if d.confidence >= min_confidence})
    for label in labels:
        items = []
        for camera_id in sorted(detections_by_camera):
            if camera_id not in calibrations:
                continue
            selected = sorted((d for d in detections_by_camera[camera_id]
                               if d.label == label and d.confidence >= min_confidence),
                              key=lambda d: (-d.confidence, d.local_detection_id))
            items.extend(selected[:max_instances_per_class])
        if not items:
            continue
        # Normalize each measurement once, irrespective of how many edges use it.
        points, observations = [], []
        for item in items:
            camera = calibrations[item.camera_id]
            x, y = item.anchor_xy(anchor)
            xy = cv2.undistortPoints(np.array([[[x, y]]], dtype=np.float64),
                                    camera.camera_matrix, camera.distortion).reshape(2)
            points.append(xy)
            observations.append(Observation(camera, Keypoint2D(x, y, item.confidence)))
        fit_cache = {}

        def fit(indices):
            key = tuple(sorted(indices))
            if key in fit_cache:
                return fit_cache[key]
            stats["group_fits"] += 1
            rows = []
            for i in key:
                x, y = points[i]
                p = calibrations[items[i].camera_id].normalized_projection
                weight = max(items[i].confidence, 1e-3) ** .5
                rows.extend((weight * (x * p[2] - p[0]), weight * (y * p[2] - p[1])))
            try:
                _, _, vt = np.linalg.svd(np.stack(rows))
                homogeneous = vt[-1]
                if abs(homogeneous[3]) < 1e-10:
                    raise ValueError("Point at infinity")
                xyz = homogeneous[:3] / homogeneous[3]
                if not np.all(np.isfinite(xyz)):
                    raise ValueError("Nonfinite point")
                for i in key:
                    camera = calibrations[items[i].camera_id]
                    if (camera.rotation @ xyz + camera.translation)[2] <= 0:
                        raise ValueError("Point behind camera")
                errors = reprojection_errors([observations[i] for i in key], xyz)
                if not np.all(np.isfinite(errors)) or np.max(errors) > max_reprojection_error_px:
                    raise ValueError("Inconsistent observations")
                value = (xyz, float(np.mean(errors)))
            except (ValueError, np.linalg.LinAlgError):
                value = None
            fit_cache[key] = value
            return value

        pair_cache = {}

        def appearance_limit(indices):
            # Colour varies by view. Relax only at bootstrap or for unchanged
            # local identities already belonging to the same global object.
            # Attaching a NEW local track to an existing object stays strict.
            if appearance_policy == "continuity":
                if not previous_track_groups.get(label):
                    return max(1.0, max_cross_view_appearance_distance)
                memberships_for_group = [memberships[i] for i in indices]
                if (all(len(value) == 1 for value in memberships_for_group)
                        and len({value[0] for value in memberships_for_group}) == 1):
                    return max(1.0, max_cross_view_appearance_distance)
            return max_cross_view_appearance_distance

        def pair_score(i, j):
            key = tuple(sorted((i, j)))
            if key not in pair_cache:
                stats["pair_checks"] += 1
                first, second = items[i], items[j]
                distance = _appearance_dispersion((first, second))
                value = None
                if (first.camera_id != second.camera_id
                        and distance <= appearance_limit(key)):
                    fitted = fit(key)
                    if fitted is not None:
                        value = fitted[1] + cross_view_appearance_weight_px * distance
                pair_cache[key] = value
            return pair_cache[key]

        def consistent(indices):
            cameras = [items[i].camera_id for i in indices]
            if len(set(cameras)) != len(cameras):
                return False
            # A full group's fit is better constrained than each narrow-baseline
            # pair. Do not require every pair to triangulate independently.
            return (all(_appearance_dispersion((items[i], items[j])) <= appearance_limit(indices)
                        for i, j in combinations(sorted(indices), 2))
                    and fit(indices) is not None)

        # Determine ownership without giving a reused historical local ID to
        # multiple objects. Ambiguous seeds are left pending for this frame.
        memberships = {i: [] for i in range(len(items))}
        for owner, old_group in enumerate(previous_track_groups.get(label, [])):
            for i, item in enumerate(items):
                if item.local_track_id is not None and old_group.get(item.camera_id) == item.local_track_id:
                    memberships[i].append(owner)
        blocked = {i for i, owners in memberships.items() if len(owners) > 1}
        stats["ownership_conflicts"] += len(blocked)
        groups = {i: {i} for i in range(len(items)) if i not in blocked}
        owners = {i: memberships[i][0] if memberships[i] else None for i in groups}
        roots = {i: i for i in groups}

        def merge(left, right):
            groups[left].update(groups.pop(right))
            if owners[left] is None:
                owners[left] = owners[right]
            owners.pop(right)
            for i in groups[left]:
                roots[i] = left

        # Fast path: validate each existing object's currently visible views.
        for owner in sorted({v for v in owners.values() if v is not None}):
            indices = [i for i in groups if owners[i] == owner]
            if len(indices) >= 2:
                if consistent(set(indices)):
                    for i in indices[1:]:
                        merge(indices[0], i)
                    stats["seeded_groups"] += 1
                else:
                    # Repair only this conflicting group's local edges below.
                    stats["seed_conflicts"] += 1

        edges = []
        for i, j in combinations(sorted(roots), 2):
            left, right = roots[i], roots[j]
            if left == right:
                continue
            a, b = owners[left], owners[right]
            if a is not None and b is not None and a != b:
                continue  # never merge two established identities implicitly
            score = pair_score(i, j)
            if score is not None:
                # Existing membership repair, then residual-to-known, then birth.
                priority = 0 if a is not None and a == b else 1 if a is not None or b is not None else 2
                edges.append((priority, score, i, j))
        for priority, _, i, j in sorted(edges):
            if priority == 2:
                continue  # new-object discovery below grows one group at a time
            left, right = roots[i], roots[j]
            if left == right:
                continue
            a, b = owners[left], owners[right]
            if a is not None and b is not None and a != b:
                continue
            union = groups[left] | groups[right]
            # Full group consistency prevents a simple connected-component
            # chain from merging A~B and B~C when A and C contradict each other.
            if consistent(union):
                merge(left, right)

        # Cold start/residual birth: grow a seed pair with at most one best
        # observation per remaining camera. Compare these local seed proposals,
        # not Cartesian products over all camera subsets. Finish one object
        # before creating another pair, avoiding premature 2+2 fragmentation.
        available = {i for i in roots if owners[roots[i]] is None}
        while len(available) >= 2:
            proposals = []
            for _, score, i, j in edges:
                if i not in available or j not in available:
                    continue
                proposal = {i, j}
                for camera_id in sorted({items[k].camera_id for k in available}
                                        - {items[i].camera_id, items[j].camera_id}):
                    choices = []
                    for k in sorted(available - proposal):
                        if items[k].camera_id != camera_id:
                            continue
                        trial = proposal | {k}
                        if consistent(trial):
                            error = fit(trial)[1]
                            appearance = _appearance_dispersion(tuple(items[t] for t in sorted(trial)))
                            choices.append((error + cross_view_appearance_weight_px * appearance, k))
                    if choices:
                        proposal.add(min(choices)[1])
                if consistent(proposal):
                    error = fit(proposal)[1]
                    appearance = _appearance_dispersion(tuple(items[t] for t in sorted(proposal)))
                    proposals.append((-len(proposal), error + cross_view_appearance_weight_px * appearance,
                                      tuple(sorted(proposal))))
            if not proposals:
                break
            chosen = min(proposals)[2]
            for k in chosen[1:]:
                merge(roots[chosen[0]], roots[k])
            available.difference_update(chosen)

        accepted_keys = set()
        accepted_owners = set()
        accepted_count = 0
        for root in sorted(groups, key=lambda r: (owners[r] is None, -len(groups[r]), r)):
            indices = groups[root]
            owner = owners[root]
            if len(indices) < 2 or accepted_count >= max_instances_per_class:
                continue
            if owner is not None and owner in accepted_owners:
                continue  # conflicting fragments of an old object stay pending
            fitted = fit(indices)
            if fitted is None:
                continue
            xyz, error = fitted
            xyz = xyz.copy()
            matched = tuple(items[i] for i in sorted(indices))
            resolved = "triangulated"
            if position_mode == "hybrid_mask":
                height = float((class_height_m or {}).get(label, default_object_height_m))
                xyz[2] -= height * .5
                if xyz[2] <= support_plane_z_m + snap_to_support_plane_m:
                    plane = [_ray_plane_intersection(calibrations[d.camera_id],
                             d.anchor_xy("mask_bottom_center"), support_plane_z_m) for d in matched]
                    plane = [p for p in plane if p is not None]
                    if plane:
                        xyz[:2] = np.median(np.stack(plane), axis=0)[:2]
                    xyz[2] = support_plane_z_m
                    resolved = "support_plane"
                else:
                    resolved = "airborne_mask_centroid"
            if workspace_min_xyz is not None and np.any(xyz < np.asarray(workspace_min_xyz)):
                continue
            if workspace_max_xyz is not None and np.any(xyz > np.asarray(workspace_max_xyz)):
                continue
            confidence = float(np.mean([d.confidence for d in matched])
                               * np.exp(-error / max(max_reprojection_error_px, 1.)))
            output.append(SceneObject3D(
                global_id="", label=label, xyz=tuple(float(v) for v in xyz),
                confidence=confidence, reprojection_error_px=error,
                source_cameras=tuple(sorted(d.camera_id for d in matched)),
                asset_uri=(asset_map or {}).get(label, fallback_asset_uri), state="observed",
                appearance=_mean_appearance(matched),
                representative_crops=tuple(d.representative_crop for d in matched
                                           if d.representative_crop is not None),
                position_mode=resolved,
                source_detection_keys=tuple(sorted((d.camera_id, d.local_detection_id) for d in matched)),
                source_track_keys=tuple(sorted(_source_track_map(matched).items())),
            ))
            accepted_keys.update(indices)
            if owner is not None:
                accepted_owners.add(owner)
            accepted_count += 1

        if single_view_min_confidence is not None:
            placed = [np.asarray(o.xyz, float) for o in output if o.label == label]
            lone = sorted((i for i in groups if i not in accepted_keys and len(groups[i]) == 1 and owners[i] is None
                           and items[i].confidence >= single_view_min_confidence),
                          key=lambda i: (-items[i].confidence, i))
            for i in lone:
                if accepted_count >= max_instances_per_class:
                    break
                d = items[i]
                hit = _ray_plane_intersection(calibrations[d.camera_id], d.anchor_xy("mask_bottom_center"),
                                              support_plane_z_m)
                if hit is None:
                    continue
                xyz = np.asarray(hit, float)
                xyz[2] = support_plane_z_m
                if workspace_min_xyz is not None and np.any(xyz < np.asarray(workspace_min_xyz)):
                    continue
                if workspace_max_xyz is not None and np.any(xyz > np.asarray(workspace_max_xyz)):
                    continue
                if any(np.linalg.norm(xyz[:2] - q[:2]) < single_view_merge_m for q in placed):
                    continue   # another camera's view of an object already placed
                output.append(SceneObject3D(
                    global_id="", label=label, xyz=tuple(float(v) for v in xyz),
                    confidence=float(d.confidence), reprojection_error_px=0.0,
                    source_cameras=(d.camera_id,),
                    asset_uri=(asset_map or {}).get(label, fallback_asset_uri), state="observed",
                    appearance=_mean_appearance((d,)),
                    representative_crops=tuple(c for c in (d.representative_crop,) if c is not None),
                    position_mode="single_view",
                    source_detection_keys=((d.camera_id, d.local_detection_id),),
                    source_track_keys=tuple(sorted(_source_track_map((d,)).items())),
                ))
                placed.append(xyz)
                accepted_keys.add(i)
                accepted_count += 1
        stats["accepted_groups"] += accepted_count
        stats["pending_detections"] += len(items) - len(accepted_keys)
    return output
