from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product

import numpy as np

from .geometry import Observation, triangulate_joint
from .types import CameraCalibration, Keypoint2D, Object2D, SceneObject3D


@dataclass(slots=True, frozen=True)
class _ObjectCandidate:
    object: SceneObject3D
    detection_keys: frozenset[tuple[str, int]]
    appearance_dispersion: float = 0.0
    track_continuity_score: float = 0.0


@dataclass(slots=True)
class _TrackState:
    object: SceneObject3D
    hits: int
    missed: int
    velocity: np.ndarray
    last_seen_ns: int | None
    local_tracks: dict[str, int]


def _mean_appearance(items: tuple[Object2D, ...]) -> tuple[float, ...] | None:
    descriptors = [
        np.asarray(item.appearance, dtype=np.float64) for item in items if item.appearance
    ]
    if not descriptors:
        return None
    mean = np.mean(np.stack(descriptors), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-8:
        mean /= norm
    return tuple(float(value) for value in mean)


def _appearance_dispersion(items: tuple[Object2D, ...]) -> float:
    descriptors = [
        np.asarray(item.appearance, dtype=np.float64) for item in items if item.appearance
    ]
    if len(descriptors) < 2:
        return 0.0
    distances = [
        1.0 - float(np.clip(np.dot(first, second), -1.0, 1.0))
        for first, second in combinations(descriptors, 2)
        if len(first) == len(second)
    ]
    return float(np.mean(distances)) if distances else 0.0


def _source_track_map(items: tuple[Object2D, ...]) -> dict[str, int]:
    return {
        item.camera_id: int(item.local_track_id)
        for item in items
        if item.local_track_id is not None
    }


def _track_group_continuity(
    items: tuple[Object2D, ...],
    previous_groups: list[dict[str, int]],
) -> float:
    """Score whether camera-local tracks have previously formed one object."""
    current = _source_track_map(items)
    if not current or not previous_groups:
        return 0.0
    best = 0.0
    for group in previous_groups:
        matches = sum(group.get(camera_id) == track_id for camera_id, track_id in current.items())
        conflicts = sum(
            camera_id in group and group[camera_id] != track_id
            for camera_id, track_id in current.items()
        )
        best = max(best, float(matches) - 0.5 * float(conflicts))
    return best


def _observation_instance_upper_bound(by_camera: dict[str, list[Object2D]]) -> int:
    """Combinatorial bound, NOT an estimate of the physical object count.

    Each candidate needs two distinct cameras and cannot reuse a detection.
    Half the total and the detections outside the largest camera both bound
    the number of disjoint pairs. A median incorrectly rejects two objects
    seen by disjoint camera pairs. Actual acceptance still requires geometry,
    appearance, and exclusive observation assignment.
    """
    counts = [len(items) for items in by_camera.values() if items]
    if len(counts) < 2:
        return 0
    total = sum(counts)
    return min(total // 2, total - max(counts))


def _ray_plane_intersection(
    camera: CameraCalibration, pixel_xy: tuple[float, float], plane_z: float
) -> np.ndarray | None:
    """Intersect an undistorted camera ray with a horizontal world plane."""
    import cv2

    point = np.array([[[pixel_xy[0], pixel_xy[1]]]], dtype=np.float64)
    normalized = cv2.undistortPoints(point, camera.camera_matrix, camera.distortion).reshape(2)
    direction_world = camera.rotation.T @ np.array([normalized[0], normalized[1], 1.0])
    camera_center = -camera.rotation.T @ camera.translation
    if abs(float(direction_world[2])) < 1e-9:
        return None
    distance = (float(plane_z) - float(camera_center[2])) / float(direction_world[2])
    if distance <= 0.0:
        return None
    return camera_center + distance * direction_world


def reconstruct_static_objects(
    detections_by_camera: dict[str, list[Object2D]],
    calibrations: dict[str, CameraCalibration],
    *,
    asset_map: dict[str, str] | None = None,
    fallback_asset_uri: str | None = None,
    anchor: str = "bottom_center",
    min_confidence: float = 0.25,
    max_reprojection_error_px: float = 50.0,
    max_instances_per_class: int = 10,
    workspace_min_xyz: tuple[float, float, float] | list[float] | None = None,
    workspace_max_xyz: tuple[float, float, float] | list[float] | None = None,
    position_mode: str = "triangulate",
    support_plane_z_m: float = 0.0,
    class_height_m: dict[str, float] | None = None,
    default_object_height_m: float = 0.10,
    snap_to_support_plane_m: float = 0.08,
    cross_view_appearance_weight_px: float = 12.0,
    max_cross_view_appearance_distance: float = 0.80,
    previous_track_groups: dict[str, list[dict[str, int]]] | None = None,
) -> list[SceneObject3D]:
    """Triangulate multiple same-class objects without reusing a 2D detection.

    Candidate correspondences are formed across every available camera subset.
    Candidates supported by more cameras win first, followed by lower reprojection
    error. This lets two cups become two independent 3D observations instead of
    discarding every detection except the strongest cup in each image.
    """
    asset_map = asset_map or {}
    class_height_m = class_height_m or {}
    previous_track_groups = previous_track_groups or {}
    if position_mode not in {"triangulate", "hybrid_mask"}:
        raise ValueError(f"Unsupported object position mode: {position_mode}")
    workspace_min = (
        np.asarray(workspace_min_xyz, dtype=np.float64)
        if workspace_min_xyz is not None
        else None
    )
    workspace_max = (
        np.asarray(workspace_max_xyz, dtype=np.float64)
        if workspace_max_xyz is not None
        else None
    )
    labels = sorted(
        {
            detection.label
            for detections in detections_by_camera.values()
            for detection in detections
            if detection.confidence >= min_confidence
        }
    )
    reconstructed: list[SceneObject3D] = []
    # One cache per synchronized input set, never per track or across frames.
    # A detection participates in many 2/3/4-view hypotheses; its immutable
    # input mask has the same anchor in all of them. Triangulation is NOT cached.
    anchors: dict[tuple[int, str], tuple[float, float]] = {}

    def anchor_xy(detection: Object2D, kind: str) -> tuple[float, float]:
        key = (id(detection), kind)
        if key not in anchors:
            anchors[key] = detection.anchor_xy(kind)
        return anchors[key]

    for label in labels:
        by_camera = {
            camera_id: sorted(
                (
                    item
                    for item in detections
                    if item.label == label and item.confidence >= min_confidence
                ),
                key=lambda item: item.confidence,
                reverse=True,
            )[:max_instances_per_class]
            for camera_id, detections in detections_by_camera.items()
            if camera_id in calibrations
        }
        by_camera = {camera_id: items for camera_id, items in by_camera.items() if items}
        camera_ids = sorted(by_camera)
        if len(camera_ids) < 2:
            continue
        instance_cap = min(
            max_instances_per_class,
            _observation_instance_upper_bound(by_camera),
        )
        if instance_cap <= 0:
            continue

        candidates_by_keys: dict[frozenset[tuple[str, int]], _ObjectCandidate] = {}
        for view_count in range(len(camera_ids), 1, -1):
            for camera_subset in combinations(camera_ids, view_count):
                for matched_detections in product(*(by_camera[name] for name in camera_subset)):
                    observations = []
                    for detection in matched_detections:
                        x, y = anchor_xy(detection, anchor)
                        observations.append(
                            Observation(
                                calibrations[detection.camera_id],
                                Keypoint2D(x=x, y=y, confidence=detection.confidence),
                            )
                        )
                    point = triangulate_joint(
                        observations,
                        min_confidence=min_confidence,
                        max_reprojection_error_px=max_reprojection_error_px,
                    )
                    # If robust triangulation rejected one member of the proposed
                    # correspondence, let its smaller camera subset compete instead.
                    if point is None or len(point.source_cameras) != view_count:
                        continue
                    xyz = np.array([point.x, point.y, point.z], dtype=np.float64)
                    resolved_position_mode = "triangulated"
                    if position_mode == "hybrid_mask":
                        estimated_height = float(
                            class_height_m.get(label, default_object_height_m)
                        )
                        xyz[2] -= estimated_height * 0.5
                        if xyz[2] <= float(support_plane_z_m) + float(snap_to_support_plane_m):
                            plane_points = [
                                value
                                for detection in matched_detections
                                if (
                                    value := _ray_plane_intersection(
                                        calibrations[detection.camera_id],
                                        anchor_xy(detection, "mask_bottom_center"),
                                        float(support_plane_z_m),
                                    )
                                )
                                is not None
                            ]
                            if plane_points:
                                plane_xy = np.median(np.stack(plane_points), axis=0)[:2]
                                xyz[:2] = plane_xy
                            xyz[2] = float(support_plane_z_m)
                            resolved_position_mode = "support_plane"
                        else:
                            resolved_position_mode = "airborne_mask_centroid"
                    if workspace_min is not None and np.any(xyz < workspace_min):
                        continue
                    if workspace_max is not None and np.any(xyz > workspace_max):
                        continue
                    keys = frozenset(
                        (item.camera_id, item.local_detection_id)
                        for item in matched_detections
                    )
                    source_track_keys = tuple(
                        sorted(_source_track_map(matched_detections).items())
                    )
                    appearance_dispersion = _appearance_dispersion(matched_detections)
                    if appearance_dispersion > max_cross_view_appearance_distance:
                        continue
                    scene_object = SceneObject3D(
                        global_id="",
                        label=label,
                        xyz=tuple(float(value) for value in xyz),
                        confidence=point.confidence,
                        reprojection_error_px=point.reprojection_error_px,
                        source_cameras=point.source_cameras,
                        asset_uri=asset_map.get(label, fallback_asset_uri),
                        state="observed",
                        appearance=_mean_appearance(matched_detections),
                        representative_crops=tuple(
                            item.representative_crop
                            for item in matched_detections
                            if item.representative_crop is not None
                        ),
                        position_mode=resolved_position_mode,
                        source_detection_keys=tuple(sorted(keys)),
                        source_track_keys=source_track_keys,
                    )
                    candidate = _ObjectCandidate(
                        scene_object,
                        keys,
                        appearance_dispersion,
                        _track_group_continuity(
                            matched_detections,
                            previous_track_groups.get(label, []),
                        ),
                    )
                    previous = candidates_by_keys.get(keys)
                    if previous is None or (
                        scene_object.reprojection_error_px
                        + cross_view_appearance_weight_px * appearance_dispersion
                        < previous.object.reprojection_error_px
                        + cross_view_appearance_weight_px * previous.appearance_dispersion
                    ):
                        candidates_by_keys[keys] = candidate

        # A detection belongs to only one physical object. Prefer the solution
        # supported by the most views, then the geometrically cleanest solution.
        candidates = sorted(
            candidates_by_keys.values(),
            key=lambda item: (
                -len(item.detection_keys),
                -item.track_continuity_score,
                item.object.reprojection_error_px
                + cross_view_appearance_weight_px * item.appearance_dispersion,
                -item.object.confidence,
            ),
        )
        used: set[tuple[str, int]] = set()
        accepted = 0
        for candidate in candidates:
            if candidate.detection_keys & used:
                continue
            reconstructed.append(candidate.object)
            used.update(candidate.detection_keys)
            accepted += 1
            if accepted >= instance_cap:
                break
    return reconstructed


class StaticObjectTracker:
    """Assign stable per-class IDs and smooth object positions across frames."""

    def __init__(
        self,
        *,
        ema_alpha: float = 0.15,
        ema_alpha_z: float | None = None,
        min_confirmations: int = 3,
        hold_frames: int | None = 30,
        hold_seconds: float | None = None,
        reid_seconds: float | None = 300.0,
        max_association_distance_m: float = 0.25,
        max_association_z_m: float | None = None,
        appearance_weight: float = 0.35,
        max_appearance_distance: float = 0.65,
        local_track_match_bonus_m: float = 0.12,
        local_track_conflict_penalty_m: float = 0.20,
        asset_variant_max_distance: float = 0.20,
        initial_variant_descriptors: dict[
            str, list[tuple[float, ...] | None]
        ] | None = None,
        merge_split_distance_m: float = 0.03,
        hide_leftover_distance_m: float = 0.15,
    ) -> None:
        self.ema_alpha = float(ema_alpha)
        self.ema_alpha_z = float(ema_alpha if ema_alpha_z is None else ema_alpha_z)
        self.min_confirmations = int(min_confirmations)
        # None: objects expire by hold_seconds only (v1.2 booth: one time-based rule).
        self.hold_frames = None if hold_frames is None else int(hold_frames)
        self.hold_seconds = None if hold_seconds is None else float(hold_seconds)
        self.reid_seconds = None if reid_seconds is None else float(reid_seconds)
        self.max_association_distance_m = float(max_association_distance_m)
        self.max_association_z_m = (
            None if max_association_z_m is None else float(max_association_z_m)
        )
        self.appearance_weight = float(appearance_weight)
        self.max_appearance_distance = float(max_appearance_distance)
        self.local_track_match_bonus_m = float(local_track_match_bonus_m)
        self.local_track_conflict_penalty_m = float(local_track_conflict_penalty_m)
        self.asset_variant_max_distance = float(asset_variant_max_distance)
        # 2026-10-25 live: one object grouped as two camera pairs at the same spot, and a
        # one-camera leftover of an old track shown beside the real object.
        self.merge_split_distance_m = float(merge_split_distance_m)
        self.hide_leftover_distance_m = float(hide_leftover_distance_m)
        # Labels observed only every few seconds (the slow prompt path): held by
        # hold_seconds alone, since hold_frames would depend on the frame rate.
        self.time_held_labels: frozenset[str] = frozenset()
        self._tracks: dict[str, _TrackState] = {}
        self._dormant_tracks: dict[str, _TrackState] = {}
        self._next_id_by_asset_key: dict[str, int] = {}
        self._variant_descriptors_by_label: dict[str, list[np.ndarray | None]] = {
            label: [
                None if value is None else np.asarray(value, dtype=np.float64)
                for value in values
            ]
            for label, values in (initial_variant_descriptors or {}).items()
        }

    def local_track_groups(self) -> dict[str, list[dict[str, int]]]:
        """Return camera-local track groups remembered for each global object."""
        groups: dict[str, list[dict[str, int]]] = {}
        for track in (*self._tracks.values(), *self._dormant_tracks.values()):
            if track.local_tracks:
                groups.setdefault(track.object.label, []).append(dict(track.local_tracks))
        return groups

    def _asset_key(self, label: str, appearance: tuple[float, ...] | None) -> str:
        namespace = label.replace(" ", "_")
        variants = self._variant_descriptors_by_label.setdefault(label, [])
        descriptor = None if not appearance else np.asarray(appearance, dtype=np.float64)
        if descriptor is not None:
            for index, prototype in enumerate(variants):
                if prototype is None or len(prototype) != len(descriptor):
                    continue
                distance = float(1.0 - np.clip(np.dot(prototype, descriptor), -1.0, 1.0))
                if distance <= self.asset_variant_max_distance:
                    combined = 0.9 * prototype + 0.1 * descriptor
                    norm = float(np.linalg.norm(combined))
                    variants[index] = combined / norm if norm > 1e-8 else combined
                    return f"{namespace}/appearance_{index:03d}"
        if descriptor is None and variants:
            return f"{namespace}/appearance_000"
        variants.append(descriptor)
        return f"{namespace}/appearance_{len(variants) - 1:03d}"

    def _new_id(self, asset_key: str) -> str:
        number = self._next_id_by_asset_key.get(asset_key, 0)
        while (
            f"{asset_key}/instance_{number:03d}" in self._tracks
            or f"{asset_key}/instance_{number:03d}" in self._dormant_tracks
        ):
            number += 1
        self._next_id_by_asset_key[asset_key] = number + 1
        return f"{asset_key}/instance_{number:03d}"

    @staticmethod
    def _appearance_distance(
        first: tuple[float, ...] | None, second: tuple[float, ...] | None
    ) -> float | None:
        if not first or not second or len(first) != len(second):
            return None
        first_array = np.asarray(first, dtype=np.float64)
        second_array = np.asarray(second, dtype=np.float64)
        return float(1.0 - np.clip(np.dot(first_array, second_array), -1.0, 1.0))

    def update(
        self,
        observations: list[SceneObject3D],
        *,
        timestamp_ns: int | None = None,
        detections_by_camera: dict[str, list[Object2D]] | None = None,
    ) -> list[SceneObject3D]:
        """Globally associate 3D observations and maintain visibility states.

        Camera-local track continuity can override a noisy 3D distance gate
        when at least two views agree.  Remaining track/observation choices are
        solved together with a Hungarian assignment instead of greedily.
        """
        # Reject invalid upstream associations before changing identity state.
        # Detection indices are camera/frame-local; tracker IDs are additionally
        # label-local because each camera/class owns a separate 2D tracker.
        claimed_detections: set[tuple[str, int]] = set()
        claimed_tracks: set[tuple[str, str, int]] = set()
        for item in observations:
            for key in item.source_detection_keys:
                if key in claimed_detections:
                    raise ValueError(f"Detection {key} already assigned to a 3D observation")
                claimed_detections.add(key)
            for camera_id, track_id in item.source_track_keys:
                key = (camera_id, item.label, track_id)
                if key in claimed_tracks:
                    raise ValueError(f"Local track {key} already assigned to a 3D observation")
                claimed_tracks.add(key)
        observations = _merge_split_observations(observations, self.merge_split_distance_m)
        if self.reid_seconds is not None and timestamp_ns is not None:
            reid_window_ns = int(self.reid_seconds * 1_000_000_000)
            self._dormant_tracks = {
                object_id: track
                for object_id, track in self._dormant_tracks.items()
                if track.last_seen_ns is None
                or timestamp_ns - track.last_seen_ns < reid_window_ns
            }
        updated: dict[str, _TrackState] = {}
        unmatched_tracks = set(self._tracks)
        unmatched_observations = set(range(len(observations)))
        association_tracks = {**self._dormant_tracks, **self._tracks}

        possible_matches: dict[tuple[str, int], float] = {}
        for object_id, track in association_tracks.items():
            old = track.object
            missed = track.missed
            velocity = track.velocity
            old_xyz = np.asarray(old.xyz)
            is_dormant = object_id in self._dormant_tracks
            predicted_xyz = old_xyz if is_dormant else old_xyz + velocity * min(missed + 1, 5)
            for index, item in enumerate(observations):
                if item.label != old.label:
                    continue
                delta = np.asarray(item.xyz) - predicted_xyz
                gate_scale = 2.0 if is_dormant else 1.0 + 0.15 * min(missed, 5)
                if self.max_association_z_m is None:
                    distance = float(np.linalg.norm(delta))
                    compatible = distance <= self.max_association_distance_m * gate_scale
                else:
                    horizontal_distance = float(np.linalg.norm(delta[:2]))
                    vertical_distance = abs(float(delta[2]))
                    compatible = (
                        horizontal_distance <= self.max_association_distance_m * gate_scale
                        and vertical_distance <= self.max_association_z_m * gate_scale
                    )
                    # Prefer the same XY footprint when an object is lifted.
                    distance = horizontal_distance + 0.25 * vertical_distance
                appearance_distance = self._appearance_distance(old.appearance, item.appearance)
                if appearance_distance is not None:
                    compatible = compatible and appearance_distance <= self.max_appearance_distance
                    distance += self.appearance_weight * appearance_distance
                item_local_tracks = dict(item.source_track_keys)
                shared_local_tracks = sum(
                    track.local_tracks.get(camera_id) == track_id
                    for camera_id, track_id in item_local_tracks.items()
                )
                conflicting_local_tracks = sum(
                    camera_id in track.local_tracks
                    and track.local_tracks[camera_id] != track_id
                    for camera_id, track_id in item_local_tracks.items()
                )
                # Two camera-local tracks continuing together are stronger
                # identity evidence than a noisy rough-calibration 3D jump.
                if shared_local_tracks >= 2:
                    compatible = True
                distance -= self.local_track_match_bonus_m * shared_local_tracks
                distance += (
                    self.local_track_conflict_penalty_m * conflicting_local_tracks
                )
                if compatible:
                    possible_matches[(object_id, index)] = float(distance)

        matched_pairs: list[tuple[str, int]] = []
        track_ids = sorted(association_tracks)
        if track_ids and observations and possible_matches:
            from scipy.optimize import linear_sum_assignment

            track_count = len(track_ids)
            observation_count = len(observations)
            size = track_count + observation_count
            invalid_cost = 1_000_000.0
            unmatched_cost = (
                1.0
                + 2.0 * self.max_association_distance_m
                + self.appearance_weight * self.max_appearance_distance
                + self.local_track_conflict_penalty_m * 4.0
            )
            costs = np.full((size, size), invalid_cost, dtype=np.float64)
            for row, object_id in enumerate(track_ids):
                for index in range(observation_count):
                    value = possible_matches.get((object_id, index))
                    if value is not None:
                        costs[row, index] = value
                costs[row, observation_count + row] = unmatched_cost
            for index in range(observation_count):
                costs[track_count + index, index] = unmatched_cost
            costs[track_count:, observation_count:] = 0.0
            rows, columns = linear_sum_assignment(costs)
            matched_pairs = [
                (track_ids[row], int(column))
                for row, column in zip(rows, columns, strict=True)
                if row < track_count
                and column < observation_count
                and costs[row, column] < invalid_cost
            ]

        for object_id, index in matched_pairs:
            item = observations[index]
            old_track = association_tracks[object_id]
            old = old_track.object
            old_hits = old_track.hits
            old_velocity = old_track.velocity
            alpha = np.array(
                [self.ema_alpha, self.ema_alpha, self.ema_alpha_z], dtype=np.float64
            )
            xyz = (1.0 - alpha) * np.asarray(old.xyz) + alpha * np.asarray(item.xyz)
            # A support-plane observation is a hard geometric constraint.  Do
            # not leave an exponentially decaying airborne Z residue after an
            # object has been put back on the table.
            if item.position_mode == "support_plane":
                xyz[2] = float(item.xyz[2])
            hits = old_hits + 1
            state = "active" if hits >= self.min_confirmations else "candidate"
            measured_velocity = np.asarray(item.xyz, dtype=np.float64) - np.asarray(
                old.xyz, dtype=np.float64
            )
            velocity = 0.65 * old_velocity + 0.35 * measured_velocity
            appearance = item.appearance
            if old.appearance and item.appearance and len(old.appearance) == len(item.appearance):
                appearance_array = 0.8 * np.asarray(old.appearance) + 0.2 * np.asarray(
                    item.appearance
                )
                norm = float(np.linalg.norm(appearance_array))
                if norm > 1e-8:
                    appearance_array /= norm
                appearance = tuple(float(value) for value in appearance_array)
            tracked = replace(
                item,
                global_id=object_id,
                asset_key=old.asset_key,
                xyz=tuple(float(value) for value in xyz),
                state=state,
                appearance=appearance,
            )
            local_tracks = dict(old_track.local_tracks)
            local_tracks.update(dict(item.source_track_keys))
            updated[object_id] = _TrackState(
                tracked,
                hits,
                0,
                velocity,
                timestamp_ns if timestamp_ns is not None else old_track.last_seen_ns,
                local_tracks,
            )
            unmatched_tracks.discard(object_id)
            unmatched_observations.remove(index)
            self._dormant_tracks.pop(object_id, None)

        # Deterministic initial numbering makes the left/lower candidate cup_0.
        for index in sorted(
            unmatched_observations,
            key=lambda value: (observations[value].label, *observations[value].xyz),
        ):
            item = observations[index]
            asset_key = self._asset_key(item.label, item.appearance)
            object_id = self._new_id(asset_key)
            state = "active" if self.min_confirmations <= 1 else "candidate"
            updated[object_id] = _TrackState(
                replace(item, global_id=object_id, asset_key=asset_key, state=state),
                1,
                0,
                np.zeros(3, dtype=np.float64),
                timestamp_ns,
                dict(item.source_track_keys),
            )

        # A successful current assignment transfers ownership of its local
        # tracks. Remove stale references from other active AND dormant tracks,
        # otherwise a later 2D-only frame can revive the same duplicate again.
        owners = {
            (camera_id, track.object.label, local_id): object_id
            for object_id, track in updated.items()
            for camera_id, local_id in track.object.source_track_keys
        }
        for object_id, track in (
            *updated.items(), *self._tracks.items(), *self._dormant_tracks.items()
        ):
            track.local_tracks = {
                camera_id: local_id
                for camera_id, local_id in track.local_tracks.items()
                if owners.get((camera_id, track.object.label, local_id), object_id)
                == object_id
            }

        current_local_detections: dict[tuple[str, str, int], Object2D] = {}
        for camera_id, detections in (detections_by_camera or {}).items():
            for detection in detections:
                if (
                    detection.local_track_id is not None
                    and (camera_id, detection.local_detection_id) not in claimed_detections
                    and (camera_id, detection.label, int(detection.local_track_id))
                    not in claimed_tracks
                ):
                    current_local_detections[
                        (camera_id, detection.label, int(detection.local_track_id))
                    ] = detection

        # If historical ownership is ambiguous and no current 3D assignment
        # resolves it, do not renew two objects from one piece of evidence.
        support_owners: dict[tuple[str, str, int], set[str]] = {}
        for object_id in unmatched_tracks:
            track = self._tracks[object_id]
            if track.hits >= self.min_confirmations:
                for camera_id, local_id in track.local_tracks.items():
                    support_owners.setdefault(
                        (camera_id, track.object.label, local_id), set()
                    ).add(object_id)

        for object_id in sorted(unmatched_tracks):
            old_track = self._tracks[object_id]
            supporting_detections = [
                detection
                for camera_id, local_track_id in old_track.local_tracks.items()
                if (
                    detection := current_local_detections.get(
                        (camera_id, old_track.object.label, local_track_id)
                    )
                )
                is not None
                and support_owners.get(
                    (camera_id, old_track.object.label, local_track_id)
                ) == {object_id}
            ]
            support_cameras = tuple(
                sorted({item.camera_id for item in supporting_detections})
            )
            if support_cameras and old_track.hits >= self.min_confirmations:
                state = "uncertain" if len(support_cameras) >= 2 else "partial"
                detection_keys = tuple(
                    sorted(
                        (item.camera_id, item.local_detection_id)
                        for item in supporting_detections
                    )
                )
                track_keys = tuple(
                    sorted(
                        (item.camera_id, int(item.local_track_id))
                        for item in supporting_detections
                        if item.local_track_id is not None
                    )
                )
                updated[object_id] = _TrackState(
                    replace(
                        old_track.object,
                        state=state,
                        source_cameras=support_cameras,
                        source_detection_keys=detection_keys,
                        source_track_keys=track_keys,
                        representative_crops=tuple(
                            item.representative_crop
                            for item in supporting_detections
                            if item.representative_crop is not None
                        ),
                    ),
                    old_track.hits,
                    0,
                    old_track.velocity * 0.5,
                    timestamp_ns
                    if timestamp_ns is not None
                    else old_track.last_seen_ns,
                    dict(old_track.local_tracks),
                )
                continue
            expired_by_time = (
                self.hold_seconds is not None
                and timestamp_ns is not None
                and old_track.last_seen_ns is not None
                and timestamp_ns - old_track.last_seen_ns
                >= int(self.hold_seconds * 1_000_000_000)
            )
            held_by_time_only = (
                old_track.object.label in self.time_held_labels
                and self.hold_seconds is not None
                and timestamp_ns is not None
            )
            expired_by_frames = (
                not held_by_time_only
                and self.hold_frames is not None
                and old_track.missed >= self.hold_frames
            )
            if expired_by_frames or expired_by_time:
                if self.reid_seconds is not None:
                    self._dormant_tracks[object_id] = _TrackState(
                        replace(
                            old_track.object,
                            state="dropped",
                            representative_crops=(),
                            source_cameras=(),
                            source_detection_keys=(),
                            source_track_keys=(),
                        ),
                        old_track.hits,
                        old_track.missed + 1,
                        np.zeros(3, dtype=np.float64),
                        old_track.last_seen_ns,
                        dict(old_track.local_tracks),
                    )
                continue
            updated[object_id] = _TrackState(
                replace(
                    old_track.object,
                    state="occluded",
                    representative_crops=(),
                    source_cameras=(),
                    source_detection_keys=(),
                    source_track_keys=(),
                ),
                old_track.hits,
                old_track.missed + 1,
                old_track.velocity * 0.8,
                old_track.last_seen_ns,
                dict(old_track.local_tracks),
            )

        self._hide_leftovers(updated)
        self._tracks = updated
        return [self._tracks[key].object for key in sorted(self._tracks)]

    def _hide_leftovers(self, tracks: dict[str, _TrackState]) -> None:
        """A one-camera ("partial") object next to a seen one of its class is its leftover."""
        seen = [track.object for track in tracks.values()
                if track.object.state in ("active", "uncertain", "candidate")]
        for track in tracks.values():
            item = track.object
            if item.state != "partial":
                continue
            if any(other.label == item.label
                   and float(np.linalg.norm(np.subtract(other.xyz[:2], item.xyz[:2])))
                   <= self.hide_leftover_distance_m
                   for other in seen):
                track.object = replace(item, state="occluded")


def _merge_split_observations(
    observations: list[SceneObject3D], distance_m: float
) -> list[SceneObject3D]:
    """Join same-class observations at one spot that no camera saw twice.

    Cross-view grouping can split one object into two camera pairs (colour differs by
    viewpoint). Two real objects that close would share at least one camera, so only
    disjoint camera sets, each triangulated from two or more cameras, are merged.
    """
    merged = list(observations)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                first, second = merged[i], merged[j]
                if (first.label != second.label
                        or min(len(first.source_cameras), len(second.source_cameras)) < 2
                        or set(first.source_cameras) & set(second.source_cameras)
                        or float(np.linalg.norm(np.subtract(first.xyz[:2], second.xyz[:2])))
                        > distance_m):
                    continue
                merged[i] = _join(first, second)
                del merged[j]
                changed = True
                break
            if changed:
                break
    return merged


def _join(first: SceneObject3D, second: SceneObject3D) -> SceneObject3D:
    w1, w2 = len(first.source_cameras), len(second.source_cameras)
    xyz = tuple(float((w1 * a + w2 * b) / (w1 + w2)) for a, b in zip(first.xyz, second.xyz))
    appearance = first.appearance
    if first.appearance and second.appearance and len(first.appearance) == len(second.appearance):
        mean = w1 * np.asarray(first.appearance) + w2 * np.asarray(second.appearance)
        norm = float(np.linalg.norm(mean))
        appearance = tuple(float(v) for v in (mean / norm if norm > 1e-8 else mean))
    return replace(
        first,
        xyz=xyz,
        confidence=max(first.confidence, second.confidence),
        reprojection_error_px=(w1 * first.reprojection_error_px
                               + w2 * second.reprojection_error_px) / (w1 + w2),
        source_cameras=tuple(sorted((*first.source_cameras, *second.source_cameras))),
        appearance=appearance,
        representative_crops=(*first.representative_crops, *second.representative_crops),
        source_detection_keys=tuple(sorted((*first.source_detection_keys,
                                            *second.source_detection_keys))),
        source_track_keys=tuple(sorted((*first.source_track_keys, *second.source_track_keys))),
    )
