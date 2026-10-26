import numpy as np
import pytest

from smartphone_mocap.geometry import project_point
from smartphone_mocap.object_reconstruction import (
    StaticObjectTracker,
    _observation_instance_upper_bound,
    reconstruct_static_objects,
)
from smartphone_mocap.synthetic import make_four_camera_rig
from smartphone_mocap.types import BoundingBox2D, Object2D, SceneObject3D


def test_anchor_cache_is_per_input_set_not_per_track(monkeypatch):
    from collections import Counter
    rig = make_four_camera_rig()
    calls = Counter()
    original = Object2D.anchor_xy

    def counted(item, kind="bottom_center"):
        calls[(id(item), kind)] += 1
        return original(item, kind)

    monkeypatch.setattr(Object2D, "anchor_xy", counted)
    detections = {}
    for name, camera in rig.items():
        x, y = project_point(camera, np.array([.12, -.08, .0]))
        detections[name] = [Object2D(name, 0, "cup", 41,
            BoundingBox2D(x-5, y-10, x+5, y), .95, local_track_id=7)]
    first = reconstruct_static_objects(detections, rig)
    assert len(first) == 1
    assert len(calls) == 4 and set(calls.values()) == {1}
    reconstruct_static_objects(detections, rig)
    assert set(calls.values()) == {2}  # no stale cache even if input objects reused


def test_multiview_object_anchor_is_triangulated():
    rig = make_four_camera_rig()
    expected = np.array([0.12, -0.08, 0.0])
    detections = {}
    for camera_id, camera in rig.items():
        x, y = project_point(camera, expected)
        detections[camera_id] = [
            Object2D(
                camera_id=camera_id,
                local_detection_id=0,
                label="bottle",
                class_id=39,
                box=BoundingBox2D(x - 20, y - 100, x + 20, y),
                confidence=0.95,
            )
        ]
    objects = reconstruct_static_objects(detections, rig, asset_map={"bottle": "bottle.usda"})
    assert len(objects) == 1
    assert np.allclose(objects[0].xyz, expected, atol=1e-6)
    assert objects[0].asset_uri == "bottle.usda"


@pytest.mark.parametrize("one_camera_false_positive", [False, True])
def test_two_same_class_objects_are_reconstructed_independently(one_camera_false_positive):
    rig = make_four_camera_rig()
    expected_positions = [np.array([-0.20, -0.08, 0.0]), np.array([0.22, 0.10, 0.0])]
    detections = {}
    for camera_index, (camera_id, camera) in enumerate(rig.items()):
        camera_detections = []
        for detection_id, expected in enumerate(expected_positions):
            x, y = project_point(camera, expected)
            camera_detections.append(
                Object2D(
                    camera_id=camera_id,
                    local_detection_id=detection_id,
                    label="cup",
                    class_id=41,
                    box=BoundingBox2D(x - 18, y - 80, x + 18, y),
                    confidence=0.95 - detection_id * 0.01,
                )
            )
        # Detection order is not guaranteed to agree between camera views.
        detections[camera_id] = (
            list(reversed(camera_detections)) if camera_index % 2 else camera_detections
        )

    if one_camera_false_positive:
        # A third detection in only one camera has no unused partner once the
        # two genuine four-view correspondences have been selected.
        detections[camera_id].append(Object2D(
            camera_id, 99, "cup", 41, BoundingBox2D(5, 5, 15, 25), .8))

    objects = reconstruct_static_objects(detections, rig)
    assert len(objects) == 2
    actual = sorted((np.asarray(item.xyz) for item in objects), key=lambda xyz: xyz[0])
    expected = sorted(expected_positions, key=lambda xyz: xyz[0])
    assert np.allclose(actual, expected, atol=1e-6)


def test_instance_upper_bound_does_not_assume_shared_visibility():
    box = BoundingBox2D(0, 0, 10, 10)

    def items(camera_id: str, count: int) -> list[Object2D]:
        return [
            Object2D(camera_id, index, "cup", 47, box, 0.9)
            for index in range(count)
        ]

    assert _observation_instance_upper_bound(
        {
            "a": items("a", 2),
            "b": items("b", 2),
            "c": items("c", 2),
            "d": items("d", 3),
        }
    ) == 4  # Upper bound, not a prediction that four objects actually exist.
    # Two of four cameras may lose one cup to occlusion; the other two views
    # are still enough to triangulate it.
    assert _observation_instance_upper_bound(
        {
            "a": items("a", 1),
            "b": items("b", 1),
            "c": items("c", 2),
            "d": items("d", 2),
        }
    ) == 3
    # With only two active views, an unmatched third detection is not a 3D
    # instance and must not increase the count.
    assert _observation_instance_upper_bound(
        {"a": items("a", 2), "b": items("b", 3)}
    ) == 2
    assert _observation_instance_upper_bound({"a": items("a", 8)}) == 0
    assert _observation_instance_upper_bound({
        "a": items("a", 20), "b": items("b", 1), "c": items("c", 1)
    }) == 2


def test_triangulation_accepts_four_three_and_two_camera_visibility():
    rig = make_four_camera_rig()
    expected = np.array([0.08, 0.04, 0.0])
    camera_items = list(rig.items())

    for view_count in (4, 3, 2):
        detections = {}
        for camera_id, camera in camera_items[:view_count]:
            x, y = project_point(camera, expected)
            detections[camera_id] = [
                Object2D(
                    camera_id=camera_id,
                    local_detection_id=0,
                    label="cup",
                    class_id=41,
                    box=BoundingBox2D(x - 15, y - 70, x + 15, y),
                    confidence=0.9,
                )
            ]

        objects = reconstruct_static_objects(detections, rig)
        assert len(objects) == 1
        assert len(objects[0].source_cameras) == view_count
        assert np.allclose(objects[0].xyz, expected, atol=1e-6)


def test_same_class_objects_survive_different_occluded_camera_subsets():
    rig = make_four_camera_rig()
    camera_items = list(rig.items())
    left = np.array([-0.22, -0.04, 0.0])
    right = np.array([0.24, 0.08, 0.0])
    # The left cup is visible in cameras 0/1/2 and the right cup in 1/2/3.
    visible_indices = {"left": {0, 1, 2}, "right": {1, 2, 3}}
    detections = {}
    for camera_index, (camera_id, camera) in enumerate(camera_items):
        camera_detections = []
        for detection_id, (name, expected) in enumerate((("left", left), ("right", right))):
            if camera_index not in visible_indices[name]:
                continue
            x, y = project_point(camera, expected)
            camera_detections.append(
                Object2D(
                    camera_id=camera_id,
                    local_detection_id=detection_id,
                    label="cup",
                    class_id=41,
                    box=BoundingBox2D(x - 15, y - 70, x + 15, y),
                    confidence=0.9,
                )
            )
        detections[camera_id] = camera_detections

    objects = reconstruct_static_objects(
        detections,
        rig,
        max_reprojection_error_px=2.0,
    )
    assert len(objects) == 2
    assert sorted(len(item.source_cameras) for item in objects) == [3, 3]
    actual = sorted((np.asarray(item.xyz) for item in objects), key=lambda xyz: xyz[0])
    assert np.allclose(actual, [left, right], atol=1e-6)


def test_two_objects_visible_in_disjoint_camera_pairs_are_not_count_capped():
    rig = make_four_camera_rig()
    expected = [np.array([-.22, -.04, 0.]), np.array([.24, .08, 0.])]
    detections = {}
    for index, (name, camera) in enumerate(rig.items()):
        x, y = project_point(camera, expected[index // 2])
        detections[name] = [Object2D(
            name, 0, "cup", 41, BoundingBox2D(x-15, y-70, x+15, y), .9)]
    objects = reconstruct_static_objects(detections, rig, max_reprojection_error_px=2.)
    assert len(objects) == 2
    assert np.allclose(sorted([o.xyz for o in objects]), expected, atol=1e-6)
    assert len({key for o in objects for key in o.source_detection_keys}) == 4


def test_static_tracker_smooths_and_confirms_position():
    tracker = StaticObjectTracker(ema_alpha=0.5, min_confirmations=2, hold_frames=2)
    first = SceneObject3D("bottle_0", "bottle", (0.0, 0.0, 0.0), 0.9, 1.0, ("a", "b"))
    second = SceneObject3D("bottle_0", "bottle", (0.2, 0.0, 0.0), 0.9, 1.0, ("a", "b"))
    assert tracker.update([first])[0].state == "candidate"
    tracked = tracker.update([second])[0]
    assert tracked.state == "active"
    assert tracked.xyz == (0.1, 0.0, 0.0)
    assert tracker.update([])[0].state == "occluded"


def test_tracker_assigns_and_preserves_two_same_class_ids():
    tracker = StaticObjectTracker(
        ema_alpha=1.0,
        min_confirmations=1,
        hold_frames=2,
        max_association_distance_m=0.3,
    )
    first = [
        SceneObject3D("", "cup", (0.05, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
        SceneObject3D("", "cup", (0.35, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
    ]
    initial = tracker.update(first)
    assert [item.global_id for item in initial] == [
        "cup/appearance_000/instance_000",
        "cup/appearance_000/instance_001",
    ]

    # Reverse detector order and move both objects slightly; IDs stay spatially attached.
    second = [
        SceneObject3D("", "cup", (0.33, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
        SceneObject3D("", "cup", (0.07, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
    ]
    tracked = tracker.update(second)
    assert tracked[0].global_id == "cup/appearance_000/instance_000"
    assert tracked[0].xyz == (0.07, 0.0, 0.0)
    assert tracked[1].global_id == "cup/appearance_000/instance_001"
    assert tracked[1].xyz == (0.33, 0.0, 0.0)


def test_tracker_preserves_id_when_an_object_is_lifted_or_reappears_nearby():
    tracker = StaticObjectTracker(
        ema_alpha=1.0,
        ema_alpha_z=0.5,
        min_confirmations=1,
        hold_frames=30,
        max_association_distance_m=0.45,
        max_association_z_m=0.60,
    )
    initial = SceneObject3D("", "cup", (0.20, 0.20, 0.0), 0.9, 1.0, ("a", "b"))
    lifted = SceneObject3D("", "cup", (0.24, 0.22, 0.32), 0.9, 1.0, ("a", "c"))
    moved = SceneObject3D("", "cup", (0.52, 0.30, 0.18), 0.9, 1.0, ("b", "c"))

    assert tracker.update([initial])[0].global_id == "cup/appearance_000/instance_000"
    assert tracker.update([])[0].state == "occluded"
    tracked_lifted = tracker.update([lifted])[0]
    assert tracked_lifted.global_id == "cup/appearance_000/instance_000"
    assert np.allclose(tracked_lifted.xyz, (0.24, 0.22, 0.16))
    tracked_moved = tracker.update([moved])[0]
    assert tracked_moved.global_id == "cup/appearance_000/instance_000"
    assert np.allclose(tracked_moved.xyz, (0.52, 0.30, 0.17))


def test_tracker_snaps_z_exactly_when_object_returns_to_support_plane():
    tracker = StaticObjectTracker(
        ema_alpha=0.5,
        ema_alpha_z=0.5,
        min_confirmations=1,
        max_association_distance_m=0.45,
        max_association_z_m=0.60,
    )
    airborne = SceneObject3D(
        "", "cup", (0.20, 0.20, 0.30), 0.9, 1.0, ("a", "b"),
        position_mode="airborne_mask_centroid",
    )
    on_table = SceneObject3D(
        "", "cup", (0.22, 0.20, 0.0), 0.9, 1.0, ("a", "b"),
        position_mode="support_plane",
    )

    tracker.update([airborne])
    tracked = tracker.update([on_table])[0]

    assert tracked.xyz[2] == 0.0
    assert tracked.position_mode == "support_plane"


def test_tracker_expires_occluded_object_by_wall_clock_time():
    tracker = StaticObjectTracker(
        min_confirmations=1,
        hold_frames=100,
        hold_seconds=2.0,
    )
    observed = SceneObject3D(
        "", "cup", (0.2, 0.2, 0.0), 0.9, 1.0, ("a", "b")
    )
    assert tracker.update([observed], timestamp_ns=1_000_000_000)
    assert tracker.update([], timestamp_ns=2_900_000_000)[0].state == "occluded"
    assert tracker.update([], timestamp_ns=3_000_000_000) == []


def test_tracker_revives_dropped_object_id_without_a_button_press():
    tracker = StaticObjectTracker(
        min_confirmations=1,
        hold_frames=0,
        reid_seconds=30.0,
        max_association_distance_m=0.25,
    )
    observed = SceneObject3D(
        "",
        "cup",
        (0.20, 0.20, 0.0),
        0.9,
        1.0,
        ("a", "b"),
        appearance=(1.0, 0.0),
    )
    first_id = tracker.update([observed], timestamp_ns=1_000_000_000)[0].global_id

    assert tracker.update([], timestamp_ns=2_000_000_000) == []
    revived = tracker.update(
        [
            SceneObject3D(
                "",
                "cup",
                (0.24, 0.21, 0.0),
                0.9,
                1.0,
                ("a", "b"),
                appearance=(0.99, 0.01),
            )
        ],
        timestamp_ns=5_000_000_000,
    )

    assert revived[0].global_id == first_id
    assert revived[0].state == "active"


def test_unknown_class_uses_fallback_asset():
    rig = make_four_camera_rig()
    expected = np.array([0.0, 0.0, 0.0])
    detections = {}
    for camera_id, camera in list(rig.items())[:2]:
        x, y = project_point(camera, expected)
        detections[camera_id] = [
            Object2D(
                camera_id=camera_id,
                local_detection_id=0,
                label="umbrella",
                class_id=25,
                box=BoundingBox2D(x - 10, y - 50, x + 10, y),
                confidence=0.9,
            )
        ]
    objects = reconstruct_static_objects(
        detections, rig, fallback_asset_uri="generic.usda"
    )
    assert objects[0].asset_uri == "generic.usda"


def test_workspace_bounds_reject_background_triangulation():
    rig = make_four_camera_rig()
    outside = np.array([1.8, 0.0, -0.4])
    detections = {}
    for camera_id, camera in list(rig.items())[:2]:
        x, y = project_point(camera, outside)
        detections[camera_id] = [
            Object2D(
                camera_id=camera_id,
                local_detection_id=0,
                label="chair",
                class_id=56,
                box=BoundingBox2D(x - 20, y - 100, x + 20, y),
                confidence=0.9,
            )
        ]

    objects = reconstruct_static_objects(
        detections,
        rig,
        workspace_min_xyz=(-0.5, -0.5, -0.1),
        workspace_max_xyz=(0.5, 0.5, 1.0),
    )
    assert objects == []


def test_hybrid_mask_position_snaps_a_resting_object_to_the_table_plane():
    rig = make_four_camera_rig()
    base = np.array([0.12, -0.06, 1.0])
    center = base + np.array([0.0, 0.0, 0.05])
    detections = {}
    for camera_id, camera in rig.items():
        cx, cy = project_point(camera, center)
        image_height, image_width = camera.image_size[1], camera.image_size[0]
        mask = np.zeros((image_height, image_width), dtype=bool)
        half_width = 12
        y1 = max(0, round(cy - 12))
        y2 = min(image_height - 1, round(cy + 12))
        x1 = max(0, round(cx - half_width))
        x2 = min(image_width - 1, round(cx + half_width))
        mask[min(y1, y2) : max(y1, y2) + 1, x1 : x2 + 1] = True
        detections[camera_id] = [
            Object2D(
                camera_id,
                0,
                "cup",
                41,
                BoundingBox2D(x1, min(y1, y2), x2, max(y1, y2)),
                0.95,
                mask=mask,
            )
        ]

    objects = reconstruct_static_objects(
        detections,
        rig,
        anchor="mask_centroid",
        position_mode="hybrid_mask",
        support_plane_z_m=1.0,
        class_height_m={"cup": 0.10},
        snap_to_support_plane_m=0.08,
    )
    assert len(objects) == 1
    assert objects[0].position_mode == "support_plane"
    assert objects[0].xyz[2] == 1.0
    assert np.allclose(objects[0].xyz[:2], base[:2], atol=0.03)


def test_appearance_prevents_different_cups_from_swapping_ids():
    tracker = StaticObjectTracker(
        ema_alpha=1.0,
        min_confirmations=1,
        max_association_distance_m=0.6,
        appearance_weight=1.0,
        max_appearance_distance=0.2,
    )
    red = (1.0, 0.0)
    blue = (0.0, 1.0)
    tracker.update(
        [
            SceneObject3D("", "cup", (0.1, 0.0, 0.0), 0.9, 1.0, ("a", "b"), appearance=red),
            SceneObject3D("", "cup", (0.5, 0.0, 0.0), 0.9, 1.0, ("a", "b"), appearance=blue),
        ]
    )
    crossed = tracker.update(
        [
            SceneObject3D("", "cup", (0.45, 0.0, 0.0), 0.9, 1.0, ("a", "b"), appearance=red),
            SceneObject3D("", "cup", (0.15, 0.0, 0.0), 0.9, 1.0, ("a", "b"), appearance=blue),
        ]
    )
    assert crossed[0].global_id == "cup/appearance_000/instance_000"
    assert crossed[0].xyz[0] == 0.45
    assert crossed[1].global_id == "cup/appearance_001/instance_000"
    assert crossed[1].xyz[0] == 0.15


def test_global_assignment_avoids_a_greedy_track_collision():
    tracker = StaticObjectTracker(
        ema_alpha=1.0,
        min_confirmations=1,
        max_association_distance_m=0.25,
    )
    tracker.update(
        [
            SceneObject3D("", "cup", (0.0, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
            SceneObject3D("", "cup", (0.22, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
        ]
    )

    # observation_0 fits either track while observation_1 only fits the left
    # track. A greedy choice loses one identity; the Hungarian solution keeps both.
    tracked = tracker.update(
        [
            SceneObject3D("", "cup", (0.10, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
            SceneObject3D("", "cup", (-0.11, 0.0, 0.0), 0.9, 1.0, ("a", "b")),
        ]
    )

    assert len(tracked) == 2
    assert tracked[0].global_id.endswith("instance_000")
    assert tracked[0].xyz[0] == -0.11
    assert tracked[1].global_id.endswith("instance_001")
    assert tracked[1].xyz[0] == 0.10


def test_two_camera_local_tracks_preserve_id_across_noisy_3d_jump():
    tracker = StaticObjectTracker(
        ema_alpha=1.0,
        min_confirmations=1,
        max_association_distance_m=0.05,
        local_track_match_bonus_m=0.12,
    )
    first = SceneObject3D(
        "",
        "cup",
        (0.10, 0.10, 0.0),
        0.9,
        1.0,
        ("a", "b"),
        source_track_keys=(("a", 11), ("b", 21)),
    )
    second = SceneObject3D(
        "",
        "cup",
        (0.60, 0.10, 0.0),
        0.9,
        1.0,
        ("a", "b"),
        source_track_keys=(("a", 11), ("b", 21)),
    )

    first_id = tracker.update([first])[0].global_id
    tracked = tracker.update([second])

    assert len(tracked) == 1
    assert tracked[0].global_id == first_id


def test_visibility_state_uses_current_camera_local_support():
    tracker = StaticObjectTracker(min_confirmations=1, hold_frames=5)
    observed = SceneObject3D(
        "",
        "cup",
        (0.20, 0.20, 0.0),
        0.9,
        1.0,
        ("a", "b", "c"),
        source_track_keys=(("a", 1), ("b", 2), ("c", 3)),
    )
    tracker.update([observed])

    box = BoundingBox2D(0, 0, 10, 10)

    def detection(camera_id: str, track_id: int) -> Object2D:
        return Object2D(
            camera_id,
            0,
            "cup",
            47,
            box,
            0.9,
            local_track_id=track_id,
        )

    uncertain = tracker.update(
        [],
        detections_by_camera={
            "a": [detection("a", 1)],
            "b": [detection("b", 2)],
        },
    )[0]
    assert uncertain.state == "uncertain"
    assert uncertain.source_cameras == ("a", "b")

    partial = tracker.update(
        [], detections_by_camera={"a": [detection("a", 1)]}
    )[0]
    assert partial.state == "partial"
    assert partial.source_cameras == ("a",)

    occluded = tracker.update([], detections_by_camera={})[0]
    assert occluded.state == "occluded"
    assert occluded.source_cameras == ()


def test_reassigned_local_track_cannot_keep_an_old_global_object_alive():
    """A changed cross-view grouping must transfer, not duplicate, 2D ownership."""
    from dataclasses import replace

    tracker = StaticObjectTracker(min_confirmations=1, hold_seconds=0.5)
    left = SceneObject3D(
        "", "cup", (-0.2, 0, 0), .9, 1., ("a", "b", "c"),
        source_track_keys=(("a", 1), ("b", 1), ("c", 1)),
    )
    right = replace(left, xyz=(0.2, 0, 0),
                    source_track_keys=(("a", 2), ("b", 2), ("c", 2)))
    initial = tracker.update([left, right], timestamp_ns=0)
    left_id, right_id = [item.global_id for item in initial]
    detection = Object2D("a", 0, "cup", 47, BoundingBox2D(0, 0, 10, 10),
                         .9, local_track_id=1)
    # The surviving right-hand global track now owns a:1, with b:2/c:2
    # continuity. The unmatched left track must not borrow the same detection.
    reassigned = replace(right, source_track_keys=(("a", 1), ("b", 2), ("c", 2)),
                         source_detection_keys=(("a", 0), ("b", 0), ("c", 0)))
    output = {item.global_id: item for item in tracker.update(
        [reassigned], timestamp_ns=100_000_000, detections_by_camera={"a": [detection]})}
    assert output[right_id].state == "active"
    assert output[left_id].state == "occluded"
    assert output[left_id].source_detection_keys == ()

    # On the next frame there is only 2D evidence, with no valid triangulation.
    # Ownership must stay with the new owner, not revert to both old histories.
    output = {item.global_id: item for item in tracker.update(
        [], timestamp_ns=200_000_000, detections_by_camera={"a": [detection]})}
    assert output[right_id].state == "partial"
    assert output[left_id].state == "occluded"
    output = {item.global_id: item for item in tracker.update(
        [], timestamp_ns=600_000_000, detections_by_camera={"a": [detection]})}
    assert left_id not in output
    assert output[right_id].state == "partial"


@pytest.mark.parametrize("field, keys", [
    ("source_detection_keys", (("a", 0),)),
    ("source_track_keys", (("a", 1),)),
])
def test_reused_observation_evidence_is_rejected_before_mutating_tracker(field, keys):
    from dataclasses import replace

    tracker = StaticObjectTracker(min_confirmations=1)
    item = SceneObject3D("", "cup", (0, 0, 0), .9, 1., ("a", "b"), **{field: keys})
    with pytest.raises(ValueError, match="already assigned"):
        tracker.update([item, replace(item, xyz=(0.2, 0, 0))])
    assert not tracker._tracks


def test_local_track_numbers_are_scoped_to_camera_and_label():
    tracker = StaticObjectTracker(min_confirmations=1)
    objects = [
        SceneObject3D("", label, (0, 0, 0), .9, 1., (camera,),
                      source_track_keys=((camera, 1),))
        for camera, label in [("a", "cup"), ("b", "cup"), ("a", "book")]
    ]
    assert len(tracker.update(objects)) == 3


def test_tracker_holds_time_held_labels_by_time_only():
    tracker = StaticObjectTracker(min_confirmations=1, hold_frames=2, hold_seconds=9.0)
    tracker.time_held_labels = frozenset({"toy robot"})
    robot = SceneObject3D("", "toy robot", (0.2, 0.2, 0.0), 0.9, 1.0, ("a", "b"))
    cup = SceneObject3D("", "cup", (0.6, 0.2, 0.0), 0.9, 1.0, ("a", "b"))
    tracker.update([robot, cup], timestamp_ns=0)
    for frame in range(1, 6):
        objects = tracker.update([], timestamp_ns=frame * 100_000_000)
    assert [o.label for o in objects] == ["toy robot"]       # the cup expired by frames
    assert tracker.update([], timestamp_ns=9_000_000_000) == []


def test_one_object_split_across_camera_pairs_is_merged():
    """2026-10-25: one dinosaur came out as two, cameras B+C and A+D, 0.5 cm apart."""
    tracker = StaticObjectTracker(min_confirmations=1, hold_frames=5)
    first = SceneObject3D("", "toy dinosaur", (0.920, 0.586, 0.0), 0.8, 2.0, ("cam_A", "cam_D"),
                          source_detection_keys=(("cam_A", 0), ("cam_D", 0)),
                          source_track_keys=(("cam_A", 1), ("cam_D", 1)))
    second = SceneObject3D("", "toy dinosaur", (0.925, 0.588, 0.0), 0.9, 3.0, ("cam_B", "cam_C"),
                           source_detection_keys=(("cam_B", 0), ("cam_C", 0)),
                           source_track_keys=(("cam_B", 4), ("cam_C", 2)))

    output = tracker.update([first, second])

    assert len(output) == 1
    merged = output[0]
    assert merged.source_cameras == ("cam_A", "cam_B", "cam_C", "cam_D")
    assert merged.xyz[0] == pytest.approx(0.9225)
    assert dict(merged.source_track_keys) == {"cam_A": 1, "cam_D": 1, "cam_B": 4, "cam_C": 2}


def test_two_close_objects_seen_by_the_same_camera_stay_apart():
    """Two real cups side by side share cameras; only disjoint camera sets are one object."""
    tracker = StaticObjectTracker(min_confirmations=1, hold_frames=5)
    output = tracker.update([
        SceneObject3D("", "cup", (0.50, 0.50, 0.0), 0.9, 1.0, ("a", "b")),
        SceneObject3D("", "cup", (0.52, 0.50, 0.0), 0.9, 1.0, ("b", "c")),
    ])
    assert len(output) == 2


def test_a_one_camera_leftover_beside_a_seen_object_is_hidden():
    """2026-10-25: a second green robot, kept alive by one camera, sat 13 cm from the real one."""
    tracker = StaticObjectTracker(min_confirmations=1, hold_frames=5)
    box = BoundingBox2D(0, 0, 10, 10)
    real = SceneObject3D("", "toy robot", (0.31, 0.53, 0.0), 0.9, 1.0, ("a", "c"),
                         source_track_keys=(("a", 1), ("c", 1)))
    stale = SceneObject3D("", "toy robot", (0.44, 0.53, 0.0), 0.9, 1.0, ("a", "d"),
                          source_track_keys=(("a", 2), ("d", 2)))
    tracker.update([real, stale])
    seen_again = SceneObject3D("", "toy robot", (0.31, 0.53, 0.0), 0.9, 1.0, ("a", "c"),
                               source_detection_keys=(("a", 0), ("c", 0)),
                               source_track_keys=(("a", 1), ("c", 1)))
    leftover = Object2D("a", 1, "toy robot", 0, box, 0.9, local_track_id=2)

    output = {item.xyz[0]: item for item in tracker.update(
        [seen_again], detections_by_camera={"a": [leftover]})}

    assert output[0.31].state == "active"
    assert output[0.44].state == "occluded"     # hidden in the twin, not shown as a second robot
