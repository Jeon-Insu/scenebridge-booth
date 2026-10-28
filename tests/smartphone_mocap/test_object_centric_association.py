from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from smartphone_mocap.geometry import project_point
from smartphone_mocap.object_centric_association import reconstruct_object_centric
from smartphone_mocap.object_reconstruction import StaticObjectTracker
from smartphone_mocap.synthetic import make_four_camera_rig
from smartphone_mocap.types import BoundingBox2D, Object2D


POSITIONS = [(-.22, -.04, 0.), (.24, .08, 0.), (.04, .3, .12)]


def test_view_dependent_colour_can_be_soft_without_merging_nearby_cups():
    # Two close cups remain distinct via one observation per camera and geometry,
    # even when colour differs drastically between the camera pairs.
    rig, detections = scene([{0, 1, 2, 3}] * 2,
                            positions=[(-.04, 0., 0.), (.04, 0., 0.)])
    for index, name in enumerate(rig):
        detections[name] = [replace(d, appearance=(1., 0.) if index < 2 else (0., 1.))
                            for d in detections[name]]
    strict = reconstruct_object_centric(detections, rig, max_reprojection_error_px=2.,
                                       max_cross_view_appearance_distance=.8)
    soft = reconstruct_object_centric(detections, rig, max_reprojection_error_px=2.,
                                     max_cross_view_appearance_distance=1.)
    assert len(strict) == 4
    assert len(soft) == 2
    assert all(len(o.source_cameras) == 4 for o in soft)
    assert np.allclose(sorted(o.xyz for o in soft), [(-.04, 0., 0.), (.04, 0., 0.)])


def test_continuity_policy_relaxes_owned_views_but_not_new_attachments():
    rig, detections = scene([{0, 1, 2, 3}])
    names = list(rig)
    for i, name in enumerate(names):
        detections[name] = [replace(d, appearance=(1., 0.) if i < 2 else (0., 1.))
                            for d in detections[name]]
    options = dict(appearance_policy="continuity", max_reprojection_error_px=2.)
    initial = reconstruct_object_centric(detections, rig, **options)
    assert len(initial) == 1 and len(initial[0].source_cameras) == 4
    groups = {"cup": [{name: 10 for name in names}]}
    continued = reconstruct_object_centric(detections, rig,
                                           previous_track_groups=groups, **options)
    assert len(continued) == 1 and len(continued[0].source_cameras) == 4
    # Changed local ID is not trusted just because XYZ fits.
    detections[names[-1]] = [replace(detections[names[-1]][0], local_track_id=99)]
    changed = reconstruct_object_centric(detections, rig,
                                         previous_track_groups=groups, **options)
    assert len(changed) == 1
    assert names[-1] not in changed[0].source_cameras


def test_invalid_appearance_policy_rejected():
    with pytest.raises(ValueError, match="appearance policy"):
        reconstruct_object_centric({}, {}, appearance_policy="invalid")


def scene(visibility, positions=POSITIONS, distinct_appearance=False):
    rig = make_four_camera_rig()
    result = {name: [] for name in rig}
    for index, (name, camera) in enumerate(rig.items()):
        for object_id, views in enumerate(visibility):
            if index not in views:
                continue
            x, y = project_point(camera, np.asarray(positions[object_id]))
            result[name].append(Object2D(
                name, object_id, "cup", 41, BoundingBox2D(x-10, y-40, x+10, y), .95,
                appearance=tuple(np.eye(3)[object_id]) if distinct_appearance else None,
                local_track_id=object_id+10))
    return rig, result


@pytest.mark.parametrize("visibility", [
    [{0, 1, 2, 3}], [{0, 1, 2}], [{1, 3}],
    [{0, 1}, {2, 3}], [{0, 1, 2, 3}, {0, 1, 2, 3}],
    [{0, 1, 2, 3}, {0, 1, 2}, {2, 3}],
])
def test_new_objects_have_variable_view_counts(visibility):
    rig, detections = scene(visibility)
    stats = {}
    output = reconstruct_object_centric(detections, rig, max_reprojection_error_px=2., diagnostics=stats)
    assert len(output) == len(visibility)
    assert np.allclose(sorted(o.xyz for o in output), sorted(POSITIONS[:len(visibility)]))
    assert sorted(len(o.source_cameras) for o in output) == sorted(map(len, visibility))
    keys = [k for o in output for k in o.source_detection_keys]
    assert len(keys) == len(set(keys))


def test_existing_ownership_and_single_view_support_preserve_id():
    rig, detections = scene([{0, 1, 2, 3}, {0, 1, 2, 3}])
    tracker = StaticObjectTracker(min_confirmations=1, ema_alpha=1.)
    first = tracker.update(reconstruct_object_centric(detections, rig, max_reprojection_error_px=2.))
    rig, visible = scene([{0}, {1, 2, 3}])
    stats = {}
    measured = reconstruct_object_centric(
        visible, rig, max_reprojection_error_px=2.,
        previous_track_groups=tracker.local_track_groups(), diagnostics=stats)
    assert len(measured) == 1  # no fabricated XYZ from the single-view cup
    output = tracker.update(measured, detections_by_camera=visible)
    assert [o.global_id for o in output] == [o.global_id for o in first]
    assert output[0].state == "partial"
    assert output[0].xyz == first[0].xyz
    assert output[1].state == "active"
    assert stats["seeded_groups"] == 1


def test_existing_tracks_bypass_cross_object_pair_search():
    rig, detections = scene([{0, 1, 2, 3}] * 3)
    groups = {"cup": [{name: object_id + 10 for name in rig} for object_id in range(3)]}
    stats = {}
    output = reconstruct_object_centric(detections, rig, previous_track_groups=groups,
                                        max_reprojection_error_px=2., diagnostics=stats)
    assert len(output) == 3
    assert stats["seeded_groups"] == 3
    assert stats["pair_checks"] == 0  # established objects need no cross-object edge search
    assert stats["group_fits"] == 3  # one full-group solve per existing object


def test_unmatched_camera_view_can_rejoin_existing_object_with_changed_local_id():
    rig, detections = scene([{0, 1, 2, 3}])
    names = list(rig)
    detections[names[3]] = [replace(detections[names[3]][0], local_track_id=99)]
    groups = {"cup": [{name: 10 for name in rig}]}
    output = reconstruct_object_centric(detections, rig, previous_track_groups=groups,
                                        max_reprojection_error_px=2.)
    assert len(output) == 1
    assert len(output[0].source_cameras) == 4
    assert (names[3], 99) in output[0].source_track_keys


def test_one_view_new_detection_stays_pending():
    rig, detections = scene([{0}])
    stats = {}
    assert reconstruct_object_centric(detections, rig, diagnostics=stats) == []
    assert stats["pending_detections"] == 1


def test_inconsistent_known_view_does_not_destroy_other_views():
    rig, detections = scene([{0, 1, 2, 3}])
    names = list(rig)
    detections[names[3]] = [replace(detections[names[3]][0], box=BoundingBox2D(0, 0, 10, 10))]
    stats = {}
    output = reconstruct_object_centric(detections, rig, max_reprojection_error_px=2.,
                previous_track_groups={"cup": [{name: 10 for name in rig}]}, diagnostics=stats)
    assert len(output) == 1
    assert len(output[0].source_cameras) == 3
    assert stats["seed_conflicts"] == 1


def test_ambiguous_historical_owner_is_not_arbitrarily_selected():
    rig, detections = scene([{0, 1, 2, 3}])
    group = {name: 10 for name in rig}
    stats = {}
    output = reconstruct_object_centric(detections, rig,
                previous_track_groups={"cup": [group, dict(group)]}, diagnostics=stats)
    assert output == []
    assert stats["ownership_conflicts"] == 4


def test_workspace_and_support_plane_are_applied_without_changing_measurements():
    rig, detections = scene([{0, 1, 2, 3}])
    output = reconstruct_object_centric(detections, rig, position_mode="hybrid_mask",
                class_height_m={"cup": .1}, max_reprojection_error_px=2.)
    assert output[0].position_mode == "support_plane"
    assert output[0].xyz[2] == 0.
    assert reconstruct_object_centric(detections, rig, workspace_min_xyz=(1, 1, 1)) == []


def test_booth_config_selects_object_centric_without_loading_detector():
    from smartphone_mocap.perception import PerceptionProcessor

    root = Path(__file__).resolve().parents[2]
    processor = PerceptionProcessor(root / "smartphone_mocap/config/booth.json",
        persist_appearance=False, object_detection_overrides={"enabled": False})
    assert processor.association_backend == "object_centric"


def test_unknown_association_method_is_not_silently_ignored():
    from smartphone_mocap.perception import PerceptionProcessor

    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="Unsupported association backend"):
        PerceptionProcessor(root / "smartphone_mocap/config/booth.json",
            persist_appearance=False,
            object_detection_overrides={"enabled": False, "association_backend": "typo"})


def test_single_view_birth_puts_a_confident_detection_on_the_table():
    rig, detections = scene([{0}])
    output = reconstruct_object_centric(detections, rig, single_view_min_confidence=0.6)
    assert len(output) == 1
    assert output[0].position_mode == "single_view" and len(output[0].source_cameras) == 1
    assert np.allclose(output[0].xyz, POSITIONS[0], atol=1e-3)     # ray hits the table plane at the base


def test_single_view_birth_needs_its_own_higher_confidence():
    rig, detections = scene([{0}])                                   # detections are 0.95
    assert reconstruct_object_centric(detections, rig, single_view_min_confidence=0.99) == []


def test_single_view_birth_does_not_duplicate_an_object_other_cameras_already_placed():
    rig, detections = scene([{0, 1, 2, 3}])
    names = list(rig)
    lone = detections[names[0]][0]
    detections[names[0]] = [replace(lone, box=BoundingBox2D(lone.box.x1 + 6, lone.box.y1, lone.box.x2 + 6, lone.box.y2))]
    output = reconstruct_object_centric(detections, rig, max_reprojection_error_px=2., single_view_min_confidence=0.6)
    assert len(output) == 1 and len(output[0].source_cameras) == 3   # the off view is not a second cup


def test_single_view_birth_leaves_known_objects_to_the_tracker():
    rig, detections = scene([{0, 1, 2, 3}, {0, 1, 2, 3}])
    tracker = StaticObjectTracker(min_confirmations=1, ema_alpha=1.)
    tracker.update(reconstruct_object_centric(detections, rig, max_reprojection_error_px=2.))
    rig, visible = scene([{0}, {1, 2, 3}])
    measured = reconstruct_object_centric(visible, rig, max_reprojection_error_px=2., single_view_min_confidence=0.6,
                                          previous_track_groups=tracker.local_track_groups())
    assert len(measured) == 1    # the lone view belongs to a known cup: the tracker keeps it as 'partial'
