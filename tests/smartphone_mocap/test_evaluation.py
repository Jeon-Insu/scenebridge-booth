import json

from smartphone_mocap.evaluation import (
    evaluate_detection_messages,
    evaluate_tracking_messages,
)


def _message(frame, objects):
    return {
        "schema": "gist.smartphone_mocap.v1",
        "frame_number": frame,
        "timestamp_ns": frame * 200_000_000,
        "objects": objects,
    }


def _object(object_id, *, state="active", xyz=(0.1, 0.2, 0.0), cameras=("a", "b")):
    return {
        "id": object_id,
        "label": "cup",
        "state": state,
        "xyz": list(xyz),
        "source_cameras": list(cameras),
        "position_mode": "support_plane" if xyz[2] == 0.0 else "airborne_mask_centroid",
        "reprojection_error_px": 4.0,
    }


def test_evaluation_counts_instances_transitions_and_coordinate_jumps():
    messages = [
        _message(0, [_object("cup/appearance_000/instance_000")]),
        _message(
            1,
            [
                _object("cup/appearance_000/instance_000", state="occluded"),
                _object("cup/appearance_000/instance_001", xyz=(0.5, 0.2, 0.0)),
            ],
        ),
        _message(
            2,
            [
                _object("cup/appearance_000/instance_000", xyz=(0.2, 0.2, 0.1)),
                _object("cup/appearance_000/instance_001", xyz=(0.51, 0.2, 0.0)),
            ],
        ),
    ]

    metrics = evaluate_tracking_messages(messages)

    assert metrics["message_count"] == 3
    assert metrics["object_count_distribution"] == {"1": 1, "2": 2}
    assert metrics["label_instance_count"]["cup"] == 2
    assert metrics["label_simultaneous_max"]["cup"] == 2
    assert metrics["state_transition_count"] == 2
    assert metrics["source_camera_count_distribution"] == {"2": 5}
    assert metrics["position_mode_distribution"] == {
        "airborne_mask_centroid": 1,
        "support_plane": 4,
    }
    assert metrics["per_id"]["cup/appearance_000/instance_000"]["step_distance_m"]["max"] > 0.14


def test_evaluation_scores_annotated_expected_counts():
    messages = [
        _message(10, [_object("cup/appearance_000/instance_000")]),
        _message(
            11,
            [
                _object("cup/appearance_000/instance_000"),
                _object("cup/appearance_000/instance_001"),
            ],
        ),
        _message(
            12,
            [
                _object("cup/appearance_000/instance_000"),
                _object("cup/appearance_000/instance_001"),
            ],
        ),
    ]
    annotations = {
        "scenarios": [
            {
                "name": "two_cups",
                "start_index": 10,
                "end_index": 12,
                "expected_counts": {"cup": 2},
            }
        ]
    }

    metrics = evaluate_tracking_messages(messages, annotations=annotations)

    scenario = metrics["scenarios"][0]
    assert scenario["evaluated_messages"] == 3
    assert scenario["count_match_ratio"] == 2 / 3
    assert scenario["distinct_ids_by_label"] == {"cup": 2}


def test_detection_evaluation_reports_per_camera_labels_and_association():
    messages = [
        {
            "frame_number": 0,
            "cameras": {
                "left": [
                    {"label": "cup", "mask_area_px": 100, "tracked_id": "cup_0"},
                    {"label": "book", "mask_area_px": 80, "tracked_id": None},
                ],
                "right": [
                    {"label": "cup", "mask_area_px": 120, "tracked_id": "cup_0"}
                ],
            },
        }
    ]

    metrics = evaluate_detection_messages(messages)

    assert metrics["total_detections"] == 3
    assert metrics["matched_detections"] == 2
    assert metrics["association_ratio"] == 2 / 3
    assert metrics["per_camera"]["left"]["labels"] == {"book": 1, "cup": 1}


import pytest


ROBOT_URI = "/props/green_robot/model_placed.usda"


def _robot(cameras, state="active", error=4.0, uri=ROBOT_URI):
    return {
        "id": "toy robot/appearance_000/instance_000",
        "label": "toy robot",
        "state": state,
        "xyz": [0.5, 0.3, 0.0],
        "source_cameras": list(cameras),
        "position_mode": "support_plane",
        "reprojection_error_px": error,
        "asset_uri": uri,
    }


def _robot_scenario(**extra):
    return {"scenarios": [{"name": "robot", "start_index": 10, "end_index": 12,
                           "expected_counts": {"toy robot": 1}, **extra}]}


def test_scenario_reports_per_prop_two_camera_detection_ratio():
    messages = [
        _message(10, [_robot(("a", "b"), error=3.0)]),
        _message(11, [_robot(("a",), error=9.0)]),   # one camera: not a 2-camera detection
        _message(12, []),
    ]
    metrics = evaluate_tracking_messages(
        messages,
        annotations=_robot_scenario(expected_props=["green_robot"]),
        prop_ids_by_uri={ROBOT_URI: "green_robot"},
    )
    prop = metrics["scenarios"][0]["props"]["green_robot"]
    assert prop["detected_ratio_2cam"] == pytest.approx(1 / 3)
    assert prop["reprojection_error_px_median"] == 3.0


def test_prop_metrics_survive_missing_uri_and_absent_prop():
    messages = [_message(10, [_robot(("a", "b"), uri=None)]), _message(11, [])]
    metrics = evaluate_tracking_messages(
        messages,
        annotations=_robot_scenario(expected_props=["green_robot", "devil_duck"]),
        prop_ids_by_uri={ROBOT_URI: "green_robot"},
    )
    props = metrics["scenarios"][0]["props"]
    assert props["green_robot"] == {"detected_ratio_2cam": 0.0, "reprojection_error_px_median": None}
    assert props["devil_duck"] == {"detected_ratio_2cam": 0.0, "reprojection_error_px_median": None}


def test_old_annotations_keep_their_output():
    messages = [_message(10, [_robot(("a", "b"))])]
    metrics = evaluate_tracking_messages(messages, annotations=_robot_scenario())
    scenario = metrics["scenarios"][0]
    assert set(scenario) == {"name", "start_index", "end_index", "expected_counts",
                             "evaluated_messages", "count_match_ratio", "distinct_ids_by_label"}


def test_placement_latency_is_time_until_prop_first_visible():
    messages = [
        _message(10, []),
        _message(11, [_robot(("a",), state="candidate")]),   # not drawn yet
        _message(12, [_robot(("a", "b"))]),
    ]
    annotations = _robot_scenario(events=[{"frame": 10, "type": "placed", "prop": "green_robot"}])
    metrics = evaluate_tracking_messages(messages, annotations=annotations,
                                         prop_ids_by_uri={ROBOT_URI: "green_robot"})
    event = metrics["scenarios"][0]["events"][0]
    assert event["latency_s"] == pytest.approx(0.4)          # frames are 0.2 s apart


def test_placement_that_never_appears_reports_none():
    annotations = _robot_scenario(events=[{"frame": 10, "type": "placed", "prop": "green_robot"}])
    metrics = evaluate_tracking_messages([_message(10, []), _message(11, [])],
                                         annotations=annotations,
                                         prop_ids_by_uri={ROBOT_URI: "green_robot"})
    assert metrics["scenarios"][0]["events"][0]["latency_s"] is None


def test_empty_table_scenario_counts_visible_false_positives():
    messages = [_message(0, []), _message(1, [_object("ghost")]),
                _message(2, [_object("ghost"), _object("ghost2", state="candidate")])]
    annotations = {"scenarios": [{"name": "empty", "start_index": 0, "end_index": 2,
                                  "expected_counts": {}}]}
    fp = evaluate_tracking_messages(messages, annotations=annotations)["scenarios"][0]["false_positives"]
    assert fp == {"frames_with_objects": 2, "max_simultaneous_objects": 1}


def test_prop_uri_map_resolves_catalog_paths(tmp_path):
    from smartphone_mocap.evaluation import prop_uri_map
    (tmp_path / "green_robot").mkdir()
    (tmp_path / "catalog.json").write_text(json.dumps({
        "schema_version": 1,
        "assets": [{"asset_id": "green_robot", "usd_uri": "green_robot/model_placed.usda",
                    "detector_class": "toy robot"}],
    }))
    mapping = prop_uri_map(tmp_path)
    assert mapping == {str((tmp_path / "green_robot/model_placed.usda").resolve()): "green_robot"}


def test_uncertain_two_camera_objects_count_as_drawn():
    # Review 2026-10-10: the twin draws "uncertain" (two cameras support a held track);
    # leaving it out made the per-prop ratio too low.
    messages = [_message(10, [_robot(("a", "b"), state="uncertain")])]
    metrics = evaluate_tracking_messages(
        messages, annotations=_robot_scenario(expected_props=["green_robot"]),
        prop_ids_by_uri={ROBOT_URI: "green_robot"})
    assert metrics["scenarios"][0]["props"]["green_robot"]["detected_ratio_2cam"] == 1.0
