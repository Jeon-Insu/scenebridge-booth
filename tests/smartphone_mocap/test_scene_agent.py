from __future__ import annotations

import base64
import math
import pytest

from smartphone_mocap.scene_agent import (
    SceneTools,
    _snapshot_image_data_url,
    _vision_enabled,
    deterministic_response,
)


def _snapshot() -> dict:
    return {
        "schema": "gist.tabletop_agent.snapshot.v1",
        "snapshot_id": "snapshot-demo",
        "frame_number": 42,
        "interaction_frame": {
            "table_axes": {
                "east": [1.0, 0.0, 0.0],
                "north": [0.0, 1.0, 0.0],
                "up": [0.0, 0.0, 1.0],
            },
            "workspace_bounds_xy": [0.0, 1.25, 0.0, 0.75],
            "viewport": {
                "available": True,
                "camera_path": "/OmniverseKit_Persp",
                "screen_right_table_xy": [0.0, 1.0],
                "screen_up_table_xy": [-1.0, 0.0],
            },
        },
        "tracking": {
            "objects": [
                {
                    "id": "cup/white/instance_000",
                    "label": "cup",
                    "state": "active",
                    "xyz": [0.2, 0.3, 0.0],
                    "asset_uri": "ceramic_mug.usda",
                    "source_cameras": ["a", "b", "c"],
                },
                {
                    "id": "cup/white/instance_001",
                    "label": "cup",
                    "state": "partial",
                    "xyz": [0.6, 0.3, 0.0],
                    "asset_uri": "ceramic_mug.usda",
                    "source_cameras": ["a", "d"],
                },
                {
                    "id": "potted plant/green/instance_000",
                    "label": "potted plant",
                    "state": "active",
                    "xyz": [0.4, 0.4, 0.0],
                    "asset_uri": "generic.usda",
                    "source_cameras": ["a", "b", "c", "d"],
                },
            ]
        },
    }


def test_scene_tools_use_snapshot_ids_and_xyz() -> None:
    tools = SceneTools(_snapshot())
    summary = tools.get_scene_summary()
    distance = tools.measure_distance(
        "cup/white/instance_000", "cup/white/instance_001"
    )
    preview = tools.preview_layout("cup", "potted plant", 0.15)

    assert summary["counts_by_label"] == {"cup": 2, "potted plant": 1}
    assert distance["distance_m"] == 0.4
    assert preview["ok"]
    assert len(preview["placements"]) == 2
    assert tools.actions == [
        {"type": "preview_layout", "preview_id": tools.pending_preview["preview_id"], "placements": preview["placements"]}
    ]


def test_fallback_korean_layout_never_moves_tracked_objects_directly() -> None:
    response = deterministic_response("화분 양옆에 컵을 배치해줘", _snapshot())

    assert response["provider"] == "deterministic-fallback"
    assert response["actions"][0]["type"] == "preview_layout"
    assert "ghost preview" in response["reply"]


def test_viewport_relative_move_uses_captured_screen_basis() -> None:
    tools = SceneTools(_snapshot())

    result = tools.preview_move(
        "cup/white/instance_000", "right", 0.10, "viewport"
    )

    assert result["ok"]
    assert result["direction_table_xy"] == [0.0, 1.0]
    assert result["preview_xyz"] == [0.2, 0.4, 0.0]
    assert tools.actions == [
        {
            "type": "preview_layout",
            "preview_id": tools.pending_preview["preview_id"],
            "placements": [
                {
                    "object_id": "cup/white/instance_000",
                    "xyz": [0.2, 0.4, 0.0],
                }
            ],
        }
    ]


def test_cardinal_move_ignores_viewport_and_can_leave_table() -> None:
    tools = SceneTools(_snapshot())

    east = tools.preview_move(
        "cup/white/instance_001", "east", 0.20, "table"
    )
    south = tools.preview_move(
        "cup/white/instance_000", "south", 0.50, "table"
    )

    assert east["preview_xyz"] == [0.8, 0.3, 0.0]
    assert east["direction_table_xy"] == [1.0, 0.0]
    assert south["preview_xyz"] == [0.2, -0.2, 0.0]
    assert not south["clamped_to_workspace"]


def test_move_requires_a_unique_object() -> None:
    result = SceneTools(_snapshot()).preview_move("cup", "right")

    assert not result["ok"]
    assert result["candidate_ids"] == [
        "cup/white/instance_000",
        "cup/white/instance_001",
    ]


def test_spatial_language_resolves_rightmost_object_in_viewport_frame() -> None:
    snapshot = _snapshot()
    snapshot["tracking"]["objects"][1]["xyz"] = [0.6, 0.55, 0.0]
    tools = SceneTools(snapshot)
    result = tools.preview_move("오른쪽 컵", "east", 0.10, "table")

    assert result["ok"]
    assert result["object_id"] == "cup/white/instance_001"


def test_spatial_language_resolves_rightmost_plant_for_height_request() -> None:
    snapshot = _snapshot()
    snapshot["tracking"]["objects"].append({
        "id": "potted plant/green/instance_001",
        "label": "potted plant",
        "state": "active",
        "xyz": [0.8, 0.6, 0.0],
        "asset_uri": "generic.usda",
        "source_cameras": ["a", "b"],
    })
    response = deterministic_response("오른쪽 화분을 10cm 위로 올려줘", snapshot)

    assert response["actions"][0]["type"] == "move_objects"
    assert response["actions"][0]["placements"][0]["object_id"] == "potted plant/green/instance_001"


def test_focused_natural_language_targets_selected_class() -> None:
    snapshot = _snapshot()
    snapshot["selection"] = {"object_ids": ["potted plant/green/instance_000"]}
    response = deterministic_response("현재 포커스 활성화된 화분을 10cm 위로 올려줘", snapshot)

    assert response["actions"][0]["type"] == "move_objects"
    assert response["actions"][0]["placements"][0]["object_id"] == "potted plant/green/instance_000"


def test_fallback_move_creates_only_a_ghost_preview() -> None:
    response = deterministic_response(
        "cup/white/instance_000을 오른쪽으로 10cm 이동해줘", _snapshot()
    )

    assert response["provider"] == "deterministic-fallback"
    assert response["actions"][0]["type"] == "preview_layout"
    assert response["actions"][0]["placements"][0]["xyz"] == [0.2, 0.4, 0.0]
    assert "원본 객체는 움직이지 않았습니다" in response["reply"]


def test_missing_asset_query_finds_generic_asset() -> None:
    result = deterministic_response("전용 애셋이 없는 객체를 찾아줘", _snapshot())

    assert "potted plant/green/instance_000" in result["reply"]


def test_apply_preview_across_requests_and_cancel() -> None:
    first = SceneTools(_snapshot())
    first.preview_move("cup/white/instance_000", "east", 0.1, "table")
    snapshot = _snapshot()
    snapshot["twin_state"] = {"preview": first.pending_preview}
    response = deterministic_response("그대로 적용해줘", snapshot)
    assert response["actions"] == [{"type": "apply_preview", "preview_id": first.pending_preview["preview_id"]}]
    second = SceneTools(snapshot)
    second.cancel_layout()
    assert not second.apply_layout()["ok"]
    assert not SceneTools(_snapshot()).apply_layout()["ok"]


def test_explicit_direct_move_uses_virtual_coordinates_without_mutating_snapshot() -> None:
    snapshot = _snapshot()
    snapshot["twin_state"] = {"offsets": {"cup/white/instance_000": [0.2, 0.0, 0.0]}}
    response = deterministic_response("cup/white/instance_000을 동쪽으로 10cm 바로 이동해줘", snapshot)
    assert len(response["actions"]) == 1
    assert response["actions"][0]["type"] == "move_objects"
    assert response["actions"][0]["placements"][0]["xyz"] == pytest.approx([0.5, 0.3, 0.0])
    assert snapshot["tracking"]["objects"][0]["xyz"] == [0.2, 0.3, 0.0]
    tools = SceneTools(snapshot)
    tools.move_object("cup/white/instance_000", "east", 0.1, "table")
    result = tools.move_object("cup/white/instance_000", "east", 0.1, "table")
    assert result["target_xyz"] == pytest.approx([0.6, 0.3, 0.0])
    tools.reset_moves("cup/white/instance_000")
    assert tools.list_objects("cup/white/instance_000")["objects"][0]["xyz"] == (0.2, 0.3, 0.0)


def test_direct_move_rejects_ambiguous_id_and_nonfinite_distance() -> None:
    tools = SceneTools(_snapshot())
    assert not tools.move_object("cup", "east")["ok"]
    assert not tools.move_object("cup/white/instance_000", "east", float("nan"))["ok"]
    assert not tools.actions


def test_height_uses_current_virtual_z_and_keeps_xy() -> None:
    snapshot = _snapshot()
    snapshot["twin_state"] = {"offsets": {"cup/white/instance_000": [0.1, 0.05, 0.2]}}
    tools = SceneTools(snapshot)
    result = tools.change_height("cup/white/instance_000", delta_m=0.3)
    assert result["target_xyz"] == pytest.approx([0.3, 0.35, 0.5])
    assert tools.actions[-1]["type"] == "move_objects"
    lowered = tools.change_height("cup/white/instance_000", delta_m=-0.1)
    assert lowered["height_m"] == pytest.approx(0.4)
    on_table = tools.change_height("cup/white/instance_000", height_m=0.0)
    assert on_table["target_xyz"] == pytest.approx([0.3, 0.35, 0.0])
    assert snapshot["tracking"]["objects"][0]["xyz"] == [0.2, 0.3, 0.0]


def test_height_preview_apply_and_invalid_requests() -> None:
    tools = SceneTools(_snapshot())
    for arguments in ({}, {"delta_m": 0.3, "height_m": 0.4}, {"height_m": float("nan")}):
        assert not tools.change_height("cup/white/instance_000", **arguments)["ok"]
    assert not tools.change_height("cup", delta_m=0.3)["ok"]
    assert not tools.actions
    tools.change_height("cup/white/instance_000", height_m=0.3, preview_only=True)
    assert tools.actions[-1]["type"] == "preview_layout"
    assert tools.list_objects("cup/white/instance_000")["objects"][0]["xyz"][2] == 0.0
    assert tools.apply_layout()["ok"]
    assert tools.list_objects("cup/white/instance_000")["objects"][0]["xyz"][2] == 0.3
    result = tools.change_height("cup/white/instance_000", delta_m=-1.0)
    assert result["height_m"] == 0 and result["clamped_to_table"]


@pytest.mark.parametrize("phrase,expected_z", [
    ("위로 30cm 올려줘", 0.5),
    ("Z축으로 30cm 들어 올려줘", 0.5),
    ("10cm 내려줘", 0.1),
    ("Z축 -10cm 이동해줘", 0.1),
    ("높이를 30cm로 맞춰줘", 0.3),
    ("lift by 300mm", 0.5),
])
def test_height_language_routes_to_z(phrase, expected_z) -> None:
    snapshot = _snapshot()
    snapshot["twin_state"] = {"offsets": {"cup/white/instance_000": [0.0, 0.0, 0.2]}}
    response = deterministic_response(f"cup/white/instance_000을 {phrase}", snapshot)
    assert response["actions"][0]["type"] == "move_objects"
    assert response["actions"][0]["placements"][0]["xyz"] == pytest.approx([0.2, 0.3, expected_z])


def test_explicit_screen_up_still_moves_on_xy_plane() -> None:
    response = deterministic_response("cup/white/instance_000을 화면 위쪽으로 30cm 바로 올려줘", _snapshot())
    assert response["actions"][0]["placements"][0]["xyz"] == pytest.approx([-0.1, 0.3, 0.0])


def test_rotation_composes_fixed_axes_and_absolute_replaces_orientation() -> None:
    tools = SceneTools(_snapshot())
    object_id = "cup/white/instance_000"
    tools.rotate_object(object_id, 90, "x")
    combined = tools.rotate_object(object_id, 90, "z")
    assert combined["rotation_wxyz"] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    replaced = tools.rotate_object(object_id, 90, "z", mode="absolute")
    assert replaced["rotation_wxyz"] == pytest.approx([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert tools.list_objects(object_id)["objects"][0]["xyz"] == (0.2, 0.3, 0.0)
    assert "xyz" not in tools.actions[-1]["placements"][0]


def test_scale_relative_absolute_axis_and_reset_are_independent() -> None:
    snapshot = _snapshot()
    snapshot["twin_state"] = {"transforms": {"cup/white/instance_000": {"scale": [2, 2, 2]}}}
    tools = SceneTools(snapshot)
    object_id = "cup/white/instance_000"
    assert tools.scale_object(object_id, 1.5)["scale"] == [3, 3, 3]
    assert tools.scale_object(object_id, 1.5, mode="absolute")["scale"] == [1.5, 1.5, 1.5]
    assert tools.scale_object(object_id, 2, axis="z")["scale"] == [1.5, 1.5, 3]
    tools.rotate_object(object_id, 90)
    tools.change_height(object_id, height_m=0.2)
    tools.reset_transform(object_id, ["scale"])
    item = tools.list_objects(object_id)["objects"][0]
    assert item["scale"] == [1, 1, 1]
    assert item["rotation_wxyz"][3] == pytest.approx(math.sqrt(0.5))
    assert item["xyz"][2] == 0.2
    assert snapshot["twin_state"]["transforms"][object_id]["scale"] == [2, 2, 2]


def test_shape_preview_applies_only_after_confirmation() -> None:
    tools = SceneTools(_snapshot())
    object_id = "cup/white/instance_000"
    tools.scale_object(object_id, 2, preview_only=True)
    assert tools.list_objects(object_id)["objects"][0]["scale"] == [1, 1, 1]
    snapshot = _snapshot()
    snapshot["twin_state"] = {"preview": tools.pending_preview}
    next_turn = SceneTools(snapshot)
    assert next_turn.apply_layout()["ok"]
    assert next_turn.list_objects(object_id)["objects"][0]["scale"] == [2, 2, 2]
    assert next_turn.list_objects(object_id)["objects"][0]["xyz"] == (0.2, 0.3, 0.0)


def test_shape_invalid_input_does_not_queue_actions() -> None:
    tools = SceneTools(_snapshot())
    for value in (0, -1, float("nan"), float("inf")):
        assert not tools.scale_object("cup/white/instance_000", value)["ok"]
    assert not tools.rotate_object("cup", 90)["ok"]
    assert not tools.rotate_object("cup/white/instance_000", float("inf"))["ok"]
    assert not tools.rotate_object("cup/white/instance_000", 90, axis="screen")["ok"]
    assert not tools.reset_transform(components=["bad"])["ok"]
    assert not tools.actions


@pytest.mark.parametrize("phrase,field,expected", [
    ("Z축으로 90도 회전해줘", "rotation_wxyz", [math.sqrt(.5), 0, 0, math.sqrt(.5)]),
    ("크기를 1.5배로 맞춰줘", "scale", [1.5, 1.5, 1.5]),
    ("높이 방향으로만 2배 늘려줘", "scale", [1, 1, 2]),
    ("현재보다 20% 작게 줄여줘", "scale", [.8, .8, .8]),
])
def test_shape_language_function_calls(phrase, field, expected) -> None:
    response = deterministic_response(f"cup/white/instance_000을 {phrase}", _snapshot())
    assert response["actions"][0]["type"] == "transform_objects"
    assert response["actions"][0]["placements"][0][field] == pytest.approx(expected)


def test_korean_alias_selection_and_same_class_distance() -> None:
    selection = deterministic_response("화분을 선택해줘", _snapshot())
    distance = deterministic_response("컵 두 개 거리를 알려줘", _snapshot())

    assert selection["actions"] == [
        {
            "type": "select_object",
            "object_id": "potted plant/green/instance_000",
        }
    ]
    assert "0.400 m" in distance["reply"]


def test_multimodal_model_detection_is_profile_independent() -> None:
    assert _vision_enabled("auto", "nvidia/Gemma-4-26B-A4B-NVFP4")
    assert _vision_enabled("auto", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    assert not _vision_enabled("auto", "nvidia/Qwen3-8B-FP4")
    assert not _vision_enabled(False, "google/gemma-4-31B-it")


def test_snapshot_mosaic_is_embedded_as_local_data_url(tmp_path) -> None:
    image = tmp_path / "camera_mosaic.jpg"
    image.write_bytes(b"small-local-jpeg")

    result = _snapshot_image_data_url({"camera_mosaic": str(image)})

    assert result == (
        "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
    )
    assert _snapshot_image_data_url(
        {"camera_mosaic": str(image)}, max_bytes=4
    ) is None


def test_sglang_starts_with_manifest_and_no_automatic_image_attachment(tmp_path, monkeypatch):
    import smartphone_mocap.scene_agent as agent
    image = tmp_path / "camera_mosaic.jpg"
    image.write_bytes(b"four-camera-paired-mosaic")
    snapshot = _snapshot()
    snapshot["camera_mosaic"] = str(image)
    snapshot["camera_files"] = {f"cam_{i}": {"raw": f"cam_{i}_raw.jpg"} for i in range(4)}
    captured = []
    monkeypatch.setattr(agent, "_resolve_model", lambda *args: "nvidia/Gemma-4-26B-A4B-NVFP4")

    def capture_request(url, payload, timeout):
        captured.append(payload)
        return {"choices": [{"message": {"content": "테스트 응답"}}]}

    monkeypatch.setattr(agent, "_post_json", capture_request)
    result = agent.sglang_response("장면 설명", snapshot, base_url="http://unused/v1", model="auto", timeout_seconds=1)
    assert not result["vision_input"]
    attachments = [part for message in captured[0]["messages"] if isinstance(message["content"], list)
                   for part in message["content"] if part["type"] == "image_url"]
    assert attachments == []
    assert "Available evidence" in captured[0]["messages"][1]["content"]
