import copy
import json

import numpy as np
import pytest

from smartphone_mocap.agent_evidence import EvidenceTools
from smartphone_mocap.scene_agent import SceneTools, sglang_response
from smartphone_mocap.agent_control import camera_evidence_metadata
from smartphone_mocap.types import CameraCalibration, FramePacket


def snapshot(tmp_path):
    result = {
        "snapshot_id": "fixed-123", "rig": {"rough_calibration_only": True},
        "tracking": {"timestamp_ns": 123, "objects": [
            {"id": "plant_0", "label": "potted plant", "xyz": [0, 0, 0], "state": "active"},
            {"id": "cup_0", "label": "cup", "xyz": [0.1, 0, 0], "state": "active"},
            {"id": "cup_1", "label": "cup", "xyz": [0.5, 0, 0], "state": "active"},
        ]},
        "twin_state": {"scene_version": "v1", "captured_at_ns": 789, "runtime_objects": {
            "plant_0": {"xyz": [0, 0, 0]}, "cup_0": {"xyz": [0.9, 0, 0]}, "cup_1": {"xyz": [0.5, 0, 0]}}},
        "camera_files": {}, "camera_metadata": {},
    }
    for i in range(4):
        cid = f"cam_{i}"
        path = tmp_path / f"{cid}.jpg"
        path.write_bytes(b"test-image-" + bytes([i]))
        result["camera_files"][cid] = {"raw": str(path)}
        result["camera_metadata"][cid] = {"corner": f"corner-{i}", "capture_time_ns": 123 + i}
    view = tmp_path / "viewport.png"
    view.write_bytes(b"virtual-pixels")
    result["twin_view"] = {"available": True, "scene_version": "v1", "path": str(view)}
    return result


def test_sources_have_different_nearest_objects_without_mutation(tmp_path):
    data = snapshot(tmp_path)
    original = copy.deepcopy(data)
    tools = EvidenceTools(data)
    real = tools.spatial_relations("physical", "plant_0")
    twin = tools.spatial_relations("twin", "plant_0")
    assert real["relations"][0]["id"] == "cup_0"
    assert twin["relations"][0]["id"] == "cup_1"
    assert real["captured_at_ns"] == 123 and twin["captured_at_ns"] == 789
    assert twin["scene_version"] == "v1"
    assert data == original


def test_missing_physical_xyz_never_uses_virtual_xyz(tmp_path):
    data = snapshot(tmp_path)
    data["tracking"]["objects"][0]["xyz"] = None
    result = EvidenceTools(data).spatial_relations("physical", "plant_0")
    assert not result["ok"]
    assert EvidenceTools(data).spatial_relations("twin", "plant_0")["ok"]


def test_unknown_and_ambiguous_objects_cannot_be_selected_or_generated(tmp_path):
    tools = SceneTools(snapshot(tmp_path))
    for query in ("missed_doll", "cup"):
        assert not tools.select_object(query)["ok"]
        assert not tools.queue_reconstruction(query)["ok"]
    assert tools.actions == []


def test_four_images_are_each_followed_by_own_metadata(tmp_path):
    tools = EvidenceTools(snapshot(tmp_path))
    result = tools.inspect_images("physical")
    assert len(result["images"]) == 4
    for i in range(4):
        assert tools.pending_images[2*i]["type"] == "image_url"
        text = tools.pending_images[2*i+1]["text"]
        assert f'"camera_id": "cam_{i}"' in text and f'corner-{i}' in text
    tools.pending_images = []
    tools.inspect_images("physical")
    assert tools.pending_images == []


def test_no_silent_fallback_for_missing_stale_or_disabled_images(tmp_path):
    data = snapshot(tmp_path)
    tools = EvidenceTools(data, vision_enabled=False)
    assert not tools.inspect_images("physical")["ok"]
    data["twin_view"]["scene_version"] = "old"
    assert not EvidenceTools(data).inspect_images("twin")["ok"]
    data["camera_files"]["cam_0"] = {}
    tools = EvidenceTools(data)
    assert not tools.inspect_images("physical", ["cam_0"])["ok"]
    assert tools.pending_images == []
    assert not tools.call("inspect_images", {"source": "physical", "camera_ids": ["/etc/passwd"]})["ok"]


def test_image_byte_budget_is_cumulative(tmp_path):
    tools = EvidenceTools(snapshot(tmp_path), max_bytes=20)
    result = tools.inspect_images("physical")
    assert len(result["images"]) == 1 and len(result["errors"]) == 3


def test_geometry_requires_source_and_unique_reference(tmp_path):
    tools = EvidenceTools(snapshot(tmp_path))
    assert not tools.call("spatial_relations", {"reference_query": "plant_0"})["ok"]
    assert not tools.call("spatial_relations", {"source": "physical", "reference_query": "cup"})["ok"]
    assert not tools.call("inspect_scene", {"source": "whatever"})["ok"]


def test_agent_can_choose_four_physical_images_then_twin_without_eager_pixels(tmp_path, monkeypatch):
    import smartphone_mocap.scene_agent as agent
    requests, events = [], []
    def model(url, payload, timeout):
        requests.append(copy.deepcopy(payload))
        if len(requests) <= 2:
            source = "physical" if len(requests) == 1 else "twin"
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": source, "type": "function", "function": {"name": "inspect_images", "arguments": json.dumps({"source": source})}}]}}]}
        return {"choices": [{"message": {"content": "실물 이미지와 트윈 이미지를 비교했습니다."}}]}
    monkeypatch.setattr(agent, "_post_json", model)
    result = sglang_response("비교해 줘", snapshot(tmp_path), base_url="http://unused", model="vision", timeout_seconds=1, progress=events.append)
    assert all(isinstance(m["content"], str) for m in requests[0]["messages"])
    images = [p for m in requests[-1]["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"]
    assert len(images) == 5 and result["vision_input"]
    assert [e["source"] for e in result["evidence_trace"]] == ["physical", "twin"]
    assert result["actions"] == [] and any(e["phase"] == "tool" for e in events)


def test_last_round_forces_answer_and_does_not_execute_extra_calls(tmp_path, monkeypatch):
    import smartphone_mocap.scene_agent as agent
    payloads = []
    def model(url, payload, timeout):
        payloads.append(payload)
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "loop", "type": "function", "function": {
            "name": "inspect_scene", "arguments": '{"source":"physical"}'}}]}}]}
    monkeypatch.setattr(agent, "_post_json", model)
    result = sglang_response("test", snapshot(tmp_path), base_url="http://unused", model="vision", timeout_seconds=1, max_tool_rounds=1)
    assert payloads[-1]["tool_choice"] == "none"
    assert len(result["evidence_trace"]) == 1


def test_camera_metadata_uses_actual_packet_and_calibration():
    transform = np.eye(4)
    transform[:3, 3] = [-1, -2, -3]
    calibration = CameraCalibration("a", (640, 480), np.eye(3), np.zeros(5), transform, time_offset_ns=10)
    packet = FramePacket("a", 17, 1234, np.zeros((480, 640, 3), np.uint8))
    meta = camera_evidence_metadata({"cameras": [{"id": "a", "corner": "top_left"}]}, {"a": calibration}, {"a": packet})["a"]
    assert meta["position_table_m"] == [1, 2, 3]
    assert meta["forward_table"] == [0, 0, 1]
    assert meta["capture_time_ns"] == 1234 and meta["adjusted_capture_time_ns"] == 1244
    assert meta["image_size_wh"] == [640, 480] and meta["frame_number"] == 17


def test_context_overflow_reports_failure_without_tracker_only_fallback(tmp_path, monkeypatch):
    import io
    import urllib.error
    import smartphone_mocap.scene_agent as agent
    def fail(*args):
        raise urllib.error.HTTPError("http://unused", 400, "Bad Request", {}, io.BytesIO(
            b'{"message":"Requested token count exceeds maximum context length"}'))
    monkeypatch.setattr(agent, "_post_json", fail)
    result = sglang_response("실제 이미지 확인", snapshot(tmp_path), base_url="http://unused", model="vision", timeout_seconds=1)
    assert "문맥 한도" in result["reply"]
    assert "context length" in result["provider_error"]
    assert result["actions"] == [] and result["provider"].startswith("sglang")


def test_selected_object_and_short_name_are_grounded_to_stable_ids(tmp_path):
    data = snapshot(tmp_path)
    data["selection"] = {"object_ids": ["cup_1"], "object_names": {"cup_0": "Cup 01", "cup_1": "Cup 02"}}
    tools = SceneTools(data)
    assert tools.list_objects("selected")["objects"][0]["id"] == "cup_1"
    assert tools.select_object("Cup 02")["ok"]
    assert tools.actions[0]["object_id"] == "cup_1"
    evidence = EvidenceTools(data)
    assert evidence.inspect_scene("physical", "선택한 객체")["objects"][0]["id"] == "cup_1"
    assert evidence.inspect_scene("twin", "Cup 01")["objects"][0]["id"] == "cup_0"
    data["selection"]["object_ids"] = []
    assert not SceneTools(data).select_object("selected")["ok"]
