"""The agent understands the Korean names written in the booth list (config/booth_props.json)."""
import json

import pytest

from smartphone_mocap import booth_props
from smartphone_mocap.agent_evidence import EvidenceTools
from smartphone_mocap.experiment_planner import ExperimentPlanner
from smartphone_mocap.scene_agent import SceneTools, _query_hint
from test_scene_agent import _snapshot


@pytest.fixture(autouse=True)
def booth_list(tmp_path, monkeypatch):
    root = tmp_path / "props"
    (root / "gist_mascot").mkdir(parents=True)
    (root / "gist_mascot/model_placed.usda").write_text("#usda 1.0\n")
    (root / "catalog.json").write_text(json.dumps({"schema_version": 1, "assets": [
        {"asset_id": "gist_mascot", "display_name_ko": "GIST 마스코트 인형", "usd_uri": "gist_mascot/model_placed.usda",
         "detector_class": "teddy bear", "height_m": 0.2, "height_source": "estimate"}]}, ensure_ascii=False))
    path = tmp_path / "booth_props.json"
    path.write_text(json.dumps({"props": [
        {"prop_id": "gist_mascot", "name_ko": "인형", "aliases_ko": ["마스코트"], "count": 1}]},
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(booth_props, "DEFAULT_PATH", path)
    monkeypatch.setattr(booth_props, "CATALOG_DIR", root)


def snapshot_with_doll():
    data = _snapshot()
    data["tracking"]["objects"].append({"id": "gist_mascot/appearance_000/instance_000", "label": "gist_mascot",
                                        "state": "active", "xyz": [0.9, 0.5, 0.0],
                                        "asset_uri": "model_placed.usda", "source_cameras": ["a", "b"]})
    return data


def test_scene_tools_find_the_doll_by_its_korean_names():
    tools = SceneTools(snapshot_with_doll())
    for word in ("인형", "마스코트"):
        ids = [o["id"] for o in tools.list_objects(word)["objects"]]
        assert ids == ["gist_mascot/appearance_000/instance_000"], word


def test_the_query_hint_picks_up_a_listed_korean_name():
    assert "인형" in _query_hint("인형을 테이블 위에서 떨어뜨려 줘")


def test_evidence_finds_the_doll_by_its_korean_name():
    objects = EvidenceTools(snapshot_with_doll()).inspect_scene("physical", "인형")["objects"]
    assert [o["id"] for o in objects] == ["gist_mascot/appearance_000/instance_000"]


def test_the_planner_counts_the_doll_as_mentioned():
    tools = SceneTools(snapshot_with_doll())
    planner = ExperimentPlanner(tools, "마스코트를 서쪽으로 10cm 밀어줘")
    assert planner._mentioned("gist_mascot/appearance_000/instance_000")
    assert not planner._mentioned("potted plant/green/instance_000")


def test_the_scene_summary_counts_objects_by_their_korean_names():
    summary = SceneTools(snapshot_with_doll()).get_scene_summary()
    # Booth names win; builtin COCO objects fall back to their builtin Korean word.
    assert summary["counts_by_name_ko"] == {"인형": 1, "컵": 2, "화분": 1}


def test_listed_objects_carry_their_korean_name():
    objects = {o["id"]: o for o in SceneTools(snapshot_with_doll()).list_objects()["objects"]}
    assert objects["gist_mascot/appearance_000/instance_000"]["name_ko"] == "인형"
    assert objects["potted plant/green/instance_000"]["name_ko"] == "화분"


def test_an_object_with_no_korean_name_keeps_its_label():
    data = snapshot_with_doll()
    data["tracking"]["objects"].append({"id": "robot/a/000", "label": "robot", "state": "active", "xyz": [1.0, 0.6, 0.0]})
    assert SceneTools(data).get_scene_summary()["counts_by_name_ko"]["robot"] == 1


def test_the_model_is_told_to_name_objects_in_korean(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    captured = []
    monkeypatch.setattr(agent, "_resolve_model", lambda *args: "test-model")
    monkeypatch.setattr(agent, "_post_json",
                        lambda url, payload, timeout: captured.append(payload) or {"choices": [{"message": {"content": "답"}}]})
    agent.sglang_response("테이블 위에 뭐가 있어?", snapshot_with_doll(), base_url="http://unused/v1", model="auto",
                          timeout_seconds=1)
    system = captured[0]["messages"][0]["content"]
    assert "name_ko" in system
    assert '"인형": 1' in captured[0]["messages"][1]["content"]


def test_the_fallback_answer_counts_objects_by_korean_name():
    from smartphone_mocap.scene_agent import deterministic_response
    reply = deterministic_response("테이블 위에 뭐가 있어?", snapshot_with_doll())["reply"]
    assert "인형: 1" in reply
    assert "gist_mascot" not in reply
