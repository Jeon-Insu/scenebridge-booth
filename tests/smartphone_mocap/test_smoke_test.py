"""Booth smoke test: the pass/fail rules for recognition and agent checks (tools/booth_smoke_test.py)."""
from smartphone_mocap.smoke_test import (
    FAIL,
    KNOWN,
    PASS,
    WARN,
    clarification_check,
    qa_checks,
    recognition_checks,
    scene_snapshot,
)

NAMES = {"giraffe": "기린", "devil_duck": "오리"}


def frame(*objects, number=0):
    return {"frame_number": number, "objects": [
        {"id": f"{label}/appearance_000/instance_{i:03d}", "label": label, "state": state, "xyz": [x, y, 0.0]}
        for i, (label, state, x, y) in enumerate(objects)]}


def by_name(checks):
    return {c.name: c for c in checks}


def test_a_prop_seen_once_in_nearly_every_frame_passes():
    rows = [frame(("giraffe", "active", 0.5, 0.3), ("devil_duck", "partial", 0.2, 0.2)) for _ in range(10)]
    checks = by_name(recognition_checks(rows, NAMES))
    assert checks["기린"].status == PASS
    assert checks["오리"].status == PASS


def test_a_prop_missing_in_too_many_frames_fails():
    rows = [frame(("giraffe", "active", 0.5, 0.3)) for _ in range(8)] + [frame() for _ in range(2)]
    assert by_name(recognition_checks(rows, {"giraffe": "기린"}))["기린"].status == FAIL


def test_two_copies_of_one_prop_do_not_count_as_seen_once():
    rows = [frame(("giraffe", "active", 0.5, 0.3), ("giraffe", "active", 0.9, 0.3)) for _ in range(10)]
    check = by_name(recognition_checks(rows, {"giraffe": "기린"}))["기린"]
    assert check.status == FAIL
    assert "2개" in check.detail


def test_candidates_are_not_shown_so_they_do_not_count():
    rows = [frame(("giraffe", "candidate", 0.5, 0.3)) for _ in range(10)]
    assert by_name(recognition_checks(rows, {"giraffe": "기린"}))["기린"].status == FAIL


def test_a_jittering_position_fails_even_when_the_count_is_right():
    rows = [frame(("giraffe", "active", 0.5 + (0.05 if i % 2 else 0.0), 0.3)) for i in range(10)]
    check = by_name(recognition_checks(rows, {"giraffe": "기린"}))["기린"]
    assert check.status == FAIL
    assert "흔들림" in check.detail


def test_a_known_undetected_prop_is_reported_as_known_not_failed():
    rows = [frame(("giraffe", "active", 0.5, 0.3)) for _ in range(10)]
    checks = by_name(recognition_checks(rows, {"giraffe": "기린"}, known_missing={"stainless_mug": "머그"}))
    assert checks["머그"].status == KNOWN


def test_a_known_missing_prop_that_now_shows_up_is_good_news():
    rows = [frame(("stainless_mug", "active", 0.5, 0.3)) for _ in range(10)]
    checks = by_name(recognition_checks(rows, {}, known_missing={"stainless_mug": "머그"}))
    assert checks["머그"].status == PASS


def test_the_snapshot_keeps_only_shown_objects_in_table_axes():
    snap = scene_snapshot(frame(("giraffe", "active", 0.5, 0.3), ("devil_duck", "candidate", 0.2, 0.2), number=7))
    assert [o["label"] for o in snap["tracking"]["objects"]] == ["giraffe"]
    assert snap["frame_number"] == 7
    assert snap["interaction_frame"]["table_axes"]["east"] == [1.0, 0.0, 0.0]


def test_a_scene_answer_naming_every_prop_in_korean_passes():
    response = {"ok": True, "task_state": "answered", "reply": "테이블에 기린 1개와 오리 1개가 있습니다."}
    checks = qa_checks(response, NAMES)
    assert [c.status for c in checks] == [PASS, PASS]


def test_a_scene_answer_using_internal_ids_is_a_warning_not_a_pass():
    response = {"ok": True, "task_state": "answered", "reply": "giraffe 1개, devil_duck 1개"}
    found, korean = qa_checks(response, NAMES)
    assert found.status == PASS
    assert korean.status == WARN


def test_a_scene_answer_missing_a_prop_fails():
    response = {"ok": True, "task_state": "answered", "reply": "기린이 있습니다."}
    found, _ = qa_checks(response, NAMES)
    assert found.status == FAIL
    assert "오리" in found.detail


def test_a_failed_agent_request_fails_the_scene_check():
    found, korean = qa_checks({"ok": False, "error": "timeout"}, NAMES)
    assert found.status == FAIL
    assert korean.status == FAIL


def test_a_force_without_duration_must_be_asked_back():
    response = {"ok": True, "task_state": "needs_clarification", "actions": [],
                "pending_experiment": {"missing": "force_duration_s"}, "reply": "몇 초 동안 가할까요?"}
    assert clarification_check(response).status == PASS


def test_running_an_experiment_without_the_duration_fails():
    response = {"ok": True, "task_state": "answered", "actions": [{"type": "physics_preview"}], "reply": "준비했습니다."}
    assert clarification_check(response).status == FAIL
