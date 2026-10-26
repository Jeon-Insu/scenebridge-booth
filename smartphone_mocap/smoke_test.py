"""Pass/fail rules for the booth smoke test (tools/booth_smoke_test.py).

The smoke test answers "is anything broken?", not "how accurate is it?". A steady position is a sanity check on a
recording where nothing moves; it says nothing about how close that position is to the real one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS, FAIL, WARN, KNOWN = "통과", "실패", "주의", "알려진 문제"
SHOWN_STATES = ("active", "partial")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _shown(row: dict) -> list[dict]:
    return [o for o in row.get("objects", []) if o.get("state") in SHOWN_STATES]


def recognition_checks(rows: list[dict], expected: dict[str, str], *, known_missing: dict[str, str] | None = None,
                       min_fraction: float = 0.9, max_spread_m: float = 0.02) -> list[Check]:
    """One check per prop: shown exactly once in at least `min_fraction` of frames, and not jittering."""
    total = max(len(rows), 1)
    checks = []
    for label, name in {**expected, **(known_missing or {})}.items():
        counts = [sum(1 for o in _shown(r) if o.get("label") == label) for r in rows]
        once = [r for r, n in zip(rows, counts) if n == 1]
        fraction = len(once) / total
        doubled = sum(1 for n in counts if n > 1)
        if label not in expected:
            status = PASS if fraction >= min_fraction else KNOWN
            detail = f"정확히 1개 {fraction:.0%}" + ("" if status == PASS else " — 지금은 검출되지 않는 소품")
            checks.append(Check(name, status, detail))
            continue
        detail = f"정확히 1개 {fraction:.0%} (기준 {min_fraction:.0%})"
        if doubled:
            detail += f", 2개 이상 {doubled}프레임"
        status = PASS if fraction >= min_fraction else FAIL
        if once:
            xy = np.array([next(o for o in _shown(r) if o.get("label") == label)["xyz"][:2] for r in once], float)
            spread = float(np.percentile(np.linalg.norm(xy - np.median(xy, axis=0), axis=1), 90))
            detail += f", 위치 흔들림 {spread * 100:.1f}cm (기준 {max_spread_m * 100:.0f}cm)"
            if spread > max_spread_m:
                status = FAIL
        checks.append(Check(name, status, detail))
    return checks


def scene_snapshot(row: dict) -> dict:
    """A minimal agent snapshot from one tracking frame (no Isaac twin state, so no physics planning)."""
    return {
        "schema": "gist.tabletop_agent.snapshot.v1",
        "snapshot_id": f"smoke-{row.get('frame_number', 0)}",
        "frame_number": row.get("frame_number", 0),
        "interaction_frame": {
            "table_axes": {"east": [1.0, 0.0, 0.0], "north": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]},
            "workspace_bounds_xy": [0.0, 1.25, 0.0, 0.75],
        },
        "tracking": {"objects": _shown(row)},
    }


def qa_checks(response: dict, expected: dict[str, str]) -> tuple[Check, Check]:
    """The scene answer names every prop on the table; and it uses the Korean names from the booth list."""
    if not response.get("ok"):
        error = str(response.get("error", "응답 없음"))
        return (Check("장면 질문", FAIL, f"에이전트 오류: {error}"), Check("한국어 이름", FAIL, "답변을 받지 못함"))
    reply = str(response.get("reply", ""))
    missing = [name for label, name in expected.items() if name not in reply and label not in reply]
    found = Check("장면 질문", FAIL if missing else PASS,
                  f"빠진 소품: {', '.join(missing)}" if missing else f"{len(expected)}개 모두 언급")
    by_id = [name for label, name in expected.items() if name not in reply and label in reply]
    korean = Check("한국어 이름", WARN if by_id or missing else PASS,
                   f"내부 ID로 답함: {', '.join(by_id)}" if by_id else
                   ("일부 소품을 언급하지 않음" if missing else "목록의 한국어 이름 사용"))
    return found, korean


def clarification_check(response: dict) -> Check:
    """A force with no duration must be asked back, with no experiment queued."""
    asked = (response.get("ok") and response.get("task_state") == "needs_clarification"
             and not response.get("actions")
             and (response.get("pending_experiment") or {}).get("missing") == "force_duration_s")
    if asked:
        return Check("힘 지속 시간 되묻기", PASS, str(response.get("reply", "")))
    return Check("힘 지속 시간 되묻기", FAIL,
                 f"상태 {response.get('task_state')}, action {len(response.get('actions') or [])}개: "
                 f"{str(response.get('reply', response.get('error', '')))[:120]}")
