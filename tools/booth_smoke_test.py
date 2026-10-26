"""부스 자동 점검: 카메라 없이 녹화로 인식이, 켜져 있는 에이전트로 대화가 망가지지 않았는지 확인합니다.

  .venv-mocap/bin/python tools/booth_smoke_test.py                                   에이전트만
  .venv-mocap/bin/python tools/booth_smoke_test.py --session recordings/<녹화> --expect giraffe zebra ...
                                                                    녹화 인식 + 에이전트

인식(--session을 줄 때만): 소품이 가만히 놓인 녹화의 앞부분(기본 30초)을 지금 설정으로 다시 인식해, --expect로 준
소품마다 "정확히 1개"가 90% 이상이고 위치가 2cm 넘게 흔들리지 않는지 봅니다. 정확도(실제 위치와의 차이) 측정이
아닙니다. 녹화 만드는 법은 docs/HANDOFF.md를 보세요.
에이전트: 에이전트 서비스(7080)가 켜져 있어야 합니다(booth_start.sh 또는 run_tabletop_demo.sh). 녹화가 없으면
smartphone_mocap/data/smoke_scene.json(실험실 소품 7종 장면)으로 장면 질문 1개와 "시간 없는 힘" 실험 요청 1개를 보냅니다. 실험 계획 생성·실제 물리 접촉은 Isaac이 필요해 여기서 보지 않습니다 —
docs/HANDOFF.md의 현장 5분 점검으로 확인합니다.

실패가 없으면 0, 하나라도 있으면 1로 끝납니다. 결과는 outputs/smoke_test/<시각>/report.json에도 남습니다.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartphone_mocap import booth_props  # noqa: E402
from smartphone_mocap.smoke_test import (  # noqa: E402
    FAIL, Check, clarification_check, qa_checks, recognition_checks, scene_snapshot)

# One tracking frame from the 2026-10-26 lab test (7 props standing still), for the agent check without a recording.
SCENE = ROOT / "smartphone_mocap/data/smoke_scene.json"


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def run_recognition(args, names: dict[str, str], out: Path) -> tuple[list[Check], dict]:
    from smartphone_mocap.replay import run_recorded_replay
    print(f"인식: {args.session.name} 앞 {args.samples}장 다시 인식 중 (1~2분)...", flush=True)
    run_recorded_replay(args.session, config_path=args.config, output_dir=out / "replay", max_samples=args.samples)
    rows = [json.loads(line) for line in (out / "replay/tracking.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = {label: names.get(label, label) for label in args.expect}
    known = {label: names.get(label, label) for label in args.known_missing}
    return recognition_checks(rows, expected, known_missing=known), rows[-1]


def run_agent(args, names: dict[str, str], row: dict) -> list[Check]:
    base = args.agent_url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return [Check("에이전트 서비스", FAIL, f"{base} 에 연결 안 됨 ({exc}). 데모를 켠 뒤 다시 실행하세요")]
    snapshot = scene_snapshot(row)
    on_table = {o["label"] for o in snapshot["tracking"]["objects"]}
    shown = {label: names.get(label, label) for label in args.expect if label in on_table}
    session = f"smoke-{time.time_ns()}"
    print("에이전트: 장면 질문 (10초 안팎)...", flush=True)
    answer = post(base + "/v1/agent/respond", {"prompt": "테이블 위에 뭐가 있어?", "snapshot": snapshot,
                                              "session_id": session, "conversation_id": session}, args.timeout)
    checks = list(qa_checks(answer, shown))
    first, second = list(shown.values())[:2]
    print("에이전트: 시간 없는 힘 요청...", flush=True)
    prompt = f"{first}를 {second} 쪽으로 0.1N의 힘으로 미는 시뮬레이션 세팅해줘"
    asked = post(base + "/v1/agent/respond", {"prompt": prompt, "snapshot": snapshot,
                                             "session_id": session + "-force", "conversation_id": session + "-force"},
                 args.timeout)
    checks.append(clarification_check(asked))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path, help="인식 점검에 쓸 녹화 폴더 (없으면 인식 점검을 건너뜀)")
    parser.add_argument("--config", type=Path, default=ROOT / "smartphone_mocap/config/booth.json")
    parser.add_argument("--samples", type=int, default=150, help="앞에서부터 다시 인식할 장 수 (5장 = 1초)")
    parser.add_argument("--expect", nargs="+", default=[], help="녹화 구간에 놓인 소품 prop_id")
    parser.add_argument("--known-missing", nargs="*", default=[], help="놓여 있지만 아직 못 찾는 소품 (예: stainless_mug)")
    parser.add_argument("--agent-url", default="http://127.0.0.1:7080")
    parser.add_argument("--timeout", type=float, default=180.0, help="에이전트 질문 하나의 최대 대기(초)")
    parser.add_argument("--skip-recognition", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    if args.session is not None and not (args.session / "session.json").is_file():
        print(f"녹화가 없습니다: {args.session}", file=sys.stderr)
        return 1
    if args.session is not None and not args.expect:
        print("--session 을 줄 때는 --expect 로 녹화에 놓인 소품 prop_id를 적으세요", file=sys.stderr)
        return 1
    names = {p.prop_id: p.name_ko for p in booth_props.load()}
    out = ROOT / "outputs/smoke_test" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    sections: dict[str, list[Check]] = {}
    row = None
    if args.session is not None and not args.skip_recognition:
        sections["인식"], row = run_recognition(args, names, out)
    elif not args.skip_recognition:
        print("인식 점검은 건너뜀: 녹화를 --session, 놓인 소품을 --expect 로 주면 함께 봅니다.")
    if not args.skip_agent:
        if row is None:
            row = json.loads(SCENE.read_text(encoding="utf-8"))
            args.expect = args.expect or sorted({o["label"] for o in row["objects"]})
        sections["에이전트"] = run_agent(args, names, row)

    print()
    for title, checks in sections.items():
        print(f"[{title}]")
        for check in checks:
            print(f"  {check.status:6s} {check.name}: {check.detail}")
    print("\n이 점검은 '망가지지 않았는지'만 봅니다. 위치 정확도, 실험 계획 생성, 실제 물리 접촉은")
    print("docs/HANDOFF.md의 '현장 5분 점검'으로 확인하세요.")
    (out / "report.json").write_text(json.dumps({k: [asdict(c) for c in v] for k, v in sections.items()},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과: {out / 'report.json'}")
    return 1 if any(c.status == FAIL for checks in sections.values() for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
