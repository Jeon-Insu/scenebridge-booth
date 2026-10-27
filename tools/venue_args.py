"""장소 프리셋의 Isaac 시작 인자를 한 줄에 하나씩 출력합니다 (tools/run_isaac_tabletop_perception.sh가 읽음).

  python3 tools/venue_args.py dongshin_ai_classroom
  python3 tools/venue_args.py --config smartphone_mocap/config/booth.json   설정의 "venue"를 씀(추적기와 같은 출처)
  --venues 다른/venues.json                                               다른 프리셋 파일
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartphone_mocap.venues import VenueError, isaac_args, load_venue  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    venues_path = None
    if "--venues" in args:  # another presets file (tests, or a second site)
        at = args.index("--venues")
        venues_path, args = args[at + 1], args[:at] + args[at + 2:]
    table_size = None
    if len(args) == 2 and args[0] == "--config":
        config = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        name = config.get("venue", "dongshin_ai_classroom")
        table_size = (config.get("rig", {}).get("workspace_size_m") or [None, None])[:2]
        table_size = table_size if None not in table_size else None
    elif len(args) == 1:
        name = args[0]
    else:
        print(__doc__)
        return 2
    try:
        print("\n".join(isaac_args(load_venue(name, venues_path), table_size)))
    except VenueError as exc:
        print(f"[문제] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
