"""부스 소품 목록(config/booth_props.json)이 바로 쓸 수 있는지 확인합니다 (v1.2: 학습한 소품만).

  python3 tools/booth_check.py           목록 확인
  python3 tools/booth_check.py --props   목록에 쓸 수 있는 학습 소품(prop_id) 보기

문제가 없으면 "준비 완료"와 함께 0으로, 문제가 있으면 고칠 내용을 한 줄씩 보여 주고 1로 끝납니다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartphone_mocap import booth_props  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, default=booth_props.DEFAULT_PATH, help="확인할 목록 파일")
    parser.add_argument("--props", action="store_true", help="쓸 수 있는 학습 소품 보기")
    args = parser.parse_args()

    if args.props:
        print("목록에 쓸 수 있는 학습 소품 — prop_id에 왼쪽 이름을 그대로 씁니다:")
        for entry in booth_props.catalog():
            print(f"  {entry['asset_id']:14s} {entry['display_name_ko']}  높이 {entry['height_m'] * 100:.0f}cm")
        return 0

    try:
        props = booth_props.load(args.list)
    except booth_props.BoothPropsError as exc:
        print(f"[문제] {exc}")
        return 1
    problems = booth_props.check(props)
    if not props:
        problems.append("목록에 소품이 하나도 없습니다")
    for line in problems:
        print(f"[문제] {line}")
    if problems:
        print(f"\n{len(problems)}가지를 고친 뒤 다시 확인하세요: {args.list}")
        return 1
    print(f"준비 완료: 소품 {len(props)}개 — {args.list}")
    for prop in props:
        words = ", ".join(prop.aliases_ko)
        print(f"  {prop.name_ko} ({prop.prop_id}, {prop.count}개, 높이 {prop.height_m * 100:.0f}cm)"
              + (f" · 다른 이름: {words}" if words else ""))
    print("데모를 다시 켜야 반영됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
