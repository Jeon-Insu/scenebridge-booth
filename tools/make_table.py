"""부스 테이블 크기로 트윈 테이블 USD를 만듭니다 (assets/tabletop_demo/table_<가로>x<세로>x<높이>.usda).

  python3 tools/make_table.py 1.8 0.75          가로 1.8m, 세로 0.75m, 높이 0.75m
  python3 tools/make_table.py 1.8 0.75 --height 0.72

트윈은 booth.json의 rig.workspace_size_m(가로·세로)에 맞는 파일을 찾습니다. 크기를 바꾸면 그 값도 함께 바꾸세요.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartphone_mocap.table_asset import table_asset_name, table_usda  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("length", type=float, help="가로(m)")
    p.add_argument("width", type=float, help="세로(m)")
    p.add_argument("--height", type=float, default=0.75)
    a = p.parse_args()
    out = ROOT / "assets/tabletop_demo" / table_asset_name(a.length, a.width, a.height)
    out.write_text(table_usda(a.length, a.width, a.height), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
