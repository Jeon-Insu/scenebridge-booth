"""학습한 소품(assets/props)의 트윈 크기를 높이로 맞춥니다.

  python3 tools/set_prop_height.py elephant 0.12      코끼리를 높이 12cm로
  python3 tools/set_prop_height.py --list             소품과 지금 높이 보기

model_placed.usda는 바닥이 Z=0에 오도록 크기와 위치를 정해 둔 파일이라, 두 값에 같은 비율을
곱하면 새 높이가 됩니다. 데모를 다시 켜야 트윈에 반영됩니다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROPS = Path(__file__).resolve().parents[1] / 'assets' / 'props'


def _rescale(text, name, ratio):
    pattern = re.compile(r'(' + re.escape(name) + r' = \()([^)]*)(\))')
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f'{name} 값을 찾지 못했습니다')
    values = ', '.join(repr(float(v) * ratio) for v in match.group(2).split(','))
    return text[:match.start()] + match.group(1) + values + match.group(3) + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('asset_id', nargs='?')
    parser.add_argument('height_m', nargs='?', type=float)
    parser.add_argument('--props', type=Path, default=PROPS)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    catalog_path = args.props / 'catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if args.list or args.asset_id is None:
        for entry in catalog['assets']:
            print(f"  {entry['asset_id']:14s} {entry['display_name_ko']:10s} 높이 {entry['height_m'] * 100:5.1f}cm ({entry['height_source']})")
        return 0
    entry = next((e for e in catalog['assets'] if e['asset_id'] == args.asset_id), None)
    if entry is None:
        print(f"없는 소품입니다: {args.asset_id} (--list로 목록 확인)")
        return 1
    if args.height_m is None or not 0.005 <= args.height_m <= 1.0:
        print('높이는 0.005~1.0 m 사이로 주세요 (예: 0.12 = 12cm)')
        return 1

    placed = args.props / entry['usd_uri']
    ratio = args.height_m / float(entry['height_m'])
    text = placed.read_text(encoding='utf-8')
    for name in ('xformOp:scale', 'xformOp:translate'):
        text = _rescale(text, name, ratio)
    placed.write_text(text, encoding='utf-8')
    entry['height_m'] = args.height_m
    entry['height_source'] = 'estimate'
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"{entry['display_name_ko']}: 높이 {args.height_m * 100:.1f}cm로 바꿨습니다. 데모를 다시 켜야 반영됩니다.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
