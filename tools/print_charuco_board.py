"""보정 보드를 A4 가로 한 장 PDF로 만듭니다. 보드 규격은 설치일 setup.json의 "board"에서 읽습니다.

  VENUE_DATE=20261025 python tools/print_charuco_board.py
  python tools/print_charuco_board.py --setup 경로/setup.json --out 보드.pdf

인쇄할 때는 "실제 크기(100%)"로 하고, 인쇄 뒤 칸 한 변을 자로 재어 확인합니다.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartphone_mocap.calibration import board_from_spec  # noqa: E402

PX_PER_MM = 20                                  # 508 dpi: every square is a whole number of pixels
PAGE_MM = (297, 210)                            # A4 landscape
TOP_MM = 12
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def render(spec):
    _, board = board_from_spec(spec)
    squares_x, squares_y = board.getChessboardSize()
    square_mm = board.getSquareLength() * 1000
    board_px = (round(squares_x * square_mm * PX_PER_MM), round(squares_y * square_mm * PX_PER_MM))
    page_px = (PAGE_MM[0] * PX_PER_MM, PAGE_MM[1] * PX_PER_MM)
    if board_px[0] > page_px[0] - 10 * PX_PER_MM or board_px[1] > page_px[1] - 30 * PX_PER_MM:
        raise SystemExit(f'보드({board_px[0] / PX_PER_MM:.0f}×{board_px[1] / PX_PER_MM:.0f}mm)가 A4에 들어가지 않습니다')
    page = Image.new('L', page_px, 255)
    left = (page_px[0] - board_px[0]) // 2
    page.paste(Image.fromarray(np.asarray(board.generateImage(board_px, marginSize=0))), (left, TOP_MM * PX_PER_MM))
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(FONT, 4 * PX_PER_MM) if Path(FONT).exists() else ImageFont.load_default()
    note = (f'보정 보드 {squares_x}×{squares_y}칸 · 칸 한 변 {square_mm:.0f}mm · {spec.get("dictionary", "DICT_5X5_100")}'
            f'   —   "실제 크기 100%"로 인쇄, 칸 한 변이 {square_mm:.0f}mm인지 자로 확인')
    draw.text((left, TOP_MM * PX_PER_MM + board_px[1] + 5 * PX_PER_MM), note, fill=0, font=font)
    return page.point(lambda v: 255 if v > 127 else 0).convert('1')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    date = os.environ.get('VENUE_DATE')
    parser.add_argument('--setup', type=Path,
                        default=ROOT / f'smartphone_mocap/calibration/venue_{date}/setup.json' if date else None)
    parser.add_argument('--out', type=Path,
                        default=ROOT / f'outputs/venue_extrinsics_{date}/charuco_board_A4.pdf' if date else None)
    args = parser.parse_args()
    if args.setup is None or args.out is None:
        parser.error('VENUE_DATE를 주거나 --setup과 --out을 지정하세요')
    spec = json.loads(args.setup.read_text(encoding='utf-8')).get('board')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(spec or {}).save(args.out, 'PDF', resolution=25.4 * PX_PER_MM)
    print(args.out)


if __name__ == '__main__':
    main()
