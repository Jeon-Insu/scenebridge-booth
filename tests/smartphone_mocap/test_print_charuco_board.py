"""tools/print_charuco_board.py writes an A4 PDF whose board prints at the setup's real size."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from smartphone_mocap.calibration import board_from_spec, detect_charuco

ROOT = Path(__file__).resolve().parents[2]
SPEC = {"squares": [5, 3], "square_length_m": 0.056, "marker_length_m": 0.042,
        "dictionary": "DICT_4X4_50", "min_corners": 4}


def test_the_pdf_is_an_a4_page_with_the_board_at_real_size(tmp_path):
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps({"board": SPEC}))
    pdf = tmp_path / "board.pdf"
    subprocess.run([sys.executable, str(ROOT / "tools/print_charuco_board.py"),
                    "--setup", str(setup), "--out", str(pdf)], check=True, cwd=ROOT)

    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    width_pt, height_pt = (float(v) for v in info.split("Page size:")[1].split("pts")[0].split(" x "))
    assert width_pt == pytest.approx(297 / 25.4 * 72, abs=1.0)    # A4 landscape
    assert height_pt == pytest.approx(210 / 25.4 * 72, abs=1.0)

    dpi = 100
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(tmp_path / "page")], check=True)
    page = cv2.imread(str(tmp_path / "page.png"))
    dictionary, board = board_from_spec(SPEC)
    corners, ids, _, _ = detect_charuco(page, dictionary, board)
    assert ids is not None and len(ids) == 8
    # Neighbouring inner corners are one square apart: 56mm at 100dpi.
    by_id = {int(i): c.ravel() for i, c in zip(ids.ravel(), corners)}
    assert np.linalg.norm(by_id[1] - by_id[0]) == pytest.approx(56 / 25.4 * dpi, rel=0.01)
