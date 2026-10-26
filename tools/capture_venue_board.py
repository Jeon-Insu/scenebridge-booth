"""Capture the fixed ChArUco board from each venue camera, one camera at a time.

The board does not move during extrinsic calibration, so the cameras need not stream
together: opening them one by one sidesteps the shared-USB bandwidth limit (and the sudo
driver quirk it needs). Every frame is run through the bloom-tolerant detector and the
frames that decode are kept for pose estimation.
"""
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
# Each install day gets its own calibration and capture folders (VENUE_DATE=YYYYMMDD).
DATE = os.environ.get('VENUE_DATE', '20260918')
sys.path.insert(0, str(ROOT))
from smartphone_mocap.calibration import board_from_spec, board_min_corners, detect_charuco  # noqa: E402

SETUP = ROOT / f'smartphone_mocap/calibration/venue_{DATE}/setup.json'
OUT = ROOT / f'outputs/venue_extrinsics_{DATE}/frames'
FRAMES = int(os.environ.get('VENUE_FRAMES', 40))


def capture(camera, dictionary, board):
    capture = cv2.VideoCapture(camera['source'], cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(40):                      # let auto-exposure settle (15 was not enough)
        capture.read()
    counts = []
    folder = OUT / camera['id']
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(FRAMES):
        ok, frame = capture.read()
        if not ok:
            counts.append(-1)
            continue
        _, ids, _, _ = detect_charuco(frame, dictionary, board)
        count = 0 if ids is None else len(ids)
        counts.append(count)
        cv2.imwrite(str(folder / f'{index:02d}_{count:02d}.jpg'), frame)
    capture.release()
    return counts


def capture_from_preview(camera, dictionary, board):
    """Collect frames the running venue preview saves, so the preview can stay open."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preview_frame_stats import read_view
    import time
    folder = OUT / camera['id']
    folder.mkdir(parents=True, exist_ok=True)
    counts, marker = [], 0.0
    for index in range(FRAMES):
        frame = read_view(camera['view'], newer_than=marker)
        marker = time.time()
        if frame is None:
            counts.append(-1)
            continue
        _, ids, _, _ = detect_charuco(frame, dictionary, board)
        count = 0 if ids is None else len(ids)
        counts.append(count)
        cv2.imwrite(str(folder / f'{index:02d}_{count:02d}.jpg'), frame)
    return counts


def main():
    setup = json.loads(SETUP.read_text(encoding='utf-8'))
    dictionary, board = board_from_spec(setup.get('board'))
    minimum = board_min_corners(setup.get('board'))
    cameras = setup['cameras']
    from_preview = '--from-preview' in sys.argv
    summary = {}
    for camera in cameras:
        counts = (capture_from_preview if from_preview else capture)(camera, dictionary, board)
        good = [c for c in counts if c >= minimum]
        summary[camera['id']] = {'view': camera['view'], 'counts': counts, 'usable_frames': len(good)}
        print(f"VIEW{camera['view']} {camera['id']}: 사용가능 {len(good)}/{FRAMES}  "
              f"코너 최대 {max(counts)}  중앙값 {sorted(counts)[len(counts)//2]}", flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
