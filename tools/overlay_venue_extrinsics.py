"""Draw the calibrated paper outline, table outline and table centre over each view.

A visual check anyone at the booth can read: the green outline should sit on the board
paper in every view, and the red dot on the board's middle.
"""
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Each install day gets its own calibration and capture folders (VENUE_DATE=YYYYMMDD).
DATE = os.environ.get('VENUE_DATE', '20260918')
CAL = ROOT / f'smartphone_mocap/calibration/venue_{DATE}'
FRAMES = ROOT / f'outputs/venue_extrinsics_{DATE}/frames'
OUT = ROOT / f'outputs/venue_extrinsics_{DATE}/overlay_final.jpg'


def main():
    report = json.loads((CAL / 'extrinsics_report.json').read_text())
    face = report['board_face_height_m_assumed']
    length, width = report['table_size_m']
    # The paper corners were located before the axis alignment; move them with it.
    turn_deg = report.get('table_axis_alignment_deg', 0.0)
    c, s = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    centre = np.array([length / 2, width / 2])
    paper = np.array(report['paper_outline_table_m'])
    paper[:, :2] = (paper[:, :2] - centre) @ np.array([[c, s], [-s, c]]) + centre
    table = np.array([[0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0]], float)
    tiles = []
    setup = json.loads((CAL / 'setup.json').read_text())
    order = sorted((c['view'], c['id']) for c in setup['cameras'])
    for view, name in order:
        document = json.loads((CAL / f'{name}.json').read_text())
        K, dist = np.array(document['camera_matrix']), np.array(document['distortion'])
        W = np.array(document['world_to_camera'])
        rvec, tvec = cv2.Rodrigues(W[:3, :3])[0], W[:3, 3]
        image = cv2.imread(sorted(glob.glob(str(FRAMES / name / '*.jpg')))[-1])
        for points, colour in ((paper, (0, 255, 0)), (table, (255, 200, 0))):
            pixels = cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [pixels], True, colour, 3)
        dot = cv2.projectPoints(np.array([[*centre, face]]), rvec, tvec, K, dist)[0].ravel().astype(int)
        cv2.circle(image, tuple(dot), 9, (0, 0, 255), -1)
        cv2.putText(image, f"VIEW{view} {name} {document['calibration_status']}", (10, 40), 0, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(image, (800, 450)))
    cv2.imwrite(str(OUT), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print(OUT)


if __name__ == '__main__':
    main()
