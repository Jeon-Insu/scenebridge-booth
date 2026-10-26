"""Extrinsics when every camera reads the board (2026-10-25 onward).

  VENUE_DATE=20261025 python tools/capture_venue_board.py      # frames per camera
  VENUE_DATE=20261025 python tools/solve_board_extrinsics.py   # this: cam_X.json + report
  VENUE_DATE=20261025 python tools/overlay_venue_extrinsics.py # check picture

Each camera's pose is the median over its board frames. The table is found on the board plane
(smartphone_mocap.board_table), so the board need not be at the table centre.
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = os.environ.get('VENUE_DATE', '20261025')
sys.path.insert(0, str(ROOT))
from smartphone_mocap.board_table import fit_table, median_pose, table_from_board  # noqa: E402
from smartphone_mocap.calibration import board_from_spec, estimate_board_extrinsics  # noqa: E402
from smartphone_mocap.types import CameraCalibration  # noqa: E402

CAL = ROOT / f'smartphone_mocap/calibration/venue_{DATE}'
FRAMES = ROOT / f'outputs/venue_extrinsics_{DATE}/frames'
OUT_IMAGES = ROOT / f'outputs/venue_extrinsics_{DATE}'
INTRINSICS = ROOT / 'smartphone_mocap/calibration/creative_4cam/cam_A_intrinsics.json'
CELL = 0.005


def centre_of(world_to_camera):
    return -world_to_camera[:3, :3].T @ world_to_camera[:3, 3]


def plane_votes(images, poses, K, dist, xs, ys):
    """Per board-plane cell: how many views see it, and the share of those that see it white."""
    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    seen = np.zeros(X.shape, int)
    white = np.zeros(X.shape)
    for image, pose in zip(images, poses):
        rvec, tvec = cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3]
        pixels = cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1, 2)
        depth = points @ pose[2, :3] + tvec[2]
        h, w = image.shape[:2]
        ok = (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w - 1) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h - 1)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        values = np.zeros(len(points))
        values[ok] = grey[pixels[ok, 1].astype(int), pixels[ok, 0].astype(int)]
        seen += ok.reshape(X.shape)
        white += ((values > threshold) & ok).reshape(X.shape)
    return np.where(seen > 0, white / np.maximum(seen, 1), 0.0), seen


def main():
    setup = json.loads((CAL / 'setup.json').read_text(encoding='utf-8'))
    spec = setup['board']
    size = (setup['table_length_m'], setup['table_width_m'])
    intrinsics = json.loads(INTRINSICS.read_text())
    K = np.array(intrinsics['camera_matrix'])
    dist = np.array(intrinsics['distortion']).reshape(-1)

    poses, errors, counts, images = {}, {}, {}, {}
    for camera in setup['cameras']:
        name = camera['id']
        calibration = CameraCalibration(name, (1280, 720), K, dist, np.eye(4))
        found = []
        for path in sorted(glob.glob(str(FRAMES / name / '*.jpg'))):
            image = cv2.imread(path)
            try:
                found.append((*estimate_board_extrinsics(image, calibration, spec), image))
            except ValueError:
                continue
        if len(found) < 5:
            raise SystemExit(f'{name}: 보드가 보이는 사진이 {len(found)}장뿐입니다 (5장 이상 필요)')
        poses[name] = median_pose([f[0] for f in found])
        errors[name] = float(np.median([f[1] for f in found]))
        counts[name] = len(found)
        images[name] = found[-1][2]
        spread = np.std([centre_of(f[0]) for f in found], axis=0).max()
        print(f'{name}: 사진 {len(found)}장, 재투영 {errors[name]:.2f}px, 위치 흔들림 {spread * 1000:.1f}mm')

    names = list(poses)
    xs = np.arange(-2.0, 2.0, CELL)
    ys = np.arange(-1.5, 1.5, CELL)
    fraction, seen = plane_votes([images[n] for n in names], [poses[n] for n in names], K, dist, xs, ys)
    centre, yaw = fit_table(fraction, seen, xs, ys, size)
    camera_xy = {n: centre_of(poses[n])[:2] for n in names}
    face = float(spec.get('face_height_m') or 0.0)
    table_from = table_from_board(centre, yaw, size, camera_xy, face)
    print(f'테이블 중심(보드 기준) ({centre[0]:+.3f}, {centre[1]:+.3f}) m, 긴 축 {yaw:+.1f}°')

    squares_x, squares_y = board_from_spec(spec)[1].getChessboardSize()
    side = board_from_spec(spec)[1].getSquareLength()
    outline = np.array([[0, 0, 0, 1], [squares_x * side, 0, 0, 1],
                        [squares_x * side, squares_y * side, 0, 1], [0, squares_y * side, 0, 1]])
    report = {
        'date': setup['date'], 'method': 'every camera reads the board; table fitted on the board plane',
        'table_size_m': list(size), 'board_face_height_m_assumed': face,
        'paper_outline_table_m': (outline @ table_from.T)[:, :3].round(4).tolist(),
        'table_axis_alignment_deg': 0.0, 'table_from_board': table_from.tolist(),
        'table_fit': {'centre_board_m': list(centre), 'yaw_deg': yaw}, 'intrinsics': str(INTRINSICS),
        'cameras': {},
    }
    for name in names:
        world_to_camera = poses[name] @ np.linalg.inv(table_from)
        document = {
            'schema_version': 1, 'camera_id': name, 'image_size': [1280, 720],
            'camera_matrix': K.tolist(), 'distortion': dist.tolist(),
            'world_to_camera': world_to_camera.tolist(), 'time_offset_ns': 0,
            'calibration_status': 'venue_measured_board_pose',
            'provenance': {'date': setup['date'], 'board_frames': counts[name],
                           'median_reprojection_px': errors[name], 'table_size_m': list(size),
                           'intrinsics': str(INTRINSICS), 'method': report['method']},
        }
        (CAL / f'{name}.json').write_text(json.dumps(document, indent=2))
        position = centre_of(world_to_camera)
        forward = world_to_camera[2, :3]
        report['cameras'][name] = {
            'position_m': position.round(3).tolist(),
            'pitch_down_deg': round(float(np.degrees(np.arcsin(-forward[2]))), 1),
            'board_frames': counts[name], 'median_reprojection_px': round(errors[name], 3),
        }
        print(f'{name}: 테이블 기준 위치 {position.round(3).tolist()} m, 내려다보는 각 {report["cameras"][name]["pitch_down_deg"]}°')
    (CAL / 'extrinsics_report.json').write_text(json.dumps(report, indent=2))

    # Top view of the plane votes with the fitted table, for a quick look.
    picture = cv2.cvtColor((fraction * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    picture[seen < 2] = (60, 0, 0)
    corners = np.array([[0, 0, 0, 1], [size[0], 0, 0, 1], [size[0], size[1], 0, 1], [0, size[1], 0, 1]])
    in_board = (corners @ np.linalg.inv(table_from).T)[:, :2]
    pixels = np.column_stack([(in_board[:, 0] - xs[0]) / CELL, (in_board[:, 1] - ys[0]) / CELL]).astype(np.int32)
    cv2.polylines(picture, [pixels], True, (0, 0, 255), 3)
    for name in names:
        x, y = camera_xy[name]
        point = (int((x - xs[0]) / CELL), int((y - ys[0]) / CELL))
        cv2.circle(picture, point, 10, (0, 255, 0), -1)
    picture = cv2.flip(picture, 0)
    for name in names:
        x, y = camera_xy[name]
        cv2.putText(picture, name[-1], (int((x - xs[0]) / CELL) + 14, picture.shape[0] - int((y - ys[0]) / CELL)),
                    0, 1.2, (0, 255, 0), 3)
    cv2.imwrite(str(OUT_IMAGES / 'table_fit_top_view.jpg'), picture)
    print(OUT_IMAGES / 'table_fit_top_view.jpg')


if __name__ == '__main__':
    main()
