"""Venue extrinsics when every camera reads the board: table placement and table frame."""
import numpy as np
import pytest

from smartphone_mocap.board_table import fit_table, median_pose, table_from_board

SIZE = (1.25, 0.75)


def _rect_map(centre, yaw_deg, size, clutter=True):
    xs = np.arange(-1.5, 1.5, 0.005)
    ys = np.arange(-1.2, 1.2, 0.005)
    X, Y = np.meshgrid(xs, ys)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    u = (X - centre[0]) * c + (Y - centre[1]) * s
    v = -(X - centre[0]) * s + (Y - centre[1]) * c
    white = (np.abs(u) <= size[0] / 2) & (np.abs(v) <= size[1] / 2)
    if clutter:   # a white chair against one long side and a dark board on top
        white |= (np.hypot(X - centre[0], Y - (centre[1] - size[1] / 2 - 0.1)) < 0.15)
        white &= ~((np.abs(X) < 0.14) & (np.abs(Y) < 0.08))
    return white.astype(float), np.ones_like(X, dtype=int) * 3, xs, ys


def test_fit_table_finds_a_known_size_table_beside_clutter():
    frac, seen, xs, ys = _rect_map((0.14, 0.09), 1.5, SIZE)

    centre, yaw_deg = fit_table(frac, seen, xs, ys, SIZE)

    assert centre == pytest.approx((0.14, 0.09), abs=0.01)
    assert yaw_deg == pytest.approx(1.5, abs=0.5)


def test_fit_table_reports_the_long_axis_within_90_degrees():
    frac, seen, xs, ys = _rect_map((0.0, 0.0), 89.0, SIZE, clutter=False)

    _, yaw_deg = fit_table(frac, seen, xs, ys, SIZE)

    assert -90 < yaw_deg <= 90
    assert abs(abs(yaw_deg) - 89.0) < 0.6


def test_table_frame_puts_b_and_d_on_the_front_edge_left_to_right():
    # Board-frame camera positions from 2026-10-25: B, D on the board's +y side.
    cams = {'cam_A': (0.85, -0.26), 'cam_B': (0.84, 0.60), 'cam_C': (-0.62, -0.43), 'cam_D': (-0.61, 0.47)}
    T = table_from_board((0.14, 0.09), 0.0, SIZE, cams)

    def to_table(xy):
        return (T @ np.array([*xy, 0.0, 1.0]))[:3]

    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    # The table's corners land on (0,0), (L,0), (L,W), (0,W).
    assert to_table((0.14 + 0.625, 0.09 + 0.375)) == pytest.approx([0, 0, 0], abs=1e-9)   # B's corner
    assert to_table((0.14 - 0.625, 0.09 - 0.375)) == pytest.approx([1.25, 0.75, 0], abs=1e-9)
    b, d, a, c = (to_table(cams[k]) for k in ('cam_B', 'cam_D', 'cam_A', 'cam_C'))
    assert b[0] < d[0] and a[0] < c[0]          # +X runs from B to D (and A to C)
    assert b[1] < a[1] and d[1] < c[1]          # +Y runs away from the operator


def test_median_pose_is_robust_to_one_bad_frame():
    good = np.eye(4); good[:3, 3] = (0.1, 0.2, 1.0)
    bad = np.eye(4); bad[:3, 3] = (0.5, -0.4, 2.0)

    pose = median_pose([good, good, bad, good, good])

    assert pose[:3, 3] == pytest.approx((0.1, 0.2, 1.0))
    assert pose[:3, :3] == pytest.approx(np.eye(3))
