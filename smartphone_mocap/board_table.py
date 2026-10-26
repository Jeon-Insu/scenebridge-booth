"""Venue extrinsics when every camera reads the board (2026-10-25 onward).

Each camera's pose comes straight from the board. The board need not sit at the table centre:
the table itself is located by back-projecting every view onto the board plane, where the
white tabletop is a known-size rectangle, and the table frame follows the booth convention —
cam_B and cam_D stand on the operator's (front) long edge, +X runs from B to D, +Y away from the
operator, +Z up, origin at the front-left corner.
"""
from __future__ import annotations

import cv2
import numpy as np


def median_pose(transforms) -> np.ndarray:
    """Element-wise median of rotation vectors and translations; one bad frame cannot pull it."""
    rotations = [cv2.Rodrigues(np.asarray(t)[:3, :3])[0].ravel() for t in transforms]
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.median(rotations, axis=0))[0]
    pose[:3, 3] = np.median([np.asarray(t)[:3, 3] for t in transforms], axis=0)
    return pose


def _score(points, weights, centre, yaw_deg, size):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    dx, dy = points[:, 0] - centre[0], points[:, 1] - centre[1]
    inside = (np.abs(dx * c + dy * s) <= size[0] / 2) & (np.abs(-dx * s + dy * c) <= size[1] / 2)
    return float(weights[inside].sum())


def fit_table(white_fraction, seen, xs, ys, size, min_views=2):
    """Centre (board metres) and long-axis yaw (degrees, in (-90, 90]) of the known-size table.

    ``white_fraction`` is, per board-plane cell, the share of the views seeing it that see it
    white; cells seen by fewer than ``min_views`` views count as unknown. The fit maximises white
    cells inside minus dark cells inside, so a white chair beside the table cannot drag it.
    """
    known = seen >= min_views
    weights = np.where(known, 2.0 * white_fraction - 1.0, 0.0)
    rows, cols = np.nonzero(known)
    points = np.stack([np.asarray(xs)[cols], np.asarray(ys)[rows]], axis=1)
    weights = weights[rows, cols]

    # Start from the largest white region's rectangle.
    mask = (known & (white_fraction >= 0.75)).astype(np.uint8)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    (cx, cy), (w, h), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    step = float(xs[1] - xs[0])
    centre = np.array([xs[0] + cx * step, ys[0] + cy * step])
    yaw = angle if w >= h else angle + 90.0

    best = _score(points, weights, centre, yaw, size)
    for move, turn in ((0.04, 2.0), (0.02, 1.0), (0.01, 0.5), (0.005, 0.25), (0.0025, 0.1)):
        improved = True
        while improved:
            improved = False
            for dx in (-move, 0, move):
                for dy in (-move, 0, move):
                    for dyaw in (-turn, 0, turn):
                        candidate = _score(points, weights, centre + (dx, dy), yaw + dyaw, size)
                        if candidate > best + 1e-9:
                            best, centre, yaw, improved = candidate, centre + (dx, dy), yaw + dyaw, True
    yaw = (yaw + 90.0) % 180.0 - 90.0
    if yaw == -90.0:
        yaw = 90.0
    return (float(centre[0]), float(centre[1])), float(yaw)


def table_from_board(centre, yaw_deg, size, camera_xy, face_height_m=0.0) -> np.ndarray:
    """4x4 transform from board coordinates to the booth table frame.

    ``camera_xy`` holds board-plane positions of cam_A..cam_D; B and D fix which long edge is
    the front and which way +X runs.
    """
    long_axis = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
    b, d = np.asarray(camera_xy['cam_B']), np.asarray(camera_xy['cam_D'])
    x_axis = long_axis if (d - b) @ long_axis > 0 else -long_axis
    y_axis = np.array([-x_axis[1], x_axis[0]])
    if (np.asarray(camera_xy['cam_A']) - b) @ y_axis <= 0:
        raise ValueError('cam_A is not behind cam_B: check which view is which camera')
    origin = np.asarray(centre) - x_axis * size[0] / 2 - y_axis * size[1] / 2
    transform = np.eye(4)
    transform[0, :2], transform[1, :2] = x_axis, y_axis
    transform[:2, 3] = -transform[:2, :2] @ origin
    transform[2, 3] = face_height_m
    return transform
