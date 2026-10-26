import cv2
import numpy as np
import pytest

from smartphone_mocap.calibration import (
    detect_charuco,
    estimate_board_extrinsics,
    make_charuco_board,
    write_charuco_board,
)
from smartphone_mocap.types import CameraCalibration


def test_charuco_detection_normalizes_opencv_5_shapes():
    dictionary, board = make_charuco_board()
    gray = board.generateImage((1680, 1200))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    corners, ids, _, _ = detect_charuco(image, dictionary, board)

    assert corners.shape == (24, 1, 2)
    assert ids.shape == (24, 1)
    cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)


def test_charuco_pose_estimation_with_current_opencv(tmp_path):
    board_path = write_charuco_board(tmp_path / "board.png")
    board_image = cv2.imread(str(board_path))
    camera_matrix = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    expected_rotation_vector = np.array([0.18, -0.10, 0.04])
    expected_translation = np.array([-0.12, -0.08, 1.2])
    board_outer_corners = np.array(
        [[0, 0, 0], [0.28, 0, 0], [0.28, 0.20, 0], [0, 0.20, 0]], dtype=np.float32
    )
    destination, _ = cv2.projectPoints(
        board_outer_corners,
        expected_rotation_vector,
        expected_translation,
        camera_matrix,
        np.zeros(5),
    )
    height, width = board_image.shape[:2]
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(source, destination.reshape(4, 2).astype(np.float32))
    camera_image = cv2.warpPerspective(
        board_image,
        homography,
        (1280, 720),
        borderValue=(255, 255, 255),
    )
    calibration = CameraCalibration(
        "test_camera", (1280, 720), camera_matrix, np.zeros(5), np.eye(4)
    )
    transform, reprojection_error = estimate_board_extrinsics(camera_image, calibration)

    charuco_to_camera = np.eye(4)
    charuco_to_camera[:3, :3], _ = cv2.Rodrigues(expected_rotation_vector)
    charuco_to_camera[:3, 3] = expected_translation
    booth_to_charuco = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.20],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    expected_transform = charuco_to_camera @ booth_to_charuco

    assert reprojection_error < 1.0
    assert np.linalg.norm(transform[:3, 3] - expected_transform[:3, 3]) < 0.02
    assert np.linalg.norm(transform[:3, :3] - expected_transform[:3, :3]) < 0.03
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0, abs=1e-6)


def _small_board_scene(board, width_px=300, bloom_px=0):
    """A board as far cameras see it: markers ~25px wide, optionally with white bloom."""
    printed = board.generateImage((1400, 1000), marginSize=60)
    small = cv2.resize(printed, (width_px, int(width_px * 1000 / 1400)), interpolation=cv2.INTER_AREA)
    scene = np.full((720, 1280), 40, np.uint8)
    scene[200:200 + small.shape[0], 400:400 + width_px] = small
    if bloom_px:
        scene = cv2.dilate(scene, np.ones((3, 3), np.uint8), iterations=bloom_px)
    return cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)


def test_charuco_detection_survives_white_bloom_on_small_markers():
    # Venue 2026-09-18: over-exposed paper bled ~1px into the black marker bits,
    # so the plain detector found 0 markers although the board was in plain view.
    dictionary, board = make_charuco_board()
    image = _small_board_scene(board, bloom_px=1)

    corners, ids, _, _ = detect_charuco(image, dictionary, board)

    assert ids is not None and len(ids) >= 20
    assert corners.shape == (len(ids), 1, 2)


def test_charuco_detection_keeps_clean_small_board_result():
    dictionary, board = make_charuco_board()

    _, ids, _, _ = detect_charuco(_small_board_scene(board), dictionary, board)

    assert ids is not None and len(ids) == 24


# Venue 2026-10-25: a board with fewer, larger squares on one A4 sheet so cameras ~1.1m
# away can read the markers (the 7x5/40mm board gave them only 2-3px per marker bit).
A4_BOARD = {"squares": [5, 3], "square_length_m": 0.056, "marker_length_m": 0.042,
            "dictionary": "DICT_4X4_50", "min_corners": 4}


def test_a_board_spec_builds_that_board():
    from smartphone_mocap.calibration import board_from_spec
    dictionary, board = board_from_spec(A4_BOARD)
    image = cv2.cvtColor(board.generateImage((1400, 840)), cv2.COLOR_GRAY2BGR)

    corners, ids, _, _ = detect_charuco(image, dictionary, board)

    assert board.getChessboardSize() == (5, 3)
    assert board.getSquareLength() == pytest.approx(0.056)
    assert ids is not None and len(ids) == 8


def test_no_board_spec_keeps_the_original_board():
    from smartphone_mocap.calibration import board_from_spec
    _, board = board_from_spec(None)

    assert board.getChessboardSize() == (7, 5)
    assert board.getSquareLength() == pytest.approx(0.04)


def test_pose_from_a_board_spec_uses_that_boards_size():
    from smartphone_mocap.calibration import board_from_spec
    _, board = board_from_spec(A4_BOARD)
    board_image = cv2.cvtColor(board.generateImage((1400, 840)), cv2.COLOR_GRAY2BGR)
    camera_matrix = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    rotation_vector = np.array([0.5, -0.10, 0.04])
    translation = np.array([-0.14, -0.08, 1.1])
    outline = np.array([[0, 0, 0], [0.28, 0, 0], [0.28, 0.168, 0], [0, 0.168, 0]], np.float32)
    destination, _ = cv2.projectPoints(outline, rotation_vector, translation, camera_matrix, np.zeros(5))
    height, width = board_image.shape[:2]
    source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(source, destination.reshape(4, 2).astype(np.float32))
    camera_image = cv2.warpPerspective(board_image, homography, (1280, 720), borderValue=(255, 255, 255))
    calibration = CameraCalibration("test_camera", (1280, 720), camera_matrix, np.zeros(5), np.eye(4))

    transform, error = estimate_board_extrinsics(camera_image, calibration, A4_BOARD)

    # BoothFrame origin is the board's bottom-left, i.e. OpenCV's (0, height) point.
    origin_in_camera = cv2.Rodrigues(rotation_vector)[0] @ np.array([0, 0.168, 0]) + translation
    assert error < 1.0
    assert np.linalg.norm(transform[:3, 3] - origin_in_camera) < 0.01


def test_a_board_spec_can_accept_fewer_corners():
    from smartphone_mocap.calibration import board_min_corners
    assert board_min_corners(A4_BOARD) == 4
    assert board_min_corners(None) == 6
