from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import CameraCalibration


def look_at_world_to_camera(
    position: np.ndarray,
    target: np.ndarray,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Return an OpenCV world-to-camera transform aimed at a world-space target."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    distance = np.linalg.norm(forward)
    if distance <= 1e-9:
        raise ValueError("Camera position and look-at target must be different")
    forward /= distance
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) <= 1e-9:
        raise ValueError("Camera cannot look exactly along the world up axis")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    if roll_deg:
        radians = np.deg2rad(float(roll_deg))
        original_right = right.copy()
        right = np.cos(radians) * original_right + np.sin(radians) * down
        down = -np.sin(radians) * original_right + np.cos(radians) * down
    rotation = np.stack((right, down, forward))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ position
    return transform


def make_approximate_calibration(
    camera_id: str,
    image_size: tuple[int, int],
    position_m: np.ndarray,
    target_m: np.ndarray,
    diagonal_fov_deg: float = 78.0,
    roll_deg: float = 0.0,
) -> CameraCalibration:
    """Make a zero-distortion pinhole calibration for pipeline smoke tests only."""
    if not 1.0 < diagonal_fov_deg < 179.0:
        raise ValueError("diagonal_fov_deg must be between 1 and 179 degrees")
    width, height = image_size
    diagonal_pixels = float(np.hypot(width, height))
    focal = diagonal_pixels / (2.0 * np.tan(np.deg2rad(diagonal_fov_deg) / 2.0))
    matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraCalibration(
        camera_id=camera_id,
        image_size=image_size,
        camera_matrix=matrix,
        distortion=np.zeros(5),
        world_to_camera=look_at_world_to_camera(position_m, target_m, roll_deg),
    )


# Fewer ChArUco corners than this triggers the bloom-tolerant marker retry.
BLOOM_RETRY_BELOW = 20


def make_charuco_board(
    squares_x: int = 7,
    squares_y: int = 5,
    square_length_m: float = 0.04,
    marker_length_m: float = 0.03,
    dictionary_name: str = "DICT_5X5_100",
):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length_m, marker_length_m, dictionary
        )
    else:
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length_m, marker_length_m, dictionary
        )
    return dictionary, board


def board_from_spec(spec: dict | None = None):
    """The (dictionary, board) an install day's setup.json "board" describes; None = 7x5/40mm."""
    spec = spec or {}
    squares_x, squares_y = spec.get("squares", (7, 5))
    return make_charuco_board(
        int(squares_x),
        int(squares_y),
        float(spec.get("square_length_m", 0.04)),
        float(spec.get("marker_length_m", 0.03)),
        spec.get("dictionary", "DICT_5X5_100"),
    )


def board_min_corners(spec: dict | None = None) -> int:
    """Corners a view needs for a pose; a board with few, large squares can accept fewer."""
    return int((spec or {}).get("min_corners", 6))


def write_charuco_board(path: str | Path, pixels_per_square: int = 240) -> Path:
    _, board = make_charuco_board()
    image_size = (7 * pixels_per_square, 5 * pixels_per_square)
    image = board.generateImage(image_size) if hasattr(board, "generateImage") else board.draw(image_size)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write board image: {destination}")
    return destination


def detect_charuco(image: np.ndarray, dictionary, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < BLOOM_RETRY_BELOW:
            # Far cameras see ~3.5px per marker bit, so over-exposed paper bleeding
            # a single pixel erases the black bits and no marker decodes. Identify
            # markers on an image with the white shaved back by 1px, then locate
            # the ChArUco corners on the untouched image so the shave cannot bias
            # their positions.
            shaved = cv2.erode(gray, np.ones((3, 3), np.uint8))
            retry_markers, retry_ids, _ = cv2.aruco.ArucoDetector(
                dictionary, cv2.aruco.DetectorParameters()
            ).detectMarkers(shaved)
            if retry_ids is not None and (marker_ids is None or len(retry_ids) > len(marker_ids)):
                retry = detector.detectBoard(gray, markerCorners=retry_markers, markerIds=retry_ids)
                if retry[1] is not None and (charuco_ids is None or len(retry[1]) > len(charuco_ids)):
                    charuco_corners, charuco_ids, marker_corners, marker_ids = retry
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if marker_ids is None:
            return None, None, marker_corners, marker_ids
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
    # OpenCV 5 Python returns ChArUco points as (N, 2) and IDs as (N,),
    # while drawDetectedCornersCharuco still expects the older vector-like
    # layouts (N, 1, 2) and (N, 1). matchImagePoints accepts both layouts.
    # Normalize here so detection, preview drawing, and calibration share one
    # representation across OpenCV 4 and 5.
    if charuco_corners is not None:
        charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    if charuco_ids is not None:
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def calibrate_intrinsics_from_images(
    camera_id: str,
    image_paths: list[str | Path],
) -> tuple[CameraCalibration, float]:
    dictionary, board = make_charuco_board()
    all_object_points = []
    all_image_points = []
    image_size: tuple[int, int] | None = None
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image_size = (image.shape[1], image.shape[0])
        corners, ids, _, _ = detect_charuco(image, dictionary, board)
        if corners is not None and ids is not None and len(ids) >= 6:
            object_points, image_points = board.matchImagePoints(corners, ids)
            all_object_points.append(object_points)
            all_image_points.append(image_points)
    if image_size is None or len(all_object_points) < 5:
        raise ValueError("At least 5 usable ChArUco images are required (10-20 recommended).")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
    )
    return (
        CameraCalibration(
            camera_id=camera_id,
            image_size=image_size,
            camera_matrix=matrix,
            distortion=distortion,
            world_to_camera=np.eye(4),
        ),
        float(error),
    )


def estimate_board_extrinsics(
    image: np.ndarray, calibration: CameraCalibration, board_spec: dict | None = None
) -> tuple[np.ndarray, float]:
    dictionary, board = board_from_spec(board_spec)
    corners, ids, _, _ = detect_charuco(image, dictionary, board)
    if corners is None or ids is None or len(ids) < board_min_corners(board_spec):
        raise ValueError("The ChArUco board was not detected clearly enough.")
    charuco_points, image_points = board.matchImagePoints(corners, ids)

    # Define BoothFrame ourselves instead of inheriting OpenCV's
    # version-dependent ChArUco board-axis convention:
    #
    #   origin: printed board's bottom-left corner
    #   +X:     toward the printed board's right edge
    #   +Y:     toward the printed board's top edge
    #   +Z:     out of the printed face (up when the board lies face-up)
    #
    # OpenCV's generated board points use X right and Y down from the
    # printed board's top-left, so flip Y before pose estimation. This gives
    # a right-handed frame whose +Z can be used as physical height.
    _, squares_y = board.getChessboardSize()
    board_height_m = float(squares_y) * float(board.getSquareLength())
    booth_points = np.asarray(charuco_points, dtype=np.float32).copy()
    booth_points[..., 1] = board_height_m - booth_points[..., 1]

    ok, rotation_vector, translation = cv2.solvePnP(
        booth_points,
        image_points,
        calibration.camera_matrix,
        calibration.distortion,
    )
    if not ok:
        raise ValueError("Could not estimate the board pose.")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    projected, _ = cv2.projectPoints(
        booth_points,
        rotation_vector,
        translation,
        calibration.camera_matrix,
        calibration.distortion,
    )
    error = float(np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1).mean())
    return transform, error
