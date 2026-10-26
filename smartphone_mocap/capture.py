from __future__ import annotations

from pathlib import Path

import cv2

from .calibration import detect_charuco, make_charuco_board
from .calibration_guide import INTRINSICS_GUIDE_20


def calibration_preview_with_status(
    preview, *, corner_count: int, total_corners: int, minimum_corners: int,
    saved_count: int, max_images: int,
    guide_pose: tuple[str, str] | None = None,
):
    """Add large status text outside the camera image; never obscure the board."""
    scale = preview.shape[1] / 1280.0
    header_height = max(1, round((260 if guide_pose else 145) * scale))
    result = cv2.copyMakeBorder(preview, header_height, 0, 0, 0,
                                cv2.BORDER_CONSTANT, value=(20, 20, 20))
    valid = corner_count >= minimum_corners
    color = (60, 240, 90) if valid else (70, 100, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, round(4 * scale))
    baseline = round(65 * scale)
    cv2.putText(result, f"CORNERS {corner_count} / {total_corners}",
                (round(20 * scale), baseline), font, 2.0 * scale, color,
                thickness, cv2.LINE_AA)
    cv2.putText(result, f"SAVED {saved_count} / {max_images}",
                (round(700 * scale), baseline), font, 2.0 * scale,
                (245, 245, 245), thickness, cv2.LINE_AA)
    hint = ("READY - SPACE to save" if valid
            else f"Need {minimum_corners}+ corners - show the board")
    cv2.putText(result, hint, (round(20 * scale), round(122 * scale)),
                font, 1.25 * scale, color, max(1, round(3 * scale)), cv2.LINE_AA)
    if guide_pose is not None:
        position, instruction = guide_pose
        cv2.putText(result, f"POSE {saved_count + 1:02d} / {max_images} - {position}",
                    (round(20 * scale), round(186 * scale)), font, 1.6 * scale,
                    (80, 220, 255), max(1, round(3 * scale)), cv2.LINE_AA)
        cv2.putText(result, instruction, (round(20 * scale), round(239 * scale)),
                    font, 1.15 * scale, (240, 240, 240),
                    max(1, round(2 * scale)), cv2.LINE_AA)
    return result


def capture_charuco_images(
    url: str,
    camera_id: str,
    output_dir: str | Path,
    *,
    max_images: int = 20,
    minimum_corners: int = 8,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    fourcc: str | None = None,
    guide_20: bool = False,
) -> list[Path]:
    """Interactively save original stream frames when a ChArUco board is visible."""
    if (width is None) != (height is None):
        raise ValueError("Specify both width and height, or neither.")
    if width is not None and (width <= 0 or height <= 0):
        raise ValueError("Image dimensions must be positive.")
    if fps is not None and fps <= 0:
        raise ValueError("FPS must be positive.")
    if fourcc is not None and len(fourcc) != 4:
        raise ValueError("FOURCC must contain exactly four characters, e.g. MJPG.")
    if guide_20 and max_images != len(INTRINSICS_GUIDE_20):
        raise ValueError("The 20-pose guide requires --max-images 20.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing = sorted(destination.glob(f"{camera_id}_*.jpg"))
    indices = [int(path.stem.rsplit("_", 1)[-1]) for path in existing
               if path.stem.rsplit("_", 1)[-1].isdigit()]
    next_index = max(indices, default=-1) + 1
    saved: list[Path] = []
    # Guided sessions resume toward 20 total files; legacy capture still counts
    # --max-images as additional images for backwards compatibility.
    starting_count = len(existing) if guide_20 else 0
    if guide_20 and starting_count >= max_images:
        print(f"Guide complete: {starting_count} images already saved in {destination}.", flush=True)
        return saved
    dictionary, board = make_charuco_board()
    squares_x, squares_y = board.getChessboardSize()
    total_corners = (squares_x - 1) * (squares_y - 1)
    source = int(url) if str(url).isdigit() else url
    backend = cv2.CAP_V4L2 if isinstance(source, int) or str(source).startswith("/dev/") else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    window_name = f"ChArUco capture - {camera_id}"
    window_open = False
    first_frame = True
    try:
        if not capture.isOpened():
            raise ConnectionError(f"Could not open video stream: {url}")
        if fourcc is not None:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            capture.set(cv2.CAP_PROP_FPS, fps)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_open = True
        cv2.resizeWindow(window_name, 960, 740 if guide_20 else 650)
        print("SPACE: save a valid board pose | Q or ESC: finish", flush=True)
        while starting_count + len(saved) < max_images:
            ok, image = capture.read()
            if not ok:
                raise ConnectionError("The video stream stopped while capturing calibration images.")
            actual_height, actual_width = image.shape[:2]
            if width is not None and (actual_width, actual_height) != (width, height):
                raise ValueError(
                    f"Camera delivered {actual_width}x{actual_height}, requested {width}x{height}. "
                    "No mismatched calibration frame was saved."
                )
            if first_frame:
                print(f"Camera {camera_id}: {actual_width}x{actual_height}, "
                      f"reported FPS={capture.get(cv2.CAP_PROP_FPS):g}", flush=True)
                first_frame = False
            corners, ids, marker_corners, marker_ids = detect_charuco(image, dictionary, board)
            preview = image.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
            corner_count = 0 if ids is None else len(ids)
            if corners is not None and ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids, (80, 240, 80))
            valid = corner_count >= minimum_corners
            saved_count = starting_count + len(saved)
            preview = calibration_preview_with_status(
                preview, corner_count=corner_count, total_corners=total_corners,
                minimum_corners=minimum_corners, saved_count=saved_count,
                max_images=max_images,
                guide_pose=INTRINSICS_GUIDE_20[saved_count] if guide_20 else None,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                if not valid:
                    print(f"Not saved: only {corner_count} ChArUco corners detected")
                    continue
                path = destination / f"{camera_id}_{next_index:03d}.jpg"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Could not save image: {path}")
                saved.append(path)
                next_index += 1
                print(f"Saved {path} ({corner_count} corners)", flush=True)
                if guide_20 and starting_count + len(saved) < max_images:
                    position, instruction = INTRINSICS_GUIDE_20[starting_count + len(saved)]
                    print(f"Next pose {starting_count + len(saved) + 1}/20: "
                          f"{position} - {instruction}", flush=True)
    finally:
        capture.release()
        if window_open:
            cv2.destroyWindow(window_name)
    return saved
