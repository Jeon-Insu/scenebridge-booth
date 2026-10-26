import cv2
import numpy as np
import pytest

from smartphone_mocap import capture as module


@pytest.fixture
def camera(monkeypatch):
    class Camera:
        opened = True
        released = False
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        def __init__(self):
            self.settings = []

        def isOpened(self):
            return self.opened

        def set(self, prop, value):
            self.settings.append((prop, value))
            return True

        def get(self, prop):
            return 30.0

        def read(self):
            return True, self.frame.copy()

        def release(self):
            self.released = True

    camera = Camera()
    camera.source = None

    def open_camera(source, backend):
        camera.source = (source, backend)
        return camera

    monkeypatch.setattr(module.cv2, "VideoCapture", open_camera)
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(module.cv2, name, lambda *args: None)
    monkeypatch.setattr(module.cv2, "waitKey", lambda *_: 32)
    monkeypatch.setattr(module.cv2, "getWindowProperty", lambda *_: 1)
    monkeypatch.setattr(module, "detect_charuco", lambda *_: (
        None, np.arange(12, dtype=np.int32).reshape(-1, 1), None, None))
    return camera


def test_usb_capture_sets_mode_and_preserves_existing_numbered_images(camera, tmp_path):
    (tmp_path / "cam_A_000.jpg").write_bytes(b"keep first")
    (tmp_path / "cam_A_002.jpg").write_bytes(b"keep last")
    paths = module.capture_charuco_images(
        "0", "cam_A", tmp_path, max_images=1,
        width=1280, height=720, fps=30, fourcc="MJPG")
    assert camera.source == (0, cv2.CAP_V4L2)
    assert paths == [tmp_path / "cam_A_003.jpg"]
    assert cv2.imread(str(paths[0])).shape == (720, 1280, 3)
    assert (tmp_path / "cam_A_002.jpg").read_bytes() == b"keep last"
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in camera.settings
    assert camera.released


def test_wrong_resolution_is_not_saved(camera, tmp_path):
    camera.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="delivered 640x480"):
        module.capture_charuco_images("/dev/video0", "cam_A", tmp_path,
                                      width=1280, height=720)
    assert not list(tmp_path.glob("*.jpg"))
    assert camera.released


def test_network_stream_keeps_default_mode(camera, tmp_path):
    module.capture_charuco_images("http://camera/video", "cam_A", tmp_path, max_images=1)
    assert camera.source == ("http://camera/video", cv2.CAP_ANY)
    assert not camera.settings


def test_failed_open_releases_device(camera, tmp_path):
    camera.opened = False
    with pytest.raises(ConnectionError):
        module.capture_charuco_images("/dev/video0", "cam_A", tmp_path)
    assert camera.released


def test_window_close_finishes_without_saving(camera, monkeypatch, tmp_path):
    monkeypatch.setattr(module.cv2, "waitKey", lambda *_: -1)
    monkeypatch.setattr(module.cv2, "getWindowProperty", lambda *_: 0)
    assert module.capture_charuco_images("0", "cam_A", tmp_path) == []
    assert camera.released


def test_dimensions_must_be_paired(tmp_path):
    with pytest.raises(ValueError, match="both width and height"):
        module.capture_charuco_images("0", "cam_A", tmp_path, width=1280)


def test_large_status_is_outside_image_and_does_not_modify_original():
    frame = np.full((720, 1280, 3), 137, dtype=np.uint8)
    original = frame.copy()
    result = module.calibration_preview_with_status(
        frame, corner_count=20, total_corners=24, minimum_corners=15,
        saved_count=3, max_images=25)
    assert result.shape == (865, 1280, 3)
    assert np.array_equal(result[145:], original)
    assert np.array_equal(frame, original)
    assert np.any(result[:145, :, 1] > 200)


def test_status_numbers_are_two_and_half_times_original_font(monkeypatch):
    labels = []
    monkeypatch.setattr(module.cv2, "putText", lambda img, text, origin, font, scale, *args:
                        labels.append((text, scale)))
    module.calibration_preview_with_status(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        corner_count=8, total_corners=24, minimum_corners=15,
        saved_count=0, max_images=25)
    assert labels[0] == ("CORNERS 8 / 24", 2.0)
    assert labels[1] == ("SAVED 0 / 25", 2.0)
    assert "Need 15+" in labels[2][0]


def test_guide_header_preserves_full_image_and_shows_current_pose(monkeypatch):
    frame = np.full((720, 1280, 3), 137, dtype=np.uint8)
    labels = []
    original_put_text = module.cv2.putText

    def record_text(image, text, *args):
        labels.append(text)
        return original_put_text(image, text, *args)

    monkeypatch.setattr(module.cv2, "putText", record_text)
    result = module.calibration_preview_with_status(
        frame, corner_count=20, total_corners=24, minimum_corners=15,
        saved_count=11, max_images=20, guide_pose=module.INTRINSICS_GUIDE_20[11])
    assert result.shape == (980, 1280, 3)
    assert np.array_equal(result[260:], frame)
    assert "POSE 12 / 20 - CENTER" in labels
    assert "Left edge toward camera ~30 deg" in labels


def test_guided_resume_saves_only_remaining_images(camera, monkeypatch, tmp_path):
    for index in range(19):
        (tmp_path / f"cam_A_{index:03d}.jpg").write_bytes(b"keep")
    displayed_counts = []
    original_render = module.calibration_preview_with_status

    def render(image, **kwargs):
        displayed_counts.append((kwargs["saved_count"], kwargs["guide_pose"]))
        return original_render(image, **kwargs)

    monkeypatch.setattr(module, "calibration_preview_with_status", render)
    paths = module.capture_charuco_images("0", "cam_A", tmp_path, guide_20=True)
    assert paths == [tmp_path / "cam_A_019.jpg"]
    assert displayed_counts == [(19, module.INTRINSICS_GUIDE_20[19])]
    assert len(list(tmp_path.glob("*.jpg"))) == 20
    assert (tmp_path / "cam_A_000.jpg").read_bytes() == b"keep"


def test_completed_guide_does_not_open_camera(camera, tmp_path):
    for index in range(20):
        (tmp_path / f"cam_A_{index:03d}.jpg").write_bytes(b"keep")
    assert module.capture_charuco_images("0", "cam_A", tmp_path, guide_20=True) == []
    assert camera.source is None


def test_guide_rejects_non_twenty_target(camera, tmp_path):
    with pytest.raises(ValueError, match="--max-images 20"):
        module.capture_charuco_images("0", "cam_A", tmp_path, guide_20=True, max_images=25)
    assert camera.source is None


def test_guide_has_twenty_steps_with_readable_fitting_text():
    assert len(module.INTRINSICS_GUIDE_20) == 20
    for index, (position, instruction) in enumerate(module.INTRINSICS_GUIDE_20):
        title = f"POSE {index + 1:02d} / 20 - {position}"
        for text, scale, thickness in [(title, 1.6, 3), (instruction, 1.15, 2)]:
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            assert size[0] <= 1240
