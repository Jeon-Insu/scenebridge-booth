import numpy as np
import cv2

from smartphone_mocap.preview import (
    make_camera_analysis_mosaic,
    make_preview_mosaic,
    write_preview_image,
)


def test_three_camera_mosaic_uses_two_by_two_grid():
    frames = {
        "a": np.full((720, 1280, 3), 50, dtype=np.uint8),
        "b": np.full((480, 640, 3), 100, dtype=np.uint8),
        "c": np.full((1080, 1920, 3), 150, dtype=np.uint8),
    }
    mosaic = make_preview_mosaic(
        ["a", "b", "c"], frames, {"a": 15.0, "b": 14.0, "c": 13.0}, {}
    )
    assert mosaic.shape == (720, 1280, 3)


def test_missing_camera_still_has_a_placeholder():
    mosaic = make_preview_mosaic(["a"], {}, {}, {"a": "camera unavailable"})
    assert mosaic.shape == (360, 640, 3)
    assert np.any(mosaic)


def test_preview_image_is_published_as_readable_jpeg(tmp_path):
    image = np.full((90, 160, 3), (20, 80, 160), dtype=np.uint8)
    output = tmp_path / "camera_mosaic.jpg"
    write_preview_image(output, image)
    decoded = cv2.imread(str(output))
    assert decoded is not None
    assert decoded.shape == image.shape
    assert not (tmp_path / ".camera_mosaic.jpg.tmp").exists()


def test_four_camera_analysis_mosaic_has_eight_views():
    camera_ids = ["top_left", "top_right", "bottom_left", "bottom_right"]
    raw = {
        camera_id: np.full((90, 160, 3), 30 + index * 20, dtype=np.uint8)
        for index, camera_id in enumerate(camera_ids)
    }
    overlays = {
        camera_id: np.full((90, 160, 3), 100 + index * 20, dtype=np.uint8)
        for index, camera_id in enumerate(camera_ids)
    }

    mosaic = make_camera_analysis_mosaic(
        camera_ids,
        raw,
        overlays,
        {camera_id: 5.0 for camera_id in camera_ids},
        {},
        tile_width=160,
        tile_height=90,
    )

    assert mosaic.shape == (360, 320, 3)
    assert not np.array_equal(mosaic[30:90, :160], mosaic[30:90, 160:])
