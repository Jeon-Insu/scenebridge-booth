import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / "exts/gist.tabletop_twin/gist/tabletop_twin/camera_layout.py"
spec = importlib.util.spec_from_file_location("camera_layout", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paired_mosaic_becomes_two_matching_2x2_grids():
    paired = np.zeros((40, 40, 3), dtype=np.uint8)
    for camera in range(4):
        paired[camera*10:(camera+1)*10, :20] = camera + 1
        paired[camera*10:(camera+1)*10, 20:] = camera + 11
    original = paired.copy()
    detection, masks = module.split_camera_analysis_mosaic(paired)
    assert detection.shape == masks.shape == (20, 40, 3)
    for camera in range(4):
        y, x = divmod(camera, 2)
        assert np.all(detection[y*10:(y+1)*10, x*20:(x+1)*20] == camera + 1)
        assert np.all(masks[y*10:(y+1)*10, x*20:(x+1)*20] == camera + 11)
    assert np.array_equal(paired, original)


@pytest.mark.parametrize("shape", [(3, 40, 3), (40, 41, 3), (41, 40, 3), (40, 40), (40, 40, 1)])
def test_reject_non_paired_mosaic(shape):
    with pytest.raises(ValueError):
        module.split_camera_analysis_mosaic(np.zeros(shape, dtype=np.uint8))
