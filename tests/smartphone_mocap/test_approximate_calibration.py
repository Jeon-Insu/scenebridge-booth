import numpy as np

from smartphone_mocap.calibration import make_approximate_calibration
from smartphone_mocap.geometry import project_point


def test_approximate_camera_projects_target_to_image_center():
    target = np.array([0.18, 0.215, 0.06])
    calibration = make_approximate_calibration(
        "top_right",
        (1280, 720),
        np.array([0.36, 0.43, 0.06]),
        target,
        78.0,
    )
    assert np.allclose(project_point(calibration, target), [640.0, 360.0])


def test_approximate_calibration_uses_camera_position():
    position = np.array([0.36, 0.0, 0.06])
    calibration = make_approximate_calibration(
        "bottom_right",
        (1280, 720),
        position,
        np.array([0.18, 0.215, 0.06]),
    )
    recovered_position = -calibration.rotation.T @ calibration.translation
    assert np.allclose(recovered_position, position)
