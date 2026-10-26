import numpy as np

from smartphone_mocap.geometry import Observation, project_point, triangulate_joint
from smartphone_mocap.synthetic import make_four_camera_rig
from smartphone_mocap.types import Keypoint2D


def test_robust_triangulation_ignores_one_outlier():
    cameras = make_four_camera_rig()
    expected = np.array([0.24, -0.18, 1.23])
    observations = []
    for index, camera in enumerate(cameras.values()):
        point = project_point(camera, expected)
        if index == 3:
            point += np.array([180.0, -120.0])
        observations.append(Observation(camera, Keypoint2D(*point, 0.95)))
    joint = triangulate_joint(observations, max_reprojection_error_px=10.0)
    assert joint is not None
    actual = np.array([joint.x, joint.y, joint.z])
    assert np.linalg.norm(actual - expected) < 1e-6
    assert len(joint.source_cameras) == 3

