# tests/smartphone_mocap/test_venue_layout.py
import numpy as np
import pytest

from smartphone_mocap.venue_layout import overlapping, table_pose_facing_wall


def test_the_table_stands_in_front_of_the_wall_with_its_back_edge_toward_it():
    # Wall at y = 5 facing -Y (normal into the room is (0, -1)); table 1.25 x 0.75, centre 1.5 m from the wall.
    offset, yaw = table_pose_facing_wall((2.0, 5.0), (0.0, -1.0), 1.5, (1.25, 0.75))
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    centre = np.array(offset) + np.array([[c, -s], [s, c]]) @ np.array([0.625, 0.375])
    assert centre == pytest.approx([2.0, 3.5])
    back_direction = np.array([[c, -s], [s, c]]) @ np.array([0.0, 1.0])     # table +Y
    assert back_direction == pytest.approx([0.0, 1.0])                        # toward the wall


def test_furniture_under_the_table_footprint_is_listed():
    boxes = {"/desk_1": (1.0, 3.0, 1.6, 3.6), "/desk_far": (5.0, 0.0, 5.6, 0.6)}
    corners = [(1.375, 3.125), (2.625, 3.125), (2.625, 3.875), (1.375, 3.875)]
    assert overlapping(boxes, corners, 0.5) == ["/desk_1"]
