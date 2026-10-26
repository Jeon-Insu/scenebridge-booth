"""배경 안에 가상 테이블을 놓는 기하(순수 함수, Isaac 없이 테스트)."""
import numpy as np


def table_pose_facing_wall(wall_centre_xy, wall_normal_xy, distance_m, table_size):
    n = np.asarray(wall_normal_xy, float)
    n /= np.linalg.norm(n)
    centre = np.asarray(wall_centre_xy, float) + n * distance_m
    y_axis = -n                                   # table +Y points at the wall
    x_axis = np.array([y_axis[1], -y_axis[0]])
    yaw = float(np.degrees(np.arctan2(x_axis[1], x_axis[0])))
    origin = centre - x_axis * table_size[0] / 2 - y_axis * table_size[1] / 2
    return (float(origin[0]), float(origin[1])), yaw


def overlapping(boxes, footprint_corners_xy, margin_m):
    pts = np.asarray(footprint_corners_xy, float)
    lo, hi = pts.min(0) - margin_m, pts.max(0) + margin_m
    return sorted(path for path, (x0, y0, x1, y1) in boxes.items()
                  if x0 <= hi[0] and x1 >= lo[0] and y0 <= hi[1] and y1 >= lo[1])
