"""Deterministic, conservative tabletop placement; no physics or policy inference.

Shared by the agent service and Kit (loaded by file path to avoid importing the
perception runtime into Kit). Points are USD leaf-bound corners in object space.
"""
from __future__ import annotations

import math

PLACEMENT_MODES = ("preserve_position", "maintain_support", "place_on_table")
CONTACT_TOLERANCE_M = 0.005


def inspect_contact(geometry, xyz, rotation_wxyz, scale):
    if not geometry or not geometry.get("points_local_m"):
        return {"available": False, "state": "unknown", "warnings": ["geometry_unavailable"]}
    w, x, y, z = rotation_wxyz
    norm = math.hypot(w, x, y, z)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid contact rotation.")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    matrix = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
              (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
              (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    points = []
    for source in geometry["points_local_m"]:
        point = [xyz[i] + sum(matrix[i][j]*source[j]*scale[j] for j in range(3)) for i in range(3)]
        if not all(math.isfinite(v) for v in point):
            raise ValueError("Nonfinite contact geometry.")
        points.append(point)
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    xmin, xmax, ymin, ymax = geometry.get("table_bounds_xy_m", [0, 1.25, 0, .75])
    inside = low[0] >= xmin-1e-6 and high[0] <= xmax+1e-6 and low[1] >= ymin-1e-6 and high[1] <= ymax+1e-6
    overlaps = high[0] >= xmin and low[0] <= xmax and high[1] >= ymin and low[1] <= ymax
    clearance = low[2]  # TableFrame surface is Z=0, not world Z.
    penetration = max(0.0, -clearance) if overlaps else 0.0
    state = ("outside_table" if not overlaps else "penetrating" if clearance < -CONTACT_TOLERANCE_M
             else "contact" if inside and abs(clearance) <= CONTACT_TOLERANCE_M
             else "airborne" if clearance > CONTACT_TOLERANCE_M else "edge_overlap")
    warnings = []
    if penetration > 1e-6:
        warnings.append("possible_table_penetration")
    if not inside:
        warnings.append("footprint_outside_table")
    return {"available": True, "method": "usd_leaf_bounds_conservative", "state": state,
            "clearance_m": clearance, "possible_penetration_m": penetration,
            "footprint_inside_table": inside, "bounds_min_m": low, "bounds_max_m": high,
            "warnings": warnings, "physics_validated": False}


def resolve_placement(geometry, current, proposed, placement="preserve_position"):
    """Resolve all requested TRS together, without mutating current or USD."""
    if placement not in PLACEMENT_MODES:
        raise ValueError("Unknown placement mode.")
    target = {key: list(proposed.get(key, current[key])) for key in ("xyz", "rotation_wxyz", "scale")}
    before = inspect_contact(geometry, **current)
    after = inspect_contact(geometry, **target)
    correction = 0.0
    if placement != "preserve_position":
        if not after["available"]:
            raise ValueError("USD geometry is unavailable; cannot compute table contact.")
        if placement == "maintain_support" and before["state"] != "contact":
            raise ValueError("Object is not currently in table contact. Keep its position, or explicitly place it on the table.")
        if not after["footprint_inside_table"]:
            raise ValueError("Object bounds extend beyond the tabletop; move inward or use a smaller size first.")
        correction = -after["clearance_m"]
        target["xyz"][2] += correction
        after = inspect_contact(geometry, **target)
    return {"target": target, "placement": placement, "z_correction_m": correction,
            "contact_before": before, "contact_after": after}
