# exts/gist.tabletop_twin/gist/tabletop_twin/tracked_pose.py
"""Pure helpers for tracked-object pose fields in tracking messages (no Kit imports)."""
import math


def tracked_yaw_deg(scene_object) -> float:
    try:
        value = float(scene_object.get("yaw_deg", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def display_label(scene_object, fallback: str) -> str:
    name = scene_object.get("display_name")
    return str(name) if name else fallback


def shown_yaw_deg(tracked_yaw: float, agent_rotation_wxyz) -> float:
    """The tracked heading is shown only while the agent has not turned the object (agent wins)."""
    w, x, y, z = (float(v) for v in agent_rotation_wxyz)
    unrotated = abs(abs(w) - 1.0) < 1e-6 and abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6
    return float(tracked_yaw) if unrotated else 0.0
