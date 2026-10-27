"""School-to-table fly-in: camera pose over time and the cut-away stages that are due (no Kit imports).

The plan comes from the venue's cut-away file (usd/venues/<venue>_cutaway.json, made by tools/make_cutaway.py):
keyframes [{"t", "eye", "target"}] in world metres and stages [{"at", "label", "paths"}] to hide.
"""
from __future__ import annotations


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)          # smoothstep: gentle start and stop


def _lerp(a, b, u):
    return [x + (y - x) * u for x, y in zip(a, b)]


def duration(keys) -> float:
    return float(keys[-1]["t"]) if keys else 0.0


def pose_at(keys, t: float):
    """(eye, target) at time t, eased between neighbouring keyframes."""
    if t <= keys[0]["t"]:
        return list(keys[0]["eye"]), list(keys[0]["target"])
    for a, b in zip(keys, keys[1:]):
        if t <= b["t"]:
            u = _ease((t - a["t"]) / max(b["t"] - a["t"], 1e-9))
            return _lerp(a["eye"], b["eye"], u), _lerp(a["target"], b["target"], u)
    return list(keys[-1]["eye"]), list(keys[-1]["target"])


def stages_due(stages, t_prev: float, t: float):
    """Stages whose time falls in (t_prev, t]: each fires exactly once while the clock moves forward."""
    return [s for s in stages if t_prev < float(s["at"]) <= t]
