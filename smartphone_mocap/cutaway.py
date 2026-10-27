"""Pick what a cut-away stage hides (used by tools/make_cutaway.py; pure, no USD).

Building parts that span several floors (outer shell, repairs, stairs) are cut per mesh: a mesh is hidden only
when it lies entirely above the cut, so the floors below keep their walls. Ceiling-mounted units are the ones
whose top is close to the ceiling; wall items hung high (photo frames) stay.
"""
from __future__ import annotations

CEILING_GAP_M = 0.35      # top within this of the ceiling
HANG_DEPTH_M = 1.5        # and bottom within this below it (walls reach down to the floor)


def above_cut(meshes, cut_z: float) -> list[str]:
    """meshes: [(path, min_z, max_z)] -> paths entirely above cut_z."""
    return [path for path, low, _high in meshes if low >= cut_z]


def ceiling_attached(min_z: float, max_z: float, *, ceiling_z: float, gap_m: float = CEILING_GAP_M,
                     hang_m: float = HANG_DEPTH_M) -> bool:
    return max_z >= ceiling_z - gap_m and ceiling_z - hang_m <= min_z < ceiling_z
