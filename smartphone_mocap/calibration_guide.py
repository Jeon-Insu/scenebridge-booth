"""Requested board poses, not measured or automatically validated poses.

Positions are in the displayed camera image. Angle estimates are sufficient:
the intrinsic solver estimates each board pose from its detected corners.
"""

INTRINSICS_GUIDE_20 = (
    ("CENTER", "Front-facing, medium size"),
    ("LEFT", "Front-facing, medium size"),
    ("RIGHT", "Front-facing, medium size"),
    ("UPPER CENTER", "Front-facing, medium size"),
    ("LOWER CENTER", "Front-facing, medium size"),
    ("UPPER LEFT", "Front-facing, medium size"),
    ("UPPER RIGHT", "Front-facing, medium size"),
    ("LOWER LEFT", "Front-facing, medium size"),
    ("LOWER RIGHT", "Front-facing, medium size"),
    ("CENTER / CLOSER", "Front-facing, larger but still sharp; do not zoom"),
    ("CENTER / FARTHER", "Front-facing, smaller but corners still clear"),
    ("CENTER", "Left edge toward camera ~30 deg"),
    ("CENTER", "Right edge toward camera ~30 deg"),
    ("CENTER", "Top edge toward camera ~30 deg"),
    ("CENTER", "Bottom edge toward camera ~30 deg"),
    ("LEFT", "Left edge toward camera ~25 deg"),
    ("RIGHT", "Right edge toward camera ~25 deg"),
    ("UPPER CENTER", "Top edge toward camera ~25 deg"),
    ("LOWER CENTER", "Bottom edge toward camera ~25 deg"),
    ("CENTER", "Top + left edges toward camera ~20 deg each"),
)
