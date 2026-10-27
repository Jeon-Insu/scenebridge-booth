"""USD-only adapter for the shared, replaceable geometric placement backend."""
import importlib.util
from itertools import product
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

_path = Path(__file__).resolve().parents[4] / "smartphone_mocap" / "table_contact.py"
_spec = importlib.util.spec_from_file_location("gist_table_contact_math", _path)
_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)
resolve_placement = _math.resolve_placement
inspect_contact = _math.inspect_contact
PLACEMENT_MODES = _math.PLACEMENT_MODES
TABLE_BOUNDS_XY = [0.0, 1.25, 0.0, 0.75]


def set_table_size(length_m, width_m):
    TABLE_BOUNDS_XY[:] = [0.0, float(length_m), 0.0, float(width_m)]


def read_contact_geometry(stage, object_prim):
    # Only the actual asset/proxy: never include ghost markers, labels or guides.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    points = []
    for name in ("Asset", "Proxy"):
        root = stage.GetPrimAtPath(object_prim.GetPath().AppendChild(name))
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            bound = cache.ComputeRelativeBound(prim, object_prim)
            bounds = bound.GetRange()
            if bounds.IsEmpty():
                continue
            low, high = bounds.GetMin(), bounds.GetMax()
            matrix = bound.GetMatrix()
            for side in product((0, 1), repeat=3):
                point = matrix.Transform(Gf.Vec3d(*(high[i] if side[i] else low[i] for i in range(3))))
                points.append(list(point))
    return {"points_local_m": points, "table_bounds_xy_m": list(TABLE_BOUNDS_XY),
            "method": "usd_leaf_bounds_conservative"}
