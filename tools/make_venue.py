"""동신여고 AI 창의교실 배경을 만든다: 액자가 가장 많은 벽 앞에 테이블 자리를 잡고, 겹치는 가구를 숨긴다.

  ~/Desktop/isaac-sim-standalone-6.0.1-linux-aarch64/python.sh tools/make_venue.py [--distance 1.6] [--margin 0.6]

원본 패키지(Dongshin_Final)는 참조만 하고 고치지 않는다. 결과: usd/venues/dongshin_ai_classroom.usda
(sublayer + 비활성화 over), 그리고 config/venues.json에 넣을 오프셋·yaw를 출력한다.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartphone_mocap.venue_layout import overlapping, table_pose_facing_wall  # noqa: E402

SOURCE = ROOT / "usd/venues/source/Dongshin_Final/Dongshin_Final.usda"
ROOM = "/Dongshin_Main/Rooms/AI_Creative_Classroom"
OTHER_ROOMS = ("Teachers_Office",)
TABLE_SIZE = (1.8, 0.75)   # default booth table; --table to change
TABLE_HEIGHT_M = 0.75
OUT = ROOT / "usd/venues/dongshin_ai_classroom.usda"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance", type=float, default=1.6, help="벽에서 테이블 중심까지(m)")
    p.add_argument("--margin", type=float, default=1.0, help="테이블 둘레에서 가구를 숨길 여유(m)")
    p.add_argument("--table", type=float, nargs=2, default=TABLE_SIZE, metavar=("L", "W"), help="테이블 가로·세로(m)")
    args = p.parse_args()

    stage = Usd.Stage.Open(str(SOURCE))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def box(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return None if r.IsEmpty() else (np.array(r.GetMin()), np.array(r.GetMax()))

    room_lo, room_hi = box(stage.GetPrimAtPath(ROOM))
    room_centre = (room_lo[:2] + room_hi[:2]) / 2

    # Frames hang on several walls: group them by their thin axis and wall coordinate, keep the biggest wall.
    walls = defaultdict(list)
    for frame in stage.GetPrimAtPath(f"{ROOM}/InteriorAssets/PhotoFrames").GetChildren():
        lo, hi = box(frame)
        size = hi - lo
        axis = int(np.argmin(size[:2]))
        walls[(axis, round(float((lo[axis] + hi[axis]) / 2), 1))].append((lo, hi))
    (axis, coord), frames = max(walls.items(), key=lambda item: len(item[1]))
    lo = np.min([f[0] for f in frames], axis=0)
    hi = np.max([f[1] for f in frames], axis=0)
    centre = (lo[:2] + hi[:2]) / 2
    normal = np.zeros(2)
    normal[axis] = 1.0 if room_centre[axis] > coord else -1.0

    # The floor under the frames: the lowest furniture base in the room.
    furniture = {}
    for group in ("Furniture", "Books"):
        for prim in stage.GetPrimAtPath(f"{ROOM}/InteriorAssets/{group}").GetChildren():
            b = box(prim)
            if b is not None:
                furniture[str(prim.GetPath())] = (b[0][0], b[0][1], b[1][0], b[1][1], b[0][2])
    floor_z = min(v[4] for v in furniture.values())

    size = tuple(args.table)
    (ox, oy), yaw = table_pose_facing_wall(tuple(centre), tuple(normal), args.distance, size)
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    R = np.array([[c, -s], [s, c]])
    corners = [tuple(np.array([ox, oy]) + R @ np.array(v))
               for v in ((0, 0), (size[0], 0), (size[0], size[1]), (0, size[1]))]
    hidden = overlapping({k: v[:4] for k, v in furniture.items()}, corners, args.margin)

    lines = ["#usda 1.0", "(", '    defaultPrim = "Dongshin_Main"', "    metersPerUnit = 1",
             f"    subLayers = [@./{SOURCE.relative_to(OUT.parent).as_posix()}@]", '    upAxis = "Z"', ")", "",
             'over "Dongshin_Main"', "{", '    over "Rooms"', "    {"]
    for room in OTHER_ROOMS:
        lines += [f'        over "{room}" (', "            active = false", "        )", "        {", "        }"]
    tree: dict = {}
    for path in hidden:
        parts = path.split("/")[3:]            # below /Dongshin_Main/Rooms
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    def emit(node, depth):
        for name, child in node.items():
            pad = "    " * depth
            if child:
                lines.extend([f'{pad}over "{name}"', f"{pad}{{"])
                emit(child, depth + 1)
                lines.append(f"{pad}}}")
            else:
                lines.extend([f'{pad}over "{name}" (', f"{pad}    active = false", f"{pad})", f"{pad}{{", f"{pad}}}"])
    emit(tree, 2)
    lines += ["    }", "}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"액자 벽: 축 {'xy'[axis]}={coord}, 액자 {len(frames)}개, 법선 {normal.tolist()}")
    print(f"바닥 z {floor_z:.2f}, 테이블 중심 {np.round(centre + normal * args.distance, 2).tolist()}")
    print(f"offset_m [{ox:.3f}, {oy:.3f}, {floor_z + TABLE_HEIGHT_M:.3f}]  yaw_deg {yaw:.1f}")
    print(f"숨긴 프림 {len(hidden)}개: {[h.split('/')[-1] for h in hidden]}")
    print(OUT)


if __name__ == "__main__":
    main()
