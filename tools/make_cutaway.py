"""동신여고 '학교에서 테이블로' 연출 계획을 만듭니다: 카메라 경로 + 층별로 숨길 프림 (Isaac python).

  $ISAAC_SIM_DIR/python.sh tools/make_cutaway.py

결과: usd/venues/dongshin_ai_classroom_cutaway.json (config/venues.json의 "cutaway"가 가리킴).
트윈의 "학교에서 테이블로" 버튼이 이 파일대로 카메라를 움직이며 위층부터 차례로 숨기고(인형의 집),
"처음으로" 버튼이 모두 되살립니다. 원본 건물 파일은 고치지 않습니다(화면에서 보이기만 바꿈).

- 층 묶음(Floor_03~06, 지붕, 교무실)은 통째로 숨긴다.
- 여러 층에 걸친 외벽·부지·보수·계단·복도 묶음은 메시마다 '그 층 높이보다 완전히 위'일 때만 숨긴다.
- 한 층을 숨길 때 바로 아래층의 천장·조명·보와 복도 천장에 붙은 것도 함께 숨겨 그 층 내부가 보이게 한다.
- 2층(AI 창의교실)은 3층과 함께 교실 천장에 매달린 물체(에어컨 등: 윗면이 천장 35cm 이내, 아랫면이 천장 1.5m 이내)도 숨긴다.
"""
import json
import sys
from pathlib import Path

from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartphone_mocap.cutaway import above_cut, ceiling_attached  # noqa: E402
from smartphone_mocap.venues import load_venue, table_to_world  # noqa: E402

VENUE = "dongshin_ai_classroom"
OUT = ROOT / "usd/venues/dongshin_ai_classroom_cutaway.json"
M = "/Dongshin_Main"
CEILING_Z = 7.05          # 2층(AI 창의교실) 천장 높이
# (시각 s, 이름, 통째로 숨길 프림, 이 높이보다 완전히 위인 공용 메시도 숨김, 천장을 함께 걷을 아래층과 그 천장 높이)
FLOOR_STAGES = [
    (4.0, "지붕·6층 + 5층 천장", [f"{M}/Floors/Floor_06", f"{M}/Common/Roof"], 17.0, ("Floor_05", 16.95)),
    (5.3, "5층 + 4층 천장", [f"{M}/Floors/Floor_05"], 13.65, ("Floor_04", 13.65)),
    (6.6, "4층 + 3층 천장", [f"{M}/Floors/Floor_04", f"{M}/Rooms/Homebase_F04", f"{M}/Rooms/Corridor_F04"], 10.35,
     ("Floor_03", 10.35)),
    (7.9, "3층·교무실 + 2층 천장", [f"{M}/Floors/Floor_03", f"{M}/Rooms/Teachers_Office"], 6.95, ("Floor_02", CEILING_Z)),
]
CEILING_GROUPS = ("Ceilings", "Lighting", "Beams")     # under each Floor_0N: taken whole
CEILING_SCAN = ("Corridors",)                         # under each Floor_0N: meshes hung from the ceiling
# Furnished rooms other than the AI classroom: their meshes go with the ceiling rule of the floor they sit on.
# Restored_Interiors spans every floor, so its meshes are also cut by height like SHARED.
ROOMS_ON_FLOOR = {"Floor_03": ["Teachers_Office"], "Floor_04": ["Homebase_F04", "Corridor_F04"]}
RESTORED = f"{M}/Rooms/Restored_Interiors"
ASSET_GROUPS = {"Furniture", "Books", "BooksAndFiles", "PhotoFrames"}   # groups under InteriorAssets
SHARED = [RESTORED, f"{M}/Common/Exterior", f"{M}/Common/CommonExteriorAndSite", f"{M}/BuildingRepairs/Geometry",
          f"{M}/Common/Stairs", f"{M}/Common/Corridors", f"{M}/Common/WestRoundStairCore"]
# 카메라: 잔디밭 쪽(남, -Y)이 정면이고 지형은 동쪽이 높다. 동쪽 끝 가까이에서 건물 전체가 사선으로 보이게 시작해,
# 건물 정면을 따라 서쪽 교실 쪽으로 붙어 이동한다. 층이 걷히는 동안 약 40~60°로 내려다봐 지나가는 구간마다
# 각 층의 방 배치가 보이고, 마지막에 천장이 열린 교실 위에서 테이블(장소 프리셋 시점)로 내려간다.
KEYS = [(0.0, [150.0, -40.0, 46.0], [85.0, 8.0, 8.0]),
        (4.0, [118.0, -24.0, 44.0], [95.0, 8.0, 10.0]),
        (6.5, [90.0, -20.0, 40.0], [70.0, 6.0, 8.0]),
        (9.0, [64.0, -13.0, 30.0], [50.0, 4.0, 6.0]),
        (11.5, [50.5, -1.0, 15.5], [46.0, 3.5, 4.6])]
FINAL_T = 14.5


def main() -> int:
    venue = load_venue(VENUE)
    stage = Usd.Stage.Open(str(venue.background))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def zrange(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return None if r.IsEmpty() else (r.GetMin()[2], r.GetMax()[2])

    meshes = []
    for group in SHARED:
        root = stage.GetPrimAtPath(group)
        if not root:
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsActive() and prim.IsA(UsdGeom.Mesh):
                z = zrange(prim)
                if z:
                    meshes.append((str(prim.GetPath()), *z))

    def ceiling_of(floor, ceiling_z):
        """The floor's ceilings, lights and beams, and corridor meshes hung from that ceiling."""
        base = f"{M}/Floors/{floor}"
        paths = [f"{base}/{g}" for g in CEILING_GROUPS if stage.GetPrimAtPath(f"{base}/{g}")]
        roots = [f"{base}/{g}" for g in CEILING_SCAN] + [f"{M}/Rooms/{r}" for r in ROOMS_ON_FLOOR.get(floor, [])]
        roots.append(RESTORED)
        for root_path in roots:
            root = stage.GetPrimAtPath(root_path)
            for prim in (Usd.PrimRange(root) if root else []):
                z = zrange(prim) if prim.IsA(UsdGeom.Mesh) else None
                if z and ceiling_attached(*z, ceiling_z=ceiling_z):
                    paths.append(str(prim.GetPath()))
        return paths

    stages, taken = [], set()
    for at, label, groups, cut, (below, below_ceiling) in FLOOR_STAGES:   # highest cut first: each mesh goes once
        paths = [p for p in groups if stage.GetPrimAtPath(p)]
        paths += [p for p in above_cut([m for m in meshes if m[0] not in taken], cut)]
        paths += ceiling_of(below, below_ceiling)
        taken.update(paths)
        stages.append({"at": at, "label": label, "paths": paths})

    ceiling = []
    room = f"{M}/Rooms/AI_Creative_Classroom"
    # Every item under InteriorAssets (loose items like the ceiling air conditioners, and those in Furniture/Books/
    # PhotoFrames groups), plus the room shell's meshes. The ceiling rule keeps wall photos and walls.
    assets = stage.GetPrimAtPath(f"{room}/InteriorAssets")
    candidates = []
    for child in assets.GetChildren():
        candidates += list(child.GetChildren()) if child.GetName() in ASSET_GROUPS else [child]
    candidates += [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{room}/Architecture")) if p.IsA(UsdGeom.Mesh)]
    for prim in candidates:
        z = zrange(prim)
        if z and ceiling_attached(*z, ceiling_z=CEILING_Z):
            ceiling.append(str(prim.GetPath()))
    stages[-1]["paths"] += [p for p in ceiling if stage.GetPrimAtPath(p)]   # with the 3rd floor and the 2nd-floor ceiling

    eye = table_to_world(venue, venue.camera["eye"])
    target = table_to_world(venue, venue.camera["target"])
    keys = [{"t": t, "eye": e, "target": g} for t, e, g in KEYS]
    keys.append({"t": FINAL_T, "eye": list(eye), "target": list(target)})
    OUT.write_text(json.dumps({"venue": VENUE, "keyframes": keys, "stages": stages}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    for s in stages:
        print(f"{s['at']:>4}s  {s['label']}: {len(s['paths'])}개")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
