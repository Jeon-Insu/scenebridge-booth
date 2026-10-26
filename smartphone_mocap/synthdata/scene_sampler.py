"""Deterministic scene descriptions for synthetic prop rendering.

Everything random about a synthetic image is decided here, in plain Python, and
written to JSON. The Isaac script only replays these descriptions, so a dataset can
be regenerated exactly and the sampling is testable without Isaac.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..calibration import look_at_world_to_camera

TABLE_SIZE_M = (1.8, 0.75)
EDGE_MARGIN_M = 0.10
MIN_SEPARATION_M = 0.08
POSE_WEIGHTS = {"upright": 0.5, "lying": 0.4, "upside_down": 0.1}
# Lying falls onto one of four sides: roll about the object's x or y axis by ±90° (2026-10-25:
# the real mascot lay on its back, a side the single x-roll never showed the detector).
LYING_TURNS = (("x", 90.0), ("x", -90.0), ("y", 90.0), ("y", -90.0))


def _pose_turn(rng, pose):
    if pose == "lying":
        return LYING_TURNS[int(rng.integers(len(LYING_TURNS)))]
    return ("x", 180.0 if pose == "upside_down" else 0.0)
# Light ranges (Isaac path tracer units). Chosen so raw renders match the venue
# recordings' mean grey level (~100-130 on 9/10); the earlier 500-5000 / 100-1500
# put the median render at 193 and a third of them near white.
# Camera layouts. "venue" jitters the calibrated cameras of one booth; "random" places a
# rig of the same cameras around a table of random size, so a model does not learn one
# booth's viewpoints; "mixed" picks venue for VENUE_SHARE of scenes and random otherwise.
CAMERA_LAYOUTS = ("venue", "random", "mixed")
VENUE_SHARE = 0.3
TABLE_SIZE_RANGE_M = ((1.2, 2.4), (0.6, 1.2))
RIG_DISTANCE_M = (0.6, 2.2)     # horizontal, from the table centre
RIG_HEIGHT_M = (0.3, 1.8)       # above the table top
RIG_ZOOM = (0.75, 1.3)          # focal length relative to the calibrated camera
RIG_AIM_JITTER_M = 0.05         # cameras aim at the table centre, within this
# Distractors (NVIDIA sample assets, unlabelled): what stands around a real booth.
FLOOR_Z_M = -0.75               # the table top is z = 0
CAMERA_CLEARANCE_M = 0.4        # nothing stands right in front of a lens
ON_TABLE_MAX_HEIGHT_M = {"electronics": 0.45, "clutter": 0.3}   # taller would hide the props
DISTRACTOR_COUNTS = {"person": (0, 3), "furniture": (0, 3), "electronics": (0, 2), "clutter": (0, 3)}
SUN_INTENSITY = (200.0, 1800.0)
AMBIENT_INTENSITY = (30.0, 300.0)


@dataclass(frozen=True)
class PropPlacement:
    asset_id: str
    detector_class: str
    usd_path: str
    x_m: float
    y_m: float
    yaw_deg: float
    roll_deg: float
    pose: str
    scale: float = 1.0      # uniform scale bringing the USD to its believable height
    train_class: str = ""   # the label the detector learns; empty means detector_class
    roll_axis: str = "x"    # the object axis roll_deg turns about ("x" or "y")

    def __post_init__(self):
        if not self.train_class:
            object.__setattr__(self, "train_class", self.detector_class)


@dataclass(frozen=True)
class DistractorPlacement:
    """An unlabelled object that is not a prop: a person, a chair, a TV, a can."""
    usd_path: str
    kind: str
    x_m: float
    y_m: float
    surface_z_m: float          # what it rests on: the floor or the table top
    yaw_deg: float              # the direction it faces, in the table frame
    meters_per_unit: float = 1.0
    up_axis: str = "Z"
    forward_yaw_deg: float = 0.0  # which way the asset faces before yaw (index metadata)


@dataclass(frozen=True)
class CameraView:
    camera_id: str
    world_to_camera: list   # 4x4, OpenCV convention (x right, y down, z forward)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Lighting:
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_intensity: float
    color_temperature_k: float
    ambient_intensity: float


@dataclass(frozen=True)
class SceneSpec:
    scene_id: int
    seed: int
    props: tuple
    cameras: tuple
    lighting: Lighting
    table_texture: str
    floor_texture: str
    wall_texture: str
    table_size_m: tuple = (1.8, 0.75)
    distractors: tuple = ()


def catalog_entries(catalog_dir: str | Path) -> list[dict]:
    root = Path(catalog_dir).resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    return [{"asset_id": e["asset_id"], "detector_class": e["detector_class"],
             "usd_path": str((root / e["usd_uri"]).resolve()),
             "height_m": float(e.get("height_m") or 0.0)} for e in catalog["assets"]]


def load_camera_views(calibration_dir: str | Path) -> list[CameraView]:
    views = []
    for path in sorted(Path(calibration_dir).glob("cam_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        k = doc["camera_matrix"]
        width, height = doc["image_size"]
        views.append(CameraView(doc["camera_id"], doc["world_to_camera"], float(k[0][0]),
                                float(k[1][1]), float(k[0][2]), float(k[1][2]), int(width), int(height)))
    if not views:
        raise FileNotFoundError(f"no cam_*.json in {calibration_dir}")
    return views


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * k @ k


def _projects_inside(view: CameraView, point: np.ndarray) -> bool:
    w2c = np.asarray(view.world_to_camera, float)
    p = w2c[:3, :3] @ point + w2c[:3, 3]
    if p[2] <= 0:
        return False
    u = view.fx * p[0] / p[2] + view.cx
    v = view.fy * p[1] / p[2] + view.cy
    return 0 <= u < view.width and 0 <= v < view.height


def _perturbed(view: CameraView, rng: np.random.Generator, max_shift_m: float,
               max_rotation_deg: float) -> CameraView:
    w2c = np.asarray(view.world_to_camera, float)
    rotation, translation = w2c[:3, :3], w2c[:3, 3]
    centre = -rotation.T @ translation
    direction = rng.normal(size=3)
    shift = direction / np.linalg.norm(direction) * rng.uniform(0.0, max_shift_m)
    delta = _rotation(rng.normal(size=3), np.radians(rng.uniform(0.0, max_rotation_deg)))
    new_rotation = delta @ rotation
    out = np.eye(4)
    out[:3, :3] = new_rotation
    out[:3, 3] = -new_rotation @ (centre + shift)
    return CameraView(view.camera_id, out.tolist(), view.fx, view.fy, view.cx, view.cy,
                      view.width, view.height)


def perturb_camera(view: CameraView, rng: np.random.Generator, target: np.ndarray,
                   max_shift_m: float = 0.15, max_rotation_deg: float = 10.0,
                   attempts: int = 20) -> CameraView:
    """Move and tilt a camera like a re-installation would, keeping the table in view."""
    for _ in range(attempts):
        candidate = _perturbed(view, rng, max_shift_m, max_rotation_deg)
        if _projects_inside(candidate, np.asarray(target, float)):
            return candidate
    return view


CLASS_UNITS = ("detector", "asset")


def training_categories(catalog: list[dict], class_unit: str = "detector") -> list[str]:
    """What the detector learns: COCO-style classes, or one class per registered prop."""
    key = {"detector": "detector_class", "asset": "asset_id"}[class_unit]
    return sorted({entry[key] for entry in catalog})


def random_rig(rng: np.random.Generator, templates: list[CameraView],
               table_size: tuple[float, float], attempts: int = 20) -> tuple:
    """Cameras spread around the table, each aimed at its centre.

    Ids, image size and principal point come from the calibrated templates; position,
    roll and focal length are random, and the aim is the table centre within
    RIG_AIM_JITTER_M. Azimuths are spread evenly with jitter so no
    side of the table is left unwatched.
    """
    width, depth = table_size
    centre = np.array([width / 2, depth / 2, 0.0])
    base_azimuth = rng.uniform(0.0, 2 * np.pi)
    views = []
    for index, template in enumerate(templates):
        azimuth = base_azimuth + 2 * np.pi * index / len(templates) + np.radians(rng.uniform(-25.0, 25.0))
        distance = rng.uniform(*RIG_DISTANCE_M)
        position = centre + np.array([distance * np.cos(azimuth), distance * np.sin(azimuth),
                                      rng.uniform(*RIG_HEIGHT_M)])
        chosen = None
        for _ in range(attempts):
            target = centre + np.array([rng.uniform(-1, 1) * RIG_AIM_JITTER_M,
                                        rng.uniform(-1, 1) * RIG_AIM_JITTER_M, 0.0])
            zoom = rng.uniform(*RIG_ZOOM)
            w2c = look_at_world_to_camera(position, target, roll_deg=float(rng.uniform(-5.0, 5.0)))
            view = CameraView(template.camera_id, w2c.tolist(), template.fx * zoom, template.fy * zoom,
                              template.cx, template.cy, template.width, template.height)
            if _projects_inside(view, centre):
                chosen = view
                break
        if chosen is None:  # aiming at the centre with the calibrated lens always frames it
            chosen = CameraView(template.camera_id, look_at_world_to_camera(position, centre).tolist(),
                                template.fx, template.fy, template.cx, template.cy,
                                template.width, template.height)
        views.append(chosen)
    return tuple(views)


def _scale_for(entry: dict, target_heights: dict[str, float] | None) -> float:
    target = (target_heights or {}).get(entry["asset_id"])
    source = float(entry.get("height_m") or 0.0)
    return float(target) / source if target and source > 0 else 1.0


def _sample_props(rng: np.random.Generator, catalog: list[dict], count: int,
                  table_size: tuple[float, float],
                  target_heights: dict[str, float] | None = None,
                  class_unit: str = "detector", unique: bool = False) -> tuple:
    placed: list[PropPlacement] = []
    poses, weights = list(POSE_WEIGHTS), list(POSE_WEIGHTS.values())
    for _ in range(200):
        if len(placed) >= count:
            break
        entry = catalog[int(rng.integers(len(catalog)))]
        if unique and any(p.asset_id == entry["asset_id"] for p in placed):
            continue
        x = float(rng.uniform(EDGE_MARGIN_M, table_size[0] - EDGE_MARGIN_M))
        y = float(rng.uniform(EDGE_MARGIN_M, table_size[1] - EDGE_MARGIN_M))
        if any(np.hypot(x - p.x_m, y - p.y_m) < MIN_SEPARATION_M for p in placed):
            continue
        pose = str(rng.choice(poses, p=weights))
        axis, roll = _pose_turn(rng, pose)
        placed.append(PropPlacement(entry["asset_id"], entry["detector_class"], entry["usd_path"],
                                    x, y, float(rng.uniform(0.0, 360.0)), roll, pose,
                                    _scale_for(entry, target_heights),
                                    entry["asset_id"] if class_unit == "asset" else entry["detector_class"],
                                    axis))
    return tuple(placed)


def sample_scene(scene_id: int, base_seed: int, catalog: list[dict], views: list[CameraView],
                 textures: dict[str, list[str]], table_size_m: tuple[float, float] = TABLE_SIZE_M,
                 max_props: int = 6, target_heights: dict[str, float] | None = None,
                 class_unit: str = "detector", camera_layout: str = "venue",
                 distractors: list[dict] | None = None, distractor_split: str = "train",
                 unique_props: bool = False, unlisted_on_table: tuple[int, int] = (0, 0)) -> SceneSpec:
    rng = np.random.default_rng([int(base_seed), int(scene_id)])
    layout = camera_layout
    if camera_layout == "mixed":
        layout = "venue" if rng.random() < VENUE_SHARE else "random"
    if camera_layout != "venue" and layout == "random":
        table_size_m = tuple(round(float(rng.uniform(*r)), 3) for r in TABLE_SIZE_RANGE_M)
    table_size_m = tuple(float(v) for v in table_size_m)
    target = np.array([table_size_m[0] / 2, table_size_m[1] / 2, 0.0])
    props = _sample_props(rng, catalog, int(rng.integers(1, max_props + 1)), table_size_m, target_heights,
                          class_unit, unique=unique_props)
    if layout == "random":
        cameras = random_rig(rng, list(views), table_size_m)
    else:
        cameras = tuple(perturb_camera(view, rng, target) for view in views)
    lighting = Lighting(
        sun_azimuth_deg=float(rng.uniform(0.0, 360.0)),
        sun_elevation_deg=float(rng.uniform(15.0, 80.0)),
        sun_intensity=float(rng.uniform(*SUN_INTENSITY)),
        color_temperature_k=float(rng.uniform(3000.0, 7000.0)),
        ambient_intensity=float(rng.uniform(*AMBIENT_INTENSITY)),
    )

    def pick(group: str) -> str:
        options = textures[group]
        return str(options[int(rng.integers(len(options)))])

    table_texture, floor_texture, wall_texture = pick("table"), pick("floor"), pick("wall")
    # Drawn last, so a scene without distractors is the same scene as before.
    placed = (sample_distractors(rng, distractors, distractor_split, table_size_m, cameras, props)
              if distractors else ())
    if unlisted_on_table[1] > 0 and distractors:
        placed = (*placed, *_sample_unlisted(rng, distractors, distractor_split, table_size_m, props,
                                             unlisted_on_table))
    return SceneSpec(int(scene_id), int(base_seed), props, cameras, lighting,
                     table_texture, floor_texture, wall_texture, table_size_m, placed)


def _sample_unlisted(rng, index, split, table_size, props, count_range) -> tuple:
    """Small table-top clutter labelled "unlisted": objects the booth list does not contain."""
    pool = [e for e in index if e["kind"] == "clutter" and e.get("split", "train") == split
            and float(e["size_m"][2]) <= ON_TABLE_MAX_HEIGHT_M.get("clutter", 0.0)]
    extra: list[DistractorPlacement] = []
    if not pool:
        return ()
    for _ in range(int(rng.integers(count_range[0], count_range[1] + 1))):
        for _attempt in range(30):
            entry = pool[int(rng.integers(len(pool)))]
            x = float(rng.uniform(EDGE_MARGIN_M, table_size[0] - EDGE_MARGIN_M))
            y = float(rng.uniform(EDGE_MARGIN_M, table_size[1] - EDGE_MARGIN_M))
            if any(np.hypot(x - p.x_m, y - p.y_m) < MIN_SEPARATION_M for p in (*props, *extra)):
                continue
            extra.append(DistractorPlacement(str(entry["usd"]), "unlisted", x, y, 0.0,
                                             float(rng.uniform(0.0, 360.0)),
                                             float(entry.get("meters_per_unit", 1.0)),
                                             str(entry.get("up_axis", "Z")),
                                             float(entry.get("forward_yaw_deg", 0.0))))
            break
    return tuple(extra)


def _around_table(rng, table_size, near_m, far_m):
    """A floor point outside the table, between near_m and far_m from its edge."""
    w, d = table_size
    gap = rng.uniform(near_m, far_m)
    side = int(rng.integers(4))
    along = rng.uniform(-0.5, 1.5)
    if side == 0:
        return along * w, -gap
    if side == 1:
        return along * w, d + gap
    if side == 2:
        return -gap, along * d
    return w + gap, along * d


def sample_distractors(rng: np.random.Generator, index: list[dict], split: str, table_size,
                       cameras, props) -> tuple:
    pool = {kind: [e for e in index if e["kind"] == kind and e.get("split", "train") == split]
            for kind in DISTRACTOR_COUNTS}
    w, d = table_size
    centre = np.array([w / 2, d / 2])
    lenses = [(-np.asarray(v.world_to_camera)[:3, :3].T @ np.asarray(v.world_to_camera)[:3, 3])[:2]
              for v in cameras]
    placed: list[DistractorPlacement] = []
    for kind, (low, high) in DISTRACTOR_COUNTS.items():
        if not pool[kind]:
            continue
        for _ in range(int(rng.integers(low, high + 1))):
            entry = pool[kind][int(rng.integers(len(pool[kind])))]
            fits_on_table = float(entry["size_m"][2]) <= ON_TABLE_MAX_HEIGHT_M.get(kind, 0.0)
            if kind == "clutter" and not fits_on_table:
                continue
            for _attempt in range(30):
                surface = FLOOR_Z_M
                if kind == "person":
                    x, y = _around_table(rng, table_size, 0.15, 1.2)
                elif kind == "furniture":
                    x, y = _around_table(rng, table_size, 0.2, 2.0)
                elif kind == "electronics" and rng.random() < 0.3 and fits_on_table:
                    edge = 0.08
                    x, y = rng.uniform(edge, w - edge), (edge if rng.random() < 0.5 else d - edge)
                    surface = 0.0
                elif kind == "electronics":
                    x, y = _around_table(rng, table_size, 0.2, 1.5)
                else:
                    x, y = rng.uniform(0.06, w - 0.06), rng.uniform(0.06, d - 0.06)
                    surface = 0.0
                if any(np.hypot(x - lx, y - ly) < CAMERA_CLEARANCE_M for lx, ly in lenses):
                    continue
                if surface == 0.0 and any(np.hypot(x - p.x_m, y - p.y_m) < MIN_SEPARATION_M + 0.05
                                          for p in props):
                    continue
                if kind == "person":
                    facing = np.degrees(np.arctan2(*(centre - (x, y))[::-1]))
                    yaw = facing + rng.uniform(-40.0, 40.0)
                else:
                    yaw = rng.uniform(0.0, 360.0)
                placed.append(DistractorPlacement(
                    str(entry["usd"]), kind, float(x), float(y), float(surface), float(yaw),
                    float(entry.get("meters_per_unit", 1.0)), str(entry.get("up_axis", "Z")),
                    float(entry.get("forward_yaw_deg", 0.0))))
                break
    return tuple(placed)


def write_specs(specs: list[SceneSpec], categories: list[str], path: str | Path) -> None:
    payload = {"schema": "scenebridge.synthetic_specs.v1", "categories": list(categories),
               "scenes": [asdict(spec) for spec in specs]}
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def read_specs(path: str | Path) -> tuple[list[SceneSpec], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = []
    for raw in payload["scenes"]:
        specs.append(SceneSpec(
            raw["scene_id"], raw["seed"],
            tuple(PropPlacement(**p) for p in raw["props"]),
            tuple(CameraView(**c) for c in raw["cameras"]),
            Lighting(**raw["lighting"]),
            raw["table_texture"], raw["floor_texture"], raw["wall_texture"],
            tuple(raw.get("table_size_m", TABLE_SIZE_M)),
            tuple(DistractorPlacement(**item) for item in raw.get("distractors", ()))))
    return specs, list(payload["categories"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, default=Path("assets/props"))
    parser.add_argument("--textures", type=Path, default=Path("assets/textures/manifest.json"))
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heights", type=Path, default=Path("assets/props/synthetic_heights.json"),
                        help="asset_id → believable height in metres, for rendering only")
    parser.add_argument("--class-unit", choices=CLASS_UNITS, default="detector",
                        help="asset: one class per registered prop (stainless_mug vs paper_cup)")
    parser.add_argument("--camera-layout", choices=CAMERA_LAYOUTS, default="venue",
                        help="venue: jitter the calibrated booth cameras; random/mixed: rigs around tables of any size")
    parser.add_argument("--distractors", type=Path,
                        help="assets/nvidia/index.json: unlabelled people, furniture, electronics, clutter")
    parser.add_argument("--distractor-split", choices=("train", "val"), default="train",
                        help="val: assets never used for training, for a held-out synthetic test set")
    parser.add_argument("--table-size", type=float, nargs=2, default=TABLE_SIZE_M, metavar=("L", "W"),
                        help="table length and width in metres (2026-10-25 lab table: 1.25 0.75)")
    parser.add_argument("--unique-props", action="store_true",
                        help="each prop at most once per scene (the booth's one-of-each premise)")
    parser.add_argument("--unlisted-on-table", type=int, nargs=2, default=(0, 0), metavar=("MIN", "MAX"),
                        help="table-top clutter labelled 'unlisted' (needs --distractors)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = catalog_entries(args.catalog_dir)
    manifest = json.loads(args.textures.read_text(encoding="utf-8"))
    textures: dict[str, list[str]] = {"table": [], "floor": [], "wall": []}
    for entry in manifest["textures"]:
        for group in entry["groups"]:
            textures[group].append(str((args.textures.parent / entry["id"]).resolve()))
    views = load_camera_views(args.calibration_dir)
    distractor_index = None
    if args.distractors:
        root = args.distractors.parent.resolve()
        distractor_index = [{**e, "usd": str(root / e["usd"])}
                            for e in json.loads(args.distractors.read_text(encoding="utf-8"))["assets"]]
    heights = (json.loads(args.heights.read_text(encoding="utf-8"))["heights_m"]
               if args.heights.is_file() else None)
    specs = [sample_scene(i, args.seed, catalog, views, textures, table_size_m=tuple(args.table_size),
                          target_heights=heights,
                          class_unit=args.class_unit, camera_layout=args.camera_layout,
                          distractors=distractor_index, distractor_split=args.distractor_split,
                          unique_props=args.unique_props, unlisted_on_table=tuple(args.unlisted_on_table))
             for i in range(args.count)]
    categories = training_categories(catalog, args.class_unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_specs(specs, categories, args.output)
    print(f"{len(specs)}개 장면, 카테고리 {categories} → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
