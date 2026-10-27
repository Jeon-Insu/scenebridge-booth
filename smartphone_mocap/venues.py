# smartphone_mocap/venues.py
"""장소 프리셋(config/venues.json): 배경 USD, 테이블 자리(오프셋), 회전."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "venues.json"
SETTING = "--/exts/gist.tabletop_twin/"


class VenueError(ValueError):
    pass


@dataclass(frozen=True)
class Venue:
    name: str
    background: Path
    offset_m: tuple[float, float, float]
    yaw_deg: float
    note: str = ""
    camera: dict | None = None     # {"eye": [x, y, z], "target": [x, y, z]} in table coordinates
    cutaway: Path | None = None    # school-to-table fly-in plan (tools/make_cutaway.py)


def table_to_world(venue: Venue, point) -> tuple[float, float, float]:
    """A table-frame point (origin at the front-left corner, z up from the table top) in the background."""
    import math
    c, s = math.cos(math.radians(venue.yaw_deg)), math.sin(math.radians(venue.yaw_deg))
    x, y, z = (float(v) for v in point)
    ox, oy, oz = venue.offset_m
    return (ox + c * x - s * y, oy + s * x + c * y, oz + z)


def load_venue(name: str, path=None, *, check_files: bool = True) -> Venue:
    """check_files=False reads the preset without the background USD (the tracker needs only the yaw)."""
    path = Path(path or DEFAULT_PATH)
    presets = json.loads(path.read_text(encoding="utf-8"))["venues"]
    if name not in presets:
        raise VenueError(f"장소 '{name}'이 없습니다. 고를 수 있는 장소: {', '.join(sorted(presets))}")
    raw = presets[name]
    background = Path(raw["background"])
    if not background.is_absolute():
        background = (ROOT / background).resolve()
    if check_files and not background.is_file():
        raise VenueError(f"장소 '{name}'의 배경 파일이 없습니다: {background}")
    import re
    for source in ([] if not check_files else re.findall(r"@([^@]+)@", background.read_text(encoding="utf-8", errors="ignore").split("\n)\n")[0])):
        if not Path(source).is_absolute():
            source = str((background.parent / source).resolve())
        if not Path(source).is_file():
            raise VenueError(f"장소 '{name}'의 원본 USD가 없습니다: {source}")
    cutaway = None
    if raw.get("cutaway"):
        cutaway = Path(raw["cutaway"])
        cutaway = cutaway if cutaway.is_absolute() else (ROOT / cutaway).resolve()
    return Venue(name, background, tuple(float(v) for v in raw["offset_m"]), float(raw.get("yaw_deg", 0.0)),
                 str(raw.get("note", "")), raw.get("camera"), cutaway)


def isaac_args(venue: Venue, table_size_m=None) -> list[str]:
    x, y, z = venue.offset_m
    args = [f"{SETTING}background_stage_path={venue.background}", f"{SETTING}world_offset_x={x:g}",
            f"{SETTING}world_offset_y={y:g}", f"{SETTING}world_offset_z={z:g}",
            f"{SETTING}table_yaw_deg={venue.yaw_deg:g}"]   # until the tracker's first message carries it
    if table_size_m is not None:  # booth.json rig.workspace_size_m: the twin draws a table of this size
        args += [f"{SETTING}table_length_m={float(table_size_m[0]):g}", f"{SETTING}table_width_m={float(table_size_m[1]):g}"]
    if venue.cutaway is not None:
        args.append(f"{SETTING}cutaway_path={venue.cutaway}")
    if venue.camera:
        for key in ("eye", "target"):
            for axis, value in zip("xyz", table_to_world(venue, venue.camera[key])):
                args.append(f"{SETTING}camera_{key}_{axis}={value:g}")
    return args
