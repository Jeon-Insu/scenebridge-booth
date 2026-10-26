# smartphone_mocap/recognition/types.py
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

POSES = ("upright", "lying", "upside_down")
# Facing is undefined for these (2026-10-25 spec: excluded from orientation scoring).
SYMMETRIC_CLASSES = frozenset({"cup", "potted plant", "vase", "bowl", "bottle"})
DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets/props"


@dataclass(frozen=True)
class PropRef:
    asset_id: str
    name_ko: str
    detector_class: str
    height_m: float
    views: tuple[Path, ...]
    count: int = 1
    symmetric: bool = False


@dataclass(frozen=True)
class Registry:
    props: tuple[PropRef, ...]

    def ids(self) -> list[str]:
        return [p.asset_id for p in self.props]

    def get(self, asset_id: str) -> PropRef:
        for prop in self.props:
            if prop.asset_id == asset_id:
                return prop
        raise KeyError(asset_id)

    def without(self, asset_id: str) -> "Registry":
        return replace(self, props=tuple(p for p in self.props if p.asset_id != asset_id))

    @classmethod
    def from_catalog(cls, catalog_dir: Path, assets_root: Path = DEFAULT_ASSETS_ROOT) -> "Registry":
        payload = json.loads((Path(catalog_dir) / "catalog.json").read_text(encoding="utf-8"))
        props = []
        for entry in payload["assets"]:
            views = tuple(sorted((Path(assets_root) / entry["asset_id"] / "views").glob("view_*.png")))
            props.append(PropRef(entry["asset_id"], entry["display_name_ko"], entry["detector_class"],
                                 float(entry["height_m"]), views,
                                 symmetric=entry["detector_class"] in SYMMETRIC_CLASSES))
        return cls(tuple(props))


@dataclass(frozen=True)
class Identity:
    asset_id: str | None
    probs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Orientation:
    pose: str | None
    yaw_deg: float | None
    probs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Crop:
    camera_id: str
    image: np.ndarray
    mask: np.ndarray | None
    hint: str | None
    score: float
