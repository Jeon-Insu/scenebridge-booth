"""Build COCO instance annotations from Replicator's instance-id images.

Replicator's COCO writer leaves ``segmentation`` empty, and its colour masks are per
class, so two elephants that overlap merge into one. BasicWriter instead writes, per
camera, an image of instance ids plus an id → class table. Each id becomes one COCO
annotation here, with a polygon mask, a tight box and its pixel area.

Render labels use underscores ("toy_duck") because the writer cannot handle spaces;
annotations carry the detector's real name ("toy duck").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IGNORED = {"BACKGROUND", "UNLABELLED"}
MIN_AREA_PX = 50


def instance_annotations(ids: np.ndarray, semantics: dict, categories: list[str],
                         min_area: int = MIN_AREA_PX) -> list[dict]:
    category_of = {name.replace(" ", "_"): index + 1 for index, name in enumerate(categories)}
    rows = []
    for raw_id, info in semantics.items():
        label = str(info.get("class", ""))
        if label in IGNORED or label not in category_of:
            continue
        mask = (ids == int(raw_id)).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]
        if not polygons:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        rows.append({"category_id": category_of[label], "bbox": [int(x), int(y), int(w), int(h)],
                     "area": area, "segmentation": polygons, "iscrowd": 0})
    return rows


def convert_scene(scene_dir: str | Path, categories: list[str]) -> Path:
    """Write ``coco.json`` for one rendered scene (every camera, every frame)."""
    scene_dir = Path(scene_dir)
    images, annotations = [], []
    for ids_path in sorted(scene_dir.glob("*/instance_segmentation/instance_segmentation_[0-9]*.png")):
        frame = ids_path.stem.rsplit("_", 1)[-1]
        product = ids_path.parent.parent
        rgb = product / "rgb" / f"rgb_{frame}.png"
        mapping = ids_path.parent / f"instance_segmentation_semantics_mapping_{frame}.json"
        if not rgb.is_file() or not mapping.is_file():
            continue
        ids = cv2.imread(str(ids_path), cv2.IMREAD_UNCHANGED)
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": str(rgb.relative_to(scene_dir)),
                       "width": int(ids.shape[1]), "height": int(ids.shape[0])})
        semantics = json.loads(mapping.read_text(encoding="utf-8"))
        for row in instance_annotations(ids, semantics, categories):
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, **row})
    payload = {"images": images, "annotations": annotations,
               "categories": [{"id": i + 1, "name": name, "supercategory": "prop"}
                              for i, name in enumerate(categories)]}
    out = scene_dir / "coco.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True, help="the specs JSON the scenes were rendered from")
    args = parser.parse_args()
    categories = json.loads(args.specs.read_text(encoding="utf-8"))["categories"]
    scenes = sorted(p for p in args.raw.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    for scene in scenes:
        convert_scene(scene, categories)
    print(f"{len(scenes)}개 장면 변환, 카테고리 {categories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
