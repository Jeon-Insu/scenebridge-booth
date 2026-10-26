"""Turn Replicator's per-scene COCO output into the dataset layout RF-DETR trains on.

RF-DETR reads ``train/``, ``valid/`` and ``test/`` folders, each with an
``_annotations.coco.json``. Splitting is by scene so the four views of one scene
never land on both sides. ``test/`` stays empty on purpose: the detector is judged
on real recordings only, never on renders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .degrade import degrade

SPLITS = ("train", "valid", "test")


def _resolve_image(json_dir: Path, scene_dir: Path, file_name: str) -> Path | None:
    direct = json_dir / file_name
    if direct.is_file():
        return direct
    matches = sorted(scene_dir.rglob(Path(file_name).name))
    return matches[0] if matches else None


def merge_scene(scene_dir: str | Path):
    scene_dir = Path(scene_dir)
    images: list[dict] = []
    annotations: list[dict] = []
    categories: dict[int, dict] = {}
    for path in sorted(scene_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"images", "annotations"} <= set(data):
            continue
        for category in data.get("categories", []):
            categories[int(category["id"])] = category
        index_of = {}
        for image in data["images"]:
            file = _resolve_image(path.parent, scene_dir, image["file_name"])
            if file is None:
                continue
            index_of[image["id"]] = len(images)
            images.append({"path": str(file), "width": int(image["width"]),
                           "height": int(image["height"])})
        for annotation in data["annotations"]:
            if annotation["image_id"] in index_of:
                annotations.append({**annotation, "image_index": index_of[annotation["image_id"]]})
    return images, annotations, categories


def package_dataset(raw_dir: str | Path, out_dir: str | Path, *, valid_fraction: float = 0.1,
                    seed: int = 0, degrade_fn=None) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    degrade_fn = degrade_fn or degrade
    scenes = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    rng = np.random.default_rng(seed)
    valid_count = int(round(len(scenes) * valid_fraction)) if len(scenes) > 1 else 0
    valid_scenes = set(rng.choice(len(scenes), size=valid_count, replace=False).tolist())
    datasets = {split: {"images": [], "annotations": [], "categories": {}} for split in SPLITS}
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    for index, scene in enumerate(scenes):
        split = "valid" if index in valid_scenes else "train"
        images, annotations, categories = merge_scene(scene)
        target = datasets[split]
        for dataset in datasets.values():
            dataset["categories"].update(categories)
        new_ids = []
        for k, image in enumerate(images):
            pixels = cv2.imread(image["path"], cv2.IMREAD_COLOR)
            pixels, applied = degrade_fn(pixels, rng)
            file_name = f"{scene.name}_rgb_{k:02d}.jpg"
            cv2.imwrite(str(out_dir / split / file_name), pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_id = len(target["images"]) + 1
            new_ids.append(image_id)
            target["images"].append({"id": image_id, "file_name": file_name,
                                     "width": image["width"], "height": image["height"],
                                     "extra": {"scene": scene.name, "degradation": applied}})
        for annotation in annotations:
            row = {key: value for key, value in annotation.items() if key != "image_index"}
            row["id"] = len(target["annotations"]) + 1
            row["image_id"] = new_ids[annotation["image_index"]]
            target["annotations"].append(row)
    counts = {}
    for split, dataset in datasets.items():
        payload = {"images": dataset["images"], "annotations": dataset["annotations"],
                   "categories": sorted(dataset["categories"].values(), key=lambda c: c["id"])}
        (out_dir / split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
        counts[split] = len(dataset["images"])
    return counts


def dataset_classes(out_dir: str | Path) -> list[str]:
    """Class names in the order RF-DETR numbers them (ascending category id)."""
    data = json.loads((Path(out_dir) / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    return [c["name"] for c in sorted(data["categories"], key=lambda c: int(c["id"]))]


def validate_dataset(out_dir: str | Path) -> list[str]:
    out_dir = Path(out_dir)
    problems = []
    for split in SPLITS:
        path = out_dir / split / "_annotations.coco.json"
        if not path.is_file():
            problems.append(f"{split}: _annotations.coco.json 없음")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        image_ids = {image["id"] for image in data["images"]}
        category_ids = {category["id"] for category in data["categories"]}
        for image in data["images"]:
            if not (out_dir / split / image["file_name"]).is_file():
                problems.append(f"{split}: 이미지 없음 {image['file_name']}")
        for annotation in data["annotations"]:
            if annotation["image_id"] not in image_ids:
                problems.append(f"{split}: 주석 {annotation['id']}의 image_id 없음")
            if annotation["category_id"] not in category_ids:
                problems.append(f"{split}: 주석 {annotation['id']}의 category_id 없음")
            if annotation["bbox"][2] <= 0 or annotation["bbox"][3] <= 0:
                problems.append(f"{split}: 주석 {annotation['id']}의 bbox 크기 0")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--raw", type=Path, required=True)
    package.add_argument("--out", type=Path, required=True)
    package.add_argument("--valid-fraction", type=float, default=0.1)
    package.add_argument("--seed", type=int, default=0)
    check = commands.add_parser("check")
    check.add_argument("--dataset", type=Path, required=True)
    check_raw = commands.add_parser("check-raw")
    check_raw.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "package":
        print(package_dataset(args.raw, args.out, valid_fraction=args.valid_fraction, seed=args.seed))
        problems = validate_dataset(args.out)
    elif args.command == "check":
        problems = validate_dataset(args.dataset)
    else:
        images, annotations, categories = merge_scene(args.scene)
        names = sorted({categories[a["category_id"]]["name"] for a in annotations
                        if a["category_id"] in categories})
        brightness = [float(cv2.imread(i["path"], cv2.IMREAD_GRAYSCALE).mean()) for i in images]
        print(f"이미지 {len(images)}장, 주석 {len(annotations)}개, 카테고리 {names}, "
              f"평균 밝기 {[round(b) for b in brightness]}")
        problems = [] if images and annotations else ["이미지 또는 주석이 없음"]
    for problem in problems:
        print("문제:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
