import json

import cv2
import numpy as np

from smartphone_mocap.synthdata.coco_dataset import merge_scene, package_dataset, validate_dataset

CATEGORIES = [{"id": 1, "name": "toy robot", "supercategory": "prop"},
              {"id": 2, "name": "cup", "supercategory": "prop"}]


def _raw_scene(root, name, images=2):
    scene = root / name
    scene.mkdir(parents=True)
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    for i in range(images):
        file = f"rgb_{i:04d}.png"
        cv2.imwrite(str(scene / file), np.full((72, 128, 3), 120, np.uint8))
        coco["images"].append({"id": i + 7, "file_name": file, "width": 128, "height": 72})
        coco["annotations"].append({"id": 100 + i, "image_id": i + 7, "category_id": 1,
                                    "bbox": [10, 10, 20, 15], "area": 300, "iscrowd": 0,
                                    "segmentation": [[10, 10, 30, 10, 30, 25, 10, 25]]})
    (scene / "coco.json").write_text(json.dumps(coco))
    return scene


def test_merge_scene_resolves_images_and_keeps_annotations(tmp_path):
    images, annotations, categories = merge_scene(_raw_scene(tmp_path, "scene_00000"))
    assert len(images) == 2 and len(annotations) == 2
    assert {a["image_index"] for a in annotations} == {0, 1}
    assert set(categories) == {1, 2}


def test_package_splits_by_scene_and_validates(tmp_path):
    raw = tmp_path / "raw"
    for k in range(10):
        _raw_scene(raw, f"scene_{k:05d}")
    calls = []
    def degrade_fn(image, rng):
        calls.append(1)
        return image, {"jpeg": 80}
    counts = package_dataset(raw, tmp_path / "out", valid_fraction=0.2, seed=1, degrade_fn=degrade_fn)
    assert counts == {"train": 16, "valid": 4, "test": 0}
    assert len(calls) == 20
    assert validate_dataset(tmp_path / "out") == []
    train = json.loads((tmp_path / "out/train/_annotations.coco.json").read_text())
    valid = json.loads((tmp_path / "out/valid/_annotations.coco.json").read_text())
    scenes = lambda data: {img["file_name"].split("_rgb")[0] for img in data["images"]}
    assert not scenes(train) & scenes(valid)          # a scene never straddles the split
    test = json.loads((tmp_path / "out/test/_annotations.coco.json").read_text())
    assert test["images"] == [] and test["categories"] == train["categories"]


def test_validate_reports_missing_image(tmp_path):
    raw = tmp_path / "raw"
    for k in range(3):
        _raw_scene(raw, f"scene_{k:05d}")
    package_dataset(raw, tmp_path / "out", valid_fraction=0.34, seed=0)
    first = json.loads((tmp_path / "out/train/_annotations.coco.json").read_text())["images"][0]
    (tmp_path / "out/train" / first["file_name"]).unlink()
    problems = validate_dataset(tmp_path / "out")
    assert any(first["file_name"] in problem for problem in problems)


def test_dataset_classes_follow_category_id_order(tmp_path):
    import json
    from smartphone_mocap.synthdata.coco_dataset import dataset_classes
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps({
        "images": [], "annotations": [],
        "categories": [{"id": 2, "name": "stainless_mug"}, {"id": 1, "name": "green_robot"}]}))
    # RF-DETR numbers classes 0.. in category-id order; the runtime reads names by that index.
    assert dataset_classes(tmp_path) == ["green_robot", "stainless_mug"]
