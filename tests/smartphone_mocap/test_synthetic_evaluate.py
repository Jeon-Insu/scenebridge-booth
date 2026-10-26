import json

import cv2
import numpy as np
import pytest

from smartphone_mocap.types import BoundingBox2D, Object2D

NAMES = {1: "paper_cup", 2: "stainless_mug"}


def _gt(category, box, area=1000):
    return {"category_id": category, "bbox": list(box), "area": area}


def _det(label, xywh, score=.9):
    x, y, w, h = xywh
    return Object2D("image", 0, label, 0, BoundingBox2D(x, y, x + w, y + h), score)


def test_matching_detection_counts_as_found_and_others_as_errors():
    from smartphone_mocap.synthdata.evaluate import score_image
    gts = [_gt(1, (10, 10, 40, 40)), _gt(2, (100, 10, 40, 40)), _gt(1, (200, 10, 40, 40))]
    dets = [_det("paper_cup", (11, 11, 40, 40)),            # found
            _det("paper_cup", (101, 10, 40, 40)),           # the mug called a paper cup
            _det("stainless_mug", (400, 300, 50, 50))]      # nothing there
    result = score_image(gts, dets, NAMES)
    assert result["found"] == {"paper_cup": 1}
    assert result["expected"] == {"paper_cup": 2, "stainless_mug": 1}
    assert result["confusion"] == {("stainless_mug", "paper_cup"): 1}
    assert result["false_positives"] == 1


def test_tiny_ground_truth_is_not_expected():
    from smartphone_mocap.synthdata.evaluate import score_image
    result = score_image([_gt(1, (10, 10, 5, 5), area=20)], [], NAMES, min_area=400)
    assert result["expected"] == {}


def test_evaluate_dataset_summarises_recall_and_false_positives(tmp_path):
    from smartphone_mocap.synthdata.evaluate import evaluate_dataset
    split = tmp_path / "valid"
    split.mkdir()
    for i in range(2):
        cv2.imwrite(str(split / f"im{i}.jpg"), np.zeros((64, 64, 3), np.uint8))
    (split / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "im0.jpg"}, {"id": 2, "file_name": "im1.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 30, 30], "area": 900},
                        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [5, 5, 30, 30], "area": 900}],
        "categories": [{"id": 1, "name": "paper_cup"}]}))

    class Detector:
        calls = 0

        def infer_batch(self, images):
            Detector.calls += 1
            found = [_det("paper_cup", (5, 5, 30, 30))] if Detector.calls == 1 else [_det("zebra", (40, 40, 10, 10))]
            return {"image": found}

    report = evaluate_dataset(tmp_path, Detector(), split="valid", min_area=100)
    assert report["props"]["paper_cup"] == {"expected": 2, "found": 1, "recall": 0.5}
    assert report["false_positives_per_image"] == pytest.approx(0.5)
    assert report["images_with_false_positive"] == 1
