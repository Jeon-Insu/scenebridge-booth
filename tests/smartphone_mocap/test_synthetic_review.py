import json

import cv2
import numpy as np


def _dataset(tmp_path, count=4):
    split = tmp_path / "train"
    split.mkdir(parents=True)
    images, annotations = [], []
    for i in range(count):
        cv2.imwrite(str(split / f"img_{i}.jpg"), np.full((72, 128, 3), 90, np.uint8))
        images.append({"id": i + 1, "file_name": f"img_{i}.jpg", "width": 128, "height": 72,
                       "extra": {"scene": f"scene_{i:05d}", "degradation": {"jpeg": 70}}})
        annotations.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [10, 10, 30, 20],
                            "area": 600, "segmentation": [[10, 10, 40, 10, 40, 30, 10, 30]], "iscrowd": 0})
    (split / "_annotations.coco.json").write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "paper_cup"}]}))
    return tmp_path


def test_ground_truth_mask_is_drawn_inside_its_polygon_only():
    from smartphone_mocap.synthdata.review import draw_ground_truth
    image = np.full((72, 128, 3), 90, np.uint8)
    ann = {"category_id": 1, "bbox": [10, 10, 30, 20], "segmentation": [[10, 10, 40, 10, 40, 30, 10, 30]]}
    out = draw_ground_truth(image, [ann], {1: "paper_cup"})
    assert out.shape == image.shape
    assert not np.array_equal(out[20, 25], image[20, 25])    # inside the mask
    assert np.array_equal(out[60, 120], image[60, 120])      # background untouched


def test_review_sheet_tiles_the_requested_number_of_images(tmp_path):
    from smartphone_mocap.synthdata.review import review_sheet
    sheet = review_sheet(_dataset(tmp_path), split="train", count=3, tile=(64, 36), columns=3)
    assert sheet.shape == (36, 192, 3)


def test_review_sheet_adds_a_prediction_column_when_given_a_detector(tmp_path):
    from smartphone_mocap.synthdata.review import review_sheet
    from smartphone_mocap.types import BoundingBox2D, Object2D

    class Detector:
        def infer_batch(self, images):
            return {k: [Object2D(k, 0, "paper_cup", 0, BoundingBox2D(5, 5, 30, 30), .9)] for k in images}

    sheet = review_sheet(_dataset(tmp_path), split="train", count=2, tile=(64, 36), columns=1,
                         detector=Detector())
    assert sheet.shape == (72, 128, 3)        # each row: ground truth | prediction
