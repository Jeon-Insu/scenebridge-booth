import json

import cv2
import numpy as np

from smartphone_mocap.synthdata.coco_dataset import merge_scene
from smartphone_mocap.synthdata.instances import convert_scene, instance_annotations

CATEGORIES = ["elephant", "toy duck"]          # ids 1, 2 — same order the renderer uses


def _ids():
    ids = np.ones((72, 128), np.uint16)            # 1 = UNLABELLED (table, floor)
    ids[0:5, 0:5] = 0                              # 0 = BACKGROUND
    ids[10:30, 10:40] = 2                          # first elephant
    ids[40:60, 60:100] = 3                         # second elephant
    ids[5:15, 110:125] = 5                         # duck
    ids[70:72, 0:3] = 6                            # a 6-pixel speck of something unregistered
    return ids


SEMANTICS = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}, "2": {"class": "elephant"},
             "3": {"class": "elephant"}, "5": {"class": "toy_duck"}, "6": {"class": "toy_duck"}}


def test_each_instance_gets_its_own_mask_box_and_area():
    rows = instance_annotations(_ids(), SEMANTICS, CATEGORIES)
    elephants = sorted((r for r in rows if r["category_id"] == 1), key=lambda r: r["bbox"][0])
    assert [r["bbox"] for r in elephants] == [[10, 10, 30, 20], [60, 40, 40, 20]]
    assert [r["area"] for r in elephants] == [600, 800]
    assert all(len(r["segmentation"]) >= 1 and len(r["segmentation"][0]) >= 6 for r in rows)


def test_underscored_render_label_maps_back_to_the_detector_name():
    duck = [r for r in instance_annotations(_ids(), SEMANTICS, CATEGORIES) if r["category_id"] == 2]
    assert len(duck) == 1 and duck[0]["bbox"] == [110, 5, 15, 10]


def test_background_unlabelled_and_specks_are_skipped():
    rows = instance_annotations(_ids(), SEMANTICS, CATEGORIES, min_area=50)
    assert len(rows) == 3                          # two elephants + one duck; the speck is dropped


def _render_product(scene, name, ids):
    (scene / name / "rgb").mkdir(parents=True)
    (scene / name / "instance_segmentation").mkdir()
    cv2.imwrite(str(scene / name / "rgb/rgb_0000.png"), np.full((72, 128, 3), 90, np.uint8))
    cv2.imwrite(str(scene / name / "instance_segmentation/instance_segmentation_0000.png"), ids)
    (scene / name / "instance_segmentation/instance_segmentation_semantics_mapping_0000.json").write_text(
        json.dumps(SEMANTICS))


def test_convert_scene_writes_coco_that_packaging_reads(tmp_path):
    scene = tmp_path / "scene_00000"
    _render_product(scene, "Replicator", _ids())
    _render_product(scene, "Replicator_01", _ids())
    convert_scene(scene, CATEGORIES)
    images, annotations, categories = merge_scene(scene)
    assert len(images) == 2 and len(annotations) == 6
    assert {c["name"] for c in categories.values()} == {"elephant", "toy duck"}
