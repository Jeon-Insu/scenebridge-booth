# tests/smartphone_mocap/test_booth_perception.py
import json
from pathlib import Path

import numpy as np
import pytest

from smartphone_mocap.perception import PerceptionProcessor
from smartphone_mocap.types import BoundingBox2D, FramePacket, Object2D

ROOT = Path(__file__).resolve().parents[2]


class FakeDetector:
    """Two cameras see one giraffe and one rhino-like thing the detector calls 'zebra' twice."""
    def __init__(self, frames):
        self.frames = frames

    def infer_batch(self, images):
        return self.frames


def _config(tmp_path, **object_overrides):
    booth = json.loads((ROOT / "smartphone_mocap/config/booth.json").read_text(encoding="utf-8"))
    booth["pose_enabled"] = False
    for cam in booth["cameras"]:
        cam["calibration"] = str((ROOT / "smartphone_mocap/config" / cam["calibration"]).resolve())
    od = booth["object_detection"]
    od.pop("checkpoint_classes", None)     # tests choose the A (two detectors) or B (one model) split themselves
    od.pop("general_checkpoint", None)
    for key in ("booth_props", "prop_catalog_dir", "checkpoint", "orientation_head"):
        od[key] = str((ROOT / "smartphone_mocap/config" / od[key]).resolve())
    od.update(object_overrides)
    path = tmp_path / "booth.json"
    path.write_text(json.dumps(booth, ensure_ascii=False), encoding="utf-8")
    return path


def test_old_object_keys_are_refused_with_a_pointer_to_v1_1(tmp_path):
    with pytest.raises(ValueError, match="v1.1"):
        PerceptionProcessor(_config(tmp_path, backend="rfdetr_seg"), detector_factory=lambda **_: FakeDetector({}),
                            orientation_factory=lambda **_: None)


def test_the_booth_list_sets_models_heights_counts_and_names(tmp_path):
    processor = PerceptionProcessor(_config(tmp_path), detector_factory=lambda **_: FakeDetector({}),
                                    orientation_factory=lambda **_: None)
    assert processor.counts["giraffe"] == 1 and processor.display_names["giraffe"] == "기린"
    assert processor.asset_map["giraffe"].endswith("giraffe/model_placed.usda")
    assert processor.class_height_m["giraffe"] == pytest.approx(0.18)


def _trained_list(tmp_path):
    path = tmp_path / "trained.json"
    path.write_text(json.dumps({"props": [{"prop_id": "giraffe", "name_ko": "기린"},
                                          {"prop_id": "zebra", "name_ko": "얼룩말"}]}, ensure_ascii=False),
                    encoding="utf-8")
    return str(path)


def test_detector_gets_the_checkpoint_and_only_listed_props(tmp_path):
    seen = {}
    PerceptionProcessor(_config(tmp_path, booth_props=_trained_list(tmp_path)),
                        detector_factory=lambda **kw: seen.update(kw) or FakeDetector({}),
                        orientation_factory=lambda **_: None)
    assert seen["checkpoint"].endswith("prop_detector_a1.pth")
    assert "giraffe" in seen["allowed_classes"] and "cup" not in seen["allowed_classes"]


def _coco_list(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps({"props": [{"prop_id": "giraffe", "name_ko": "기린"},
                                          {"prop_id": "cell_phone", "name_ko": "휴대폰"}]}, ensure_ascii=False),
                    encoding="utf-8")
    return str(path)


def test_general_objects_get_a_second_coco_detector_with_our_names(tmp_path):
    calls = []
    PerceptionProcessor(_config(tmp_path, booth_props=_coco_list(tmp_path)),
                        detector_factory=lambda **kw: calls.append(kw) or FakeDetector({}),
                        orientation_factory=lambda **_: None)
    props, coco = calls
    assert props["checkpoint"].endswith("prop_detector_a1.pth") and props["allowed_classes"] == ["giraffe"]
    assert coco["checkpoint"] is None and coco["allowed_classes"] == ["cell phone"]
    assert coco["class_aliases"] == {"cell phone": "cell_phone"}


def test_a_list_of_trained_props_only_makes_one_detector(tmp_path):
    calls = []
    PerceptionProcessor(_config(tmp_path, booth_props=_trained_list(tmp_path)),
                        detector_factory=lambda **kw: calls.append(kw) or FakeDetector({}),
                        orientation_factory=lambda **_: None)
    assert len(calls) == 1


def test_the_coco_detector_gets_per_prop_thresholds(tmp_path):
    calls = []
    PerceptionProcessor(_config(tmp_path, booth_props=_coco_list(tmp_path), coco_class_confidence={"cell_phone": 0.3}),
                        detector_factory=lambda **kw: calls.append(kw) or FakeDetector({}),
                        orientation_factory=lambda **_: None)
    props, coco = calls
    assert coco["class_confidence"] == {"cell_phone": 0.3}
    assert "class_confidence" not in props or not props["class_confidence"]


def test_a_checkpoint_that_knows_general_objects_replaces_the_coco_detector(tmp_path):
    calls = []
    PerceptionProcessor(_config(tmp_path, booth_props=_coco_list(tmp_path),
                                checkpoint_classes=["cell_phone", "giraffe", "zebra"],
                                class_confidence={"cell_phone": 0.3}),
                        detector_factory=lambda **kw: calls.append(kw) or FakeDetector({}),
                        orientation_factory=lambda **_: None)
    (only,) = calls                                  # one model, no COCO detector
    assert only["allowed_classes"] == ["cell_phone", "giraffe"]
    assert only["class_confidence"] == {"cell_phone": 0.3}


def test_association_keeps_detections_down_to_the_lowest_prop_threshold(tmp_path):
    processor = PerceptionProcessor(_config(tmp_path, booth_props=_coco_list(tmp_path), confidence=0.5,
                                            coco_class_confidence={"cell_phone": 0.3}),
                                    detector_factory=lambda **_: FakeDetector({}), orientation_factory=lambda **_: None)
    assert processor.association_min_confidence == 0.3   # else a 0.3 phone is dropped after detection


def test_general_objects_can_come_from_our_own_trained_model(tmp_path):
    calls = []
    PerceptionProcessor(_config(tmp_path, booth_props=_coco_list(tmp_path), general_checkpoint="/models/b3.pth",
                                coco_class_confidence={"cell_phone": 0.3}),
                        detector_factory=lambda **kw: calls.append(kw) or FakeDetector({}),
                        orientation_factory=lambda **_: None)
    figures, general = calls
    assert figures["allowed_classes"] == ["giraffe"]
    assert general["checkpoint"] == "/models/b3.pth"
    assert general["allowed_classes"] == ["cell_phone"] and not general["class_aliases"]   # it already uses our ids
    assert general["class_confidence"] == {"cell_phone": 0.3}
