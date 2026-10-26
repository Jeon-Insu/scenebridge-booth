import sys
import types

import numpy as np


class _Prediction:
    def __init__(self):
        self.xyxy = np.array([[10., 10., 40., 60.]])
        self.confidence = np.array([.9])
        self.class_id = np.array([1])
        mask = np.zeros((80, 80), bool)
        mask[10:60, 10:40] = True
        self.mask = np.array([mask])


def _fake_rfdetr(monkeypatch, created):
    module = types.ModuleType("rfdetr")

    class FakeSegNano:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.class_names = ["green_robot", "stainless_mug"] if "pretrain_weights" in kwargs else None

        def inference(self, **kwargs):
            pass

        def predict(self, images, **kwargs):
            return [_Prediction() for _ in images]

    module.RFDETRSegNano = FakeSegNano
    assets = types.ModuleType("rfdetr.assets")
    coco = types.ModuleType("rfdetr.assets.coco_classes")
    coco.COCO_CLASSES = {1: "person", 47: "cup"}
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", assets)
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)


def test_custom_checkpoint_is_loaded_and_names_its_classes(monkeypatch):
    from smartphone_mocap.object_detection import RFDETRSegObjectDetector
    created = []
    _fake_rfdetr(monkeypatch, created)
    detector = RFDETRSegObjectDetector(device="cpu", checkpoint="/models/props.pth")
    assert created[0]["pretrain_weights"] == "/models/props.pth"
    image = np.zeros((80, 80, 3), np.uint8)
    labels = [d.label for d in detector.infer_batch({"cam_A": image})["cam_A"]]
    assert labels == ["stainless_mug"]          # class_id 1 → second checkpoint class, 0-based


def test_without_checkpoint_the_coco_model_is_used(monkeypatch):
    from smartphone_mocap.object_detection import RFDETRSegObjectDetector
    created = []
    _fake_rfdetr(monkeypatch, created)
    RFDETRSegObjectDetector(device="cpu")
    assert "pretrain_weights" not in created[0]
