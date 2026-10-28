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


def _fake_with(monkeypatch, class_ids, confidences, calls):
    module = types.ModuleType("rfdetr")

    class Prediction:
        def __init__(self):
            n = len(class_ids)
            self.xyxy = np.array([[10. + 30 * i, 10., 35. + 30 * i, 60.] for i in range(n)])
            self.confidence = np.array(confidences)
            self.class_id = np.array(class_ids)
            masks = np.zeros((n, 80, 120), bool)
            for i in range(n):
                masks[i, 10:60, 10 + 30 * i:35 + 30 * i] = True
            self.mask = masks

    class FakeSegNano:
        def __init__(self, **kwargs):
            self.class_names = ["green_robot", "stainless_mug"]

        def inference(self, **kwargs):
            pass

        def predict(self, images, **kwargs):
            calls.append(kwargs)
            return [Prediction() for _ in images]

    module.RFDETRSegNano = FakeSegNano
    coco = types.ModuleType("rfdetr.assets.coco_classes")
    coco.COCO_CLASSES = {}
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    monkeypatch.setitem(sys.modules, "rfdetr.assets", types.ModuleType("rfdetr.assets"))
    monkeypatch.setitem(sys.modules, "rfdetr.assets.coco_classes", coco)


def test_one_class_can_have_a_lower_threshold_than_the_rest(monkeypatch):
    from smartphone_mocap.object_detection import RFDETRSegObjectDetector
    calls = []
    _fake_with(monkeypatch, [0, 1], [.35, .35], calls)
    detector = RFDETRSegObjectDetector(device="cpu", checkpoint="/m.pth", confidence=0.5,
                                       class_confidence={"stainless_mug": 0.3})
    labels = [d.label for d in detector.infer_batch({"cam_A": np.zeros((80, 120, 3), np.uint8)})["cam_A"]]
    assert calls[0]["threshold"] == 0.3            # the model is asked down to the lowest threshold ...
    assert labels == ["stainless_mug"]             # ... and the other classes still need 0.5


def test_a_class_id_outside_the_class_list_is_skipped(monkeypatch):
    from smartphone_mocap.object_detection import RFDETRSegObjectDetector
    _fake_with(monkeypatch, [1, 2], [.9, .9], [])
    detector = RFDETRSegObjectDetector(device="cpu", checkpoint="/m.pth")
    labels = [d.label for d in detector.infer_batch({"cam_A": np.zeros((80, 120, 3), np.uint8)})["cam_A"]]
    assert labels == ["stainless_mug"]             # low thresholds can return an id past the list; no crash
