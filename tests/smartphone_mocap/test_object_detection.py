import numpy as np

from smartphone_mocap.object_detection import (
    _appearance_descriptor,
    _deduplicate_detections,
    _estimate_tabletop_foreground_mask,
)
from smartphone_mocap.types import BoundingBox2D, Object2D


def test_yolo_batch_masks_keep_original_pixels_and_camera_order():
    from types import SimpleNamespace
    from smartphone_mocap.object_detection import UltralyticsObjectDetector

    class Array:
        def __init__(self, values): self.values = np.asarray(values)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.values

    mask = np.zeros((1,40,60), dtype=np.float32)
    mask[0,10:30,20:40] = 1
    result = SimpleNamespace(boxes=SimpleNamespace(xyxy=Array([[20,10,40,30]]),
        conf=Array([.9]), cls=Array([41])),names={41:"cup"}, masks=SimpleNamespace(data=Array(mask)))
    calls = []
    detector = UltralyticsObjectDetector.__new__(UltralyticsObjectDetector)
    detector.model = SimpleNamespace(predict=lambda **kw: calls.append(kw) or [result,result])
    detector.device,detector.image_size,detector.confidence = "cpu",640,.5
    detector.allowed_classes,detector.excluded_classes,detector.class_aliases = set(),set(),{}
    images = {"B":np.zeros((40,60,3),np.uint8),"A":np.zeros((40,60,3),np.uint8)}
    out = detector.infer_batch(images)
    assert list(out) == ["B","A"]
    assert calls[0]["retina_masks"] and not calls[0]["half"]
    assert out["B"][0].mask.shape == (40,60)
    assert out["B"][0].anchor_xy("mask_centroid") == (29.5,19.5)
    assert out["A"][0].camera_id == "A"


def test_yolo_rejects_letterboxed_mask_shape():
    import pytest
    from types import SimpleNamespace
    from smartphone_mocap.object_detection import UltralyticsObjectDetector
    class Array:
        def __init__(self, values): self.values=np.asarray(values)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.values
    r = SimpleNamespace(boxes=SimpleNamespace(xyxy=Array([[0,0,5,5]]),conf=Array([.9]),cls=Array([41])),
                        names={41:"cup"},masks=SimpleNamespace(data=Array(np.zeros((1,32,32)))))
    detector=UltralyticsObjectDetector.__new__(UltralyticsObjectDetector)
    with pytest.raises(ValueError,match="original image"):
        detector._decode("A",np.zeros((40,60,3),np.uint8),r)


def test_mask_anchor_and_descriptor_ignore_most_background():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[20:80, 40:70] = (0, 0, 255)
    mask = np.zeros((100, 120), dtype=bool)
    mask[20:80, 40:70] = True
    box = BoundingBox2D(35, 15, 75, 85)
    descriptor, crop = _appearance_descriptor(image, box, mask)
    detection = Object2D("cam", 0, "cup", 1, box, 0.9, mask=mask)

    cx, cy = detection.anchor_xy("mask_centroid")
    bx, by = detection.anchor_xy("mask_bottom_center")
    assert abs(cx - 54.5) < 0.1
    assert abs(cy - 49.5) < 0.1
    assert abs(bx - 54.5) < 0.1
    assert 76.0 <= by <= 78.0
    assert len(descriptor) == 50
    assert crop.shape == (70, 40, 3)
    assert np.count_nonzero(crop[:, :4]) == 0


def test_alias_collisions_are_deduplicated_without_merging_nearby_objects():
    mask_large = np.zeros((100, 160), dtype=bool)
    mask_large[15:80, 20:75] = True
    mask_nested = np.zeros_like(mask_large)
    mask_nested[25:70, 28:68] = True
    mask_other = np.zeros_like(mask_large)
    mask_other[20:75, 95:145] = True
    detections = [
        Object2D(
            "cam",
            0,
            "potted plant",
            64,
            BoundingBox2D(20, 15, 75, 80),
            0.92,
            mask=mask_large,
        ),
        Object2D(
            "cam",
            1,
            "potted plant",
            75,
            BoundingBox2D(28, 25, 68, 70),
            0.74,
            mask=mask_nested,
        ),
        Object2D(
            "cam",
            2,
            "potted plant",
            64,
            BoundingBox2D(95, 20, 145, 75),
            0.89,
            mask=mask_other,
        ),
    ]

    deduplicated = _deduplicate_detections(detections)

    assert [item.local_detection_id for item in deduplicated] == [0, 2]


def test_fast_tabletop_mask_separates_coloured_prop_from_white_surface():
    image = np.full((120, 180, 3), 240, dtype=np.uint8)
    image[35:105, 70:115] = (35, 80, 180)
    mask = _estimate_tabletop_foreground_mask(image, (60, 25, 125, 112))
    assert mask.shape == image.shape[:2]
    assert mask[60, 90]
    assert not mask[28, 63]
    assert np.count_nonzero(mask) < 65 * 87 * 0.8
