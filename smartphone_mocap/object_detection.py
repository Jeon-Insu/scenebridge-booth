from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import BoundingBox2D, Object2D, SceneObject3D


def _appearance_descriptor(
    image: np.ndarray, box: BoundingBox2D, mask: np.ndarray | None
) -> tuple[tuple[float, ...], np.ndarray]:
    """Build a small, model-independent colour/shape descriptor and preview crop."""
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(np.floor(box.x1))))
    y1 = max(0, min(height - 1, int(np.floor(box.y1))))
    x2 = max(x1 + 1, min(width, int(np.ceil(box.x2))))
    y2 = max(y1 + 1, min(height, int(np.ceil(box.y2))))
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = None if mask is None else mask[y1:y2, x1:x2].astype(np.uint8)
    if crop_mask is not None and np.any(crop_mask):
        # Remove a little boundary background from the colour signature.
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(crop_mask, kernel, iterations=1)
        if np.any(eroded):
            crop_mask = eroded
        representative = cv2.bitwise_and(crop, crop, mask=crop_mask * 255)
        occupancy = float(np.mean(crop_mask > 0))
    else:
        representative = crop
        crop_mask = None
        occupancy = 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None if crop_mask is None else crop_mask, [12, 4], [0, 180, 0, 256]
    ).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    if norm > 1e-8:
        histogram /= norm
    aspect = min(4.0, (x2 - x1) / max(float(y2 - y1), 1.0)) / 4.0
    descriptor = np.concatenate([histogram, np.array([aspect, occupancy]) * 0.25])
    descriptor_norm = float(np.linalg.norm(descriptor))
    if descriptor_norm > 1e-8:
        descriptor /= descriptor_norm
    return tuple(float(value) for value in descriptor), representative


def _make_object(
    *,
    camera_id: str,
    local_detection_id: int,
    label: str,
    class_id: int,
    xyxy,
    confidence: float,
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> Object2D:
    box = BoundingBox2D(*(float(value) for value in xyxy))
    appearance, crop = _appearance_descriptor(image, box, mask)
    return Object2D(
        camera_id=camera_id,
        local_detection_id=local_detection_id,
        label=label,
        class_id=class_id,
        box=box,
        confidence=confidence,
        mask=mask,
        appearance=appearance,
        representative_crop=crop,
    )


def _estimate_tabletop_foreground_mask(image: np.ndarray, xyxy) -> np.ndarray:
    """Estimate a fast foreground mask for props on a uniform tabletop."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    crop = image[top:bottom, left:right]
    crop_height, crop_width = crop.shape[:2]
    full_mask = np.zeros((height, width), dtype=bool)
    if crop_height < 3 or crop_width < 3:
        full_mask[top:bottom, left:right] = True
        return full_mask

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_size = max(1, min(crop_height, crop_width) // 12)
    border_pixels = np.concatenate(
        [
            lab[:border_size].reshape(-1, 3),
            lab[-border_size:].reshape(-1, 3),
            lab[:, :border_size].reshape(-1, 3),
            lab[:, -border_size:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.linalg.norm(border_pixels - background, axis=1)
    threshold = max(12.0, float(np.percentile(border_distance, 95.0)) + 6.0)
    local_mask = (distance > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        local_mask, connectivity=8
    )
    minimum_area = max(9, round(crop_height * crop_width * 0.025))
    if component_count > 1:
        candidates = [
            index
            for index in range(1, component_count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        local_mask = (
            np.isin(component_labels, candidates).astype(np.uint8)
            if candidates
            else np.zeros_like(local_mask)
        )

    if np.count_nonzero(local_mask) < minimum_area:
        # Preserve a usable centroid for low-contrast white props.
        inset_x = max(1, crop_width // 12)
        inset_y = max(1, crop_height // 12)
        local_mask[inset_y : crop_height - inset_y, inset_x : crop_width - inset_x] = 1
    full_mask[top:bottom, left:right] = local_mask.astype(bool)
    return full_mask


def _overlap_ratio(first: Object2D, second: Object2D) -> float:
    """Return intersection over the smaller detection area.

    This is intentionally stronger than ordinary IoU for alias collisions such
    as COCO ``vase`` and ``potted plant``: one prediction is often wholly inside
    the other even when their IoU is only moderate.
    """
    if first.mask is not None and second.mask is not None:
        intersection = float(np.count_nonzero(first.mask & second.mask))
        smaller = float(
            min(np.count_nonzero(first.mask), np.count_nonzero(second.mask))
        )
        return intersection / smaller if smaller > 0.0 else 0.0
    x1 = max(first.box.x1, second.box.x1)
    y1 = max(first.box.y1, second.box.y1)
    x2 = min(first.box.x2, second.box.x2)
    y2 = min(first.box.y2, second.box.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first.box.x2 - first.box.x1) * max(
        0.0, first.box.y2 - first.box.y1
    )
    second_area = max(0.0, second.box.x2 - second.box.x1) * max(
        0.0, second.box.y2 - second.box.y1
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0.0 else 0.0


def _deduplicate_detections(
    detections: list[Object2D], *, overlap_threshold: float = 0.72
) -> list[Object2D]:
    """Remove overlapping predictions that collapse to the same alias label."""
    kept: list[Object2D] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            detection.label == existing.label
            and _overlap_ratio(detection, existing) >= overlap_threshold
            for existing in kept
        ):
            continue
        kept.append(detection)
    return kept


class UltralyticsObjectDetector:
    """Batched YOLO detection/instance segmentation with original-image masks."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | int | None = None,
        image_size: int = 640,
        confidence: float = 0.25,
        allowed_classes: list[str] | None = None,
        excluded_classes: list[str] | None = None,
        class_aliases: dict[str, str] | None = None,
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO
            from ultralytics.cfg import DEFAULT_CFG_DICT
        except ImportError as exc:
            raise RuntimeError(
                "Detection dependencies are missing. Run tools/setup_smartphone_mocap.sh first."
            ) from exc
        self.model = YOLO(str(model_path))
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.precision_kwargs = (
            {"quantize": 32 if str(self.device) == "cpu" else 16}
            if "quantize" in DEFAULT_CFG_DICT else {"half": str(self.device) != "cpu"}
        )
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.allowed_classes = set(allowed_classes or [])
        self.excluded_classes = set(excluded_classes or [])
        self.class_aliases = dict(class_aliases or {})

    def infer(self, camera_id: str, image: np.ndarray) -> list[Object2D]:
        return self.infer_batch({camera_id: image}).get(camera_id, [])

    def infer_batch(self, images: dict[str, np.ndarray]) -> dict[str, list[Object2D]]:
        if not images:
            return {}
        camera_ids = list(images)
        results = self.model.predict(
            source=[images[name] for name in camera_ids],
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            retina_masks=True,  # Avoid reinterpreting letterboxed masks as image pixels.
            **getattr(self, "precision_kwargs", {"half":str(self.device) != "cpu"}),
            verbose=False,
        )
        return {
            camera_id: self._decode(camera_id, images[camera_id], result)
            for camera_id, result in zip(camera_ids, results, strict=True)
        }

    def _decode(self, camera_id: str, image: np.ndarray, result) -> list[Object2D]:
        if result.boxes is None or result.boxes.xyxy is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names
        masks = getattr(result, "masks", None)
        masks = None if masks is None else masks.data.detach().cpu().numpy()
        if masks is not None and (len(masks) != len(boxes) or masks.shape[1:] != image.shape[:2]):
            raise ValueError("YOLO instance masks must align with boxes and original image size")
        detections: list[Object2D] = []
        for index, (xyxy, confidence, class_id) in enumerate(
            zip(boxes, confidences, class_ids, strict=True)
        ):
            label = str(names[int(class_id)])
            if self.allowed_classes and label not in self.allowed_classes:
                continue
            if label in self.excluded_classes:
                continue
            label = self.class_aliases.get(label, label)
            detections.append(
                _make_object(
                    camera_id=camera_id,
                    local_detection_id=index,
                    label=label,
                    class_id=int(class_id),
                    xyxy=xyxy,
                    confidence=float(confidence),
                    image=image,
                    mask=None if masks is None else masks[index] >= 0.5,
                )
            )
        return _deduplicate_detections(detections)


class TorchvisionMaskRCNNObjectDetector:
    """Permissively licensed TorchVision instance-segmentation backend.

    The detector returns a full-resolution binary mask for every object, plus a
    lightweight appearance descriptor used only for temporal association.
    """

    def __init__(
        self,
        *,
        device: str | int | None = None,
        image_size: int = 640,
        confidence: float = 0.5,
        mask_threshold: float = 0.5,
        use_amp: bool = True,
        allowed_classes: list[str] | None = None,
        excluded_classes: list[str] | None = None,
        class_aliases: dict[str, str] | None = None,
    ) -> None:
        try:
            import torch
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_V2_Weights,
                maskrcnn_resnet50_fpn_v2,
            )
        except ImportError as exc:
            raise RuntimeError("TorchVision Mask R-CNN dependencies are missing.") from exc
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.categories = tuple(str(name) for name in weights.meta["categories"])
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            min_size=int(image_size),
            max_size=int(image_size),
        ).to(self.device)
        self.model.eval()
        self.torch = torch
        self.confidence = float(confidence)
        self.mask_threshold = float(mask_threshold)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.allowed_classes = set(allowed_classes or [])
        self.excluded_classes = set(excluded_classes or [])
        self.class_aliases = dict(class_aliases or {})

    def infer(self, camera_id: str, image: np.ndarray) -> list[Object2D]:
        return self.infer_batch({camera_id: image}).get(camera_id, [])

    def infer_batch(self, images: dict[str, np.ndarray]) -> dict[str, list[Object2D]]:
        camera_ids = list(images)
        tensors = [
            self.torch.from_numpy(images[name][:, :, ::-1].copy())
            .permute(2, 0, 1)
            .to(self.device, dtype=self.torch.float32)
            .div_(255.0)
            for name in camera_ids
        ]
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.float16,
            enabled=self.use_amp,
        ):
            predictions = self.model(tensors)
        result: dict[str, list[Object2D]] = {}
        for camera_id, prediction in zip(camera_ids, predictions, strict=True):
            image = images[camera_id]
            scores = prediction["scores"].detach().cpu().numpy()
            boxes = prediction["boxes"].detach().cpu().numpy()
            labels = prediction["labels"].detach().cpu().numpy().astype(int)
            masks = prediction["masks"][:, 0].detach().cpu().numpy()
            detections: list[Object2D] = []
            for index, (score, xyxy, class_id, probability_mask) in enumerate(
                zip(scores, boxes, labels, masks, strict=True)
            ):
                if float(score) < self.confidence:
                    break
                label = self.categories[int(class_id)]
                if label == "N/A":
                    continue
                if self.allowed_classes and label not in self.allowed_classes:
                    continue
                if label in self.excluded_classes:
                    continue
                label = self.class_aliases.get(label, label)
                mask = probability_mask >= self.mask_threshold
                if not np.any(mask):
                    continue
                detections.append(
                    _make_object(
                        camera_id=camera_id,
                        local_detection_id=index,
                        label=label,
                        class_id=int(class_id),
                        xyxy=xyxy,
                        confidence=float(score),
                        image=image,
                        mask=mask,
                    )
                )
            result[camera_id] = _deduplicate_detections(detections)
        return result


class RFDETRSegObjectDetector:
    """Apache-2.0 RF-DETR instance-segmentation backend.

    RF-DETR performs class detection and mask prediction in one pass.  The
    four synchronized camera images are submitted as one batch so the shared
    GPU is used efficiently.
    """

    _VARIANTS = {
        "nano": "RFDETRSegNano",
        "small": "RFDETRSegSmall",
        "medium": "RFDETRSegMedium",
        "large": "RFDETRSegLarge",
    }

    def __init__(
        self,
        *,
        device: str | int | None = None,
        image_size: int = 312,
        confidence: float = 0.5,
        mask_threshold: float = 0.5,
        use_amp: bool = True,
        variant: str = "nano",
        compile_model: bool = False,
        allowed_classes: list[str] | None = None,
        excluded_classes: list[str] | None = None,
        class_aliases: dict[str, str] | None = None,
        checkpoint: str | None = None,
    ) -> None:
        try:
            import torch
            import rfdetr
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR dependencies are missing. Run tools/setup_smartphone_mocap.sh first."
            ) from exc
        normalized_variant = str(variant).lower()
        if normalized_variant not in self._VARIANTS:
            choices = ", ".join(sorted(self._VARIANTS))
            raise ValueError(
                f"Unsupported RF-DETR-Seg variant: {variant}. Choose one of: {choices}"
            )
        if isinstance(device, int):
            device_name = f"cuda:{device}"
        elif device is None:
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_name = str(device)
        model_class = getattr(rfdetr, self._VARIANTS[normalized_variant])
        # RF-DETR Nano's native 312px resolution is used in the live config.
        # Custom resolutions are accepted by RF-DETR when compatible with the
        # selected variant's patch/window layout.
        if checkpoint:
            # A fine-tuned checkpoint carries its own class list; its class ids are
            # 0-based indices into that list, not COCO category ids.
            self.model = model_class(device=device_name, resolution=int(image_size),
                                     pretrain_weights=str(checkpoint))
            self.categories = list(self.model.class_names)
        else:
            self.model = model_class(device=device_name, resolution=int(image_size))
            self.categories = COCO_CLASSES
        self.device = torch.device(device_name)
        self.torch = torch
        self.confidence = float(confidence)
        # RF-DETR already returns binary masks. Keep the option for backend
        # parity and for future probability-mask releases.
        self.mask_threshold = float(mask_threshold)
        self.allowed_classes = set(allowed_classes or [])
        self.excluded_classes = set(excluded_classes or [])
        self.class_aliases = dict(class_aliases or {})
        dtype = torch.float16 if bool(use_amp) and self.device.type == "cuda" else torch.float32
        # JIT tracing fixes the batch size and adds a long first-run compile.
        # Eager FP16 is already fast on GB10 and remains valid if cameras drop.
        self.model.inference(
            compile=bool(compile_model),
            batch_size=4,
            dtype=dtype,
            inplace=not bool(compile_model),
        )

    def infer(self, camera_id: str, image: np.ndarray) -> list[Object2D]:
        return self.infer_batch({camera_id: image}).get(camera_id, [])

    def infer_batch(self, images: dict[str, np.ndarray]) -> dict[str, list[Object2D]]:
        camera_ids = list(images)
        if not camera_ids:
            return {}
        # OpenCV frames are BGR; RF-DETR accepts RGB numpy images.
        rgb_images = [
            cv2.cvtColor(images[camera_id], cv2.COLOR_BGR2RGB)
            for camera_id in camera_ids
        ]
        predictions = self.model.predict(
            rgb_images,
            threshold=self.confidence,
            include_source_image=False,
        )
        if len(camera_ids) == 1 and not isinstance(predictions, list):
            predictions = [predictions]
        result: dict[str, list[Object2D]] = {}
        for camera_id, prediction in zip(camera_ids, predictions, strict=True):
            image = images[camera_id]
            masks = getattr(prediction, "mask", None)
            detections: list[Object2D] = []
            for index, (xyxy, confidence, class_id) in enumerate(
                zip(
                    prediction.xyxy,
                    prediction.confidence,
                    prediction.class_id,
                    strict=True,
                )
            ):
                label = str(self.categories[int(class_id)])
                if self.allowed_classes and label not in self.allowed_classes:
                    continue
                if label in self.excluded_classes:
                    continue
                label = self.class_aliases.get(label, label)
                mask = None if masks is None else np.asarray(masks[index])
                if mask is not None:
                    mask = mask >= self.mask_threshold
                    if not np.any(mask):
                        continue
                detections.append(
                    _make_object(
                        camera_id=camera_id,
                        local_detection_id=index,
                        label=label,
                        class_id=int(class_id),
                        xyxy=xyxy,
                        confidence=float(confidence),
                        image=image,
                        mask=mask,
                    )
                )
            result[camera_id] = _deduplicate_detections(detections)
        return result


class TorchvisionMobileNetObjectDetector:
    """Fast BSD-licensed detector with a controlled-table mask approximation."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        image_size: int = 320,
        confidence: float = 0.45,
        allowed_classes: list[str] | None = None,
        excluded_classes: list[str] | None = None,
        class_aliases: dict[str, str] | None = None,
    ) -> None:
        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
            )
        except ImportError as exc:
            raise RuntimeError("TorchVision detection dependencies are missing.") from exc
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.categories = tuple(str(name) for name in weights.meta["categories"])
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(
            self.device
        )
        requested_size = int(image_size)
        self.model.transform.min_size = (requested_size,)
        self.model.transform.max_size = requested_size
        self.model.eval()
        self.torch = torch
        self.confidence = float(confidence)
        self.allowed_classes = set(allowed_classes or [])
        self.excluded_classes = set(excluded_classes or [])
        self.class_aliases = dict(class_aliases or {})

    def infer(self, camera_id: str, image: np.ndarray) -> list[Object2D]:
        return self.infer_batch({camera_id: image}).get(camera_id, [])

    def infer_batch(self, images: dict[str, np.ndarray]) -> dict[str, list[Object2D]]:
        camera_ids = list(images)
        tensors = [
            self.torch.from_numpy(images[name][:, :, ::-1].copy())
            .permute(2, 0, 1)
            .to(self.device, dtype=self.torch.float32)
            .div_(255.0)
            for name in camera_ids
        ]
        with self.torch.inference_mode():
            predictions = self.model(tensors)
        result: dict[str, list[Object2D]] = {}
        for camera_id, prediction in zip(camera_ids, predictions, strict=True):
            image = images[camera_id]
            scores = prediction["scores"].detach().cpu().numpy()
            boxes = prediction["boxes"].detach().cpu().numpy()
            labels = prediction["labels"].detach().cpu().numpy().astype(int)
            detections: list[Object2D] = []
            for index, (score, xyxy, class_id) in enumerate(
                zip(scores, boxes, labels, strict=True)
            ):
                if float(score) < self.confidence:
                    break
                label = self.categories[int(class_id)]
                if label == "N/A":
                    continue
                if self.allowed_classes and label not in self.allowed_classes:
                    continue
                if label in self.excluded_classes:
                    continue
                label = self.class_aliases.get(label, label)
                detections.append(
                    _make_object(
                        camera_id=camera_id,
                        local_detection_id=index,
                        label=label,
                        class_id=int(class_id),
                        xyxy=xyxy,
                        confidence=float(score),
                        image=image,
                        mask=_estimate_tabletop_foreground_mask(image, xyxy),
                    )
                )
            result[camera_id] = _deduplicate_detections(detections)
        return result


def draw_object_detections(
    image: np.ndarray,
    detections: list[Object2D],
    tracked_by_detection: dict[tuple[str, int], SceneObject3D] | None = None,
    *,
    show_masks: bool = True,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_anchors: bool = True,
    mask_alpha: float = 0.55,
) -> np.ndarray:
    preview = image.copy()
    tracked_by_detection = tracked_by_detection or {}
    for detection in detections:
        tracked = tracked_by_detection.get(
            (detection.camera_id, detection.local_detection_id)
        )
        colour = (50, 210, 255)
        if tracked is not None:
            seed = sum(ord(character) for character in tracked.global_id)
            colour = (
                60 + (seed * 47) % 180,
                60 + (seed * 83) % 180,
                60 + (seed * 131) % 180,
            )
        if show_masks and detection.mask is not None:
            mask_layer = np.zeros_like(preview)
            mask_layer[detection.mask] = colour
            preview = cv2.addWeighted(
                preview, 1.0, mask_layer, float(mask_alpha), 0.0
            )
            contours, _ = cv2.findContours(
                detection.mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(preview, contours, -1, colour, 2)
        box = detection.box
        first = (round(box.x1), round(box.y1))
        second = (round(box.x2), round(box.y2))
        if show_boxes:
            cv2.rectangle(preview, first, second, colour, 2)
        local_suffix = (
            f"/L{detection.local_track_id}"
            if detection.local_track_id is not None
            else ""
        )
        label = f"{detection.label}{local_suffix} {detection.confidence:.2f}"
        if tracked is not None:
            instance_name = tracked.global_id.rsplit("/", 1)[-1]
            x, y, z = tracked.xyz
            label = (
                f"{detection.label}{local_suffix}/{instance_name} {tracked.state} "
                f"xyz=({x:.2f},{y:.2f},{z:.2f})m"
            )
        else:
            label += " unmatched"
        if show_labels:
            cv2.putText(
                preview,
                label,
                (first[0], max(22, first[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )
        if show_anchors:
            anchor = detection.anchor_xy(
                "mask_centroid" if detection.mask is not None else "bottom_center"
            )
            cv2.circle(
                preview, (round(anchor[0]), round(anchor[1])), 5, (30, 40, 255), -1
            )
    return preview
