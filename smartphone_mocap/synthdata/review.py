"""Contact sheet for checking a synthetic dataset by eye, optionally with a model's detections.

  .venv-mocap/bin/python -m smartphone_mocap.synthdata.review --dataset datasets/props_synth_<date> \
      --out outputs/synthetic/review/<date>.jpg [--count 12] [--checkpoint <model>.pth]

Each tile is one training image with its ground truth (mask, box, prop id) and the
degradation applied to it. With --checkpoint, a second tile per image shows what that
fine-tuned detector finds, so a dataset error and a model error can be told apart.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _colour(key) -> tuple[int, int, int]:
    rng = np.random.default_rng(zlib.crc32(str(key).encode()))
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def _label(image, text, x, y, colour):
    y = max(18, int(y) - 6)
    cv2.putText(image, text, (int(x), y), FONT, 0.7, (0, 0, 0), 4)
    cv2.putText(image, text, (int(x), y), FONT, 0.7, colour, 2)


def draw_ground_truth(image: np.ndarray, annotations: list[dict], names: dict[int, str]) -> np.ndarray:
    out, overlay = image.copy(), image.copy()
    for ann in annotations:
        colour = _colour(names[ann["category_id"]])
        for polygon in ann["segmentation"]:
            cv2.fillPoly(overlay, [np.asarray(polygon).reshape(-1, 2).astype(np.int32)], colour)
    mask = np.any(overlay != image, axis=2)
    out[mask] = (0.45 * overlay[mask] + 0.55 * image[mask]).astype(np.uint8)
    for ann in annotations:
        colour = _colour(names[ann["category_id"]])
        x, y, w, h = (int(v) for v in ann["bbox"])
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        _label(out, names[ann["category_id"]], x, y, colour)
    return out


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    out = image.copy()
    for detection in detections:
        name = getattr(detection, "asset_id", None) or detection.label
        colour = _colour(name)
        if detection.mask is not None:
            out[detection.mask] = (0.45 * np.asarray(colour) + 0.55 * out[detection.mask]).astype(np.uint8)
        box = detection.box
        cv2.rectangle(out, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), colour, 2)
        _label(out, f"{name} {detection.confidence:.2f}", box.x1, box.y1, colour)
    return out


def _caption(tile, text):
    cv2.putText(tile, text[:80], (6, 18), FONT, 0.5, (0, 0, 0), 3)
    cv2.putText(tile, text[:80], (6, 18), FONT, 0.5, (255, 255, 255), 1)
    return tile


def review_sheet(dataset_dir, *, split="train", count=12, tile=(640, 360), columns=3,
                 detector=None, seed=0) -> np.ndarray:
    root = Path(dataset_dir) / split
    data = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    rng = np.random.default_rng(seed)
    picked = [data["images"][i] for i in rng.choice(len(data["images"]), min(count, len(data["images"])),
                                                    replace=False)]
    tiles = []
    for info in picked:
        image = cv2.imread(str(root / info["file_name"]), cv2.IMREAD_COLOR)
        annotations = [a for a in data["annotations"] if a["image_id"] == info["id"]]
        degradation = info.get("extra", {}).get("degradation", {})
        tiles.append(_caption(cv2.resize(draw_ground_truth(image, annotations, names), tile),
                              f"GT {info['file_name']} {degradation}"))
        if detector is not None:
            found = detector.infer_batch({"image": image}).get("image", [])
            tiles.append(_caption(cv2.resize(draw_detections(image, found), tile), f"model: {len(found)}"))
    per_row = columns * (2 if detector is not None else 1)
    blank = np.zeros((tile[1], tile[0], 3), np.uint8)
    while len(tiles) % per_row:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    return np.vstack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "valid"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, help="fine-tuned RF-DETR-Seg checkpoint to draw beside the truth")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()
    detector = None
    if args.checkpoint:
        from ..object_detection import RFDETRSegObjectDetector
        detector = RFDETRSegObjectDetector(checkpoint=str(args.checkpoint), confidence=args.confidence)
    sheet = review_sheet(args.dataset, split=args.split, count=args.count, columns=2 if detector else 3,
                         detector=detector, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
