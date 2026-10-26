"""Score a prop detector on a held-out synthetic dataset.

  .venv-mocap/bin/python -m smartphone_mocap.synthdata.evaluate --dataset datasets/props_synth_val_<date> \
      --checkpoint <model>.pth --out outputs/synthetic/eval/<name>.json [--split valid]

Per prop: how many visible instances (mask area >= --min-area) the model found with the
right prop id (box IoU >= 0.5). Also: which prop was mistaken for which, and detections
where there is no prop at all (people, chairs, TVs, cans in the distractor set).
This is a synthetic stand-in until real validation recordings exist; it is not venue
accuracy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from ..types import BoundingBox2D

MATCH_IOU = 0.5
NEAR_IOU = 0.3



def box_iou(a, b):
    a = (a.x1, a.y1, a.x2, a.y2)
    b = (b.x1, b.y1, b.x2, b.y2)
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.

def _box(xywh):
    x, y, w, h = (float(v) for v in xywh)
    return BoundingBox2D(x, y, x + w, y + h)


def score_image(ground_truth, detections, names, *, min_area=400):
    expected = [(names[g["category_id"]], _box(g["bbox"])) for g in ground_truth
                if float(g.get("area", 0)) >= min_area]
    every_gt = [_box(g["bbox"]) for g in ground_truth]
    found, confusion, false_positives = Counter(), Counter(), 0
    matched = set()
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        label = getattr(detection, "asset_id", None) or detection.label
        best, best_iou = None, 0.0
        for index, (name, box) in enumerate(expected):
            iou = box_iou(detection.box, box)
            if index not in matched and iou > best_iou:
                best, best_iou = index, iou
        if best is not None and best_iou >= MATCH_IOU:
            matched.add(best)
            name = expected[best][0]
            if name == label:
                found[name] += 1
            else:
                confusion[(name, label)] += 1
        elif not any(box_iou(detection.box, box) >= NEAR_IOU for box in every_gt):
            false_positives += 1
    return {"expected": dict(Counter(name for name, _ in expected)), "found": dict(found),
            "confusion": dict(confusion), "false_positives": false_positives}


def evaluate_dataset(dataset_dir, detector, *, split="valid", min_area=400):
    root = Path(dataset_dir) / split
    data = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    expected, found, confusion = Counter(), Counter(), Counter()
    false_positives, images_with_fp = 0, 0
    for info in data["images"]:
        image = cv2.imread(str(root / info["file_name"]), cv2.IMREAD_COLOR)
        detections = detector.infer_batch({"image": image}).get("image", [])
        result = score_image([a for a in data["annotations"] if a["image_id"] == info["id"]],
                             detections, names, min_area=min_area)
        expected.update(result["expected"])
        found.update(result["found"])
        confusion.update(result["confusion"])
        false_positives += result["false_positives"]
        images_with_fp += result["false_positives"] > 0
    count = len(data["images"])
    return {
        "dataset": str(dataset_dir), "split": split, "images": count, "min_area_px": min_area,
        "props": {name: {"expected": expected[name], "found": found[name],
                         "recall": found[name] / expected[name] if expected[name] else None}
                  for name in sorted(expected)},
        "confusion": [{"truth": t, "predicted": p, "count": n} for (t, p), n in confusion.most_common()],
        "false_positives_per_image": false_positives / count if count else 0.0,
        "images_with_false_positive": images_with_fp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=400)
    args = parser.parse_args()
    from ..object_detection import RFDETRSegObjectDetector
    detector = RFDETRSegObjectDetector(checkpoint=str(args.checkpoint), confidence=args.confidence)
    report = evaluate_dataset(args.dataset, detector, split=args.split, min_area=args.min_area)
    report.update({"checkpoint": str(args.checkpoint), "confidence": args.confidence})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, row in report["props"].items():
        recall = "-" if row["recall"] is None else f"{row['recall']:.2f}"
        print(f"{name:15s} {row['found']:4d}/{row['expected']:<4d} recall {recall}")
    print(f"이미지당 오검출 {report['false_positives_per_image']:.2f}, "
          f"오검출 있는 이미지 {report['images_with_false_positive']}/{report['images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
