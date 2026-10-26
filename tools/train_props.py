"""Fine-tune RF-DETR-Seg on a synthetic prop dataset (smartphone_mocap.synthdata).

  .venv-mocap/bin/python tools/train_props.py --dataset datasets/props_synth_<date> \
      --output models/props_<date> --epochs 30

Writes RF-DETR's checkpoints and ``classes.json`` (class names in the model's index
order) into --output. Copy the best checkpoint to outputs/models/ and point ``object_detection.checkpoint`` at
``checkpoint_best_total.pth``. Synthetic validation scores are not venue accuracy;
judge the model on real recordings with ``evaluate-recording``.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartphone_mocap.synthdata.coco_dataset import dataset_classes, validate_dataset  # noqa: E402

VARIANTS = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=312, help="same as the runtime image_size")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="nano")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    problems = validate_dataset(args.dataset)
    if problems:
        for problem in problems[:20]:
            print("문제:", problem)
        return 1
    import rfdetr
    args.output.mkdir(parents=True, exist_ok=True)
    classes = dataset_classes(args.dataset)
    (args.output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")
    model = getattr(rfdetr, VARIANTS[args.variant])(resolution=args.resolution)
    model.train(dataset_dir=str(args.dataset), epochs=args.epochs, batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps, lr=args.lr, num_workers=args.num_workers,
                output_dir=str(args.output))
    checkpoints = sorted(args.output.rglob("checkpoint*.pth"))
    print("클래스:", classes)
    print("체크포인트:", [str(path) for path in checkpoints])
    return 0 if checkpoints else 1


if __name__ == "__main__":
    raise SystemExit(main())
