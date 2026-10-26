"""학습 렌더의 소품 잘라낸 사진 → (자세 3 × 카메라 기준 8방향) 분류층. DINOv2-small 특징은 고정.

  .venv-mocap/bin/python tools/train_orientation.py --specs outputs/synthetic/specs_pose400.json --raw outputs/synthetic/raw_pose400
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smartphone_mocap.prop_encoder import PropEncoder  # noqa: E402
from smartphone_mocap.recognition.evalset import load_scenes  # noqa: E402
from smartphone_mocap.recognition.orientation import camera_azimuth_deg, relative_bin  # noqa: E402
from smartphone_mocap.recognition.types import POSES  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/recognition/orientation_head.pt"))
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    encoder = PropEncoder()
    features, labels = [], []
    for scene in load_scenes(args.specs, args.raw, degrade_seed=1):
        for truth in scene.truth:
            if truth.asset_id is None:
                continue
            for cam, mask in truth.masks.items():
                ys, xs = np.nonzero(mask)
                if len(xs) < 400:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crop = scene.images[cam][y0:y1, x0:x1].copy()
                crop[~mask[y0:y1, x0:x1]] = 114
                azimuth = camera_azimuth_deg(scene.calibrations[cam], truth.xyz)
                features.append(encoder.encode([Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))])[0])
                labels.append(POSES.index(truth.pose) * 8 + relative_bin(truth.yaw_deg, azimuth))
    X = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(labels)
    head = torch.nn.Linear(X.shape[1], 24)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(args.epochs):
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(X), y)
        loss.backward()
        optimiser.step()
    accuracy = float((head(X).argmax(-1) == y).float().mean())
    print(f"샘플 {len(y)}, 마지막 손실 {float(loss):.3f}, 학습 정확도 {accuracy:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
