from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import CameraCalibration


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: str | Path) -> CameraCalibration:
    return CameraCalibration.from_dict(load_json(path))


def save_calibration(path: str | Path, calibration: CameraCalibration) -> None:
    save_json(path, calibration.to_dict())

