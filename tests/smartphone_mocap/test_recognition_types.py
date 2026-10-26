# tests/smartphone_mocap/test_recognition_types.py
import json
from pathlib import Path

import pytest

from smartphone_mocap.recognition.types import Registry


def _catalog(tmp_path):
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat/catalog.json").write_text(json.dumps({"schema_version": 1, "assets": [
        {"asset_id": "giraffe", "display_name_ko": "기린 피규어", "usd_uri": "giraffe/model_placed.usda",
         "detector_class": "giraffe", "height_m": 0.18, "height_source": "estimate"},
        {"asset_id": "paper_cup", "display_name_ko": "흰 종이컵", "usd_uri": "paper_cup/model_placed.usda",
         "detector_class": "cup", "height_m": 0.1, "height_source": "class_prior"}]}), encoding="utf-8")
    for asset in ("giraffe", "paper_cup"):
        (tmp_path / "assets" / asset / "views").mkdir(parents=True)
        for i in range(2):
            (tmp_path / "assets" / asset / "views" / f"view_{i}.png").write_bytes(b"x")
    return Registry.from_catalog(tmp_path / "cat", tmp_path / "assets")


def test_registry_reads_props_and_their_reference_views(tmp_path):
    registry = _catalog(tmp_path)
    assert registry.ids() == ["giraffe", "paper_cup"]
    assert [p.name for p in registry.get("giraffe").views] == ["view_0.png", "view_1.png"]
    assert registry.get("paper_cup").symmetric          # cups and pots have no facing


def test_registry_without_a_prop_is_the_unregistered_case(tmp_path):
    registry = _catalog(tmp_path).without("giraffe")
    assert registry.ids() == ["paper_cup"]
    with pytest.raises(KeyError):
        registry.get("giraffe")
