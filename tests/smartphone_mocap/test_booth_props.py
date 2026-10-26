"""v1.2 booth list: trained props only, merged with assets/props/catalog.json."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from smartphone_mocap import booth_props

ROOT = Path(__file__).resolve().parents[2]


def _catalog(tmp_path):
    root = tmp_path / "props"
    for asset in ("giraffe", "paper_cup"):
        (root / asset).mkdir(parents=True)
        (root / asset / "model_placed.usda").write_text("#usda 1.0\n")
    (root / "catalog.json").write_text(json.dumps({"schema_version": 1, "assets": [
        {"asset_id": "giraffe", "display_name_ko": "기린 피규어", "usd_uri": "giraffe/model_placed.usda",
         "detector_class": "giraffe", "height_m": 0.18, "height_source": "estimate"},
        {"asset_id": "paper_cup", "display_name_ko": "흰 종이컵", "usd_uri": "paper_cup/model_placed.usda",
         "detector_class": "cup", "height_m": 0.1, "height_source": "class_prior"}]}, ensure_ascii=False))
    return root


def _list(tmp_path, props):
    path = tmp_path / "booth_props.json"
    path.write_text(json.dumps({"props": props}, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_listed_prop_takes_model_height_and_symmetry_from_the_catalog(tmp_path):
    root = _catalog(tmp_path)
    props = booth_props.load(_list(tmp_path, [
        {"prop_id": "giraffe", "name_ko": "기린", "aliases_ko": ["기린 인형"], "count": 1},
        {"prop_id": "paper_cup", "name_ko": "컵"}]), catalog_dir=root)
    giraffe, cup = props
    assert (giraffe.prop_id, giraffe.height_m, giraffe.symmetric, giraffe.count) == ("giraffe", 0.18, False, 1)
    assert giraffe.model == root / "giraffe/model_placed.usda"
    assert cup.symmetric and cup.count == 1 and cup.aliases_ko == ()


@pytest.mark.parametrize("entry, words", [
    ({"prop_id": "unicorn", "name_ko": "유니콘"}, "학습한 소품이 아닙니다"),
    ({"prop_id": "giraffe", "name_ko": "기린", "count": 0}, "count"),
    ({"prop_id": "giraffe"}, "name_ko"),
])
def test_bad_entries_are_refused_in_korean(tmp_path, entry, words):
    with pytest.raises(booth_props.BoothPropsError, match=words):
        booth_props.load(_list(tmp_path, [entry]), catalog_dir=_catalog(tmp_path))


def test_check_finds_a_repeated_prop_and_a_shared_word(tmp_path):
    props = booth_props.load(_list(tmp_path, [
        {"prop_id": "giraffe", "name_ko": "동물"}, {"prop_id": "giraffe", "name_ko": "기린"},
        {"prop_id": "paper_cup", "name_ko": "컵", "aliases_ko": ["동물"]}]), catalog_dir=_catalog(tmp_path))
    problems = booth_props.check(props)
    assert any("두 번" in p for p in problems) and any("'동물'" in p for p in problems)


def test_korean_aliases_map_names_and_catalog_names_to_prop_ids(tmp_path):
    path = _list(tmp_path, [{"prop_id": "giraffe", "name_ko": "기린", "aliases_ko": ["기린 인형"]}])
    aliases = booth_props.korean_aliases(path, catalog_dir=_catalog(tmp_path))
    assert {w: t for w, t in aliases.items() if t == "giraffe"} == {"기린": "giraffe", "기린 인형": "giraffe",
                                                                   "기린 피규어": "giraffe"}
    assert booth_props.korean_names("giraffe", aliases) == ["기린", "기린 인형", "기린 피규어"]
    assert booth_props.korean_aliases(tmp_path / "missing.json") == booth_props.BUILTIN_ALIASES


def test_a_list_name_wins_over_a_builtin_word(tmp_path):
    path = _list(tmp_path, [{"prop_id": "paper_cup", "name_ko": "컵"}])
    assert booth_props.korean_aliases(path, catalog_dir=_catalog(tmp_path))["컵"] == "paper_cup"


def test_the_project_list_is_valid():
    props = booth_props.load(ROOT / "config/booth_props.json")
    assert {p.prop_id for p in props} <= {e["asset_id"] for e in booth_props.catalog()}
    assert booth_props.check(props) == []


def test_booth_check_cli_lists_the_trained_props():
    result = subprocess.run([sys.executable, str(ROOT / "tools/booth_check.py"), "--props"],
                            capture_output=True, text=True, check=True)
    assert "giraffe" in result.stdout and "기린" in result.stdout


@pytest.mark.parametrize("entry, words", [
    ({"prop_id": "giraffe", "name_ko": "기린", "aliases_ko": "기린 인형"}, "aliases_ko"),
    ({"prop_id": "giraffe", "name_ko": "기린", "aliases_ko": ["", "인형"]}, "aliases_ko"),
    ("giraffe", "중괄호"),
])
def test_malformed_entries_are_refused_in_korean(tmp_path, entry, words):
    with pytest.raises(booth_props.BoothPropsError, match=words):
        booth_props.load(_list(tmp_path, [entry]), catalog_dir=_catalog(tmp_path))
