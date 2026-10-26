"""부스 소품 목록(config/booth_props.json) — v1.2: 학습한 소품(assets/props/catalog.json)만.

학생은 prop_id(학습 소품 이름), 한국어 이름, 다른 이름, 개수만 적는다. 3D 모델·높이·대칭 여부는
카탈로그에서 가져온다(높이는 tools/set_prop_height.py로 바꾼다).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "booth_props.json"
CATALOG_DIR = ROOT / "assets" / "props"
SYMMETRIC_IDS = frozenset({"paper_cup", "stainless_mug"})
# Generic words the scene agent has always understood; the booth list's own names win over them.
BUILTIN_ALIASES = {"화분": "potted plant", "식물": "potted plant", "컵": "cup", "책": "book", "시계": "clock"}


class BoothPropsError(ValueError):
    pass


@dataclass(frozen=True)
class BoothProp:
    prop_id: str
    name_ko: str
    aliases_ko: tuple[str, ...]
    count: int
    model: Path
    height_m: float
    symmetric: bool


def catalog(catalog_dir=None) -> list[dict]:
    path = Path(catalog_dir or CATALOG_DIR) / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))["assets"]


def load(path=None, *, catalog_dir=None) -> list[BoothProp]:
    path = Path(path or DEFAULT_PATH)
    root = Path(catalog_dir or CATALOG_DIR)
    if not path.is_file():
        raise BoothPropsError(f"소품 목록 파일이 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BoothPropsError(f"소품 목록의 {exc.lineno}번째 줄 형식이 잘못됐습니다: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("props"), list):
        raise BoothPropsError('소품 목록에 "props" 목록이 없습니다')
    entries = {e["asset_id"]: e for e in catalog(root)}
    props = []
    for index, raw in enumerate(payload["props"], start=1):
        if not isinstance(raw, dict):
            raise BoothPropsError(f"{index}번째 소품은 {{ }}(중괄호) 안에 적어야 합니다")
        aliases = raw.get("aliases_ko", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
            raise BoothPropsError(f'{index}번째 소품의 aliases_ko는 ["이름1", "이름2"]처럼 비어 있지 않은 이름 목록이어야 합니다')
        prop_id = raw.get("prop_id")
        if prop_id not in entries:
            raise BoothPropsError(f"{index}번째 소품 '{prop_id}'은 학습한 소품이 아닙니다 "
                                  "(tools/booth_check.py --props 로 목록 확인)")
        if not raw.get("name_ko"):
            raise BoothPropsError(f"{index}번째 소품 '{prop_id}'에 name_ko(한국어 이름)가 없습니다")
        count = raw.get("count", 1)
        if not isinstance(count, int) or count < 1:
            raise BoothPropsError(f"{index}번째 소품 '{prop_id}'의 count는 1 이상의 정수여야 합니다")
        entry = entries[prop_id]
        props.append(BoothProp(prop_id, str(raw["name_ko"]), tuple(aliases), count,
                               root / entry["usd_uri"], float(entry["height_m"]), prop_id in SYMMETRIC_IDS))
    return props


def check(props) -> list[str]:
    problems, seen_ids, words = [], set(), {}
    for prop in props:
        if prop.prop_id in seen_ids:
            problems.append(f"'{prop.prop_id}'이 목록에 두 번 있습니다. 여러 개면 count를 늘리세요")
        seen_ids.add(prop.prop_id)
        for word in (prop.name_ko, *prop.aliases_ko):
            other = words.setdefault(word, prop.prop_id)
            if other != prop.prop_id:
                problems.append(f"이름 '{word}'이 '{other}'와 '{prop.prop_id}'에 함께 쓰였습니다")
    return problems


def korean_aliases(path=None, *, catalog_dir=None) -> dict[str, str]:
    try:
        props = load(path, catalog_dir=catalog_dir)
        names = {e["asset_id"]: e["display_name_ko"] for e in catalog(catalog_dir)}
    except (BoothPropsError, OSError, KeyError, json.JSONDecodeError):
        return dict(BUILTIN_ALIASES)
    aliases: dict[str, str] = {}
    for prop in props:
        for word in (prop.name_ko, *prop.aliases_ko, names.get(prop.prop_id)):
            if word:
                aliases.setdefault(word, prop.prop_id)
    for word, label in BUILTIN_ALIASES.items():
        aliases.setdefault(word, label)
    return aliases


def korean_names(prop_id, aliases=None) -> list[str]:
    aliases = korean_aliases() if aliases is None else aliases
    return [word for word, target in aliases.items() if target == prop_id]
