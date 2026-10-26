"""Curated demo props: the gallery that instance retrieval searches.

Only promoted assets live here. A bad asset in the gallery can win a match and silently
break retrieval, so entries are added deliberately rather than on every generation.
"""
import json
from pathlib import Path

REQUIRED = ('asset_id', 'display_name_ko', 'usd_uri', 'detector_class', 'height_m', 'height_source')
# Keeps a measured dimension distinguishable from an estimate, as the rest of the project does.
HEIGHT_SOURCES = ('measured', 'meshy_auto_size', 'class_prior')
SCHEMA_VERSION = 1


class PropCatalog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / 'catalog.json'

    def _load(self) -> dict:
        if not self.path.is_file():
            return {'schema_version': SCHEMA_VERSION, 'assets': []}
        return json.loads(self.path.read_text(encoding='utf-8'))

    def entries(self) -> list[dict]:
        return list(self._load().get('assets', []))

    def get(self, asset_id: str) -> dict | None:
        return next((e for e in self.entries() if e.get('asset_id') == asset_id), None)

    def by_class(self, detector_class: str) -> list[dict]:
        wanted = detector_class.casefold()
        return [e for e in self.entries() if str(e.get('detector_class', '')).casefold() == wanted]

    def validate(self, entry: dict) -> None:
        """Check an entry without writing, so callers can fail before copying files."""
        missing = [field for field in REQUIRED if not entry.get(field)]
        if missing:
            raise ValueError(f'필수 항목이 없습니다: {", ".join(missing)}')
        if entry['height_source'] not in HEIGHT_SOURCES:
            raise ValueError(f'height_source는 {HEIGHT_SOURCES} 중 하나여야 합니다.')
        if not 0.005 <= float(entry['height_m']) <= 3:
            raise ValueError('height_m는 0.005~3 m 범위여야 합니다.')
        if self.get(entry['asset_id']) is not None:
            raise ValueError(f'asset_id "{entry["asset_id"]}"가 이미 있습니다.')

    def set_usd_uri(self, asset_id: str, usd_uri: str) -> dict:
        """Point an entry at its normalized asset, keeping the raw export recorded."""
        catalog = self._load()
        for entry in catalog['assets']:
            if entry.get('asset_id') != asset_id:
                continue
            entry.setdefault('source_usd_uri', entry['usd_uri'])
            entry['usd_uri'] = usd_uri
            self.path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
            return entry
        raise ValueError(f'asset_id "{asset_id}"가 카탈로그에 없습니다.')

    def add(self, entry: dict) -> dict:
        self.validate(entry)
        catalog = self._load()
        catalog['assets'].append(dict(entry))
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
        return entry
