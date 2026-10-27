"""Place promoted props into the twin's convention (Z-up, metres, base at Z=0).

Retrieval hands ``asset_uri`` straight to the extension, which references it as-is — only the
Asset Studio path runs ``normalized_asset``. A raw Meshy export is Y-up in feet with a centred
origin, so without this step a retrieved prop shows up oversized and lying on its side.

Reuses the extension's own ``normalized_asset`` rather than reimplementing the transform.
Run through tools/normalize_prop_assets.sh, which supplies Pixar USD from Isaac Sim.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartphone_mocap.prop_catalog import PropCatalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROPS = ROOT / 'assets' / 'props'
EXTENSION_MODULE = ROOT / 'exts/gist.tabletop_twin/gist/tabletop_twin/asset_studio.py'
PLACED_NAME = 'model_placed.usda'


def load_normalizer():
    """Import the module file directly; the package __init__ pulls in Kit-only modules."""
    spec = importlib.util.spec_from_file_location('ext_asset_studio', EXTENSION_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalized_asset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--props', type=Path, default=PROPS)
    parser.add_argument('--force', action='store_true', help='이미 정규화된 자산도 다시 만듭니다')
    args = parser.parse_args()

    catalog = PropCatalog(args.props)
    normalized_asset = load_normalizer()
    done = 0
    for entry in catalog.entries():
        asset_id = entry['asset_id']
        if entry.get('source_usd_uri') and not args.force:
            print(f'  {asset_id}: 이미 정규화됨 — 건너뜀')
            continue
        source = args.props / (entry.get('source_usd_uri') or entry['usd_uri'])
        if not source.is_file():
            print(f'  {asset_id}: 원본 없음 {source}', file=sys.stderr)
            continue
        output = source.parent / PLACED_NAME
        output.unlink(missing_ok=True)
        normalized_asset(str(source), str(output), float(entry['height_m']))
        # Keep the repo portable: reference the source next to the placed file, not this machine's path.
        output.write_text(output.read_text(encoding='utf-8').replace(f'@{source}@', f'@./{source.name}@'), encoding='utf-8')
        catalog.set_usd_uri(asset_id, f'{asset_id}/{PLACED_NAME}')
        print(f'  {asset_id}: 높이 {entry["height_m"]} m로 정규화 → {output.name}')
        done += 1
    print(f'정규화 완료: {done}개')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
