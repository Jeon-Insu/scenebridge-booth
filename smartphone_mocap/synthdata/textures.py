"""Fetch CC0 PBR textures from ambientCG for the synthetic table, floor and walls.

Textures are drawn per scene from several categories so the detector does not learn
one particular cloth or floor. Images are not committed; ``manifest.json`` is, so the
set can be fetched again exactly.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

API = "https://ambientcg.com/api/v2/full_json"
LICENSE = "CC0 1.0 (ambientCG)"
USER_AGENT = "scenebridge-texture-fetch/1.0"
# Category → which surfaces it may dress.
CATEGORY_GROUPS = {"Fabric": ("table",), "Wood": ("table", "floor"),
                   "Tiles": ("floor", "wall"), "Concrete": ("floor", "wall")}
GROUPS = {group: tuple(c for c, groups in CATEGORY_GROUPS.items() if group in groups)
          for group in ("table", "floor", "wall")}
MAP_SUFFIXES = {"color": "_Color.jpg", "roughness": "_Roughness.jpg", "normal": "_NormalGL.jpg"}


def search_url(category: str, limit: int) -> str:
    return (f"{API}?type=Material&category={category}&limit={int(limit)}"
            f"&sort=Popular&include=downloadData")


def parse_search(payload: dict, resolution: str = "1K-JPG") -> list[tuple[str, str]]:
    found = []
    for asset in payload.get("foundAssets", []):
        downloads = (asset.get("downloadFolders", {}).get("default", {})
                     .get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", []))
        link = next((d["downloadLink"] for d in downloads if d.get("attribute") == resolution), None)
        if link:
            found.append((asset["assetId"], link))
    return found


def pick_maps(names: list[str]) -> dict[str, str]:
    picked = {}
    for key, suffix in MAP_SUFFIXES.items():
        match = next((name for name in names if name.endswith(suffix)), None)
        if match:
            picked[key] = match
    return picked


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch(out_dir: Path, per_category: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for category, groups in CATEGORY_GROUPS.items():
        payload = json.loads(_get(search_url(category, per_category)))
        for asset_id, link in parse_search(payload):
            target = out_dir / asset_id
            if not all((target / f"{key}.jpg").exists() for key in MAP_SUFFIXES):
                archive = zipfile.ZipFile(io.BytesIO(_get(link)))
                maps = pick_maps(archive.namelist())
                if set(maps) != set(MAP_SUFFIXES):
                    print(f"건너뜀 {asset_id}: 필요한 맵 없음 {sorted(maps)}")
                    continue
                target.mkdir(exist_ok=True)
                for key, name in maps.items():
                    (target / f"{key}.jpg").write_bytes(archive.read(name))
            entries.append({"id": asset_id, "category": category, "groups": list(groups),
                            "source": link})
            print(f"{category:9s} {asset_id}")
    manifest = {"license": LICENSE, "textures": entries}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("assets/textures"))
    parser.add_argument("--per-category", type=int, default=6)
    args = parser.parse_args()
    manifest = fetch(args.out, args.per_category)
    print(f"{len(manifest['textures'])}개 질감 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
