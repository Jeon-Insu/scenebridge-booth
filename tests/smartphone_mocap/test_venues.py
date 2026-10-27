# tests/smartphone_mocap/test_venues.py
import json
import subprocess
import sys
from pathlib import Path

import pytest

from smartphone_mocap import venues

ROOT = Path(__file__).resolve().parents[2]


def _file(tmp_path, background):
    path = tmp_path / "venues.json"
    path.write_text(json.dumps({"venues": {"room": {"background": str(background), "offset_m": [1, 2, .75],
                                                    "yaw_deg": 90, "note": "x"}}}), encoding="utf-8")
    return path


def test_a_venue_gives_isaac_arguments(tmp_path):
    bg = tmp_path / "bg.usda"
    bg.write_text("#usda 1.0\n")
    venue = venues.load_venue("room", _file(tmp_path, bg))
    args = venues.isaac_args(venue)
    assert f"--/exts/gist.tabletop_twin/background_stage_path={bg}" in args
    assert "--/exts/gist.tabletop_twin/world_offset_z=0.75" in args and venue.yaw_deg == 90


def test_unknown_names_and_missing_backgrounds_are_refused_in_korean(tmp_path):
    path = _file(tmp_path, tmp_path / "missing.usda")
    with pytest.raises(venues.VenueError, match="장소"):
        venues.load_venue("hall", path)
    with pytest.raises(venues.VenueError, match="배경"):
        venues.load_venue("room", path)


def test_the_project_preset_is_the_dongshin_classroom():
    # The background USD is delivered separately, so read the preset without requiring it.
    venue = venues.load_venue("dongshin_ai_classroom", check_files=False)
    assert venue.offset_m == (45.011, 2.92, 4.535) and venue.yaw_deg == 90.0  # 1.8 m table


def test_the_tracker_can_read_a_venue_whose_background_is_not_installed(tmp_path):
    venue = venues.load_venue("room", _file(tmp_path, tmp_path / "missing.usda"), check_files=False)
    assert venue.yaw_deg == 90


def test_venue_args_cli_prints_one_argument_per_line(tmp_path):
    background = tmp_path / "bg.usda"
    background.write_text("#usda 1.0\n")
    out = subprocess.run([sys.executable, str(ROOT / "tools/venue_args.py"), "room",
                          "--venues", str(_file(tmp_path, background))],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0].startswith("--/exts/gist.tabletop_twin/background_stage_path=") and len(out) == 4


def test_venue_args_cli_refuses_a_background_that_is_not_installed(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/venue_args.py"), "room",
                             "--venues", str(_file(tmp_path, tmp_path / "missing.usda"))],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "배경 파일이 없습니다" in result.stderr


def test_table_points_turn_into_world_points():
    venue = venues.Venue("room", Path("/x.usda"), (10.0, 20.0, 4.0), 90.0)
    assert venues.table_to_world(venue, (1.0, 0.0, 0.5)) == pytest.approx((10.0, 21.0, 4.5))


def test_a_venue_camera_becomes_world_eye_and_target_settings(tmp_path):
    bg = tmp_path / "bg.usda"
    bg.write_text("#usda 1.0\n")
    path = tmp_path / "venues.json"
    path.write_text(json.dumps({"venues": {"room": {"background": str(bg), "offset_m": [10, 20, 4], "yaw_deg": 90,
                                                    "camera": {"eye": [0, -1, 1], "target": [0, 1, 0]}}}}))
    args = venues.isaac_args(venues.load_venue("room", path))
    assert "--/exts/gist.tabletop_twin/camera_eye_x=11" in args and "--/exts/gist.tabletop_twin/camera_eye_z=5" in args
    assert "--/exts/gist.tabletop_twin/camera_target_x=9" in args and len(args) == 10


def test_a_wrapper_whose_source_usd_is_missing_is_refused(tmp_path):
    wrapper = tmp_path / "wrap.usda"
    wrapper.write_text('#usda 1.0\n(\n    subLayers = [@/nowhere/Dongshin_Final.usda@]\n)\n')
    with pytest.raises(venues.VenueError, match="원본"):
        venues.load_venue("room", _file(tmp_path, wrapper))


def test_the_launcher_reads_the_venue_from_the_tracker_config(tmp_path):
    config = tmp_path / "booth.json"
    config.write_text(json.dumps({"venue": "room"}))
    background = tmp_path / "bg.usda"
    background.write_text("#usda 1.0\n")
    out = subprocess.run([sys.executable, str(ROOT / "tools/venue_args.py"), "--config", str(config),
                          "--venues", str(_file(tmp_path, background))],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert any("world_offset_x=1" in line for line in out)


def test_isaac_gets_the_table_size_too(tmp_path):
    background = tmp_path / "bg.usda"
    background.write_text("#usda 1.0\n")
    args = venues.isaac_args(venues.load_venue("room", _file(tmp_path, background)), table_size_m=(1.8, 0.75))
    assert "--/exts/gist.tabletop_twin/table_length_m=1.8" in args
    assert "--/exts/gist.tabletop_twin/table_width_m=0.75" in args


def test_the_launcher_passes_the_booth_table_size(tmp_path):
    config = tmp_path / "booth.json"
    config.write_text(json.dumps({"venue": "room", "rig": {"workspace_size_m": [1.8, 0.75, 0.35]}}))
    background = tmp_path / "bg.usda"
    background.write_text("#usda 1.0\n")
    out = subprocess.run([sys.executable, str(ROOT / "tools/venue_args.py"), "--config", str(config),
                          "--venues", str(_file(tmp_path, background))],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert "--/exts/gist.tabletop_twin/table_length_m=1.8" in out


def test_a_venue_cutaway_file_reaches_isaac(tmp_path):
    background = tmp_path / "bg.usda"
    background.write_text("#usda 1.0\n")
    path = tmp_path / "venues.json"
    path.write_text(json.dumps({"venues": {"room": {"background": str(background), "offset_m": [1, 2, .75],
                                                    "cutaway": str(tmp_path / "cut.json")}}}), encoding="utf-8")
    args = venues.isaac_args(venues.load_venue("room", path))
    assert f"--/exts/gist.tabletop_twin/cutaway_path={tmp_path / 'cut.json'}" in args
