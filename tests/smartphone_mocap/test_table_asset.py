"""The twin's table is generated from the booth table size (rig.workspace_size_m)."""
import re

from smartphone_mocap.table_asset import table_asset_name, table_usda


def test_the_table_top_matches_the_requested_size():
    text = table_usda(1.8, 0.75)
    assert "double3 xformOp:scale = (1.8, 0.75, 0.04)" in text
    assert "double3 xformOp:translate = (0.9, 0.375, -0.02)" in text


def test_legs_sit_inside_the_corners_of_a_long_table():
    xs = {float(x) for x in re.findall(r"translate = \(([\d.]+), [\d.]+, -0\.395\)", table_usda(1.8, 0.75))}
    assert xs == {0.09, 1.71}


def test_the_file_name_says_the_size_in_centimetres():
    assert table_asset_name(1.8, 0.75) == "table_180x75x75.usda"
    assert table_asset_name(1.25, 0.75) == "table_125x75x75.usda"


def test_side_rails_join_the_legs_along_the_long_edge():
    text = table_usda(1.8, 0.75)
    assert text.count("double3 xformOp:scale = (1.62, 0.035, 0.035)") == 2
    assert "double3 xformOp:translate = (0.9, 0.66, -0.18)" in text
