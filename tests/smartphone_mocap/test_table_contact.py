from itertools import product
import math

import pytest

from smartphone_mocap.table_contact import resolve_placement, inspect_contact
from smartphone_mocap.scene_agent import SceneTools, deterministic_response


OBJECT_ID = "cup/white/instance_000"
GEOMETRY = {"points_local_m": [list(p) for p in product((-.04, .04), (-.04, .04), (0, .10))]}
POSE = {"xyz": [.6, .35, 0], "rotation_wxyz": [1, 0, 0, 0], "scale": [1, 1, 1]}
TILT = [math.sqrt(.5), math.sqrt(.5), 0, 0]


def snapshot(z=0):
    return {"tracking": {"objects": [{"id": OBJECT_ID, "label": "cup", "xyz": [.6, .35, z]}]},
            "twin_state": {"runtime_objects": {OBJECT_ID: {"xyz": [.6, .35, z], "contact_geometry": GEOMETRY}}}}


def test_preserve_position_warns_but_never_snaps():
    result = resolve_placement(GEOMETRY, POSE, {"rotation_wxyz": TILT, "scale": [2, 2, 2]})
    assert result["target"]["xyz"] == POSE["xyz"]
    assert result["z_correction_m"] == 0
    assert result["contact_after"]["possible_penetration_m"] == pytest.approx(.08)
    assert "possible_table_penetration" in result["contact_after"]["warnings"]


def test_combined_contact_rotation_scale_and_shrink_are_not_cumulative():
    result = resolve_placement(GEOMETRY, POSE, {"rotation_wxyz": TILT, "scale": [2, 2, 2]}, "maintain_support")
    assert result["target"]["xyz"] == pytest.approx([.6, .35, .08])
    assert result["contact_after"]["clearance_m"] == pytest.approx(0)
    smaller = resolve_placement(GEOMETRY, result["target"], {"scale": [1, 1, 1]}, "maintain_support")
    assert smaller["target"]["xyz"][2] == pytest.approx(.04)
    upright = resolve_placement(GEOMETRY, smaller["target"], {"rotation_wxyz": [1, 0, 0, 0]}, "maintain_support")
    assert upright["target"]["xyz"][2] == pytest.approx(0)


def test_airborne_rotation_preserves_pivot_not_bottom_clearance():
    floating = {**POSE, "xyz": [.6, .35, .3]}
    result = resolve_placement(GEOMETRY, floating, {"rotation_wxyz": TILT})
    assert result["target"]["xyz"] == floating["xyz"]
    assert result["contact_before"]["clearance_m"] == .3
    assert result["contact_after"]["clearance_m"] == pytest.approx(.26)
    with pytest.raises(ValueError, match="not currently"):
        resolve_placement(GEOMETRY, floating, {"rotation_wxyz": TILT}, "maintain_support")
    landed = resolve_placement(GEOMETRY, result["target"], {}, "place_on_table")
    assert landed["target"]["xyz"] == pytest.approx([.6, .35, .04])


@pytest.mark.parametrize("geometry,pose,policy", [
    ({}, POSE, "place_on_table"),
    (GEOMETRY, {**POSE, "xyz": [0, 0, .3]}, "place_on_table"),
    (GEOMETRY, POSE, "magic"),
])
def test_invalid_contact_requests_fail_without_changing_input(geometry, pose, policy):
    original = list(pose["xyz"])
    with pytest.raises(ValueError):
        resolve_placement(geometry, pose, {}, policy)
    assert pose["xyz"] == original


def test_no_geometry_preserve_position_is_allowed_but_not_claimed_verified():
    result = resolve_placement({}, POSE, {"rotation_wxyz": TILT})
    assert result["target"]["xyz"] == POSE["xyz"]
    assert not result["contact_after"]["available"]


def test_shifted_asset_origin_can_be_below_table_without_penetration():
    geometry = {"points_local_m": [[x, y, z+.05] for x, y, z in GEOMETRY["points_local_m"]]}
    result = resolve_placement(geometry, POSE, {}, "place_on_table")
    assert result["target"]["xyz"][2] == pytest.approx(-.05)
    assert result["contact_after"]["state"] == "contact"


def test_agent_defaults_free_and_explicit_support_changes_height():
    tools = SceneTools(snapshot())
    result = tools.transform_object(OBJECT_ID, angle_deg=90, rotation_axis="x", scale_factor=2,
                                    placement="maintain_support")
    assert result["ok"]
    assert result["xyz"] == pytest.approx([.6, .35, .08])
    assert len(tools.actions) == 1
    assert tools.inspect_table_contact(OBJECT_ID)["contact"]["state"] == "contact"
    tools.change_height(OBJECT_ID, delta_m=.3)
    result = tools.rotate_object(OBJECT_ID, 90, axis="x")
    assert result["placement"] == "preserve_position"
    assert tools.list_objects()["objects"][0]["xyz"][2] == pytest.approx(.38)
    assert not tools.rotate_object(OBJECT_ID, 90, axis="x", placement="maintain_support")["ok"]


def test_preview_keeps_actual_state_and_commit_uses_solved_target():
    tools = SceneTools(snapshot())
    result = tools.rotate_object(OBJECT_ID, 90, axis="x", placement="maintain_support", preview_only=True)
    assert result["xyz"][2] == pytest.approx(.04)
    assert tools.list_objects()["objects"][0]["xyz"][2] == 0
    tools.apply_layout()
    assert tools.list_objects()["objects"][0]["xyz"][2] == pytest.approx(.04)


def test_explicit_korean_landing_uses_geometry_not_zero_pivot():
    value = snapshot(.3)
    value["twin_state"]["transforms"] = {OBJECT_ID: {"rotation_wxyz": TILT}}
    result = deterministic_response(f"{OBJECT_ID}을 테이블 위에 내려놔", value)
    proposal = result["actions"][0]["placements"][0]
    assert proposal["placement"] == "place_on_table"
    assert proposal["xyz"][2] == pytest.approx(.04)


def test_agent_invalid_geometry_or_placement_does_not_queue():
    tools = SceneTools(snapshot(.3))
    assert not tools.rotate_object(OBJECT_ID, 90, placement="maintain_support")["ok"]
    assert not tools.scale_object(OBJECT_ID, 2, placement="magic")["ok"]
    assert not tools.actions


def test_runtime_virtual_position_overrides_old_frozen_tracking_offset():
    value = snapshot(.3)
    value["tracking"]["objects"][0]["xyz"][2] = .1
    value["twin_state"]["offsets"] = {OBJECT_ID: [0, 0, .3]}
    assert SceneTools(value).list_objects()["objects"][0]["xyz"][2] == .3
