from smartphone_mocap.publisher import make_tracking_message
from smartphone_mocap.reconstruction import reconstruct_primary_person
from smartphone_mocap.synthetic import make_four_camera_rig, project_synthetic_person, synthetic_joint_positions
from smartphone_mocap.types import SceneObject3D


def test_four_view_synthetic_pipeline_produces_complete_message():
    rig = make_four_camera_rig()
    positions = synthetic_joint_positions(12)
    views = {
        name: [project_synthetic_person(camera, positions)]
        for name, camera in rig.items()
    }
    person = reconstruct_primary_person(views, rig)
    assert person is not None
    assert len(person.joints) == 17
    message = make_tracking_message(1234, 12, [person])
    assert message["schema"] == "gist.smartphone_mocap.v1"
    assert message["coordinate_frame"] == "BoothFrame"
    assert len(message["people"][0]["joints"]) == 17


def test_tracking_message_encodes_detected_object_asset():
    scene_object = SceneObject3D(
        global_id="bottle_0",
        label="bottle",
        xyz=(0.18, 0.21, 0.0),
        confidence=0.9,
        reprojection_error_px=2.0,
        source_cameras=("a", "b"),
        asset_uri="/tmp/bottle.usda",
    )
    placement = {"translation_m": [1.0, 2.0, 0.0], "yaw_deg": 90.0}
    message = make_tracking_message(1234, 0, [], [scene_object], placement)
    assert message["objects"][0]["id"] == "bottle_0"
    assert message["objects"][0]["asset_uri"] == "/tmp/bottle.usda"
    assert message["rig_to_scene"] == placement
