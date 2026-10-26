import numpy as np

from smartphone_mocap.calibration import look_at_world_to_camera
from smartphone_mocap.synthdata.scene_sampler import (CameraView, SceneSpec, perturb_camera, read_specs,
                                                      sample_scene, write_specs)

CATALOG = [
    {"asset_id": "green_robot", "detector_class": "toy robot", "usd_path": "/props/robot.usda"},
    {"asset_id": "paper_cup", "detector_class": "cup", "usd_path": "/props/cup.usda"},
]
TEXTURES = {"table": ["/tex/fabric"], "floor": ["/tex/tiles"], "wall": ["/tex/concrete"]}
TARGET = np.array([0.9, 0.375, 0.0])


def _view():
    w2c = look_at_world_to_camera(np.array([0.3, 0.1, 0.8]), TARGET)
    return CameraView("cam_B", w2c.tolist(), 926.0, 932.0, 650.0, 324.0, 1280, 720)


def _angle_deg(a, b):
    r = np.asarray(a)[:3, :3] @ np.asarray(b)[:3, :3].T
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


def _centre(view):
    w2c = np.asarray(view.world_to_camera)
    return -w2c[:3, :3].T @ w2c[:3, 3]


def test_perturbed_camera_stays_within_limits_and_sees_the_table():
    base = _view()
    rng = np.random.default_rng(3)
    for _ in range(200):
        moved = perturb_camera(base, rng, TARGET)
        assert _angle_deg(moved.world_to_camera, base.world_to_camera) <= 10.0 + 1e-6
        assert np.linalg.norm(_centre(moved) - _centre(base)) <= 0.15 + 1e-9
        w2c = np.asarray(moved.world_to_camera)
        p = w2c[:3, :3] @ TARGET + w2c[:3, 3]
        u, v = moved.fx * p[0] / p[2] + moved.cx, moved.fy * p[1] / p[2] + moved.cy
        assert p[2] > 0 and 0 <= u < 1280 and 0 <= v < 720


def test_unreachable_target_keeps_the_original_camera():
    base = _view()
    behind = np.array([-50.0, -50.0, 5.0])
    assert perturb_camera(base, np.random.default_rng(0), behind) == base


def test_props_stay_on_the_table_and_apart():
    for scene_id in range(50):
        spec = sample_scene(scene_id, 11, CATALOG, [_view()], TEXTURES)
        for prop in spec.props:
            assert 0.10 <= prop.x_m <= 1.70 and 0.10 <= prop.y_m <= 0.65
        positions = [(p.x_m, p.y_m) for p in spec.props]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert np.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]) >= 0.08


def test_lying_and_upside_down_poses_are_sampled():
    poses = [p.pose for scene_id in range(400)
             for p in sample_scene(scene_id, 5, CATALOG, [_view()], TEXTURES).props]
    lying = poses.count("lying") / len(poses)
    assert 0.3 < lying < 0.5 and "upside_down" in poses   # spec 2026-10-25: lying 0.4


def test_same_seed_same_scene_and_json_roundtrip(tmp_path):
    a = sample_scene(4, 99, CATALOG, [_view()], TEXTURES)
    b = sample_scene(4, 99, CATALOG, [_view()], TEXTURES)
    assert a == b
    path = tmp_path / "specs.json"
    write_specs([a], ["cup", "toy robot"], path)
    specs, categories = read_specs(path)
    assert specs == [a] and categories == ["cup", "toy robot"]
    assert isinstance(specs[0], SceneSpec)


SIZED_CATALOG = [
    {"asset_id": "elephant", "detector_class": "elephant", "usd_path": "/props/elephant.usda", "height_m": 0.448},
    {"asset_id": "devil_duck", "detector_class": "toy duck", "usd_path": "/props/duck.usda", "height_m": 0.018},
]


def test_props_are_scaled_to_their_target_height():
    # Meshy exports arrive at arbitrary sizes (a 45 cm elephant beside a 2 cm duck);
    # synthetic scenes must render props at believable sizes.
    heights = {"elephant": 0.12, "devil_duck": 0.08}
    for scene_id in range(20):
        for prop in sample_scene(scene_id, 2, SIZED_CATALOG, [_view()], TEXTURES, target_heights=heights).props:
            source = 0.448 if prop.asset_id == "elephant" else 0.018
            assert abs(prop.scale * source - heights[prop.asset_id]) < 1e-9


def test_props_without_a_target_height_keep_their_size():
    for prop in sample_scene(0, 2, SIZED_CATALOG, [_view()], TEXTURES).props:
        assert prop.scale == 1.0


def test_asset_class_unit_labels_props_by_asset_id():
    from smartphone_mocap.synthdata.scene_sampler import training_categories
    assert training_categories(CATALOG, "asset") == ["green_robot", "paper_cup"]
    props = [p for scene_id in range(10)
             for p in sample_scene(scene_id, 1, CATALOG, [_view()], TEXTURES, class_unit="asset").props]
    assert all(p.train_class == p.asset_id for p in props)


def test_detector_class_unit_keeps_the_previous_labels():
    from smartphone_mocap.synthdata.scene_sampler import training_categories
    assert training_categories(CATALOG, "detector") == ["cup", "toy robot"]
    props = sample_scene(0, 1, CATALOG, [_view()], TEXTURES).props
    assert all(p.train_class == p.detector_class for p in props)
    assert sample_scene(0, 1, CATALOG, [_view()], TEXTURES) == sample_scene(
        0, 1, CATALOG, [_view()], TEXTURES, class_unit="detector")


def test_specs_written_before_train_class_existed_still_load(tmp_path):
    import json
    scene = sample_scene(4, 99, CATALOG, [_view()], TEXTURES)
    path = tmp_path / "specs.json"
    write_specs([scene], ["cup", "toy robot"], path)
    payload = json.loads(path.read_text())
    for prop in payload["scenes"][0]["props"]:
        del prop["train_class"]
    path.write_text(json.dumps(payload))
    specs, _ = read_specs(path)
    assert [p.train_class for p in specs[0].props] == [p.detector_class for p in scene.props]


def test_lighting_stays_in_the_venue_brightness_range():
    # 2026-10-10: 500-5000 sun / 100-1500 dome rendered a median grey of 193 (venue ~100-130).
    from smartphone_mocap.synthdata.scene_sampler import AMBIENT_INTENSITY, SUN_INTENSITY
    for scene_id in range(50):
        light = sample_scene(scene_id, 3, CATALOG, [_view()], TEXTURES).lighting
        assert SUN_INTENSITY[0] <= light.sun_intensity <= SUN_INTENSITY[1] <= 1800.0
        assert AMBIENT_INTENSITY[0] <= light.ambient_intensity <= AMBIENT_INTENSITY[1] <= 300.0


def _project(view, point):
    w2c = np.asarray(view.world_to_camera)
    p = w2c[:3, :3] @ np.asarray(point, float) + w2c[:3, 3]
    return p[2], view.fx * p[0] / p[2] + view.cx, view.fy * p[1] / p[2] + view.cy


def test_random_rig_surrounds_the_table_within_distance_and_height_limits():
    from smartphone_mocap.synthdata.scene_sampler import RIG_DISTANCE_M, RIG_HEIGHT_M, random_rig
    templates = [_view(), _view(), _view(), _view()]
    rng = np.random.default_rng(1)
    for _ in range(100):
        table = (float(rng.uniform(1.2, 2.4)), float(rng.uniform(0.6, 1.2)))
        rig = random_rig(rng, templates, table)
        centre = np.array([table[0] / 2, table[1] / 2, 0.0])
        azimuths = []
        for view in rig:
            c = _centre(view)
            horizontal = np.hypot(*(c - centre)[:2])
            assert RIG_DISTANCE_M[0] - 1e-9 <= horizontal <= RIG_DISTANCE_M[1] + 1e-9
            assert RIG_HEIGHT_M[0] - 1e-9 <= c[2] <= RIG_HEIGHT_M[1] + 1e-9
            depth, u, v = _project(view, centre)
            assert depth > 0 and 0 <= u < view.width and 0 <= v < view.height
            azimuths.append(np.degrees(np.arctan2(c[1] - centre[1], c[0] - centre[0])) % 360)
        gaps = np.diff(sorted(azimuths) + [sorted(azimuths)[0] + 360])
        assert gaps.max() < 180                     # no half of the table is unwatched


def test_random_rig_keeps_camera_ids_and_image_size():
    from smartphone_mocap.synthdata.scene_sampler import random_rig
    templates = [replace_view(_view(), camera_id) for camera_id in ("cam_A", "cam_B", "cam_C", "cam_D")]
    rig = random_rig(np.random.default_rng(0), templates, (1.8, 0.75))
    assert [v.camera_id for v in rig] == ["cam_A", "cam_B", "cam_C", "cam_D"]
    assert all((v.width, v.height) == (1280, 720) for v in rig)


def replace_view(view, camera_id):
    from dataclasses import replace
    return replace(view, camera_id=camera_id)


def test_mixed_layout_varies_table_size_and_uses_both_rig_kinds():
    tables, venue_like = set(), 0
    base = _view()
    for scene_id in range(60):
        spec = sample_scene(scene_id, 9, CATALOG, [base], TEXTURES, camera_layout="mixed")
        tables.add(spec.table_size_m)
        venue_like += np.linalg.norm(_centre(spec.cameras[0]) - _centre(base)) <= 0.15 + 1e-9
        for prop in spec.props:
            assert 0 < prop.x_m < spec.table_size_m[0] and 0 < prop.y_m < spec.table_size_m[1]
    assert len(tables) > 30 and 5 < venue_like < 40


def test_venue_layout_and_old_specs_keep_the_fixed_table(tmp_path):
    import json
    spec = sample_scene(0, 1, CATALOG, [_view()], TEXTURES)
    assert spec.table_size_m == (1.8, 0.75)
    path = tmp_path / "specs.json"
    write_specs([spec], ["cup"], path)
    payload = json.loads(path.read_text())
    del payload["scenes"][0]["table_size_m"]
    path.write_text(json.dumps(payload))
    assert read_specs(path)[0][0].table_size_m == (1.8, 0.75)


def test_random_rig_aims_at_the_table_centre():
    # 2026-10-10 user: cameras look at the table centre; the aim used to wander +-30%.
    from smartphone_mocap.synthdata.scene_sampler import random_rig
    rng = np.random.default_rng(5)
    for _ in range(100):
        table = (float(rng.uniform(1.2, 2.4)), float(rng.uniform(0.6, 1.2)))
        centre = np.array([table[0] / 2, table[1] / 2, 0.0])
        for view in random_rig(rng, [_view()] * 4, table):
            w2c = np.asarray(view.world_to_camera)
            position = -w2c[:3, :3].T @ w2c[:3, 3]
            forward = w2c[2, :3]
            hit = position + forward * (-position[2] / forward[2])
            assert np.hypot(*(hit - centre)[:2]) <= 0.05 * np.sqrt(2) + 1e-9


DISTRACTORS = [
    {"usd": "/nv/person_a.usd", "kind": "person", "split": "train", "meters_per_unit": 0.01, "up_axis": "Y",
     "size_m": [0.5, 0.3, 1.7]},
    {"usd": "/nv/person_b.usd", "kind": "person", "split": "val", "meters_per_unit": 0.01, "up_axis": "Y",
     "size_m": [0.5, 0.3, 1.7]},
    {"usd": "/nv/chair.usd", "kind": "furniture", "split": "train", "meters_per_unit": 0.01, "up_axis": "Z",
     "size_m": [0.6, 0.6, 0.9]},
    {"usd": "/nv/tv.usd", "kind": "electronics", "split": "train", "meters_per_unit": 1.0, "up_axis": "Z",
     "size_m": [0.9, 0.2, 0.6]},
    {"usd": "/nv/mug.usd", "kind": "clutter", "split": "train", "meters_per_unit": 1.0, "up_axis": "Z",
     "size_m": [0.1, 0.1, 0.1]},
    {"usd": "/nv/can.usd", "kind": "clutter", "split": "val", "meters_per_unit": 1.0, "up_axis": "Z",
     "size_m": [0.07, 0.07, 0.1]},
]


def _scene_with_distractors(scene_id, split="train"):
    return sample_scene(scene_id, 11, CATALOG, [_view()] * 4, TEXTURES, camera_layout="mixed",
                        distractors=DISTRACTORS, distractor_split=split)


def test_distractors_come_only_from_the_requested_split():
    for split in ("train", "val"):
        allowed = {d["usd"] for d in DISTRACTORS if d["split"] == split}
        used = {d.usd_path for i in range(40) for d in _scene_with_distractors(i, split).distractors}
        assert used and used <= allowed


def test_people_and_furniture_stand_on_the_floor_off_the_table():
    from smartphone_mocap.synthdata.scene_sampler import FLOOR_Z_M
    seen = set()
    for i in range(60):
        spec = _scene_with_distractors(i)
        w, d = spec.table_size_m
        for item in spec.distractors:
            seen.add(item.kind)
            on_table = 0 <= item.x_m <= w and 0 <= item.y_m <= d
            if item.kind in ("person", "furniture"):
                assert item.surface_z_m == FLOOR_Z_M and not on_table
            if item.kind == "clutter":
                assert item.surface_z_m == 0.0 and on_table
    assert seen == {"person", "furniture", "electronics", "clutter"}


def test_distractors_keep_clear_of_cameras_and_props():
    from smartphone_mocap.synthdata.scene_sampler import CAMERA_CLEARANCE_M, MIN_SEPARATION_M
    for i in range(60):
        spec = _scene_with_distractors(i)
        centres = [_centre(v) for v in spec.cameras]
        for item in spec.distractors:
            assert all(np.hypot(item.x_m - c[0], item.y_m - c[1]) >= CAMERA_CLEARANCE_M for c in centres)
            if item.kind == "clutter":
                assert all(np.hypot(item.x_m - p.x_m, item.y_m - p.y_m) >= MIN_SEPARATION_M + 0.05
                           for p in spec.props)


def test_people_face_the_table():
    for i in range(40):
        spec = _scene_with_distractors(i)
        cx, cy = spec.table_size_m[0] / 2, spec.table_size_m[1] / 2
        for item in spec.distractors:
            if item.kind == "person":
                to_table = np.degrees(np.arctan2(cy - item.y_m, cx - item.x_m))
                assert abs((item.yaw_deg - to_table + 180) % 360 - 180) <= 40 + 1e-6


def test_no_distractor_index_keeps_scenes_unchanged_and_old_specs_load(tmp_path):
    import json
    plain = sample_scene(3, 11, CATALOG, [_view()] * 4, TEXTURES, camera_layout="mixed")
    assert plain.distractors == ()
    spec = _scene_with_distractors(3)
    assert (spec.props, spec.cameras, spec.lighting) == (plain.props, plain.cameras, plain.lighting)
    path = tmp_path / "specs.json"
    write_specs([spec], ["cup"], path)
    assert read_specs(path)[0][0] == spec
    payload = json.loads(path.read_text())
    del payload["scenes"][0]["distractors"]
    path.write_text(json.dumps(payload))
    assert read_specs(path)[0][0].distractors == ()


def test_nothing_taller_than_the_limit_is_put_on_the_table():
    # Probe 2026-10-10: a 1 m wall TV stood on the table and hid the props.
    from smartphone_mocap.synthdata.scene_sampler import ON_TABLE_MAX_HEIGHT_M
    tall_tv = {"usd": "/nv/tv_wall.usd", "kind": "electronics", "split": "train", "meters_per_unit": 1.0,
               "up_axis": "Z", "size_m": [1.2, 0.1, 0.9]}
    index = DISTRACTORS + [tall_tv]
    heights = {e["usd"]: e["size_m"][2] for e in index}
    for i in range(80):
        spec = sample_scene(i, 13, CATALOG, [_view()] * 4, TEXTURES, camera_layout="mixed",
                            distractors=index)
        for item in spec.distractors:
            if item.surface_z_m == 0.0:
                assert heights[item.usd_path] <= ON_TABLE_MAX_HEIGHT_M[item.kind]


def _flat_catalog():
    return [{"asset_id": f"p{i}", "detector_class": "toy", "usd_path": f"/p{i}.usda", "height_m": .1}
            for i in range(10)]


def test_lying_props_fall_over_on_four_sides():
    import numpy as np
    from smartphone_mocap.synthdata import scene_sampler as ss
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        for p in ss._sample_props(rng, _flat_catalog(), 6, (1.25, 0.75)):
            if p.pose == "lying":
                seen.add((p.roll_axis, p.roll_deg))
    assert seen == {("x", 90.0), ("x", -90.0), ("y", 90.0), ("y", -90.0)}


def test_unique_props_never_repeat_an_asset_in_a_scene():
    import numpy as np
    from smartphone_mocap.synthdata import scene_sampler as ss
    rng = np.random.default_rng(1)
    for _ in range(30):
        ids = [p.asset_id for p in ss._sample_props(rng, _flat_catalog(), 8, (1.25, 0.75), unique=True)]
        assert len(ids) == len(set(ids))
