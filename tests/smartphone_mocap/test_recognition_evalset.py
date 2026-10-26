# tests/smartphone_mocap/test_recognition_evalset.py
import json

import cv2
import numpy as np

from smartphone_mocap.recognition.evalset import load_scenes
from smartphone_mocap.synthdata.scene_sampler import (CameraView, DistractorPlacement, Lighting,
                                                      PropPlacement, SceneSpec, write_specs)


def test_scene_ground_truth_links_masks_to_props_and_unlisted_objects(tmp_path):
    view = CameraView("cam_A", np.eye(4).tolist(), 500., 500., 32., 24., 64, 48)
    spec = SceneSpec(0, 0, (PropPlacement("giraffe", "giraffe", "/g.usda", .3, .2, 45., 90., "lying", 1., "giraffe", "y"),),
                     (view,), Lighting(0, 45, 500, 5000, 100), "t", "f", "w", (1.25, .75),
                     (DistractorPlacement("/can.usd", "unlisted", .9, .5, 0., 0.),))
    write_specs([spec], ["giraffe"], tmp_path / "specs.json")
    product = tmp_path / "raw/scene_00000/Replicator"
    (product / "rgb").mkdir(parents=True)
    (product / "instance_segmentation").mkdir()
    cv2.imwrite(str(product / "rgb/rgb_0000.png"), np.zeros((48, 64, 3), np.uint8))
    ids = np.zeros((48, 64), np.uint16)
    ids[5:10, 5:10], ids[20:30, 40:50] = 2, 3
    cv2.imwrite(str(product / "instance_segmentation/instance_segmentation_0000.png"), ids)
    (product / "instance_segmentation/instance_segmentation_semantics_mapping_0000.json").write_text(
        json.dumps({"0": {"class": "BACKGROUND"}, "2": {"class": "giraffe"}, "3": {"class": "unlisted"}}))

    scene = next(load_scenes(tmp_path / "specs.json", tmp_path / "raw", degrade_seed=None))

    giraffe, unlisted = sorted(scene.truth, key=lambda t: t.asset_id is None)
    assert (giraffe.asset_id, giraffe.pose, giraffe.yaw_deg) == ("giraffe", "lying", 45.0)
    assert giraffe.masks["cam_A"].sum() == 25     # unlisted outlines: test_each_unlisted_object_gets_its_own_outline
    assert unlisted.asset_id is None and unlisted.xyz[:2] == (.9, .5)
    assert scene.calibrations["cam_A"].camera_matrix[0, 0] == 500.


def test_rendered_placement_overrides_the_spec_position(tmp_path):
    """2026-10-25: a lying prop's footprint centre moves off its spec x,y by half its height."""
    view = CameraView("cam_A", np.eye(4).tolist(), 500., 500., 32., 24., 64, 48)
    spec = SceneSpec(0, 0, (PropPlacement("giraffe", "giraffe", "/g.usda", .3, .2, 45., 90., "lying", 1., "giraffe", "y"),),
                     (view,), Lighting(0, 45, 500, 5000, 100), "t", "f", "w", (1.25, .75),
                     (DistractorPlacement("/can.usd", "unlisted", .9, .5, 0., 0.),))
    write_specs([spec], ["giraffe"], tmp_path / "specs.json")
    product = tmp_path / "raw/scene_00000/Replicator"
    (product / "rgb").mkdir(parents=True)
    (product / "instance_segmentation").mkdir()
    cv2.imwrite(str(product / "rgb/rgb_0000.png"), np.zeros((48, 64, 3), np.uint8))
    cv2.imwrite(str(product / "instance_segmentation/instance_segmentation_0000.png"), np.zeros((48, 64), np.uint16))
    (product / "instance_segmentation/instance_segmentation_semantics_mapping_0000.json").write_text("{}")
    (tmp_path / "raw/scene_00000/placed.json").write_text(json.dumps(
        {"props": [{"centre": [.33, .25, .04]}], "distractors": [{"kind": "unlisted", "centre": [.91, .52, .05, .1]}]}))

    scene = next(load_scenes(tmp_path / "specs.json", tmp_path / "raw", degrade_seed=None))

    assert [t.xyz for t in scene.truth] == [(.33, .25, 0.0), (.91, .52, 0.0)]


def test_a_flat_unlisted_object_is_not_counted_as_truth(tmp_path):
    test_rendered_placement_overrides_the_spec_position(tmp_path)
    placed = tmp_path / "raw/scene_00000/placed.json"
    placed.write_text(json.dumps({"props": [{"centre": [.33, .25, .04, .18]}],
                                  "distractors": [{"kind": "unlisted", "centre": [.91, .52, .003, .007]}]}))
    scene = next(load_scenes(tmp_path / "specs.json", tmp_path / "raw", degrade_seed=None))
    assert [t.asset_id for t in scene.truth] == ["giraffe"]


def test_each_unlisted_object_gets_its_own_outline(tmp_path):
    """Review 2026-10-25: all unlisted objects share one label, so one mask overwrote the other."""
    down = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]]      # 1 m above, looking down
    view = CameraView("cam_A", down, 100., 100., 32., 24., 64, 48)
    spec = SceneSpec(0, 0, (), (view,), Lighting(0, 45, 500, 5000, 100), "t", "f", "w", (1.25, .75),
                     (DistractorPlacement("/a.usd", "unlisted", .1, -.1, 0., 0.),
                      DistractorPlacement("/b.usd", "unlisted", -.2, .1, 0., 0.)))
    write_specs([spec], [], tmp_path / "specs.json")
    product = tmp_path / "raw/scene_00000/Replicator"
    (product / "rgb").mkdir(parents=True)
    (product / "instance_segmentation").mkdir()
    cv2.imwrite(str(product / "rgb/rgb_0000.png"), np.zeros((48, 64, 3), np.uint8))
    ids = np.zeros((48, 64), np.uint16)
    ids[31:37, 39:45] = 3            # around (42, 34): the first object
    ids[11:17, 9:15] = 4             # around (12, 14): the second object
    cv2.imwrite(str(product / "instance_segmentation/instance_segmentation_0000.png"), ids)
    (product / "instance_segmentation/instance_segmentation_semantics_mapping_0000.json").write_text(
        json.dumps({"3": {"class": "unlisted"}, "4": {"class": "unlisted"}}))

    first, second = next(load_scenes(tmp_path / "specs.json", tmp_path / "raw", degrade_seed=None)).truth

    assert first.masks["cam_A"][34, 42] and not first.masks["cam_A"][14, 12]
    assert second.masks["cam_A"][14, 12] and not second.masks["cam_A"][34, 42]
