# tests/smartphone_mocap/test_recognition_objects.py
import numpy as np

from smartphone_mocap.recognition.objects import apply_counts, crops_for
from smartphone_mocap.recognition.types import Identity, PropRef, Registry
from smartphone_mocap.types import BoundingBox2D, Object2D, SceneObject3D


def _obj(x, cams):
    return SceneObject3D("", "object", (x, 0.3, 0.0), .9, 1., cams)


def test_a_prop_listed_once_keeps_only_its_best_sighting():
    registry = Registry((PropRef("toy_duck", "오리", "toy duck", .08, ()),))
    named = [(_obj(.7, ("a", "b", "c", "d")), Identity("toy_duck", {"toy_duck": .8})),
             (_obj(.5, ("a", "b")), Identity("toy_duck", {"toy_duck": .9}))]
    kept = apply_counts(named, registry)
    assert [i.asset_id for _, i in kept] == ["toy_duck", None]     # the 4-camera one wins


def test_crops_grey_out_the_background_and_survive_a_single_camera():
    image = np.full((50, 50, 3), 255, np.uint8)
    mask = np.zeros((50, 50), bool)
    mask[20:30, 20:30] = True
    det = Object2D("a", 0, "object", 0, BoundingBox2D(20, 20, 30, 30), .9, mask=mask, asset_id="x")
    crops = crops_for([det], {"a": image}, pad=4)
    assert len(crops) == 1 and crops[0].hint == "x"
    assert crops[0].image.shape[:2] == (18, 18)
    assert tuple(crops[0].image[0, 0]) == (114, 114, 114) and tuple(crops[0].image[9, 9]) == (255, 255, 255)
    assert crops_for([], {"a": image}) == []
