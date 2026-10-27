"""Which building meshes a cut-away stage hides (tools/make_cutaway.py)."""
from smartphone_mocap.cutaway import above_cut, ceiling_attached


def test_only_meshes_entirely_above_the_cut_are_hidden():
    meshes = [("roof", 20.2, 23.4), ("wall_2f", 3.7, 7.0), ("pipe", 0.4, 20.2), ("wall_4f", 11.4, 14.6)]
    assert above_cut(meshes, 11.3) == ["roof", "wall_4f"]


def test_ceiling_units_go_but_high_photo_frames_stay():
    assert ceiling_attached(5.98, 6.76, ceiling_z=7.05)          # ceiling AC/light unit
    assert not ceiling_attached(5.83, 6.46, ceiling_z=7.05)      # upper row of wall photos
    assert not ceiling_attached(3.79, 4.6, ceiling_z=7.05)       # desk


def test_walls_reaching_the_ceiling_are_not_ceiling_units():
    assert not ceiling_attached(3.75, 7.05, ceiling_z=7.05)      # classroom wall, floor to ceiling
    assert ceiling_attached(6.98, 7.05, ceiling_z=7.05)          # the ceiling panel itself
