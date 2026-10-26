import numpy as np

from smartphone_mocap.synthdata.degrade import (bloom_whites, defocus, degrade, jpeg, lift_blacks,
                                                overexpose)


def _gradient():
    row = np.linspace(0, 255, 200).astype(np.uint8)
    return np.dstack([np.tile(row, (100, 1))] * 3)


def test_overexpose_increases_clipped_pixels():
    image = _gradient()
    assert np.mean(overexpose(image, 1.5) >= 250) > np.mean(image >= 250)


def test_bloom_grows_a_white_square_by_one_pixel():
    image = np.zeros((40, 40, 3), np.uint8)
    image[10:20, 10:20] = 255
    grown = bloom_whites(image, pixels=1)
    assert grown[9, 15].min() == 255 and grown[15, 20].min() == 255
    assert grown[30, 30].max() == 0


def test_lift_blacks_sets_the_floor_and_keeps_white():
    image = np.zeros((10, 10, 3), np.uint8)
    image[0, 0] = 255
    lifted = lift_blacks(image, 40)
    assert lifted[5, 5].min() == 40 and lifted[0, 0].min() == 255


def test_jpeg_and_defocus_keep_shape_and_dtype():
    image = _gradient()
    for out in (jpeg(image, 70), defocus(image, 1.2), defocus(image, 0)):
        assert out.shape == image.shape and out.dtype == np.uint8


def test_degrade_is_deterministic_for_a_seed():
    image = _gradient()
    a, applied_a = degrade(image, np.random.default_rng(7))
    b, applied_b = degrade(image, np.random.default_rng(7))
    assert applied_a == applied_b and np.array_equal(a, b)
    assert a.shape == image.shape and a.dtype == np.uint8
