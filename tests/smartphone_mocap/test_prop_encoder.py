import pytest
from PIL import Image

from smartphone_mocap.prop_encoder import FILL_RGB, flatten_alpha


def test_transparent_background_becomes_mean_pixel_not_black():
    """Black is itself a strong signal and skews the embedding, so cutouts get the mean fill."""
    cutout = Image.new('RGBA', (4, 4), (0, 0, 0, 0))

    flattened = flatten_alpha(cutout)

    assert flattened.mode == 'RGB'
    assert flattened.getpixel((0, 0)) == FILL_RGB


def test_the_object_itself_keeps_its_colour():
    cutout = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
    cutout.putpixel((1, 1), (10, 200, 30, 255))

    assert flatten_alpha(cutout).getpixel((1, 1)) == (10, 200, 30)


def test_a_photo_without_alpha_passes_through():
    photo = Image.new('RGB', (4, 4), (10, 20, 30))

    assert flatten_alpha(photo).getpixel((0, 0)) == (10, 20, 30)
