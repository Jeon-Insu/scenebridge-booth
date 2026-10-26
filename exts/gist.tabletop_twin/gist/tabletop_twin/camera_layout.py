"""Repack the synchronized 4-row paired mosaic without touching inference."""
import numpy as np


def split_camera_analysis_mosaic(paired):
    """Return detection and segmentation 2x2 grids from one coherent image.

    Input rows: [cam0 detection | cam0 mask], ..., [cam3 detection | cam3 mask].
    Output camera order: cam0/cam1 on top, cam2/cam3 below, in BOTH tabs.
    """
    if paired.ndim != 3 or paired.shape[2] not in (3, 4):
        raise ValueError("Expected a BGR/RGBA paired camera mosaic.")
    height, width = paired.shape[:2]
    if height < 4 or width < 2 or height % 4 or width % 2:
        raise ValueError("Expected exactly four equally sized detection/mask row pairs.")
    tile_height, tile_width = height // 4, width // 2
    grids = []
    for column in range(2):
        tiles = [paired[row*tile_height:(row+1)*tile_height,
                        column*tile_width:(column+1)*tile_width] for row in range(4)]
        grids.append(np.concatenate((np.concatenate(tiles[:2], axis=1),
                                     np.concatenate(tiles[2:], axis=1)), axis=0))
    return tuple(grids)
