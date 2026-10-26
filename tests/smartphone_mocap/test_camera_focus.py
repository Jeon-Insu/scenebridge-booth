import base64
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from smartphone_mocap.camera_focus import (
    make_camera_links, read_linked_preview, render_camera_focus, resolve_focus_ids,
)
from smartphone_mocap.preview import make_camera_analysis_mosaic, write_preview_image, MosaicFilePublisher
from smartphone_mocap.types import BoundingBox2D, Object2D


def fixture():
    ids = ["a", "b", "c", "d"]
    frames = {cid: np.full((240, 320, 3), 120, np.uint8) for cid in ids[:3]}
    mask = np.zeros((240, 320), np.uint8)
    mask[60:140, 40:100] = 1
    mask[80:100, 60:80] = 0  # Preserve a hole in an instance mask.
    def detection(cid, index, second=False):
        return Object2D(cid, index, "cup", 1,
                        BoundingBox2D(200, 60, 280, 140) if second else BoundingBox2D(40, 60, 100, 140),
                        .9, mask=None if second else mask)
    detections = {"a": [detection("a", 0), detection("a", 1, True)],
                  "b": [detection("b", 0, True)], "c": [detection("c", 0), detection("c", 1)]}
    tracked = {(c, i): SimpleNamespace(global_id=oid) for c, i, oid in [
        ("a", 0, "cup/A"), ("a", 1, "cup/B"), ("b", 0, "cup/B"), ("c", 0, "cup/A"), ("c", 1, "cup/A")]}
    meta = make_camera_links(ids, frames, detections, tracked, tile_width=160, tile_height=120,
                             frame_number=42, timestamp_ns=1234)
    mosaic = make_camera_analysis_mosaic(ids, frames, frames, {}, {}, tile_width=160, tile_height=120)
    return mosaic, meta, frames, detections, tracked


def test_geometry_and_lossless_mask_use_actual_tile_coordinates():
    _, meta, *_ = fixture()
    entry = meta["cameras"][0]["objects"][0]
    assert entry["box_tile_xyxy"] == [20, 30, 50, 70]
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(entry["mask_png_base64"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (120, 160)
    assert mask[35, 25] == 255 and mask[45, 35] == 0
    assert meta["frame_number"] == 42 and meta["timestamp_ns"] == 1234


def test_letterbox_uses_same_rounded_size_and_offsets_as_preview():
    frame = np.zeros((240, 320, 3), np.uint8)
    detection = Object2D("a", 0, "cup", 1, BoundingBox2D(0, 0, 320, 240), .9)
    meta = make_camera_links(["a"], {"a": frame}, {"a": [detection]},
                             {("a", 0): SimpleNamespace(global_id="cup/A")}, tile_width=320, tile_height=180)
    assert meta["cameras"][0]["objects"][0]["box_tile_xyxy"] == [40, 0, 280, 180]


def test_no_class_based_link_for_unassociated_detection():
    _, _, frames, detections, _ = fixture()
    meta = make_camera_links(["a"], frames, detections, {})
    assert meta["cameras"][0]["objects"] == []


def test_selected_cup_does_not_highlight_other_cup_or_duplicate_links():
    base, meta, *_ = fixture()
    original = base.copy()
    output, status, thumbnail = render_camera_focus(base, meta, ["cup/A"], {"cup/A": "Cup 01"})
    assert status["a"]["observed_ids"] == ["cup/A"]
    assert status["b"]["state"] == "not_observed"
    assert status["c"]["state"] == "ambiguous"
    assert status["c"]["observed_ids"] == []
    assert status["d"]["state"] == "no_frame"
    assert thumbnail is not None
    assert np.array_equal(base, original) and not np.array_equal(output, base)
    # Selected raw interior is restored, other cup stays dimmed.
    assert np.array_equal(output[55, 35], base[55, 35])
    assert not np.array_equal(output[55, 115], base[55, 115])
    # Segmentation interior is tinted, while its hole stays dim.
    assert output[35, 160+25, 2] > output[45, 160+35, 2]


def test_four_observed_views_highlight_in_both_columns():
    base, meta, *_ = fixture()
    for camera in meta["cameras"]:
        camera["has_frame"] = True
        camera["objects"] = [meta["cameras"][0]["objects"][0]]
    output, status, _ = render_camera_focus(base, meta, ["cup/A"])
    assert all(s["observed_ids"] == ["cup/A"] for s in status.values())
    for row in range(4):
        for col in range(2):
            assert not np.array_equal(output[row*120+1, col*160+1], base[row*120+1, col*160+1])


def test_clear_selection_restores_exact_base_and_unknown_ids_never_guess():
    base, meta, *_ = fixture()
    result, status, thumb = render_camera_focus(base, meta, [])
    assert np.array_equal(result, base) and status == {} and thumb is None
    _, status, thumb = render_camera_focus(base, meta, ["cup/deleted"])
    assert all(not s["observed_ids"] for s in status.values()) and thumb is None


def test_missing_metadata_has_no_false_link():
    base, *_ = fixture()
    _, status, thumb = render_camera_focus(base, None, ["cup/A"])
    assert all(s["state"] == "link_unavailable" for s in status.values())
    assert thumb is None


def test_preview_digest_guards_image_metadata_coherence(tmp_path):
    base, meta, *_ = fixture()
    path = tmp_path / "preview.jpg"
    write_preview_image(path, base, links=meta)
    before = path.read_bytes()
    loaded, links = read_linked_preview(path)
    assert links["frame_number"] == 42
    assert links["image_sha256"] == hashlib.sha256(before).hexdigest()
    render_camera_focus(loaded, links, ["cup/A"])
    assert path.read_bytes() == before
    write_preview_image(path, np.zeros_like(base))  # Old links + new JPEG must fail closed.
    assert read_linked_preview(path)[1] is None


def test_corrupt_sidecar_is_ignored(tmp_path):
    base, meta, *_ = fixture()
    path = tmp_path / "preview.jpg"
    write_preview_image(path, base, links=meta)
    path.with_suffix(".links.json").write_text("{bad")
    assert read_linked_preview(path)[1] is None


def test_selection_resolves_mesh_descendants_groups_and_exact_prefixes():
    paths = {"a": "/World/Tracking/Objects/Cup1", "b": "/World/Tracking/Objects/Cup10"}
    assert resolve_focus_ids([paths["a"] + "/Asset/Handle"], paths) == ["a"]
    assert resolve_focus_ids([paths["b"]], paths) == ["b"]
    assert resolve_focus_ids(["/World/Tracking/Objects"], paths) == ["a", "b"]
    assert resolve_focus_ids(["/World", "/World/TabletopDemo/Table"], paths) == []
    assert resolve_focus_ids([paths["a"]], {}) == []


def test_live_publisher_snapshots_links_with_overlays():
    _, _, frames, detections, tracked = fixture()
    publisher = MosaicFilePublisher([SimpleNamespace(camera_id=c) for c in ["a", "b", "c", "d"]],
                                    "/unused.jpg", tile_width=160, tile_height=120, layout="raw_and_perception")
    publisher.update_overlays(detections, tracked, frames, frame_number=42, timestamp_ns=1234)
    assert publisher._perception_links["frame_number"] == 42
    assert publisher._perception_links["cameras"][0]["objects"][0]["object_id"] == "cup/A"
    publisher.set_paused(True)
    assert publisher._paused.is_set()
    publisher.set_paused(False)
    assert not publisher._paused.is_set()
