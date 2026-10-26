"""Coherent preview/ID links and UI-only selection effects (no inference changes)."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

LINK_SCHEMA = "gist.tabletop_camera_links.v1"
FOCUS_COLOR = (40, 220, 255)  # BGR gold; different from ordinary green detections.


def fit_parameters(image_width, image_height, tile_width, tile_height):
    scale = min(tile_width / image_width, tile_height / image_height)
    width, height = max(1, round(image_width * scale)), max(1, round(image_height * scale))
    return width, height, (tile_width - width) // 2, (tile_height - height) // 2


def make_camera_links(camera_ids, source_frames, detections, tracked_by_detection, *,
                      tile_width=320, tile_height=180, frame_number=None, timestamp_ns=None):
    """Exact geometry for the same source frames used by make_camera_analysis_mosaic.

    Only already-associated tracker IDs are eligible. An unassociated detection
    is never linked by class, appearance or a guessed projected 3D position.
    Masks are lossless binary PNGs at preview resolution, not model-input masks.
    """
    cameras = []
    for cid in camera_ids:
        image = source_frames.get(cid)
        camera = {"camera_id": cid, "has_frame": image is not None, "objects": []}
        cameras.append(camera)
        if image is None:
            continue
        ih, iw = image.shape[:2]
        width, height, dx, dy = fit_parameters(iw, ih, tile_width, tile_height)
        camera["source_size_wh"] = [iw, ih]
        for detection in detections.get(cid, []):
            tracked = tracked_by_detection.get((cid, detection.local_detection_id))
            if tracked is None:
                continue
            b = detection.box
            box = [dx + b.x1 * width / iw, dy + b.y1 * height / ih,
                   dx + b.x2 * width / iw, dy + b.y2 * height / ih]
            if not all(np.isfinite(box)):
                continue
            entry = {"object_id": str(tracked.global_id), "label": detection.label,
                     "local_detection_id": int(detection.local_detection_id),
                     "box_tile_xyxy": [float(v) for v in box], "mask_png_base64": None}
            mask = detection.mask
            if mask is not None and mask.shape == (ih, iw):
                tile = np.zeros((tile_height, tile_width), dtype=np.uint8)
                tile[dy:dy+height, dx:dx+width] = cv2.resize(
                    (mask > 0).astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
                ok, encoded = cv2.imencode(".png", tile)
                if ok:
                    entry["mask_png_base64"] = base64.b64encode(encoded).decode("ascii")
            camera["objects"].append(entry)
    return {"schema": LINK_SCHEMA, "layout": "paired_rows", "tile_size_wh": [tile_width, tile_height],
            "frame_number": frame_number, "timestamp_ns": timestamp_ns, "cameras": cameras}


def read_linked_preview(path):
    """Always read a complete JPEG; reject mismatched/torn sidecar metadata."""
    path = Path(path)
    encoded = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid preview JPEG")
    metadata = None
    try:
        candidate = json.loads(path.with_suffix(".links.json").read_text())
        if (isinstance(candidate, dict) and candidate.get("schema") == LINK_SCHEMA and candidate.get("layout") == "paired_rows"
                and candidate.get("image_sha256") == hashlib.sha256(encoded).hexdigest()):
            tw, th = candidate["tile_size_wh"]
            if (isinstance(tw, int) and isinstance(th, int) and tw > 0 and th > 0
                    and isinstance(candidate["cameras"], list) and len(candidate["cameras"]) == 4
                    and all(isinstance(c, dict) for c in candidate["cameras"])
                    and image.shape[:2] == (4*th, 2*tw)):
                metadata = candidate
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return image, metadata


def resolve_focus_ids(selected_paths, object_paths, *, objects_root="/World/Tracking/Objects"):
    """Map mesh descendants / managed groups to IDs without prefix collisions."""
    found = []
    for selected in selected_paths:
        selected = str(selected).rstrip("/")
        for oid, path in object_paths.items():
            if (selected == path or selected.startswith(path + "/")
                    or (selected == objects_root or selected.startswith(objects_root + "/"))
                    and path.startswith(selected + "/")):
                if oid not in found:
                    found.append(oid)
    return found


def _valid_box(entry, width, height):
    try:
        box = [float(v) for v in entry["box_tile_xyxy"]]
        if len(box) != 4 or not all(np.isfinite(box)):
            return None
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(width-1, round(x1))), max(0, min(width-1, round(x2)))
        y1, y2 = max(0, min(height-1, round(y1))), max(0, min(height-1, round(y2)))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
    except (KeyError, TypeError, ValueError):
        return None


def render_camera_focus(paired, metadata, selected_ids, names=None):
    """Paint a COPY for the two UI grids. Never writes back to preview files.

    Duplicate links for an ID in one camera are ambiguous: no guessed box.
    Empty views explicitly mean not observed, not necessarily physically absent.
    """
    output = paired.copy()
    if not selected_ids:
        return output, {}, None
    selected_ids = list(dict.fromkeys(selected_ids))
    names = names or {}
    th, tw = paired.shape[0] // 4, paired.shape[1] // 2
    statuses, thumbnail, best_area = {}, None, -1
    cameras = metadata.get("cameras", []) if metadata else []
    for row in range(4):
        camera = cameras[row] if row < len(cameras) else {}
        cid = camera.get("camera_id", f"view_{row}")
        matches, ambiguous = [], []
        for oid in selected_ids:
            candidates = [o for o in camera.get("objects", []) if o.get("object_id") == oid]
            if len(candidates) > 1:
                ambiguous.append(oid)
            elif candidates and _valid_box(candidates[0], tw, th):
                matches.append(candidates[0])
        observed = [entry["object_id"] for entry in matches]
        state = ("link_unavailable" if metadata is None else "no_frame" if not camera.get("has_frame") else
                 "ambiguous" if ambiguous else "observed" if observed else "not_observed")
        statuses[cid] = {"state": state, "observed_ids": observed, "ambiguous_ids": ambiguous,
                         "unobserved_ids": [o for o in selected_ids if o not in observed and o not in ambiguous]}
        footer = {"link_unavailable": "LINK UNAVAILABLE", "no_frame": "NO FRAME",
                  "ambiguous": "AMBIGUOUS LINK", "not_observed": "NOT OBSERVED"}.get(state,
                  "FOCUS: " + ", ".join(names.get(oid, oid.rsplit("/", 1)[-1]) for oid in observed))
        for column in range(2):
            y, x = row * th, column * tw
            source = paired[y:y+th, x:x+tw]
            tile = output[y:y+th, x:x+tw]
            # Preserve the camera header; soften other content without hiding it.
            tile[30:] = (tile[30:].astype(np.float32) * 0.68).astype(np.uint8)
            for entry in matches:
                x1, y1, x2, y2 = _valid_box(entry, tw, th)
                mask = None
                if column == 1 and entry.get("mask_png_base64"):
                    try:
                        encoded = base64.b64decode(entry["mask_png_base64"], validate=True)
                        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
                        if decoded is not None and decoded.shape == (th, tw):
                            mask = decoded > 0
                    except (ValueError, cv2.error):
                        pass
                if mask is not None and np.any(mask):
                    tile[mask] = (source[mask] * 0.70 + np.array(FOCUS_COLOR) * 0.30).astype(np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(tile, contours, -1, FOCUS_COLOR, 2)
                else:
                    tile[y1:y2+1, x1:x2+1] = source[y1:y2+1, x1:x2+1]
                    cv2.rectangle(tile, (x1, y1), (x2, y2), FOCUS_COLOR, 3)
                if column == 0:
                    cv2.rectangle(tile, (x1, y1), (x2, y2), FOCUS_COLOR, 3)
                    area = (x2-x1) * (y2-y1)
                    if len(selected_ids) == 1 and area > best_area:
                        thumbnail = source[y1:y2+1, x1:x2+1].copy()
                        best_area = area
            cv2.rectangle(tile, (0, th-20), (tw-1, th-1), (12, 18, 22), -1)
            cv2.putText(tile, footer[:45], (6, th-6), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        FOCUS_COLOR if observed else (180, 180, 180), 1, cv2.LINE_AA)
            if observed:
                cv2.rectangle(tile, (1, 1), (tw-2, th-2), FOCUS_COLOR, 2)
    return output, statuses, thumbnail
