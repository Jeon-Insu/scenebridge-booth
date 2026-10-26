from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .recorded_session import RecordedSession

# States in which an object is drawn in the twin, the same set as
# scene_agent.VISIBLE_STATES and the extension. "candidate", "tentative", "occluded"
# and "dropped" are bookkeeping states the viewer never sees.
VISIBLE_STATES = frozenset({"active", "partial", "uncertain"})


def prop_uri_map(catalog_dir: str | Path) -> dict[str, str]:
    """Map each prop's resolved USD path to its asset_id.

    Prop retrieval writes the resolved USD path into a twin object's ``asset_uri``,
    so this is how an evaluation knows which registered prop it saw.
    """
    root = Path(catalog_dir).resolve()
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    return {
        str((root / entry["usd_uri"]).resolve()): str(entry["asset_id"])
        for entry in catalog.get("assets", [])
    }


def _prop_id(item: dict, prop_ids_by_uri: dict[str, str]) -> str | None:
    uri = item.get("asset_uri")
    if not uri:
        return None
    return prop_ids_by_uri.get(str(uri)) or prop_ids_by_uri.get(str(Path(uri).resolve()))


def _prop_metrics(selected: list[dict], expected_props: list[str],
                  prop_ids_by_uri: dict[str, str]) -> dict[str, dict]:
    """Per prop: share of frames where two or more cameras see it, and its error there."""
    metrics = {}
    for prop in expected_props:
        seen_frames = 0
        errors: list[float] = []
        for message in selected:
            hits = [
                item for item in message.get("objects", [])
                if item.get("state") in VISIBLE_STATES
                and len(item.get("source_cameras", [])) >= 2
                and _prop_id(item, prop_ids_by_uri) == prop
            ]
            if hits:
                seen_frames += 1
                errors.extend(float(item.get("reprojection_error_px", 0.0)) for item in hits)
        metrics[str(prop)] = {
            "detected_ratio_2cam": seen_frames / len(selected) if selected else 0.0,
            "reprojection_error_px_median": float(np.median(errors)) if errors else None,
        }
    return metrics


def _event_latencies(messages: list[dict], events: list[dict],
                     prop_ids_by_uri: dict[str, str]) -> list[dict]:
    """Seconds from each annotated placement until that prop is first drawn in the twin."""
    ordered = sorted(messages, key=lambda message: int(message.get("frame_number", -1)))
    results = []
    for event in events:
        frame = int(event["frame"])
        prop = str(event.get("prop", ""))
        after = [message for message in ordered if int(message.get("frame_number", -1)) >= frame]
        latency = None
        for message in after:
            if any(item.get("state") in VISIBLE_STATES and _prop_id(item, prop_ids_by_uri) == prop
                   for item in message.get("objects", [])):
                latency = (int(message["timestamp_ns"]) - int(after[0]["timestamp_ns"])) / 1e9
                break
        results.append({"frame": frame, "type": str(event.get("type", "")),
                        "prop": prop, "latency_s": latency})
    return results


def _false_positives(selected: list[dict]) -> dict[str, int]:
    """On an empty table every drawn object is a false positive."""
    counts = [sum(1 for item in message.get("objects", []) if item.get("state") in VISIBLE_STATES)
              for message in selected]
    return {"frames_with_objects": sum(1 for count in counts if count > 0),
            "max_simultaneous_objects": max(counts, default=0)}


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_tracking_messages(
    messages: list[dict], *, annotations: dict | None = None,
    prop_ids_by_uri: dict[str, str] | None = None,
) -> dict:
    object_counts: Counter[int] = Counter()
    states: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    source_camera_counts: Counter[int] = Counter()
    label_ids: dict[str, set[str]] = defaultdict(set)
    label_simultaneous_max: Counter[str] = Counter()
    per_id_raw: dict[str, dict] = {}
    last_state: dict[str, str] = {}
    state_transition_count = 0

    for message in messages:
        objects = list(message.get("objects", []))
        object_counts[len(objects)] += 1
        per_label = Counter(str(item.get("label", "unknown")) for item in objects)
        for label, count in per_label.items():
            label_simultaneous_max[label] = max(label_simultaneous_max[label], count)
        for item in objects:
            object_id = str(item["id"])
            label = str(item.get("label", "unknown"))
            state = str(item.get("state", "unknown"))
            xyz = np.asarray(item.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
            states[state] += 1
            modes[str(item.get("position_mode", "unknown"))] += 1
            source_camera_counts[len(item.get("source_cameras", []))] += 1
            label_ids[label].add(object_id)
            raw = per_id_raw.setdefault(
                object_id,
                {
                    "label": label,
                    "frames": [],
                    "states": Counter(),
                    "xyz": [],
                    "reprojection_errors_px": [],
                    "state_transition_count": 0,
                },
            )
            raw["frames"].append(int(message.get("frame_number", len(raw["frames"]))))
            raw["states"][state] += 1
            raw["xyz"].append(xyz)
            raw["reprojection_errors_px"].append(
                float(item.get("reprojection_error_px", 0.0))
            )
            if object_id in last_state and last_state[object_id] != state:
                raw["state_transition_count"] += 1
                state_transition_count += 1
            last_state[object_id] = state

    per_id: dict[str, dict] = {}
    for object_id, raw in sorted(per_id_raw.items()):
        xyz = np.stack(raw["xyz"])
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else []
        per_id[object_id] = {
            "label": raw["label"],
            "sample_count": len(raw["frames"]),
            "first_frame": min(raw["frames"]),
            "last_frame": max(raw["frames"]),
            "state_counts": dict(sorted(raw["states"].items())),
            "state_transition_count": raw["state_transition_count"],
            "xyz_min_m": [float(value) for value in np.min(xyz, axis=0)],
            "xyz_max_m": [float(value) for value in np.max(xyz, axis=0)],
            "xyz_span_m": [float(value) for value in np.ptp(xyz, axis=0)],
            "step_distance_m": _summary(steps),
            "reprojection_error_px": _summary(raw["reprojection_errors_px"]),
        }

    scenario_results = []
    for scenario in (annotations or {}).get("scenarios", []):
        start = int(scenario["start_index"])
        end = int(scenario["end_index"])
        selected = [
            message
            for message in messages
            if start <= int(message.get("frame_number", -1)) <= end
        ]
        expected = {
            str(label): int(count)
            for label, count in scenario.get("expected_counts", {}).items()
        }
        matched = 0
        ids_by_label: dict[str, set[str]] = defaultdict(set)
        for message in selected:
            actual = Counter(
                str(item.get("label", "unknown"))
                for item in message.get("objects", [])
                if item.get("state") != "dropped"
            )
            if all(actual[label] == count for label, count in expected.items()):
                matched += 1
            for item in message.get("objects", []):
                ids_by_label[str(item.get("label", "unknown"))].add(str(item["id"]))
        row = {
            "name": str(scenario.get("name", "unnamed")),
            "start_index": start,
            "end_index": end,
            "expected_counts": expected,
            "evaluated_messages": len(selected),
            "count_match_ratio": matched / len(selected) if selected else 0.0,
            "distinct_ids_by_label": {
                label: len(values) for label, values in sorted(ids_by_label.items())
            },
        }
        if "expected_props" in scenario:
            row["props"] = _prop_metrics(selected, list(scenario["expected_props"]),
                                         prop_ids_by_uri or {})
        if "events" in scenario:
            row["events"] = _event_latencies(messages, list(scenario["events"]),
                                             prop_ids_by_uri or {})
        if scenario.get("expected_counts") == {} and not scenario.get("expected_props"):
            row["false_positives"] = _false_positives(selected)
        scenario_results.append(row)

    return {
        "message_count": len(messages),
        "object_count_distribution": {
            str(key): value for key, value in sorted(object_counts.items())
        },
        "state_distribution": dict(sorted(states.items())),
        "position_mode_distribution": dict(sorted(modes.items())),
        "source_camera_count_distribution": {
            str(key): value for key, value in sorted(source_camera_counts.items())
        },
        "label_instance_count": {
            label: len(values) for label, values in sorted(label_ids.items())
        },
        "label_simultaneous_max": dict(sorted(label_simultaneous_max.items())),
        "state_transition_count": state_transition_count,
        "per_id": per_id,
        "scenarios": scenario_results,
    }


def evaluate_detection_messages(messages: list[dict]) -> dict:
    per_camera: dict[str, dict] = {}
    total_detections = 0
    matched_detections = 0
    for message in messages:
        for camera_id, detections in message.get("cameras", {}).items():
            raw = per_camera.setdefault(
                camera_id,
                {
                    "frame_count": 0,
                    "detection_count": 0,
                    "matched_count": 0,
                    "labels": Counter(),
                    "frames_with_label": Counter(),
                    "mask_areas_px": [],
                    "detections_per_frame": [],
                },
            )
            raw["frame_count"] += 1
            raw["detection_count"] += len(detections)
            raw["detections_per_frame"].append(len(detections))
            frame_labels = set()
            for detection in detections:
                label = str(detection.get("label", "unknown"))
                raw["labels"][label] += 1
                frame_labels.add(label)
                raw["mask_areas_px"].append(
                    float(detection.get("mask_area_px", 0.0))
                )
                total_detections += 1
                if detection.get("tracked_id"):
                    raw["matched_count"] += 1
                    matched_detections += 1
            for label in frame_labels:
                raw["frames_with_label"][label] += 1

    encoded = {}
    for camera_id, raw in sorted(per_camera.items()):
        encoded[camera_id] = {
            "frame_count": raw["frame_count"],
            "detection_count": raw["detection_count"],
            "matched_count": raw["matched_count"],
            "association_ratio": (
                raw["matched_count"] / raw["detection_count"]
                if raw["detection_count"]
                else 0.0
            ),
            "labels": dict(sorted(raw["labels"].items())),
            "frames_with_label": dict(sorted(raw["frames_with_label"].items())),
            "mask_area_px": _summary(raw["mask_areas_px"]),
            "detections_per_frame": _summary(raw["detections_per_frame"]),
        }
    return {
        "message_count": len(messages),
        "total_detections": total_detections,
        "matched_detections": matched_detections,
        "association_ratio": (
            matched_detections / total_detections if total_detections else 0.0
        ),
        "per_camera": encoded,
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
    ]


def evaluate_recorded_session(
    session_dir: str | Path,
    tracking_path: str | Path,
    *,
    detections_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    processing_seconds: float | None = None,
    prop_catalog_dir: str | Path | None = None,
) -> dict:
    root = Path(session_dir).resolve()
    tracking = Path(tracking_path).resolve()
    annotations = (
        json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        if annotations_path is not None
        else None
    )
    with RecordedSession(root) as session:
        integrity = session.validate()
        capture_skews_ms = []
        for row in session.timestamp_rows:
            times = [
                int(row["cameras"][camera_id]["capture_time_ns"])
                for camera_id in session.camera_ids
            ]
            capture_skews_ms.append((max(times) - min(times)) / 1_000_000.0)
        timing = {"capture_skew_ms": _summary(capture_skews_ms)}
    messages = _load_jsonl(tracking)
    result = {
        "schema": "gist.tabletop_evaluation.v1",
        "session_dir": str(root),
        "tracking_path": str(tracking),
        "annotations_path": (
            str(Path(annotations_path).resolve())
            if annotations_path is not None
            else None
        ),
        "integrity": integrity,
        "timing": timing,
        "tracking": evaluate_tracking_messages(
            messages, annotations=annotations,
            prop_ids_by_uri=prop_uri_map(prop_catalog_dir) if prop_catalog_dir else None,
        ),
    }
    if detections_path is not None:
        detection_messages = _load_jsonl(Path(detections_path).resolve())
        result["detections"] = evaluate_detection_messages(detection_messages)
    if processing_seconds is not None:
        result["performance"] = {
            "processing_seconds": float(processing_seconds),
            "messages_per_second": (
                len(messages) / processing_seconds if processing_seconds > 0.0 else 0.0
            ),
        }
    return result


def write_evaluation_report(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "evaluation.json"
    markdown_path = destination / "evaluation.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    integrity = result["integrity"]
    timing = result["timing"]["capture_skew_ms"]
    tracking = result["tracking"]
    lines = [
        "# Tabletop recording evaluation",
        "",
        f"- Session: `{result['session_dir']}`",
        f"- Tracking: `{result['tracking_path']}`",
        f"- Annotations: `{result['annotations_path'] or 'none'}`",
        f"- Integrity: **{'PASS' if integrity['valid'] else 'FAIL'}**",
        f"- Timestamp rows: {integrity['timestamp_rows']}",
        f"- Capture skew p50/p95/max: {timing['p50']:.1f} / {timing['p95']:.1f} / {timing['max']:.1f} ms",
        f"- Tracking messages: {tracking['message_count']}",
        f"- State transitions: {tracking['state_transition_count']}",
        "",
        "## Labels",
        "",
        "| Label | Distinct IDs | Maximum simultaneous |",
        "|---|---:|---:|",
    ]
    for label, count in tracking["label_instance_count"].items():
        lines.append(
            f"| {label} | {count} | {tracking['label_simultaneous_max'].get(label, 0)} |"
        )
    if result.get("detections"):
        detections = result["detections"]
        lines.extend(
            [
                "",
                "## 2D detection and segmentation",
                "",
                f"- Total detections: {detections['total_detections']}",
                f"- Associated with a 3D track: {detections['association_ratio']:.1%}",
                "",
                "| Camera | Detections | Associated | Labels |",
                "|---|---:|---:|---|",
            ]
        )
        for camera_id, item in detections["per_camera"].items():
            lines.append(
                f"| {camera_id} | {item['detection_count']} | "
                f"{item['association_ratio']:.1%} | "
                f"`{json.dumps(item['labels'], ensure_ascii=False)}` |"
            )
    lines.extend(
        [
            "",
            "## Tracks",
            "",
            "| ID | Samples | State changes | Step p95/max (m) | Reprojection p95 (px) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for object_id, item in tracking["per_id"].items():
        lines.append(
            f"| `{object_id}` | {item['sample_count']} | {item['state_transition_count']} | "
            f"{item['step_distance_m']['p95']:.3f} / {item['step_distance_m']['max']:.3f} | "
            f"{item['reprojection_error_px']['p95']:.1f} |"
        )
    if tracking["scenarios"]:
        lines.extend(
            [
                "",
                "## Annotated scenarios",
                "",
                "| Scenario | Messages | Count match | Distinct IDs |",
                "|---|---:|---:|---|",
            ]
        )
        for scenario in tracking["scenarios"]:
            lines.append(
                f"| {scenario['name']} | {scenario['evaluated_messages']} | "
                f"{scenario['count_match_ratio']:.1%} | "
                f"`{json.dumps(scenario['distinct_ids_by_label'], ensure_ascii=False)}` |"
            )
    if result.get("performance"):
        performance = result["performance"]
        lines.extend(
            [
                "",
                "## Performance",
                "",
                f"- Processing time: {performance['processing_seconds']:.1f} s",
                f"- Output rate: {performance['messages_per_second']:.2f} messages/s",
            ]
        )
    if integrity["errors"] or integrity["warnings"]:
        lines.extend(["", "## Recording diagnostics", ""])
        lines.extend(f"- ERROR: {value}" for value in integrity["errors"])
        lines.extend(f"- WARNING: {value}" for value in integrity["warnings"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
