from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
from .agent_control import PerceptionAgentControl, encode_detections, camera_evidence_metadata
from .camera_focus import make_camera_links
from .evaluation import evaluate_recorded_session, write_evaluation_report
from .object_detection import draw_object_detections
from .perception import PerceptionProcessor
from .preview import (
    make_camera_analysis_mosaic,
    make_preview_mosaic,
    write_preview_image,
)
from .publisher import JsonLinesWriter, UdpJsonPublisher
from .recorded_session import RecordedSession
from .sync import FrameSynchronizer
from .rate_limit import PerceptionRateGate


def run_recorded_replay(
    session_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    annotations_path: str | Path | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    max_samples: int | None = None,
    preview_every: int = 0,
    embedded_preview: bool = False,
    publish_udp: bool = False,
    realtime: bool = False,
    speed: float = 1.0,
    async_perception: bool = False,
    recorded_lockstep: bool = False,
) -> dict:
    """Run the production perception stack over a recorded camera session."""
    root = Path(session_dir).resolve()
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    selected_annotations = annotations_path
    if selected_annotations is None and manifest.get("annotations_file"):
        candidate = root / str(manifest["annotations_file"])
        if candidate.is_file():
            selected_annotations = candidate
    selected_config = Path(config_path or manifest.get("config_path", "")).expanduser()
    if not selected_config.is_absolute():
        selected_config = (root / selected_config).resolve()
    if not selected_config.is_file():
        raise FileNotFoundError(
            f"Perception config was not found: {selected_config}. Pass --config explicitly."
        )
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "replay_latest"
    )
    destination.mkdir(parents=True, exist_ok=True)
    if async_perception and not realtime:
        raise ValueError("async_perception requires realtime playback")
    tracking_path = destination / "tracking.jsonl"
    detections_path = destination / "detections.jsonl"
    preview_dir = destination / "previews"
    if preview_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    processor = PerceptionProcessor(selected_config, persist_appearance=False)
    config = processor.config
    rate_gate = PerceptionRateGate(config.get("perception_max_hz", 0))
    configured_control = Path(
        str(config.get("agent_control_dir", "../../outputs/tabletop_agent"))
    ).expanduser()
    if not configured_control.is_absolute():
        configured_control = (selected_config.parent / configured_control).resolve()
    agent_control = PerceptionAgentControl(configured_control)
    agent_control.start_runtime("recorded_replay")
    embedded_preview_path = None
    if embedded_preview:
        configured_preview = config.get("embedded_preview_path")
        if not configured_preview:
            raise ValueError(
                "The selected config has no embedded_preview_path for --embedded-preview"
            )
        embedded_preview_path = Path(str(configured_preview)).expanduser()
        if not embedded_preview_path.is_absolute():
            embedded_preview_path = (
                selected_config.parent / embedded_preview_path
            ).resolve()
    tolerance_ns = int(float(config.get("sync_tolerance_ms", 60.0)) * 1_000_000)
    synchronizer = FrameSynchronizer(
        list(processor.calibrations),
        tolerance_ns,
        time_offsets_ns={
            name: calibration.time_offset_ns
            for name, calibration in processor.calibrations.items()
        },
        max_queue_size=int(config.get("sync_queue_size", 8)),
    )
    publishers = [JsonLinesWriter(tracking_path)]
    if publish_udp:
        udp = config.get("udp", {})
        publishers.append(
            UdpJsonPublisher(udp.get("host", "127.0.0.1"), int(udp.get("port", 7001)))
        )

    started = time.monotonic()
    processed = 0
    read_samples = 0
    first_recorded_ns: int | None = None
    with RecordedSession(root) as session:
        integrity = session.validate()
        if not integrity["valid"]:
            raise ValueError("Invalid recording: " + "; ".join(integrity["errors"]))
        if set(session.camera_ids) != set(processor.calibrations):
            raise ValueError(
                "Recording cameras and calibration cameras differ: "
                f"recording={sorted(session.camera_ids)}, "
                f"config={sorted(processor.calibrations)}"
            )
        with detections_path.open("w", encoding="utf-8") as detections_file:
            latest_detection_frames: dict[str, np.ndarray] = {}
            latest_segmentation_frames: dict[str, np.ndarray] = {}
            latest_camera_links = None
            preview_initialized = False
            last_saved_processed = -1
            fps_by_camera = {
                camera_id: float(session.manifest.get("record_fps", 0.0))
                for camera_id in session.camera_ids
            }

            def emit_result(result, frame_number: int, packets: dict) -> None:
                nonlocal processed, latest_detection_frames, latest_segmentation_frames, latest_camera_links
                for publisher in publishers:
                    publisher.publish(result.message)
                encoded_detections = encode_detections(
                    frame_number,
                    int(result.message["timestamp_ns"]),
                    result.detections_by_camera,
                    result.tracked_by_detection,
                )
                detections_file.write(
                    json.dumps(encoded_detections, ensure_ascii=False) + "\n"
                )
                latest_detection_frames = {
                    camera_id: draw_object_detections(
                        packet.image,
                        result.detections_by_camera.get(camera_id, []),
                        result.tracked_by_detection,
                        show_masks=False,
                    )
                    for camera_id, packet in packets.items()
                    if packet.image is not None
                }
                latest_segmentation_frames = {
                    camera_id: draw_object_detections(
                        packet.image,
                        result.detections_by_camera.get(camera_id, []),
                        result.tracked_by_detection,
                        show_boxes=False,
                        show_labels=False,
                        show_anchors=False,
                        mask_alpha=0.62,
                    )
                    for camera_id, packet in packets.items()
                    if packet.image is not None
                }
                latest_camera_links = make_camera_links(
                    list(session.camera_ids), latest_detection_frames, result.detections_by_camera,
                    result.tracked_by_detection,
                    tile_width=int(config.get("embedded_preview_tile_width", 320)),
                    tile_height=int(config.get("embedded_preview_tile_height", 180)),
                    frame_number=frame_number, timestamp_ns=int(result.message["timestamp_ns"]))
                if agent_control.pending_snapshot_request() is not None:
                    if embedded_preview_path is not None:
                        publish_preview(frame_number)
                    snapshot_path = agent_control.capture_snapshot(
                        source="recorded_replay",
                        frame_number=frame_number,
                        tracking=result.message,
                        detections=encoded_detections,
                        raw_frames={
                            camera_id: packet.image
                            for camera_id, packet in packets.items()
                            if packet.image is not None
                        },
                        detection_frames=latest_detection_frames,
                        segmentation_frames=latest_segmentation_frames,
                        preview_path=embedded_preview_path,
                        camera_metadata=camera_evidence_metadata(config, processor.calibrations, packets),
                        rig=config.get("rig", {}),
                    )
                    if snapshot_path is not None:
                        print(f"Agent snapshot frozen: {snapshot_path}")
                processed += 1
                if processed % 100 == 0:
                    print(
                        f"Replay: processed={processed} recording_index={frame_number}"
                    )

            def publish_preview(frame_number: int, *, save: bool = False) -> None:
                if config.get("embedded_preview_layout") == "raw_and_perception":
                    mosaic = make_camera_analysis_mosaic(
                        list(session.camera_ids),
                        latest_detection_frames,
                        latest_segmentation_frames,
                        fps_by_camera,
                        {
                            camera_id: "waiting for instance segmentation..."
                            for camera_id in session.camera_ids
                        },
                        tile_width=int(
                            config.get("embedded_preview_tile_width", 320)
                        ),
                        tile_height=int(
                            config.get("embedded_preview_tile_height", 180)
                        ),
                        detection_label="RAW + OBJECT DETECTION",
                        perception_label="INSTANCE SEGMENTATION",
                    )
                else:
                    mosaic = make_preview_mosaic(
                        list(session.camera_ids),
                        latest_segmentation_frames or latest_detection_frames,
                        fps_by_camera,
                        {},
                        tile_width=int(config.get("embedded_preview_tile_width", 320)),
                        tile_height=int(config.get("embedded_preview_tile_height", 180)),
                    )
                if save:
                    cv2.imwrite(str(preview_dir / f"frame_{frame_number:06d}.jpg"), mosaic)
                if embedded_preview_path is not None:
                    write_preview_image(
                        embedded_preview_path,
                        mosaic,
                        jpeg_quality=int(config.get("embedded_preview_jpeg_quality", 78)),
                        links=latest_camera_links if config.get("embedded_preview_layout") == "raw_and_perception" else None,
                    )

            executor = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="instance-segmentation")
                if async_perception
                else None
            )
            future: Future | None = None
            future_context: tuple[int, dict] | None = None
            try:
                for sample in session.iter_samples(
                    start=start, stop=stop, stride=stride
                ):
                    pause_started = time.monotonic()
                    while agent_control.is_frozen():
                        time.sleep(0.02)
                    pause_duration = time.monotonic() - pause_started
                    if pause_duration > 0.05:
                        started += pause_duration
                        agent_control.mark_live("recorded_replay")
                    if max_samples is not None and read_samples >= max_samples:
                        break
                    if realtime:
                        if speed <= 0.0:
                            raise ValueError("speed must be positive")
                        if first_recorded_ns is None:
                            first_recorded_ns = sample.recorded_monotonic_ns
                        target = (
                            sample.recorded_monotonic_ns - first_recorded_ns
                        ) / 1_000_000_000.0 / speed
                        remaining = target - (time.monotonic() - started)
                        if remaining > 0.0:
                            time.sleep(remaining)
                    read_samples += 1
                    if recorded_lockstep:
                        # The recorder writes one frame per camera into the same
                        # sample index. Timestamp skew is retained as metadata,
                        # but must not discard a camera view during offline
                        # frame-index evaluation.
                        synced = sample.packets
                    else:
                        for packet in sample.packets.values():
                            synchronizer.push(packet)
                        synced = synchronizer.pop_synced(session.camera_ids)
                    if synced is None:
                        continue
                    perception_updated = False
                    if executor is None:
                        if not rate_gate.ready(sample.recorded_monotonic_ns):
                            continue
                        result = processor.process(synced, sample.recording_index)
                        emit_result(result, sample.recording_index, synced)
                        perception_updated = True
                    else:
                        if future is not None and future.done():
                            assert future_context is not None
                            emit_result(future.result(), *future_context)
                            perception_updated = True
                            future = None
                            future_context = None
                        if future is None and rate_gate.ready(time.monotonic_ns()):
                            future = executor.submit(
                                processor.process, synced, sample.recording_index
                            )
                            future_context = (sample.recording_index, synced)
                    save_preview = (
                        preview_every > 0
                        and processed > 0
                        and processed % preview_every == 0
                        and processed != last_saved_processed
                    )
                    should_publish_embedded = (
                        embedded_preview_path is not None
                        and (
                            executor is None
                            or perception_updated
                            or not preview_initialized
                        )
                    )
                    if save_preview or should_publish_embedded:
                        publish_preview(sample.recording_index, save=save_preview)
                        preview_initialized = True
                        if save_preview:
                            last_saved_processed = processed
                if future is not None:
                    assert future_context is not None
                    emit_result(future.result(), *future_context)
                    if embedded_preview_path is not None:
                        publish_preview(future_context[0])
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                for publisher in publishers:
                    publisher.close()

    elapsed = time.monotonic() - started
    detection_config = json.loads(selected_config.read_text(encoding="utf-8")).get("object_detection", {})
    catalog_setting = detection_config.get("prop_catalog_dir")
    prop_catalog_dir = (selected_config.parent / catalog_setting).resolve() if catalog_setting else None
    evaluation = evaluate_recorded_session(
        root,
        tracking_path,
        detections_path=detections_path,
        annotations_path=selected_annotations,
        processing_seconds=elapsed,
        prop_catalog_dir=prop_catalog_dir,
    )
    evaluation["replay"] = {
        "config_path": str(selected_config),
        "start": start,
        "stop": stop,
        "stride": stride,
        "read_samples": read_samples,
        "processed_samples": processed,
        "async_perception": async_perception,
        "recorded_lockstep": recorded_lockstep,
        "sync_dropped_frames": synchronizer.dropped_frames,
        "detections_path": str(detections_path),
        "preview_every": preview_every,
        "embedded_preview_path": (
            str(embedded_preview_path)
            if embedded_preview_path is not None
            else None
        ),
        "annotations_path": (
            str(Path(selected_annotations).resolve())
            if selected_annotations is not None
            else None
        ),
    }
    json_report, markdown_report = write_evaluation_report(evaluation, destination)
    run_result = {
        "session_dir": str(root),
        "output_dir": str(destination),
        "tracking_path": str(tracking_path),
        "detections_path": str(detections_path),
        "evaluation_json": str(json_report),
        "evaluation_markdown": str(markdown_report),
        "read_samples": read_samples,
        "processed_samples": processed,
        "async_perception": async_perception,
        "elapsed_seconds": elapsed,
    }
    (destination / "replay_run.json").write_text(
        json.dumps(run_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_result
