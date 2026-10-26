from __future__ import annotations

import signal
import time
from pathlib import Path

import cv2

from .agent_control import PerceptionAgentControl, encode_detections, camera_evidence_metadata
from .camera_focus import make_camera_links
from .io import load_json
from .object_detection import draw_object_detections
from .perception import PerceptionProcessor
from .preview import (
    MosaicFilePublisher,
    make_camera_analysis_mosaic,
    make_preview_mosaic,
    write_preview_image,
)
from .publisher import JsonLinesWriter, UdpJsonPublisher
from .session_recording import MultiCameraSessionRecorder
from .source import OpenCVCaptureSource
from .sync import FrameSynchronizer
from .rate_limit import PerceptionRateGate


def run_live(config_path: str | Path, *, max_frames: int | None = None) -> int:
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    base = config_path.parent

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()

    camera_configs = config["cameras"]
    processor = PerceptionProcessor(config_path, persist_appearance=True)
    rate_gate = PerceptionRateGate(config.get("perception_max_hz", 0))
    calibrations = processor.calibrations
    sources = [
        OpenCVCaptureSource(
            camera_id=item["id"],
            uri=item["source"],
            width=item.get("width"),
            height=item.get("height"),
            fps=item.get("fps"),
            fourcc=item.get("fourcc"),
            autofocus=item.get("autofocus"),
            focus=item.get("focus"),
            queue_size=int(item.get("queue_size", config.get("camera_queue_size", 2))),
            loop_file=bool(item.get("loop", False)),
        )
        for item in camera_configs
    ]
    synchronizer = FrameSynchronizer(
        list(calibrations),
        int(float(config.get("sync_tolerance_ms", 60.0)) * 1_000_000),
        time_offsets_ns={name: calibration.time_offset_ns for name, calibration in calibrations.items()},
        max_queue_size=int(config.get("sync_queue_size", 8)),
    )
    show_preview = bool(config.get("show_preview", False))
    preview_title = str(config.get("preview_window_title", "Scene perception preview"))
    minimum_active_cameras = max(1, int(config.get("minimum_active_cameras", 2)))
    camera_timeout_ns = int(float(config.get("camera_timeout_seconds", 1.5)) * 1_000_000_000)
    udp = config.get("udp", {})
    publishers = [UdpJsonPublisher(udp.get("host", "127.0.0.1"), int(udp.get("port", 7001)))]
    if config.get("record_jsonl"):
        publishers.append(JsonLinesWriter(resolve(config["record_jsonl"])))
    recording_config = config.get("recording", {})
    session_recorder = (
        MultiCameraSessionRecorder(
            sources,
            resolve(recording_config.get("output_dir", "../../recordings")),
            session_label=str(
                recording_config.get("session_label", "tabletop_scenarios")
            ),
            fps=float(recording_config.get("fps", 5.0)),
            codec=str(recording_config.get("codec", "MJPG")),
            metadata={
                "config_path": str(config_path),
                "rig_name": str(config.get("rig", {}).get("name", "unnamed")),
            },
        )
        if bool(recording_config.get("enabled", False))
        else None
    )
    if session_recorder is not None:
        publishers.append(JsonLinesWriter(session_recorder.session_dir / "tracking.jsonl"))
    embedded_preview = (
        MosaicFilePublisher(
            sources,
            resolve(config["embedded_preview_path"]),
            update_hz=float(config.get("embedded_preview_update_hz", 5.0)),
            tile_width=int(config.get("embedded_preview_tile_width", 320)),
            tile_height=int(config.get("embedded_preview_tile_height", 180)),
            jpeg_quality=int(config.get("embedded_preview_jpeg_quality", 78)),
            layout=str(config.get("embedded_preview_layout", "grid")),
        )
        if config.get("embedded_preview_path")
        else None
    )
    control_root = resolve(
        str(config.get("agent_control_dir", "../../outputs/tabletop_agent"))
    )
    agent_control = PerceptionAgentControl(control_root)
    agent_control.start_runtime("live")
    embedded_preview_path = (
        resolve(config["embedded_preview_path"])
        if config.get("embedded_preview_path")
        else None
    )

    stopping = False

    def request_stop(_signal, _frame) -> None:
        nonlocal stopping
        stopping = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    processed = 0
    last_received_ns: dict[str, int] = {}
    previous_active: tuple[str, ...] = ()
    was_frozen = False
    for source in sources:
        source.start()
    if session_recorder is not None:
        session_recorder.start()
        print(f"Recording camera session: {session_recorder.session_dir}")
    if embedded_preview is not None:
        embedded_preview.start()
    if show_preview:
        cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
    print(f"Started {len(sources)} sources. Press Ctrl+C to stop.")
    try:
        while not stopping and (max_frames is None or processed < max_frames):
            frozen = agent_control.is_frozen()
            if embedded_preview is not None:
                embedded_preview.set_paused(frozen)
            if was_frozen and not frozen:
                agent_control.mark_live("live")
            was_frozen = frozen
            received = False
            for source in sources:
                if (packet := source.poll()) is not None:
                    last_received_ns[source.camera_id] = time.monotonic_ns()
                    if not frozen:
                        synchronizer.push(packet)
                    received = True
            if frozen:
                # Capture threads continue filling their bounded queues. Draining
                # them here keeps USB cameras healthy without spending GPU time.
                time.sleep(0.01)
                continue
            now_ns = time.monotonic_ns()
            active = tuple(
                camera_id
                for camera_id in calibrations
                if now_ns - last_received_ns.get(camera_id, 0) <= camera_timeout_ns
            )
            if active != previous_active:
                inactive = [camera_id for camera_id in calibrations if camera_id not in active]
                print(f"Active cameras: {list(active)}; waiting/reconnecting: {inactive}")
                previous_active = active
            synced = (
                synchronizer.pop_synced(active)
                if len(active) >= minimum_active_cameras
                else None
            )
            if synced is None:
                if not received:
                    time.sleep(0.002)
                continue
            if not rate_gate.ready(time.monotonic_ns()):
                continue  # Consume skipped sets; never accumulate a stale FIFO.
            result = processor.process(synced, processed)
            person = result.person
            objects = result.objects
            objects_by_camera = result.detections_by_camera
            tracked_by_detection = result.tracked_by_detection
            if embedded_preview is not None:
                embedded_preview.update_overlays(
                    objects_by_camera,
                    tracked_by_detection,
                    {
                        camera_id: packet.image
                        for camera_id, packet in synced.items()
                        if packet.image is not None
                    },
                    frame_number=processed, timestamp_ns=int(result.message.get("timestamp_ns", 0)),
                )
            message = result.message
            for publisher in publishers:
                publisher.publish(message)
            if agent_control.pending_snapshot_request() is not None:
                if embedded_preview is not None:
                    embedded_preview.set_paused(True)
                raw_frames = {
                    camera_id: packet.image
                    for camera_id, packet in synced.items()
                    if packet.image is not None
                }
                detection_frames = {
                    camera_id: draw_object_detections(
                        packet.image,
                        objects_by_camera.get(camera_id, []),
                        tracked_by_detection,
                        show_masks=False,
                    )
                    for camera_id, packet in synced.items()
                    if packet.image is not None
                }
                segmentation_frames = {
                    camera_id: draw_object_detections(
                        packet.image,
                        objects_by_camera.get(camera_id, []),
                        tracked_by_detection,
                        show_boxes=False,
                        show_labels=False,
                        show_anchors=False,
                        mask_alpha=0.62,
                    )
                    for camera_id, packet in synced.items()
                    if packet.image is not None
                }
                if embedded_preview_path is not None:
                    frozen_mosaic = make_camera_analysis_mosaic(
                        list(calibrations),
                        detection_frames,
                        segmentation_frames,
                        {},
                        {},
                        tile_width=int(config.get("embedded_preview_tile_width", 320)),
                        tile_height=int(config.get("embedded_preview_tile_height", 180)),
                    )
                    write_preview_image(
                        embedded_preview_path,
                        frozen_mosaic,
                        jpeg_quality=int(config.get("embedded_preview_jpeg_quality", 78)),
                        links=make_camera_links(list(calibrations), raw_frames, objects_by_camera, tracked_by_detection,
                            tile_width=int(config.get("embedded_preview_tile_width", 320)),
                            tile_height=int(config.get("embedded_preview_tile_height", 180)),
                            frame_number=processed, timestamp_ns=int(message.get("timestamp_ns", 0))),
                    )
                snapshot_path = agent_control.capture_snapshot(
                    source="live",
                    frame_number=processed,
                    tracking=message,
                    detections=encode_detections(
                        processed,
                        int(message.get("timestamp_ns", 0) or 0),
                        objects_by_camera,
                        tracked_by_detection,
                    ),
                    raw_frames=raw_frames,
                    detection_frames=detection_frames,
                    segmentation_frames=segmentation_frames,
                    preview_path=embedded_preview_path,
                    camera_metadata=camera_evidence_metadata(config, calibrations, synced),
                    rig=config.get("rig", {}),
                )
                if snapshot_path is not None:
                    print(f"Agent snapshot frozen: {snapshot_path}")
                    if embedded_preview is not None:
                        embedded_preview.set_paused(True)
            if show_preview:
                preview_frames = {
                    camera_id: draw_object_detections(
                        packet.image,
                        objects_by_camera.get(camera_id, []),
                        tracked_by_detection,
                    )
                    for camera_id, packet in synced.items()
                    if packet.image is not None
                }
                mosaic = make_preview_mosaic(
                    list(calibrations),
                    preview_frames,
                    {},
                    {source.camera_id: source.error for source in sources},
                    tile_width=int(config.get("preview_tile_width", 640)),
                    tile_height=int(config.get("preview_tile_height", 360)),
                )
                cv2.imshow(preview_title, mosaic)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    stopping = True
            processed += 1
            if processed % 30 == 0:
                joint_count = len(person.joints) if person else 0
                object_summary = ", ".join(
                    f"{item.global_id}=({item.xyz[0]:.3f},{item.xyz[1]:.3f},{item.xyz[2]:.3f})"
                    f"/{item.state}/{item.position_mode}/{item.reprojection_error_px:.1f}px"
                    for item in objects
                ) or "none"
                print(
                    f"frames={processed} joints={joint_count} objects={len(objects)} "
                    f"active={list(active)} dropped={synchronizer.dropped_frames} "
                    f"tracks=[{object_summary}]"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if session_recorder is not None:
            session_recorder.stop()
            if session_recorder.error:
                print(f"Recording stopped with error: {session_recorder.error}")
        for source in sources:
            source.stop()
        for publisher in publishers:
            publisher.close()
        if embedded_preview is not None:
            embedded_preview.stop()
        if show_preview:
            cv2.destroyWindow(preview_title)
    return processed
