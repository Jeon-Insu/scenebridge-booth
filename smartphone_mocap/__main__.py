from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import (
    calibrate_intrinsics_from_images,
    estimate_board_extrinsics,
    make_approximate_calibration,
    write_charuco_board,
)
from .capture import capture_charuco_images
from .evaluation import evaluate_recorded_session, write_evaluation_report
from .io import load_calibration, load_json, save_calibration
from .pipeline import run_live
from .preview import run_preview
from .replay import run_recorded_replay
from .publisher import JsonLinesWriter, UdpJsonPublisher, make_tracking_message
from .reconstruction import reconstruct_primary_person
from .synthetic import make_four_camera_rig, project_synthetic_person, synthetic_joint_positions


def command_synthetic(args: argparse.Namespace) -> None:
    rig = make_four_camera_rig()
    rng = np.random.default_rng(args.seed)
    publishers = [UdpJsonPublisher(args.udp_host, args.udp_port)]
    if args.output:
        publishers.append(JsonLinesWriter(args.output))
    try:
        for frame_number in range(args.frames):
            positions = synthetic_joint_positions(frame_number, args.fps)
            views = {
                camera_id: [project_synthetic_person(camera, positions, noise_px=args.noise_px, rng=rng)]
                for camera_id, camera in rig.items()
            }
            person = reconstruct_primary_person(views, rig)
            message = make_tracking_message(time.time_ns(), frame_number, [person] if person else [])
            for publisher in publishers:
                publisher.publish(message)
            if args.realtime:
                time.sleep(1.0 / args.fps)
    finally:
        for publisher in publishers:
            publisher.close()
    print(f"Published {args.frames} synthetic frames to UDP {args.udp_host}:{args.udp_port}")


def command_intrinsics(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(args.images))
    calibration, error = calibrate_intrinsics_from_images(args.camera_id, paths)
    save_calibration(args.output, calibration)
    print(f"Saved {args.output}; used {len(paths)} inputs; reprojection RMS={error:.3f}px")


def command_extrinsics(args: argparse.Namespace) -> None:
    calibration = load_calibration(args.calibration)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    calibration.world_to_camera, error = estimate_board_extrinsics(image, calibration)
    save_calibration(args.output or args.calibration, calibration)
    print(f"Saved world-to-camera pose; mean reprojection error={error:.3f}px")


def command_capture_calibration(args: argparse.Namespace) -> None:
    saved = capture_charuco_images(
        args.url,
        args.camera_id,
        args.output_dir,
        max_images=args.max_images,
        minimum_corners=args.minimum_corners,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        guide_20=args.guide_20,
    )
    print(f"Captured {len(saved)} calibration images in {args.output_dir}")


def command_approximate_calibrations(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = load_json(config_path)
    base = config_path.parent
    rig = config.get("rig", {})
    target = np.asarray(
        rig.get("look_at_target_m", config.get("approximate_target_m")), dtype=np.float64
    )
    for item in config["cameras"]:
        mount = item.get("mount", {})
        output = Path(item["calibration"])
        if not output.is_absolute():
            output = (base / output).resolve()
        calibration = make_approximate_calibration(
            camera_id=str(item["id"]),
            image_size=(int(item["width"]), int(item["height"])),
            position_m=np.asarray(
                mount.get("position_m", item.get("approximate_position_m")), dtype=np.float64
            ),
            target_m=np.asarray(mount.get("look_at_target_m", target), dtype=np.float64),
            diagonal_fov_deg=float(item.get("diagonal_fov_deg", 78.0)),
            roll_deg=float(mount.get("roll_deg", 0.0)),
        )
        save_calibration(output, calibration)
        print(f"Saved approximate calibration: {output}")
    print("WARNING: these calibrations validate data flow only; coordinates are not measurements.")


def command_show_rig(args: argparse.Namespace) -> None:
    config = load_json(args.config.resolve())
    rig = config.get("rig", {})
    default_target = np.asarray(
        rig.get("look_at_target_m", config.get("approximate_target_m")), dtype=np.float64
    )
    print(f"rig={rig.get('name', 'unnamed')}")
    print(f"workspace_size_m={rig.get('workspace_size_m', 'unspecified')}")
    print(f"origin={rig.get('origin', 'unspecified')}")
    for item in config["cameras"]:
        mount = item.get("mount", {})
        position = np.asarray(
            mount.get("position_m", item.get("approximate_position_m")), dtype=np.float64
        )
        target = np.asarray(mount.get("look_at_target_m", default_target), dtype=np.float64)
        direction = target - position
        horizontal = float(np.hypot(direction[0], direction[1]))
        yaw_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))
        pitch_deg = float(np.degrees(np.arctan2(direction[2], horizontal)))
        print(
            f"{item['id']}: corner={mount.get('corner', 'unspecified')} "
            f"position_m={position.tolist()} target_m={target.tolist()} "
            f"yaw_deg={yaw_deg:.1f} pitch_deg={pitch_deg:.1f} "
            f"roll_deg={float(mount.get('roll_deg', 0.0)):.1f}"
        )


def command_replay(args: argparse.Namespace) -> None:
    result = run_recorded_replay(
        args.session,
        config_path=args.config,
        output_dir=args.output_dir,
        annotations_path=args.annotations,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        max_samples=args.max_samples,
        preview_every=args.preview_every,
        embedded_preview=args.embedded_preview,
        publish_udp=args.publish_udp,
        realtime=args.realtime,
        speed=args.speed,
        async_perception=args.async_perception,
        recorded_lockstep=args.recorded_lockstep,
    )
    print(
        f"Replay complete: {result['processed_samples']} outputs in "
        f"{result['elapsed_seconds']:.1f}s"
    )
    print(f"Evaluation: {result['evaluation_markdown']}")


def command_evaluate_recording(args: argparse.Namespace) -> None:
    result = evaluate_recorded_session(
        args.session,
        args.tracking,
        detections_path=args.detections,
        annotations_path=args.annotations,
        prop_catalog_dir=args.prop_catalog,
    )
    json_report, markdown_report = write_evaluation_report(result, args.output_dir)
    print(f"Evaluation JSON: {json_report}")
    print(f"Evaluation Markdown: {markdown_report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Four-phone 3D skeleton proof-of-concept")
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("synthetic", help="publish a moving synthetic skeleton")
    synthetic.add_argument("--frames", type=int, default=150)
    synthetic.add_argument("--fps", type=float, default=15.0)
    synthetic.add_argument("--noise-px", type=float, default=0.5)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--udp-host", default="127.0.0.1")
    synthetic.add_argument("--udp-port", type=int, default=7001)
    synthetic.add_argument("--output", type=Path)
    synthetic.add_argument("--realtime", action="store_true")
    synthetic.set_defaults(func=command_synthetic)

    live = commands.add_parser("live", help="run configured cameras or phone streams")
    live.add_argument("--config", type=Path, default=Path("smartphone_mocap/config/smartphones.example.json"))
    live.add_argument("--max-frames", type=int)
    live.set_defaults(func=lambda args: run_live(args.config, max_frames=args.max_frames))

    replay = commands.add_parser(
        "replay", help="deterministically replay a recorded four-camera session"
    )
    replay.add_argument("--session", type=Path, required=True)
    replay.add_argument("--config", type=Path)
    replay.add_argument("--output-dir", type=Path)
    replay.add_argument("--annotations", type=Path)
    replay.add_argument("--start", type=int, default=0)
    replay.add_argument("--stop", type=int)
    replay.add_argument("--stride", type=int, default=1)
    replay.add_argument("--max-samples", type=int)
    replay.add_argument("--preview-every", type=int, default=0)
    replay.add_argument(
        "--embedded-preview",
        action="store_true",
        help="continuously update the configured Isaac Sim camera preview JPEG",
    )
    replay.add_argument("--publish-udp", action="store_true")
    replay.add_argument("--realtime", action="store_true")
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument(
        "--recorded-lockstep",
        action="store_true",
        help="offline evaluation: pair cameras by recorder sample index, retaining timestamp skew as metadata",
    )
    replay.add_argument(
        "--async-perception",
        action="store_true",
        help="keep realtime RAW playback while processing only the latest available frame set",
    )
    replay.set_defaults(func=command_replay)

    evaluate = commands.add_parser(
        "evaluate-recording",
        help="score existing replay JSONL files without rerunning inference",
    )
    evaluate.add_argument("--session", type=Path, required=True)
    evaluate.add_argument("--tracking", type=Path, required=True)
    evaluate.add_argument("--detections", type=Path)
    evaluate.add_argument("--annotations", type=Path)
    evaluate.add_argument("--prop-catalog", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate_recording)

    preview = commands.add_parser("preview", help="show all configured camera views in one window")
    preview.add_argument(
        "--config", type=Path, default=Path("smartphone_mocap/config/usb_webcams_3.json")
    )
    preview.add_argument("--headless", action="store_true", help="test capture without opening a window")
    preview.add_argument("--duration", type=float, help="stop automatically after this many seconds")
    preview.set_defaults(
        func=lambda args: run_preview(
            args.config, headless=args.headless, duration_seconds=args.duration
        )
    )

    approximate = commands.add_parser(
        "make-approx-calibrations",
        help="generate assumed pinhole camera poses for a pipeline smoke test",
    )
    approximate.add_argument(
        "--config", type=Path, default=Path("smartphone_mocap/config/usb_webcams_3.json")
    )
    approximate.set_defaults(func=command_approximate_calibrations)

    show_rig = commands.add_parser(
        "show-rig", help="print workspace, camera mounts, and derived yaw/pitch angles"
    )
    show_rig.add_argument(
        "--config", type=Path, default=Path("smartphone_mocap/config/usb_webcams_3.json")
    )
    show_rig.set_defaults(func=command_show_rig)

    board = commands.add_parser("make-board", help="write a printable ChArUco calibration board")
    board.add_argument("--output", type=Path, default=Path("smartphone_mocap/calibration/charuco.png"))
    board.set_defaults(func=lambda args: print(write_charuco_board(args.output)))

    capture = commands.add_parser(
        "capture-calibration", help="save valid ChArUco poses from the exact live stream"
    )
    capture.add_argument("--url", required=True)
    capture.add_argument("--camera-id", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--max-images", type=int, default=20)
    capture.add_argument("--minimum-corners", type=int, default=8)
    capture.add_argument("--width", type=int, help="USB capture width; use with --height")
    capture.add_argument("--height", type=int, help="USB capture height; use with --width")
    capture.add_argument("--fps", type=float, help="Requested camera frame rate (device may negotiate another rate)")
    capture.add_argument("--fourcc", help="USB capture format, e.g. MJPG")
    capture.add_argument("--guide-20", action="store_true",
                         help="Show 20 planned poses; resume until 20 total images are saved")
    capture.set_defaults(func=command_capture_calibration)

    intrinsics = commands.add_parser("calibrate-intrinsics", help="calibrate one phone from board photos")
    intrinsics.add_argument("--camera-id", required=True)
    intrinsics.add_argument("--images", required=True, help="quoted glob, e.g. 'captures/cam_0/*.jpg'")
    intrinsics.add_argument("--output", type=Path, required=True)
    intrinsics.set_defaults(func=command_intrinsics)

    extrinsics = commands.add_parser("calibrate-extrinsics", help="make the fixed board the BoothFrame")
    extrinsics.add_argument("--calibration", required=True)
    extrinsics.add_argument("--image", required=True)
    extrinsics.add_argument("--output")
    extrinsics.set_defaults(func=command_extrinsics)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
