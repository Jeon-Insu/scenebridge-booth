from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from .io import load_calibration, load_json
from .local_object_tracking import PerCameraObjectTracker
from .object_centric_association import reconstruct_object_centric
from .object_reconstruction import StaticObjectTracker, reconstruct_static_objects
from .pose import UltralyticsPoseEstimator
from .publisher import make_tracking_message
from .recognition.objects import keep_counted, merge_detections
from .recognition.orientation import OrientationRunner
from .reconstruction import reconstruct_primary_person
from .types import FramePacket, Person3D, SceneObject3D

# v1.1 object keys: v1.2 refuses them rather than silently ignoring an old booth config.
OLD_OBJECT_KEYS = frozenset({
    "backend", "classes", "asset_manifest", "asset_map", "class_height_m", "class_aliases",
    "excluded_classes", "prop_detector", "prompt_detection", "appearance_asset_dir",
    "representative_image_dir", "representative_image_views", "disabled_prop_catalog_dir",
    "fallback_asset", "hold_frames",
})


def _rfdetr(**kwargs):
    from .object_detection import RFDETRSegObjectDetector
    return RFDETRSegObjectDetector(**kwargs)


@dataclass(slots=True)
class PerceptionResult:
    message: dict
    person: Person3D | None
    objects: list[SceneObject3D]
    detections_by_camera: dict
    tracked_by_detection: dict


class PerceptionProcessor:
    """Shared live/replay perception stack for one synchronized camera set."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        persist_appearance: bool = True,
        object_detection_overrides: dict | None = None,
        detector_factory=None,
        orientation_factory=None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = load_json(self.config_path)
        if object_detection_overrides:
            self.config["object_detection"] = {
                **self.config.get("object_detection", {}), **object_detection_overrides,
            }
        self.base = self.config_path.parent
        self.rig_to_scene = self.config.get("rig", {}).get("scene_placement")
        if self.config.get("venue"):
            # The venue preset decides how the table turns inside the background (config/venues.json).
            from .venues import load_venue
            venue = load_venue(self.config["venue"], check_files=False)  # Isaac checks the background
            self.rig_to_scene = {"translation_m": [0.0, 0.0, 0.0], "yaw_deg": venue.yaw_deg}
        camera_configs = self.config["cameras"]
        self.calibrations = {
            item["id"]: load_calibration(self.resolve(item["calibration"]))
            for item in camera_configs
        }

        self.estimator = (
            UltralyticsPoseEstimator(
                self.resolve(
                    self.config.get("pose_model", "../models/yolo11n-pose.pt")
                ),
                device=self.config.get("device"),
                image_size=int(self.config.get("inference_image_size", 640)),
                detection_confidence=float(
                    self.config.get("detection_confidence", 0.25)
                ),
                keypoint_confidence=float(
                    self.config.get("keypoint_confidence", 0.15)
                ),
            )
            if bool(self.config.get("pose_enabled", True))
            else None
        )

        self.object_config = self.config.get("object_detection", {})
        stale = sorted(set(self.object_config) & OLD_OBJECT_KEYS)
        if stale:
            raise ValueError(f"v1.2 설정이 아닙니다 (예전 키: {', '.join(stale)}). "
                             "예전 설정은 v1.1 브랜치에서 실행하세요")
        self.association_backend = str(self.object_config.get("association_backend", "object_centric"))
        if self.association_backend not in {"legacy", "object_centric"}:
            raise ValueError(f"Unsupported association backend: {self.association_backend}")
        object_enabled = bool(self.object_config.get("enabled", False))
        self.counts: dict[str, int] = {}
        self.display_names: dict[str, str] = {}
        self.asset_map: dict[str, str] = {}
        self.class_height_m: dict[str, float] = {}
        self.symmetric: set[str] = set()
        trained_ids: list[str] = []
        coco_names: dict[str, str] = {}   # COCO class name -> our prop_id
        if object_enabled:
            from .booth_props import BoothPropsError, check, load
            props = load(self.resolve(self.object_config["booth_props"]),
                         catalog_dir=self.resolve(self.object_config.get("prop_catalog_dir", "../../assets/props")))
            problems = check(props)
            if problems:
                raise BoothPropsError("; ".join(problems))
            for prop in props:
                self.counts[prop.prop_id] = prop.count
                self.display_names[prop.prop_id] = prop.name_ko
                self.asset_map[prop.prop_id] = str(prop.model)
                self.class_height_m[prop.prop_id] = prop.height_m
                if prop.symmetric:
                    self.symmetric.add(prop.prop_id)
                if prop.detector == "coco":
                    coco_names[prop.coco_class] = prop.prop_id
                else:
                    trained_ids.append(prop.prop_id)
        factory = detector_factory or _rfdetr
        common = dict(device=self.config.get("device"),
                      image_size=int(self.object_config.get("image_size", 312)),
                      mask_threshold=float(self.object_config.get("mask_threshold", 0.5)),
                      use_amp=bool(self.object_config.get("use_amp", True)))
        # Our trained props (A1) ...
        self.object_detector = factory(
            checkpoint=str(self.resolve(self.object_config["checkpoint"])),
            allowed_classes=sorted(trained_ids),
            confidence=float(self.object_config.get("confidence", 0.5)), **common,
        ) if object_enabled and trained_ids else None
        # ... and general objects (phone, cup, ...) from the COCO-pretrained detector, renamed to our prop ids.
        self.coco_detector = factory(
            checkpoint=None, allowed_classes=sorted(coco_names), class_aliases=dict(coco_names),
            confidence=float(self.object_config.get("coco_confidence", self.object_config.get("confidence", 0.5))),
            class_confidence=dict(self.object_config.get("coco_class_confidence", {})),  # e.g. phone lower
            **common,
        ) if object_enabled and coco_names else None
        orient = orientation_factory or OrientationRunner.from_files
        self.orientation = orient(
            head_path=self.resolve(self.object_config.get("orientation_head",
                                                          "../../outputs/models/orientation_head.pt")),
            symmetric=self.symmetric,
            interval_s=float(self.object_config.get("orientation_interval_s", 0.5)),
        ) if object_enabled else None
        self.fallback_asset_uri = None
        from .table_region import build_table_region_filter
        self.table_region = build_table_region_filter(
            self.object_config.get("table_region", {}), self.config.get("rig", {}), self.calibrations
        )

        local_tracking_config = self.object_config.get("local_tracking", {})
        self.local_object_tracker = (
            PerCameraObjectTracker(
                backend=str(local_tracking_config.get("backend", "bytetrack")),
                frame_rate=float(local_tracking_config.get("frame_rate", 5.0)),
                track_activation_threshold=float(
                    local_tracking_config.get("track_activation_threshold", 0.25)
                ),
                lost_track_buffer=int(
                    local_tracking_config.get("lost_track_buffer", 30)
                ),
                minimum_matching_threshold=float(
                    local_tracking_config.get("minimum_matching_threshold", 0.8)
                ),
                minimum_consecutive_frames=int(
                    local_tracking_config.get("minimum_consecutive_frames", 1)
                ),
            )
            if object_enabled and bool(local_tracking_config.get("enabled", True))
            else None
        )

        self.object_tracker = StaticObjectTracker(
            ema_alpha=float(self.object_config.get("ema_alpha", 0.15)),
            ema_alpha_z=(
                float(self.object_config["ema_alpha_z"])
                if "ema_alpha_z" in self.object_config
                else None
            ),
            min_confirmations=int(self.object_config.get("min_confirmations", 3)),
            hold_frames=None,
            hold_seconds=float(self.object_config.get("hold_seconds", 1.5)),
            reid_seconds=(
                float(self.object_config["reid_seconds"])
                if "reid_seconds" in self.object_config
                else 300.0
            ),
            max_association_distance_m=float(
                self.object_config.get("max_association_distance_m", 0.15)
            ),
            max_association_z_m=(
                float(self.object_config["max_association_z_m"])
                if "max_association_z_m" in self.object_config
                else None
            ),
            appearance_weight=float(self.object_config.get("appearance_weight", 0.35)),
            max_appearance_distance=float(
                self.object_config.get("max_appearance_distance", 0.65)
            ),
            local_track_match_bonus_m=float(
                self.object_config.get("local_track_match_bonus_m", 0.12)
            ),
            local_track_conflict_penalty_m=float(
                self.object_config.get("local_track_conflict_penalty_m", 0.20)
            ),
            asset_variant_max_distance=float(
                self.object_config.get("asset_variant_max_distance", 0.20)
            ),
        )

    @staticmethod
    def _detect(detector, images, frame_time_ns) -> dict:
        if detector is None:
            return {}
        if hasattr(detector, "set_frame_time"):
            detector.set_frame_time(frame_time_ns)
        if hasattr(detector, "infer_batch"):
            return detector.infer_batch(images)
        return {camera_id: detector.infer(camera_id, image) for camera_id, image in images.items()}

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.base / path).resolve()

    def process(
        self, packets: dict[str, FramePacket], frame_number: int
    ) -> PerceptionResult:
        started = perf_counter()
        people_by_camera = (
            {
                camera_id: self.estimator.infer(camera_id, packet.image)
                for camera_id, packet in packets.items()
                if packet.image is not None
            }
            if self.estimator is not None
            else {}
        )
        person = reconstruct_primary_person(
            people_by_camera,
            self.calibrations,
            min_keypoint_confidence=float(
                self.config.get("keypoint_confidence", 0.15)
            ),
            max_reprojection_error_px=float(
                self.config.get("max_reprojection_error_px", 25.0)
            ),
        )
        images = {
            camera_id: packet.image
            for camera_id, packet in packets.items()
            if packet.image is not None
        }
        frame_time_ns = int(
            sum(packet.capture_time_ns for packet in packets.values()) / len(packets)
        )
        detection_started = perf_counter()
        detections_by_camera = merge_detections(
            self._detect(self.object_detector, images, frame_time_ns),
            self._detect(self.coco_detector, images, frame_time_ns),
            overlap_iou=float(self.object_config.get("coco_overlap_iou", 0.5)),
        )
        if self.table_region is not None:
            # Before tracking: people, monitors and tripods around the booth never become objects.
            detections_by_camera = self.table_region.filter(detections_by_camera)
        detection_finished = perf_counter()
        if self.local_object_tracker is not None:
            detections_by_camera = self.local_object_tracker.update(
                detections_by_camera
            )
        timestamp_ns = frame_time_ns
        reconstruction_started = perf_counter()
        association_diagnostics = {"method": self.association_backend}
        reconstruct = (reconstruct_object_centric if self.association_backend == "object_centric"
                       else reconstruct_static_objects)
        extra = {"diagnostics": association_diagnostics,
                 "appearance_policy": str(self.object_config.get("appearance_policy", "strict"))
                 } if self.association_backend == "object_centric" else {}
        object_observations = reconstruct(
            detections_by_camera,
            self.calibrations,
            asset_map=self.asset_map,
            fallback_asset_uri=self.fallback_asset_uri,
            anchor=str(self.object_config.get("anchor", "bottom_center")),
            min_confidence=float(self.object_config.get("confidence", 0.25)),
            max_reprojection_error_px=float(
                self.object_config.get("max_reprojection_error_px", 100.0)
            ),
            max_instances_per_class=int(
                self.object_config.get("max_instances_per_class", 10)
            ),
            workspace_min_xyz=self.object_config.get("workspace_min_xyz"),
            workspace_max_xyz=self.object_config.get("workspace_max_xyz"),
            position_mode=str(
                self.object_config.get("position_mode", "triangulate")
            ),
            support_plane_z_m=float(
                self.object_config.get(
                    "support_plane_z_m",
                    self.config.get("rig", {}).get("support_plane_z_m", 0.0),
                )
            ),
            class_height_m=self.class_height_m,
            default_object_height_m=float(
                self.object_config.get("default_object_height_m", 0.10)
            ),
            snap_to_support_plane_m=float(
                self.object_config.get("snap_to_support_plane_m", 0.08)
            ),
            cross_view_appearance_weight_px=float(
                self.object_config.get("cross_view_appearance_weight_px", 12.0)
            ),
            max_cross_view_appearance_distance=float(
                self.object_config.get("max_cross_view_appearance_distance", 0.80)
            ),
            previous_track_groups=self.object_tracker.local_track_groups(),
            **extra,
        )
        object_observations = keep_counted(object_observations, self.counts)
        reconstruction_finished = perf_counter()
        objects = self.object_tracker.update(
            object_observations,
            timestamp_ns=timestamp_ns,
            detections_by_camera=detections_by_camera,
        )
        if self.orientation is not None:
            objects = self.orientation.apply(objects, detections_by_camera, images, self.calibrations, timestamp_ns)
        objects = [replace(o, display_name=self.display_names.get(o.label)) for o in objects]
        tracked_by_detection = {
            key: item
            for item in objects
            if item.state != "occluded"
            for key in item.source_detection_keys
        }
        message = make_tracking_message(
                timestamp_ns,
                frame_number,
                [person] if person else [],
                objects,
                self.rig_to_scene,
            )
        message["performance"] = {
            "detector_adapter_ms": (detection_finished - detection_started) * 1000,
            "local_tracking_ms": (reconstruction_started - detection_finished) * 1000,
            "reconstruction_ms": (reconstruction_finished - reconstruction_started) * 1000,
            "global_tracking_assets_ms": (perf_counter() - reconstruction_finished) * 1000,
            "processor_ms": (perf_counter() - started) * 1000,
            "timing_basis": "CPU wall clock including adapter transfers; not isolated GPU kernel time",
        }
        message["association"] = association_diagnostics
        return PerceptionResult(
            message=message,
            person=person,
            objects=objects,
            detections_by_camera=detections_by_camera,
            tracked_by_detection=tracked_by_detection,
        )
