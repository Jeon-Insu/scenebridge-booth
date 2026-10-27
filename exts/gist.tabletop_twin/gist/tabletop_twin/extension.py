from __future__ import annotations

import json
import asyncio
import hashlib
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import carb
import cv2
import omni.ext
import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

from .compass import ViewportCompass
from .asset_studio import AssetStudioMixin
from .simulation_mode import SimulationMode
import omni.timeline
from .camera_layout import split_camera_analysis_mosaic
from .focus import read_linked_preview, resolve_focus_ids, render_camera_focus
from .contact import read_contact_geometry, resolve_placement, inspect_contact, PLACEMENT_MODES, set_table_size
from .tracked_pose import display_label, shown_yaw_deg, tracked_yaw_deg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
# Booth table size (settings table_length_m / table_width_m, from booth.json rig.workspace_size_m via venue_args).
TABLE_SIZE_M = [1.25, 0.75, 0.75]


def table_asset_path() -> Path:
    cm = [int(round(v * 100)) for v in TABLE_SIZE_M]
    return PROJECT_ROOT / "assets" / "tabletop_demo" / f"table_{cm[0]}x{cm[1]}x{cm[2]}.usda"
CATALOG_PATH = PROJECT_ROOT / "assets" / "tabletop_decor" / "catalog.json"
STATE_PATH = PROJECT_ROOT / "outputs" / "tabletop_twin" / "scene_state.json"
CAMERA_PREVIEW_PATH = PROJECT_ROOT / "outputs" / "tabletop_camera_mosaic.jpg"
DEFAULT_AGENT_ROOT = PROJECT_ROOT / "outputs" / "tabletop_agent"
OBJECTS_ROOT = Sdf.Path("/World/Tracking/Objects")
AGENT_PREVIEW_ROOT = Sdf.Path("/World/AgentPreview")
DEMO_CAMERA_PATH = Sdf.Path("/World/TabletopDemo/DemoCamera")
HIDDEN_DEMO_PANELS = ("Layer", "Layers", "Render Settings", "Robot Inspector", "Property", "Properties",
                      "Main ToolBar", "Physics Toolbar")
# A docked layout narrower than this share of the main window left empty panes beside it.
MIN_VIEWPORT_WIDTH_SHARE = 0.5
MAX_LAYOUT_REDOS = 5
FORMER_OFFICE_PATH = (
    "/World/Environment/Visual/Buildings/"
    "Building_Former_South_Jeolla_Province_Office"
)


def _safe_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "unknown"
    return sanitized if not sanitized[0].isdigit() else f"Item_{sanitized}"


def _capture_viewport_interaction_frame(
    stage,
    viewport,
    message_translation: tuple[float, float, float],
    offset: tuple[float, float, float],
    message_yaw_rad: float,
) -> dict[str, Any]:
    """Project Viewport screen axes onto the tabletop and return TableFrame vectors."""
    result: dict[str, Any] = {
        "captured_at_ns": time.time_ns(),
        "table_axes": {
            "east": [1.0, 0.0, 0.0],
            "north": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
        "workspace_bounds_xy": [0.0, TABLE_SIZE_M[0], 0.0, TABLE_SIZE_M[1]],
        "viewport": {"available": False},
    }
    if stage is None or viewport is None:
        return result
    camera_path = Sdf.Path(str(viewport.camera_path))
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        return result
    matrix = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    origin = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    forward = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    camera_right = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
    camera_up = matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized()
    plane_z = message_translation[2] + offset[2]

    def plane_intersection(ray: Gf.Vec3d) -> Gf.Vec3d | None:
        if abs(float(ray[2])) < 1e-6:
            return None
        distance = (plane_z - float(origin[2])) / float(ray[2])
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        return origin + ray * distance

    center = plane_intersection(forward)
    right_point = plane_intersection((forward + camera_right * 0.05).GetNormalized())
    up_point = plane_intersection((forward + camera_up * 0.05).GetNormalized())
    if center is None or right_point is None or up_point is None:
        return result

    cosine = math.cos(message_yaw_rad)
    sine = math.sin(message_yaw_rad)

    def to_table_unit(delta: Gf.Vec3d) -> list[float] | None:
        world_x, world_y = float(delta[0]), float(delta[1])
        table_x = cosine * world_x + sine * world_y
        table_y = -sine * world_x + cosine * world_y
        length = math.hypot(table_x, table_y)
        if not math.isfinite(length) or length < 1e-6:
            return None
        return [round(table_x / length, 6), round(table_y / length, 6)]

    screen_right = to_table_unit(right_point - center)
    screen_up = to_table_unit(up_point - center)
    if screen_right is None or screen_up is None:
        return result
    result["viewport"] = {
        "available": True,
        "camera_path": str(camera_path),
        "screen_right_table_xy": screen_right,
        "screen_up_table_xy": screen_up,
    }
    return result


@dataclass
class ObjectRecord:
    object_id: str
    label: str
    prim_path: str
    state: str
    asset_uri: str
    asset_key: str
    uses_catalog_fallback: bool
    xyz_m: tuple[float, float, float]
    position_mode: str
    source_camera_count: int
    last_seen_monotonic: float
    last_seen_timestamp_ns: int
    tracked_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    table_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    agent_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    last_placement: str = "preserve_position"
    display_name: str = ""
    tracked_yaw_deg: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("last_seen_monotonic", None)
        return result


class TabletopTwinExtension(AssetStudioMixin, omni.ext.IExt):
    """User-facing object-only twin for the tabletop booth demo.

    The perception process remains replaceable. It sends the existing
    gist.smartphone_mocap.v1 UDP schema; this extension owns the bounded USD
    object registry, UI selection, and the human-approved reconstruction queue.
    """

    def on_startup(self, _ext_id: str) -> None:
        self._simulation = SimulationMode(omni.timeline.get_timeline_interface())
        self._agent_session_id = f"tabletop-{time.time_ns()}"
        self._agent_conversation_id = f"conversation-{time.time_ns()}"
        self._applied_action_ids = set()
        self._physics_properties = {}
        self._physics_table_material = None
        settings = carb.settings.get_settings()
        prefix = "/exts/gist.tabletop_twin"
        self._port = settings.get_as_int(f"{prefix}/udp_port")
        self._offset = (
            settings.get_as_float(f"{prefix}/world_offset_x"),
            settings.get_as_float(f"{prefix}/world_offset_y"),
            settings.get_as_float(f"{prefix}/world_offset_z"),
        )
        # Venue preset camera (config/venues.json → tools/venue_args.py): world eye and target.
        eye = [settings.get(f"{prefix}/camera_eye_{axis}") for axis in "xyz"]
        target = [settings.get(f"{prefix}/camera_target_{axis}") for axis in "xyz"]
        self._venue_camera = (
            (tuple(float(v) for v in eye), tuple(float(v) for v in target))
            if all(v is not None for v in (*eye, *target)) else None
        )
        self._background_stage_path = settings.get_as_string(
            f"{prefix}/background_stage_path"
        )
        for index, key in enumerate(("table_length_m", "table_width_m")):
            value = settings.get(f"{prefix}/{key}")
            if value is not None:
                TABLE_SIZE_M[index] = float(value)
        set_table_size(TABLE_SIZE_M[0], TABLE_SIZE_M[1])
        self._stale_remove_seconds = max(
            0.1, settings.get_as_float(f"{prefix}/stale_remove_seconds")
        )
        self._max_tracked_objects = max(
            1, settings.get_as_int(f"{prefix}/max_tracked_objects")
        )
        self._auto_open_window = settings.get_as_bool(f"{prefix}/auto_open_window")
        self._auto_frame_demo_context = settings.get_as_bool(
            f"{prefix}/auto_frame_demo_context"
        )
        camera_preview_hz = max(
            1.0, settings.get_as_float(f"{prefix}/camera_preview_refresh_hz")
        )
        self._camera_preview_interval = 1.0 / camera_preview_hz
        configured_control_dir = settings.get_as_string(f"{prefix}/agent_control_dir")
        self._agent_root = Path(configured_control_dir or DEFAULT_AGENT_ROOT)
        if not self._agent_root.is_absolute():
            self._agent_root = (PROJECT_ROOT / self._agent_root).resolve()
        self._agent_url = settings.get_as_string(f"{prefix}/agent_service_url")
        self._agent_timeout_seconds = max(
            1.0, settings.get_as_float(f"{prefix}/agent_timeout_seconds")
        )
        self._agent_snapshot_wait_seconds = max(
            1.0, settings.get_as_float(f"{prefix}/agent_snapshot_wait_seconds")
        )
        self._agent_auto_resume = settings.get_as_bool(
            f"{prefix}/agent_auto_resume_after_response"
        )

        self._catalog = self._load_catalog()
        self._catalog_by_class = self._index_catalog(self._catalog)
        self._records: dict[str, ObjectRecord] = {}
        self._focused_object_ids = []
        self._focus_signature = None
        self._object_short_names = {}
        self._object_name_counts = {}
        self._focus_view_status = {}
        self._focus_buttons_signature = None
        self._camera_preview_file_signature = None
        self._camera_base_image = None
        self._camera_link_metadata = None
        self._camera_focus_dirty = True
        self._focus_thumbnail_provider = ui.ByteImageProvider()
        self._focus_thumbnail_provider.set_bytes_data(bytearray([24, 24, 24, 255]) * 72 * 48, [72, 48])
        self._agent_offsets: dict[str, tuple[float, float, float]] = {}
        self._agent_transforms: dict[str, dict[str, Any]] = {}
        self._agent_preview: dict[str, Any] = {}
        self._compass = ViewportCompass()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._messages: deque[dict[str, Any]] = deque(maxlen=2)
        self._latest_tracking_message: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._message_translation = (0.0, 0.0, 0.0)
        self._message_yaw_rad = 0.0
        # Observation/replay input is independent of the Isaac timeline.  A
        # timeline pause alone therefore does not freeze the mirrored twin.
        # Keep an explicit gate so Pause also freezes incoming UDP/replay data.
        self._observation_paused = False
        self._last_stage_id: int | None = None
        self._frame_context_pending = False
        self._frame_context_next_at = 0.0
        self._frame_context_stop_at = 0.0
        self._camera_focus_at = 0.0
        self._background_open_pending = bool(self._background_stage_path)
        self._background_open_at = time.monotonic() + 1.0
        self._background_open_attempts = 0
        self._last_housekeeping = 0.0
        self._last_ui_refresh = 0.0
        self._last_camera_preview_check = 0.0
        self._camera_preview_mtime_ns = 0
        self._ui_dirty = True
        self._state_dirty = True
        self._status = f"Waiting for tracking data on UDP {self._port}"
        self._agent_messages: deque[dict[str, str]] = deque(maxlen=200)
        self._agent_messages.append(
            {
                "role": "Agent",
                "text": "준비되었습니다. 질문을 보내면 현재 장면을 고정한 뒤 답변합니다.",
            }
        )
        self._agent_input_model = ui.SimpleStringModel("")
        self._agent_input_editing = False
        self._agent_input_model.add_begin_edit_fn(self._on_agent_input_begin)
        self._agent_input_model.add_end_edit_fn(self._on_agent_input_end)
        self._agent_mode = "live"
        self._agent_provider = "agent service"
        self._agent_busy = False
        self._agent_request_id = ""
        self._agent_snapshot_id = ""
        self._agent_snapshot_path = ""
        self._agent_results: deque[dict[str, Any]] = deque(maxlen=8)
        self._agent_capture_jobs = deque()
        self._agent_capture_tasks = set()
        self._agent_query_id = ""
        self._agent_evidence_status = "정보원: 질문에 따라 선택"
        self._agent_evidence_trace = []
        self._agent_thread: threading.Thread | None = None
        self._last_interaction_frame: dict[str, Any] = {}
        self._last_agent_runtime_check = 0.0
        self._last_agent_runtime_mtime_ns = 0
        self._layout_pending = self._auto_open_window
        self._layout_attempts = 0
        self._layout_phase = 0
        self._layout_redos = 0
        self._layout_apply_at = time.monotonic() + 10.0
        # Full App restores built-in windows in several startup phases. Suppress
        # auxiliary panes during that brief period, not just at our first dock.
        self._startup_layout_guard_until = time.monotonic() + 25.0
        self._agent_ui_built = False
        self._agent_history_signature = None
        self._agent_scroll_restore = None
        self._agent_scroll_restore_frames = 0

        self._window = ui.Window(
            "Tabletop Agent",
            width=550,
            height=850,
            visible=self._auto_open_window,
        )
        self._window.set_key_pressed_fn(self._on_agent_key_pressed)
        self._settings_window = ui.Window("Tabletop Settings", width=960, height=340,
                                          visible=self._auto_open_window)
        self._rebuild_ui()

        self._camera_providers = {}
        for kind in ("detection", "segmentation"):
            provider = ui.ByteImageProvider()
            provider.set_bytes_data([32, 34, 38, 255] * 4, [2, 2])
            self._camera_providers[kind] = provider
        self._camera_status_text = "Waiting for the four-camera preview stream..."
        self._camera_status_labels = {}
        self._camera_grid_widgets = {}
        self._camera_window = ui.Window(
            "Tabletop Cameras",
            width=1200,
            height=340,
            visible=self._auto_open_window,
        )
        self._build_camera_ui()
        self._start_receiver()
        self._subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update, name="GIST Tabletop Twin scene update"
            )
        )
        carb.log_info(
            f"[GIST Tabletop Twin] Started with {len(self._catalog.get('assets', []))} "
            f"curated assets; UDP 0.0.0.0:{self._port}"
        )

    def on_shutdown(self) -> None:
        task = getattr(self, "_agent_apply_task", None)
        if task is not None:
            task.cancel()
        self._simulation.leave()
        self._compass.destroy()
        self._stop.set()
        for task in self._agent_capture_tasks:
            task.cancel()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._agent_thread is not None:
            self._agent_thread.join(timeout=0.25)
        self._subscription = None
        for name in ("_window", "_settings_window", "_camera_window"):
            window = getattr(self, name, None)
            if window is not None:
                window.destroy()
            setattr(self, name, None)
        self._camera_providers = {}
        self._camera_status_labels = {}
        self._camera_grid_widgets = {}
        self._agent_input_model = None
        self._window = None

    def _load_catalog(self) -> dict[str, Any]:
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            carb.log_error(f"[GIST Tabletop Twin] Could not load {CATALOG_PATH}: {exc}")
            return {"assets": [], "theme": "unavailable"}

    @staticmethod
    def _index_catalog(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for asset in catalog.get("assets", []):
            for class_name in asset.get("expected_detector_classes", []):
                index.setdefault(str(class_name).casefold(), asset)
        return index

    def _start_receiver(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(0.25)
        try:
            self._socket.bind(("0.0.0.0", self._port))
        except OSError as exc:
            carb.log_error(f"[GIST Tabletop Twin] Could not bind UDP {self._port}: {exc}")
            self._status = f"Failed to open UDP port {self._port}"
            self._socket.close()
            self._socket = None
            self._ui_dirty = True
            return
        self._thread = threading.Thread(
            target=self._receive_loop, name="gist-tabletop-twin-udp", daemon=True
        )
        self._thread.start()

    def _receive_loop(self) -> None:
        while not self._stop.is_set() and self._socket is not None:
            try:
                payload, _address = self._socket.recvfrom(65507)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                message = json.loads(payload.decode("utf-8"))
                if message.get("schema") != "gist.smartphone_mocap.v1":
                    continue
                with self._lock:
                    self._messages.append(message)
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
                carb.log_warn(f"[GIST Tabletop Twin] Ignored malformed UDP packet: {exc}")

    def _on_update(self, _event) -> None:
        now = time.monotonic()
        if self._auto_open_window and now < self._startup_layout_guard_until:
            self._hide_auxiliary_panels()
        if self._agent_scroll_restore_frames:
            self._agent_scroll_restore_frames -= 1
            scroll = self._agent_history_scroll
            scroll.scroll_y = scroll.scroll_y_max if self._agent_scroll_restore is None else self._agent_scroll_restore
        if self._layout_pending:
            self._try_apply_default_layout()
        usd_context = omni.usd.get_context()
        if self._background_open_pending and now >= self._background_open_at:
            self._background_open_attempts += 1
            if usd_context.get_stage_url() == self._background_stage_path:
                self._background_open_pending = False
            elif usd_context.can_open_stage():
                if usd_context.open_stage(self._background_stage_path):
                    carb.log_info(
                        "[GIST Tabletop Twin] Opened background stage: "
                        f"{self._background_stage_path}"
                    )
                    self._background_open_pending = False
                    return
            self._background_open_at = now + 1.0
            if self._background_open_attempts >= 30:
                carb.log_error(
                    "[GIST Tabletop Twin] Could not open background stage: "
                    f"{self._background_stage_path}"
                )
                self._background_open_pending = False

        stage = usd_context.get_stage()
        if stage is None:
            return
        if self._simulation.active and (stage != self._simulation.stage or
                any(not stage.GetPrimAtPath(path).IsValid() for path in self._simulation.body_paths)):
            self._simulation.leave()
            self._agent_messages.append({"role":"System", "text":"실험 장면이 변경되어 중단했습니다. 관측 장면에서 실험을 다시 준비해 주세요."})
            self._ui_dirty = True
            self._state_dirty = True

        if now - self._last_camera_preview_check >= self._camera_preview_interval:
            self._update_camera_preview()
            self._last_camera_preview_check = now
        if now - self._last_agent_runtime_check >= 0.25:
            self._poll_agent_runtime()
            progress = self._read_json_file(self._agent_root / "progress.json")
            if self._agent_busy and progress and progress.get("request_id") == self._agent_query_id:
                entries = progress.get("evidence_trace", []) + ([progress["current"]] if progress.get("current") else [])
                names = []
                for entry in entries:
                    source = entry.get("source", "")
                    label = {"conversation archive":"대화 기록", "user statement":"사용자 정보", "backend execution":"실행 기록"}.get(source)
                    label = label or ("실물 카메라" if source == "physical" and entry.get("tool") == "inspect_images" else "실물 관측" if source == "physical" else "트윈 이미지" if entry.get("tool") == "inspect_images" else "트윈 상태")
                    if label not in names:
                        names.append(label)
                self._agent_evidence_status = "정보원: " + (" → ".join(names) if names else "선택 중")
                self._ui_dirty = True
            self._last_agent_runtime_check = now
        self._drain_agent_results(stage)
        self._poll_asset_studio(stage, now)

        if self._simulation.active:
            # Drain/discard observations; simulation transforms have sole ownership.
            with self._lock:
                self._messages.clear()
            while self._agent_capture_jobs:
                job = self._agent_capture_jobs.popleft()
                task = asyncio.ensure_future(self._prepare_agent_evidence(job))
                self._agent_capture_tasks.add(task)
                task.add_done_callback(self._agent_capture_tasks.discard)
            if self._ui_dirty:
                self._rebuild_ui()
                self._ui_dirty = False
            self._write_state_snapshot()
            return

        if self._observation_paused:
            # The replay producer may continue publishing UDP frames, but a
            # paused observation must not mutate the twin or its object IDs.
            with self._lock:
                self._messages.clear()
            while self._agent_capture_jobs:
                job = self._agent_capture_jobs.popleft()
                task = asyncio.ensure_future(self._prepare_agent_evidence(job))
                self._agent_capture_tasks.add(task)
                task.add_done_callback(self._agent_capture_tasks.discard)
            if self._ui_dirty:
                self._rebuild_ui()
                self._ui_dirty = False
            self._write_state_snapshot()
            return

        stage_id = id(stage)
        if stage_id != self._last_stage_id:
            self._last_stage_id = stage_id
            self._records.clear()
            self._object_short_names.clear()
            self._object_name_counts.clear()
            self._focus_signature = None
            self._agent_offsets.clear()
            self._agent_transforms.clear()
            self._clear_agent_preview(stage)
            self._ensure_scene(stage)
            self._frame_context_pending = self._auto_frame_demo_context
            # Isaac's GUI Viewport becomes usable a few seconds after the USD
            # stage first appears. Reframe several times across that interval;
            # an early successful command can otherwise be overwritten by the
            # Full App's own initial camera setup.
            self._frame_context_next_at = now + 5.0
            self._frame_context_stop_at = now + 18.0
            self._camera_focus_at = now + 19.0
            self._ui_dirty = True
            self._state_dirty = True

        if self._frame_context_pending and now >= self._frame_context_next_at:
            self._frame_demo_context()
            self._frame_context_next_at = now + 2.0
            if now >= self._frame_context_stop_at:
                self._frame_context_pending = False

        if self._camera_focus_at and now >= self._camera_focus_at:
            viewport_window = ui.Workspace.get_window("Viewport")
            if viewport_window is not None:
                viewport_window.focus()
            if self._camera_window is not None:
                self._camera_window.focus()
            if self._window is not None:
                self._window.focus()
            self._camera_focus_at = 0.0

        message = None
        with self._lock:
            if self._messages:
                message = self._messages.pop()
                self._messages.clear()
        # A hot reload clears in-memory records while the producer can remain
        # frozen. Restore that exact frozen tracking snapshot once; do not
        # switch video sources or overwrite an existing/cleared operator scene.
        if (message is None and self._latest_tracking_message is None
                and not self._records and self._agent_mode == "snapshot"
                and self._agent_snapshot_path):
            saved = self._read_json_file(Path(self._agent_snapshot_path)) or {}
            tracking = saved.get("tracking", {})
            if saved.get("snapshot_id") == self._agent_snapshot_id and tracking.get("objects"):
                message = tracking
                self._agent_messages.append({"role": "System",
                    "text": "확장 재시작 후 고정된 입력 장면에서 3D 객체를 복구했습니다."})
        if message is not None:
            self._apply_message(stage, message, now)
        self._update_focus_selection(stage)
        while self._agent_capture_jobs:
            job = self._agent_capture_jobs.popleft()
            task = asyncio.ensure_future(self._prepare_agent_evidence(job))
            self._agent_capture_tasks.add(task)
            task.add_done_callback(self._agent_capture_tasks.discard)
        self._compass.update(stage, self._message_yaw_rad)

        if self._agent_mode == "live" and now - self._last_housekeeping >= 0.5:
            self._remove_stale_objects(stage, now)
            self._last_housekeeping = now
        if self._ui_dirty and now - self._last_ui_refresh >= 0.25:
            self._rebuild_ui()
            self._last_ui_refresh = now
            self._ui_dirty = False
        if self._state_dirty:
            self._write_state_snapshot()
            self._state_dirty = False

    def _try_apply_default_layout(self) -> None:
        """Full-height Agent/Stage right; camera/settings/content tabs below Viewport."""
        self._layout_attempts += 1
        if time.monotonic() < self._layout_apply_at:
            return
        viewport = ui.Workspace.get_window("Viewport")
        root = ui.Workspace.get_window("DockSpace")
        stage_window = ui.Workspace.get_window("Stage")
        if viewport is None or root is None or self._window is None:
            return
        try:
            self._hide_auxiliary_panels()
            # Dock calls are deferred by Kit. Let each split settle before using
            # its new node; otherwise hiding old tabs can leave Agent floating.
            if self._layout_phase == 0:
                viewport.dock_in(root, ui.DockPosition.SAME)
            elif self._layout_phase == 1:
                self._window.dock_in(viewport, ui.DockPosition.RIGHT, 0.30)
            elif self._layout_phase == 2:
                if stage_window is not None:
                    stage_window.dock_in(self._window, ui.DockPosition.SAME)
                self._camera_window.dock_in(viewport, ui.DockPosition.BOTTOM, 0.35)
            elif self._layout_phase == 3:
                for window in (self._settings_window,
                               ui.Workspace.get_window("Content"), ui.Workspace.get_window("Console")):
                    if window is not None:
                        window.dock_in(self._camera_window, ui.DockPosition.SAME)
            else:
                handles = [ui.Workspace.get_window(title) for title in
                           ("Viewport", "Tabletop Agent", "Tabletop Cameras")]
                valid = all(handle is not None and handle.docked and handle.dock_id != 0 for handle in handles)
                if not valid or len({handle.dock_id for handle in handles if handle}) != 3:
                    self._layout_phase = 0
                    self._layout_apply_at = time.monotonic() + 0.5
                    return
                # 2026-10-25: docking too early at startup left the old Viewport node as
                # empty panes and squeezed the Viewport into a side column; dock again.
                if (viewport.width < MIN_VIEWPORT_WIDTH_SHARE * ui.Workspace.get_main_window_width()
                        and self._layout_redos < MAX_LAYOUT_REDOS):
                    self._layout_redos += 1
                    self._layout_phase = 0
                    self._layout_apply_at = time.monotonic() + 1.0
                    carb.log_warn(f"[GIST Tabletop Twin] Viewport too narrow ({viewport.width:.0f}px); docking again")
                    return
                viewport.focus()
                self._camera_window.focus()
                self._window.focus()
                self._layout_pending = False
                return
            self._layout_phase += 1
            self._layout_apply_at = time.monotonic() + 0.15
        except (RuntimeError, AttributeError) as exc:
            if self._layout_attempts % 120 == 0:
                carb.log_warn(f"[GIST Tabletop Twin] Waiting to apply UI layout: {exc}")

    @staticmethod
    def _hide_auxiliary_panels() -> None:
        changed = False
        for title in HIDDEN_DEMO_PANELS:
            window = ui.Workspace.get_window(title)
            if window is not None and window.visible:
                window.visible = False
                changed = True
        if changed:
            viewport = ui.Workspace.get_window("Viewport")
            if viewport is not None:
                viewport.focus()

    def _ensure_scene(self, stage) -> None:
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/TabletopDemo")
        camera = UsdGeom.Camera.Define(stage, DEMO_CAMERA_PATH)
        camera.GetFocalLengthAttr().Set(24.0)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))
        camera_prim = camera.GetPrim()
        camera_xform = UsdGeom.Xformable(camera_prim)
        camera_transform = camera_prim.GetAttribute("xformOp:transform")
        if not camera_transform:
            camera_transform = camera_xform.AddTransformOp().GetAttr()
        camera_transform.Set(
            Gf.Matrix4d(1.0)
            .SetLookAt(
                Gf.Vec3d(30.0, 25.0, 22.0),
                Gf.Vec3d(43.0, -8.0, 1.5),
                Gf.Vec3d(0.0, 0.0, 1.0),
            )
            .GetInverse()
        )
        table_prim = stage.GetPrimAtPath("/World/TabletopDemo/Table")
        if not table_prim.IsValid():
            table_prim = stage.DefinePrim("/World/TabletopDemo/Table", "Xform")
            table_prim.GetReferences().AddReference(str(table_asset_path()))
        self._set_table_transform(table_prim)
        marker = table_prim.GetAttribute("gist:managedByTabletopTwin")
        if not marker:
            marker = table_prim.CreateAttribute(
                "gist:managedByTabletopTwin", Sdf.ValueTypeNames.Bool, custom=True
            )
        marker.Set(True)
        UsdGeom.Xform.Define(stage, "/World/Tracking")
        UsdGeom.Xform.Define(stage, OBJECTS_ROOT)

    def _set_table_transform(self, table_prim) -> None:
        translation = Gf.Vec3d(
            self._message_translation[0] + self._offset[0],
            self._message_translation[1] + self._offset[1],
            self._message_translation[2] + self._offset[2],
        )
        xformable = UsdGeom.Xformable(table_prim)
        translate = table_prim.GetAttribute("xformOp:translate")
        if not translate:
            translate = xformable.AddTranslateOp().GetAttr()
        translate.Set(translation)
        rotation = table_prim.GetAttribute("xformOp:rotateZ")
        if not rotation:
            rotation = xformable.AddRotateZOp().GetAttr()
        rotation.Set(math.degrees(self._message_yaw_rad))

    def _apply_message(self, stage, message: dict[str, Any], now: float) -> None:
        self._latest_tracking_message = message
        placement = message.get("rig_to_scene", {})
        translation = placement.get("translation_m", [0.0, 0.0, 0.0])
        if (
            isinstance(translation, list)
            and len(translation) == 3
            and all(math.isfinite(float(value)) for value in translation)
        ):
            self._message_translation = tuple(float(value) for value in translation)
        else:
            self._message_translation = (0.0, 0.0, 0.0)
        yaw_deg = float(placement.get("yaw_deg", 0.0))
        self._message_yaw_rad = math.radians(yaw_deg if math.isfinite(yaw_deg) else 0.0)
        table_prim = stage.GetPrimAtPath("/World/TabletopDemo/Table")
        if table_prim.IsValid():
            self._set_table_transform(table_prim)

        timestamp_ns = int(message.get("timestamp_ns", 0) or 0)
        received_ids: set[str] = set()
        for scene_object in message.get("objects", []):
            raw_id = str(scene_object.get("id", "unknown"))
            received_ids.add(raw_id)
            self._update_object(stage, raw_id, scene_object, now, timestamp_ns)

        # Tracking packets are complete scene snapshots, not incremental
        # updates.  Once the tracker has dropped an ID, remove its prim now;
        # keeping it until the generic connection timeout created 30-second
        # "ghost" objects in the viewport.
        for missing_id in set(self._records) - received_ids:
            if missing_id.startswith('asset_'):
                continue  # Manually registered assets have no detector lifecycle.
            self._remove_record(stage, missing_id)

        self._enforce_object_limit(stage)
        visible_count = sum(
            record.state in {"active", "partial", "uncertain"}
            for record in self._records.values()
        )
        self._status = (
            f"Connected | visible {visible_count} | remembered {len(self._records)}"
        )
        # Do not rebuild the chat view for every camera packet; doing so would
        # steal keyboard focus from the agent input field while the user types.
        self._ui_dirty = True  # Only Settings rebuilds; Agent input stays alive.
        self._state_dirty = True

    def _object_path(self, raw_id: str, label: str) -> Sdf.Path:
        parts = [part for part in raw_id.split("/") if part]
        if len(parts) < 2:
            parts = [label or "object", raw_id]
        path = OBJECTS_ROOT
        for part in parts:
            path = path.AppendChild(_safe_identifier(part))
        return path

    def _with_offset(self, xyz: list[Any]) -> Gf.Vec3d:
        cosine = math.cos(self._message_yaw_rad)
        sine = math.sin(self._message_yaw_rad)
        rotated_x = cosine * float(xyz[0]) - sine * float(xyz[1])
        rotated_y = sine * float(xyz[0]) + cosine * float(xyz[1])
        return Gf.Vec3d(
            rotated_x + self._message_translation[0] + self._offset[0],
            rotated_y + self._message_translation[1] + self._offset[1],
            float(xyz[2]) + self._message_translation[2] + self._offset[2],
        )

    def _catalog_asset(self, label: str) -> dict[str, Any] | None:
        return self._catalog_by_class.get(label.casefold())

    def _resolve_asset(self, label: str, raw_uri: Any) -> tuple[str, bool]:
        supplied_uri = str(raw_uri) if isinstance(raw_uri, str) else ""
        catalog_asset = self._catalog_asset(label)
        uses_fallback = not supplied_uri or Path(supplied_uri).name == "generic.usda"
        if not uses_fallback:
            return supplied_uri, False
        if catalog_asset is None:
            return supplied_uri, True
        return str(CATALOG_PATH.parent / catalog_asset["usd_uri"]), True

    def _update_object(
        self,
        stage,
        raw_id: str,
        scene_object: dict[str, Any],
        now: float,
        timestamp_ns: int,
    ) -> None:
        xyz = scene_object.get("xyz")
        if not isinstance(xyz, list) or len(xyz) != 3:
            return
        try:
            if not all(math.isfinite(float(value)) for value in xyz):
                return
        except (TypeError, ValueError):
            return

        label = str(scene_object.get("label", "object"))
        object_path = self._object_path(raw_id, label)
        for element in object_path.GetPrefixes():
            if element == OBJECTS_ROOT or element.HasPrefix(OBJECTS_ROOT):
                UsdGeom.Xform.Define(stage, element)
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        state = str(scene_object.get("state", "active"))
        raw_asset_uri = scene_object.get("asset_uri")
        asset_uri, uses_fallback = self._resolve_asset(label, raw_asset_uri)

        UsdGeom.Imageable(object_prim).GetVisibilityAttr().Set(
            UsdGeom.Tokens.invisible
            if state in {"candidate", "tentative", "occluded", "dropped"}
            else UsdGeom.Tokens.inherited
        )
        metadata = {
            "gist:isTrackedObject": (Sdf.ValueTypeNames.Bool, True),
            "gist:label": (Sdf.ValueTypeNames.String, label),
            "gist:state": (Sdf.ValueTypeNames.String, state),
            "gist:assetKey": (
                Sdf.ValueTypeNames.String,
                str(scene_object.get("asset_key") or ""),
            ),
            "gist:positionMode": (
                Sdf.ValueTypeNames.String,
                str(scene_object.get("position_mode") or ""),
            ),
            "gist:confidence": (
                Sdf.ValueTypeNames.Double,
                float(scene_object.get("confidence", 0.0)),
            ),
            "gist:reprojectionErrorPx": (
                Sdf.ValueTypeNames.Double,
                float(scene_object.get("reprojection_error_px", 0.0)),
            ),
            "gist:sourceCameras": (
                Sdf.ValueTypeNames.String,
                ",".join(
                    str(value) for value in scene_object.get("source_cameras", [])
                ),
            ),
            "gist:sourceTracks": (
                Sdf.ValueTypeNames.String,
                ",".join(
                    f"{value.get('camera_id')}:{value.get('local_track_id')}"
                    for value in scene_object.get("source_tracks", [])
                    if isinstance(value, dict)
                ),
            ),
            "gist:usesCatalogFallback": (Sdf.ValueTypeNames.Bool, uses_fallback),
        }
        for name, (value_type, value) in metadata.items():
            attribute = object_prim.GetAttribute(name)
            if not attribute:
                attribute = object_prim.CreateAttribute(name, value_type, custom=True)
            attribute.Set(value)

        xformable = UsdGeom.Xformable(object_prim)
        translate = object_prim.GetAttribute("xformOp:translate")
        if not translate:
            translate = xformable.AddTranslateOp().GetAttr()
        agent_offset = self._agent_offsets.get(raw_id, (0.0, 0.0, 0.0))
        table_xyz = tuple(float(xyz[i]) + agent_offset[i] for i in range(3))
        point = self._with_offset(table_xyz)
        translate.Set(point)
        object_prim.CreateAttribute("gist:agentOffsetM", Sdf.ValueTypeNames.Double3, custom=True).Set(Gf.Vec3d(*agent_offset))
        transform = self._agent_transforms.get(raw_id, {})
        self._write_shape_transform(object_prim, transform, tracked_yaw_deg(scene_object))

        asset_path = object_path.AppendChild("Asset")
        proxy_path = object_path.AppendChild("Proxy")
        if asset_uri:
            if stage.GetPrimAtPath(proxy_path).IsValid():
                stage.RemovePrim(proxy_path)
            asset_prim = stage.DefinePrim(asset_path, "Xform")
            uri_attribute = asset_prim.GetAttribute("gist:assetUri")
            if not uri_attribute:
                uri_attribute = asset_prim.CreateAttribute(
                    "gist:assetUri", Sdf.ValueTypeNames.String, custom=True
                )
            if uri_attribute.Get() != asset_uri:
                asset_prim.GetReferences().ClearReferences()
                asset_prim.GetReferences().AddReference(asset_uri)
                uri_attribute.Set(asset_uri)
            # Deliberately not instanceable: changing prototype sets during long
            # live sessions caused unstable renderer instances in prior builds.
            asset_prim.SetInstanceable(False)
        else:
            if stage.GetPrimAtPath(asset_path).IsValid():
                stage.RemovePrim(asset_path)
            proxy = UsdGeom.Cube.Define(stage, proxy_path)
            proxy.GetSizeAttr().Set(0.08)
            proxy.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.50, 0.10)])
            proxy_translate = proxy.GetPrim().GetAttribute("xformOp:translate")
            if not proxy_translate:
                proxy_translate = UsdGeom.Xformable(proxy.GetPrim()).AddTranslateOp().GetAttr()
            proxy_translate.Set(Gf.Vec3d(0.0, 0.0, 0.04))

        self._records[raw_id] = ObjectRecord(
            object_id=raw_id,
            label=label,
            prim_path=str(object_path),
            state=state,
            asset_uri=asset_uri,
            asset_key=str(scene_object.get("asset_key") or ""),
            uses_catalog_fallback=uses_fallback,
            xyz_m=(float(point[0]), float(point[1]), float(point[2])),
            position_mode=str(scene_object.get("position_mode") or ""),
            source_camera_count=len(scene_object.get("source_cameras", [])),
            display_name=display_label(scene_object, ""),
            tracked_yaw_deg=tracked_yaw_deg(scene_object),
            last_seen_monotonic=now,
            last_seen_timestamp_ns=timestamp_ns,
            tracked_xyz_m=tuple(float(value) for value in xyz),
            table_xyz_m=table_xyz,
            agent_offset_m=agent_offset,
            rotation_wxyz=tuple(transform.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])),
            scale=tuple(transform.get("scale", [1.0, 1.0, 1.0])),
            last_placement=self._records[raw_id].last_placement if raw_id in self._records else "preserve_position",
        )

    def _enforce_object_limit(self, stage) -> None:
        overflow = len(self._records) - self._max_tracked_objects
        if overflow <= 0:
            return
        oldest = sorted(
            (record for record in self._records.values() if not record.object_id.startswith('asset_')),
            key=lambda record: record.last_seen_monotonic
        )[:overflow]
        for record in oldest:
            self._remove_record(stage, record.object_id)

    def _remove_stale_objects(self, stage, now: float) -> None:
        stale_ids = [
            object_id
            for object_id, record in self._records.items()
            if not object_id.startswith('asset_') and now - record.last_seen_monotonic >= self._stale_remove_seconds
        ]
        for object_id in stale_ids:
            self._remove_record(stage, object_id)
        if stale_ids:
            self._status = (
                f"Removed {len(stale_ids)} stale object(s) | managed {len(self._records)}"
            )
            self._ui_dirty = True
            self._state_dirty = True

    def _remove_record(self, stage, object_id: str) -> None:
        self._agent_offsets.pop(object_id, None)
        self._agent_transforms.pop(object_id, None)
        if any(item.get("object_id") == object_id for item in self._agent_preview.get("placements", [])):
            self._clear_agent_preview(stage)
        record = self._records.pop(object_id, None)
        if record is None:
            return
        path = Sdf.Path(record.prim_path)
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
        parent = path.GetParentPath()
        while parent != OBJECTS_ROOT and parent.HasPrefix(OBJECTS_ROOT):
            prim = stage.GetPrimAtPath(parent)
            if not prim.IsValid() or prim.GetChildren():
                break
            next_parent = parent.GetParentPath()
            stage.RemovePrim(parent)
            parent = next_parent

    def _select_object(self, prim_path: str) -> None:
        omni.usd.get_context().get_selection().set_selected_prim_paths(
            [prim_path], True
        )
        self._update_focus_selection(omni.usd.get_context().get_stage())

    def _clear_object_focus(self) -> None:
        omni.usd.get_context().get_selection().set_selected_prim_paths([], True)
        self._update_focus_selection(omni.usd.get_context().get_stage())

    def _update_focus_selection(self, stage) -> None:
        """Stage and Viewport share USD selection; map mesh hits to managed IDs."""
        paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        record_paths = {oid: record.prim_path for oid, record in self._records.items()}
        signature = (tuple(paths), tuple(record_paths.items()), self._agent_preview.get("preview_id"))
        if signature == self._focus_signature:
            return
        self._focus_signature = signature
        for oid, record in self._records.items():
            if oid not in self._object_short_names:
                label = record.display_name or {"potted plant": "Plant"}.get(record.label, record.label.title())
                self._object_name_counts[label] = self._object_name_counts.get(label, 0) + 1
                self._object_short_names[oid] = f"{label} {self._object_name_counts[label]:02d}"
        focused = resolve_focus_ids(paths, record_paths)
        # Ghost previews retain their source object's identity, never their displaced image position.
        for path in paths:
            if str(path) == str(AGENT_PREVIEW_ROOT):
                focused.extend(p["object_id"] for p in self._agent_preview.get("placements", []) if p.get("object_id") in self._records)
            elif stage and str(path).startswith(str(AGENT_PREVIEW_ROOT) + "/"):
                prim = stage.GetPrimAtPath(path)
                while prim and prim.GetPath().HasPrefix(AGENT_PREVIEW_ROOT):
                    attr = prim.GetAttribute("gist:sourceObjectId")
                    oid = str(attr.Get()) if attr else ""
                    if oid in self._records:
                        focused.append(oid)
                        break
                    prim = prim.GetParent()
        self._focused_object_ids = list(dict.fromkeys(focused))
        self._camera_focus_dirty = True
        self._update_camera_preview()
        self._ui_dirty = True
        self._state_dirty = True

    def _selection_evidence(self) -> dict[str, Any]:
        return {"object_ids": list(self._focused_object_ids),
                "object_names": {oid: self._object_short_names[oid] for oid in self._records if oid in self._object_short_names},
                "source": "USD selection from Stage, viewport or agent; not physical movement"}

    def _frame_demo_context(self) -> bool:
        """Fit the physical table tightly in the active Viewport."""
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        prim_paths = ["/World/TabletopDemo/Table"]
        if not all(stage.GetPrimAtPath(path).IsValid() for path in prim_paths):
            return False
        try:
            from omni.kit.viewport.utility import (
                frame_viewport_prims,
                get_active_viewport,
            )

            viewport = get_active_viewport()
            if viewport is None:
                return False
            viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
            if self._venue_camera is not None:
                from omni.kit.viewport.utility.camera_state import ViewportCameraState
                eye, target = self._venue_camera
                state = ViewportCameraState("/OmniverseKit_Persp", viewport)
                state.set_position_world(Gf.Vec3d(*eye), True)
                state.set_target_world(Gf.Vec3d(*target), True)
                return True
            return bool(frame_viewport_prims(viewport, prims=prim_paths))
        except (ImportError, RuntimeError, AttributeError) as exc:
            carb.log_warn(f"[GIST Tabletop Twin] Could not frame demo context: {exc}")
            return False

    def _queue_reconstruction(self, object_id: str) -> None:
        record = self._records.get(object_id)
        if record is None:
            return
        self._jobs[object_id] = {
            "object_id": object_id,
            "label": record.label,
            "prim_path": record.prim_path,
            "status": "awaiting_approval",
            "requested_at_ns": time.time_ns(),
            "provider": None,
        }
        self._ui_dirty = True
        self._state_dirty = True

    def _set_job_status(self, object_id: str, status: str) -> None:
        job = self._jobs.get(object_id)
        if job is None:
            return
        if status == 'approved_waiting_provider':
            status = 'select_generation_inputs'
            self._open_asset_studio()
        job["status"] = status
        job["updated_at_ns"] = time.time_ns()
        self._ui_dirty = True
        self._state_dirty = True

    def _set_bottom_tab(self, tab_name: str) -> None:
        window = {"settings": self._settings_window, "agent": self._window}.get(tab_name)
        if window is not None:
            window.visible = True
            window.focus()

    def _clear_tracked_objects(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        removed = len(self._records)
        for object_id in list(self._records):
            self._remove_record(stage, object_id)
        self._status = f"Cleared {removed} tracked object(s); waiting for new data"
        self._ui_dirty = True
        self._state_dirty = True

    def _on_agent_input_begin(self, _model) -> None:
        self._agent_input_editing = True

    def _on_agent_input_end(self, _model) -> None:
        self._agent_input_editing = False

    def _on_agent_key_pressed(self, key: int, modifiers: int, pressed: bool) -> None:
        if not pressed or not self._agent_input_editing:
            return
        enter_keys = {
            int(carb.input.KeyboardInput.ENTER),
            int(carb.input.KeyboardInput.NUMPAD_ENTER),
        }
        blocked_modifiers = (
            carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT
            | carb.input.KEYBOARD_MODIFIER_FLAG_CONTROL
            | carb.input.KEYBOARD_MODIFIER_FLAG_ALT
        )
        if int(key) in enter_keys and not (int(modifiers) & blocked_modifiers):
            self._submit_agent_prompt()

    def _submit_agent_prompt(self, prompt: str | None = None) -> None:
        if prompt is None and self._agent_input_model is not None:
            prompt = self._agent_input_model.get_value_as_string()
        prompt = str(prompt or "").strip()
        if not prompt:
            return
        if self._agent_busy:
            self._agent_messages.append(
                {
                    "role": "Agent",
                    "text": "The previous request is still running on the fixed snapshot.",
                }
            )
            self._ui_dirty = True
            return
        if self._simulation.active:
            self._simulation.command("pause")
        conversation = list(self._agent_messages)[-10:]
        self._agent_messages.append({"role": "You", "text": prompt})
        self._agent_input_editing = False
        if self._agent_input_model is not None:
            self._agent_input_model.set_value("")
        fallback_snapshot = self._fallback_agent_snapshot()
        interaction_frame = self._capture_interaction_frame()
        self._last_interaction_frame = interaction_frame
        if self._agent_mode == "snapshot" and self._agent_snapshot_path:
            request_id = self._agent_snapshot_id
            snapshot_path = self._agent_snapshot_path
            wait_for_snapshot = False
            status_text = f"고정 장면 {request_id[-12:]}을 분석하는 중..."
        else:
            request_id = self._request_agent_snapshot()
            snapshot_path = ""
            wait_for_snapshot = True
            status_text = "다음 동기화 카메라 장면을 고정하는 중..."
        self._agent_busy = True
        self._agent_query_id = f"query-{time.time_ns()}"
        self._agent_evidence_status = "정보원: 고정 스냅샷 준비 중"
        self._agent_evidence_trace = []
        self._agent_messages.append({"role": "System", "text": status_text})
        self._agent_thread = threading.Thread(
            target=self._agent_request_worker,
            args=(
                prompt,
                request_id,
                snapshot_path,
                wait_for_snapshot,
                fallback_snapshot,
                conversation,
                interaction_frame,
            ),
            name="gist-tabletop-scene-agent",
            daemon=True,
        )
        self._agent_thread.start()
        self._ui_dirty = True
        self._state_dirty = True

    def _capture_interaction_frame(self) -> dict[str, Any]:
        """Capture screen directions as TableFrame XY vectors on the UI thread."""
        result = _capture_viewport_interaction_frame(
            None,
            None,
            self._message_translation,
            self._offset,
            self._message_yaw_rad,
        )
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return result
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            result = _capture_viewport_interaction_frame(
                stage,
                viewport,
                self._message_translation,
                self._offset,
                self._message_yaw_rad,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            carb.log_warn(
                f"[GIST Tabletop Twin] Could not capture Viewport direction frame: {exc}"
            )
        return result

    def _fallback_agent_snapshot(self) -> dict[str, Any]:
        if self._latest_tracking_message is not None:
            tracking = json.loads(json.dumps(self._latest_tracking_message))
        else:
            tracking = {
                "schema": "gist.smartphone_mocap.v1",
                "timestamp_ns": time.time_ns(),
                "objects": [
                    {
                        "id": record.object_id,
                        "label": record.label,
                        "state": record.state,
                        "xyz": list(record.tracked_xyz_m),
                        "asset_uri": record.asset_uri,
                        "source_cameras": [None] * record.source_camera_count,
                    }
                    for record in self._records.values()
                ],
            }
        return {
            "schema": "gist.tabletop_agent.snapshot.v1",
            "snapshot_id": self._agent_request_id or f"extension-{time.time_ns()}",
            "captured_at_ns": time.time_ns(),
            "source": "extension_fallback",
            "frame_number": int(tracking.get("frame_number", 0) or 0),
            "tracking": tracking,
            "detections": {},
            "camera_files": {},
            # Never attach a mutable live mosaic to an immutable fallback snapshot.
            "camera_mosaic": None,
            "twin_state": self._capture_twin_edits(),
            "selection": self._selection_evidence(),
        }

    def _capture_twin_edits(self) -> dict[str, Any]:
        stage = omni.usd.get_context().get_stage()
        runtime = {}
        physics_paths = ({b["object_id"]: p for b, p in
                          zip(self._simulation.scenario["bodies"], self._simulation.body_paths)}
                         if self._simulation.active else {})
        active_properties = ({b["object_id"]: b for b in self._simulation.scenario["bodies"]}
                             if self._simulation.active else {})
        for record in self._records.values():
            prim = stage.GetPrimAtPath(record.prim_path) if stage else None
            if not prim:
                continue
            cached = active_properties.get(record.object_id, self._physics_properties.get(record.object_id))
            if self._simulation.active:
                geometry = cached.get("source_contact_geometry", {}) if cached else {}
            else:
                geometry = read_contact_geometry(stage, prim)
            pose = self._record_pose(record)
            if record.object_id in physics_paths:
                prim = stage.GetPrimAtPath(physics_paths[record.object_id])
            # Read the actual USD origin, including manual Stage edits and parent transforms.
            world_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world = world_matrix.ExtractTranslation()
            if record.object_id in physics_paths:
                q = world_matrix.ExtractRotationQuat()
                pose["rotation_wxyz"] = [float(q.GetReal()), *map(float, q.GetImaginary())]
            dx = float(world[0]) - self._message_translation[0] - self._offset[0]
            dy = float(world[1]) - self._message_translation[1] - self._offset[1]
            cosine, sine = math.cos(self._message_yaw_rad), math.sin(self._message_yaw_rad)
            pose["xyz"] = [cosine * dx + sine * dy, -sine * dx + cosine * dy,
                           float(world[2]) - self._message_translation[2] - self._offset[2]]
            runtime[record.object_id] = {**pose, "label": record.label, "asset_uri": record.asset_uri,
                "position_mode": record.position_mode, "prim_path": str(prim.GetPath()),
                "usd_world_transform": [[float(world_matrix[r][c]) for c in range(4)] for r in range(4)],
                "contact_geometry": geometry,
                "table_contact": inspect_contact(geometry, **pose), "last_placement": record.last_placement}
        state = json.loads(json.dumps({"offsets": self._agent_offsets, "preview": self._agent_preview,
                                      "transforms": self._agent_transforms, "runtime_objects": runtime}))
        state["session_id"] = self._agent_session_id
        state["simulation"] = {"active": self._simulation.active, "status": self._simulation.status}
        if self._simulation.active:
            state["simulation"].update({
                "initial_bodies": self._simulation.initial_bodies,
                "release_origins": {b["object_id"]: b.get("release_origin_xyz", b["xyz"])
                                    for b in self._simulation.scenario["bodies"]},
                "result": self._simulation.result()})
        state["physics_properties"] = self._physics_properties
        state["physics_table_material"] = self._physics_table_material
        state["scene_version"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:20]
        state["captured_at_ns"] = time.time_ns()
        state["tracking_timestamp_ns"] = (self._latest_tracking_message or {}).get("timestamp_ns")
        return state

    async def _prepare_agent_evidence(self, job):
        """Main-thread USD + viewport capture after perception has frozen.

        Capture once per question, but send pixels to the model only if it asks.
        No webcam opening, camera movement, or scene edits are performed here.
        """
        result = {}
        try:
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            await omni.kit.app.get_app().next_update_async()
            result = self._fallback_agent_snapshot()
            result["interaction_frame"] = self._capture_interaction_frame()
            view = {"available": False, "source": "twin", "scene_version": result["twin_state"]["scene_version"]}
            result["twin_view"] = view
            viewport = get_active_viewport()
            stage = omni.usd.get_context().get_stage()
            if viewport is not None and stage is not None:
                path = self._agent_root / "requests" / job["query_id"] / "viewport.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                camera_path = str(viewport.camera_path)
                camera = stage.GetPrimAtPath(camera_path)
                before_camera = str(UsdGeom.Camera(camera).GetCamera(Usd.TimeCode.Default()))
                # Project known USD origins into the captured view for ID grounding.
                frustum = UsdGeom.Camera(camera).GetCamera(Usd.TimeCode.Default()).frustum
                matrix = frustum.ComputeViewMatrix() * frustum.ComputeProjectionMatrix()
                projected = []
                for oid, obj in result["twin_state"]["runtime_objects"].items():
                    prim = stage.GetPrimAtPath(obj["prim_path"])
                    origin = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                    clip = Gf.Vec4d(*origin, 1.0) * matrix
                    if clip[3] > 0:
                        uv = [float(clip[0] / clip[3] + 1) / 2, float(1 - clip[1] / clip[3]) / 2]
                        projected.append({"id": oid, "uv_top_left": uv,
                                          "in_frame": all(0 <= c <= 1 for c in uv), "visibility": "not tested for occlusion"})
                view.update(camera_path=camera_path, path=str(path), captured_at_ns=time.time_ns(),
                            projected_object_origins=projected, interaction_frame=result["interaction_frame"])
                capture = capture_viewport_to_file(viewport, file_path=str(path))
                await asyncio.wait_for(capture.wait_for_result(), timeout=8.0)
                # Kit's capture future can complete before its asynchronous PNG writer.
                deadline = time.monotonic() + 3.0
                previous_size = -1
                while time.monotonic() < deadline:
                    await omni.kit.app.get_app().next_update_async()
                    size = path.stat().st_size if path.is_file() else 0
                    if size > 0 and size == previous_size and cv2.imread(str(path)) is not None:
                        break
                    previous_size = size
                after = self._capture_twin_edits()
                unchanged = (after["scene_version"] == view["scene_version"] and str(viewport.camera_path) == camera_path
                             and str(UsdGeom.Camera(camera).GetCamera(Usd.TimeCode.Default())) == before_camera)
                view["available"] = path.is_file() and path.stat().st_size > 0 and unchanged
                view["capture_completed_at_ns"] = time.time_ns()
                if not unchanged:
                    view["error"] = "Scene or camera changed during capture; image cannot verify this state"
        except Exception as exc:
            result["twin_view"] = {"available": False, "error": str(exc), "source": "twin"}
            carb.log_warn(f"[GIST Tabletop Twin] Evidence capture unavailable: {exc}")
        finally:
            job["result"] = result
            job["done"].set()

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _write_agent_control(self, desired_mode: str, request_id: str) -> None:
        try:
            self._atomic_write_json(
                self._agent_root / "control.json",
                {
                    "schema": "gist.tabletop_agent.control.v1",
                    "desired_mode": desired_mode,
                    "request_id": request_id,
                    "requested_at_ns": time.time_ns(),
                },
            )
        except OSError as exc:
            carb.log_warn(f"[GIST Tabletop Twin] Agent control write failed: {exc}")

    def _request_agent_snapshot(self) -> str:
        request_id = f"snapshot-{time.time_ns()}"
        self._agent_request_id = request_id
        self._agent_snapshot_id = ""
        self._agent_snapshot_path = ""
        self._agent_mode = "snapshot_pending"
        self._write_agent_control("snapshot", request_id)
        self._ui_dirty = True
        self._state_dirty = True
        return request_id

    def _refresh_agent_snapshot(self) -> None:
        request_id = self._request_agent_snapshot()
        self._agent_messages.append(
            {
                "role": "System",
                "text": f"장면 정보를 갱신하는 중 ({request_id[-12:]}).",
            }
        )
        self._ui_dirty = True

    def _clear_agent_history(self) -> None:
        """Start a clean conversational turn without restarting Isaac Sim.

        The physical tracker and USD scene are intentionally preserved. A new
        snapshot can be requested separately with ``장면 갱신``.
        """
        if self._agent_busy:
            self._agent_messages.append(
                {"role": "System", "text": "현재 요청이 끝난 뒤 대화를 초기화해 주세요."}
            )
            self._ui_dirty = True
            return
        self._agent_conversation_id = f"conversation-{time.time_ns()}"
        self._agent_messages.clear()
        self._agent_messages.append(
            {"role": "Agent", "text": "새 대화를 시작했습니다. 이전 대화는 자동 입력에서 제외됩니다. 장면·물성·실험 상태는 유지됩니다. 실험을 처음부터 돌리려면 “다시 실행해줘”, 실시간 장면으로 돌아가려면 “관측으로 복귀해줘”라고 요청하세요."}
        )
        self._agent_history_signature = None
        self._agent_evidence_trace = []
        self._agent_evidence_status = "정보원: 새 대화 · 장면 갱신 필요 시 버튼 사용"
        self._ui_dirty = True
        self._state_dirty = True

    def _resume_agent_live(self) -> None:
        request_id = f"live-{time.time_ns()}"
        self._agent_request_id = request_id
        self._write_agent_control("live", request_id)
        self._agent_mode = "live"
        self._agent_snapshot_id = ""
        self._agent_snapshot_path = ""
        resumed_at = time.monotonic()
        for record in self._records.values():
            record.last_seen_monotonic = resumed_at
        self._agent_messages.append(
            {"role": "System", "text": "실시간 모드가 재개되어 인식 결과를 갱신합니다."}
        )
        self._ui_dirty = True
        self._state_dirty = True

    def _agent_request_worker(
        self,
        prompt: str,
        request_id: str,
        snapshot_path: str,
        wait_for_snapshot: bool,
        fallback_snapshot: dict[str, Any],
        conversation: list[dict[str, str]],
        interaction_frame: dict[str, Any],
    ) -> None:
        snapshot = None
        selected_path = snapshot_path
        if wait_for_snapshot:
            deadline = time.monotonic() + self._agent_snapshot_wait_seconds
            while time.monotonic() < deadline and not self._stop.is_set():
                runtime = self._read_json_file(self._agent_root / "runtime_state.json")
                if (
                    runtime
                    and runtime.get("mode") == "snapshot"
                    and str(runtime.get("request_id", "")) == request_id
                ):
                    selected_path = str(runtime.get("snapshot_path") or "")
                    if selected_path:
                        snapshot = self._read_json_file(Path(selected_path))
                    if snapshot is not None:
                        break
                time.sleep(0.05)
        elif selected_path:
            snapshot = self._read_json_file(Path(selected_path))
        if snapshot is None:
            snapshot = fallback_snapshot
            snapshot["snapshot_id"] = request_id
        else:
            snapshot = dict(snapshot)
        job = {"query_id": self._agent_query_id, "done": threading.Event()}
        self._agent_capture_jobs.append(job)
        job["done"].wait(12.0)
        captured = job.get("result", {})
        snapshot["interaction_frame"] = captured.get("interaction_frame", interaction_frame)
        snapshot["twin_state"] = captured.get("twin_state", fallback_snapshot.get("twin_state", {}))
        snapshot["twin_view"] = captured.get("twin_view", {"available": False, "error": "Capture timed out"})
        snapshot["selection"] = captured.get("selection", fallback_snapshot.get("selection", {}))
        self._last_interaction_frame = snapshot["interaction_frame"]
        # Persist exactly what this question can inspect; physical and virtual capture times remain separate.
        request_dir = self._agent_root / "requests" / job["query_id"]
        self._atomic_write_json(request_dir / "evidence.json", snapshot)

        request_payload = {
            "prompt": prompt,
            "snapshot": snapshot,
            "snapshot_path": selected_path or None,
            "conversation": conversation,
            "request_id": job["query_id"],
            "session_id": self._agent_session_id,
            "conversation_id": self._agent_conversation_id,
        }
        try:
            request = urllib.request.Request(
                self._agent_url,
                data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                request, timeout=self._agent_timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not result.get("ok"):
                raise ValueError(str(result.get("error", "Agent request failed")))
            self._agent_results.append(
                {
                    "reply": str(result.get("reply", "No response text.")),
                    "actions": result.get("actions", []),
                    "provider": str(result.get("provider", "agent service")),
                    "provider_error": str(result.get("provider_error", "")),
                    "evidence_trace": result.get("evidence_trace", []),
                    "snapshot_id": str(snapshot.get("snapshot_id", request_id)),
                    "snapshot_path": selected_path,
                    "resume": self._agent_auto_resume,
                }
            )
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            self._agent_results.append(
                {
                    "reply": f"Agent service unavailable: {exc}",
                    "actions": [],
                    "provider": "unavailable",
                    "snapshot_id": str(snapshot.get("snapshot_id", request_id)),
                    "snapshot_path": selected_path,
                    "resume": False,
                }
            )

    def _poll_agent_runtime(self) -> None:
        runtime_path = self._agent_root / "runtime_state.json"
        try:
            modified_ns = runtime_path.stat().st_mtime_ns
        except OSError:
            return
        if modified_ns == self._last_agent_runtime_mtime_ns:
            return
        self._last_agent_runtime_mtime_ns = modified_ns
        runtime = self._read_json_file(runtime_path)
        if runtime is None:
            return
        request_id = str(runtime.get("request_id", ""))
        if self._agent_request_id and request_id != self._agent_request_id:
            return
        if runtime.get("mode") == "snapshot":
            self._agent_mode = "snapshot"
            self._agent_snapshot_id = request_id
            self._agent_snapshot_path = str(runtime.get("snapshot_path") or "")
        elif runtime.get("mode") == "live" and self._agent_mode != "snapshot_pending":
            self._agent_mode = "live"
        self._ui_dirty = True
        self._state_dirty = True

    def _drain_agent_results(self, stage) -> None:
        task = getattr(self, "_agent_apply_task", None)
        if task is not None and not task.done():
            return
        if not self._agent_results:
            return
        self._agent_apply_task = asyncio.ensure_future(self._apply_agent_results(stage))

    async def _apply_agent_results(self, stage):
        while self._agent_results:
            result = self._agent_results.popleft()
            self._agent_busy = True
            self._agent_provider = result["provider"]
            self._agent_evidence_trace = result.get("evidence_trace", [])
            labels = []
            for entry in self._agent_evidence_trace:
                if entry.get("ok", False):
                    label = {"conversation archive":"대화 기록", "user statement":"사용자 정보", "backend execution":"실행 기록"}.get(entry.get("source"))
                    label = label or (("실물" if entry.get("source") == "physical" else "트윈") + (" 이미지" if entry.get("tool") == "inspect_images" else " 상태"))
                    if label not in labels:
                        labels.append(label)
            self._agent_evidence_status = "확인한 정보원: " + (" · ".join(labels) if labels else "추가 조회 없음")
            self._agent_snapshot_id = result["snapshot_id"]
            if result.get("snapshot_path"):
                self._agent_snapshot_path = result["snapshot_path"]
            if self._agent_mode != "live":
                self._agent_mode = "snapshot"
            failure = None
            applied = []
            for action in result.get("actions", []):
                if isinstance(action, dict):
                    action_id = action.get("action_id")
                    if action_id and action_id in self._applied_action_ids:
                        continue
                    try:
                        # Timeline commands are deferred until Kit's next update.
                        # Stop before rebuilding bodies and let PhysX release them.
                        if action.get("type") == "physics_preview" or (
                                action.get("type") == "physics_control" and action.get("command") == "reset"):
                            self._simulation.timeline.stop()
                            await omni.kit.app.get_app().next_update_async()
                            await omni.kit.app.get_app().next_update_async()
                        self._apply_agent_action(stage, action)
                        await omni.kit.app.get_app().next_update_async()
                        await omni.kit.app.get_app().next_update_async()
                        if action.get("type") == "physics_control" and action.get("command") == "play":
                            if not self._simulation.timeline.is_playing():
                                raise RuntimeError("PhysX timeline did not start")
                        applied.append(action.get("command", action.get("type", "action")))
                        if action_id:
                            self._applied_action_ids.add(action_id)
                            self._ack_agent_action(action_id, True, "Applied to Isaac stage")
                    except Exception as exc:
                        failure = str(exc)
                        self._agent_messages.append({"role": "System", "text": f"장면 적용 실패: {exc}"})
                        if action_id:
                            self._applied_action_ids.add(action_id)
                            self._ack_agent_action(action_id, False, str(exc))
                        self._applied_action_ids.update(
                            pending["action_id"] for pending in result.get("actions", [])
                            if isinstance(pending, dict) and pending.get("action_id"))
                        break  # Do not play after a failed preparation.
            if failure:
                reply = "요청한 작업을 완료하지 못했습니다. 장면 적용 오류: " + failure
            elif applied:
                labels = {"physics_preview": "실험 준비", "play": "재생", "pause": "일시정지",
                          "reset": "초기 상태 복원", "return": "관측 복귀"}
                reply = "Isaac 적용 확인: " + " → ".join(labels.get(a, a) for a in applied)
                if not any(a in labels for a in applied):
                    reply = result["reply"] + "\n" + reply
            else:
                reply = result["reply"]
            self._agent_messages.append({"role": "Agent", "text": reply})
            self._agent_busy = False
            if result.get("provider_error"):
                carb.log_warn(
                    "[GIST Tabletop Twin] Agent backend reported an error: "
                    + result["provider_error"]
                )
            if result.get("resume"):
                self._resume_agent_live()
        self._ui_dirty = True
        self._state_dirty = True

    def _ack_agent_action(self, action_id, ok, detail):
        payload = {"session_id":self._agent_session_id,"action_id":action_id,"ok":ok,"detail":detail}
        url = self._agent_url.rsplit("/", 1)[0] + "/ack"
        def send():
            try:
                request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                                 headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
            except (OSError, urllib.error.URLError) as exc:
                carb.log_warn(f"[GIST Tabletop Twin] Action acknowledgement pending: {exc}")
        threading.Thread(target=send, daemon=True).start()

    def _apply_agent_action(self, stage, action: dict[str, Any]) -> None:
        action_type = str(action.get("type", ""))
        if action_type == "physics_preview":
            if abs(self._message_yaw_rad) > 1e-6:
                raise ValueError("Physics viewport currently requires an unrotated table coordinate frame")
            offset = [self._offset[i] + self._message_translation[i] for i in range(3)]
            self._simulation.replace(stage, action["scenario"], offset)
            if action["scenario"].get("properties_scope") != "experiment":
                for body in action["scenario"]["bodies"]:
                    self._physics_properties[body["object_id"]] = json.loads(json.dumps(body))
                self._physics_table_material = dict(action["scenario"]["table_material"])
            self._observation_paused = True
            self._agent_messages.append({"role": "System", "text": self._simulation.status})
            if self._simulation.preflight_warnings:
                self._agent_messages.append({"role": "System", "text":
                    "물리 실험 초기 배치 보정 (단위 mm):\n" +
                    "\n".join(self._simulation.preflight_warnings)})
            self._ui_dirty = True
            return
        if action_type == "physics_control":
            self._physics_command(action["command"], propagate_error=True)
            return
        if self._simulation.active:
            raise ValueError("Return to observation before editing mirrored objects")
        if action_type == "select_object":
            record = self._records.get(str(action.get("object_id", "")))
            if record is not None:
                self._select_object(record.prim_path)
        elif action_type == "queue_reconstruction":
            self._queue_reconstruction(str(action.get("object_id", "")))
        elif action_type == "preview_layout":
            self._create_agent_preview(stage, action.get("placements", []), action.get("preview_id"))
        elif action_type == "apply_preview":
            self._commit_agent_preview(stage, str(action.get("preview_id", "")))
        elif action_type in {"move_objects", "transform_objects"}:
            self._move_agent_objects(stage, action.get("placements", []))
        elif action_type == "reset_transforms":
            self._reset_agent_transforms(stage, action.get("object_ids", []), action.get("components", []))
        elif action_type == "reset_moves":
            self._reset_agent_moves(stage, action.get("object_ids", []))
        elif action_type == "cancel_preview":
            self._clear_agent_preview(stage)

    def _clear_agent_preview(self, stage=None) -> None:
        self._agent_preview = {}
        stage = stage or omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath(AGENT_PREVIEW_ROOT).IsValid():
            stage.RemovePrim(AGENT_PREVIEW_ROOT)
        self._state_dirty = True

    def _validated_agent_placements(self, stage, placements: Any) -> list[dict[str, Any]]:
        if not isinstance(placements, list) or not placements:
            raise ValueError("이동할 객체와 좌표가 없습니다.")
        validated = []
        seen = set()
        for item in placements:
            if not isinstance(item, dict):
                raise ValueError("잘못된 이동 데이터입니다.")
            object_id = str(item.get("object_id", ""))
            record = self._records.get(object_id)
            if record is None or not stage.GetPrimAtPath(record.prim_path).IsValid():
                raise ValueError(f"객체가 현재 장면에 없습니다: {object_id}")
            if object_id in seen:
                raise ValueError(f"중복된 객체입니다: {object_id}")
            target = {"object_id": object_id}
            for field, length in (("xyz", 3), ("rotation_wxyz", 4), ("scale", 3)):
                if field not in item:
                    continue
                values = item[field]
                if not isinstance(values, (list, tuple)) or len(values) != length:
                    raise ValueError(f"{field}: {length}개 값이 필요합니다.")
                values = [float(value) for value in values]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{field}: 값이 유한한 수여야 합니다.")
                if field == "xyz":
                    if not (0.0 <= values[0] <= TABLE_SIZE_M[0] and 0.0 <= values[1] <= TABLE_SIZE_M[1]):
                        raise ValueError("이동 대상은 테이블 범위 안에 있어야 합니다.")
                    # An arbitrary USD pivot can be below Z=0 while all geometry
                    # is above the table. Contact checks use geometry, not pivot Z.
                elif field == "rotation_wxyz":
                    norm = math.hypot(*values)
                    if not math.isfinite(norm) or norm < 1e-8:
                        raise ValueError("회전 quaternion의 길이가 0이거나 유효하지 않습니다.")
                    values = [v/norm for v in values]
                elif any(v <= 0 for v in values):
                    raise ValueError("크기 배율은 0보다 커야 합니다.")
                target[field] = values
            placement = item.get("placement", "preserve_position")
            if placement not in PLACEMENT_MODES:
                raise ValueError("알 수 없는 접촉 처리 방식입니다.")
            if len(target) == 1 and placement == "preserve_position":
                raise ValueError("이동, 회전 또는 크기 변환 값이 필요합니다.")
            geometry = read_contact_geometry(stage, stage.GetPrimAtPath(record.prim_path))
            resolution = resolve_placement(geometry, self._record_pose(record), target, placement)
            if placement != "preserve_position":
                target["xyz"] = resolution["target"]["xyz"]
            target.update(placement=placement, contact=resolution["contact_after"],
                          z_correction_m=resolution["z_correction_m"])
            seen.add(object_id)
            validated.append(target)
        return validated

    @staticmethod
    def _record_pose(record):
        return {"xyz": list(record.table_xyz_m), "rotation_wxyz": list(record.rotation_wxyz),
                "scale": list(record.scale)}

    def _report_contact(self, item, preview=False):
        contact = item["contact"]
        xyz = item.get("xyz", self._records[item["object_id"]].table_xyz_m)
        text = ("미리보기" if preview else "변환 적용") + f" [{item['object_id']}]: {item['placement']}; "
        text += f"최종 XYZ=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})m. "
        if contact.get("available"):
            text += f"하단 간격 {contact['clearance_m']:.3f}m. "
            if "possible_table_penetration" in contact["warnings"]:
                text += f"주의: 테이블 관통 가능성 {contact['possible_penetration_m']:.3f}m (보수적 경계 검사). 자동 보정하지 않았습니다. "
            if "footprint_outside_table" in contact["warnings"]:
                text += "주의: 객체 경계가 테이블 밖으로 나갑니다. "
        else:
            text += "경계 정보가 없어 접촉 여부를 검사하지 못했습니다. "
        self._agent_messages.append({"role": "System", "text": text})

    def _write_shape_transform(self, prim, transform, tracked_yaw_deg=0.0) -> None:
        """Author T * R(table yaw + tracked yaw) * R(edit) * S about the asset's existing origin."""
        xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        yaw = ops.get("xformOp:rotateZ:tableFrame")
        if yaw is None:
            yaw = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, "tableFrame")
        heading = shown_yaw_deg(tracked_yaw_deg, transform.get("rotation_wxyz", (1, 0, 0, 0)))
        yaw.Set(math.degrees(self._message_yaw_rad) + heading)
        orientation = ops.get("xformOp:orient:agent")
        if orientation is None:
            orientation = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "agent")
        w, x, y, z = transform.get("rotation_wxyz", (1, 0, 0, 0))
        orientation.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        scale = ops.get("xformOp:scale:agent")
        if scale is None:
            scale = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "agent")
        scale.Set(Gf.Vec3d(*transform.get("scale", (1, 1, 1))))
        translate = ops.get("xformOp:translate")
        xform.SetXformOpOrder(([translate] if translate is not None else []) + [yaw, orientation, scale])

    def _set_agent_offset(self, stage, record, offset) -> None:
        offset = tuple(float(value) for value in offset)
        if any(abs(value) > 1e-9 for value in offset):
            self._agent_offsets[record.object_id] = offset
        else:
            self._agent_offsets.pop(record.object_id, None)
        record.agent_offset_m = offset
        record.table_xyz_m = tuple(record.tracked_xyz_m[i] + offset[i] for i in range(3))
        point = self._with_offset(record.table_xyz_m)
        record.xyz_m = tuple(point)
        prim = stage.GetPrimAtPath(record.prim_path)
        prim.GetAttribute("xformOp:translate").Set(point)
        prim.CreateAttribute("gist:agentOffsetM", Sdf.ValueTypeNames.Double3, custom=True).Set(Gf.Vec3d(*offset))
        self._state_dirty = True
        self._ui_dirty = True

    def _move_agent_objects(self, stage, placements: Any) -> None:
        placements = self._validated_agent_placements(stage, placements)
        for item in placements:
            record = self._records[item["object_id"]]
            record.last_placement = item["placement"]
            if "xyz" in item:
                offset = tuple(item["xyz"][i] - record.tracked_xyz_m[i] for i in range(3))
                self._set_agent_offset(stage, record, offset)
            if "rotation_wxyz" in item or "scale" in item:
                transform = self._agent_transforms.setdefault(record.object_id, {})
                for field in ("rotation_wxyz", "scale"):
                    if field in item:
                        transform[field] = list(item[field])
                        setattr(record, field, tuple(item[field]))
                self._write_shape_transform(stage.GetPrimAtPath(record.prim_path), transform,
                                            record.tracked_yaw_deg)
            self._report_contact(item)
        self._clear_agent_preview(stage)
        self._ui_dirty = self._state_dirty = True
        self._agent_messages.append({"role": "System", "text": f"USD 객체 {len(placements)}개 변환 적용 완료."})
        if len(placements) == 1:
            record = self._records.get(placements[0]["object_id"])
            if record:
                self._select_object(record.prim_path)

    def _reset_agent_transforms(self, stage=None, object_ids=None, components=None) -> None:
        stage = stage or omni.usd.get_context().get_stage()
        if stage is None:
            return
        components = ["rotation", "scale"] if components is None else components
        if not components or any(v not in {"translation", "rotation", "scale"} for v in components):
            raise ValueError("초기화 항목이 잘못되었습니다.")
        object_ids = list(self._records) if object_ids is None else object_ids
        for object_id in object_ids:
            record = self._records.get(str(object_id))
            if record is None:
                continue
            if "translation" in components:
                self._set_agent_offset(stage, record, (0.0, 0.0, 0.0))
            transform = self._agent_transforms.setdefault(record.object_id, {})
            if "rotation" in components:
                transform.pop("rotation_wxyz", None)
                record.rotation_wxyz = (1.0, 0.0, 0.0, 0.0)
            if "scale" in components:
                transform.pop("scale", None)
                record.scale = (1.0, 1.0, 1.0)
            self._write_shape_transform(stage.GetPrimAtPath(record.prim_path), transform,
                                        record.tracked_yaw_deg)
            record.last_placement = "preserve_position"
            geometry = read_contact_geometry(stage, stage.GetPrimAtPath(record.prim_path))
            self._report_contact({"object_id": record.object_id, "placement": "preserve_position",
                                  "contact": inspect_contact(geometry, **self._record_pose(record))})
            if not transform:
                self._agent_transforms.pop(record.object_id, None)
        self._clear_agent_preview(stage)
        self._agent_messages.append({"role": "System", "text": "선택한 변환 항목을 초기화했습니다."})
        self._ui_dirty = self._state_dirty = True

    def _commit_agent_preview(self, stage=None, preview_id: str | None = None) -> None:
        stage = stage or omni.usd.get_context().get_stage()
        if stage is None or not self._agent_preview:
            raise ValueError("적용할 미리보기가 없습니다.")
        if preview_id is not None and preview_id != self._agent_preview.get("preview_id"):
            raise ValueError("미리보기가 변경되었거나 취소되었습니다. 현재 미리보기로 다시 요청해 주세요.")
        self._move_agent_objects(stage, self._agent_preview["placements"])

    def _reset_agent_moves(self, stage=None, object_ids=None) -> None:
        stage = stage or omni.usd.get_context().get_stage()
        if stage is None:
            return
        object_ids = list(self._agent_offsets) if object_ids is None else object_ids
        count = 0
        for object_id in object_ids:
            record = self._records.get(str(object_id))
            if record is not None:
                self._set_agent_offset(stage, record, (0.0, 0.0, 0.0))
                record.last_placement = "preserve_position"
                geometry = read_contact_geometry(stage, stage.GetPrimAtPath(record.prim_path))
                self._report_contact({"object_id": record.object_id, "placement": "preserve_position",
                                      "contact": inspect_contact(geometry, **self._record_pose(record))})
                count += 1
        self._clear_agent_preview(stage)
        self._agent_messages.append({"role": "System", "text": f"객체 {count}개를 카메라 추적 위치로 복귀했습니다."})
        self._ui_dirty = True

    def _apply_preview_button(self) -> None:
        try:
            self._commit_agent_preview()
        except (ValueError, RuntimeError) as exc:
            self._agent_messages.append({"role": "System", "text": str(exc)})
        self._ui_dirty = True

    def _create_agent_preview(self, stage, placements: Any, preview_id=None) -> None:
        placements = self._validated_agent_placements(stage, placements)
        self._clear_agent_preview(stage)
        self._agent_preview = {"preview_id": str(preview_id or time.time_ns()), "placements": placements}
        root = UsdGeom.Xform.Define(stage, AGENT_PREVIEW_ROOT).GetPrim()
        preview_marker = root.CreateAttribute(
            "gist:isAgentPreview", Sdf.ValueTypeNames.Bool, custom=True
        )
        preview_marker.Set(True)
        for index, placement in enumerate(placements if isinstance(placements, list) else []):
            if not isinstance(placement, dict):
                continue
            object_id = str(placement.get("object_id", ""))
            record = self._records.get(object_id)
            xyz = placement.get("xyz", list(record.table_xyz_m) if record else None)
            if record is None or not isinstance(xyz, list) or len(xyz) != 3:
                continue
            path = AGENT_PREVIEW_ROOT.AppendChild(
                f"Proposal_{index:02d}_{_safe_identifier(object_id)}"
            )
            proposal = UsdGeom.Xform.Define(stage, path).GetPrim()
            translate = UsdGeom.Xformable(proposal).AddTranslateOp().GetAttr()
            translate.Set(self._with_offset(xyz))
            transform = {**self._agent_transforms.get(object_id, {}), **placement}
            self._write_shape_transform(proposal, transform, record.tracked_yaw_deg)
            source_id = proposal.CreateAttribute(
                "gist:sourceObjectId", Sdf.ValueTypeNames.String, custom=True
            )
            source_id.Set(object_id)
            if record.asset_uri:
                asset = stage.DefinePrim(path.AppendChild("Asset"), "Xform")
                asset.GetReferences().AddReference(record.asset_uri)
                asset.SetInstanceable(False)
            marker = UsdGeom.Cube.Define(stage, path.AppendChild("GhostMarker"))
            marker.GetSizeAttr().Set(0.11)
            marker.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.85, 1.0)])
            marker.GetDisplayOpacityAttr().Set([0.28])
            marker_translate = UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp().GetAttr()
            marker_translate.Set(Gf.Vec3d(0.0, 0.0, 0.055))
            self._report_contact(placement, preview=True)
        self._select_object(str(AGENT_PREVIEW_ROOT))
        self._state_dirty = True

    def _write_state_snapshot(self) -> None:
        payload = {
            "simulation": {
                "active": self._simulation.active,
                "status": self._simulation.status,
                "observation_paused": self._observation_paused,
            },
            "schema": "gist.tabletop_twin.state.v1",
            "written_at_ns": time.time_ns(),
            "table": {
                "prim_path": "/World/TabletopDemo/Table",
                "size_m": list(TABLE_SIZE_M),
                "asset_uri": str(table_asset_path()),
            },
            "background_stage_path": self._background_stage_path,
            "catalog": {
                "theme": self._catalog.get("theme", ""),
                "asset_count": len(self._catalog.get("assets", [])),
                "uri": str(CATALOG_PATH),
            },
            "objects": [
                record.public_dict()
                for record in sorted(
                    self._records.values(), key=lambda item: item.object_id
                )
            ],
            "reconstruction_jobs": list(self._jobs.values()),
            "agent_messages": list(self._agent_messages),
            "selection": {**self._selection_evidence(), "camera_observations": self._focus_view_status},
            "agent_runtime": {
                "mode": self._agent_mode,
                "provider": self._agent_provider,
                "busy": self._agent_busy,
                "snapshot_id": self._agent_snapshot_id,
                "snapshot_path": self._agent_snapshot_path,
                "preview_prim_path": str(AGENT_PREVIEW_ROOT),
                "interaction_frame": self._last_interaction_frame,
                "twin_state": self._capture_twin_edits(),
                "evidence_trace": self._agent_evidence_trace,
                "query_id": self._agent_query_id,
            },
        }
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = STATE_PATH.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_path.replace(STATE_PATH)
        except OSError as exc:
            carb.log_warn(f"[GIST Tabletop Twin] Could not write scene state: {exc}")

    def _build_camera_ui(self) -> None:
        with self._camera_window.frame:
            with ui.VStack(spacing=3):
                with ui.HStack(height=50, spacing=8):
                    ui.ImageWithProvider(self._focus_thumbnail_provider, width=72, height=48,
                        fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT, identifier="tabletop_focus_thumbnail")
                    with ui.VStack(width=ui.Fraction(1), spacing=2):
                        self._focus_label = ui.Label("객체를 선택하면 실제 카메라에서도 함께 강조합니다.", height=20)
                        self._focus_buttons_frame = ui.Frame(height=25)
                    ui.Button("선택 해제", width=78, clicked_fn=self._clear_object_focus)
                with ui.HStack(spacing=8):
                    for index, kind in enumerate(("detection", "segmentation")):
                        if index:
                            ui.Rectangle(width=1, style={"background_color": 0xFF656565})
                        with ui.VStack(width=ui.Fraction(1), spacing=4):
                            self._camera_status_labels[kind] = ui.Label(self._camera_status_text, height=22)
                            self._camera_grid_widgets[kind] = ui.ImageWithProvider(self._camera_providers[kind],
                                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                                identifier=f"tabletop_{kind}_grid")

    def _refresh_focus_card(self, thumbnail=None) -> None:
        if not hasattr(self, "_focus_label"):
            return
        ids = self._focused_object_ids
        if ids:
            names = ", ".join(self._object_short_names.get(oid, oid) for oid in ids)
            count = sum(bool(s.get("observed_ids")) for s in self._focus_view_status.values())
            text = f"선택: {names} | {count}/4 뷰에서 연결된 관측 | 황금색 강조"
            if len(ids) == 1 and ids[0] in self._records:
                record = self._records[ids[0]]
                xyz = record.tracked_xyz_m
                text += f" | 실물 추정 XYZ=({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) m"
            self._focus_label.text = text
        else:
            self._focus_label.text = "객체 선택 → 카메라와 트윈 함께 강조 | 아래 이름 또는 Stage/뷰포트에서 선택"
        if thumbnail is not None and thumbnail.size:
            rgba = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGBA)
            self._focus_thumbnail_provider.set_bytes_data(bytearray(rgba.tobytes()), [rgba.shape[1], rgba.shape[0]])
        else:
            self._focus_thumbnail_provider.set_bytes_data(bytearray([24, 24, 24, 255]) * 72 * 48, [72, 48])
        signature = (tuple(self._records), tuple(ids))
        if signature != self._focus_buttons_signature:
            self._focus_buttons_signature = signature
            with self._focus_buttons_frame:
                with ui.HStack(spacing=5):
                    for oid, record in list(self._records.items())[:8]:
                        ui.Button(self._object_short_names.get(oid, record.label), width=100,
                            clicked_fn=lambda path=record.prim_path: self._select_object(path),
                            style={"background_color": 0xFF387E98 if oid in ids else 0xFF414141})
                    ui.Spacer()

    def _update_camera_preview(self) -> None:
        try:
            modified_ns = CAMERA_PREVIEW_PATH.stat().st_mtime_ns
            links_path = CAMERA_PREVIEW_PATH.with_suffix(".links.json")
            links_modified_ns = links_path.stat().st_mtime_ns if links_path.is_file() else 0
        except OSError:
            return
        signature = (modified_ns, links_modified_ns)
        if signature == self._camera_preview_file_signature and not self._camera_focus_dirty:
            return
        try:
            if signature != self._camera_preview_file_signature:
                self._camera_base_image, self._camera_link_metadata = read_linked_preview(CAMERA_PREVIEW_PATH)
                self._camera_preview_file_signature = signature
            if self._camera_base_image is None:
                return
            image, statuses, thumbnail = render_camera_focus(self._camera_base_image, self._camera_link_metadata,
                self._focused_object_ids, self._object_short_names)
            self._focus_view_status = statuses
            grids = split_camera_analysis_mosaic(image)
            for kind, grid in zip(("detection", "segmentation"), grids):
                rgba = cv2.cvtColor(grid, cv2.COLOR_BGR2RGBA)
                height, width = rgba.shape[:2]
                self._camera_providers[kind].set_bytes_data(bytearray(rgba.tobytes()), [width, height])
                name = "RAW + OBJECT DETECTION" if kind == "detection" else "INSTANCE SEGMENTATION"
                self._camera_status_labels[kind].text = f"{name} | 4 synchronized cameras · 2 x 2"
            self._refresh_focus_card(thumbnail)
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            carb.log_warn(f"[GIST Tabletop Twin] Camera preview update failed: {exc}")
            return
        self._camera_preview_mtime_ns = modified_ns
        self._camera_focus_dirty = False
        self._camera_status_text = "4 synchronized camera pairs · detection and masks side by side"

    def _rebuild_ui(self) -> None:
        if self._window is None:
            return
        if not self._agent_ui_built:
            with self._window.frame:
                self._build_agent_tab()
            self._agent_ui_built = True
        self._agent_status_label.text = f"{'답변 중' if self._agent_busy else '준비'} · {self._agent_mode.upper()}"
        self._agent_provider_label.text = f"모델: {self._agent_provider}"
        self._agent_evidence_label.text = self._agent_evidence_status
        self._agent_send_button.enabled = not self._agent_busy
        self._refresh_agent_history()
        with self._settings_window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(height=0, spacing=8):
                    self._build_settings_tab()
                    ui.Separator(height=1)
                    self._build_agent_settings()

    def _physics_command(self, command, propagate_error=False):
        try:
            if not self._simulation.active:
                if command == "pause":
                    # Pause the observation/replay gate as well as Isaac's
                    # timeline.  The latter alone has no effect on UDP input.
                    omni.timeline.get_timeline_interface().pause()
                    self._observation_paused = True
                    self._agent_messages.append({"role": "System", "text": "Observation paused"})
                    self._ui_dirty = True
                    self._state_dirty = True
                    return
                if command == "return":
                    self._observation_paused = False
                    with self._lock:
                        self._messages.clear()
                    self._resume_agent_live()
                    self._ui_dirty = True
                    self._state_dirty = True
                    return
            self._simulation.command(command)
            if command == "return":
                self._observation_paused = False
                with self._lock:
                    self._messages.clear()
                self._resume_agent_live()
            self._agent_messages.append({"role": "System", "text": self._simulation.status})
        except (ValueError, RuntimeError) as exc:
            if propagate_error:
                raise
            self._agent_messages.append({"role": "System", "text": f"Physics: {exc}"})
        self._ui_dirty = True
        self._state_dirty = True

    def _build_settings_tab(self) -> None:
        observation_status = "Observation paused" if self._observation_paused else "Observation live"
        ui.Label(f"{self._simulation.status} | {observation_status}", height=24)
        ui.Button("Prepare Physics Preview (Agent)", height=30,
                  enabled=not self._simulation.active and not self._agent_busy,
                  clicked_fn=lambda: self._submit_agent_prompt("현재 트윈의 실험 대상 물체들을 inspect_physics_inputs로 확인하고 물성과 근거를 추정해서 prepare_physics 후 preview_physics로 화면에 물리 실험을 준비해 줘. 아직 실행하지 마."))
        with ui.HStack(height=30, spacing=5):
            for label, command in [("Run", "play"), ("Pause", "pause"), ("Reset Experiment", "reset"), ("Return to Observation", "return")]:
                ui.Button(label, enabled=self._simulation.active or command in {"pause", "return"},
                          clicked_fn=lambda command=command: self._physics_command(command))
        ui.Label("새 대화: 대화 입력만 초기화 | Reset Experiment: 실험 시작 상태 | Return to Observation: 실시간 관측 복귀",
                 height=36, word_wrap=True)
        with ui.HStack(spacing=12):
            with ui.VStack(width=330, spacing=5):
                ui.Label("TABLETOP TWIN · STATUS", height=24)
                ui.Label(self._status, height=22)
                ui.Label(
                    f"UDP {self._port} | stale {self._stale_remove_seconds:.1f}s | "
                    f"max objects {self._max_tracked_objects}",
                    height=22,
                )
                ui.Label(
                    "Detector: RF-DETR-Seg-Nano | Mask: learned per-instance",
                    height=22,
                )
                ui.Label(f"Table {TABLE_SIZE_M[0]:g} x {TABLE_SIZE_M[1]:g} m | tabletop Z=0", height=22)
                with ui.HStack(height=30, spacing=5):
                    ui.Button(
                        "Select Table",
                        clicked_fn=lambda: self._select_object(
                            "/World/TabletopDemo/Table"
                        ),
                    )
                    ui.Button("Clear Objects", clicked_fn=self._clear_tracked_objects)
                ui.Button(
                    "Reset Table View",
                    height=28,
                    clicked_fn=self._frame_demo_context,
                )
            with ui.VStack(spacing=4):
                ui.Label(f"TRACKED OBJECTS ({len(self._records)})", height=24)
                records = sorted(
                    self._records.values(), key=lambda item: item.object_id
                )[:8]
                if not records:
                    ui.Label("Waiting for the first tracked object...", height=22)
                for record in records:
                    with ui.HStack(height=28, spacing=5):
                        fallback = " | fallback" if record.uses_catalog_fallback else ""
                        z_mode = (
                            "plane Z"
                            if record.position_mode == "support_plane"
                            else "estimated Z"
                        )
                        ui.Label(
                            f"{record.label} | {record.state} | "
                            f"xyz=({record.xyz_m[0]:.2f}, {record.xyz_m[1]:.2f}, "
                            f"{record.xyz_m[2]:.2f}) | {z_mode} | "
                            f"{record.source_camera_count} cams{fallback}",
                            width=0,
                        )
                        ui.Button(
                            "Select",
                            width=60,
                            clicked_fn=lambda path=record.prim_path: self._select_object(path),
                        )
                        ui.Button(
                            "Request 3D",
                            width=82,
                            clicked_fn=lambda object_id=record.object_id: self._queue_reconstruction(
                                object_id
                            ),
                        )

    def _build_agent_tab(self) -> None:
        # Build once: tracking/status/history updates must not recreate the input.
        with ui.VStack(spacing=6):
            ui.Label("장면 에이전트 대화", height=26, style={"font_size": 20})
            self._agent_status_label = ui.Label("", height=20, style={"color": 0xFF8FD8FF})
            self._agent_provider_label = ui.Label("", height=0, word_wrap=True)
            self._agent_evidence_label = ui.Label("", height=0, word_wrap=True, style={"color": 0xFF8FD8FF})
            ui.Separator(height=1)
            with ui.ZStack(height=ui.Fraction(1)):
                ui.Rectangle(style={"background_color": 0xFF20242A,
                                    "border_color": 0xFF456579, "border_width": 1, "border_radius": 6})
                self._agent_history_scroll = ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    identifier="tabletop_agent_history",
                )
            ui.Label("질문 입력 · Enter로 전송", height=20, style={"color": 0xFF8FD8FF})
            with ui.HStack(height=38, spacing=6):
                self._agent_input_field = ui.StringField(
                    model=self._agent_input_model, height=38,
                    identifier="tabletop_agent_prompt",
                    style={"background_color": 0xFF171A1F, "border_color": 0xFF58B7EA,
                           "border_width": 2, "border_radius": 5, "color": 0xFFFFFFFF, "font_size": 16})
                self._agent_send_button = ui.Button("전송", width=68,
                    clicked_fn=self._submit_agent_prompt, identifier="tabletop_agent_send",
                    style={"background_color": 0xFF1679A8, "border_radius": 5})
            with ui.HStack(height=30, spacing=6):
                ui.Button("새 대화", clicked_fn=self._clear_agent_history,
                          tooltip="이전 대화를 자동 입력에서 제외합니다. 저장 기록·물성·현재 실험은 유지됩니다.")
                ui.Button("실시간 재개", clicked_fn=self._resume_agent_live)
                ui.Button("미리보기 적용", clicked_fn=self._apply_preview_button)

    def _refresh_agent_history(self) -> None:
        signature = tuple((item.get("role", ""), item.get("text", "")) for item in self._agent_messages)
        if signature == self._agent_history_signature:
            return
        scroll = self._agent_history_scroll
        at_bottom = self._agent_history_signature is None or scroll.scroll_y_max - scroll.scroll_y <= 24
        new_user_message = bool(signature and signature[-1][0] == "You")
        self._agent_scroll_restore = None if at_bottom or new_user_message else scroll.scroll_y
        self._agent_scroll_restore_frames = 8
        self._agent_history_signature = signature
        with scroll:
            with ui.VStack(height=0, spacing=8):
                ui.Spacer(height=6)
                for index, (role, text) in enumerate(signature):
                    with ui.HStack(height=0):
                        ui.Spacer(width=8)
                        with ui.VStack(height=0, spacing=3):
                            ui.Label({"You": "사용자", "Agent": "에이전트", "System": "시스템"}.get(role, role),
                                height=18, style={"color": 0xFF8FD8FF if role == "You" else 0xFFB8C9D5})
                            ui.Label(text, height=0, word_wrap=True,
                                identifier=f"tabletop_agent_message_{index}",
                                style={"color": 0xFFF2F4F5, "font_size": 16})
                            ui.Separator(height=1)
                        ui.Spacer(width=8)
                ui.Spacer(height=6)

    def _build_agent_settings(self) -> None:
        with ui.HStack(height=0, spacing=16):
            with ui.VStack(width=330, height=0, spacing=5):
                ui.Label("AGENT · ACTIONS", height=24)
                ui.Label(f"고정 장면: {self._agent_snapshot_id or 'none'}", height=0, word_wrap=True)
                ui.Label("화면 방향: 전송 순간 Viewport | 동 +X | 북 +Y", height=0, word_wrap=True)
                ui.Label("올려/내려 = 높이 Z | 테이블 표면 = Z 0", height=20)
                with ui.HStack(height=28, spacing=5):
                    ui.Button("장면 갱신", clicked_fn=self._refresh_agent_snapshot)
                    ui.Button("Ghost 취소", clicked_fn=self._clear_agent_preview)
                with ui.HStack(height=28, spacing=5):
                    ui.Button("추적 위치 복귀", clicked_fn=self._reset_agent_moves)
                    ui.Button("회전/크기 초기화", clicked_fn=self._reset_agent_transforms)
                with ui.HStack(height=28, spacing=5):
                    ui.Button("현재 장면 요약", clicked_fn=lambda: self._submit_agent_prompt("현재 장면을 요약해줘"))
                    ui.Button("누락 애셋 찾기", clicked_fn=lambda: self._submit_agent_prompt("전용 애셋이 없는 객체를 찾아줘"))
            with ui.VStack(height=0, spacing=5):
                ui.Button("객체 등록 · 3D 자산", clicked_fn=self._open_asset_studio)
                ui.Label("3D 복원 승인", height=24)
                if not self._jobs:
                    ui.Label("객체 등록 · 3D 자산에서 입력 이미지를 선택하세요", height=22)
                for object_id, job in list(self._jobs.items())[-6:]:
                    ui.Label(f"{job['label']} · {job['status']}", height=0, word_wrap=True)
                    if job["status"] == "awaiting_approval":
                        with ui.HStack(height=28, spacing=5):
                            ui.Button("입력 선택·생성", clicked_fn=lambda item_id=object_id: self._set_job_status(item_id, "approved_waiting_provider"))
                            ui.Button("거절", clicked_fn=lambda item_id=object_id: self._set_job_status(item_id, "rejected"))
                ui.Button("Reset Demo Layout", height=28, clicked_fn=self._request_layout_reset)

    def _request_layout_reset(self) -> None:
        self._layout_apply_at = time.monotonic()
        self._layout_redos = 0
        self._layout_phase = 0
        self._layout_pending = True
        for window in (self._window, self._settings_window, self._camera_window):
            window.visible = True
