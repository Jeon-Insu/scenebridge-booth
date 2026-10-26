"""Kit adapter for the independently testable camera-selection helpers."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[4] / "smartphone_mocap/camera_focus.py"
_spec = importlib.util.spec_from_file_location("gist_camera_focus", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
read_linked_preview = _module.read_linked_preview
resolve_focus_ids = _module.resolve_focus_ids
render_camera_focus = _module.render_camera_focus
