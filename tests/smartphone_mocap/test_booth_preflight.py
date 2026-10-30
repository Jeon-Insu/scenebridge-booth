import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from booth_preflight import check  # noqa: E402


def _project(tmp_path, *, models=("a1.pth", "head.pt", "b3.pth"), cameras_exist=True, venv=True):
    cfg_dir = tmp_path / "smartphone_mocap/config"
    cfg_dir.mkdir(parents=True)
    (tmp_path / "outputs/models").mkdir(parents=True)
    for name in models:
        (tmp_path / "outputs/models" / name).write_bytes(b"x")
    device = tmp_path / "video0"
    if cameras_exist:
        device.write_bytes(b"")
    if venv:
        (tmp_path / ".venv-mocap/bin").mkdir(parents=True)
        (tmp_path / ".venv-mocap/bin/python").write_text("")
    od = {"checkpoint": "../../outputs/models/a1.pth", "orientation_head": "../../outputs/models/head.pt",
          "general_checkpoint": "../../outputs/models/b3.pth"}
    config = cfg_dir / "booth.json"
    config.write_text(json.dumps({"object_detection": od, "cameras": [{"id": "cam_A", "source": str(device)}]}))
    return config


def test_a_complete_install_has_no_problems(tmp_path):
    errors, warnings = check(_project(tmp_path), tmp_path)
    assert errors == [] and warnings == []


def test_missing_figure_model_and_cameras_are_errors_in_korean(tmp_path):
    errors, _ = check(_project(tmp_path, models=("head.pt", "b3.pth"), cameras_exist=False), tmp_path)
    assert any("a1.pth" in e for e in errors)
    assert any("cam_A" in e and "카메라" in e for e in errors)


def test_missing_general_model_is_only_a_warning(tmp_path):
    errors, warnings = check(_project(tmp_path, models=("a1.pth", "head.pt")), tmp_path)
    assert errors == [] and any("b3.pth" in w for w in warnings)   # the tracker falls back to COCO


def test_missing_venv_is_an_error(tmp_path):
    errors, _ = check(_project(tmp_path, venv=False), tmp_path)
    assert any(".venv-mocap" in e for e in errors)


def test_language_model_download_needs_a_token_when_not_cached(tmp_path):
    from booth_preflight import llm_warning
    assert "HF_TOKEN" in llm_warning(tmp_path, {})
    assert llm_warning(tmp_path, {"HF_TOKEN": "x"}) is None
    (tmp_path / ".cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4").mkdir(parents=True)
    assert llm_warning(tmp_path, {}) is None
