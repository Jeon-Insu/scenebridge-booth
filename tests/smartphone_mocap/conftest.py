"""Agent tests describe generic scenes (cups, plants): keep them independent of the project's booth list."""
import pytest


@pytest.fixture(autouse=True)
def _no_project_booth_list(tmp_path_factory, monkeypatch):
    from smartphone_mocap import booth_props
    monkeypatch.setattr(booth_props, "DEFAULT_PATH", tmp_path_factory.mktemp("no_list") / "booth_props.json")
