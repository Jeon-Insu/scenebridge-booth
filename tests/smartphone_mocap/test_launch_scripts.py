"""The booth is installed on fresh machines: launch scripts may use only what Ubuntu ships plus the documented installs."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "tools").glob("*.sh"))


def test_launch_scripts_do_not_need_ripgrep():
    users = [p.name for p in SCRIPTS if re.search(r"(^|[|;&(\s])rg\s", p.read_text(encoding="utf-8"), re.M)]
    assert users == []          # rg was only on the development computer (~/.local/bin)


def test_isaac_readiness_wait_is_long_enough_for_a_first_start():
    text = (ROOT / "tools/run_tabletop_demo.sh").read_text(encoding="utf-8")
    assert "ISAAC_READY_TIMEOUT_S:-300" in text   # first start compiles shaders for minutes


def test_isaac_is_looked_for_on_the_desktop_too():
    text = (ROOT / "tools/run_isaac_tabletop_perception.sh").read_text(encoding="utf-8")
    assert '"${HOME}/Desktop"' in text
