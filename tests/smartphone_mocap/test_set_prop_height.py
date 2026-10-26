"""tools/set_prop_height.py rescales a placed prop to a new height and records it in the catalog."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLACED = '''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Asset"
{
    float3 xformOp:scale = (0.3048, 0.3048, 0.3048)
    double3 xformOp:translate = (-0.001, -0.002, 0.225)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
}
'''


def _props(tmp_path):
    (tmp_path / 'elephant').mkdir()
    (tmp_path / 'elephant/model_placed.usda').write_text(PLACED)
    (tmp_path / 'catalog.json').write_text(json.dumps({'schema_version': 1, 'assets': [
        {'asset_id': 'elephant', 'display_name_ko': '코끼리', 'usd_uri': 'elephant/model_placed.usda',
         'detector_class': 'elephant', 'height_m': 0.45, 'height_source': 'meshy_auto_size'}]}))
    return tmp_path


def _numbers(text, name):
    return [float(v) for v in re.search(name + r' = \(([^)]*)\)', text).group(1).split(',')]


def test_a_new_height_scales_the_model_and_its_placement(tmp_path):
    props = _props(tmp_path)
    subprocess.run([sys.executable, str(ROOT / 'tools/set_prop_height.py'), 'elephant', '0.12',
                    '--props', str(props)], check=True)

    text = (props / 'elephant/model_placed.usda').read_text()
    ratio = 0.12 / 0.45
    assert _numbers(text, 'xformOp:scale') == pytest.approx([0.3048 * ratio] * 3)
    assert _numbers(text, 'xformOp:translate') == pytest.approx([-0.001 * ratio, -0.002 * ratio, 0.225 * ratio])
    entry = json.loads((props / 'catalog.json').read_text())['assets'][0]
    assert entry['height_m'] == 0.12
    assert entry['height_source'] == 'estimate'


def test_an_unknown_prop_is_refused(tmp_path):
    props = _props(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / 'tools/set_prop_height.py'), 'unicorn', '0.1',
                             '--props', str(props)], capture_output=True, text=True)
    assert result.returncode == 1
    assert '없는 소품' in result.stdout + result.stderr
