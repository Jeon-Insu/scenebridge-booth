import io
import json
import urllib.request

import pytest

from smartphone_mocap.meshy_provider import MeshyProvider


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv('MESHY_API_KEY', 'test-placeholder')
    return MeshyProvider()


@pytest.fixture
def captured(monkeypatch):
    """Capture outgoing requests instead of billing the real API."""
    sent = []

    def fake_urlopen(request, timeout=None):
        sent.append(request)
        body = json.dumps({'result': 'task-1'}).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    return sent


def test_cutout_request_targets_image_endpoint_and_asks_for_transparency(provider, captured):
    provider.submit_cutout('data:image/jpeg;base64,AAA', model='nano-banana')

    request = captured[0]
    payload = json.loads(request.data)
    assert request.full_url == 'https://api.meshy.ai/openapi/v1/image-to-image'
    assert payload['remove_background'] is True
    assert payload['ai_model'] == 'nano-banana'
    assert payload['reference_image_urls'] == ['data:image/jpeg;base64,AAA']


def test_model_submission_still_targets_multi_image_endpoint(provider, captured):
    provider.submit(['data:image/jpeg;base64,AAA'])

    assert captured[0].full_url == 'https://api.meshy.ai/openapi/v1/multi-image-to-3d'


def test_submission_remeshes_to_a_physics_friendly_budget(provider, captured):
    """meshy-6 leaves should_remesh off, which returned a 198k-triangle cup."""
    provider.submit(['data:image/jpeg;base64,AAA'], target_polycount=5000)

    payload = json.loads(captured[0].data)
    assert payload['should_remesh'] is True
    assert payload['target_polycount'] == 5000


def test_submission_requests_pbr_so_metal_does_not_bake_to_flat_grey(provider, captured):
    provider.submit(['data:image/jpeg;base64,AAA'])

    payload = json.loads(captured[0].data)
    assert payload['should_texture'] is True
    assert payload['enable_pbr'] is True
