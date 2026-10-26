import pytest

from smartphone_mocap.prop_catalog import PropCatalog

ENTRY = {
    'asset_id': 'stainless_mug',
    'display_name_ko': '스테인리스 머그',
    'usd_uri': 'stainless_mug/model.usdz',
    'detector_class': 'cup',
    'height_m': 0.095,
    'height_source': 'measured',
}


@pytest.fixture
def catalog(tmp_path):
    return PropCatalog(tmp_path / 'props')


def test_added_entry_survives_a_reload(catalog, tmp_path):
    catalog.add(ENTRY)

    assert PropCatalog(tmp_path / 'props').get('stainless_mug')['height_m'] == 0.095


def test_duplicate_asset_id_is_refused_so_the_gallery_keeps_one_truth(catalog):
    catalog.add(ENTRY)

    with pytest.raises(ValueError, match='이미 있습니다'):
        catalog.add({**ENTRY, 'height_m': 0.2})

    assert catalog.get('stainless_mug')['height_m'] == 0.095


def test_estimated_height_cannot_be_recorded_as_a_measurement(catalog):
    with pytest.raises(ValueError, match='height_source'):
        catalog.add({**ENTRY, 'height_source': 'eyeballed'})


def test_class_lookup_returns_every_instance_of_that_class(catalog):
    catalog.add(ENTRY)
    catalog.add({**ENTRY, 'asset_id': 'paper_cup', 'usd_uri': 'paper_cup/model.usdz'})
    catalog.add({**ENTRY, 'asset_id': 'mini_plant', 'usd_uri': 'mini_plant/model.usdz',
                 'detector_class': 'potted plant'})

    assert [e['asset_id'] for e in catalog.by_class('cup')] == ['stainless_mug', 'paper_cup']
    assert [e['asset_id'] for e in catalog.by_class('CUP')] == ['stainless_mug', 'paper_cup']


def test_missing_required_field_is_refused(catalog):
    with pytest.raises(ValueError, match='detector_class'):
        catalog.add({k: v for k, v in ENTRY.items() if k != 'detector_class'})


def test_usd_uri_can_be_repointed_at_the_normalized_asset(catalog):
    """Retrieval hands the URI straight to the twin, so it must be the placed version."""
    catalog.add(ENTRY)

    catalog.set_usd_uri('stainless_mug', 'stainless_mug/model_placed.usda')

    assert catalog.get('stainless_mug')['usd_uri'] == 'stainless_mug/model_placed.usda'
    assert catalog.get('stainless_mug')['source_usd_uri'] == 'stainless_mug/model.usdz'


def test_repointing_an_unknown_asset_is_refused(catalog):
    with pytest.raises(ValueError, match='없습니다'):
        catalog.set_usd_uri('ghost', 'ghost/model_placed.usda')
