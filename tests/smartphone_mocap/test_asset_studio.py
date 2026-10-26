import json
from pathlib import Path
import pytest
from PIL import Image
from smartphone_mocap.asset_studio import AssetStudio

@pytest.fixture
def studio(tmp_path):
    snapshots=tmp_path/'snapshots';p=snapshots/'snap';p.mkdir(parents=True)
    Image.new('RGB',(100,80),'red').save(p/'raw.png')
    (p/'snapshot.json').write_text(json.dumps({'camera_files':{'a':{'raw':str(p/'raw.png')}}}))
    return AssetStudio(tmp_path/'assets',snapshots)

def register(studio):
    return studio.register({'name':'종이컵','snapshot_id':'snap','same_object_confirmed':True,
        'views':[{'camera':'a','polygon':[[.1,.1],[.8,.1],[.8,.8],[.1,.8]]}]})

def test_registration_preserves_source_and_mask(studio):
    d=register(studio);p=studio.path(d['id'])
    assert d['position_source'].startswith('manual')
    assert Image.open(p/'raw_0.png').size==(100,80)
    assert Image.open(p/'mask_0.png').size==(100,80)
    assert Image.open(p/'input_0.png').mode=='RGBA'
    assert studio.load(d['id'])['snapshot_id']=='snap'
    with pytest.raises(ValueError):studio.image_path('../snap','a')

def test_missing_key_and_ambiguous_submission_cannot_duplicate(studio,monkeypatch):
    d=register(studio);monkeypatch.delenv('MESHY_API_KEY',raising=False)
    with pytest.raises(ValueError):studio.generate(d['id'])
    assert studio.load(d['id'])['status']=='registered'
    monkeypatch.setenv('MESHY_API_KEY','test-placeholder')
    calls=[]
    def fail(images):calls.append(images);raise TimeoutError()
    monkeypatch.setattr(studio.backend,'submit',fail)
    with pytest.raises(ValueError):studio.generate(d['id'])
    assert studio.generate(d['id'])['status']=='submission_unknown'
    assert len(calls)==1

def test_import_preview_apply_rollback_and_no_ready_before_output(studio,tmp_path):
    d=register(studio);oid=d['id']
    args={'id':oid,'xyz':[.5,.3,0],'height_m':.1}
    with pytest.raises(ValueError):studio.command({'action':'apply',**args})
    asset=tmp_path/'cube.usda';asset.write_text('#usda 1.0')
    studio.command({'action':'import','id':oid,'path':str(asset)})
    assert studio.command({'action':'preview',**args})['revision']==1
    assert studio.command({'action':'apply',**args})['desired']=='apply'
    assert studio.command({'action':'rollback','id':oid})['desired']=='rollback'
    with pytest.raises(ValueError):studio.command({'action':'apply',**args,'height_m':float('nan')})

def test_provider_task_and_failed_retry_are_persistent(studio,monkeypatch):
    d=register(studio);monkeypatch.setenv('MESHY_API_KEY','test-placeholder')
    monkeypatch.setattr(studio.backend,'submit',lambda images:'task123')
    assert studio.generate(d['id'])['provider_task_id']=='task123'
    monkeypatch.setattr(studio.backend,'poll',lambda task:{'status':'FAILED'})
    assert studio.refresh(d['id'])['status']=='failed'
    assert studio.command({'action':'retry','id':d['id']})['status']=='registered'
