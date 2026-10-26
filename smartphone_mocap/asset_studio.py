"""Persistent, local asset workflow. Images are selected from immutable snapshots."""
from __future__ import annotations
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid
from PIL import Image, ImageDraw
from .agent_control import atomic_write_json
from .meshy_provider import MeshyProvider

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.RLock()

class AssetStudio:
    def __init__(self, root=None, snapshots=None, provider=None):
        self.backend = provider or MeshyProvider()
        self.root = Path(root or ROOT/'outputs/asset_studio').resolve()
        self.snapshots = Path(snapshots or ROOT/'outputs/tabletop_agent/snapshots').resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, oid):
        if not re.fullmatch(r'asset_[a-f0-9]{32}', oid):
            raise ValueError('Invalid asset ID')
        return self.root/oid

    def load(self, oid):
        return json.loads((self.path(oid)/'record.json').read_text())

    def save(self, record):
        record['updated_ns'] = time.time_ns()
        atomic_write_json(self.path(record['id'])/'record.json', record)
        return record

    def snapshot(self, sid):
        p = (self.snapshots/sid/'snapshot.json').resolve()
        if p.parent.parent != self.snapshots:
            raise ValueError('Invalid snapshot')
        return json.loads(p.read_text())

    def image_path(self, sid, camera):
        s = self.snapshot(sid)
        p = Path(s['camera_files'][camera]['raw']).resolve()
        if not p.is_relative_to((self.snapshots/sid).resolve()):
            raise ValueError('Image is outside snapshot')
        return p

    def listing(self):
        snapshots = []
        for p in sorted(self.snapshots.glob('*/snapshot.json'), key=lambda p:p.stat().st_mtime, reverse=True)[:40]:
            try:
                d=json.loads(p.read_text())
                snapshots.append({'id':p.parent.name,'frame':d.get('frame_number'),
                    'cameras':list(d.get('camera_files',{})), 'modified_ns':p.stat().st_mtime_ns})
            except (OSError,ValueError): pass
        records=[json.loads(p.read_text()) for p in sorted(self.root.glob('asset_*/record.json'))]
        ack=self.root/'isaac_status.json'
        return {'snapshots':snapshots,'records':records,'provider_configured':self.backend.configured(),
                'isaac':json.loads(ack.read_text()) if ack.exists() else {}}

    def register(self, data):
        name=str(data.get('name','')).strip()
        if not name or len(name)>100: raise ValueError('객체 이름을 입력하세요 (100자 이하).')
        views=data.get('views',[])
        if not 1<=len(views)<=4: raise ValueError('같은 객체의 시점을 1~4개 선택하세요.')
        if data.get('same_object_confirmed') is not True: raise ValueError('선택한 시점이 같은 객체인지 확인하세요.')
        if len({v['camera'] for v in views})!=len(views): raise ValueError('카메라별로 한 영역만 선택하세요.')
        sid=data['snapshot_id'];self.snapshot(sid)
        prepared=[]
        for v in views:
            with Image.open(self.image_path(sid,v['camera'])) as source:
                im=source.convert('RGB')
            polygon=v.get('polygon',[])
            if len(polygon)<3: raise ValueError('객체 윤곽을 최소 3점으로 지정하세요.')
            if not all(len(p)==2 and all(math.isfinite(float(n)) and 0<=float(n)<=1 for n in p) for p in polygon):
                raise ValueError('윤곽 좌표는 이미지 내부의 정규화 좌표여야 합니다.')
            points=[(round(p[0]*(im.width-1)),round(p[1]*(im.height-1))) for p in polygon]
            mask=Image.new('L',im.size);ImageDraw.Draw(mask).polygon(points,fill=255)
            box=mask.getbbox()
            if not box or min(box[2]-box[0],box[3]-box[1])<16: raise ValueError('선택 영역이 너무 작습니다.')
            rgba=im.convert('RGBA');rgba.putalpha(mask)
            prepared.append((v,im,mask,rgba.crop(box),box))
        oid='asset_'+uuid.uuid4().hex;folder=self.path(oid);folder.mkdir()
        evidence=[]
        for i,(v,im,mask,crop,box) in enumerate(prepared):
            im.save(folder/f'raw_{i}.png');mask.save(folder/f'mask_{i}.png');crop.save(folder/f'input_{i}.png')
            evidence.append({'camera':v['camera'],'polygon':v['polygon'],'bbox_px':box,
                'input':f'input_{i}.png','sha256':hashlib.sha256((folder/f'input_{i}.png').read_bytes()).hexdigest()})
        return self.save({'id':oid,'label':name,'snapshot_id':sid,'views':evidence,
            'schema':'gist.asset_studio.v1','provider':self.backend.name,
            'status':'registered','created_ns':time.time_ns(),'desired':'none','revision':0,
            'position_source':'manual placement; not triangulated or automatically tracked'})

    def generate(self, oid):
        with LOCK:
            d=self.load(oid)
            if d.get('provider_task_id') or d['status'] in ('submitting','submission_unknown'):
                return d # Never automatically resubmit a potentially billable request.
            if not self.backend.configured():raise ValueError('먼저 생성 서비스 API 키를 설정하세요.')
            if d['status'] not in ('registered','failed'):raise ValueError('현재 상태에서 생성할 수 없습니다.')
            d['status']='submitting';self.save(d)
        images=['data:image/png;base64,'+base64.b64encode((self.path(oid)/v['input']).read_bytes()).decode() for v in d['views']]
        try:
            task_id=self.backend.submit(images)
            with LOCK:
                d=self.load(oid);d.update(provider_task_id=task_id,status='generating');return self.save(d)
        except Exception:
            with LOCK:
                d=self.load(oid);d.update(status='submission_unknown',error='생성 접수 여부를 확인하지 못했습니다. 중복 결제를 피하려면 Meshy 작업 ID를 조회해 연결하세요.');self.save(d)
            raise ValueError(d['error'])

    def refresh(self, oid):
        d=self.load(oid)
        task=d.get('provider_task_id')
        if not task:return d
        if not re.fullmatch(r'[A-Za-z0-9_-]+',task):raise ValueError('Invalid task ID')
        if d.get('provider', self.backend.name) != self.backend.name:
            raise ValueError('이 작업을 생성한 제공자 어댑터로 조회하세요.')
        result=self.backend.poll(task)
        with LOCK:
            d=self.load(oid);d['progress']=result.get('progress',0)
            status=result.get('status')
            if status in ('FAILED','CANCELED'):
                d.update(status='failed',error=str(result.get('error',{})))
            elif status=='SUCCEEDED' and not d.get('asset_file'):
                url=result.get('asset_url')
                if not url or not url.startswith('https://'):raise ValueError('USDZ 출력이 없습니다.')
                # Generated download URLs contain no API credentials.
                with urllib.request.urlopen(url,timeout=90) as response:
                    body=response.read(200*1024*1024+1)
                if len(body)>200*1024*1024:raise ValueError('자산이 200MB를 초과합니다.')
                target=self.path(oid)/'generated.usdz';target.write_bytes(body)
                d.update(status='ready',asset_file=str(target))
            return self.save(d)

    def command(self, data):
        action=data['action'];oid=data.get('id')
        if action=='register':return self.register(data)
        if action=='generate':return self.generate(oid)
        if action=='refresh':return self.refresh(oid)
        with LOCK:
            d=self.load(oid)
            if action=='attach_task':
                task=str(data['task_id'])
                if not re.fullmatch(r'[A-Za-z0-9_-]+',task):raise ValueError('Invalid task ID')
                if d.get('asset_file'):raise ValueError('기존 자산이 있습니다. 새 객체로 등록하세요.')
                d.update(provider_task_id=task,status='generating')
            elif action=='retry':
                if d['status']!='failed':raise ValueError('제공자가 실패를 확정한 작업만 재시도할 수 있습니다.')
                d.setdefault('previous_tasks',[]).append(d.pop('provider_task_id',None));d.update(status='registered',error=None)
            elif action=='import':
                path=Path(str(data['path'])).expanduser().resolve()
                if not path.is_file() or path.suffix.lower() not in ('.usd','.usda','.usdc','.usdz'):raise ValueError('로컬 USD/USDZ 파일을 지정하세요.')
                d.update(asset_file=str(path),status='ready')
            elif action in ('preview','apply','rollback'):
                if action!='rollback':
                    if not d.get('asset_file'):raise ValueError('생성 또는 가져오기를 먼저 완료하세요.')
                    xyz=[float(n) for n in data['xyz']];height=float(data['height_m']);yaw=float(data.get('yaw_deg',0))
                    if len(xyz)!=3 or not all(math.isfinite(n) for n in xyz+[height,yaw]) or not .005<=height<=3:
                        raise ValueError('유효한 XYZ와 높이(0.005~3m)를 입력하세요.')
                    d.update(xyz=xyz,height_m=height,yaw_deg=yaw)
                d['desired']=action;d['revision']+=1
            else:raise ValueError('지원하지 않는 작업입니다.')
            return self.save(d)
