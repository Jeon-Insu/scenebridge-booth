"""Apply persistent manual assets on Kit's update thread; never during physics."""
import json
import math
from pathlib import Path
import time
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def normalized_asset(source, output, height, yaw=0):
    original=Usd.Stage.Open(str(source))
    if not original or not original.GetDefaultPrim():
        raise ValueError('자산에 유효한 defaultPrim이 필요합니다.')
    wrapper=Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(wrapper,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(wrapper,1)
    root=UsdGeom.Xform.Define(wrapper,'/Asset');wrapper.SetDefaultPrim(root.GetPrim())
    model=UsdGeom.Xform.Define(wrapper,'/Asset/Model')
    model.GetPrim().GetReferences().AddReference(str(Path(source).resolve()))
    # Orient under the root before measuring the resulting dimensions.
    rotate=UsdGeom.Xformable(model.GetPrim())
    if UsdGeom.GetStageUpAxis(original)==UsdGeom.Tokens.y:
        rotate.AddRotateXOp(opSuffix='studioUp').Set(90)
    if yaw:rotate.AddRotateZOp(opSuffix='studioYaw').Set(yaw)
    for prim in wrapper.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
    lo,hi=bounds.GetMin(),bounds.GetMax();extent=hi-lo
    if not all(math.isfinite(float(n)) for n in (*lo,*hi)) or extent[2]<=1e-8:
        raise ValueError('자산의 유효한 3D 경계를 계산할 수 없습니다.')
    scale=height/extent[2]
    root.AddTranslateOp().Set(Gf.Vec3d(-(lo[0]+hi[0])*.5*scale,-(lo[1]+hi[1])*.5*scale,-lo[2]*scale))
    root.AddScaleOp().Set(Gf.Vec3f(scale))
    wrapper.GetRootLayer().Save()
    return str(output)


class AssetStudioMixin:
    def _open_asset_studio(self):
        import webbrowser
        webbrowser.open('http://127.0.0.1:7080/asset-studio')

    def _poll_asset_studio(self, stage, now):
        if now-getattr(self,'_asset_studio_checked',0)<1:return
        self._asset_studio_checked=now
        root=Path(__file__).resolve().parents[4]/'outputs/asset_studio'
        if not root.exists():return
        if not hasattr(self,'_asset_studio_revisions'):
            self._asset_studio_revisions={};self._asset_studio_status={}
        for path in root.glob('asset_*/record.json'):
            try:
                d=json.loads(path.read_text());oid=d['id'];revision=d['revision']
                if d['desired']=='none':continue
                if self._asset_studio_revisions.get(oid)==revision:continue
                if self._simulation.active:
                    self._asset_studio_status[oid]={'state':'deferred','revision':revision,'detail':'물리 실험 종료 후 적용'};continue
                preview='/World/AssetStudioPreview/'+oid
                if stage.GetPrimAtPath(preview):stage.RemovePrim(preview)
                if d['desired']=='rollback':
                    if oid in self._records:self._remove_record(stage,oid)
                else:
                    output=path.parent/f'normalized_{revision}.usda'
                    if output.exists():output.unlink()
                    uri=normalized_asset(d['asset_file'],output,d['height_m'],d['yaw_deg'])
                    if d['desired']=='preview':
                        prim=UsdGeom.Xform.Define(stage,preview)
                        UsdGeom.Xform.Define(stage,preview+'/Visual').GetPrim().GetReferences().AddReference(uri)
                        prim.AddTranslateOp(opSuffix='placement').Set(self._with_offset(d['xyz']))
                    else:
                        self._update_object(stage,oid,{'id':oid,'label':d['label'],'xyz':d['xyz'],
                            'asset_uri':uri,'state':'active','position_mode':'manual_asset_placement',
                            'source_cameras':[]},now,time.time_ns())
                        # New geometry requires new physics preparation, not stale cached bounds.
                        self._physics_properties.pop(oid,None)
                self._asset_studio_revisions[oid]=revision
                self._asset_studio_status[oid]={'state':'applied','command':d['desired'],'revision':revision,'updated_ns':time.time_ns()}
                self._ui_dirty=True;self._state_dirty=True
            except Exception as exc:
                self._asset_studio_status[path.parent.name]={'state':'failed','detail':str(exc),'updated_ns':time.time_ns()}
                if 'd' in locals() and d.get('id') == path.parent.name:
                    self._asset_studio_revisions[d['id']] = d['revision']
                    self._asset_studio_status[d['id']]['revision'] = d['revision']
        tmp=root/'isaac_status.tmp';tmp.write_text(json.dumps(self._asset_studio_status,ensure_ascii=False));tmp.replace(root/'isaac_status.json')
