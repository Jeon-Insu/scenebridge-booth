"""Meshy adapter. A replacement provider implements request/submit/poll translation here."""
import os
import json
import urllib.request

BASE = "https://api.meshy.ai/openapi/v1"
MODEL_PATH = "/multi-image-to-3d"
IMAGE_PATH = "/image-to-image"

# image-to-image is generative, so the prompt has to forbid restyling the subject.
CUTOUT_PROMPT = ("Remove the background only. Keep the object itself completely unchanged: "
                 "same shape, proportions, colors, texture, and camera angle. Do not restyle, "
                 "redraw, crop, or add anything.")

class MeshyProvider:
    name = 'meshy'

    def configured(self):
        return bool(os.environ.get('MESHY_API_KEY'))

    # Tabletop props get simulated, so the raw mesh (198k triangles on the first cup) is remeshed
    # to a physics-friendly budget. PBR is requested so metal does not bake down to flat grey.
    DEFAULT_POLYCOUNT = 5000

    def submit(self, images, target_polycount=DEFAULT_POLYCOUNT, enable_pbr=True):
        result = self.request(MODEL_PATH, payload={'image_urls': images, 'ai_model':'meshy-6',
            'should_texture':True, 'target_formats':['usdz'], 'image_enhancement':False,
            'should_remesh':True, 'target_polycount':int(target_polycount),
            'topology':'triangle', 'enable_pbr':bool(enable_pbr)})
        return str(result['result'])

    def poll(self, task_id):
        result = self.request(f'{MODEL_PATH}/{task_id}')
        return {'status':result.get('status'), 'progress':result.get('progress',0),
                'error':result.get('task_error'), 'asset_url':result.get('model_urls',{}).get('usdz')}

    def submit_cutout(self, image, model='nano-banana'):
        result = self.request(IMAGE_PATH, payload={'ai_model':model, 'prompt':CUTOUT_PROMPT,
            'reference_image_urls':[image], 'remove_background':True})
        return str(result['result'])

    def poll_cutout(self, task_id):
        result = self.request(f'{IMAGE_PATH}/{task_id}')
        return {'status':result.get('status'), 'progress':result.get('progress',0),
                'error':result.get('task_error'), 'image_urls':result.get('image_urls',[]),
                'consumed_credits':result.get('consumed_credits')}

    def balance(self):
        return self.request('/balance').get('balance')

    def request(self, path='', payload=None):
        key=os.environ.get('MESHY_API_KEY')
        if not key: raise ValueError('MESHY_API_KEY가 설정되지 않았습니다. 로컬 USD 가져오기는 사용할 수 있습니다.')
        req=urllib.request.Request(BASE+path, data=json.dumps(payload).encode() if payload else None,
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)
