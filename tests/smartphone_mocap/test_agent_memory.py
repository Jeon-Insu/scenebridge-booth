import copy
import io
import json
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from smartphone_mocap.agent_memory import AgentMemory
from smartphone_mocap.agent_context import post_with_budget, preflight_budget
from smartphone_mocap.scene_agent import SceneTools, sglang_response
from test_physics_scenario import snapshot, decision


def test_prepared_survives_provider_failure_and_store_reopen(tmp_path, monkeypatch):
    import smartphone_mocap.scene_agent as agent
    db = AgentMemory(tmp_path/'state.db')
    db.claim('s','one','물리 준비','simulation',snapshot())
    def model(*args):
        if not hasattr(model,'called'):
            model.called=True
            return {'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'prep','type':'function','function':{
                'name':'prepare_physics','arguments':json.dumps({'decisions':[decision()]})}}]}}]}
        raise urllib.error.HTTPError('http://unused',400,'Bad',{},io.BytesIO(b'{"message":"maximum context length"}'))
    monkeypatch.setattr(agent,'_post_json',model)
    result = sglang_response('물리 준비',snapshot(),base_url='http://unused',model='test',timeout_seconds=1,
                             memory=db,session_id='s',request_id='one')
    assert result['actions'] == [] and result['provider_error']
    db.finish('s','one',result)
    restored=AgentMemory(tmp_path/'state.db').hydrate('s',snapshot())
    assert SceneTools(restored,physics_requested=True).physics.decisions['cup/one/000']['mass_kg']==.01
    assert not AgentMemory(tmp_path/'state.db').hydrate('other',snapshot())['twin_state']['physics_properties']


def test_fact_invalidates_estimate_but_experiment_assumption_does_not(tmp_path):
    db=AgentMemory(tmp_path/'state.db')
    tools=SceneTools(snapshot(),physics_requested=True)
    tools.physics.prepare_physics([decision()])
    db.save_properties('s','one',tools.physics.decisions)
    db.fact('s','two','cup/one/000','material','ceramic','세라믹이라고 가정하면','experiment')
    assert db.hydrate('s',snapshot())['twin_state']['physics_properties']
    db.fact('s','three','cup/one/000','material','plastic','이 컵은 플라스틱이야','object')
    stale=copy.deepcopy(snapshot());stale['twin_state']['physics_properties']=tools.physics.decisions
    hydrated=db.hydrate('s',stale)
    assert not hydrated['twin_state']['physics_properties']
    assert hydrated['twin_state']['object_facts'][0]['value']=='plastic'
    changed={**tools.physics.decisions['cup/one/000'],'material':'plastic'}
    db.save_properties('s','four',{'cup/one/000':changed})
    assert db.hydrate('s',stale)['twin_state']['physics_properties']['cup/one/000']['material']=='plastic'
    with db.connect() as conn:
        assert conn.execute('SELECT count(*) FROM properties').fetchone()[0]==2


def test_claim_is_atomic_and_execution_is_distinct_from_planning(tmp_path):
    db=AgentMemory(tmp_path/'state.db')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:db.claim('s','r','run','simulation',snapshot()), range(4)))
    assert results.count(None)==1
    assert db.claim('s','r2','another','qa',snapshot())['busy']
    result=db.finish('s','r',{'reply':'queued','actions':[{'type':'physics_control','command':'play'}]})
    action_id=result['actions'][0]['action_id']
    assert db.backend_context('s','r')['actions'][0]['state']=='queued'
    assert not db.acknowledge('other',action_id,True,'wrong session')
    assert db.acknowledge('s',action_id,False,'Prepare a simulation first')
    assert not db.acknowledge('s',action_id,True,'duplicate')
    assert db.backend_context('s','r')['actions'][0]['state']=='failed'
    assert db.claim('s','r2','another','qa',snapshot()) is None
    db.recover_interrupted()
    assert db.backend_context('s','r2')['request']['state']=='interrupted'


def test_context_rejection_reduces_output_without_repeating_tools():
    payload={'messages':[{'role':'user','content':'question'}],'max_tokens':2048}
    calls=[]
    def post(url,data,timeout):
        calls.append(copy.deepcopy(data))
        if len(calls)==1:
            raise urllib.error.HTTPError(url,400,'Bad',{},io.BytesIO(json.dumps({'message':
                "Requested token count exceeds the model's maximum context length of 16384 tokens. 14447 tokens from the input messages"}).encode()))
        return {'choices':[]}
    post_with_budget(post,'http://unused',payload,1)
    assert len(calls)==2 and calls[1]['max_tokens']==1873
    assert calls[0]['messages']==calls[1]['messages']
    assert payload['max_tokens']==2048


def test_preflight_preserves_current_question_and_images():
    messages=[{'role':'system','content':'policy'},{'role':'user','content':'state'},
              {'role':'user','content':'old'},{'role':'assistant','content':'old reply'},
              {'role':'user','content':'CURRENT'}, {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:...'}}]}]
    payload={'model':'x','messages':messages,'tools':[],'tool_choice':'auto','chat_template_kwargs':{},'max_tokens':2048}
    def count(url,data,timeout):
        assert url.endswith('/v1/tokenize')
        return {'count':14000 if len(data['messages'])==6 else 12000,'max_model_len':16384}
    result=preflight_budget(count,'http://local/v1/chat/completions',payload,2,history_count=2)
    assert result['history_removed']==2
    assert payload['messages'][2]['content']=='CURRENT'
    assert payload['messages'][3]['content'][0]['type']=='image_url'


def test_hypothetical_prepare_is_not_persisted(tmp_path,monkeypatch):
    import smartphone_mocap.scene_agent as agent
    db=AgentMemory(tmp_path/'state.db')
    calls=iter([{'role':'assistant','tool_calls':[{'id':'p','type':'function','function':{'name':'prepare_physics','arguments':json.dumps({'decisions':[decision()]})}}]}, {'role':'assistant','content':'가정된 실험 준비'}])
    monkeypatch.setattr(agent,'_post_json',lambda *args:{'choices':[{'message':next(calls)}]})
    sglang_response('종이컵이라고 가정해서 물리 준비',snapshot(),base_url='http://unused',model='test',timeout_seconds=1,memory=db,session_id='s',request_id='r')
    assert not db.hydrate('s',snapshot())['twin_state']['physics_properties']


def test_fact_tool_uses_exact_id_and_preserves_explicit_negation(tmp_path,monkeypatch):
    import smartphone_mocap.scene_agent as agent
    db=AgentMemory(tmp_path/'state.db')
    prompt='이 컵은 종이컵이야. 가정이 아니라 확인한 사실이야.'
    calls=iter([
        {'role':'assistant','tool_calls':[{'id':'bad','type':'function','function':{'name':'record_object_fact','arguments':json.dumps({'object_id':'','field':'material','value':'paper','scope':'object','user_quote':prompt})}}]},
        {'role':'assistant','tool_calls':[{'id':'good','type':'function','function':{'name':'record_object_fact','arguments':json.dumps({'object_id':'cup/one/000','field':'material','value':'paper','scope':'object','user_quote':prompt})}}]},
        {'role':'assistant','content':'기록했습니다.'}])
    monkeypatch.setattr(agent,'_post_json',lambda *args:{'choices':[{'message':next(calls)}]})
    result=sglang_response(prompt,snapshot(),base_url='http://unused',model='test',timeout_seconds=1,memory=db,session_id='s',request_id='r')
    assert [e['ok'] for e in result['evidence_trace']]==[False,True]
    assert db.hydrate('s',snapshot())['twin_state']['object_facts'][0]['value']=='paper'


def test_property_versions_ignore_pose_and_record_revisions(tmp_path):
    db=AgentMemory(tmp_path/'state.db')
    tools=SceneTools(snapshot(),physics_requested=True)
    tools.physics.prepare_physics([decision()])
    assert db.save_properties('s','one',tools.physics.decisions)=={'cup/one/000':1}
    tools.physics.decisions['cup/one/000']['xyz']=[1,2,3]
    assert db.save_properties('s','two',tools.physics.decisions)=={'cup/one/000':1}
    tools.physics.decisions['cup/one/000']['mass_kg']=.02
    assert db.save_properties('s','three',tools.physics.decisions)=={'cup/one/000':2}


def test_http_duplicate_and_ack_do_not_reexecute(tmp_path,monkeypatch):
    import threading
    import urllib.request
    import smartphone_mocap.agent_service as service
    calls=[]
    def respond(*args,**kwargs):
        calls.append(1)
        return {'reply':'queued','actions':[{'type':'physics_control','command':'play'}]}
    monkeypatch.setattr(service,'respond',respond)
    server=service.TabletopAgentServer(('127.0.0.1',0),{'memory_db_path':str(tmp_path/'http.db')})
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/v1/agent'
    def post(path,payload):
        request=urllib.request.Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
    try:
        payload={'prompt':'실행','snapshot':snapshot(),'session_id':'s','request_id':'r'}
        first=post('/respond',payload)
        assert first['actions'][0]['action_id']
        assert post('/respond',payload)['actions']==[] and len(calls)==1
        post('/ack',{'session_id':'s','action_id':first['actions'][0]['action_id'],'ok':True,'detail':'done'})
        with urllib.request.urlopen(base+'/task?session_id=s&request_id=r') as response:
            assert json.load(response)['execution']=='applied'
    finally:
        server.shutdown();server.server_close();thread.join()


def test_user_material_fact_cannot_be_overwritten_by_a_guess(tmp_path):
    db=AgentMemory(tmp_path/'state.db')
    db.fact('s','r','cup/one/000','material','paper','종이컵이야','object')
    planner=SceneTools(db.hydrate('s',snapshot()),physics_requested=True)
    response=planner.call('prepare_physics',{'decisions':[{**decision(),'material':'ceramic'}]})
    assert not response['ok'] and "'paper'" in response['error']
    assert planner.call('prepare_physics',{'decisions':[decision()]})['ok']


def test_new_conversation_resets_automatic_history_not_object_state(tmp_path):
    db=AgentMemory(tmp_path/'state.db')
    s=snapshot();s['conversation_id']='old'
    db.claim('session','r','old question','qa',s)
    db.finish('session','r',{'reply':'old answer','actions':[]})
    planner=SceneTools(s,physics_requested=True);planner.physics.prepare_physics([decision()])
    db.save_properties('session','r',planner.physics.decisions)
    assert len(db.history('session',conversation_id='old'))==1
    assert db.history('session',conversation_id='new')==[]
    assert db.hydrate('session',snapshot())['twin_state']['physics_properties']
    assert db.history('session','old question')[0]['request_id']=='r'


def test_run_request_cannot_be_acknowledged_without_an_action(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    requests=[]
    def model(url,payload,timeout):
        requests.append(copy.deepcopy(payload))
        return {'choices':[{'message':{'content':'시뮬레이션을 실행했습니다.'}}]}
    monkeypatch.setattr(agent,'_post_json',model)
    result=sglang_response('시뮬레이션 실행해 줘',snapshot(),base_url='http://unused',model='test',timeout_seconds=1)
    assert len(requests)==2 and requests[1]['tool_choice']=='required'
    assert result['actions']==[] and '실행하지 않았습니다' in result['reply']


def test_explicit_mass_accepts_unused_empty_fill_metadata():
    planner=SceneTools(snapshot(),physics_requested=True)
    result=planner.call('prepare_physics',{'decisions':[{**decision(),'fill_fraction':0}]})
    assert result['ok'] and result['decisions'][0]['mass_kg']==.01


def test_images_released_after_geometry_tool_before_physics_preparation(tmp_path, monkeypatch):
    import smartphone_mocap.scene_agent as agent
    image=tmp_path/'raw.jpg';image.write_bytes(b'test-jpeg')
    s=snapshot();s['camera_files']={'camera':{'raw':str(image)}}
    responses=iter([
        {'role':'assistant','tool_calls':[{'id':'images','type':'function','function':{'name':'inspect_images','arguments':'{"source":"physical"}'}}]},
        {'role':'assistant','content':'실물 이미지에서 컵이 보이며 재질은 불확실합니다.', 'tool_calls':[{'id':'geometry','type':'function','function':{'name':'inspect_physics_inputs','arguments':'{}'}}]},
        {'role':'assistant','tool_calls':[{'id':'again','type':'function','function':{'name':'inspect_images','arguments':'{"source":"physical"}'}}]},
        {'role':'assistant','content':'물성은 추가 확인이 필요합니다.'},
    ])
    counts=[];requests=[]
    def post(url,payload,timeout):
        requests.append(copy.deepcopy(payload))
        counts.append(sum(p.get('type')=='image_url' for m in payload['messages'] if isinstance(m.get('content'),list) for p in m['content']))
        return {'choices':[{'message':next(responses)}]}
    monkeypatch.setattr(agent,'_post_json',post)
    result=agent.sglang_response('물리 실험에 쓸 재질을 설명해줘',s,base_url='http://unused',model='gemma-4',timeout_seconds=1,vision_input=True)
    assert counts==[0,1,0,1]
    assert any(m.get('content')=='실물 이미지에서 컵이 보이며 재질은 불확실합니다.' for m in requests[2]['messages'])
    assert image.read_bytes()==b'test-jpeg'
    assert result['actions']==[]
