from itertools import product
import copy
import pytest
from smartphone_mocap.scene_agent import SceneTools
from smartphone_mocap.physics_scenario import geometry_input, validate_decision, get_job


def snapshot():
    return {"snapshot_id":"test", "tracking":{"objects":[{"id":"cup/one/000","label":"cup","xyz":[.4,.3,.2],"state":"active"}]},
            "interaction_frame":{"workspace_bounds_xy":[0,1.25,0,.75]},
            "twin_state":{"runtime_objects":{"cup/one/000":{"contact_geometry":{
                "points_local_m":[list(p) for p in product([-.04,.04],[-.04,.04],[0,.1])]}}}}}


def decision():
    return {"object_id":"cup/one/000","material":"paper","collider":"cylinder", "mass_kg":.01,
            "static_friction":.5,"dynamic_friction":.4,"restitution":.1,"confidence":.5,
            "rationale":"Thin empty shell", "evidence":"Synthetic test fixture, not a real image",
            "assumptions":["Empty cup, approximate geometry"]}


def test_agent_decides_no_class_defaults_no_live_mutation(monkeypatch):
    s = snapshot()
    original = copy.deepcopy(s)
    tools = SceneTools(s, physics_requested=True)
    assert tools.call("inspect_physics_inputs", {})["material_defaults"] is None
    assert tools.call("prepare_physics", {"decisions":[decision()]})["ok"]
    received = []
    monkeypatch.setattr("smartphone_mocap.physics_scenario.submit_job",lambda scenario:received.append(scenario) or {"ok":True})
    assert tools.call("simulate_what_if", {"object_ids":["cup/one/000"], "table_material":{
        "static_friction":.5,"dynamic_friction":.4,"restitution":.1,"rationale":"test surface"}})["ok"]
    assert received[0]["bodies"][0]["mass_kg"] == .01
    assert s == original and tools.actions == []


def test_on_demand_authority_and_stale_geometry(monkeypatch):
    assert not SceneTools(snapshot()).call("prepare_physics",{"decisions":[decision()]})["ok"]
    tools = SceneTools(snapshot(), physics_requested=True)
    tools.call("prepare_physics",{"decisions":[decision()]})
    tools.objects[0]["scale"] = [2,2,2]
    result = tools.call("simulate_what_if", {"object_ids":["cup/one/000"],"table_material":{}})
    assert not result["ok"] and "changed" in result["error"]


def test_density_requires_fill_fraction_and_invalid_estimates_fail():
    tools = SceneTools(snapshot(), physics_requested=True)
    item = geometry_input(tools.objects[0])
    d = decision()
    del d["mass_kg"]
    d["density_kg_m3"] = 800
    with pytest.raises((ValueError,TypeError)):
        validate_decision(d,item)
    d["fill_fraction"] = .01
    assert validate_decision(d,item)["mass_kg"] < .01
    for field,value in [("mass_kg",float("nan")),("dynamic_friction",2),("restitution",1.1)]:
        with pytest.raises(ValueError):
            validate_decision({**decision(),field:value},item)
    with pytest.raises(ValueError):
        get_job("../../other")


def test_pending_execution_cannot_be_announced_as_completed(monkeypatch):
    tools=SceneTools(snapshot(),physics_requested=True)
    tools.physics.jobs=[{"job_id":"a"*32}]
    monkeypatch.setattr("smartphone_mocap.physics_scenario.get_job",lambda _: {"status":"running"})
    reply=tools.physics.finalize_reply("시뮬레이션이 성공적으로 완료되었습니다.")
    assert "실행 중" in reply and "완료" not in reply


def test_explicit_mass_priority_is_recorded():
    geometry = geometry_input(SceneTools(snapshot()).objects[0])
    result = validate_decision({**decision(), "density_kg_m3": 800, "fill_fraction": .000001}, geometry)
    assert result["mass_kg"] == .01
    assert "explicit_mass_used_density_and_fill_not_used_for_dynamics" in result["warnings"]


def test_viewport_preview_and_controls_are_actions_without_live_mutation():
    tools = SceneTools(snapshot(), physics_requested=True)
    tools.call("prepare_physics", {"decisions": [decision()]})
    surface = {"static_friction": .5, "dynamic_friction": .4,
               "restitution": .1, "rationale": "laminated tabletop estimate"}
    preview = tools.call("preview_physics", {"object_ids": ["cup/one/000"],
                                              "table_material": surface,
                                              "release_height_m": .2})
    assert preview["ok"] and preview["status"] == "queued_for_isaac"
    action = tools.actions[-1]
    assert action["type"] == "physics_preview"
    assert action["scenario"]["bodies"][0]["xyz"][2] == pytest.approx(.4)
    assert tools.call("control_physics", {"command": "play"})["ok"]
    assert tools.actions[-1] == {"type": "physics_control", "command": "play"}


def prepared_snapshot():
    s = snapshot()
    tools = SceneTools(s, physics_requested=True)
    tools.physics.prepare_physics([decision()])
    s['twin_state']['physics_properties'] = copy.deepcopy(tools.physics.decisions)
    s['twin_state']['physics_table_material'] = {
        'static_friction': .5, 'dynamic_friction': .4, 'restitution': .1, 'rationale': 'test'}
    s['twin_state']['simulation'] = {'active': True}
    return s


def test_new_question_reuses_properties_and_moves_full_meter_off_table():
    s = prepared_snapshot()
    s['twin_state']['runtime_objects']['cup/one/000']['xyz'] = [.4, .3, .2]
    original = copy.deepcopy(s)
    tools = SceneTools(s, physics_requested=True)
    result = tools.call('preview_physics_move', {'query': 'cup/one/000', 'direction': 'south', 'distance_m': 1})
    assert result['ok'], result
    body = tools.actions[-1]['scenario']['bodies'][0]
    assert body['xyz'] == pytest.approx([.4, -.7, .2])
    assert body['mass_kg'] == decision()['mass_kg']
    assert body['material'] == 'paper'
    assert tools.actions[-1]['type'] == 'physics_preview'
    assert s == original
    assert not tools.call('preview_move', {'query': 'cup', 'direction': 'south'})['ok']
    assert len(tools.actions) == 1


def test_properties_are_not_reestimated_without_explicit_update():
    tools = SceneTools(prepared_snapshot(), physics_requested=True)
    changed = {**decision(), 'mass_kg': .25, 'material': 'ceramic'}
    result = tools.physics.prepare_physics([changed])
    assert result['decisions'][0]['material'] == 'paper'
    assert result['decisions'][0]['mass_kg'] == .01
    result = tools.physics.prepare_physics([changed], update_existing=True)
    assert result['decisions'][0]['mass_kg'] == .25


def test_pose_changes_reuse_properties_but_scale_change_invalidates():
    s = prepared_snapshot()
    runtime = s['twin_state']['runtime_objects']['cup/one/000']
    runtime['xyz'] = [.6, -.4, .1]
    runtime['rotation_wxyz'] = [0, 1, 0, 0]
    tools = SceneTools(s, physics_requested=True)
    assert tools.physics.decisions['cup/one/000']['xyz'] == [.6, -.4, .1]
    assert tools.physics.decisions['cup/one/000']['rotation_wxyz'] == [0, 1, 0, 0]
    runtime['scale'] = [2, 2, 2]
    assert not SceneTools(s, physics_requested=True).physics.decisions


def test_interaction_overrides_are_isolated_and_mass_repeat_preserves_push():
    s = prepared_snapshot()
    original = copy.deepcopy(s)
    tools = SceneTools(s, physics_requested=True)
    result = tools.call('configure_experiment', {'bodies': [
        {'object_id':'cup/one/000', 'mass_kg':.02, 'impulse_ns':[.004,0,0]}]})
    assert result['ok'], result
    body = result['scenario']['bodies'][0]
    assert body['mass_kg'] == .02 and body['impulse_ns'] == [.004,0,0]
    assert result['scenario']['properties_scope'] == 'experiment'
    assert s == original
    s['twin_state']['simulation']['initial_bodies'] = {body['object_id']: body}
    s['twin_state']['runtime_objects']['cup/one/000']['xyz'] = [1,.5,0]
    again = SceneTools(s, physics_requested=True).call('configure_experiment', {'bodies': [
        {'object_id':'cup/one/000', 'mass_kg':.04}]})
    repeated = again['scenario']['bodies'][0]
    assert repeated['xyz'] == body['xyz']
    assert repeated['impulse_ns'] == body['impulse_ns']
    assert repeated['mass_kg'] == .04
    assert s['twin_state']['physics_properties']['cup/one/000']['mass_kg'] == .01


@pytest.mark.parametrize('config', [
    {'mass_kg':0}, {'force_n':[1,0,0]}, {'force_n':[float('nan'),0,0]},
    {'velocity_m_s':[1,2]}, {'force_duration_s':4},
    {'motion':'static','impulse_ns':[1,0,0]}, {'dynamic_friction':9},
])
def test_invalid_interaction_is_atomic(config):
    tools = SceneTools(prepared_snapshot(), physics_requested=True)
    result = tools.call('configure_experiment', {'bodies':[{'object_id':'cup/one/000', **config}]})
    assert not result['ok'], result
    assert not tools.actions


def test_release_height_uses_original_baseline_after_repeat():
    s = prepared_snapshot()
    s['twin_state']['runtime_objects']['cup/one/000']['xyz'] = [.4,.3,.5]
    s['twin_state']['simulation']['release_origins'] = {'cup/one/000':[.4,.3,.2]}
    tools = SceneTools(s, physics_requested=True)
    result = tools.call('preview_physics', {'object_ids':['cup/one/000'], 'release_height_m':.3})
    assert result['ok'], result
    assert tools.actions[0]['scenario']['bodies'][0]['xyz'][2] == pytest.approx(.5)


def test_static_collision_partner_is_explicit_and_result_is_measured():
    s = prepared_snapshot()
    second = copy.deepcopy(s['tracking']['objects'][0])
    second.update(id='cup/two/000',xyz=[.8,.3,0])
    s['tracking']['objects'].append(second)
    s['twin_state']['runtime_objects'][second['id']] = copy.deepcopy(s['twin_state']['runtime_objects']['cup/one/000'])
    tools = SceneTools(s, physics_requested=True)
    tools.call('prepare_physics', {'decisions':[{**decision(), 'object_id':second['id']}]})
    result = tools.call('configure_experiment', {'bodies':[
        {'object_id':'cup/one/000','force_n':[.1,0,0],'force_duration_s':.2},
        {'object_id':second['id'],'motion':'static'}]})
    assert result['ok'], result
    assert [b['motion'] for b in result['scenario']['bodies']] == ['dynamic','static']
    assert not tools.call('get_experiment_result', {})['ok']
    tools.twin_state['simulation']['result'] = {'contacts':[], 'elapsed_simulation_s':0}
    assert tools.call('get_experiment_result', {})['status'] == 'awaiting_isaac_execution'
    tools.actions.clear()
    assert tools.call('get_experiment_result', {})['result']['contacts'] == []


def test_push_target_direction_uses_initial_poses_not_model_cardinal_guess():
    s = prepared_snapshot()
    second = {**s['tracking']['objects'][0], 'id':'cup/two/000', 'xyz':[.1,.1,.2]}
    s['tracking']['objects'].append(second)
    s['twin_state']['runtime_objects'][second['id']] = copy.deepcopy(s['twin_state']['runtime_objects']['cup/one/000'])
    tools = SceneTools(s, physics_requested=True)
    tools.call('prepare_physics', {'decisions':[{**decision(), 'object_id':second['id']}]})
    result = tools.call('configure_experiment', {'bodies':[
        {'object_id':'cup/one/000', 'toward_object_id':second['id'], 'impulse_magnitude_ns':.01},
        {'object_id':second['id'], 'motion':'static'}]})
    assert result['ok'], result
    vector = result['scenario']['bodies'][0]['impulse_ns']
    assert vector[0] < 0 and vector[1] < 0
    assert sum(v*v for v in vector) == pytest.approx(.01**2)
    assert vector[0]/vector[1] == pytest.approx(1.5)


def test_experiment_table_override_does_not_replace_cached_table():
    tools = SceneTools(prepared_snapshot(), physics_requested=True)
    saved = []
    tools.physics.on_table = saved.append
    result = tools.call('configure_experiment', {'bodies':[{'object_id':'cup/one/000'}],
        'table_material':{'static_friction':.1,'dynamic_friction':.1,'restitution':0,'rationale':'experiment only'}})
    assert result['ok'], result
    assert saved == []
    assert tools.physics.table_material['static_friction'] == .5


def test_short_play_followup_has_physics_tools_and_only_queues_play(monkeypatch):
    import json
    import smartphone_mocap.scene_agent as agent
    responses = iter([
        {'role':'assistant','tool_calls':[{'id':'play','type':'function','function':{'name':'control_physics','arguments':json.dumps({'command':'play'})}}]},
        {'role':'assistant','content':'재생을 요청했습니다.'}])
    def respond(endpoint,payload,timeout):
        assert any(t['function']['name']=='control_physics' for t in payload['tools'])
        return {'choices':[{'message':next(responses)}]}
    monkeypatch.setattr(agent,'_post_json',respond)
    result=agent.sglang_response('이어서 재생해',prepared_snapshot(),base_url='http://unused',model='test',timeout_seconds=1)
    assert result['actions'] == [{'type':'physics_control','command':'play'}]


@pytest.mark.parametrize('prompt', ['물리 실험을 종료하고 관측 모드로 돌아가줘.', '관측으로 복귀해줘.'])
def test_return_to_observation_requires_real_action(monkeypatch, prompt):
    import smartphone_mocap.scene_agent as agent
    responses = iter([
        {'role': 'assistant', 'content': '관측으로 돌아갑니다.'},
        {'role': 'assistant', 'tool_calls': [{'id': 'return', 'type': 'function',
          'function': {'name': 'control_physics', 'arguments': '{"command":"return"}'}}]},
        {'role': 'assistant', 'content': '복귀 요청을 전달했습니다.'},
    ])
    def respond(endpoint, payload, timeout):
        assert any(t['function']['name'] == 'control_physics' for t in payload['tools'])
        return {'choices': [{'message': next(responses)}]}
    monkeypatch.setattr(agent, '_post_json', respond)
    result = agent.sglang_response(prompt, prepared_snapshot(), base_url='http://unused', model='test', timeout_seconds=1)
    assert result['actions'] == [{'type': 'physics_control', 'command': 'return'}]


def test_high_level_push_can_prepare_first_table_without_fake_object():
    s=prepared_snapshot()
    s['twin_state'].pop('physics_table_material')
    other={**s['tracking']['objects'][0],'id':'cup/two/000','xyz':[.1,.1,.2]}
    s['tracking']['objects'].append(other)
    s['twin_state']['runtime_objects'][other['id']]=copy.deepcopy(s['twin_state']['runtime_objects']['cup/one/000'])
    tools=SceneTools(s,physics_requested=True)
    tools.call('prepare_physics',{'decisions':[{**decision(),'object_id':other['id']}]})
    table={'static_friction':.4,'dynamic_friction':.3,'restitution':.2,'rationale':'explicit estimate'}
    result=tools.call('push_toward',{'object_id':'cup/one/000','target_object_id':other['id'],'impulse_ns':.03,'table_material':table})
    assert result['ok'],result
    assert result['scenario']['table_material']==table
    assert result['scenario']['bodies'][0]['impulse_ns'][0]<0


@pytest.mark.parametrize('prompt', [
    'get_experiment_result로 방금 실행한 실험에서 컵끼리 충돌했는지 확인해 줘.',
    '방금 완료한 물리 실험의 실제 결과만 조회해서 알려줘. 시뮬레이션을 새로 준비하거나 재생하지 말고, 실제 이동 변위와 컵끼리 접촉 여부를 확인해줘.',
    '실험을 준비하지 말고 결과만 알려줘.',
    '초기화하지 말고 실험 결과를 확인해줘.',
    "Do not configure or play. Use get_experiment_result for the physics experiment result.",
])
def test_past_execution_result_question_does_not_request_another_run(monkeypatch, prompt):
    import json
    import smartphone_mocap.scene_agent as agent
    s=prepared_snapshot();s['twin_state']['simulation']['result']={'completed':True,'contacts':[]}
    calls=iter([
        # A model ignoring the offered schema must not trigger a mutation.
        {'role':'assistant','tool_calls':[{'id':'unwanted','type':'function','function':{'name':'control_physics','arguments':'{"command":"play"}'}}]},
        {'role':'assistant','tool_calls':[{'id':'result','type':'function','function':{'name':'get_experiment_result','arguments':'{}'}}]},
        {'role':'assistant','content':'컵끼리의 접촉 기록은 없습니다.'}])
    def response(url,payload,timeout):
        assert [t['function']['name'] for t in payload['tools']]==['get_experiment_result']
        return {'choices':[{'message':next(calls)}]}
    monkeypatch.setattr(agent,'_post_json',response)
    result=agent.sglang_response(prompt,s,base_url='http://unused',model='test',timeout_seconds=1)
    assert result['actions']==[]
    assert result['evidence_trace'][0]['status']=='blocked'


def test_measured_distance_is_vector_norm_not_single_axis():
    s=prepared_snapshot()
    s['twin_state']['simulation']['result']={'objects':[{'displacement_m':[.03,.04,0]}]}
    result=SceneTools(s,physics_requested=True).call('get_experiment_result',{})
    body=result['result']['objects'][0]
    assert body['net_displacement_m']==pytest.approx(.05)
    assert body['horizontal_displacement_m']==pytest.approx(.05)
    assert 'net_displacement_m' not in s['twin_state']['simulation']['result']['objects'][0]


@pytest.mark.parametrize("completed,prepared,expected", [
    (True, False, ["reset", "play"]),
    (False, False, ["play"]),
    (True, True, ["play"]),
])
def test_replay_resets_only_completed_existing_experiment(completed, prepared, expected):
    s = prepared_snapshot()
    s["twin_state"]["simulation"]["result"] = {"completed": completed}
    tools = SceneTools(s, physics_requested=True)
    if prepared:
        assert tools.call("configure_experiment", {"bodies": [{"object_id": "cup/one/000"}]})["ok"]
    assert tools.call("control_physics", {"command": "play"})["ok"]
    assert [a["command"] for a in tools.actions if a["type"] == "physics_control"] == expected


@pytest.mark.parametrize("prompt", ["엉 진행해줘", "다시 세팅해줘", "컵도 포함해줘"])
def test_short_action_cannot_claim_completion_without_action(monkeypatch, prompt):
    import smartphone_mocap.scene_agent as agent
    monkeypatch.setattr(agent, "_post_json", lambda *args: {
        "choices": [{"message": {"role": "assistant", "content": "설정 완료했습니다."}}]})
    result = agent.sglang_response(prompt, prepared_snapshot(), base_url="http://unused",
                                  model="test", timeout_seconds=1)
    assert not result["actions"]
    assert "실행하지 않았습니다" in result["reply"]


def test_required_participant_blocks_partial_preview_and_play():
    tools = SceneTools(prepared_snapshot(), physics_requested=True)
    tools.physics.required_object_ids = {"cup/missing/000"}
    result = tools.call("configure_experiment", {"bodies": [{"object_id": "cup/one/000"}]})
    assert not result["ok"]
    assert not tools.call("control_physics", {"command": "play"})["ok"]
    assert tools.actions == []


def test_push_retains_third_participant_without_old_impulse():
    s = prepared_snapshot()
    for key, xyz in [("book/one/000", [.1,.1,.2]), ("plant/one/000", [.6,.1,.2])]:
        s["tracking"]["objects"].append({"id":key, "label":key.split("/")[0], "xyz":xyz, "state":"active"})
        s["twin_state"]["runtime_objects"][key] = copy.deepcopy(s["twin_state"]["runtime_objects"]["cup/one/000"])
    tools = SceneTools(s, physics_requested=True)
    assert tools.call("prepare_physics", {"decisions":[
        {**decision(), "object_id":key} for key in ["book/one/000", "plant/one/000"]]})["ok"]
    first = tools.call("configure_experiment", {"bodies":[
        {"object_id":"cup/one/000", "impulse_ns":[.1,0,0]}]})
    tools.twin_state["simulation"]["initial_bodies"] = {b["object_id"]:b for b in first["scenario"]["bodies"]}
    result = tools.call("push_toward", {"object_id":"book/one/000",
        "target_object_id":"plant/one/000", "impulse_ns":.03})
    assert result["ok"], result
    bodies = {b["object_id"]:b for b in result["scenario"]["bodies"]}
    assert len(bodies) == 3
    assert bodies["cup/one/000"]["impulse_ns"] == [0,0,0]


def test_inclusion_survives_followup_and_blocks_play(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    responses = iter([
        {"role":"assistant","tool_calls":[{"id":"play","type":"function",
          "function":{"name":"control_physics","arguments":'{"command":"play"}'}}]},
        {"role":"assistant","content":"완료했습니다."},
        {"role":"assistant","content":"완료했습니다."},
    ])
    monkeypatch.setattr(agent, "_post_json", lambda *args: {"choices":[{"message":next(responses)}]})
    result = agent.sglang_response("진행해줘", prepared_snapshot(),
        conversation=[{"role":"you","text":"컵도 포함해줘"}],
        base_url="http://unused", model="test", timeout_seconds=1)
    assert not result["actions"]
    assert "실행하지 않았습니다" in result["reply"]


def test_queued_preview_reply_cannot_claim_runtime_completion():
    tools = SceneTools(prepared_snapshot(), physics_requested=True)
    assert tools.call("configure_experiment", {"bodies":[{"object_id":"cup/one/000"}]})["ok"]
    reply = tools.physics.finalize_reply("완료했습니다. 컵이 충돌했습니다.")
    assert "아직 실행 완료가 확인된 것은 아닙니다" in reply
    assert "컵이 충돌했습니다" not in reply


def test_multiple_object_query_accepts_comma_separated_ids_and_labels():
    tools = SceneTools(snapshot(), physics_requested=True)
    assert len(tools._matching("book, cup")) == 1
    assert len(tools._matching("cup/one/000, cup")) == 1
    assert len(tools.call("inspect_physics_inputs", {"query":"cup/one/000, book"})["objects"]) == 1


def test_empty_twin_is_reported_before_model_inference(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    s = snapshot()
    s["twin_state"]["runtime_objects"] = {}
    def unexpected(*args):
        raise AssertionError("Model cannot repair missing USD geometry")
    monkeypatch.setattr(agent, "_post_json", unexpected)
    result = agent.sglang_response("시뮬레이션 세팅해줘", s, base_url="http://unused",
                                  model="test", timeout_seconds=1)
    assert not result["actions"]
    assert "3D 장면" in result["reply"]
    assert result["evidence_trace"][0]["error"] == "empty_runtime_objects"


def test_throw_cannot_fall_back_to_gravity_preview(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    responses = iter([
        {"role":"assistant", "tool_calls":[{"id":"drop","type":"function",
          "function":{"name":"preview_physics_move","arguments":"{}"}}]},
        {"role":"assistant","content":"준비했습니다."},
        {"role":"assistant","content":"준비했습니다."},
    ])
    def respond(endpoint, payload, timeout):
        names = {t["function"]["name"] for t in payload["tools"]}
        assert "plan_experiment" in names
        assert "configure_experiment" not in names
        assert "preview_physics_move" not in names
        assert "preview_physics" not in names
        return {"choices":[{"message":next(responses)}]}
    monkeypatch.setattr(agent, "_post_json", respond)
    result = agent.sglang_response("책을 화분쪽으로 강하게 던지는 시뮬레이션 세팅해줘",
                                  prepared_snapshot(), base_url="http://unused",
                                  model="test", timeout_seconds=1)
    assert not result["actions"]
    assert "plan_experiment" in result["reply"]


def participant_snapshot():
    s = prepared_snapshot()
    for key, label in [("book/one/000", "book"), ("plant/one/000", "potted plant"),
                       ("cup/two/000", "cup")]:
        s["tracking"]["objects"].append({"id":key, "label":label, "xyz":[.7,.5,0], "state":"active"})
        s["twin_state"]["runtime_objects"][key] = copy.deepcopy(s["twin_state"]["runtime_objects"]["cup/one/000"])
    tools = SceneTools(s, physics_requested=True)
    tools.call("prepare_physics", {"decisions":[{**decision(), "object_id":o["id"]} for o in tools.objects]})
    s["twin_state"]["physics_properties"] = copy.deepcopy(tools.physics.decisions)
    s["twin_state"]["simulation"]["initial_bodies"] = copy.deepcopy(tools.physics.decisions)
    return s


@pytest.mark.parametrize("prompt,removed", [
    ("책을 화분에 던지는 실험 세팅해줘", False),
    ("컵은 제외하지 마", False),
    ("컵을 제외해달라고 한 적 없어", False),
    ("don't exclude cups", False),
    ("do not remove cups", False),
    ("컵들을 실험에서 제외해줘", True),
    ("remove cups", True),
])
def test_removal_authority_is_grounded_in_user_prompt(prompt, removed):
    from smartphone_mocap.scene_agent import _participant_contract
    tools = SceneTools(participant_snapshot(), physics_requested=True)
    contract = _participant_contract(tools, prompt)
    assert set(contract["authorized_removal_ids"]) == ({"cup/one/000", "cup/two/000"} if removed else set())
    assert len(contract["required_participant_ids"]) == (2 if removed else 4)


def test_agent_retries_omission_with_explicit_full_list(monkeypatch):
    import json
    import smartphone_mocap.scene_agent as agent
    s = participant_snapshot()
    ids = list(s["twin_state"]["simulation"]["initial_bodies"])
    def call(keys, ident):
        return {"role":"assistant", "tool_calls":[{"id":ident, "type":"function",
            "function":{"name":"configure_experiment", "arguments":json.dumps({
                "bodies":[{"object_id":key} for key in keys]})}}]}
    responses = iter([call(["book/one/000","plant/one/000"], "bad"),
                      call(ids, "fixed"), {"role":"assistant", "content":"요청했습니다."}])
    payloads = []
    def respond(endpoint, payload, timeout):
        payloads.append(copy.deepcopy(payload))
        return {"choices":[{"message":next(responses)}]}
    monkeypatch.setattr(agent, "_post_json", respond)
    result = agent.sglang_response("현재 물리 실험 세팅해줘", s,
                                  base_url="http://unused",model="test",timeout_seconds=1)
    assert len(result["actions"]) == 1
    assert result["evidence_trace"][0]["status"] == "blocked"
    assert {b["object_id"] for b in result["actions"][0]["scenario"]["bodies"]} == set(ids)
    assert "Participant contract violation" in str(payloads[1]["messages"])
    # The physics function itself does not auto-add cups.
    raw = SceneTools(s, physics_requested=True)
    subset = raw.call("configure_experiment", {"bodies":[{"object_id":"book/one/000"}]})
    assert len(subset["scenario"]["bodies"]) == 1
