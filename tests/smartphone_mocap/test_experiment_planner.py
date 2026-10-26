import copy
import json
import math
import pytest
from test_physics_scenario import participant_snapshot
from smartphone_mocap.scene_agent import SceneTools, sglang_response
from smartphone_mocap.experiment_planner import ExperimentPlanner, force_value
from smartphone_mocap.agent_memory import AgentMemory

PROMPT="화분을 테이블에 닿도록 서쪽으로 10cm 움직이고 책을 화분으로 0.1N의 힘으로 1초 동안 움직이는 시뮬레이션 세팅해줘"
def setup(prompt=PROMPT):
    s=participant_snapshot()
    # Separate source and target, and float the target above the tabletop.
    for key,xyz in [("book/one/000",[.8,.4,0]),("plant/one/000",[.5,.5,.15])]:
        s["twin_state"]["simulation"]["initial_bodies"][key]["xyz"]=xyz
    tools=SceneTools(s,physics_requested=True)
    return tools,ExperimentPlanner(tools,prompt)

def changes():
    return [{"object_id":"plant/one/000","translate":{"direction":"west","distance":10,"unit":"cm"},"support":"table"},
            {"object_id":"book/one/000","dynamics":{"kind":"force","magnitude":.1,"toward_object_id":"plant/one/000","duration_s":1}}]

def test_patch_preserves_cups_and_computes_force_after_contact_placement():
    tools,planner=setup()
    original=copy.deepcopy(tools.twin_state)
    result=planner.plan_experiment(changes())
    assert result["ok"]
    bodies={b["object_id"]:b for b in result["scenario"]["bodies"]}
    assert len(bodies)==4
    for key in ("cup/one/000","cup/two/000"):
        ref=original["simulation"]["initial_bodies"][key]
        assert bodies[key]["xyz"]==ref["xyz"]
        assert bodies[key]["mass_kg"]==ref["mass_kg"]
        assert bodies[key]["motion"]==ref.get("motion","dynamic")
        assert bodies[key]["force_n"]==[0,0,0]
    plant,book=bodies["plant/one/000"],bodies["book/one/000"]
    assert plant["xyz"]==pytest.approx([.4,.5,0])
    assert math.hypot(*book["force_n"])==pytest.approx(.1)
    assert book["force_n"][1]/book["force_n"][0]==pytest.approx(-.25)
    assert book["force_duration_s"]==1
    assert book["impulse_ns"]==[0,0,0]
    assert tools.twin_state==original
    assert len(tools.actions)==1 and tools.actions[0]["type"]=="physics_preview"

@pytest.mark.parametrize("mutation",["impulse","wrong_force","wrong_distance","wrong_direction","missing_support","cup_static","raw_xyz"])
def test_invalid_request_changes_do_not_queue_actions(mutation):
    tools,planner=setup();c=changes()
    if mutation=="impulse":c[1]["dynamics"]["kind"]="impulse"
    if mutation=="wrong_force":c[1]["dynamics"]["magnitude"]=10
    if mutation=="wrong_distance":c[0]["translate"]["distance"]=20
    if mutation=="wrong_direction":c[0]["translate"]["direction"]="east"
    if mutation=="missing_support":c[0].pop("support")
    if mutation=="cup_static":c.append({"object_id":"cup/one/000","properties":{"motion":"static"}})
    if mutation=="raw_xyz":c[0]["position_m"]=[0,0,0]
    with pytest.raises(ValueError):planner.plan_experiment(c)
    assert not tools.actions

def test_contact_uses_rotated_scaled_bounds():
    tools,planner=setup()
    ref=tools.twin_state["simulation"]["initial_bodies"]["plant/one/000"]
    ref["rotation_wxyz"]=[math.sqrt(.5),math.sqrt(.5),0,0]
    ref["scale"]=[2,2,2]
    result=planner.plan_experiment(changes())
    assert result["ok"]
    b=next(b for b in result["scenario"]["bodies"] if b["object_id"]=="plant/one/000")
    assert b["xyz"][2]==pytest.approx(.08)
    assert b["scale"]==[2,2,2]

def test_missing_duration_asks_without_any_model_or_actions(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    monkeypatch.setattr(agent,"_post_json",lambda *a:pytest.fail("No inference needed for missing force duration"))
    result=sglang_response(PROMPT.replace("1초 동안 ",""),participant_snapshot(),
                          base_url="http://unused",model="test",timeout_seconds=1)
    assert result["task_state"]=="needs_clarification"
    assert "몇 초" in result["reply"]
    assert result["actions"]==[]
    assert force_value("0.1N의 힘")==.1
    assert force_value("0.1Ns 충격량") is None
    assert force_value("0.1 N s 충격량") is None
    assert force_value("0.1 N·s 충격량") is None

def test_clarification_survives_restart_but_not_new_conversation(tmp_path,monkeypatch):
    import smartphone_mocap.scene_agent as agent
    s=participant_snapshot();s["conversation_id"]="one"
    db=AgentMemory(tmp_path/"memory.sqlite")
    first=PROMPT.replace("1초 동안 ","")
    db.claim("s","r1",first,"simulation",s)
    result=sglang_response(first,s,base_url="http://unused",model="test",timeout_seconds=1,
                          memory=db,session_id="s",request_id="r1")
    db.finish("s","r1",result)
    reopened=AgentMemory(tmp_path/"memory.sqlite")
    assert reopened.pending_experiment("s","one")["missing"]=="force_duration_s"
    assert reopened.pending_experiment("s","two") is None
    def respond(url,payload,timeout):
        assert any("서쪽으로 10cm" in str(m["content"]) and "1초 동안" in str(m["content"]) for m in payload["messages"])
        return {"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"q","type":"function","function":{
            "name":"ask_experiment_question","arguments":json.dumps({"question":"대상을 확인해 주세요.","missing_field":"target"})}}]}}]}
    monkeypatch.setattr(agent,"_post_json",respond)
    answer=sglang_response("1초 동안",s,base_url="http://unused",model="test",timeout_seconds=1,
                          memory=reopened,session_id="s",request_id="r2")
    assert answer["task_state"]=="needs_clarification"
    assert not answer["actions"]

def test_repeated_invalid_call_stops_without_forcing_execution(monkeypatch):
    import smartphone_mocap.scene_agent as agent
    count=[]
    def respond(*args):
        count.append(1)
        return {"choices":[{"message":{"role":"assistant","tool_calls":[{"id":str(len(count)),"type":"function","function":{
            "name":"configure_experiment","arguments":"{}"}}]}}]}
    monkeypatch.setattr(agent,"_post_json",respond)
    result=sglang_response(PROMPT,participant_snapshot(),base_url="http://unused",model="test",timeout_seconds=1)
    assert len(count)==2
    assert not result["actions"]
    assert result["task_state"]=="failed"


def test_throw_assumed_speed_cannot_be_applied_as_impulse():
    tools,planner=setup("책을 화분으로 던지는 실험 세팅해줘")
    with pytest.raises(ValueError,match="unit and assumption"):
        planner.plan_experiment([{"object_id":"book/one/000","dynamics":{
            "kind":"impulse","magnitude":2,"toward_object_id":"plant/one/000"}}],
            assumptions=["2 m/s 속도로 던진다고 가정"])
    assert not tools.actions


def test_korean_distance_suffix_still_constrains_numeric_value():
    tools,planner=setup(PROMPT.replace('10cm', '10 CM가량'))
    c=changes();c[0]['translate']['distance']=20
    with pytest.raises(ValueError,match='distance differs'):
        planner.plan_experiment(c)
    assert not tools.actions
