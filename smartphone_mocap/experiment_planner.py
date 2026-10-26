"""Task-scoped experiment patches, grounded constraints and deterministic compilation."""
from __future__ import annotations
import copy
import math
import re
from .booth_props import korean_names
from .physics_scenario import _number
from .table_contact import resolve_placement

DIRECTIONS = {"east":(1,0,0), "west":(-1,0,0), "north":(0,1,0),
              "south":(0,-1,0), "up":(0,0,1), "down":(0,0,-1)}
FORCE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:N|뉴턴)(?![A-Za-z]|\s*(?:[·⋅*]\s*)?s\b)", re.I)
DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(?:초|seconds?|sec|s)(?:\s*동안)?", re.I)

def force_value(prompt):
    match = FORCE.search(prompt)
    return float(match[1]) if match else None

def duration_value(prompt):
    match = DURATION.search(prompt)
    return float(match[1]) if match else None

def clarification_reply(prompt, question, missing):
    return {"reply":question, "actions":[], "physics_jobs":[],
            "provider":"experiment-planner", "task_state":"needs_clarification",
            "pending_experiment":{"prompt":prompt, "missing":missing},
            "evidence_trace":[{"tool":"experiment_contract","source":"user request",
                               "ok":True,"status":"needs_clarification"}]}

def dynamic_edit_requested(prompt):
    return bool(re.search(r"힘|충격|속도|던지|던져|던지는|밀어|force|impulse|velocity|throw|push|질량|마찰|mass|friction", prompt, re.I)
                and re.search(r"세팅|설정|실행|돌려|준비|줘|해|configure|prepare|run|simulate|apply|set",prompt,re.I)
                and not re.search(r"결과만|결과.*(?:알려|보여)|result.only",prompt,re.I))

class ExperimentPlanner:
    def __init__(self, scene, prompt):
        self.scene, self.prompt = scene, prompt
        self.pending = None
        self.plan = None

    def ask(self, question, missing_field):
        if self.scene.actions:
            raise ValueError("Cannot ask after queuing changes. Ask before planning.")
        if not isinstance(question,str) or not question.strip():
            raise ValueError("Provide a specific question.")
        self.pending = clarification_reply(self.prompt, question, missing_field)
        return {"ok":True, "status":"needs_clarification", "question":question}

    def _mentioned(self, key):
        obj = next((o for o in self.scene.objects if o["id"]==key), {})
        names = [key,obj.get("display_name"),obj.get("label"),*korean_names(obj.get("label"))]
        return any(n and n.casefold() in self.prompt.casefold() for n in names) or key in self.scene.snapshot.get("selection",{}).get("object_ids",[])

    def plan_experiment(self, changes, seconds=3, assumptions=None, table_material=None):
        if table_material is not None and self.scene.physics.table_material is not None:
            if table_material != self.scene.physics.table_material:
                raise ValueError("Prepared table properties must remain unchanged.")
        if assumptions is not None and (not isinstance(assumptions,list) or not all(isinstance(v,str) for v in assumptions)):
            raise ValueError("Assumptions must be a list of explanations.")
        if self.pending:
            return {"ok":False,"error":"Answer the pending clarification before planning."}
        if self.plan or self.scene.actions:
            raise ValueError("Only one atomic experiment plan is allowed per request.")
        if not self.scene.physics.allowed:
            raise ValueError("An explicit experiment request is required.")
        if not isinstance(changes,list) or not changes:
            raise ValueError("Provide the requested changes, not a full body list.")
        ids=[c["object_id"] for c in changes]
        if len(ids)!=len(set(ids)):
            raise ValueError("Combine changes to each object into one entry.")
        simulation=self.scene.twin_state.get("simulation",{})
        initial=copy.deepcopy(simulation.get("initial_bodies",{}))
        baseline=initial or copy.deepcopy(self.scene.physics.decisions)
        # A new experiment includes the currently visible scene; do not silently hide
        # objects whose properties have not yet been prepared.
        participants=set(initial) if initial else {o["id"] for o in self.scene._matching()}
        participants |= set(ids) | self.scene.physics.required_object_ids
        unavailable=participants-set(self.scene.physics.decisions)
        if unavailable:
            raise ValueError("Inspect and prepare missing properties first: "+", ".join(sorted(unavailable)))
        for key in participants:
            baseline.setdefault(key,copy.deepcopy(self.scene.physics.decisions[key]))
        configs={key:{"object_id":key} for key in sorted(participants)}
        requested_force=force_value(self.prompt)
        requires_support=bool(re.search(r"테이블.{0,20}(?:닿|내려|놓)|place.{0,15}table",self.prompt,re.I))
        motions=[]
        diff=[]
        for change in changes:
            unknown=set(change)-{"object_id","translate","support","dynamics","properties"}
            if unknown:
                raise ValueError("Unsupported patch fields: "+", ".join(sorted(unknown)))
            key=change["object_id"]
            if not self._mentioned(key):
                raise ValueError("Unmentioned object must remain unchanged: "+key)
            ref=baseline[key]; config=configs[key]
            xyz=list(ref["xyz"])
            if "translate" in change:
                move=change["translate"]
                if set(move)-{"direction","distance","unit"}:raise ValueError("Unknown translation field")
                direction=move["direction"]
                if direction not in DIRECTIONS:raise ValueError("Use table-axis cardinal direction.")
                distance=_number(move["distance"],"distance",0,100)
                unit=move.get("unit","m")
                if unit not in {"m","cm","mm"}:raise ValueError("Use m, cm or mm.")
                meters=distance*{"m":1,"cm":.01,"mm":.001}[unit]
                # Explicit source measurements constrain model-generated translations.
                specified=re.findall(r"(\d+(?:\.\d+)?)\s*(cm|mm|m|센티미터|미터)(?![A-Za-z])",self.prompt,re.I)
                values=[float(n)*{"cm":.01,"mm":.001,"m":1,"센티미터":.01,"미터":1}[u.lower()] for n,u in specified]
                if values and not any(abs(meters-v)<1e-6 for v in values):
                    raise ValueError("Translation distance differs from the user's specified distance.")
                for label,axis in [("서쪽","west"),("동쪽","east"),("북쪽","north"),("남쪽","south")]:
                    if label in self.prompt and direction!=axis:
                        raise ValueError("Translation direction conflicts with user request.")
                xyz=[xyz[i]+DIRECTIONS[direction][i]*meters for i in range(3)]
                config["position_m"]=xyz
            if "support" in change:
                if change["support"]!="table" or not requires_support:
                    raise ValueError("Table placement requires an explicit contact/placement request.")
                pose={"xyz":list(ref["xyz"]),"rotation_wxyz":ref["rotation_wxyz"],"scale":ref["scale"]}
                resolution=resolve_placement(ref["source_contact_geometry"],pose,{"xyz":xyz},"place_on_table")
                config["position_m"]=resolution["target"]["xyz"]
            if "properties" in change:
                props=change["properties"]
                vocabulary={"mass_kg":r"질량|무게|mass|weight",
                            "static_friction":r"마찰|friction","dynamic_friction":r"마찰|friction",
                            "restitution":r"반발|탄성|restitution","motion":r"고정|정적|동적|static|dynamic"}
                if not props or set(props)-set(vocabulary):raise ValueError("Unsupported experiment property.")
                for name,value in props.items():
                    if not re.search(vocabulary[name],self.prompt,re.I):
                        raise ValueError("Unrequested property change: "+name)
                    config[name]=value
            if "dynamics" in change:
                motions.append((key,change["dynamics"]))
            diff.append(copy.deepcopy(change))
        if requires_support and not any(c.get("support")=="table" for c in changes):
            raise ValueError("Requested table contact is missing from the plan.")
        if re.search(r"(?:서쪽|동쪽|북쪽|남쪽|west|east|north|south).{0,20}\d",self.prompt,re.I) and not any("translate" in c for c in changes):
            raise ValueError("Requested translation is missing from the plan.")
        if requested_force is not None and not motions:
            raise ValueError("Requested force is missing from the plan.")
        if re.search(r"던지|던져|던지는|throw",self.prompt,re.I) and not motions:
            raise ValueError("Throw requires motion; a gravity-only plan is incomplete.")
        for key,motion in motions:
            if set(motion)-{"kind","magnitude","toward_object_id","direction","duration_s"}:
                raise ValueError("Unknown dynamics field.")
            kind=motion["kind"]
            if kind not in {"force","impulse","velocity"}:raise ValueError("Unknown dynamics kind.")
            if re.search(r"충격량|impulse|N\s*[·⋅*]?\s*s\b",self.prompt,re.I) and kind != "impulse":
                raise ValueError("An impulse request must not be replaced with force or velocity.")
            if requested_force is not None and kind!="force":
                raise ValueError("Force in N must not be replaced with impulse or velocity.")
            magnitude=_number(motion["magnitude"],"magnitude",0,100 if kind=="velocity" else 1000)
            if requested_force is not None and abs(magnitude-requested_force)>1e-8:
                raise ValueError("Force magnitude differs from the user's request.")
            if kind != "force" and not re.search(r"\d+(?:\.\d+)?\s*(?:m/s|미터|N\s*[·⋅*]?\s*s)",self.prompt,re.I) and not assumptions:
                raise ValueError("Qualitative motion strength needs an explicit assumption explaining the chosen magnitude and unit.")
            # An illustrative assumption is also a contract: do not describe
            # velocity while issuing an impulse of the same numeric value.
            unit_text = self.prompt + " " + " ".join(assumptions or [])
            for expected, pattern in [
                ("velocity", r"(\d+(?:\.\d+)?)\s*m/s"),
                ("impulse", r"(\d+(?:\.\d+)?)\s*N\s*[·⋅*]?\s*s(?![A-Za-z])")]:
                declared = re.findall(pattern, unit_text, re.I)
                if declared and (kind != expected or not any(abs(magnitude-float(v))<1e-8 for v in declared)):
                    raise ValueError("Dynamics kind/magnitude conflicts with the stated unit and assumption: " + expected)
            if kind=="force":
                duration=duration_value(self.prompt)
                if duration is None:
                    return self.ask("힘을 몇 초 동안 가할까요?", "force_duration_s")
                if abs(_number(motion.get("duration_s",duration),"duration_s",.001,10)-duration)>1e-8:
                    raise ValueError("Force duration differs from the user's request.")
                seconds=max(float(seconds),duration)
            if motion.get("toward_object_id"):
                target=motion["toward_object_id"]
                if target not in configs or target==key:raise ValueError("Choose another participant as the target.")
                if not self._mentioned(target):raise ValueError("Unrequested direction target.")
                origin=configs[key].get("position_m",baseline[key]["xyz"])
                dest=configs[target].get("position_m",baseline[target]["xyz"])
                delta=[dest[0]-origin[0],dest[1]-origin[1],0]
                length=math.hypot(*delta)
                if length<1e-6:raise ValueError("Coincident objects have no horizontal direction.")
                vector=[v*magnitude/length for v in delta]
            else:
                direction=motion.get("direction")
                if direction not in DIRECTIONS:raise ValueError("Specify target object or cardinal direction.")
                if re.search(r"화분.{0,8}(?:으로|쪽|향)|toward",self.prompt,re.I):
                    raise ValueError("Use toward_object_id for the requested object target.")
                vector=[v*magnitude for v in DIRECTIONS[direction]]
            # Replacing the driven body's motion is explicit in this patch;
            # surrounding objects' motion and properties are untouched.
            configs[key].update({n:[0,0,0] for n in ["velocity_m_s","angular_velocity_rad_s","impulse_ns","force_n","torque_nm"]})
            configs[key].update(motion="dynamic",force_duration_s=duration if kind=="force" else 0)
            configs[key][{"force":"force_n","impulse":"impulse_ns","velocity":"velocity_m_s"}[kind]]=vector
        result=self.scene.physics.configure_experiment(list(configs.values()),seconds=seconds,table_material=table_material)
        if result.get("ok"):
            self.plan={"state":"queued_for_isaac","scope":"experiment","changes":diff,
                       "participant_ids":sorted(participants),"assumptions":assumptions or [],
                       "source_snapshot_id":self.scene.snapshot.get("snapshot_id")}
            result["plan"]=self.plan
        return result

def schema(name,description,properties,required):
    return {"type":"function","function":{"name":name,"description":description,
            "parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":False}}}

EXPERIMENT_SCHEMAS=[
 schema("plan_experiment",
        "Atomically PATCH the current experiment. Send ONLY user-requested changes. Other objects, mass, friction and motion remain unchanged. Code computes final target direction and contact height. No raw XYZ/vectors. Queues paused; control_physics(play) runs after validation.",
        {"changes":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
            "object_id":{"type":"string"},
            "translate":{"type":"object","additionalProperties":False,"properties":{
                "direction":{"type":"string","enum":list(DIRECTIONS)},"distance":{"type":"number"},
                "unit":{"type":"string","enum":["m","cm","mm"]}},"required":["direction","distance","unit"]},
            "support":{"type":"string","enum":["table"]},
            "dynamics":{"type":"object","additionalProperties":False,"properties":{
                "kind":{"type":"string","enum":["force","impulse","velocity"]},"magnitude":{"type":"number"},
                "toward_object_id":{"type":"string"},"direction":{"type":"string","enum":list(DIRECTIONS)},
                "duration_s":{"type":"number"}},"required":["kind","magnitude"]},
            "properties":{"type":"object","additionalProperties":False,"properties":{
                **{k:{"type":"number"} for k in ["mass_kg","static_friction","dynamic_friction","restitution"]},
                "motion":{"type":"string","enum":["static","dynamic"]}}}
            },"required":["object_id"]}},
         "table_material":{"type":"object","description":"Only for the FIRST experiment if no table material is prepared. Existing table values must not change.",
            "properties":{**{k:{"type":"number"} for k in ["static_friction","dynamic_friction","restitution"]},"rationale":{"type":"string"}},
            "required":["static_friction","dynamic_friction","restitution","rationale"],"additionalProperties":False},
         "seconds":{"type":"number"},"assumptions":{"type":"array","items":{"type":"string"}}},["changes"]),
 schema("ask_experiment_question","Ask for missing or ambiguous experiment conditions BEFORE queuing actions. This is a valid needs_clarification outcome, not execution failure.",
        {"question":{"type":"string"},"missing_field":{"type":"string"}},["question","missing_field"])
]
