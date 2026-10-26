"""Agent-decided physics with deterministic validation and isolated jobs.

No class-to-mass defaults. Geometry is snapshot-bound; all material estimates
come from the agent's explicit decision, with assumptions and provenance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = ROOT / "outputs/tabletop_physics"
_WORKER_LOCK = threading.Lock()

PHYSICS_POLICY = """
Physics is on-demand and NEVER changes live mirrored objects. For an explicit
what-if/simulation request, inspect_physics_inputs and inspect_images first when
visual material/shape evidence is needed. YOU infer material, mass or density,
fill fraction, friction, restitution and collider; there is no class preset.
Use units SI, explain hollow/filled assumptions, uncertainty and evidence. Never
claim a guessed mass is measured, or fabricate a web search/reference. Dimensions
are current USD geometry bounds, not measured real-world dimensions. Simple box,
cylinder or sphere colliders are intentional proxies, not detailed mesh physics.
Call prepare_physics with one decision per selected object; use stable IDs from
inspection. Then simulate_what_if to release the selected bodies under gravity in
an isolated table scene. Include supporting/nearby objects needed for the question;
omitted objects cannot collide. Supply table friction/restitution explicitly.
Result is asynchronous: a queued job is NOT a successful simulation. Use
get_physics_result for facts, or give the job ID and let the user ask later.
Do not apply live move/scale/rotate tools merely to create a counterfactual.
Supported experiments include release, velocity, impulse and finite-duration force.
Liquid and deformation are not supported.
"""


def simulation_requested(prompt: str) -> bool:
    return bool(re.search(r"시뮬|물리|what[ -]?if|simulat|physics|놓으면|떨어뜨리면|넘어질|떨어질", prompt, re.I))


def _number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not boolean")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a nonempty explanation")
    return value.strip()[:4000]


def geometry_input(item):
    points = item.get("contact_geometry", {}).get("points_local_m", [])
    if not points:
        raise ValueError("USD geometry bounds are unavailable; do not invent dimensions")
    if len(points) > 100000 or any(len(p) != 3 for p in points):
        raise ValueError("Invalid geometry bounds")
    scale = item.get("scale", [1, 1, 1])
    xyz = item.get("xyz")
    rotation = item.get("rotation_wxyz", [1, 0, 0, 0])
    if not xyz or len(xyz) != 3 or len(scale) != 3 or len(rotation) != 4:
        raise ValueError("Complete position, rotation and scale required")
    scale = [_number(v, "scale", .001, 1000) for v in scale]
    points = [[_number(p[i], "bounds", -100, 100)*scale[i] for i in range(3)] for p in points]
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    size = [_number(high[i]-low[i], "proxy size (m)", .001, 5) for i in range(3)]
    pose = {"xyz": [_number(v, "xyz", -10, 10) for v in xyz],
            "rotation_wxyz": [_number(v, "rotation", -1, 1) for v in rotation]}
    norm = math.hypot(*pose["rotation_wxyz"])
    if norm < 1e-8:
        raise ValueError("Invalid zero quaternion")
    pose["rotation_wxyz"] = [v/norm for v in pose["rotation_wxyz"]]
    value = {"object_id": str(item["id"]), "label": item.get("label"),
             "asset_uri": item.get("asset_uri"), "size_m": size,
             "center_local_m": [(low[i]+high[i])/2 for i in range(3)],
             **pose, "scale": scale, "geometry_source": "USD conservative bounds; not measured physical dimensions",
             "source_contact_geometry": copy.deepcopy(item.get("contact_geometry", {}))}
    value["geometry_digest"] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value


def validate_decision(decision, geometry):
    material = _text(decision.get("material"), "material")
    collider = decision.get("collider")
    if collider not in {"box", "cylinder", "sphere"}:
        raise ValueError("Supported proxy colliders: box, cylinder (local Z), sphere")
    size = geometry["size_m"]
    volume = math.prod(size)
    if collider == "cylinder":
        volume = math.pi*(max(size[:2])/2)**2*size[2]
    elif collider == "sphere":
        volume = 4/3*math.pi*(max(size)/2)**3
    density = decision.get("density_kg_m3")
    fill = decision.get("fill_fraction")
    mass = decision.get("mass_kg")
    if density is not None:
        density = _number(density, "density_kg_m3", .001, 50000)
    if fill is not None:
        fill = _number(fill, "fill_fraction", 0 if mass is not None else .000001, 1)
    if mass is None:
        density = _number(density, "density_kg_m3", .001, 50000)
        fill = _number(fill, "fill_fraction", .000001, 1)
        mass = density*volume*fill
        basis = "agent density * proxy volume * agent fill fraction"
    else:
        basis = "agent mass estimate (not measured unless supplied evidence establishes it)"
    warnings = ["agent_estimated_not_measured", "simple_proxy_collider",
                "uniform_proxy_inertia_and_center_of_mass", "conditional_simulation_only"]
    if decision.get("mass_kg") is not None and (density is not None or fill is not None):
        warnings.append("explicit_mass_used_density_and_fill_not_used_for_dynamics")
    mass = _number(mass, "mass_kg", .000001, 1000)
    static = _number(decision.get("static_friction"), "static_friction", 0, 10)
    dynamic = _number(decision.get("dynamic_friction"), "dynamic_friction", 0, static)
    assumptions = decision.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions:
        raise ValueError("At least one assumption is required")
    return {**geometry, "material": material, "collider": collider, "mass_kg": mass,
            "mass_basis": basis, "density_kg_m3": density, "fill_fraction": fill,
            "static_friction": static, "dynamic_friction": dynamic,
            "restitution": _number(decision.get("restitution"), "restitution", 0, 1),
            "confidence": _number(decision.get("confidence"), "confidence", 0, 1),
            "rationale": _text(decision.get("rationale"), "rationale"),
            "evidence": _text(decision.get("evidence"), "evidence"),
            "assumptions": [_text(a, "assumption") for a in assumptions],
            "warnings": warnings}


def _write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def submit_job(scenario):
    launcher = Path(os.environ.get("ISAAC_SIM_DIR", str(Path.home() / "isaac-sim-standalone-6.0.1-linux-aarch64"))) / "python.sh"
    if not launcher.is_file():
        return {"ok": False, "error": "Isaac runtime unavailable; set ISAAC_SIM_DIR"}
    if not _WORKER_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "A physics job is already running; wait for its result"}
    job_id = uuid.uuid4().hex
    folder = JOB_ROOT / job_id
    try:
        folder.mkdir(parents=True, exist_ok=False)
        _write_json(folder / "request.json", scenario)
        _write_json(folder / "status.json", {"status": "queued", "job_id": job_id})
    except Exception:
        _WORKER_LOCK.release()
        raise

    def run():
        try:
            _write_json(folder / "status.json", {"status": "running", "job_id": job_id})
            with (folder / "worker.log").open("w") as log:
                process = subprocess.Popen([str(launcher), str(ROOT / "tools/run_tabletop_physics.py"),
                    "--job", str(folder)], stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, cwd=ROOT)
                try:
                    returncode = process.wait(timeout=150)
                except subprocess.TimeoutExpired:
                    # Only the process group created for this job, never an
                    # existing Isaac application or another user's worker.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError("Physics worker exceeded its 150-second budget")
            if returncode != 0 or not (folder / "result.json").is_file():
                if (folder / "failure.json").is_file():
                    raise RuntimeError(json.loads((folder / "failure.json").read_text())["error"])
                raise RuntimeError(f"Physics worker failed (exit {returncode}); see worker.log")
            _write_json(folder / "status.json", {"status": "completed", "job_id": job_id})
        except Exception as exc:
            _write_json(folder / "status.json", {"status": "failed", "job_id": job_id, "error": str(exc)})
        finally:
            _WORKER_LOCK.release()

    threading.Thread(target=run, name="tabletop-physics-job", daemon=True).start()
    return {"ok": True, "status": "queued", "job_id": job_id,
            "source_unchanged": True, "result_directory": str(folder)}


def get_job(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", str(job_id)):
        raise ValueError("Invalid physics job ID")
    folder = JOB_ROOT / job_id
    if not (folder / "status.json").is_file():
        return {"ok": False, "error": "Unknown physics job"}
    status = json.loads((folder / "status.json").read_text())
    # Recover durable results if a service/CLI exited after the worker finished.
    if status["status"] in {"running", "queued"}:
        if (folder / "result.json").is_file():
            status["status"] = "completed"
        elif (folder / "failure.json").is_file():
            status.update(status="failed", error=json.loads((folder/"failure.json").read_text())["error"])
    if status["status"] == "completed":
        status["result"] = json.loads((folder / "result.json").read_text())
    return {"ok": True, **status}


class PhysicsScenarioTools:
    def __init__(self, scene, allowed=False):
        self.scene, self.allowed = scene, allowed
        self.decisions = {}
        self.jobs = []
        self.on_prepared = None
        self.on_table = None
        self.experiment_only = False
        self.required_object_ids = set()
        self.material_facts = {f["object_id"]:str(f["value"]) for f in scene.twin_state.get("object_facts", []) if f.get("field") == "material"}
        self.table_material = copy.deepcopy(scene.twin_state.get("physics_table_material"))
        for item in scene._matching():
            cached = scene.twin_state.get("physics_properties", {}).get(item.get("id"))
            if not cached:
                continue
            try:
                geometry = geometry_input(item)
                # Pose changes do not invalidate material/mass. A changed asset
                # or scaled shape does require a fresh decision.
                if all(geometry.get(k) == cached.get(k) for k in
                       ("asset_uri", "scale", "size_m", "center_local_m")):
                    self.decisions[item["id"]] = {**copy.deepcopy(cached), **geometry}
            except (ValueError, TypeError):
                continue

    def inspect_physics_inputs(self, query=""):
        result = []
        for item in self.scene._matching(query):
            try:
                result.append({"available": True, **geometry_input(item),
                               "prepared_properties": self.decisions.get(item["id"]),
                               "experiment_initial": {k:v for k,v in self.scene.twin_state.get("simulation", {}).get("initial_bodies", {}).get(item["id"], {}).items()
                                   if k in {"xyz", "motion", "mass_kg", "velocity_m_s", "impulse_ns", "force_n", "torque_nm", "force_duration_s"}}})
            except ValueError as exc:
                result.append({"object_id": item.get("id"), "available": False, "error": str(exc)})
        return {"objects": result, "on_demand_authorized": self.allowed,
                "supported_colliders": ["box", "cylinder", "sphere"],
                "material_defaults": None, "prepared_table_material": self.table_material,
                "physics_applied_to_live_scene": False}

    def prepare_physics(self, decisions, update_existing=False):
        if not self.allowed:
            return {"ok": False, "error": "Explicit simulation/physics request required"}
        if not isinstance(decisions, list) or not 1 <= len(decisions) <= 20:
            raise ValueError("Provide 1..20 physics decisions")
        accepted = {}
        for decision in decisions:
            if not isinstance(decision, dict):
                raise ValueError("Each physics decision must be an object")
            target = str(decision.get("object_id", ""))
            matches = [o for o in self.scene._matching() if o.get("id") == target]
            if len(matches) != 1 or target in accepted:
                raise ValueError("Each decision needs one unique inspected stable object ID")
            material_fact = self.material_facts.get(target)
            if material_fact and not self.experiment_only and str(decision.get("material", "")).strip().casefold() != material_fact.strip().casefold():
                raise ValueError(f"Use the explicit user material value {material_fact!r} for {target}; do not replace it with an estimate")
            if target in self.decisions and not update_existing:
                accepted[target] = self.decisions[target]
            else:
                accepted[target] = validate_decision(decision, geometry_input(matches[0]))
        if self.on_prepared:
            versions = self.on_prepared(accepted) or {}
            for key, version in versions.items():
                accepted[key]["property_version"] = version
        self.decisions.update(accepted)
        return {"ok": True, "decisions": list(accepted.values()), "status": "prepared_not_simulated"}

    def _scenario(self, object_ids, table_material, seconds=3.0, gravity_m_s2=9.81, *, persist_table=True):
        if not self.allowed:
            return {"ok": False, "error": "Explicit simulation request required"}
        if not isinstance(object_ids, list) or not 1 <= len(object_ids) <= 20 or len(set(object_ids)) != len(object_ids):
            raise ValueError("Provide unique selected object IDs")
        missing = self.required_object_ids - set(object_ids)
        if missing:
            raise ValueError("Requested participants missing: " + ", ".join(sorted(missing)))
        if table_material is None:
            table_material = self.table_material
        if not isinstance(table_material, dict):
            raise ValueError("First experiment requires table_material with static_friction, dynamic_friction, restitution, and rationale. Supply explicit estimates; no table material has been saved yet.")
        items = {o["id"]:o for o in self.scene._matching()}
        bodies = []
        for key in object_ids:
            if key not in items or key not in self.decisions:
                raise ValueError(f"Inspect and prepare physics for {key} first")
            current = geometry_input(items[key])
            if current["geometry_digest"] != self.decisions[key]["geometry_digest"]:
                raise ValueError("Object pose/shape changed; inspect and prepare again")
            bodies.append(self.decisions[key])
        static = _number(table_material.get("static_friction"), "table static friction", 0, 10)
        surface = {"static_friction":static,
                   "dynamic_friction": _number(table_material.get("dynamic_friction"), "table dynamic friction", 0, static),
                   "restitution":_number(table_material.get("restitution"), "table restitution", 0, 1),
                   "rationale":_text(table_material.get("rationale"), "table rationale")}
        bounds = self.scene.snapshot.get("interaction_frame", {}).get("workspace_bounds_xy")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError("Snapshot must provide table XY bounds")
        bounds = [_number(v, "table bounds", -10, 10) for v in bounds]
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("Invalid table bounds")
        scenario = {"schema":"gist.tabletop_physics.v1", "snapshot_id":self.scene.snapshot.get("snapshot_id"),
                    "bodies":bodies, "table_bounds_xy":bounds, "table_material":surface,
                    "seconds":_number(seconds, "seconds", .1, 10),
                    "gravity_m_s2":_number(gravity_m_s2, "gravity", 0, 30),
                    "omitted_object_ids":sorted(set(items)-set(object_ids)), "action":"release",
                    "source_unchanged":True, "properties_scope":"experiment" if self.experiment_only else "object"}
        if self.on_table and persist_table:
            self.on_table(surface)
        return scenario

    def preview_physics(self, object_ids, table_material=None, release_height_m=0.0, placements=None):
        scenario = self._scenario(object_ids, table_material)
        if scenario.get("ok") is False:
            return scenario
        lift = _number(release_height_m, "release_height_m", 0, 2)
        scenario = json.loads(json.dumps(scenario))
        targets = {}
        for placement in placements or []:
            key = placement["object_id"]
            if key not in object_ids or key in targets or len(placement["xyz"]) != 3:
                raise ValueError("Placements require unique selected IDs and three coordinates")
            targets[key] = [_number(v, "experiment position", -100, 100) for v in placement["xyz"]]
        for body in scenario["bodies"]:
            if body["object_id"] in targets:
                body["xyz"] = targets[body["object_id"]]
            reference = self.scene.twin_state.get("simulation", {}).get("release_origins", {}).get(body["object_id"])
            if reference is not None and body["object_id"] not in targets:
                body["xyz"] = list(reference)
            body["release_origin_xyz"] = list(body["xyz"])
            body["xyz"][2] += lift
        self.scene.actions.append({"type": "physics_preview", "scenario": scenario})
        return {"ok": True, "status": "queued_for_isaac", "release_height_m": lift,
                "note": "Queued preview. If the user requested execution, call control_physics(command=play) now. Do not ask again. Otherwise remain paused. Source stays unchanged."}

    def preview_physics_move(self, query, direction, distance_m, table_material=None, direction_frame="table"):
        plan = self.scene._plan_move(query, direction, distance_m, direction_frame)
        if not plan.get("ok"):
            return plan
        if plan["object_id"] not in self.decisions:
            return {"ok": False, "error": "Inspect and prepare physics for the selected object first"}
        result = self.preview_physics(list(self.decisions), table_material,
                                      placements=[{"object_id": plan["object_id"], "xyz": plan["preview_xyz"]}])
        return {**result, "movement": plan, "experiment": "relocate then release under gravity; not a push"}

    def configure_experiment(self, bodies, seconds=3.0, gravity_m_s2=9.81, table_material=None):
        """Atomic experiment configuration; overrides never update object properties."""
        if not isinstance(bodies, list) or not bodies:
            raise ValueError("Choose at least one body")
        ids = [b["object_id"] for b in bodies]
        if len(ids) != len(set(ids)):
            raise ValueError("Each object may appear only once")
        scenario = self._scenario(ids, table_material, seconds, gravity_m_s2, persist_table=False)
        if scenario.get("ok") is False:
            return scenario
        scenario = copy.deepcopy(scenario)
        scenario["properties_scope"] = "experiment"
        initial = self.scene.twin_state.get("simulation", {}).get("initial_bodies", {})
        def vector(value, name, limit):
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(name + " requires three table-axis components")
            return [_number(v, name, -limit, limit) for v in value]
        for body, config in zip(scenario["bodies"], bodies):
            reference = initial.get(body["object_id"], body)
            body["xyz"] = list(reference["xyz"])
            body["rotation_wxyz"] = list(reference["rotation_wxyz"])
            body["scale"] = list(reference["scale"])
            if "position_m" in config:
                body["xyz"] = vector(config["position_m"], "position_m", 100)
            config = dict(config)
            if "toward_object_id" in config:
                target_id = config["toward_object_id"]
                if target_id not in ids or target_id == body["object_id"]:
                    raise ValueError("Direction target must be another included collision participant")
                target_index = ids.index(target_id)
                target_body = initial.get(target_id, scenario["bodies"][target_index])
                target_xyz = vector(bodies[target_index].get("position_m", target_body["xyz"]), "target_position", 100)
                dx, dy = target_xyz[0]-body["xyz"][0], target_xyz[1]-body["xyz"][1]
                distance = math.hypot(dx, dy)
                if distance < 1e-6:
                    raise ValueError("Cannot resolve horizontal direction for coincident positions")
                if not any(k in config for k in ("impulse_magnitude_ns", "speed_m_s")):
                    raise ValueError("Specify impulse_magnitude_ns or speed_m_s for a direction target")
                for scalar, name, limit in [("impulse_magnitude_ns", "impulse_ns", 1000), ("speed_m_s", "velocity_m_s", 100)]:
                    if scalar in config:
                        if name in config:
                            raise ValueError("Choose a magnitude toward a target OR an explicit vector")
                        magnitude = _number(config[scalar], scalar, 0, limit)
                        config[name] = [magnitude*dx/distance, magnitude*dy/distance, 0.0]
            elif any(k in config for k in ("impulse_magnitude_ns", "speed_m_s")):
                raise ValueError("A scalar push requires toward_object_id")
            motion = config.get("motion", reference.get("motion", "dynamic"))
            if motion not in {"dynamic", "static"}:
                raise ValueError("motion must be dynamic or static")
            body["motion"] = motion
            for name, low, high in [("mass_kg", .000001, 1000), ("static_friction", 0, 10),
                                    ("dynamic_friction", 0, 10), ("restitution", 0, 1)]:
                body[name] = _number(config.get(name, reference.get(name, body[name])), name, low, high)
            if body["dynamic_friction"] > body["static_friction"]:
                raise ValueError("dynamic friction cannot exceed static friction")
            for name, limit in [("velocity_m_s", 100), ("angular_velocity_rad_s", 1000),
                                ("impulse_ns", 1000), ("force_n", 10000), ("torque_nm", 10000)]:
                body[name] = vector(config.get(name, reference.get(name, [0, 0, 0])), name, limit)
            body["force_duration_s"] = _number(config.get("force_duration_s", reference.get("force_duration_s", 0)), "force_duration_s", 0, scenario["seconds"])
            if any(body["force_n"] + body["torque_nm"]) and not body["force_duration_s"]:
                raise ValueError("Nonzero force/torque needs force_duration_s")
            if motion == "static" and any(body[k] != [0,0,0] for k in
                    ("velocity_m_s", "angular_velocity_rad_s", "impulse_ns", "force_n", "torque_nm")):
                raise ValueError("A static obstacle cannot receive motion or forces")
        scenario["action"] = "interaction"
        if self.on_table and self.table_material is None:
            self.on_table(scenario["table_material"])
        self.scene.actions.append({"type":"physics_preview", "scenario":scenario})
        return {"ok":True, "status":"queued_for_isaac", "scenario":scenario,
                "note":"Prepared only. Run with control_physics(play). All vectors use table axes; impulse and force act at center of mass. Overrides apply only to this experiment."}

    def push_toward(self, object_id, target_object_id, impulse_ns, seconds=3.0, target_motion="static", table_material=None, additional_object_ids=None):
        """High-level push: required target, scalar impulse, cached properties."""
        retained = self.scene.twin_state.get("simulation", {}).get("initial_bodies", {})
        extras = set(retained) | self.required_object_ids | set(additional_object_ids or [])
        extras -= {object_id, target_object_id}
        return self.configure_experiment([
            {"object_id":object_id, "motion":"dynamic", "toward_object_id":target_object_id,
             "impulse_magnitude_ns":impulse_ns, "velocity_m_s":[0,0,0],
             "angular_velocity_rad_s":[0,0,0], "force_n":[0,0,0], "torque_nm":[0,0,0], "force_duration_s":0},
            {"object_id":target_object_id, "motion":target_motion, "impulse_ns":[0,0,0],
             "velocity_m_s":[0,0,0], "angular_velocity_rad_s":[0,0,0],
             "force_n":[0,0,0], "torque_nm":[0,0,0], "force_duration_s":0}
        ] + [{"object_id":key, "impulse_ns":[0,0,0], "velocity_m_s":[0,0,0],
              "angular_velocity_rad_s":[0,0,0], "force_n":[0,0,0], "torque_nm":[0,0,0],
              "force_duration_s":0} for key in sorted(extras)],
            seconds=seconds, table_material=table_material)

    def get_experiment_result(self):
        if any(a.get("type") == "physics_preview" or
               (a.get("type") == "physics_control" and a.get("command") in {"play", "reset", "return"})
               for a in self.scene.actions):
            return {"ok":False, "status":"awaiting_isaac_execution",
                    "note":"This request queued a new experiment or execution. Its result does not exist in this snapshot; ask again after Isaac applies it."}
        result = copy.deepcopy(self.scene.twin_state.get("simulation", {}).get("result"))
        if result is not None:
            for body in result.get("objects", []):
                delta = body.get("displacement_m")
                if isinstance(delta, list) and len(delta) == 3:
                    body["net_displacement_m"] = math.hypot(*delta)
                    body["horizontal_displacement_m"] = math.hypot(*delta[:2])
        return {"ok":result is not None, "result":result,
                "note":"Measured runtime snapshot, not a prediction. Contact events are from PhysX."}

    def control_physics(self, command):
        if command not in {"play", "pause", "reset", "return"}:
            raise ValueError("Use play, pause, reset or return")
        if not self.allowed:
            return {"ok": False, "error": "Explicit physics request required"}
        simulation = self.scene.twin_state.get("simulation", {})
        if command == "play" and self.required_object_ids:
            previews = [a["scenario"] for a in self.scene.actions if a.get("type") == "physics_preview"]
            participants = ({b["object_id"] for b in previews[-1]["bodies"]} if previews
                            else set(simulation.get("initial_bodies", {})))
            missing = self.required_object_ids - participants
            if missing:
                return {"ok":False, "error":"Configure missing participants before play: " + ", ".join(sorted(missing))}
        fresh = any(a.get("type") == "physics_preview" or
                    (a.get("type") == "physics_control" and a.get("command") == "reset")
                    for a in self.scene.actions)
        completed = simulation.get("completed") or (simulation.get("result") or {}).get("completed")
        if command == "play" and completed and not fresh:
            self.scene.actions.append({"type": "physics_control", "command": "reset"})
        self.scene.actions.append({"type": "physics_control", "command": command})
        return {"ok": True, "status": "queued_for_isaac"}

    def simulate_what_if(self, object_ids, table_material, seconds=3.0, gravity_m_s2=9.81):
        scenario = self._scenario(object_ids, table_material, seconds, gravity_m_s2)
        if scenario.get("ok") is False:
            return scenario
        result = submit_job(scenario)
        if result.get("ok"):
            self.jobs.append(result)
        return result

    def get_physics_result(self, job_id):
        return get_job(job_id)

    def finalize_reply(self, text, *, korean=True):
        """Execution status is a runtime fact, not a model-generated verdict."""
        if not self.jobs:
            actions = [a for a in self.scene.actions if a.get("type") in {"physics_preview", "physics_control"}]
            if not actions:
                return text
            labels = {"play":"재생", "pause":"일시정지", "reset":"실험 초기화", "return":"관측 복귀"}
            steps = ["실험 준비" if a["type"] == "physics_preview" else labels[a["command"]] for a in actions]
            reply = "Isaac에 " + " → ".join(steps) + " 요청을 전달합니다. 아직 실행 완료가 확인된 것은 아닙니다."
            previews = [a["scenario"] for a in actions if a["type"] == "physics_preview"]
            if previews:
                reply += "\n참가 객체: " + ", ".join(b["object_id"] for b in previews[-1]["bodies"])
            return reply
        lines = ["물리 속성을 에이전트가 추정하여 별도 실험을 제출했습니다. 추정치이지 실측값은 아닙니다."
                 if korean else "Submitted an isolated experiment using agent-estimated, not measured, properties."]
        for job in self.jobs:
            status = get_job(job["job_id"])
            label = {"queued":"대기 중", "running":"실행 중", "completed":"계산 완료", "failed":"실행 실패"}.get(status.get("status"),"알 수 없음") if korean else status.get("status","unknown")
            lines.append(f"{job['job_id']}: {label}")
            if status.get("error"):
                lines.append(status["error"])
        for item in self.decisions.values():
            lines.append(f"{item['object_id']}: {item['material']}, {item['mass_kg']:.4g} kg, {item['collider']}")
            lines.append(item["rationale"])
        lines.append("원본 장면은 변경하지 않았습니다. 작업 ID로 결과 조회를 요청할 수 있습니다."
                     if korean else "The original scene is unchanged. Ask for the result using the job ID.")
        return "\n".join(lines)


_num = {"type":"number"}
_text_schema = {"type":"string"}
_decision = {"type":"object", "properties":{
    "object_id":_text_schema, "material":_text_schema,
    "collider":{"type":"string","enum":["box","cylinder","sphere"]},
    "mass_kg":_num, "density_kg_m3":_num, "fill_fraction":_num,
    "static_friction":_num, "dynamic_friction":_num, "restitution":_num,
    "confidence":_num, "rationale":_text_schema, "evidence":_text_schema,
    "assumptions":{"type":"array","items":_text_schema}},
    "required":["object_id","material","collider","static_friction","dynamic_friction","restitution","confidence","rationale","evidence","assumptions"]}
PHYSICS_SCHEMAS = [
    {"type":"function", "function":{"name":name,"description":description,
     "parameters":{"type":"object","properties":properties,"required":required}}}
    for name,description,properties,required in [
        ("inspect_physics_inputs","Inspect snapshot geometry and IDs before estimating physics.",{"query":_text_schema},[]),
        ("prepare_physics","Agent supplies material estimates and assumptions; validates only, no live changes. Supply mass_kg OR density_kg_m3 with fill_fraction.",{"decisions":{"type":"array","items":_decision}},["decisions"]),
        ("simulate_what_if","Queue isolated gravity release/settle of prepared bodies. No live scene changes. Omitted bodies do not collide.",
         {"object_ids":{"type":"array","items":_text_schema},"seconds":_num,"gravity_m_s2":_num,
          "table_material":{"type":"object","properties":{"static_friction":_num,"dynamic_friction":_num,"restitution":_num,"rationale":_text_schema},"required":["static_friction","dynamic_friction","restitution","rationale"]}},["object_ids","table_material"]),
        ("get_physics_result","Read actual asynchronous status/results; queued is not success.",{"job_id":_text_schema},["job_id"])
    ]
]
PHYSICS_SCHEMAS += [
    {"type": "function", "function": {"name": "preview_physics",
     "description": "Open prepared bodies in an isolated viewport physics layer, paused. Optional release_height_m lifts selected bodies for a drop experiment. Table and ground static. Requires inspect and prepare first.",
     "parameters": {"type": "object", "properties": {
         "object_ids": {"type": "array", "items": _text_schema},
         "table_material": PHYSICS_SCHEMAS[2]["function"]["parameters"]["properties"]["table_material"],
         "release_height_m": _num}, "required": ["object_ids", "table_material"]}}},
    {"type": "function", "function": {"name": "control_physics",
     "description": "Control viewport physics: play, pause, reset to initial experiment pose, return to observation. Execution acknowledged by Isaac, not by queueing.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "enum": ["play", "pause", "reset", "return"]}}, "required": ["command"]}}}
]
PHYSICS_POLICY += "\nFor a visible interactive experiment use preview_physics after prepare_physics; this now supports main-viewport release simulation. Use control_physics for play/pause/reset/return. Check Isaac execution feedback. Live tracking must not overwrite simulation. Do not claim collision-free until preview preflight succeeds.\n"

PHYSICS_SCHEMAS.append({"type": "function", "function": {
    "name": "preview_physics_move",
    "description": "Prepare an unrestricted horizontal relocation then gravity release in the viewport experiment. Can leave the table and fall to the ground. Replaces an existing experiment; never moves observation objects. Requires inspect_physics_inputs and prepare_physics. Use control_physics play to run.",
    "parameters": {"type": "object", "properties": {
        "query": _text_schema, "direction": _text_schema, "distance_m": _num,
        "direction_frame": {"type": "string", "enum": ["table", "viewport"]},
        "table_material": PHYSICS_SCHEMAS[2]["function"]["parameters"]["properties"]["table_material"]
    }, "required": ["query", "direction", "distance_m", "table_material"]}
}})
PHYSICS_POLICY += """
For 'move south 1m and simulate', inspect/prepare then use preview_physics_move
with direction=south, distance_m=1, direction_frame=table. It relocates ONLY
experiment bodies and releases them under gravity. There is no tabletop boundary
clamp or invisible wall. Off-table bodies can fall to ground. This is relocation
then release, not a force/velocity push. Never substitute preview_move or other
observation editing tools for a physics request. New previews replace the current
experiment. Call control_physics(play) if the user asked to run it. Report tool
errors as errors, never as completed movement. Material labels are not inferred
from proxy asset shape: inspect physical images when estimating material; honor
explicit user material facts and mark uncertain choices as assumptions.
"""

# Existing objects keep their prepared values across questions. A table estimate
# is required only for the first experiment.
for _tool in PHYSICS_SCHEMAS:
    _function = _tool["function"]
    if _function["name"] in {"preview_physics", "preview_physics_move"}:
        _function["parameters"]["required"] = [k for k in _function["parameters"]["required"] if k != "table_material"]
    if _function["name"] == "prepare_physics":
        _function["parameters"]["properties"]["update_existing"] = {"type": "boolean", "description": "True only when the user explicitly requests changing existing material/properties. Otherwise cached values are reused."}
PHYSICS_POLICY += """
Previously assigned physics_properties and physics_table_material persist across
requests and are authoritative estimates until explicitly revised. Check the
prepared_properties returned by inspect_physics_inputs. If already prepared,
skip prepare_physics, image/material estimation and table re-estimation; directly
call preview_physics_move or preview_physics with cached properties. Merely moving,
rotating, pausing or resetting an object must not change mass/friction/restitution.
Use prepare_physics(update_existing=true) only for an explicit property/material
revision (e.g. user corrects ceramic to paper, changes contents), or prepare an
invalidated resized/replaced shape. Never re-estimate just because it is a new
question. Optional table_material defaults to its existing prepared values.
"""

_VECTOR_SCHEMA = {"type":"array", "items":{"type":"number"}, "minItems":3, "maxItems":3}
PHYSICS_SCHEMAS += [{"type":"function", "function":{
    "name":"configure_experiment", "description":"Configure the FULL participant list, not just changed objects. Omitting an object removes it from the experiment. Preserve current participants unless the user explicitly asks to exclude them. For unchanged participants pass only object_id to retain their existing initial settings. Overrides are experiment-only. Static objects are immovable obstacles. Table axes: east +X, north +Y, up +Z. Force/impulse act at center of mass; torque is separate. Starts paused; use control_physics play. Repeats start from saved initial poses, not final simulated poses.",
    "parameters":{"type":"object", "properties":{
        "bodies":{"type":"array", "items":{"type":"object", "properties":{
            "object_id":{"type":"string"}, "toward_object_id":{"type":"string", "description":"Compute horizontal direction toward this included object from saved initial positions. Prefer this for pushing toward another object; do not guess direction vectors."},
            "impulse_magnitude_ns":{"type":"number"}, "speed_m_s":{"type":"number"}, "motion":{"type":"string","enum":["dynamic","static"]},
            **{k:_VECTOR_SCHEMA for k in ["position_m","velocity_m_s","angular_velocity_rad_s","impulse_ns","force_n","torque_nm"]},
            **{k:{"type":"number"} for k in ["mass_kg","static_friction","dynamic_friction","restitution","force_duration_s"]}
        }, "required":["object_id"], "additionalProperties":False}},
        "seconds":{"type":"number"}, "gravity_m_s2":{"type":"number"},
        "table_material":PHYSICS_SCHEMAS[2]["function"]["parameters"]["properties"]["table_material"]
    }, "required":["bodies"]}}},
    {"type":"function","function":{"name":"get_experiment_result", "description":"Read the latest actual physics run: elapsed simulated time, final positions and recorded contact pairs. Do not infer collision from proximity alone.", "parameters":{"type":"object","properties":{}}}}]
PHYSICS_POLICY += "\nUse configure_experiment for initial velocity, impulse (N s), finite-duration force (N), torque (N m), static collision partners and temporary mass/friction changes. Prepare properties for ALL participants first. ONLY a pure resume/play request with NO new impulse, force, mass, target or scenario uses control_physics(play) alone. A request specifying an impulse, force or new settings MUST configure the experiment before play, even if another experiment is active. Read get_experiment_result for outcomes. Never claim execution based on queued tool output. Answer in Korean."

PHYSICS_POLICY += "\nWhen the experiment is completed, replay requires control_physics(reset) followed by control_physics(play). A paused unfinished experiment resumes with play alone. configure_experiment preserves omitted initial motion settings for listed bodies, so a mass-only repeat keeps the same impulse. Set vectors explicitly to zero to remove a previous force or velocity.\n"

PHYSICS_POLICY += "\nFor a push TOWARD another object, always use toward_object_id plus impulse_magnitude_ns or speed_m_s in configure_experiment. The backend computes the horizontal unit direction from the participants initial positions. Do not guess a cardinal vector. Collision is not guaranteed by a push: inspect actual contact events afterwards, and report insufficient travel honestly.\n"

PHYSICS_SCHEMAS.append({"type":"function", "function":{
    "name":"push_toward", "description":"Preferred tool for pushing one prepared object toward another. Computes direction from initial positions, uses cached properties, includes both collision partners. Prepares paused; call control_physics play to run. No model vector arithmetic needed.",
    "parameters":{"type":"object", "properties":{
        "object_id":{"type":"string"}, "target_object_id":{"type":"string"},
        "additional_object_ids":{"type":"array", "items":{"type":"string"}, "description":"Additional collision participants. Existing experiment participants are retained; use configure_experiment to explicitly replace the participant set."},
        "impulse_ns":{"type":"number", "description":"Scalar impulse magnitude in N s"},
        "seconds":{"type":"number"}, "target_motion":{"type":"string", "enum":["static","dynamic"]},
        "table_material":PHYSICS_SCHEMAS[2]["function"]["parameters"]["properties"]["table_material"]
    }, "required":["object_id","target_object_id","impulse_ns"]}}})
PHYSICS_POLICY += "\nPrefer push_toward(object_id,target_object_id,impulse_ns,seconds,target_motion) for a simple impulse toward another object. The scalar impulse and required target are enough; cached properties are reused automatically. Use configure_experiment for more general force/velocity/mass experiments.\n"

PHYSICS_POLICY += "\nThe table is NOT an object to prepare_physics. Never invent a table object_id. If no prepared table material exists, pass table_material with explicit friction, restitution and rationale directly to push_toward or configure_experiment.\n"

PHYSICS_POLICY += "\nFor measured travel, report the provided net_displacement_m as 직선 변위. It is the 3D distance between initial and final positions, NOT the X component and NOT total trajectory length. Use horizontal_displacement_m for horizontal distance. Do not invent a total path length or a reason for absent contact from endpoints alone.\n"
