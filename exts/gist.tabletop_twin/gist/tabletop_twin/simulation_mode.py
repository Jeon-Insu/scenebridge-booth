"""Reversible viewport experiment authored only in an anonymous session sublayer."""
import copy
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

ROOT = "/World/TabletopPhysicsExperiment"

# Initial penetration is common when vision-derived poses are used.  Small
# penetrations are left to PhysX's depenetration solver; larger ones are
# separated upward in the temporary experiment, with an explicit warning.
AUTO_DEPENETRATION_M = 0.005
AUTO_DEPENETRATION_RATIO = 0.10
OVERLAP_CORRECTION_CLEARANCE_M = 0.005


class SimulationMode:
    def __init__(self, timeline):
        self.timeline = timeline
        self.layer = None
        self.stage = None
        self.scenario = None
        self.status = "Observation"
        self.preflight_warnings = []
        self._step_subscription = None
        self._pre_step_subscription = None
        self._contact_subscription = None

    @property
    def active(self):
        return self.layer is not None

    def enter(self, stage, scenario, offset):
        if self.active or self.timeline.is_playing():
            raise ValueError("Return to observation and stop the timeline before preparing a new experiment")
        if scenario.get("schema") != "gist.tabletop_physics.v1" or not scenario.get("bodies"):
            raise ValueError("A prepared physics scenario is required")
        if UsdGeom.GetStageUpAxis(stage) != "Z" or abs(UsdGeom.GetStageMetersPerUnit(stage)-1) > 1e-6:
            raise ValueError("Viewport experiment requires a Z-up meter stage")
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("Existing rigid bodies require a separate stage; stop before affecting unrelated physics")
        self.stage, self.scenario = stage, copy.deepcopy(scenario)
        self.offset = tuple(offset)
        self.old_time = self.timeline.get_current_time()
        self.old_edit_target = stage.GetEditTarget()
        self.layer = Sdf.Layer.CreateAnonymous("tabletop-physics-experiment.usda")
        stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        try:
            with Usd.EditContext(stage, self.layer):
                self._build()
                self._preflight()
                for path in ["/World/Tracking/Objects", "/World/AgentPreview"]:
                    if stage.GetPrimAtPath(path):
                        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set("invisible")
            suffix = f" | {len(self.preflight_warnings)} initial overlap warning(s)" if self.preflight_warnings else ""
            self.status = "Simulation ready | agent-estimated properties" + suffix
            stage.SetEditTarget(self.layer)
        except Exception:
            self.leave()
            raise

    def replace(self, stage, scenario, offset):
        if not self.active:
            return self.enter(stage, scenario, offset)
        if stage != self.stage or scenario.get("schema") != "gist.tabletop_physics.v1" or not scenario.get("bodies"):
            raise ValueError("A valid scenario on the current stage is required")
        self.timeline.stop()
        self.scenario = copy.deepcopy(scenario)
        self.offset = tuple(offset)
        try:
            with Usd.EditContext(stage, self.layer):
                self._clear_bodies()
                self._build()
                self._preflight()
            self.status = "Simulation ready | reused properties; updated experiment pose"
        except Exception:
            self.leave()
            raise

    def _clear_bodies(self):
        # Keep the PhysicsScene prim alive: SimulationManager retains its
        # handle between updates, including prepare+play in one response.
        root = self.stage.GetPrimAtPath(ROOT)
        for child in list(root.GetChildren()):
            if child.GetName() != "PhysicsScene":
                self.stage.RemovePrim(child.GetPath())

    def _material(self, path, values):
        mat = UsdShade.Material.Define(self.stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(values["static_friction"])
        api.CreateDynamicFrictionAttr(values["dynamic_friction"])
        api.CreateRestitutionAttr(values["restitution"])
        return mat

    def _box(self, path, size, center, material):
        shape = UsdGeom.Cube.Define(self.stage, path)
        shape.CreateSizeAttr(1.)
        shape.AddTranslateOp().Set(Gf.Vec3d(*center))
        shape.AddScaleOp().Set(Gf.Vec3d(*size))
        self._collider(shape, material)
        return shape

    def _collider(self, shape, material):
        api = UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        api.CreateSimulationOwnerRel().SetTargets([ROOT+"/PhysicsScene"])
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose="physics")
        shape.CreateVisibilityAttr().Set("invisible")

    def _build(self):
        s = self.scenario
        self.elapsed = 0.0
        self.contacts = []
        self._contact_pairs = set()
        self.completed = False
        self.forces = []
        self.initial_bodies = {b["object_id"]:copy.deepcopy(b) for b in s["bodies"]}
        root = UsdGeom.Xform.Define(self.stage, ROOT)
        root.ClearXformOpOrder()
        root.AddTranslateOp().Set(Gf.Vec3d(*self.offset))
        scene = UsdPhysics.Scene.Define(self.stage, ROOT+"/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(float(s["gravity_m_s2"]))
        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableCCDAttr(True)
        table_mat = self._material(ROOT+"/TableMaterial", s["table_material"])
        x0,x1,y0,y1 = s["table_bounds_xy"]
        self._box(ROOT+"/TableCollider", [x1-x0,y1-y0,.05], [(x0+x1)/2,(y0+y1)/2,-.025], table_mat)
        self._box(ROOT+"/GroundCollider", [20,20,.1], [(x0+x1)/2,(y0+y1)/2,-self.offset[2]-.05], table_mat)
        self.body_paths = []
        for i,b in enumerate(s["bodies"]):
            path = ROOT+f"/Body_{i:03d}"
            self.body_paths.append(path)
            xf = UsdGeom.Xform.Define(self.stage, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(*b["xyz"]))
            q = b["rotation_wxyz"]
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
            if b.get("motion", "dynamic") != "static":
                rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
                rb.CreateSimulationOwnerRel().SetTargets([ROOT+"/PhysicsScene"])
                UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(b["mass_kg"])
                velocity = [v+j/b["mass_kg"] for v,j in zip(b.get("velocity_m_s", [0,0,0]), b.get("impulse_ns", [0,0,0]))]
                rb.CreateVelocityAttr(Gf.Vec3f(*velocity))
                rb.CreateAngularVelocityAttr(Gf.Vec3f(*[math.degrees(v) for v in b.get("angular_velocity_rad_s", [0,0,0])]))
                dynamics = PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim())
                dynamics.CreateEnableCCDAttr(True)
                dynamics.CreateSleepThresholdAttr(1e-8)
                dynamics.CreateAngularDampingAttr(0.0)
                PhysxSchema.PhysxContactReportAPI.Apply(xf.GetPrim()).CreateThresholdAttr(0.0)
                if any(b.get("force_n", [0,0,0]) + b.get("torque_nm", [0,0,0])):
                    force = PhysxSchema.PhysxForceAPI.Apply(xf.GetPrim())
                    force.CreateModeAttr("force")
                    force.CreateWorldFrameEnabledAttr(True)
                    force.CreateForceEnabledAttr(True)
                    force.CreateForceAttr(Gf.Vec3f(*b.get("force_n", [0,0,0])))
                    force.CreateTorqueAttr(Gf.Vec3f(0,0,0))
                    self.forces.append((force, b))
            xf.GetPrim().SetCustomDataByKey("tabletop:object_id", b["object_id"])
            material = self._material(path+"/Material", b)
            size, center = b["size_m"], b["center_local_m"]
            if b["collider"] == "box":
                shape = self._box(path+"/Collider", size, center, material)
            else:
                if b["collider"] == "sphere":
                    shape = UsdGeom.Sphere.Define(self.stage,path+"/Collider")
                    shape.CreateRadiusAttr(max(size)/2)
                elif b["collider"] == "cylinder":
                    shape = UsdGeom.Cylinder.Define(self.stage,path+"/Collider")
                    shape.CreateRadiusAttr(max(size[:2])/2)
                    shape.CreateHeightAttr(size[2])
                    shape.CreateAxisAttr("Z")
                else:
                    raise ValueError("Unsupported collider")
                shape.AddTranslateOp().Set(Gf.Vec3d(*center))
                self._collider(shape, material)
            asset = b.get("asset_uri")
            if asset and Path(asset).is_file():
                visual = UsdGeom.Xform.Define(self.stage,path+"/Visual")
                visual.GetPrim().GetReferences().AddReference(asset)
                visual.ClearXformOpOrder()
                visual.AddScaleOp().Set(Gf.Vec3d(*b.get("scale", [1,1,1])))
                # Imported assets must not add a second nested rigid body/collider.
                for prim in Usd.PrimRange(visual.GetPrim()):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                        raise ValueError("Asset contains physics; use a visual-only asset")
            else:
                shape.CreateVisibilityAttr().Set("inherited")
                shape.CreateDisplayColorAttr([Gf.Vec3f(.3,.6,.9)])

        self._step_subscription = None
        self._pre_step_subscription = None
        self._contact_subscription = None
        import omni.physx
        self._pre_step_subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(self._before_step, True, 0)
        self._step_subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(self._on_step, False, 0)
        self._contact_subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self._on_contacts)

    def _before_step(self, dt):
        if not self.active or self.completed:
            return
        # Use simulated time, never wall clock; pause/resume preserves the schedule.
        for api, body in self.forces:
            remaining = max(0.0, body.get("force_duration_s", 0.0) - self.elapsed)
            fraction = min(1.0, remaining / max(float(dt), 1e-9))
            api.GetForceAttr().Set(Gf.Vec3f(*[v*fraction for v in body.get("force_n", [0,0,0])]))
            if fraction and any(body.get("torque_nm", [0,0,0])):
                import omni.physx, omni.usd
                from pxr import PhysicsSchemaTools
                simulation = omni.physx.get_physx_simulation_interface()
                stage_id = omni.usd.get_context().get_stage_id()
                encoded = PhysicsSchemaTools.sdfPathToInt(api.GetPrim().GetPath())
                simulation.wake_up(stage_id, encoded)
                simulation.apply_torque(stage_id, encoded,
                    Gf.Vec3f(*[v*fraction for v in body["torque_nm"]]))

    def _on_step(self, dt):
        if not self.active or self.completed:
            return
        self.elapsed += float(dt)
        if self.elapsed + 1e-7 >= self.scenario.get("seconds", 3.0):
            self.completed = True
            self.timeline.pause()
            self.status = "Simulation completed"

    def _on_contacts(self, headers, data):
        from pxr import PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        names = {key:b["object_id"] for p,b in zip(self.body_paths, self.scenario["bodies"])
                 for key in (p, p+"/Collider")}
        for header in headers:
            if header.type != ContactEventType.CONTACT_FOUND:
                continue
            paths = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (header.actor0, header.actor1)]
            if not all(p.startswith(ROOT+"/") for p in paths):
                continue
            pair = tuple(sorted(names.get(p, p.rsplit("/",1)[-1]) for p in paths))
            if pair in self._contact_pairs:
                continue
            self._contact_pairs.add(pair)
            self.contacts.append({"objects":list(pair), "time_s":self.elapsed, "source":"PhysX CONTACT_FOUND"})

    def result(self):
        if not self.active:
            return None
        objects = []
        for path, body in zip(self.body_paths, self.scenario["bodies"]):
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return {"status":"Simulation invalidated", "completed":False,
                        "error":"Experiment body no longer exists: " + path,
                        "elapsed_simulation_s":self.elapsed, "contacts":list(self.contacts), "objects":objects}
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xyz = [float(world.ExtractTranslation()[i])-self.offset[i] for i in range(3)]
            origin = self.initial_bodies[body["object_id"]]["xyz"]
            rotation = world.ExtractRotationQuat()
            rb = UsdPhysics.RigidBodyAPI(prim)
            velocity = list(rb.GetVelocityAttr().Get() or [0,0,0]) if rb else [0,0,0]
            angular = [math.radians(v) for v in (rb.GetAngularVelocityAttr().Get() or [0,0,0])] if rb else [0,0,0]
            objects.append({"object_id":body["object_id"], "initial_xyz_m":list(origin), "final_xyz_m":xyz,
                            "final_rotation_wxyz":[float(rotation.GetReal()), *map(float,rotation.GetImaginary())],
                            "velocity_m_s":velocity, "angular_velocity_rad_s":angular,
                            "displacement_m":[xyz[i]-origin[i] for i in range(3)], "motion":body.get("motion","dynamic"), "mass_kg":body["mass_kg"]})
        return {"status":self.status, "completed":self.completed, "elapsed_simulation_s":self.elapsed,
                "contacts":list(self.contacts), "objects":objects, "measurement_source":"USD rigid-body transforms and PhysX contact events"}

    def _preflight(self):
        # Include invisible colliders, even when their visual asset is visible.
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False, ignoreVisibility=True,
        )
        self.preflight_warnings = []
        boxes = [
            (name, cache.ComputeWorldBound(
                self.stage.GetPrimAtPath(ROOT + "/" + name)
            ).ComputeAlignedRange())
            for name in ("TableCollider", "GroundCollider")
        ]
        for index, path in enumerate(self.body_paths):
            body = self.scenario["bodies"][index]
            # Upward corrections are monotonic. Recheck every obstacle after
            # each move, since clearing one can enter an earlier obstacle.
            for attempt in range(len(boxes) + 1):
                cache.Clear()
                box = cache.ComputeWorldBound(
                    self.stage.GetPrimAtPath(path + "/Collider")
                ).ComputeAlignedRange()
                low, high = box.GetMin(), box.GetMax()
                corrected = False
                small_overlaps = []
                for name, other in boxes:
                    overlap = [min(high[i], other.GetMax()[i]) -
                               max(low[i], other.GetMin()[i]) for i in range(3)]
                    penetration = min(overlap)
                    if penetration <= 0:
                        continue
                    characteristic = max(min(min(box.GetSize()[i], other.GetSize()[i])
                                             for i in range(3)), 1e-6)
                    soft_limit = max(AUTO_DEPENETRATION_M,
                                     AUTO_DEPENETRATION_RATIO * characteristic)
                    if penetration <= soft_limit:
                        small_overlaps.append(
                            f"{body['object_id']} overlaps {name} by {penetration*1000:.1f} mm; "
                            "PhysX depenetration allowed"
                        )
                        continue
                    # Intersection width is insufficient for a contained body.
                    # Move its lower face completely above the obstacle.
                    delta = other.GetMax()[2] - low[2] + OVERLAP_CORRECTION_CLEARANCE_M
                    body["xyz"][2] += delta
                    UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).GetOrderedXformOps()[0].Set(
                        Gf.Vec3d(*body["xyz"])
                    )
                    self.preflight_warnings.append(
                        f"{body['object_id']} overlapped {name}; "
                        f"physics-only upward correction {delta*1000:.1f} mm"
                    )
                    corrected = True
                    break
                if not corrected:
                    self.preflight_warnings.extend(small_overlaps)
                    break
            else:
                raise ValueError("Initial overlap could not be resolved")
            boxes.append((path, box))
        self.initial_bodies = {b["object_id"]:copy.deepcopy(b) for b in self.scenario["bodies"]}

    def command(self, name):
        if name == "return":
            self.leave()
            return
        if not self.active:
            raise ValueError("Prepare a simulation first")
        if name == "play":
            if self.completed:
                raise ValueError("Experiment completed; reset before another trial")
            self.timeline.play()
            self.status = "Simulation running"
        elif name == "pause":
            self.timeline.pause()
            self.status = "Simulation paused"
        elif name == "reset":
            self.timeline.stop()
            with Usd.EditContext(self.stage, self.layer):
                self._clear_bodies()
                self._build()
            self.timeline.set_current_time(0)
            self.status = "Simulation reset to initial pose"
        else:
            raise ValueError("Unknown simulation command")

    def leave(self):
        self._step_subscription = None
        self._pre_step_subscription = None
        self._contact_subscription = None
        if self.active:
            self.timeline.stop()
            self.stage.SetEditTarget(self.old_edit_target)
            session = self.stage.GetSessionLayer()
            session.subLayerPaths = [p for p in session.subLayerPaths if p != self.layer.identifier]
            self.timeline.set_current_time(self.old_time)
        self.layer = None
        self.stage = None
        self.preflight_warnings = []
        self.status = "Observation"
