"""Render sampled synthetic scenes with Isaac Replicator. Run with Isaac's python.sh.

  $ISAAC_SIM_DIR/python.sh \
      tools/generate_synthetic_props.py --specs outputs/synthetic/specs.json \
      --raw-dir outputs/synthetic/raw [--limit 1]

Scene content comes entirely from the specs JSON (smartphone_mocap.synthdata.scene_sampler);
this script only builds the stage and renders the four calibrated cameras with an
instance-id image per camera; smartphone_mocap.synthdata.instances turns those into COCO. Do not run it while
the demo's Isaac is up: both compete for the GPU.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--specs", type=Path, required=True)
parser.add_argument("--raw-dir", type=Path, required=True)
parser.add_argument("--limit", type=int)
parser.add_argument("--spp", type=int, default=64, help="path-traced samples per pixel")
parser.add_argument("--subframes", type=int, default=8)
parser.add_argument("--scenes", type=int, nargs="*", help="render only these scene ids")
parser.add_argument("--skip-done", action="store_true",
                    help="skip scenes whose four cameras are already rendered (resume after a crash)")
parser.add_argument("--save-stage", action="store_true",
                    help="also write each scene as <raw-dir>/scene_XXXXX/stage.usda, to open in Isaac Sim")
args = parser.parse_args()
# Replicator resolves a relative output_dir against ~/omni.replicator_out, not the cwd.
args.raw_dir = args.raw_dir.resolve()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

from smartphone_mocap.synthdata.scene_sampler import read_specs  # noqa: E402

CV_TO_USD = np.diag([1.0, -1.0, -1.0])     # OpenCV camera (y down, z forward) → USD (y up, -z forward)
HORIZONTAL_APERTURE_MM = 20.955
TABLE_HEIGHT_M = 0.75


def camera_to_world_usd(world_to_camera):
    w2c = np.asarray(world_to_camera, float)
    out = np.eye(4)
    out[:3, :3] = w2c[:3, :3].T @ CV_TO_USD
    out[:3, 3] = -w2c[:3, :3].T @ w2c[:3, 3]
    return out


def matrix(m):
    return Gf.Matrix4d(np.asarray(m, float).T.tolist())   # USD uses row vectors


def define_camera(stage, path, view):
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(view.fx * HORIZONTAL_APERTURE_MM / view.width)
    camera.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(HORIZONTAL_APERTURE_MM * view.height / view.width)
    # Measured 2026-10-25: with the opposite signs the rendered objects sat (-2·Δcx, -2·Δcy) px
    # from their calibrated projection (dx -20, dy +72 for cx 649.8, cy 323.9).
    camera.CreateHorizontalApertureOffsetAttr(-(view.cx - view.width / 2) * HORIZONTAL_APERTURE_MM / view.width)
    camera.CreateVerticalApertureOffsetAttr((view.cy - view.height / 2) * HORIZONTAL_APERTURE_MM / view.width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix(camera_to_world_usd(view.world_to_camera)))


def textured_material(stage, path, texture_dir):
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def texture(name, file, output, colour_space, scale=None, bias=None):
        node = UsdShade.Shader.Define(stage, f"{path}/{name}")
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(Path(texture_dir) / file))
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colour_space)
        node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if scale is not None:
            node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        kind = Sdf.ValueTypeNames.Float3 if output == "rgb" else Sdf.ValueTypeNames.Float
        return node.CreateOutput(output, kind)

    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture("color", "color.jpg", "rgb", "sRGB"))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        texture("rough", "roughness.jpg", "r", "raw"))
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        texture("normal", "normal.jpg", "rgb", "raw", (2, 2, 2, 1), (-1, -1, -1, 0)))
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def textured_plane(stage, path, corners, tile_m, material):
    """A quad with UVs that repeat every ``tile_m`` metres, so textures keep real scale."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    width = float(np.linalg.norm(np.subtract(corners[1], corners[0])))
    height = float(np.linalg.norm(np.subtract(corners[3], corners[0])))
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(width / tile_m, 0),
                          Gf.Vec2f(width / tile_m, height / tile_m), Gf.Vec2f(0, height / tile_m)]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def apply_class_label(prim, label):
    """Label for Replicator. Newer builds read UsdSemantics, older ones the Semantics API."""
    try:
        from pxr import UsdSemantics
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([label])
    except ImportError:
        pass
    try:
        from pxr import Semantics
        api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        api.CreateSemanticTypeAttr().Set("class")
        api.CreateSemanticDataAttr().Set(label)
    except ImportError:
        pass


def semantic_label(detector_class):
    """Replicator's COCO writer drops labels containing spaces into "other"; use underscores.

    The COCO category keeps the detector's real name ("toy duck"), so the dataset is
    labelled exactly as the runtime detector names things.
    """
    return detector_class.replace(" ", "_")


def place_prop(stage, path, placement):
    # The asset's own root carries the transform that brings the Meshy export to its
    # catalog size (the mug's is x0.053). Referencing it onto the prim we transform
    # would let our xformOpOrder override that, rendering the mug 19x too big, so the
    # reference sits on a child and our placement on its parent.
    prim = stage.DefinePrim(path, "Xform")
    stage.DefinePrim(f"{path}/Asset", "Xform").GetReferences().AddReference(placement.usd_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp()
    xform.AddRotateZOp().Set(placement.yaw_deg)
    if getattr(placement, "roll_axis", "x") == "y":
        xform.AddRotateYOp().Set(placement.roll_deg)
    else:
        xform.AddRotateXOp().Set(placement.roll_deg)
    xform.AddScaleOp().Set(Gf.Vec3f(placement.scale, placement.scale, placement.scale))
    translate.Set(Gf.Vec3d(placement.x_m, placement.y_m, 0.0))
    # Rest it on the table: a lying or upside-down prop has a different lowest point.
    bound = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    translate.Set(Gf.Vec3d(placement.x_m, placement.y_m, -bound.GetMin()[2]))
    apply_class_label(prim, semantic_label(placement.train_class or placement.detector_class))


def place_distractor(stage, path, item):
    """An unlabelled NVIDIA asset in its real size, upright, resting on the floor or table.

    No semantic label: in the instance image it is background, so the detector learns it
    is not a prop. Units and up axis differ between asset packs (ArchVis is centimetres).
    Evaluation scenes label table-top "unlisted" clutter so its outline can be scored.
    """
    prim = stage.DefinePrim(path, "Xform")
    stage.DefinePrim(f"{path}/Asset", "Xform").GetReferences().AddReference(item.usd_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp()
    xform.AddRotateZOp().Set(item.yaw_deg - item.forward_yaw_deg)
    xform.AddRotateXOp().Set(90.0 if item.up_axis == "Y" else 0.0)
    unit = item.meters_per_unit
    xform.AddScaleOp().Set(Gf.Vec3f(unit, unit, unit))
    translate.Set(Gf.Vec3d(item.x_m, item.y_m, 0.0))
    bound = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    if not bound.IsEmpty():
        translate.Set(Gf.Vec3d(item.x_m, item.y_m, item.surface_z_m - bound.GetMin()[2]))
    if item.kind == "unlisted":
        apply_class_label(prim, "unlisted")


def build_scene(spec):
    omni.usd.get_context().new_stage()
    app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    table = textured_material(stage, "/World/Looks/Table", spec.table_texture)
    floor = textured_material(stage, "/World/Looks/Floor", spec.floor_texture)
    wall = textured_material(stage, "/World/Looks/Wall", spec.wall_texture)
    w, d = spec.table_size_m
    z = -TABLE_HEIGHT_M
    textured_plane(stage, "/World/Table", [(0, 0, 0), (w, 0, 0), (w, d, 0), (0, d, 0)], 0.5, table)
    # Floor and back wall reach past the farthest random camera (2.2 m from the table).
    x0, x1, y0, y1 = -4.0, w + 4.0, -4.0, d + 4.0
    textured_plane(stage, "/World/Floor", [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], 1.0, floor)
    textured_plane(stage, "/World/Wall", [(x0, y1, z), (x1, y1, z), (x1, y1, 2.5), (x0, y1, 2.5)], 1.0, wall)
    for i, placement in enumerate(spec.props):
        place_prop(stage, f"/World/Props/prop_{i:02d}", placement)
    for i, item in enumerate(spec.distractors):
        place_distractor(stage, f"/World/Distractors/{item.kind}_{i:02d}", item)
    light = spec.lighting
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(light.sun_intensity)
    sun.CreateEnableColorTemperatureAttr(True)
    sun.CreateColorTemperatureAttr(light.color_temperature_k)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(
        Gf.Vec3f(90.0 - light.sun_elevation_deg, 0.0, light.sun_azimuth_deg))
    UsdLux.DomeLight.Define(stage, "/World/Ambient").CreateIntensityAttr(light.ambient_intensity)
    for view in spec.cameras:
        define_camera(stage, f"/World/Cameras/{view.camera_id}", view)
    app.update()


def write_placed(spec, scene_dir):
    """Where each prop and distractor really rests: world bounding-box centre, for evaluation truth."""
    import json
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def centre(path):
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        if box.IsEmpty():
            return None
        lo, hi = box.GetMin(), box.GetMax()
        return [(lo[i] + hi[i]) / 2 for i in range(3)] + [hi[2] - lo[2]]

    payload = {"props": [{"asset_id": p.asset_id, "centre": centre(f"/World/Props/prop_{i:02d}")}
                         for i, p in enumerate(spec.props)],
               "distractors": [{"kind": d.kind, "centre": centre(f"/World/Distractors/{d.kind}_{i:02d}")}
                               for i, d in enumerate(spec.distractors)]}
    Path(scene_dir).mkdir(parents=True, exist_ok=True)
    (Path(scene_dir) / "placed.json").write_text(json.dumps(payload, indent=1))


def scene_done(scene_dir, cameras):
    rgb = list(scene_dir.glob("*/rgb/rgb_*.png"))
    ids = list(scene_dir.glob("*/instance_segmentation/instance_segmentation_semantics_mapping_*.json"))
    return len(rgb) >= cameras and len(ids) >= cameras


def main():
    specs, _ = read_specs(args.specs)
    rep.settings.set_render_pathtraced(samples_per_pixel=args.spp)
    if args.scenes:
        specs = [spec for spec in specs if spec.scene_id in set(args.scenes)]
    for spec in specs[: args.limit] if args.limit else specs:
        scene_dir = args.raw_dir / f"scene_{spec.scene_id:05d}"
        if args.skip_done and scene_done(scene_dir, len(spec.cameras)):
            continue
        if scene_dir.exists():
            shutil.rmtree(scene_dir)          # a half-written scene from a crash
        build_scene(spec)
        if args.save_stage:
            out = args.raw_dir / f"scene_{spec.scene_id:05d}"
            out.mkdir(parents=True, exist_ok=True)
            omni.usd.get_context().get_stage().GetRootLayer().Export(str(out / "stage.usda"))
        products = [rep.create.render_product(f"/World/Cameras/{view.camera_id}", (view.width, view.height))
                    for view in spec.cameras]
        # BasicWriter, not CocoWriter: the COCO writer leaves "segmentation" empty, and
        # RF-DETR-Seg needs a mask per instance. Instance ids are turned into COCO
        # polygons afterwards by smartphone_mocap.synthdata.instances.
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(args.raw_dir / f"scene_{spec.scene_id:05d}"),
                          rgb=True, instance_segmentation=True,
                          colorize_instance_segmentation=False, semantic_types=["class"])
        writer.attach(products)
        rep.orchestrator.step(rt_subframes=args.subframes)
        rep.orchestrator.wait_until_complete()
        writer.detach()
        for product in products:
            product.destroy()
        write_placed(spec, args.raw_dir / f"scene_{spec.scene_id:05d}")
        print(f"scene {spec.scene_id:05d}: props {len(spec.props)}", flush=True)
    app.close()


main()
