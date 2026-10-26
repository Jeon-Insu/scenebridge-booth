"""Small camera-relative compass for the tabletop's fixed cardinal axes."""
from __future__ import annotations

import math

import omni.ui as ui
from omni.ui import scene as sc
from omni.kit.viewport.utility import get_active_viewport_window
from pxr import Gf, Usd, UsdGeom


def screen_cardinal_directions(camera_to_world, table_yaw: float):
    """Project TableFrame cardinal vectors into camera XY (no translation)."""
    world_to_camera = camera_to_world.GetInverse()
    c, s = math.cos(table_yaw), math.sin(table_yaw)
    result = {}
    for name, x, y in (("N", 0, 1), ("E", 1, 0), ("S", 0, -1), ("W", -1, 0)):
        projected = world_to_camera.TransformDir(Gf.Vec3d(c*x-s*y, s*x+c*y, 0))
        result[name] = (float(projected[0]), float(projected[1]))
    return result


class ViewportCompass:
    FRAME_NAME = "gist.tabletop_twin.compass"

    def __init__(self):
        self.window = None
        self.frame = None
        self.scene_view = None
        self._signature = None

    def destroy(self):
        if self.scene_view is not None:
            self.scene_view.scene.clear()
        if self.frame is not None:
            self.frame.clear()
        self.window = self.frame = self.scene_view = None

    def update(self, stage, table_yaw: float):
        window = get_active_viewport_window()
        if window is None:
            return
        if window is not self.window:
            self.destroy()
            self.window = window
            self.frame = window.get_frame(self.FRAME_NAME)
            with self.frame:
                with ui.VStack():
                    ui.Spacer(height=46)
                    with ui.HStack(height=164):
                        ui.Spacer(width=18)
                        with ui.ZStack(width=162):
                            ui.Rectangle(style={"background_color": 0xBF171A20, "border_radius": 8})
                            with ui.VStack(spacing=1):
                                ui.Label("TABLE COMPASS", height=22, alignment=ui.Alignment.CENTER,
                                         style={"font_size": 14, "color": 0xFFFAE7BC})
                                identity = [float(i % 5 == 0) for i in range(16)]
                                self.scene_view = sc.SceneView(
                                    sc.CameraModel(identity, identity), height=114,
                                    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
                                )
                                ui.Label("N +Y   E +X", height=18, alignment=ui.Alignment.CENTER)
                        ui.Spacer()
                    ui.Spacer()
            self._signature = None
        camera = stage.GetPrimAtPath(str(window.viewport_api.camera_path))
        if not camera.IsValid():
            return
        matrix = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        directions = screen_cardinal_directions(matrix, table_yaw)
        signature = tuple(round(value, 3) for pair in directions.values() for value in pair)
        if signature == self._signature:
            return
        self._signature = signature
        self.scene_view.scene.clear()
        with self.scene_view.scene:
            sc.Arc(0.70, begin=0, end=2*math.pi, wireframe=True, thickness=1, color=0x557F7F7F)
            for name, (x, y) in directions.items():
                color = 0xFF6D8CFF if name == "N" else 0xFFFFD781 if name == "E" else 0xFFD0D0D0
                # Foreshortening follows the actual view so labels agree with moves.
                sc.Line([0, 0, 0], [x*0.60, y*0.60, 0], thickness=3, color=color)
                with sc.Transform(transform=sc.Matrix44.get_translation_matrix(x*0.84, y*0.84, 0)):
                    sc.Label(name, size=19, color=color, alignment=ui.Alignment.CENTER)
