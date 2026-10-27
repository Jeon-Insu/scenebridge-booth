"""The twin's table USD, generated from the booth table size (rig.workspace_size_m in booth.json).

TableFrame: X=[0, length], Y=[0, width], tabletop Z=0, floor Z=-height (same layout as the original 1.25 m table).
Write one with: python3 tools/make_table.py 1.8 0.75
"""
from __future__ import annotations

TOP_THICKNESS_M = 0.04
LEG_SIZE_M = 0.055
LEG_INSET_M = 0.09


def table_asset_name(length_m: float, width_m: float, height_m: float = 0.75) -> str:
    cm = lambda v: int(round(v * 100))
    return f"table_{cm(length_m)}x{cm(width_m)}x{cm(height_m)}.usda"


def _num(value: float) -> str:
    return f"{round(value, 6):g}"


def table_usda(length_m: float, width_m: float, height_m: float = 0.75) -> str:
    name = table_asset_name(length_m, width_m, height_m)[:-5].replace("table", "Table") + "cm"
    leg_h = height_m - TOP_THICKNESS_M
    leg_z = -(TOP_THICKNESS_M + leg_h / 2)
    legs = []
    for tag, x, y in (("X0_Y0", LEG_INSET_M, LEG_INSET_M), ("X1_Y0", length_m - LEG_INSET_M, LEG_INSET_M),
                      ("X0_Y1", LEG_INSET_M, width_m - LEG_INSET_M), ("X1_Y1", length_m - LEG_INSET_M, width_m - LEG_INSET_M)):
        legs.append(f'''
        def Cube "Leg_{tag}"
        {{
            double size = 1
            color3f[] primvars:displayColor = [(0.22, 0.23, 0.25)]
            double3 xformOp:scale = ({_num(LEG_SIZE_M)}, {_num(LEG_SIZE_M)}, {_num(leg_h)})
            double3 xformOp:translate = ({_num(x)}, {_num(y)}, {_num(leg_z)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}''')
    for tag, y in (("Front", LEG_INSET_M), ("Back", width_m - LEG_INSET_M)):
        legs.append(f'''
        def Cube "Rail_{tag}"
        {{
            double size = 1
            color3f[] primvars:displayColor = [(0.22, 0.23, 0.25)]
            double3 xformOp:scale = ({_num(length_m - 2 * LEG_INSET_M)}, 0.035, 0.035)
            double3 xformOp:translate = ({_num(length_m / 2)}, {_num(y)}, -0.18)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}''')
    return f'''#usda 1.0
(
    defaultPrim = "{name}"
    doc = "Booth table {_num(length_m)} m x {_num(width_m)} m x {_num(height_m)} m. Local X/Y begin at the front-left corner and the top surface is local Z=0."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{name}"
{{
    custom double gist:heightM = {_num(height_m)}
    custom string gist:localFrame = "TableFrame: X=[0,{_num(length_m)}], Y=[0,{_num(width_m)}], tabletop Z=0, floor Z=-{_num(height_m)}"
    custom double gist:lengthM = {_num(length_m)}
    custom double gist:topThicknessM = {_num(TOP_THICKNESS_M)}
    custom double gist:widthM = {_num(width_m)}

    def Xform "Geometry"
    {{
        def Cube "Top"
        {{
            double size = 1
            color3f[] primvars:displayColor = [(0.69, 0.61, 0.50)]
            double3 xformOp:scale = ({_num(length_m)}, {_num(width_m)}, {_num(TOP_THICKNESS_M)})
            double3 xformOp:translate = ({_num(length_m / 2)}, {_num(width_m / 2)}, {_num(-TOP_THICKNESS_M / 2)})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}{"".join(legs)}
    }}
}}
'''
