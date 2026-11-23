"""EEVEE animation along the pattern tree.

1 Blender unit = 1 µm. Z is exaggerated x4, as in the stills. Cells are
interpolated between the published patch layouts; this is a visualization of
the similarity path, not a tracked cell lineage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCENE_PATH = Path(r"E:\annotile\results\blender_cache\scene.json")
TREE_PATH = Path(r"E:\annotile\results\trajectory.json")
OUT = Path(r"E:\annotile\results\figures\evolution")
ZSCALE = 4.0
SPAN = 40.0
HOLD = 4
STEPS = 7


def hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


def grape(color: str) -> tuple[float, float, float]:
    source = hex_rgb(color)
    target = (0.72, 0.28, 0.62)
    return tuple(0.78 * source[i] + 0.22 * target[i] for i in range(3))


def mix(a, b, t: float):
    return tuple((1 - t) * a[i] + t * b[i] for i in range(3))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            block.remove(item)


def configure() -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if name in engines:
            scene.render.engine = name
            break
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 16
        if hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = True
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.4
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.93, 0.88, 0.91, 1.0)
    background.inputs["Strength"].default_value = 0.7


def material(name: str, color, transmission: float, subsurface: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    rgba = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.08 if transmission > 0.5 else 0.28
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
    if "IOR" in bsdf.inputs and transmission > 0.5:
        bsdf.inputs["IOR"].default_value = 1.04
    if transmission > 0.5:
        for attr, value in (
            ("blend_method", "BLEND"),
            ("surface_render_method", "BLENDED"),
            ("use_transparency_overlap", True),
            ("use_raytrace_refraction", True),
            ("use_screen_refraction", True),
        ):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, value)
                except (TypeError, ValueError):
                    pass
        bsdf.inputs["Base Color"].default_value = (0.97, 0.95, 0.99, 0.15)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 0.08
    return mat


def pack(model: dict) -> list[dict]:
    span = float(model["patch_um"])
    cells = []
    for cell in model["cells"]:
        cells.append(
            {
                "x": float(cell["x"]) / span * SPAN,
                "y": float(cell["y"]) / span * SPAN,
                "z": float(cell["z"]),
                "r": float(cell["r"]),
                "color": grape(cell["color"]),
                "type": cell["type"],
            }
        )
    cells.sort(key=lambda item: item["r"], reverse=True)
    return cells[:36]


def match(source: list[dict], target: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    used = set()
    pairs = []
    for index, cell in enumerate(source):
        best = None
        for other_index, other in enumerate(target):
            if other_index in used:
                continue
            gap = ((cell["x"] - other["x"]) ** 2 + (cell["y"] - other["y"]) ** 2) ** 0.5
            cost = gap + (0.0 if cell["type"] == other["type"] else 14.0)
            if best is None or cost < best[0]:
                best = (cost, other_index)
        if best is None:
            continue
        used.add(best[1])
        pairs.append((index, best[1]))
    extra_source = [index for index in range(len(source)) if index not in {pair[0] for pair in pairs}]
    extra_target = [index for index in range(len(target)) if index not in used]
    return pairs, extra_source, extra_target


def lerp_cell(left: dict, right: dict, t: float, radius_scale: float = 1.0) -> dict:
    return {
        "x": (1 - t) * left["x"] + t * right["x"],
        "y": (1 - t) * left["y"] + t * right["y"],
        "z": (1 - t) * left["z"] + t * right["z"],
        "r": ((1 - t) * left["r"] + t * right["r"]) * radius_scale,
        "color": mix(left["color"], right["color"], t),
    }


def transition(source: list[dict], target: list[dict]) -> list[list[dict]]:
    pairs, extra_source, extra_target = match(source, target)
    frames = []
    for step in range(STEPS):
        t = step / (STEPS - 1)
        cells = [lerp_cell(source[i], target[j], t) for i, j in pairs]
        for index in extra_source:
            cells.append(lerp_cell(source[index], source[index], t, 1 - t))
        for index in extra_target:
            born = dict(target[index])
            born["r"] = target[index]["r"] * t
            cells.append(born)
        frames.append([cell for cell in cells if cell["r"] > 0.35])
    return frames


def build(cells: list[dict], caption: str) -> None:
    clear_scene()
    configure()
    height = 10.0 * ZSCALE
    for index, cell in enumerate(cells):
        color = cell["color"]
        location = Vector((cell["x"], cell["y"], cell["z"] * ZSCALE))
        body = material(f"body_{index}", color, 0.12, 0.35)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=cell["r"], location=location)
        obj = bpy.context.active_object
        obj.data.materials.append(body)
        bpy.ops.object.shade_smooth()
        nucleus = material(f"nuc_{index}", tuple(channel * 0.25 for channel in color), 0.0, 0.2)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=cell["r"] * 0.4, location=location)
        core = bpy.context.active_object
        core.data.materials.append(nucleus)
        bpy.ops.object.shade_smooth()
    center = Vector((SPAN / 2, SPAN / 2, height / 2))
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    cube = bpy.context.active_object
    cube.scale = (SPAN + 3, SPAN + 3, height + 3)
    glass = material("glass", (0.97, 0.95, 0.99), 1.0, 0.0)
    cube.data.materials.append(glass)
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    wire = bpy.context.active_object
    wire.scale = (SPAN + 3, SPAN + 3, height + 3)
    frame = wire.modifiers.new("wire", "WIREFRAME")
    frame.thickness = 0.012
    frame.use_replace = True
    wire.data.materials.append(material("edge", (0.45, 0.22, 0.52), 0.0, 0.0))
    bpy.ops.object.light_add(type="AREA", location=center + Vector((-50, -80, 60)))
    key = bpy.context.active_object
    key.data.energy = 4000
    key.data.size = 40
    bpy.ops.object.light_add(type="AREA", location=center + Vector((40, 20, 16)))
    fill = bpy.context.active_object
    fill.data.energy = 1200
    fill.data.size = 36
    fill.data.color = (0.82, 0.62, 0.95)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 36
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((68, -94, 58))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    bpy.ops.object.text_add(location=Vector((SPAN / 2, SPAN / 2, height + 9)))
    text = bpy.context.active_object
    text.data.body = caption
    text.data.size = 6.5
    text.data.align_x = "CENTER"
    text.data.extrude = 0.08
    facing = camera.location - text.location
    text.rotation_euler = facing.to_track_quat("Z", "Y").to_euler()
    text.data.materials.append(material("label", (0.22, 0.08, 0.24), 0.0, 0.0))


def main() -> None:
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    tree = json.loads(TREE_PATH.read_text(encoding="utf-8"))
    models = {item["pattern"]: pack(item) for item in scene["patterns"]}
    frames: list[tuple[str, list[dict]]] = []
    for segment in tree["video_segments"]:
        path = segment["path"]
        slice_name = segment["slice"]
        for index, name in enumerate(path):
            for _ in range(HOLD):
                frames.append((f"{slice_name}  {name}", models[name]))
            if index + 1 < len(path):
                nxt = path[index + 1]
                for cells in transition(models[name], models[nxt])[1:-1]:
                    frames.append((f"{slice_name}  {name} -> {nxt}", cells))
    OUT.mkdir(parents=True, exist_ok=True)
    for index, (caption, cells) in enumerate(frames, start=1):
        build(cells, caption)
        target = OUT / f"frame_{index:04d}.png"
        bpy.context.scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        print("FRAME", index, caption, len(cells), flush=True)
    print("ANIM_DONE", len(frames), flush=True)


if __name__ == "__main__":
    main()
