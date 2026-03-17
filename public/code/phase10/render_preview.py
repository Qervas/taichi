"""Phase 10 — Batch preview renderer for textured OBJ models.

Imports each OBJ one at a time, auto-frames the camera, and renders a
clean preview image. Uses Cycles GPU with low samples for speed.

Usage:
    blender --background --python render_preview.py -- \
        --input ./assets --output ./renders --samples 32
"""
import bpy
import bmesh
import math
import os
import sys
import glob
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

import argparse
p = argparse.ArgumentParser()
p.add_argument("--input", default="./assets", help="Directory with OBJ+MTL+PNG files")
p.add_argument("--output", default="./renders", help="Output directory for preview images")
p.add_argument("--samples", type=int, default=32)
p.add_argument("--resolution", default="1920x1080")
p.add_argument("--only", default=None, help="Render only this OBJ (basename, no ext)")
args = p.parse_args(argv)

RES_X, RES_Y = map(int, args.resolution.split("x"))

BL_VERSION = bpy.app.version
BL_IS_3X = BL_VERSION[0] < 4
print(f"Blender {BL_VERSION[0]}.{BL_VERSION[1]}.{BL_VERSION[2]}"
      f" ({'3.x' if BL_IS_3X else '4.x+'})")


def clear_scene():
    """Remove everything."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                  bpy.data.images]:
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = args.samples
    sc.render.resolution_x = RES_X
    sc.render.resolution_y = RES_Y

    # GPU setup
    sc.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        for device_type in ('OPTIX', 'CUDA'):
            try:
                prefs.preferences.compute_device_type = device_type
                prefs.preferences.get_devices()
                devices = [d for d in prefs.preferences.devices
                           if d.type == device_type]
                if devices:
                    for d in prefs.preferences.devices:
                        d.use = (d.type == device_type)
                    print(f"  GPU: {device_type} ({len(devices)} devices)")
                    break
            except Exception:
                continue

    sc.cycles.use_denoising = True
    sc.render.film_transparent = True  # transparent background
    sc.view_settings.view_transform = 'Filmic'
    sc.view_settings.look = 'Medium High Contrast'


def setup_world():
    """HDRI-like procedural sky for clean lighting."""
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    N = world.node_tree.nodes
    L = world.node_tree.links
    N.clear()

    out = N.new("ShaderNodeOutputWorld"); out.location = (400, 0)
    bg = N.new("ShaderNodeBackground"); bg.location = (200, 0)
    bg.inputs["Strength"].default_value = 1.2

    sky = N.new("ShaderNodeTexSky"); sky.location = (0, 0)
    sky.sky_type = 'HOSEK_WILKIE'
    sky.sun_direction = (0.5, 0.3, 0.7)
    sky.turbidity = 3.0
    L.new(sky.outputs["Color"], bg.inputs["Color"])
    L.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_lights(center, size):
    """Sun + soft fill."""
    cx, cy, cz = center
    s = max(size, 1.0)

    bpy.ops.object.light_add(type='SUN', location=(cx + s, cy - s, cz + s * 2))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(2.0)
    sun.data.color = (1.0, 0.95, 0.88)
    sun.rotation_euler = (math.radians(50), math.radians(15), math.radians(-25))

    bpy.ops.object.light_add(type='AREA', location=(cx - s, cy + s, cz + s))
    fill = bpy.context.active_object
    fill.name = "Fill"
    fill.data.energy = 100
    fill.data.size = s * 3
    fill.data.color = (0.85, 0.9, 1.0)


def import_obj(obj_path):
    """Import OBJ with texture. Returns list of imported mesh objects."""
    if BL_IS_3X:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Y', axis_up='Z')
    else:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='NEGATIVE_Y', up_axis='Z')

    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']

    if not imported:
        print(f"  WARNING: No meshes imported from {obj_path}")
        return []

    # Fix texture paths — MTL might reference relative PNG
    assets_dir = os.path.dirname(obj_path)
    for obj in imported:
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        img = node.image
                        if not os.path.isabs(img.filepath):
                            candidate = os.path.join(assets_dir,
                                                      os.path.basename(img.filepath))
                            if os.path.exists(candidate):
                                img.filepath = candidate
                                img.reload()

    return imported


def get_bounds(objects):
    """Get combined bounding box of all objects."""
    all_coords = []
    for obj in objects:
        if obj.type == 'MESH':
            bbox_world = [obj.matrix_world @ Vector(corner)
                          for corner in obj.bound_box]
            all_coords.extend(bbox_world)

    if not all_coords:
        return Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))

    coords = np.array([(v.x, v.y, v.z) for v in all_coords])
    bmin = Vector(coords.min(axis=0))
    bmax = Vector(coords.max(axis=0))
    center = (bmin + bmax) / 2
    return bmin, bmax, center


def setup_camera(center, size):
    """Auto-frame camera: diagonal view looking at center."""
    cx, cy, cz = center
    d = size * 1.8  # distance from center

    # Diagonal view — slightly elevated
    cam_loc = (cx + d * 0.6, cy - d * 0.7, cz + d * 0.5)

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam

    # Track to center
    empty = bpy.data.objects.new("CamTarget", None)
    empty.location = (cx, cy, cz)
    bpy.context.collection.objects.link(empty)

    constraint = cam.constraints.new('TRACK_TO')
    constraint.target = empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    cam.data.lens = 35
    cam.data.clip_start = 0.1
    cam.data.clip_end = size * 20

    return cam


def render_one(obj_path, output_dir):
    """Import, frame, and render a single OBJ model."""
    name = os.path.splitext(os.path.basename(obj_path))[0]
    out_path = os.path.join(output_dir, f"{name}.png")

    if os.path.exists(out_path):
        print(f"  SKIP {name} — already rendered")
        return True

    print(f"\n{'='*60}")
    print(f"  Rendering: {name}")
    print(f"{'='*60}")

    clear_scene()
    setup_render()
    setup_world()

    # Import model
    objects = import_obj(obj_path)
    if not objects:
        print(f"  FAIL: no meshes in {obj_path}")
        return False

    n_verts = sum(len(o.data.vertices) for o in objects)
    n_faces = sum(len(o.data.polygons) for o in objects)
    print(f"  Meshes: {len(objects)}, verts: {n_verts:,}, faces: {n_faces:,}")

    # Get bounds
    bmin, bmax, center = get_bounds(objects)
    dims = bmax - bmin
    size = max(dims.x, dims.y, dims.z)
    print(f"  Bounds: ({bmin.x:.1f}, {bmin.y:.1f}, {bmin.z:.1f}) → "
          f"({bmax.x:.1f}, {bmax.y:.1f}, {bmax.z:.1f})")
    print(f"  Size: {dims.x:.1f} x {dims.y:.1f} x {dims.z:.1f}")

    # Setup camera and lights
    setup_camera(center, size)
    setup_lights(center, size)

    # Add ground plane (below model)
    bpy.ops.mesh.primitive_plane_add(
        size=size * 5,
        location=(center.x, center.y, bmin.z - 0.01))
    ground = bpy.context.active_object
    ground.name = "Ground"
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(mat)

    # Render
    sc = bpy.context.scene
    sc.render.filepath = out_path
    sc.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    print(f"  -> {out_path}")
    return True


def main():
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    obj_files = sorted(glob.glob(os.path.join(input_dir, "*.obj")))
    print(f"Found {len(obj_files)} OBJ files in {input_dir}")

    if args.only:
        obj_files = [f for f in obj_files if args.only in os.path.basename(f)]
        print(f"  Filtered to {len(obj_files)} matching '{args.only}'")

    ok = 0
    for i, obj_path in enumerate(obj_files):
        print(f"\n[{i+1}/{len(obj_files)}]")
        if render_one(obj_path, output_dir):
            ok += 1

    print(f"\n{'='*60}")
    print(f"Done! {ok}/{len(obj_files)} models rendered to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
