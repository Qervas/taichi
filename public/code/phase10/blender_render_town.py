"""Phase 10 — Town Renderer (Blender Cycles).

Loads the scene descriptor from generate_town.py and renders a cinematic
overhead or street-level view of the town.

Usage:
    blender --background --python blender_render_town.py -- \
        --scene ./town_export/scene.json --output ./renders_town --samples 128
"""
import bpy
import bmesh
import math
import os
import sys
import json
import numpy as np
from mathutils import Vector, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

import argparse
p = argparse.ArgumentParser()
p.add_argument("--scene", required=True, help="Path to scene.json")
p.add_argument("--output", default="./renders_town")
p.add_argument("--samples", type=int, default=128)
p.add_argument("--resolution", default="1920x1080")
p.add_argument("--cpu", action="store_true")
p.add_argument("--camera", default="drone",
               choices=["drone", "street", "closeup", "overview"])
args = p.parse_args(argv)

RES_X, RES_Y = map(int, args.resolution.split("x"))

BL_VERSION = bpy.app.version
BL_IS_3X = BL_VERSION[0] < 4
print(f"Blender {BL_VERSION[0]}.{BL_VERSION[1]}.{BL_VERSION[2]}"
      f" ({'3.x' if BL_IS_3X else '4.x+'})")

MODELS_DIR = os.path.expanduser("~/Downloads/models")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _set_coat(bsdf, weight, ior, roughness):
    if BL_IS_3X:
        bsdf.inputs["Clearcoat"].default_value = weight
        bsdf.inputs["Clearcoat Roughness"].default_value = roughness
    else:
        bsdf.inputs["Coat Weight"].default_value = weight
        bsdf.inputs["Coat IOR"].default_value = ior
        bsdf.inputs["Coat Roughness"].default_value = roughness


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in list(bpy.data.collections):
        if c != bpy.context.scene.collection:
            bpy.data.collections.remove(c)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def setup_render():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = args.samples
    sc.render.resolution_x = RES_X
    sc.render.resolution_y = RES_Y

    if args.cpu:
        sc.cycles.device = 'CPU'
    else:
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
    sc.render.use_persistent_data = True
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'Filmic'
    sc.view_settings.look = 'Medium High Contrast'


def setup_sky():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    N = world.node_tree.nodes
    L = world.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputWorld"); out.location = (600, 0)
    bg = N.new("ShaderNodeBackground"); bg.location = (400, 0)
    bg.inputs["Strength"].default_value = 1.2
    sky = N.new("ShaderNodeTexSky"); sky.location = (0, 0)
    sky.sky_type = 'HOSEK_WILKIE'
    sky.sun_direction = (0.5, -0.3, 0.6)
    sky.turbidity = 4.0
    L.new(sky.outputs["Color"], bg.inputs["Color"])
    L.new(bg.outputs["Background"], out.inputs["Surface"])


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def mat_asphalt():
    mat = bpy.data.materials.new("Asphalt")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    bsdf = N.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.045, 1)
    bsdf.inputs["Roughness"].default_value = 0.9

    # Subtle noise bump for texture
    tc = N.new("ShaderNodeTexCoord"); tc.location = (-600, -300)
    noise = N.new("ShaderNodeTexNoise"); noise.location = (-400, -300)
    noise.inputs["Scale"].default_value = 80.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = N.new("ShaderNodeBump"); bump.location = (-100, -300)
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.01
    L.new(noise.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def mat_concrete():
    mat = bpy.data.materials.new("Concrete")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.35, 0.33, 0.30, 1)
    bsdf.inputs["Roughness"].default_value = 0.85
    return mat


def mat_sidewalk():
    mat = bpy.data.materials.new("Sidewalk")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    bsdf = N.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.45, 0.42, 0.38, 1)
    bsdf.inputs["Roughness"].default_value = 0.8

    # Tile pattern
    tc = N.new("ShaderNodeTexCoord"); tc.location = (-600, -300)
    brick = N.new("ShaderNodeTexBrick"); brick.location = (-300, -300)
    brick.inputs["Scale"].default_value = 15.0
    brick.inputs["Color1"].default_value = (0.42, 0.40, 0.36, 1)
    brick.inputs["Color2"].default_value = (0.48, 0.45, 0.40, 1)
    brick.inputs["Mortar"].default_value = (0.35, 0.33, 0.30, 1)
    brick.inputs["Mortar Size"].default_value = 0.02
    L.new(tc.outputs["Object"], brick.inputs["Vector"])
    L.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    bump = N.new("ShaderNodeBump"); bump.location = (-100, -300)
    bump.inputs["Strength"].default_value = 0.1
    L.new(brick.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def mat_tree_trunk():
    mat = bpy.data.materials.new("TreeTrunk")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.15, 0.08, 0.03, 1)
    bsdf.inputs["Roughness"].default_value = 0.95
    return mat


def mat_tree_crown():
    mat = bpy.data.materials.new("TreeCrown")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    bsdf = N.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.08, 0.22, 0.05, 1)
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Specular IOR Level" if not BL_IS_3X else "Specular"].default_value = 0.1

    # Color variation with noise
    tc = N.new("ShaderNodeTexCoord"); tc.location = (-500, 0)
    noise = N.new("ShaderNodeTexNoise"); noise.location = (-300, 0)
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 4.0
    L.new(tc.outputs["Object"], noise.inputs["Vector"])

    if BL_IS_3X:
        mix = N.new("ShaderNodeMixRGB"); mix.location = (-100, 0)
        mix.inputs["Fac"].default_value = 0.4
        mix.inputs["Color1"].default_value = (0.08, 0.22, 0.05, 1)
        mix.inputs["Color2"].default_value = (0.12, 0.30, 0.08, 1)
        L.new(noise.outputs["Fac"], mix.inputs["Fac"])
        L.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        mix = N.new("ShaderNodeMix"); mix.location = (-100, 0)
        mix.data_type = 'RGBA'
        mix.inputs["Factor"].default_value = 0.4
        mix.inputs[6].default_value = (0.08, 0.22, 0.05, 1)
        mix.inputs[7].default_value = (0.12, 0.30, 0.08, 1)
        L.new(noise.outputs["Fac"], mix.inputs["Factor"])
        L.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def mat_road_marking():
    mat = bpy.data.materials.new("RoadMarking")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.85, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def mat_ground():
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    bsdf = N.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.18, 0.22, 0.12, 1)
    bsdf.inputs["Roughness"].default_value = 0.95

    # Grass-like noise
    tc = N.new("ShaderNodeTexCoord"); tc.location = (-600, -200)
    noise = N.new("ShaderNodeTexNoise"); noise.location = (-400, -200)
    noise.inputs["Scale"].default_value = 50.0
    noise.inputs["Detail"].default_value = 10.0
    L.new(tc.outputs["Object"], noise.inputs["Vector"])

    if BL_IS_3X:
        mix = N.new("ShaderNodeMixRGB"); mix.location = (-200, -200)
        mix.inputs["Fac"].default_value = 0.3
        mix.inputs["Color1"].default_value = (0.15, 0.20, 0.10, 1)
        mix.inputs["Color2"].default_value = (0.22, 0.28, 0.15, 1)
        L.new(noise.outputs["Fac"], mix.inputs["Fac"])
        L.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        mix = N.new("ShaderNodeMix"); mix.location = (-200, -200)
        mix.data_type = 'RGBA'
        mix.inputs["Factor"].default_value = 0.3
        mix.inputs[6].default_value = (0.15, 0.20, 0.10, 1)
        mix.inputs[7].default_value = (0.22, 0.28, 0.15, 1)
        L.new(noise.outputs["Fac"], mix.inputs["Factor"])
        L.new(mix.outputs[2], bsdf.inputs["Base Color"])

    bump = N.new("ShaderNodeBump"); bump.location = (-100, -400)
    bump.inputs["Strength"].default_value = 0.2
    L.new(noise.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# ---------------------------------------------------------------------------
# Import functions
# ---------------------------------------------------------------------------
def import_obj_zup(filepath):
    """Import OBJ with Z-up convention."""
    if BL_IS_3X:
        bpy.ops.import_scene.obj(filepath=filepath,
                                  axis_forward='-Y', axis_up='Z')
    else:
        bpy.ops.wm.obj_import(filepath=filepath,
                               forward_axis='NEGATIVE_Y', up_axis='Z')
    return list(bpy.context.selected_objects)


def create_ground_plane(town_size, material):
    """Create a large ground plane directly in Blender (more reliable than OBJ)."""
    tw, th = town_size
    margin = max(tw, th) * 0.5
    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(tw / 2, th / 2, -0.05))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (tw + 2 * margin, th + 2 * margin, 1)
    ground.data.materials.append(material)
    return ground


def import_infrastructure(scene_dir, infra_dict, materials, town_size):
    """Import roads, curbs, sidewalks, trees OBJs + create ground."""
    # Create ground as Blender primitive (reliable)
    create_ground_plane(town_size, materials["ground"])
    print("  Ground plane created")

    mat_map = {
        "roads_obj": materials["asphalt"],
        "curbs_obj": materials["concrete"],
        "sidewalks_obj": materials["sidewalk"],
        "trees_obj": materials["tree_crown"],
        "crosswalks_obj": materials["road_marking"],
        "centerlines_obj": materials["road_marking"],
    }

    imported = {}
    for key, mat in mat_map.items():
        obj_file = infra_dict.get(key)
        if not obj_file:
            continue
        path = os.path.join(scene_dir, obj_file)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        objs = import_obj_zup(path)
        for obj in objs:
            if obj.type == 'MESH':
                obj.data.materials.clear()
                obj.data.materials.append(mat)
                if "tree" in key:
                    for poly in obj.data.polygons:
                        poly.use_smooth = True
                # Fix coordinate flip: import_obj_zup negates X,Y
                # Directly negate vertex coords to restore original positions
                mesh = obj.data
                n_verts = len(mesh.vertices)
                cos = np.empty(n_verts * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", cos)
                cos = cos.reshape(-1, 3)
                cos[:, 0] = -cos[:, 0]  # flip X
                cos[:, 1] = -cos[:, 1]  # flip Y
                mesh.vertices.foreach_set("co", cos.ravel())
                mesh.update()
                # Also flip normals since we flipped 2 axes
                # (flipping even number of axes preserves winding,
                #  but need to recalc normals)
                mesh.calc_normals()
        bpy.context.view_layer.update()
        # Debug: verify bounds
        for obj in objs:
            if obj.type == 'MESH':
                wverts = [obj.matrix_world @ v.co for v in obj.data.vertices]
                if wverts:
                    xs = [v.x for v in wverts]
                    ys = [v.y for v in wverts]
                    zs = [v.z for v in wverts]
                    print(f"    {key} bounds: "
                          f"X=[{min(xs):.1f},{max(xs):.1f}] "
                          f"Y=[{min(ys):.1f},{max(ys):.1f}] "
                          f"Z=[{min(zs):.2f},{max(zs):.2f}]")
        imported[key] = objs
        print(f"  {key}: {len(objs)} objects")

    return imported


def import_buildings(scene_dir, buildings_list):
    """Import all building OBJs at their placed positions."""
    all_objs = []
    total = len(buildings_list)
    scene_bmin = np.array([1e10, 1e10, 1e10])
    scene_bmax = np.array([-1e10, -1e10, -1e10])

    for i, bldg in enumerate(buildings_list):
        obj_path = os.path.join(MODELS_DIR, bldg["file"])
        if not os.path.exists(obj_path):
            print(f"  [{i+1}/{total}] {bldg['file']} — NOT FOUND, skip")
            continue

        objs = import_obj_zup(obj_path)
        meshes = [o for o in objs if o.type == 'MESH']

        if not meshes:
            continue

        # Fix texture paths
        obj_dir = os.path.dirname(obj_path)
        for obj in meshes:
            for mat_slot in obj.material_slots:
                mat = mat_slot.material
                if mat and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            img = node.image
                            if not os.path.exists(img.filepath_from_user()):
                                candidate = os.path.join(
                                    obj_dir, os.path.basename(img.filepath))
                                if os.path.exists(candidate):
                                    img.filepath = candidate
                                    img.reload()

        # Compute the building's native bounds (in imported space)
        bpy.context.view_layer.update()
        all_verts = []
        for obj in meshes:
            for v in obj.data.vertices:
                all_verts.append(obj.matrix_world @ v.co)
        verts = np.array([(v.x, v.y, v.z) for v in all_verts])
        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        bcenter_xy = (bmin[:2] + bmax[:2]) / 2
        base_z = bmin[2]

        # Target position from layout
        tx, ty, tz = bldg["position"]
        rot_deg = bldg["rotation_deg"]

        # Offset to center building at target pos with base at Z=0
        offset_x = tx - bcenter_xy[0]
        offset_y = ty - bcenter_xy[1]
        offset_z = -base_z

        for obj in meshes:
            obj.location.x += offset_x
            obj.location.y += offset_y
            obj.location.z += offset_z

        # Apply rotation around the building center
        if rot_deg != 0:
            bpy.context.view_layer.update()
            pivot = Vector((tx, ty, 0))
            rot_mat = Euler((0, 0, math.radians(rot_deg))).to_matrix().to_4x4()
            for obj in meshes:
                # Translate to pivot, rotate, translate back
                obj.location -= pivot
                obj.location = rot_mat @ obj.location
                obj.location += pivot
                obj.rotation_euler.z += math.radians(rot_deg)

        # Track scene bounds for debug
        bpy.context.view_layer.update()
        for obj in meshes:
            for v in obj.data.vertices:
                wv = obj.matrix_world @ v.co
                scene_bmin = np.minimum(scene_bmin, [wv.x, wv.y, wv.z])
                scene_bmax = np.maximum(scene_bmax, [wv.x, wv.y, wv.z])

        all_objs.extend(meshes)

        if i % 10 == 0 or i < 3:
            print(f"  [{i+1}/{total}] {bldg['file']} at "
                  f"({tx:.0f}, {ty:.0f}) rot={rot_deg:.0f}° "
                  f"native_center=({bcenter_xy[0]:.0f},{bcenter_xy[1]:.0f}) "
                  f"offset=({offset_x:.0f},{offset_y:.0f},{offset_z:.1f})")

    print(f"\n  Scene bounds after placement:")
    print(f"    min: ({scene_bmin[0]:.0f}, {scene_bmin[1]:.0f}, {scene_bmin[2]:.1f})")
    print(f"    max: ({scene_bmax[0]:.0f}, {scene_bmax[1]:.0f}, {scene_bmax[2]:.1f})")
    return all_objs


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def setup_camera(town_size, mode="drone"):
    tw, th = town_size
    cx, cy = tw / 2, th / 2

    if mode == "drone":
        # Elevated drone view, looking down at ~45 degrees
        cam_loc = (cx - tw * 0.3, cy - th * 0.3, max(tw, th) * 0.35)
        target = (cx, cy, 0)
        lens = 28
    elif mode == "street":
        # Street level view looking into the town
        cam_loc = (30, cy, 1.7)
        target = (cx * 0.5, cy, 8)
        lens = 28
    elif mode == "closeup":
        # Mid-range cinematic — 3/4 aerial of main intersection (330, 293)
        # Elevated enough to see road surface + crosswalks, low enough for building detail
        cam_loc = (280, 240, 45)
        target = (345, 310, 0)
        lens = 50
    else:  # overview
        # High overhead
        cam_loc = (cx, cy - th * 0.1, max(tw, th) * 0.9)
        target = (cx, cy, 0)
        lens = 20

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam

    empty = bpy.data.objects.new("CamTarget", None)
    empty.location = target
    bpy.context.collection.objects.link(empty)

    constraint = cam.constraints.new('TRACK_TO')
    constraint.target = empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    cam.data.lens = lens
    cam.data.clip_end = 2000
    return cam


def setup_lights(town_size):
    tw, th = town_size
    cx, cy = tw / 2, th / 2

    bpy.ops.object.light_add(type='SUN',
                              location=(cx + tw, cy - th, max(tw, th)))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(1.5)
    sun.data.color = (1.0, 0.95, 0.85)
    sun.rotation_euler = (math.radians(55), math.radians(10),
                          math.radians(-20))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def render_town():
    scene_path = os.path.abspath(args.scene)
    scene_dir = os.path.dirname(scene_path)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    with open(scene_path) as f:
        scene = json.load(f)

    town_size = scene["town_size"]
    buildings = scene["buildings"]
    infra = scene["infrastructure"]
    stats = scene["stats"]

    print("=" * 60)
    print("Phase 10 — Town Renderer")
    print(f"  Town: {town_size[0]:.0f} x {town_size[1]:.0f}m")
    print(f"  Buildings: {stats['n_buildings']}")
    print(f"  Trees: {stats['n_trees']}")
    print("=" * 60)

    # Setup
    clear_scene()
    setup_render()
    setup_sky()

    # Create materials
    print("\nCreating materials...")
    materials = {
        "asphalt": mat_asphalt(),
        "concrete": mat_concrete(),
        "sidewalk": mat_sidewalk(),
        "tree_trunk": mat_tree_trunk(),
        "tree_crown": mat_tree_crown(),
        "ground": mat_ground(),
        "road_marking": mat_road_marking(),
    }

    # Import infrastructure
    print("\nImporting infrastructure...")
    import_infrastructure(scene_dir, infra, materials, town_size)

    # Import buildings
    print(f"\nImporting {len(buildings)} buildings...")
    import_buildings(scene_dir, buildings)

    # Camera and lights
    setup_camera(town_size, mode=args.camera)
    setup_lights(town_size)

    # Render
    out_path = os.path.join(output_dir, f"town_{args.camera}.png")
    print(f"\nRendering to {out_path}...")
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)

    print("Done!")


if __name__ == "__main__":
    render_town()
