"""Phase 11 — Fast Cycles renderer with incremental scene updates.

Optimizations over blender_render.py:
  1. Binary PLY loading (no OBJ text parsing — 10-50x faster I/O)
  2. Static scene persists (building, ground, camera, lights built once)
  3. Only water mesh geometry swapped per frame
  4. Cycles persistent_data keeps BVH/textures in GPU memory
  5. Lower default samples (64) + OIDN denoiser = same quality, less time

Usage:
    blender --background --python blender_render_fast.py -- \
        --meshes ./export/flood/meshes_gpu --frame 150 --output ./renders_fast
"""
import bpy
import math
import os
import sys
import json
import time
import struct
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

import argparse
p = argparse.ArgumentParser()
p.add_argument("--meshes", default="./export/flood/meshes_gpu")
p.add_argument("--meta", default="./export/flood/meta.json")
p.add_argument("--frame", default=None)
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=None)
p.add_argument("--step", type=int, default=1)
p.add_argument("--output", default="./renders_fast")
p.add_argument("--samples", type=int, default=64)
p.add_argument("--resolution", default="1920x1080")
p.add_argument("--cpu", action="store_true")
p.add_argument("--gpus", type=int, default=4, help="Max GPUs to use")
p.add_argument("--foam-dir", default=None,
               help="Directory with foam_NNNNNN.npy (default: same as --meshes)")
args = p.parse_args(argv)

RES_X, RES_Y = map(int, args.resolution.split("x"))
BL_VERSION = bpy.app.version
BL_IS_3X = BL_VERSION[0] < 4
MODELS_DIR = os.path.expanduser("~/Downloads/models")

print(f"  Blender {BL_VERSION[0]}.{BL_VERSION[1]}.{BL_VERSION[2]}")


# ---------------------------------------------------------------------------
# Fast binary PLY loader
# ---------------------------------------------------------------------------
def load_ply_mesh(ply_path):
    """Load binary little-endian PLY triangle mesh.

    Returns (verts, faces) as numpy arrays:
      verts: (V, 3) float32
      faces: (F, 3) int32
    """
    with open(ply_path, "rb") as f:
        # Parse header
        n_verts = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        if n_verts == 0:
            return None, None

        # Read vertex block: N x 3 float32 = N x 12 bytes
        verts = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(-1, 3).copy()

        # Read face block: each face = 1 byte (count=3) + 3 x 4 bytes (int32) = 13 bytes
        if n_faces > 0:
            face_raw = f.read(n_faces * 13)
            faces = np.zeros((n_faces, 3), dtype=np.int32)
            for i in range(n_faces):
                off = i * 13
                # skip byte 0 (count=3)
                faces[i, 0] = struct.unpack_from("<i", face_raw, off + 1)[0]
                faces[i, 1] = struct.unpack_from("<i", face_raw, off + 5)[0]
                faces[i, 2] = struct.unpack_from("<i", face_raw, off + 9)[0]
        else:
            faces = None

    return verts, faces


def load_ply_mesh_fast(ply_path):
    """Load binary PLY — vectorized face parsing."""
    with open(ply_path, "rb") as f:
        n_verts = 0
        n_faces = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        if n_verts == 0:
            return None, None

        # Vertices: contiguous float32 block
        verts = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(-1, 3).copy()

        # Faces: each = 1 byte (uchar count) + 3 x int32 = 13 bytes
        if n_faces > 0:
            raw = np.frombuffer(f.read(n_faces * 13), dtype=np.uint8).reshape(n_faces, 13)
            # Extract the 3 int32s starting at byte 1, 5, 9
            faces = np.zeros((n_faces, 3), dtype=np.int32)
            faces[:, 0] = np.frombuffer(raw[:, 1:5].tobytes(), dtype=np.int32)
            faces[:, 1] = np.frombuffer(raw[:, 5:9].tobytes(), dtype=np.int32)
            faces[:, 2] = np.frombuffer(raw[:, 9:13].tobytes(), dtype=np.int32)
        else:
            faces = None

    return verts, faces


# ---------------------------------------------------------------------------
# Scene setup (called ONCE)
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
                        gpu_count = 0
                        for d in prefs.preferences.devices:
                            if d.type == device_type and gpu_count < args.gpus:
                                d.use = True
                                gpu_count += 1
                            else:
                                d.use = False
                        print(f"  Render device: {device_type} ({gpu_count} GPUs)")
                        break
                except Exception:
                    continue

    # Tone mapping
    if BL_IS_3X:
        sc.view_settings.view_transform = 'Filmic'
        sc.view_settings.look = 'Medium High Contrast'
    else:
        try:
            sc.view_settings.view_transform = 'AgX'
            sc.view_settings.look = 'AgX - Medium High Contrast'
        except Exception:
            sc.view_settings.view_transform = 'Filmic'
            sc.view_settings.look = 'Medium High Contrast'

    # Light path
    sc.cycles.max_bounces = 12
    sc.cycles.transmission_bounces = 4
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.glossy_bounces = 6
    sc.cycles.diffuse_bounces = 4
    sc.cycles.volume_bounces = 2

    # Shadow caustics
    if hasattr(sc.cycles, 'caustics_refractive'):
        sc.cycles.caustics_refractive = True
        sc.cycles.caustics_reflective = False

    # Path guiding
    if hasattr(sc.cycles, 'use_guiding'):
        sc.cycles.use_guiding = True

    # Adaptive sampling
    if hasattr(sc.cycles, 'use_adaptive_sampling'):
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.01

    # Denoising (OIDN — lets us use fewer samples)
    sc.cycles.use_denoising = True

    # *** PERSISTENT DATA — key optimization ***
    # Keeps BVH + textures in GPU memory between frames
    sc.render.use_persistent_data = True

    sc.render.film_transparent = False

    # Sky
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    N = world.node_tree.nodes
    L = world.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputWorld"); out.location = (600, 0)
    bg = N.new("ShaderNodeBackground"); bg.location = (400, 0)
    bg.inputs["Strength"].default_value = 2.5
    sky = N.new("ShaderNodeTexSky"); sky.location = (0, 0)
    sky.sky_type = 'HOSEK_WILKIE'
    sky.sun_direction = (0.50, -0.30, 0.64)
    sky.turbidity = 3.0
    L.new(sky.outputs["Color"], bg.inputs["Color"])
    L.new(bg.outputs["Background"], out.inputs["Surface"])

    print(f"  Cycles: {args.samples} samples, adaptive, OIDN denoiser, persistent data")


# ---------------------------------------------------------------------------
# Building import (called ONCE)
# ---------------------------------------------------------------------------
def import_building(meta):
    obj_file = meta["building_obj"]
    obj_path = os.path.join(MODELS_DIR, obj_file)

    if not os.path.exists(obj_path):
        print(f"  ERROR: Building OBJ not found: {obj_path}")
        return [], None, None, None

    if BL_IS_3X:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='Y', axis_up='Z')
    else:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')

    objects = list(bpy.context.selected_objects)
    meshes = [o for o in objects if o.type == 'MESH']

    obj_dir = os.path.dirname(obj_path)
    for obj in meshes:
        for mat_slot in obj.material_slots:
            m = mat_slot.material
            if m and m.use_nodes:
                for node in m.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        img = node.image
                        if not os.path.exists(img.filepath_from_user()):
                            candidate = os.path.join(obj_dir, os.path.basename(img.filepath))
                            if os.path.exists(candidate):
                                img.filepath = candidate
                                img.reload()

    bpy.context.view_layer.update()
    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)
    verts = np.array([(v.x, v.y, v.z) for v in all_verts])
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    bcenter = verts.mean(axis=0)

    print(f"  Building: {len(meshes)} meshes, {len(verts):,} verts")
    return objects, bmin, bmax, bcenter


# ---------------------------------------------------------------------------
# Water material
# ---------------------------------------------------------------------------
def mat_water(edge_bounds=None):
    mat = bpy.data.materials.new("FloodWater")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial"); out.location = (1200, 0)

    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 200)
    bsdf.inputs["Base Color"].default_value = (0.10, 0.07, 0.03, 1)
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Metallic"].default_value = 0.0
    if BL_IS_3X:
        bsdf.inputs["Transmission"].default_value = 0.15
    else:
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.15
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = 0.15
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.15
        bsdf.inputs["Coat Roughness"].default_value = 0.02
        bsdf.inputs["Coat IOR"].default_value = 1.5

    # Foam BSDF
    foam_bsdf = N.new("ShaderNodeBsdfPrincipled"); foam_bsdf.location = (300, -200)
    foam_bsdf.inputs["Base Color"].default_value = (0.85, 0.80, 0.72, 1)
    foam_bsdf.inputs["Roughness"].default_value = 0.85

    mix_sh = N.new("ShaderNodeMixShader"); mix_sh.location = (700, 100)
    L.new(bsdf.outputs["BSDF"], mix_sh.inputs[1])
    L.new(foam_bsdf.outputs["BSDF"], mix_sh.inputs[2])

    foam_attr = N.new("ShaderNodeAttribute"); foam_attr.location = (-200, -200)
    foam_attr.attribute_name = "foam"
    foam_attr.attribute_type = 'GEOMETRY'
    foam_ramp = N.new("ShaderNodeMapRange"); foam_ramp.location = (0, -200)
    foam_ramp.inputs["From Min"].default_value = 0.03
    foam_ramp.inputs["From Max"].default_value = 0.4
    foam_ramp.inputs["To Min"].default_value = 0.0
    foam_ramp.inputs["To Max"].default_value = 1.0
    foam_ramp.clamp = True
    L.new(foam_attr.outputs["Fac"], foam_ramp.inputs["Value"])
    L.new(foam_ramp.outputs["Result"], mix_sh.inputs["Fac"])

    # Multi-scale bump
    texcoord = N.new("ShaderNodeTexCoord"); texcoord.location = (-800, -400)
    noise1 = N.new("ShaderNodeTexNoise"); noise1.location = (-500, -300)
    noise1.inputs["Scale"].default_value = 8.0
    noise1.inputs["Detail"].default_value = 3.0
    noise1.inputs["Roughness"].default_value = 0.5
    L.new(texcoord.outputs["Object"], noise1.inputs["Vector"])

    noise2 = N.new("ShaderNodeTexNoise"); noise2.location = (-500, -500)
    noise2.inputs["Scale"].default_value = 25.0
    noise2.inputs["Detail"].default_value = 6.0
    noise2.inputs["Roughness"].default_value = 0.55
    L.new(texcoord.outputs["Object"], noise2.inputs["Vector"])

    noise3 = N.new("ShaderNodeTexNoise"); noise3.location = (-500, -700)
    noise3.inputs["Scale"].default_value = 80.0
    noise3.inputs["Detail"].default_value = 8.0
    noise3.inputs["Roughness"].default_value = 0.6
    L.new(texcoord.outputs["Object"], noise3.inputs["Vector"])

    if BL_IS_3X:
        mix_12 = N.new("ShaderNodeMixRGB"); mix_12.location = (-250, -400)
        mix_12.inputs["Fac"].default_value = 0.35
        L.new(noise1.outputs["Fac"], mix_12.inputs["Color1"])
        L.new(noise2.outputs["Fac"], mix_12.inputs["Color2"])
        mix_123 = N.new("ShaderNodeMixRGB"); mix_123.location = (-100, -500)
        mix_123.inputs["Fac"].default_value = 0.15
        L.new(mix_12.outputs["Color"], mix_123.inputs["Color1"])
        L.new(noise3.outputs["Fac"], mix_123.inputs["Color2"])
        bump_input = mix_123.outputs["Color"]
    else:
        mix_12 = N.new("ShaderNodeMix"); mix_12.location = (-250, -400)
        mix_12.data_type = 'FLOAT'
        mix_12.inputs["Factor"].default_value = 0.35
        L.new(noise1.outputs["Fac"], mix_12.inputs["A"])
        L.new(noise2.outputs["Fac"], mix_12.inputs["B"])
        mix_123 = N.new("ShaderNodeMix"); mix_123.location = (-100, -500)
        mix_123.data_type = 'FLOAT'
        mix_123.inputs["Factor"].default_value = 0.15
        L.new(mix_12.outputs["Result"], mix_123.inputs["A"])
        L.new(noise3.outputs["Fac"], mix_123.inputs["B"])
        bump_input = mix_123.outputs["Result"]

    bump = N.new("ShaderNodeBump"); bump.location = (100, -450)
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.01
    L.new(bump_input, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Edge fade
    if edge_bounds is not None:
        x_lo, x_hi, y_lo, y_hi = edge_bounds
        fade_w = 0.15

        transparent = N.new("ShaderNodeBsdfTransparent"); transparent.location = (700, -100)
        mix_edge = N.new("ShaderNodeMixShader"); mix_edge.location = (1000, 0)

        tc = N.new("ShaderNodeTexCoord"); tc.location = (-600, 300)
        sep = N.new("ShaderNodeSeparateXYZ"); sep.location = (-400, 300)
        L.new(tc.outputs["Object"], sep.inputs["Vector"])

        x_fade_lo = N.new("ShaderNodeMapRange"); x_fade_lo.location = (-200, 400)
        x_fade_lo.inputs["From Min"].default_value = x_lo
        x_fade_lo.inputs["From Max"].default_value = x_lo + fade_w
        x_fade_lo.inputs["To Min"].default_value = 0.0
        x_fade_lo.inputs["To Max"].default_value = 1.0
        x_fade_lo.clamp = True
        L.new(sep.outputs["X"], x_fade_lo.inputs["Value"])

        x_fade_hi = N.new("ShaderNodeMapRange"); x_fade_hi.location = (-200, 300)
        x_fade_hi.inputs["From Min"].default_value = x_hi - fade_w
        x_fade_hi.inputs["From Max"].default_value = x_hi
        x_fade_hi.inputs["To Min"].default_value = 1.0
        x_fade_hi.inputs["To Max"].default_value = 0.0
        x_fade_hi.clamp = True
        L.new(sep.outputs["X"], x_fade_hi.inputs["Value"])

        y_fade_lo = N.new("ShaderNodeMapRange"); y_fade_lo.location = (-200, 200)
        y_fade_lo.inputs["From Min"].default_value = y_lo
        y_fade_lo.inputs["From Max"].default_value = y_lo + fade_w
        y_fade_lo.inputs["To Min"].default_value = 0.0
        y_fade_lo.inputs["To Max"].default_value = 1.0
        y_fade_lo.clamp = True
        L.new(sep.outputs["Y"], y_fade_lo.inputs["Value"])

        y_fade_hi = N.new("ShaderNodeMapRange"); y_fade_hi.location = (-200, 100)
        y_fade_hi.inputs["From Min"].default_value = y_hi - fade_w
        y_fade_hi.inputs["From Max"].default_value = y_hi
        y_fade_hi.inputs["To Min"].default_value = 1.0
        y_fade_hi.inputs["To Max"].default_value = 0.0
        y_fade_hi.clamp = True
        L.new(sep.outputs["Y"], y_fade_hi.inputs["Value"])

        x_min = N.new("ShaderNodeMath"); x_min.location = (0, 350)
        x_min.operation = 'MINIMUM'
        L.new(x_fade_lo.outputs["Result"], x_min.inputs[0])
        L.new(x_fade_hi.outputs["Result"], x_min.inputs[1])

        y_min = N.new("ShaderNodeMath"); y_min.location = (0, 150)
        y_min.operation = 'MINIMUM'
        L.new(y_fade_lo.outputs["Result"], y_min.inputs[0])
        L.new(y_fade_hi.outputs["Result"], y_min.inputs[1])

        xy_min = N.new("ShaderNodeMath"); xy_min.location = (200, 250)
        xy_min.operation = 'MINIMUM'
        L.new(x_min.outputs["Value"], xy_min.inputs[0])
        L.new(y_min.outputs["Value"], xy_min.inputs[1])

        L.new(xy_min.outputs["Value"], mix_edge.inputs["Fac"])
        L.new(transparent.outputs["BSDF"], mix_edge.inputs[1])
        L.new(mix_sh.outputs["Shader"], mix_edge.inputs[2])
        L.new(mix_edge.outputs["Shader"], out.inputs["Surface"])

        if hasattr(mat, 'blend_method'):
            mat.blend_method = 'HASHED'
    else:
        L.new(mix_sh.outputs["Shader"], out.inputs["Surface"])

    # Volume absorption
    vol_abs = N.new("ShaderNodeVolumeAbsorption"); vol_abs.location = (800, -300)
    vol_abs.inputs["Color"].default_value = (0.12, 0.18, 0.08, 1)
    vol_abs.inputs["Density"].default_value = 3.0
    L.new(vol_abs.outputs["Volume"], out.inputs["Volume"])

    return mat


def mat_flood_base():
    mat = bpy.data.materials.new("FloodBase")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Base Color"].default_value = (0.07, 0.05, 0.02, 1)
    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["IOR"].default_value = 1.333
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = N.new("ShaderNodeTexCoord"); texcoord.location = (-500, -300)
    noise = N.new("ShaderNodeTexNoise"); noise.location = (-300, -300)
    noise.inputs["Scale"].default_value = 10.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.5
    L.new(texcoord.outputs["Object"], noise.inputs["Vector"])
    bump = N.new("ShaderNodeBump"); bump.location = (100, -300)
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.008
    L.new(noise.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# ---------------------------------------------------------------------------
# Static scene elements (called ONCE)
# ---------------------------------------------------------------------------
def create_ground(center_xy, z=0.0, size=10.0):
    center = (center_xy[0], center_xy[1], z - 0.001)
    bpy.ops.mesh.primitive_plane_add(size=size, location=center)
    ground = bpy.context.active_object
    ground.name = "Ground"
    mat = bpy.data.materials.new("Pavement")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.08, 0.06, 0.04, 1)
    bsdf.inputs["Roughness"].default_value = 0.65
    ground.data.materials.append(mat)
    return ground


def setup_camera(building_center, building_size):
    cx, cy, cz = building_center
    d = building_size * 2.5
    cam_loc = (cx - 0.65 * d, cy - 0.65 * d, cz + 0.40 * d)
    target = (cx, cy, cz * 0.3)

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

    cam.data.lens = 24
    cam.data.clip_end = 500
    return cam


def setup_lights(center, size):
    cx, cy, cz = center
    s = size

    bpy.ops.object.light_add(type='SUN', location=(cx + s * 3, cy - s * 2, cz + s * 4))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 5.0
    sun.data.angle = math.radians(1.5)
    sun.data.color = (1.0, 0.95, 0.88)
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
    if hasattr(sun.data, 'shadow_caustics'):
        sun.data.shadow_caustics = True

    bpy.ops.object.light_add(type='AREA',
                              location=(cx - s * 2, cy + s * 2, cz + s))
    fill = bpy.context.active_object
    fill.name = "Fill"
    fill.data.energy = 50
    fill.data.size = s * 4
    fill.data.color = (0.85, 0.90, 1.0)


# ---------------------------------------------------------------------------
# Flood base (updated per frame — just move Z)
# ---------------------------------------------------------------------------
flood_base_obj = None
_flood_base_mat = None


def create_or_update_flood_base(water_z, center_xy, size=8.0):
    global flood_base_obj, _flood_base_mat
    if _flood_base_mat is None:
        _flood_base_mat = mat_flood_base()
    if flood_base_obj is None:
        bpy.ops.mesh.primitive_plane_add(size=size,
                                          location=(center_xy[0], center_xy[1], water_z))
        flood_base_obj = bpy.context.active_object
        flood_base_obj.name = "FloodBase"
        flood_base_obj.data.materials.append(_flood_base_mat)
    else:
        flood_base_obj.location.z = water_z


# ---------------------------------------------------------------------------
# Water mesh — FAST in-place update from binary PLY
# ---------------------------------------------------------------------------
water_obj = None
water_mat = None
_inv_scale = 1.0
_inv_offset = (0.0, 0.0, 0.0)
_edge_bounds = None


def _build_mesh_from_arrays(name, verts_np, faces_np):
    """Build Blender mesh from numpy arrays using foreach_set (fast)."""
    mesh = bpy.data.meshes.new(name)
    n_verts = len(verts_np)
    n_faces = len(faces_np)
    n_loops = n_faces * 3

    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", verts_np.ravel())
    mesh.loops.add(n_loops)
    mesh.loops.foreach_set("vertex_index", faces_np.ravel())
    mesh.polygons.add(n_faces)
    mesh.polygons.foreach_set("loop_start", np.arange(0, n_loops, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(n_faces, 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(n_faces, dtype=np.bool_))
    mesh.update()
    mesh.validate()
    return mesh


def _apply_foam(mesh, foam_data):
    """Apply foam vertex colors."""
    if foam_data is None or len(foam_data) == 0:
        return
    n_verts = len(mesh.vertices)
    if len(foam_data) != n_verts:
        return

    if BL_IS_3X:
        if "foam" not in mesh.vertex_colors:
            mesh.vertex_colors.new(name="foam")
        vcol = mesh.vertex_colors["foam"]
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vi = mesh.loops[loop_idx].vertex_index
                f = foam_data[vi] if vi < len(foam_data) else 0.0
                vcol.data[loop_idx].color = (f, f, f, 1.0)
    else:
        if "foam" not in mesh.color_attributes:
            mesh.color_attributes.new(name="foam", type='FLOAT_COLOR', domain='POINT')
        attr = mesh.color_attributes["foam"]
        n = len(attr.data)
        if n == n_verts:
            colors = np.zeros(n * 4, dtype=np.float32)
            colors[0::4] = foam_data
            colors[1::4] = foam_data
            colors[2::4] = foam_data
            colors[3::4] = 1.0
            attr.data.foreach_set("color", colors)


def update_water(ply_path, foam_path=None):
    """Load binary PLY and swap water mesh geometry. Fast path."""
    global water_obj, water_mat

    t0 = time.time()

    if water_mat is None:
        water_mat = mat_water(edge_bounds=_edge_bounds)

    if not os.path.exists(ply_path):
        return None

    # Fast binary PLY load
    verts, faces = load_ply_mesh_fast(ply_path)
    if verts is None or faces is None:
        return None
    t_load = time.time() - t0

    # Load foam
    foam_data = None
    if foam_path and os.path.exists(foam_path):
        foam_data = np.load(foam_path)

    # No coordinate transform needed — PLY meshes from gpu_mesh_warp.py
    # are already in native world coords (transform applied in Warp kernel)

    t_mesh = time.time()
    if water_obj is not None:
        # HOT PATH: swap mesh data in-place
        old_mesh = water_obj.data
        new_mesh = _build_mesh_from_arrays("WaterTmp", verts, faces)
        new_mesh.materials.clear()
        new_mesh.materials.append(water_mat)
        if foam_data is not None:
            _apply_foam(new_mesh, foam_data)
        water_obj.data = new_mesh
        bpy.data.meshes.remove(old_mesh)
    else:
        # First frame: create object + modifiers (once)
        mesh = _build_mesh_from_arrays("WaterMesh", verts, faces)
        mesh.materials.clear()
        mesh.materials.append(water_mat)
        if foam_data is not None:
            _apply_foam(mesh, foam_data)

        water_obj = bpy.data.objects.new("WaterSurface", mesh)
        bpy.context.scene.collection.objects.link(water_obj)

        # Smooth + SubSurf (created once, persist across frames)
        mod = water_obj.modifiers.new("Smooth", 'SMOOTH')
        mod.factor = 0.5
        mod.iterations = 3

        sub = water_obj.modifiers.new("SubSurf", 'SUBSURF')
        sub.levels = 0
        sub.render_levels = 1

    t_mesh = time.time() - t_mesh
    t_total = time.time() - t0

    n_v = len(verts)
    n_f = len(faces)
    print(f"    Water: {n_v:,} verts, {n_f:,} tris  "
          f"(load={t_load:.3f}s mesh={t_mesh:.3f}s total={t_total:.3f}s)")

    return water_obj


def get_water_level_z(obj):
    if obj is None:
        return None
    verts = obj.data.vertices
    if len(verts) == 0:
        return None
    n = len(verts)
    zs = np.empty(n, dtype=np.float32)
    # foreach_get is faster than list comprehension
    cos = np.empty(n * 3, dtype=np.float32)
    verts.foreach_get("co", cos)
    zs = cos[2::3]
    return float(np.percentile(zs, 80))


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------
def render_frames():
    meshes_dir = os.path.abspath(args.meshes)
    output_dir = os.path.abspath(args.output)
    meta_path = os.path.abspath(args.meta)
    foam_dir = os.path.abspath(args.foam_dir) if args.foam_dir else meshes_dir

    with open(meta_path) as f:
        meta = json.load(f)

    # Frame list
    if args.frame:
        frame_list = [int(f.strip()) for f in args.frame.split(",")]
    else:
        ply_files = sorted(f for f in os.listdir(meshes_dir)
                           if f.startswith("water_") and f.endswith(".ply"))
        all_frames = [int(f.split("_")[1].split(".")[0]) for f in ply_files]
        end = args.end if args.end is not None else max(all_frames) if all_frames else 0
        frame_list = [f for f in all_frames if args.start <= f <= end][::args.step]

    if not frame_list:
        frame_list = [0]

    print("=" * 60)
    print("Phase 11 — Fast Incremental Cycles Renderer")
    print(f"  Meshes: {meshes_dir} (binary PLY)")
    print(f"  Frames: {len(frame_list)} ({frame_list[0]}..{frame_list[-1]})")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {args.samples}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # STATIC SCENE SETUP (done ONCE)
    # ==========================================
    t_setup = time.time()
    clear_scene()
    setup_render()

    print("\n--- Static scene setup (one-time) ---")
    building_objs, bmin, bmax, bcenter = import_building(meta)
    if bmin is None:
        print("ERROR: No building loaded")
        return
    building_size = (bmax - bmin).max()

    create_ground(center_xy=(bcenter[0], bcenter[1]), z=bmin[2], size=building_size * 8)

    global _inv_scale, _inv_offset, _edge_bounds
    _inv_scale = meta["transform"]["inv_scale"]
    _inv_offset = tuple(meta["transform"]["inv_offset"])

    sim_edge = 0.08
    sim_inner = 0.92
    _edge_bounds = (
        sim_edge * _inv_scale + _inv_offset[0],
        sim_inner * _inv_scale + _inv_offset[0],
        sim_edge * _inv_scale + _inv_offset[1],
        sim_inner * _inv_scale + _inv_offset[1],
    )

    setup_camera(bcenter, building_size)
    setup_lights(bcenter, building_size)

    t_setup = time.time() - t_setup
    print(f"--- Setup done in {t_setup:.2f}s ---\n")

    # ==========================================
    # RENDER LOOP (only water changes)
    # ==========================================
    t_all = time.time()
    n_rendered = 0

    for fi, frame in enumerate(frame_list):
        out_path = os.path.join(output_dir, f"frame_{frame:06d}.png")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f"  [{fi+1}/{len(frame_list)}] Frame {frame} — skip (exists)")
            continue

        print(f"  [{fi+1}/{len(frame_list)}] Frame {frame}")

        # Only thing that changes: water mesh
        ply_path = os.path.join(meshes_dir, f"water_{frame:06d}.ply")
        foam_path = os.path.join(foam_dir, f"foam_{frame:06d}.npy")
        w = update_water(ply_path, foam_path=foam_path)

        if w:
            wz = get_water_level_z(w)
            if wz is not None:
                create_or_update_flood_base(wz, (bcenter[0], bcenter[1]),
                                             size=building_size * 8)

        t_render = time.time()
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t_render
        n_rendered += 1
        print(f"    Render: {dt:.2f}s -> {out_path}")

    total_time = time.time() - t_all
    if n_rendered > 0:
        avg = total_time / n_rendered
        print(f"\nDone! {n_rendered} frames in {total_time:.1f}s ({avg:.1f}s/frame)")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    render_frames()
