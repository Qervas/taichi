"""Phase 9 — Blender Cycles renderer for 3D MPM flood.

Imports building GLB + water mesh OBJ sequence, renders cinematic frames.
Photorealistic flood water with mirror reflections, procedural sky dome.

Usage:
    blender --background --python blender_render.py -- \
        --meshes ./export/meshes --frame 0,30,60,90 --output ./renders
"""
import bpy
import math
import os
import sys
import json
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

import argparse
p = argparse.ArgumentParser()
p.add_argument("--meshes", default="./export/meshes")
p.add_argument("--meta", default="./export/meta.json")
p.add_argument("--frame", default=None)
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=None)
p.add_argument("--step", type=int, default=1)
p.add_argument("--output", default="./renders")
p.add_argument("--samples", type=int, default=128)
p.add_argument("--resolution", default="1920x1080")
p.add_argument("--cpu", action="store_true", help="Force CPU rendering")
args = p.parse_args(argv)

RES_X, RES_Y = map(int, args.resolution.split("x"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# Blender version compatibility (3.6 vs 4.x+)
BL_VERSION = bpy.app.version
BL_IS_3X = BL_VERSION[0] < 4
print(f"  Blender {BL_VERSION[0]}.{BL_VERSION[1]}.{BL_VERSION[2]}"
      f" ({'3.x compat' if BL_IS_3X else '4.x+'})")


def _set_coat(bsdf, weight, ior, roughness):
    """Set clearcoat/coat params — handles 3.x vs 4.x naming."""
    if BL_IS_3X:
        bsdf.inputs["Clearcoat"].default_value = weight
        bsdf.inputs["Clearcoat Roughness"].default_value = roughness
        # Blender 3.x has no Clearcoat IOR input
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
        print(f"  Render device: CPU")
    else:
        sc.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            # Try OptiX first (faster on RTX/Ampere), fall back to CUDA
            for device_type in ('OPTIX', 'CUDA'):
                try:
                    prefs.preferences.compute_device_type = device_type
                    prefs.preferences.get_devices()
                    devices = [d for d in prefs.preferences.devices
                               if d.type == device_type]
                    if devices:
                        for d in prefs.preferences.devices:
                            d.use = (d.type == device_type)
                        print(f"  Render device: {device_type} ({len(devices)} GPUs)")
                        break
                except Exception:
                    continue

    # Denoiser
    sc.cycles.use_denoising = True

    # Persistent data — cache unchanged objects (building, ground) between frames
    sc.render.use_persistent_data = True

    # Film
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'Filmic'
    sc.view_settings.look = 'Medium High Contrast'

    # World: procedural sky dome — Hosek-Wilkie sky with warm sun
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    N = world.node_tree.nodes
    L = world.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputWorld"); out.location = (600, 0)
    bg = N.new("ShaderNodeBackground"); bg.location = (400, 0)
    bg.inputs["Strength"].default_value = 1.5

    # Sky texture — overcast/hazy for realistic flood conditions
    sky = N.new("ShaderNodeTexSky"); sky.location = (0, 0)
    sky.sky_type = 'HOSEK_WILKIE'
    sky.sun_direction = (0.4, -0.2, 0.55)  # late afternoon sun angle
    sky.turbidity = 5.0  # hazy/overcast — flood conditions
    L.new(sky.outputs["Color"], bg.inputs["Color"])
    L.new(bg.outputs["Background"], out.inputs["Surface"])


def import_building(meta):
    """Import building GLB at native coordinates (no repositioning)."""
    glb_path = os.path.join(C.ASSETS_DIR, meta["building_glb"])
    bpy.ops.import_scene.gltf(filepath=glb_path)
    objects = list(bpy.context.selected_objects)
    meshes = [o for o in objects if o.type == 'MESH']

    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)

    verts = np.array([(v.x, v.y, v.z) for v in all_verts])
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    bcenter = verts.mean(axis=0)

    print(f"  Building: {len(meshes)} meshes (native coords)")
    print(f"  Bounds: {bmin.round(4)} → {bmax.round(4)}")
    print(f"  Center: {bcenter.round(4)}")
    return objects, bmin, bmax, bcenter


# ---------------------------------------------------------------------------
# Photorealistic flood water material
# ---------------------------------------------------------------------------
def mat_water(edge_bounds=None):
    """Turbid flood water — muddy brown, mirror-reflective, opaque.

    Based on real flood reference photos:
    - Murky brown/olive color (NOT blue)
    - Very smooth, near-perfect mirror reflections
    - Opaque — can't see the bottom
    - Edge fade: transparent near domain boundaries → blends into flood base plane

    edge_bounds: (x_min, x_max, y_min, y_max) in native coords — fade region
    """
    mat = bpy.data.materials.new("FloodWater")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial"); out.location = (800, 0)

    # Principled BSDF — dielectric water with murky color
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Base Color"].default_value = (0.06, 0.045, 0.025, 1)
    bsdf.inputs["Roughness"].default_value = 0.02
    bsdf.inputs["IOR"].default_value = 1.33
    bsdf.inputs["Metallic"].default_value = 0.0
    _set_coat(bsdf, 0.8, 1.5, 0.015)

    # Very subtle bump
    texcoord = N.new("ShaderNodeTexCoord"); texcoord.location = (-500, -300)
    noise1 = N.new("ShaderNodeTexNoise"); noise1.location = (-300, -200)
    noise1.inputs["Scale"].default_value = 15.0
    noise1.inputs["Detail"].default_value = 3.0
    noise1.inputs["Roughness"].default_value = 0.4
    L.new(texcoord.outputs["Object"], noise1.inputs["Vector"])

    noise2 = N.new("ShaderNodeTexNoise"); noise2.location = (-300, -400)
    noise2.inputs["Scale"].default_value = 40.0
    noise2.inputs["Detail"].default_value = 5.0
    noise2.inputs["Roughness"].default_value = 0.5
    L.new(texcoord.outputs["Object"], noise2.inputs["Vector"])

    mix_n = N.new("ShaderNodeMix"); mix_n.location = (-100, -300)
    mix_n.data_type = 'FLOAT'
    mix_n.inputs["Factor"].default_value = 0.3
    L.new(noise1.outputs["Fac"], mix_n.inputs["A"])
    L.new(noise2.outputs["Fac"], mix_n.inputs["B"])

    bump = N.new("ShaderNodeBump"); bump.location = (100, -300)
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.005
    L.new(mix_n.outputs["Result"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Edge fade: transparent near domain boundaries
    if edge_bounds is not None:
        x_lo, x_hi, y_lo, y_hi = edge_bounds
        fade_w = 0.15  # fade width in native coords

        transparent = N.new("ShaderNodeBsdfTransparent"); transparent.location = (300, -200)
        mix_sh = N.new("ShaderNodeMixShader"); mix_sh.location = (600, 0)

        # Object coords → separate XY
        tc = N.new("ShaderNodeTexCoord"); tc.location = (-500, 200)
        sep = N.new("ShaderNodeSeparateXYZ"); sep.location = (-300, 200)
        L.new(tc.outputs["Object"], sep.inputs["Vector"])

        # X fade: 1 in center, 0 at edges
        x_fade_lo = N.new("ShaderNodeMapRange"); x_fade_lo.location = (-100, 300)
        x_fade_lo.inputs["From Min"].default_value = x_lo
        x_fade_lo.inputs["From Max"].default_value = x_lo + fade_w
        x_fade_lo.inputs["To Min"].default_value = 0.0
        x_fade_lo.inputs["To Max"].default_value = 1.0
        x_fade_lo.clamp = True
        L.new(sep.outputs["X"], x_fade_lo.inputs["Value"])

        x_fade_hi = N.new("ShaderNodeMapRange"); x_fade_hi.location = (-100, 200)
        x_fade_hi.inputs["From Min"].default_value = x_hi - fade_w
        x_fade_hi.inputs["From Max"].default_value = x_hi
        x_fade_hi.inputs["To Min"].default_value = 1.0
        x_fade_hi.inputs["To Max"].default_value = 0.0
        x_fade_hi.clamp = True
        L.new(sep.outputs["X"], x_fade_hi.inputs["Value"])

        # Y fade
        y_fade_lo = N.new("ShaderNodeMapRange"); y_fade_lo.location = (-100, 100)
        y_fade_lo.inputs["From Min"].default_value = y_lo
        y_fade_lo.inputs["From Max"].default_value = y_lo + fade_w
        y_fade_lo.inputs["To Min"].default_value = 0.0
        y_fade_lo.inputs["To Max"].default_value = 1.0
        y_fade_lo.clamp = True
        L.new(sep.outputs["Y"], y_fade_lo.inputs["Value"])

        y_fade_hi = N.new("ShaderNodeMapRange"); y_fade_hi.location = (-100, 0)
        y_fade_hi.inputs["From Min"].default_value = y_hi - fade_w
        y_fade_hi.inputs["From Max"].default_value = y_hi
        y_fade_hi.inputs["To Min"].default_value = 1.0
        y_fade_hi.inputs["To Max"].default_value = 0.0
        y_fade_hi.clamp = True
        L.new(sep.outputs["Y"], y_fade_hi.inputs["Value"])

        # min(x_lo_fade, x_hi_fade) → X opacity
        x_min = N.new("ShaderNodeMath"); x_min.location = (50, 250)
        x_min.operation = 'MINIMUM'
        L.new(x_fade_lo.outputs["Result"], x_min.inputs[0])
        L.new(x_fade_hi.outputs["Result"], x_min.inputs[1])

        # min(y_lo_fade, y_hi_fade) → Y opacity
        y_min = N.new("ShaderNodeMath"); y_min.location = (50, 50)
        y_min.operation = 'MINIMUM'
        L.new(y_fade_lo.outputs["Result"], y_min.inputs[0])
        L.new(y_fade_hi.outputs["Result"], y_min.inputs[1])

        # Combined: min(X, Y) — opaque only when both X and Y are in center
        xy_min = N.new("ShaderNodeMath"); xy_min.location = (200, 150)
        xy_min.operation = 'MINIMUM'
        L.new(x_min.outputs["Value"], xy_min.inputs[0])
        L.new(y_min.outputs["Value"], xy_min.inputs[1])

        L.new(xy_min.outputs["Value"], mix_sh.inputs["Fac"])
        L.new(transparent.outputs["BSDF"], mix_sh.inputs[1])
        L.new(bsdf.outputs["BSDF"], mix_sh.inputs[2])
        L.new(mix_sh.outputs["Shader"], out.inputs["Surface"])

        mat.blend_method = 'HASHED'
    else:
        L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def mat_flood_base():
    """Separate material for the flood base plane — same look, no mesh bump."""
    mat = bpy.data.materials.new("FloodBase")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial"); out.location = (600, 0)

    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Base Color"].default_value = (0.06, 0.045, 0.025, 1)
    bsdf.inputs["Roughness"].default_value = 0.02
    bsdf.inputs["IOR"].default_value = 1.33
    bsdf.inputs["Metallic"].default_value = 0.0
    _set_coat(bsdf, 0.8, 1.5, 0.015)
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Gentle procedural ripples for the flat plane
    texcoord = N.new("ShaderNodeTexCoord"); texcoord.location = (-500, -300)

    noise = N.new("ShaderNodeTexNoise"); noise.location = (-300, -300)
    noise.inputs["Scale"].default_value = 20.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.5
    L.new(texcoord.outputs["Object"], noise.inputs["Vector"])

    bump = N.new("ShaderNodeBump"); bump.location = (100, -300)
    bump.inputs["Strength"].default_value = 0.05
    bump.inputs["Distance"].default_value = 0.003
    L.new(noise.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


# ---------------------------------------------------------------------------
# Ground
# ---------------------------------------------------------------------------
def create_ground(center_xy, z=0.0, size=10.0):
    """Procedural pavement/street ground plane."""
    center = (center_xy[0], center_xy[1], z - 0.001)  # slightly below water
    bpy.ops.mesh.primitive_plane_add(size=size, location=center)
    ground = bpy.context.active_object
    ground.name = "Ground"

    mat = bpy.data.materials.new("Pavement")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    bsdf = N.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.10, 0.10, 0.09, 1)
    bsdf.inputs["Roughness"].default_value = 0.85

    texcoord = N.new("ShaderNodeTexCoord"); texcoord.location = (-600, 0)

    # Large-scale variation (stains, patches)
    noise1 = N.new("ShaderNodeTexNoise"); noise1.location = (-400, 100)
    noise1.inputs["Scale"].default_value = 3.0
    noise1.inputs["Detail"].default_value = 4.0
    noise1.inputs["Roughness"].default_value = 0.7
    L.new(texcoord.outputs["Object"], noise1.inputs["Vector"])

    # Fine grain (asphalt texture)
    noise2 = N.new("ShaderNodeTexNoise"); noise2.location = (-400, -100)
    noise2.inputs["Scale"].default_value = 30.0
    noise2.inputs["Detail"].default_value = 10.0
    noise2.inputs["Roughness"].default_value = 0.9
    L.new(texcoord.outputs["Object"], noise2.inputs["Vector"])

    # Mix into color variation
    mix = N.new("ShaderNodeMix"); mix.location = (-200, 0)
    mix.data_type = 'FLOAT'
    mix.inputs["Factor"].default_value = 0.3
    L.new(noise1.outputs["Fac"], mix.inputs["A"])
    L.new(noise2.outputs["Fac"], mix.inputs["B"])

    cr = N.new("ShaderNodeMapRange"); cr.location = (0, 0)
    cr.inputs["From Min"].default_value = 0.2
    cr.inputs["From Max"].default_value = 0.8
    cr.inputs["To Min"].default_value = 0.05
    cr.inputs["To Max"].default_value = 0.13
    L.new(mix.outputs["Result"], cr.inputs["Value"])

    combine = N.new("ShaderNodeCombineColor"); combine.location = (150, 0)
    L.new(cr.outputs["Result"], combine.inputs["Red"])
    L.new(cr.outputs["Result"], combine.inputs["Green"])
    L.new(cr.outputs["Result"], combine.inputs["Blue"])
    L.new(combine.outputs["Color"], bsdf.inputs["Base Color"])

    # Bump
    bump = N.new("ShaderNodeBump"); bump.location = (0, -200)
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.003
    L.new(noise2.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    ground.data.materials.append(mat)
    return ground


# ---------------------------------------------------------------------------
# Flood base plane — infinite water look
# ---------------------------------------------------------------------------
flood_base_obj = None
flood_base_mat = None


def create_or_update_flood_base(water_z, center_xy, size=8.0):
    """Large flat plane at water surface level — extends beyond sim domain.

    This plane extends far beyond the sim domain, providing the illusion of
    infinite flood water. The dynamic water mesh (from SplashSurf) sits on top,
    providing detailed surface near the building.
    """
    global flood_base_obj, flood_base_mat

    if flood_base_mat is None:
        flood_base_mat = mat_flood_base()

    if flood_base_obj is None:
        bpy.ops.mesh.primitive_plane_add(size=size,
                                          location=(center_xy[0], center_xy[1], water_z))
        flood_base_obj = bpy.context.active_object
        flood_base_obj.name = "FloodBase"
        flood_base_obj.data.materials.append(flood_base_mat)
    else:
        flood_base_obj.location.z = water_z


# ---------------------------------------------------------------------------
# Camera & Lights
# ---------------------------------------------------------------------------
def setup_camera(building_center, building_size):
    """Cinematic camera — diagonal SW, locked view."""
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

    print(f"  Camera at {cam_loc} → target {target}")
    return cam


def setup_lights(center, size):
    """Sun + fill — sky dome provides most illumination."""
    cx, cy, cz = center
    s = size

    # Sun — warm afternoon, matches sky sun_direction
    bpy.ops.object.light_add(type='SUN', location=(cx + s * 3, cy - s * 2, cz + s * 4))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(1.5)  # sharp shadows
    sun.data.color = (1.0, 0.95, 0.85)  # warm
    sun.rotation_euler = (math.radians(55), math.radians(10), math.radians(-20))

    # Soft fill from opposite side
    bpy.ops.object.light_add(type='AREA',
                              location=(cx - s * 2, cy + s * 2, cz + s))
    fill = bpy.context.active_object
    fill.name = "Fill"
    fill.data.energy = 50
    fill.data.size = s * 4
    fill.data.color = (0.85, 0.90, 1.0)  # cool fill


# ---------------------------------------------------------------------------
# Water mesh loading
# ---------------------------------------------------------------------------
water_obj = None
water_mat = None
_inv_scale = 1.0
_inv_offset = (0.0, 0.0, 0.0)
_edge_bounds = None  # (x_lo, x_hi, y_lo, y_hi) in native coords for edge fade


def _parse_obj_fast(obj_path):
    """Parse OBJ file with numpy — much faster than bpy.ops.import_scene."""
    verts = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)
    return np.array(verts, dtype=np.float32), faces


def _build_mesh_fast(name, verts_np, faces):
    """Build a Blender mesh using foreach_set — 10-50x faster than from_pydata."""
    mesh = bpy.data.meshes.new(name)

    n_verts = len(verts_np)
    n_faces = len(faces)
    # All triangles → 3 loops per face
    n_loops = n_faces * 3

    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", verts_np.ravel())

    mesh.loops.add(n_loops)
    # Flatten face indices: [f0v0, f0v1, f0v2, f1v0, f1v1, f1v2, ...]
    loop_verts = np.array(faces, dtype=np.int32).ravel()
    mesh.loops.foreach_set("vertex_index", loop_verts)

    mesh.polygons.add(n_faces)
    loop_starts = np.arange(0, n_loops, 3, dtype=np.int32)
    loop_totals = np.full(n_faces, 3, dtype=np.int32)
    mesh.polygons.foreach_set("loop_start", loop_starts)
    mesh.polygons.foreach_set("loop_total", loop_totals)

    mesh.update()
    mesh.validate()
    return mesh


def load_water_mesh(obj_path):
    """Load water mesh from OBJ, transform to native coords, smooth heavily.

    Optimized: parses OBJ with numpy and builds Blender mesh directly,
    avoiding the slow bpy.ops.import_scene/wm.obj_import operator.
    On first call, creates the object + modifiers. On subsequent calls,
    replaces only the mesh data in-place.
    """
    global water_obj, water_mat

    if water_mat is None:
        water_mat = mat_water(edge_bounds=_edge_bounds)

    if not os.path.exists(obj_path):
        return None

    # Parse OBJ directly — 10-50x faster than bpy.ops import
    verts_np, faces_raw = _parse_obj_fast(obj_path)
    if len(verts_np) == 0:
        return None

    # Transform from sim [0,1] → native coords (apply inline, no Blender ops)
    verts_np = verts_np * _inv_scale + np.array(_inv_offset, dtype=np.float32)

    if water_obj is not None:
        # In-place mesh replacement — reuse the existing object and modifiers
        old_mesh = water_obj.data
        new_mesh = _build_mesh_fast("WaterMeshTmp", verts_np, faces_raw)
        # Smooth shading
        new_mesh.polygons.foreach_set("use_smooth",
                                       [True] * len(new_mesh.polygons))
        new_mesh.update()
        # Assign material
        new_mesh.materials.clear()
        new_mesh.materials.append(water_mat)
        # Swap mesh data
        water_obj.data = new_mesh
        bpy.data.meshes.remove(old_mesh)
    else:
        # First frame: create the object, modifiers, etc.
        mesh = _build_mesh_fast("WaterMesh", verts_np, faces_raw)
        # Smooth shading
        mesh.polygons.foreach_set("use_smooth",
                                   [True] * len(mesh.polygons))
        mesh.update()
        mesh.materials.clear()
        mesh.materials.append(water_mat)

        water_obj = bpy.data.objects.new("WaterSurface", mesh)
        bpy.context.scene.collection.objects.link(water_obj)

        # Smoothing — eliminate surface lumps
        mod = water_obj.modifiers.new("Smooth", 'SMOOTH')
        mod.factor = 0.5
        mod.iterations = 8

        # SubSurf for truly smooth surface
        sub = water_obj.modifiers.new("SubSurf", 'SUBSURF')
        sub.levels = 0       # viewport
        sub.render_levels = 1  # render only

    return water_obj


def get_water_level_z(water_mesh_obj):
    """Get the 80th percentile Z — approximate water surface level."""
    if water_mesh_obj is None:
        return None
    verts = water_mesh_obj.data.vertices
    if len(verts) == 0:
        return None
    zs = [v.co.z for v in verts]
    zs.sort()
    return zs[int(len(zs) * 0.80)]  # 80th percentile


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------
def render_frames():
    meshes_dir = os.path.abspath(args.meshes)
    output_dir = os.path.abspath(args.output)
    meta_path = os.path.abspath(args.meta)

    with open(meta_path) as f:
        meta = json.load(f)

    # Frame list
    if args.frame:
        frame_list = [int(f.strip()) for f in args.frame.split(",")]
    else:
        obj_files = sorted(f for f in os.listdir(meshes_dir)
                           if f.startswith("water_") and f.endswith(".obj"))
        all_frames = [int(f.split("_")[1].split(".")[0]) for f in obj_files]
        end = args.end if args.end is not None else max(all_frames) if all_frames else 0
        frame_list = [f for f in all_frames if args.start <= f <= end][::args.step]

    if not frame_list:
        frame_list = [0]

    print("=" * 60)
    print("Phase 9 — Photorealistic Flood Renderer")
    print(f"  Meshes: {meshes_dir}")
    print(f"  Frames: {len(frame_list)} ({frame_list[0]}..{frame_list[-1]})")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Setup scene
    clear_scene()
    setup_render()

    # Import building
    print("\nImporting building...")
    building_objs, bmin, bmax, bcenter = import_building(meta)
    building_size = (bmax - bmin).max()

    # Ground plane — large, under everything
    create_ground(center_xy=(bcenter[0], bcenter[1]), z=bmin[2], size=building_size * 8)

    # Set inverse transform
    global _inv_scale, _inv_offset, _edge_bounds
    _inv_scale = meta["transform"]["inv_scale"]
    _inv_offset = tuple(meta["transform"]["inv_offset"])
    print(f"  Inverse transform: scale={_inv_scale:.4f}, "
          f"offset=({_inv_offset[0]:.4f}, {_inv_offset[1]:.4f}, {_inv_offset[2]:.4f})")

    # Compute edge bounds in native coords for shader fade
    # Sim domain edges where particles pile up: ~[0.05, 0.95]
    sim_edge = 0.08  # start fade at this sim coord
    sim_inner = 0.92
    _edge_bounds = (
        sim_edge * _inv_scale + _inv_offset[0],   # x_lo native
        sim_inner * _inv_scale + _inv_offset[0],   # x_hi native
        sim_edge * _inv_scale + _inv_offset[1],    # y_lo native
        sim_inner * _inv_scale + _inv_offset[1],   # y_hi native
    )
    print(f"  Edge fade bounds (native): X=[{_edge_bounds[0]:.3f}, {_edge_bounds[1]:.3f}], "
          f"Y=[{_edge_bounds[2]:.3f}, {_edge_bounds[3]:.3f}]")

    # Camera and lights
    setup_camera(bcenter, building_size)
    setup_lights(bcenter, building_size)

    sc = bpy.context.scene

    # Render each frame
    for fi, frame in enumerate(frame_list):
        print(f"\n  [{fi+1}/{len(frame_list)}] Frame {frame}...")

        # Load water mesh
        obj_path = os.path.join(meshes_dir, f"water_{frame:06d}.obj")
        w = load_water_mesh(obj_path)
        if w:
            n_v = len(w.data.vertices)
            n_f = len(w.data.polygons)
            print(f"    Water: {n_v:,} verts, {n_f:,} faces")

            # Update flood base plane at water median Z
            wz = get_water_level_z(w)
            if wz is not None:
                create_or_update_flood_base(wz, (bcenter[0], bcenter[1]),
                                             size=building_size * 8)
                print(f"    Flood base Z: {wz:.4f}")
        else:
            print(f"    Water: no mesh")

        # Render
        out_path = os.path.join(output_dir, f"frame_{frame:06d}.png")
        sc.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print(f"    -> {out_path}")

        # Clean orphans
        for m in list(bpy.data.meshes):
            if m.users == 0:
                bpy.data.meshes.remove(m)

    print(f"\nDone! {len(frame_list)} frames rendered to {output_dir}/")


if __name__ == "__main__":
    render_frames()
