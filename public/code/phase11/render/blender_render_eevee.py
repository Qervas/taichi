"""Phase 11 — EEVEE renderer for fast flood visualization.

Replaces Cycles with EEVEE for massive speedup.
Same scene setup (building, water, camera, lights) but real-time engine.

Usage:
    blender --background --python blender_render_eevee.py -- \
        --meshes ./export/flood/meshes --frame 150 --output ./renders_eevee
"""
import bpy
import math
import os
import sys
import json
import time
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

import argparse
p = argparse.ArgumentParser()
p.add_argument("--meshes", default="./export/flood/meshes")
p.add_argument("--meta", default="./export/flood/meta.json")
p.add_argument("--frame", default=None)
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=None)
p.add_argument("--step", type=int, default=1)
p.add_argument("--output", default="./renders_eevee")
p.add_argument("--samples", type=int, default=64)
p.add_argument("--resolution", default="1920x1080")
args = p.parse_args(argv)

RES_X, RES_Y = map(int, args.resolution.split("x"))
BL_VERSION = bpy.app.version
BL_IS_3X = BL_VERSION[0] < 4
MODELS_DIR = os.path.expanduser("~/Downloads/models")

print(f"  Blender {BL_VERSION[0]}.{BL_VERSION[1]}.{BL_VERSION[2]}"
      f" ({'3.x' if BL_IS_3X else '4.x+'})")


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

    # --- EEVEE engine ---
    sc.render.engine = 'BLENDER_EEVEE'

    sc.render.resolution_x = RES_X
    sc.render.resolution_y = RES_Y

    # EEVEE sampling
    if hasattr(sc.eevee, 'taa_render_samples'):
        sc.eevee.taa_render_samples = args.samples
    elif hasattr(sc.eevee, 'sampling_render_samples'):
        sc.eevee.sampling_render_samples = args.samples

    # Screen-space reflections — critical for water
    if hasattr(sc.eevee, 'use_ssr'):
        sc.eevee.use_ssr = True
        sc.eevee.use_ssr_refraction = True
        sc.eevee.ssr_quality = 1.0
        sc.eevee.ssr_thickness = 0.2
        sc.eevee.ssr_max_roughness = 0.5

    # Shadows
    if hasattr(sc.eevee, 'shadow_cube_size'):
        sc.eevee.shadow_cube_size = '2048'
    if hasattr(sc.eevee, 'shadow_cascade_size'):
        sc.eevee.shadow_cascade_size = '2048'
    if hasattr(sc.eevee, 'use_shadow_high_bitdepth'):
        sc.eevee.use_shadow_high_bitdepth = True
    if hasattr(sc.eevee, 'use_soft_shadows'):
        sc.eevee.use_soft_shadows = True

    # Ambient occlusion
    if hasattr(sc.eevee, 'use_gtao'):
        sc.eevee.use_gtao = True
        sc.eevee.gtao_distance = 0.5

    # Bloom
    if hasattr(sc.eevee, 'use_bloom'):
        sc.eevee.use_bloom = True
        sc.eevee.bloom_threshold = 0.8
        sc.eevee.bloom_intensity = 0.05

    # Volumetrics (for light shafts)
    if hasattr(sc.eevee, 'use_volumetric_lights'):
        sc.eevee.use_volumetric_lights = True

    # --- Tone mapping ---
    if BL_IS_3X:
        sc.view_settings.view_transform = 'Filmic'
        sc.view_settings.look = 'Medium High Contrast'
    else:
        try:
            sc.view_settings.view_transform = 'AgX'
            sc.view_settings.look = 'AgX - Medium High Contrast'
            print("  Tone mapping: AgX")
        except Exception:
            sc.view_settings.view_transform = 'Filmic'
            sc.view_settings.look = 'Medium High Contrast'
            print("  Tone mapping: Filmic")

    sc.render.film_transparent = False

    # --- Sky ---
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

    print(f"  Engine: EEVEE, samples={args.samples}")


# ---------------------------------------------------------------------------
# Building import
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
    print(f"  Bounds: {bmin.round(2)} -> {bmax.round(2)}")
    return objects, bmin, bmax, bcenter


# ---------------------------------------------------------------------------
# Water material — EEVEE-optimized
# ---------------------------------------------------------------------------
def mat_water(edge_bounds=None):
    """Muddy flood water for EEVEE — Principled BSDF with SSR."""
    mat = bpy.data.materials.new("FloodWater")
    mat.use_nodes = True
    N = mat.node_tree.nodes
    L = mat.node_tree.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial"); out.location = (1200, 0)

    # --- Water BSDF ---
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 200)
    bsdf.inputs["Base Color"].default_value = (0.10, 0.07, 0.03, 1)
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Metallic"].default_value = 0.0
    # Transmission for EEVEE SSR refraction
    if BL_IS_3X:
        bsdf.inputs["Transmission"].default_value = 0.15
    else:
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 0.15
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = 0.15
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5

    # EEVEE screen-space refraction flag
    if hasattr(mat, 'use_screen_refraction'):
        mat.use_screen_refraction = True

    # --- Foam BSDF ---
    foam_bsdf = N.new("ShaderNodeBsdfPrincipled"); foam_bsdf.location = (300, -200)
    foam_bsdf.inputs["Base Color"].default_value = (0.85, 0.80, 0.72, 1)
    foam_bsdf.inputs["Roughness"].default_value = 0.85

    # --- Mix water + foam ---
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

    # --- Bump (same multi-scale as Cycles version) ---
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

    # --- Edge fade ---
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
# Ground
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


# ---------------------------------------------------------------------------
# Flood base plane
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
# Camera & Lights
# ---------------------------------------------------------------------------
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
    print(f"  Camera at ({cam_loc[0]:.1f}, {cam_loc[1]:.1f}, {cam_loc[2]:.1f})")
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

    # EEVEE: enable shadow + contact shadows
    sun.data.use_shadow = True
    if hasattr(sun.data, 'use_contact_shadow'):
        sun.data.use_contact_shadow = True

    bpy.ops.object.light_add(type='AREA',
                              location=(cx - s * 2, cy + s * 2, cz + s))
    fill = bpy.context.active_object
    fill.name = "Fill"
    fill.data.energy = 50
    fill.data.size = s * 4
    fill.data.color = (0.85, 0.90, 1.0)


# ---------------------------------------------------------------------------
# Water mesh loading
# ---------------------------------------------------------------------------
water_obj = None
water_mat = None
_inv_scale = 1.0
_inv_offset = (0.0, 0.0, 0.0)
_edge_bounds = None


def _parse_obj_fast(obj_path):
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
    mesh = bpy.data.meshes.new(name)
    n_verts = len(verts_np)
    n_faces = len(faces)
    n_loops = n_faces * 3

    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", verts_np.ravel())
    mesh.loops.add(n_loops)
    loop_verts = np.array(faces, dtype=np.int32).ravel()
    mesh.loops.foreach_set("vertex_index", loop_verts)
    mesh.polygons.add(n_faces)
    mesh.polygons.foreach_set("loop_start", np.arange(0, n_loops, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(n_faces, 3, dtype=np.int32))
    mesh.update()
    mesh.validate()
    return mesh


def _apply_foam_vertex_colors(mesh, foam_data):
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
                vert_idx = mesh.loops[loop_idx].vertex_index
                f = foam_data[vert_idx] if vert_idx < len(foam_data) else 0.0
                vcol.data[loop_idx].color = (f, f, f, 1.0)
    else:
        if "foam" not in mesh.color_attributes:
            mesh.color_attributes.new(name="foam", type='FLOAT_COLOR', domain='POINT')
        attr = mesh.color_attributes["foam"]
        n = len(attr.data)
        if n == n_verts:
            colors = np.zeros(n * 4, dtype=np.float32)
            for i in range(n):
                colors[i*4] = foam_data[i]
                colors[i*4+1] = foam_data[i]
                colors[i*4+2] = foam_data[i]
                colors[i*4+3] = 1.0
            attr.data.foreach_set("color", colors)


def load_water_mesh(obj_path, foam_path=None):
    global water_obj, water_mat

    if water_mat is None:
        water_mat = mat_water(edge_bounds=_edge_bounds)

    if not os.path.exists(obj_path):
        return None

    verts_np, faces_raw = _parse_obj_fast(obj_path)
    if len(verts_np) == 0:
        return None

    foam_data = None
    if foam_path and os.path.exists(foam_path):
        foam_data = np.load(foam_path)

    verts_np = verts_np * _inv_scale + np.array(_inv_offset, dtype=np.float32)

    if water_obj is not None:
        old_mesh = water_obj.data
        new_mesh = _build_mesh_fast("WaterMeshTmp", verts_np, faces_raw)
        new_mesh.polygons.foreach_set("use_smooth", [True] * len(new_mesh.polygons))
        new_mesh.update()
        new_mesh.materials.clear()
        new_mesh.materials.append(water_mat)
        if foam_data is not None:
            _apply_foam_vertex_colors(new_mesh, foam_data)
        water_obj.data = new_mesh
        bpy.data.meshes.remove(old_mesh)
    else:
        mesh = _build_mesh_fast("WaterMesh", verts_np, faces_raw)
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.update()
        mesh.materials.clear()
        mesh.materials.append(water_mat)
        if foam_data is not None:
            _apply_foam_vertex_colors(mesh, foam_data)

        water_obj = bpy.data.objects.new("WaterSurface", mesh)
        bpy.context.scene.collection.objects.link(water_obj)

        mod = water_obj.modifiers.new("Smooth", 'SMOOTH')
        mod.factor = 0.5
        mod.iterations = 3

        sub = water_obj.modifiers.new("SubSurf", 'SUBSURF')
        sub.levels = 0
        sub.render_levels = 1

    return water_obj


def get_water_level_z(water_mesh_obj):
    if water_mesh_obj is None:
        return None
    verts = water_mesh_obj.data.vertices
    if len(verts) == 0:
        return None
    zs = [v.co.z for v in verts]
    zs.sort()
    return zs[int(len(zs) * 0.80)]


# ---------------------------------------------------------------------------
# Main
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
    print("Phase 11 — EEVEE Flood Renderer")
    print(f"  Meshes: {meshes_dir}")
    print(f"  Frames: {len(frame_list)} ({frame_list[0]}..{frame_list[-1]})")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {args.samples}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    clear_scene()
    setup_render()

    # Import building
    print("\nImporting building...")
    building_objs, bmin, bmax, bcenter = import_building(meta)
    if bmin is None:
        print("ERROR: No building loaded")
        return
    building_size = (bmax - bmin).max()

    # Ground
    create_ground(center_xy=(bcenter[0], bcenter[1]), z=bmin[2], size=building_size * 8)

    # Inverse transform
    global _inv_scale, _inv_offset, _edge_bounds
    _inv_scale = meta["transform"]["inv_scale"]
    _inv_offset = tuple(meta["transform"]["inv_offset"])
    print(f"  Inverse transform: scale={_inv_scale:.4f}")

    sim_edge = 0.08
    sim_inner = 0.92
    _edge_bounds = (
        sim_edge * _inv_scale + _inv_offset[0],
        sim_inner * _inv_scale + _inv_offset[0],
        sim_edge * _inv_scale + _inv_offset[1],
        sim_inner * _inv_scale + _inv_offset[1],
    )

    # Camera + lights
    setup_camera(bcenter, building_size)
    setup_lights(bcenter, building_size)

    # Render loop
    for fi, frame in enumerate(frame_list):
        out_path = os.path.join(output_dir, f"frame_{frame:06d}.png")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f"\n  [{fi+1}/{len(frame_list)}] Frame {frame} — already rendered, skip")
            continue
        print(f"\n  [{fi+1}/{len(frame_list)}] Frame {frame}...")

        obj_path = os.path.join(meshes_dir, f"water_{frame:06d}.obj")
        foam_path = os.path.join(meshes_dir, f"foam_{frame:06d}.npy")
        w = load_water_mesh(obj_path, foam_path=foam_path)
        if w:
            n_v = len(w.data.vertices)
            has_foam = os.path.exists(foam_path)
            print(f"    Water: {n_v:,} verts, foam={'yes' if has_foam else 'no'}")

            wz = get_water_level_z(w)
            if wz is not None:
                create_or_update_flood_base(wz, (bcenter[0], bcenter[1]),
                                             size=building_size * 8)
        else:
            print(f"    Water: no mesh")

        t0 = time.time()
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        print(f"    -> {out_path} ({dt:.2f}s)")

        for m in list(bpy.data.meshes):
            if m.users == 0:
                bpy.data.meshes.remove(m)

    print(f"\nDone! {len(frame_list)} frames rendered to {output_dir}/")


if __name__ == "__main__":
    render_frames()
