"""Phase 10 — FAST Blender Cycles renderer using heightmap displacement.

Instead of loading a new OBJ mesh every frame (→ BVH rebuild ~40s),
uses a FIXED subdivided plane with per-frame height map displacement.
The plane mesh never changes → BVH stays cached → ~2-3s/frame.

Usage:
    blender --background --python blender_render_fast.py -- \
        --heightmaps ./export/flood/heightmaps \
        --meta ./export/flood/meta.json \
        --building ~/Downloads/models/0_1_cluster_texture.obj \
        --output ./renders_fast --samples 64
"""
import bpy
import bmesh
import math
import os
import sys
import json
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

import argparse
p = argparse.ArgumentParser()
p.add_argument("--heightmaps", required=True, help="Dir with height_XXXXXX.npy files")
p.add_argument("--meta", default="./export/flood/meta.json")
p.add_argument("--building", default=None)
p.add_argument("--frame", default=None)
p.add_argument("--start", type=int, default=0)
p.add_argument("--end", type=int, default=None)
p.add_argument("--step", type=int, default=1)
p.add_argument("--output", default="./renders_fast")
p.add_argument("--samples", type=int, default=64)
p.add_argument("--resolution", default="1920x1080")
p.add_argument("--cpu", action="store_true")
args = p.parse_args(argv)

RES_X, RES_Y = map(int, args.resolution.split("x"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

BL_VERSION = bpy.app.version
BL_IS_3X = BL_VERSION[0] < 4
print(f"Blender {BL_VERSION[0]}.{BL_VERSION[1]}.{BL_VERSION[2]}"
      f" ({'3.x' if BL_IS_3X else '4.x+'})")


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

    # Sky
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    N = world.node_tree.nodes
    L = world.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputWorld"); out.location = (600, 0)
    bg = N.new("ShaderNodeBackground"); bg.location = (400, 0)
    bg.inputs["Strength"].default_value = 1.5
    sky = N.new("ShaderNodeTexSky"); sky.location = (0, 0)
    sky.sky_type = 'HOSEK_WILKIE'
    sky.sun_direction = (0.4, -0.2, 0.55)
    sky.turbidity = 5.0
    L.new(sky.outputs["Color"], bg.inputs["Color"])
    L.new(bg.outputs["Background"], out.inputs["Surface"])


def import_building(meta):
    if args.building:
        obj_path = os.path.expanduser(args.building)
    else:
        obj_path = os.path.join(C.MODELS_DIR, meta.get("building_obj", ""))

    print(f"  Loading building: {obj_path}")
    if BL_IS_3X:
        bpy.ops.import_scene.obj(filepath=obj_path,
                                  axis_forward='-Y', axis_up='Z')
    else:
        bpy.ops.wm.obj_import(filepath=obj_path,
                               forward_axis='NEGATIVE_Y', up_axis='Z')

    objects = list(bpy.context.selected_objects)
    meshes = [o for o in objects if o.type == 'MESH']

    obj_dir = os.path.dirname(obj_path)
    for obj in meshes:
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        img = node.image
                        if not os.path.exists(img.filepath_from_user()):
                            candidate = os.path.join(obj_dir,
                                                      os.path.basename(img.filepath))
                            if os.path.exists(candidate):
                                img.filepath = candidate
                                img.reload()

    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)

    verts = np.array([(v.x, v.y, v.z) for v in all_verts])
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    bcenter = verts.mean(axis=0)

    print(f"  Building: {len(meshes)} meshes")
    print(f"  Bounds: {bmin.round(2)} → {bmax.round(2)}")
    return objects, bmin, bmax, bcenter


# ---------------------------------------------------------------------------
# Water plane (FIXED topology — never changes)
# ---------------------------------------------------------------------------
water_plane = None
water_mat = None


def create_water_plane(center_xy, z, size, subdivisions=256):
    """Create a fixed subdivided plane for heightmap displacement.

    This plane is created ONCE and never changes topology.
    Only vertex Z positions are updated per frame.
    """
    global water_plane, water_mat

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=subdivisions,
        y_subdivisions=subdivisions,
        size=size,
        location=(center_xy[0], center_xy[1], z))

    water_plane = bpy.context.active_object
    water_plane.name = "WaterPlane"

    # Smooth shading
    for poly in water_plane.data.polygons:
        poly.use_smooth = True

    # Water material
    water_mat = bpy.data.materials.new("FloodWater")
    water_mat.use_nodes = True
    N = water_mat.node_tree.nodes
    L = water_mat.node_tree.links
    N.clear()

    out = N.new("ShaderNodeOutputMaterial"); out.location = (800, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    bsdf.inputs["Base Color"].default_value = (0.06, 0.045, 0.025, 1)
    bsdf.inputs["Roughness"].default_value = 0.02
    bsdf.inputs["IOR"].default_value = 1.33
    bsdf.inputs["Metallic"].default_value = 0.0
    _set_coat(bsdf, 0.8, 1.5, 0.015)

    # Bump from noise
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

    if BL_IS_3X:
        mix_n = N.new("ShaderNodeMixRGB"); mix_n.location = (-100, -300)
        mix_n.inputs["Fac"].default_value = 0.3
        L.new(noise1.outputs["Fac"], mix_n.inputs["Color1"])
        L.new(noise2.outputs["Fac"], mix_n.inputs["Color2"])
        bump = N.new("ShaderNodeBump"); bump.location = (100, -300)
        bump.inputs["Strength"].default_value = 0.08
        bump.inputs["Distance"].default_value = 0.005
        L.new(mix_n.outputs["Color"], bump.inputs["Height"])
    else:
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
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    water_plane.data.materials.append(water_mat)

    print(f"  Water plane: {subdivisions}x{subdivisions} = "
          f"{len(water_plane.data.vertices):,} verts (FIXED topology)")
    return water_plane


def apply_heightmap(heightmap, inv_scale, inv_offset,
                    sim_xy=(0.05, 0.95), hide_threshold=0.001):
    """Apply height map to the fixed water plane by updating vertex Z.

    Only modifies vertex positions — topology stays the same.
    Cycles does BVH REFIT (fast) instead of REBUILD (slow).

    NOTE: foreach_get("co") returns LOCAL object-space coords (centered at 0).
    Must add object location to get world coords for the inv_transform mapping.
    """
    if water_plane is None:
        return

    mesh = water_plane.data
    n_verts = len(mesh.vertices)

    # Get current vertex positions (LOCAL object space)
    cos = np.zeros(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", cos)
    cos = cos.reshape(-1, 3)

    # Convert local XY to world XY, then to sim coords
    loc = water_plane.location
    world_x = cos[:, 0] + loc.x
    world_y = cos[:, 1] + loc.y
    sim_x = (world_x - inv_offset[0]) / inv_scale
    sim_y = (world_y - inv_offset[1]) / inv_scale

    # Sample heightmap at each vertex's sim XY position
    h, w = heightmap.shape
    sim_lo, sim_hi = sim_xy
    sim_span = sim_hi - sim_lo

    # Map sim coords to heightmap pixel coords
    px = np.clip(((sim_x - sim_lo) / sim_span * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(((sim_y - sim_lo) / sim_span * (h - 1)).astype(int), 0, h - 1)

    # Look up Z from heightmap (in sim coords)
    z_sim = heightmap[px, py]

    # Convert Z from sim coords to world coords, then to local Z
    z_world = z_sim * inv_scale + inv_offset[2]
    z_local = z_world - loc.z

    # For cells with no water (z_sim near 0), push below ground to hide
    no_water = z_sim < hide_threshold
    z_local[no_water] = -10.0  # well below object origin

    # Update only Z (local coords)
    cos[:, 2] = z_local
    mesh.vertices.foreach_set("co", cos.ravel())
    mesh.update()


# ---------------------------------------------------------------------------
# Ground & lights (same as before)
# ---------------------------------------------------------------------------
def create_ground(center_xy, z=0.0, size=10.0):
    center = (center_xy[0], center_xy[1], z - 0.001)
    bpy.ops.mesh.primitive_plane_add(size=size, location=center)
    ground = bpy.context.active_object
    ground.name = "Ground"
    mat = bpy.data.materials.new("Pavement")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.10, 0.10, 0.09, 1)
    bsdf.inputs["Roughness"].default_value = 0.85
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
    sun.data.energy = 3.0
    sun.data.angle = math.radians(1.5)
    sun.data.color = (1.0, 0.95, 0.85)
    sun.rotation_euler = (math.radians(55), math.radians(10), math.radians(-20))

    bpy.ops.object.light_add(type='AREA',
                              location=(cx - s * 2, cy + s * 2, cz + s))
    fill = bpy.context.active_object
    fill.name = "Fill"
    fill.data.energy = 50
    fill.data.size = s * 4
    fill.data.color = (0.85, 0.90, 1.0)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------
def render_frames():
    hmap_dir = os.path.abspath(args.heightmaps)
    output_dir = os.path.abspath(args.output)
    meta_path = os.path.abspath(args.meta)

    with open(meta_path) as f:
        meta = json.load(f)

    # Frame list
    if args.frame:
        frame_list = [int(f.strip()) for f in args.frame.split(",")]
    else:
        npy_files = sorted(f for f in os.listdir(hmap_dir)
                           if f.startswith("height_") and f.endswith(".npy"))
        all_frames = [int(f.split("_")[1].split(".")[0]) for f in npy_files]
        end = args.end if args.end is not None else max(all_frames) if all_frames else 0
        frame_list = [f for f in all_frames if args.start <= f <= end][::args.step]

    if not frame_list:
        print("No frames to render!")
        return

    print("=" * 60)
    print("Phase 10 — FAST Heightmap Displacement Renderer")
    print(f"  Heightmaps: {hmap_dir}")
    print(f"  Frames: {len(frame_list)} ({frame_list[0]}..{frame_list[-1]})")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Setup scene (ONE TIME)
    clear_scene()
    setup_render()

    # Import building
    print("\nImporting building...")
    building_objs, bmin, bmax, bcenter = import_building(meta)
    building_size = (bmax - bmin).max()

    # Inverse transform
    inv_scale = meta["transform"]["inv_scale"]
    inv_offset = tuple(meta["transform"]["inv_offset"])
    print(f"  Inverse transform: scale={inv_scale:.4f}, offset={inv_offset}")

    # Ground
    create_ground(center_xy=(bcenter[0], bcenter[1]), z=bmin[2],
                  size=building_size * 8)

    # Create FIXED water plane (never changes topology)
    water_z = bmin[2]  # start at ground level
    create_water_plane(
        center_xy=(bcenter[0], bcenter[1]),
        z=water_z,
        size=building_size * 4,
        subdivisions=256)

    # Camera and lights
    setup_camera(bcenter, building_size)
    setup_lights(bcenter, building_size)

    sc = bpy.context.scene

    # Render loop — only update vertex Z positions, never change topology
    import time
    t0 = time.time()

    for fi, frame in enumerate(frame_list):
        out_path = os.path.join(output_dir, f"frame_{frame:06d}.png")

        # Skip already rendered
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            continue

        # Load heightmap
        hmap_path = os.path.join(hmap_dir, f"height_{frame:06d}.npy")
        if not os.path.exists(hmap_path):
            print(f"  [{fi+1}/{len(frame_list)}] Frame {frame} — no heightmap, skip")
            continue

        heightmap = np.load(hmap_path)

        # Apply heightmap to fixed plane (only vertex Z changes)
        apply_heightmap(heightmap, inv_scale, inv_offset)

        # Render
        sc.render.filepath = out_path
        bpy.ops.render.render(write_still=True)

        elapsed = time.time() - t0
        fps = (fi + 1) / elapsed if elapsed > 0 else 0
        if fi % 10 == 0 or fi < 3:
            print(f"  [{fi+1}/{len(frame_list)}] Frame {frame} — "
                  f"{elapsed:.0f}s total ({fps:.1f} fr/s)")

    elapsed = time.time() - t0
    print(f"\nDone! {len(frame_list)} frames in {elapsed:.0f}s "
          f"({len(frame_list)/elapsed:.1f} fr/s)")


if __name__ == "__main__":
    render_frames()
