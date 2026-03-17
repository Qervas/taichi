"""
Blender Cycles PBR renderer for Phase 5 water wheel simulation.

Usage:
    blender --background --python blender_render.py -- \
        --input ./export --frames 300 --output ./renders

    Options after '--':
      --input    Directory with water*.ply + wheel_*.obj files
      --frames   Number of frames to render (default: 300)
      --output   Output directory for rendered PNGs (default: ./renders)
      --samples  Cycles render samples (default: 256)
      --preview  Use instanced spheres instead of meshed fluid (fast)
      --no-caustics  Disable caustics for faster renders
      --resolution WxH  e.g. 1920x1080 (default)
"""

import bpy
import bmesh
import math
import os
import sys
import numpy as np
from mathutils import Vector, Matrix

# ---------------------------------------------------------------------------
# Parse CLI args (everything after '--')
# ---------------------------------------------------------------------------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./export")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--output", default="./renders")
parser.add_argument("--samples", type=int, default=256)
parser.add_argument("--preview", action="store_true")
parser.add_argument("--no-caustics", action="store_true")
parser.add_argument("--start", type=int, default=0, help="Start from this frame (skip earlier)")
parser.add_argument("--resolution", default="1920x1080")
args = parser.parse_args(argv)

RES_X, RES_Y = [int(x) for x in args.resolution.split("x")]


# ---------------------------------------------------------------------------
# Blender version compatibility
# ---------------------------------------------------------------------------
BL_VERSION = bpy.app.version  # e.g. (4, 0, 0) or (3, 6, 5)
IS_BL4 = BL_VERSION[0] >= 4


def principled_input_name(name):
    """Map Principled BSDF input names for 3.x vs 4.x compatibility."""
    remap_4x = {
        "Transmission": "Transmission Weight",
        "Specular": "Specular IOR Level",
        "Emission Strength": "Emission Strength",
    }
    if IS_BL4 and name in remap_4x:
        return remap_4x[name]
    return name


def set_principled(node, name, value):
    """Set a Principled BSDF input by name with version compatibility."""
    actual = principled_input_name(name)
    if actual in node.inputs:
        node.inputs[actual].default_value = value
    else:
        print(f"  Warning: Principled BSDF has no input '{actual}' (tried '{name}')")


# ---------------------------------------------------------------------------
# Scene cleanup
# ---------------------------------------------------------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.node_groups:
        bpy.data.node_groups.remove(block)


# ---------------------------------------------------------------------------
# Renderer setup
# ---------------------------------------------------------------------------
def setup_cycles():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'

    cycles = scene.cycles
    cycles.device = 'GPU'
    cycles.samples = args.samples
    cycles.use_denoising = True
    cycles.denoiser = 'OPENIMAGEDENOISE'
    cycles.max_bounces = 12
    cycles.diffuse_bounces = 4
    cycles.glossy_bounces = 6
    cycles.transmission_bounces = 8
    cycles.volume_bounces = 2

    if args.no_caustics:
        cycles.caustics_reflective = False
        cycles.caustics_refractive = False
    else:
        cycles.caustics_reflective = True
        cycles.caustics_refractive = True

    # Filmic color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium Contrast'

    # Enable GPU compute
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        cprefs = prefs.preferences
        cprefs.compute_device_type = 'CUDA'
        cprefs.get_devices()
        for device in cprefs.devices:
            device.use = True


# ---------------------------------------------------------------------------
# PBR Water Material
# ---------------------------------------------------------------------------
def create_water_material():
    mat = bpy.data.materials.new(name="WaterPBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (200, 0)
    set_principled(bsdf, "Base Color", (1.0, 1.0, 1.0, 1.0))
    set_principled(bsdf, "Roughness", 0.0)
    set_principled(bsdf, "Transmission", 1.0)
    set_principled(bsdf, "IOR", 1.33)
    set_principled(bsdf, "Specular", 0.5)
    set_principled(bsdf, "Alpha", 0.85)

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


# ---------------------------------------------------------------------------
# PBR Wood Material
# ---------------------------------------------------------------------------
def create_wood_material():
    mat = bpy.data.materials.new(name="WoodPBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (400, 0)
    set_principled(bsdf, "Roughness", 0.35)
    set_principled(bsdf, "Specular", 0.3)

    # Vertex color as base color
    vcol = nodes.new('ShaderNodeVertexColor')
    vcol.location = (0, 100)
    vcol.layer_name = "Col"
    links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])

    # Micro-grain bump from noise texture
    texcoord = nodes.new('ShaderNodeTexCoord')
    texcoord.location = (-400, -200)

    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-200, -200)
    noise.inputs['Scale'].default_value = 200.0
    noise.inputs['Detail'].default_value = 8.0
    links.new(texcoord.outputs['Object'], noise.inputs['Vector'])

    bump = nodes.new('ShaderNodeBump')
    bump.location = (200, -200)
    bump.inputs['Strength'].default_value = 0.05
    bump.inputs['Distance'].default_value = 0.01
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


# ---------------------------------------------------------------------------
# Floor Material (dark reflective)
# ---------------------------------------------------------------------------
def create_floor_material():
    mat = bpy.data.materials.new(name="FloorPBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    set_principled(bsdf, "Base Color", (0.05, 0.05, 0.06, 1.0))
    set_principled(bsdf, "Roughness", 0.15)
    set_principled(bsdf, "Specular", 0.8)

    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ---------------------------------------------------------------------------
# Geometry Nodes: Point cloud → Volume → Mesh (fluid surface)
# ---------------------------------------------------------------------------
def create_fluid_geonodes():
    """Build a Geometry Nodes modifier: vertices → points → volume → smooth mesh."""
    tree = bpy.data.node_groups.new(name="FluidMesher", type='GeometryNodeTree')

    # Create interface sockets (Blender 4.0+ vs 3.x)
    if IS_BL4:
        tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:
        tree.inputs.new('NodeSocketGeometry', "Geometry")
        tree.outputs.new('NodeSocketGeometry', "Geometry")

    nodes = tree.nodes
    links = tree.links

    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-800, 0)

    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (800, 0)

    # Step 1: Mesh to Points — convert vertex mesh to point cloud
    m2p = nodes.new('GeometryNodeMeshToPoints')
    m2p.location = (-600, 0)
    m2p.inputs['Radius'].default_value = 0.008  # particle influence radius
    links.new(input_node.outputs[0], m2p.inputs['Mesh'])

    # Step 2: Points to Volume — SPH-like density field
    p2v = nodes.new('GeometryNodePointsToVolume')
    p2v.location = (-300, 0)
    if 'Resolution Mode' in p2v.inputs:
        p2v.inputs['Resolution Mode'].default_value = 'Size'
    elif hasattr(p2v, 'resolution_mode'):
        p2v.resolution_mode = 'VOXEL_SIZE'
    p2v.inputs['Voxel Size'].default_value = 0.004  # finer volume grid
    p2v.inputs['Radius'].default_value = 0.045  # larger merge radius — smooths isolated blobs
    if 'Density' in p2v.inputs:
        p2v.inputs['Density'].default_value = 1.0
    links.new(m2p.outputs['Points'], p2v.inputs['Points'])

    # Step 3: Volume to Mesh — isosurface extraction
    v2m = nodes.new('GeometryNodeVolumeToMesh')
    v2m.location = (0, 0)
    if 'Resolution Mode' in v2m.inputs:
        v2m.inputs['Resolution Mode'].default_value = 'Size'
    elif hasattr(v2m, 'resolution_mode'):
        v2m.resolution_mode = 'VOXEL_SIZE'
    v2m.inputs['Voxel Size'].default_value = 0.004  # match volume voxel for clean surface
    v2m.inputs['Threshold'].default_value = 0.02  # lower threshold captures more of the field
    links.new(p2v.outputs['Volume'], v2m.inputs['Volume'])

    # Step 4: Set Shade Smooth
    smooth = nodes.new('GeometryNodeSetShadeSmooth')
    smooth.location = (250, 0)
    smooth_in = 'Mesh' if 'Mesh' in smooth.inputs else 'Geometry'
    smooth_out = 'Mesh' if 'Mesh' in smooth.outputs else 'Geometry'
    links.new(v2m.outputs['Mesh'], smooth.inputs[smooth_in])

    # Step 5: Set Material
    setmat = nodes.new('GeometryNodeSetMaterial')
    setmat.location = (500, 0)
    setmat.inputs['Material'].default_value = bpy.data.materials.get("WaterPBR")
    setmat_in = 'Mesh' if 'Mesh' in setmat.inputs else 'Geometry'
    setmat_out = 'Mesh' if 'Mesh' in setmat.outputs else 'Geometry'
    links.new(smooth.outputs[smooth_out], setmat.inputs[setmat_in])

    links.new(setmat.outputs[setmat_out], output_node.inputs[0])

    return tree


# ---------------------------------------------------------------------------
# PLY binary reader (no external deps)
# ---------------------------------------------------------------------------
def read_ply_binary(filepath):
    """Read binary little-endian PLY. Returns dict of numpy arrays."""
    with open(filepath, 'rb') as f:
        # Parse header
        properties = []
        num_vertices = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
            elif line == 'end_header':
                break

        if num_vertices == 0:
            return {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}

        # Build numpy dtype from properties
        type_map = {
            'float': np.float32, 'float32': np.float32,
            'double': np.float64, 'float64': np.float64,
            'int': np.int32, 'int32': np.int32,
            'uchar': np.uint8, 'uint8': np.uint8,
        }
        dtype = [(name, type_map.get(t, np.float32)) for name, t in properties]
        data = np.frombuffer(f.read(), dtype=np.dtype(dtype), count=num_vertices)

        result = {}
        for name, _ in properties:
            result[name] = data[name].astype(np.float32)
        return result


def read_ply_ascii(filepath):
    """Fallback ASCII PLY reader."""
    with open(filepath, 'r') as f:
        properties = []
        num_vertices = 0
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append(parts[2])
            elif line == 'end_header':
                break

        data = {p: [] for p in properties}
        for _ in range(num_vertices):
            vals = f.readline().strip().split()
            for j, p in enumerate(properties):
                data[p].append(float(vals[j]))

        return {p: np.array(v, dtype=np.float32) for p, v in data.items()}


def load_ply(filepath):
    """Load PLY, trying binary first then ASCII fallback."""
    with open(filepath, 'rb') as f:
        header_start = f.read(256).decode('ascii', errors='replace')

    if 'binary_little_endian' in header_start:
        return read_ply_binary(filepath)
    else:
        return read_ply_ascii(filepath)


# ---------------------------------------------------------------------------
# Water mesh loading
# ---------------------------------------------------------------------------
def create_water_pointcloud(ply_data, name="WaterPoints"):
    """Create a mesh object from PLY particle positions (one vertex per particle)."""
    verts = list(zip(
        ply_data['x'].tolist(),
        ply_data['y'].tolist(),
        ply_data['z'].tolist()
    ))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def update_water_mesh(obj, ply_data):
    """Update existing water mesh vertices in-place, or rebuild if count changed."""
    n = len(ply_data['x'])
    mesh = obj.data
    if len(mesh.vertices) != n:
        # Rebuild
        bm = bmesh.new()
        for i in range(n):
            bm.verts.new((ply_data['x'][i], ply_data['y'][i], ply_data['z'][i]))
        bm.to_mesh(mesh)
        bm.free()
    else:
        # Update in-place (faster)
        coords = np.column_stack([ply_data['x'], ply_data['y'], ply_data['z']])
        mesh.vertices.foreach_set("co", coords.ravel())
    mesh.update()


def create_water_preview(ply_data, name="WaterPreview"):
    """Preview mode: instanced ico-spheres for each particle."""
    obj = create_water_pointcloud(ply_data, name)
    # Add ico-sphere as instancer
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.005, subdivisions=2)
    sphere = bpy.context.active_object
    sphere.name = "WaterParticle"
    sphere.data.materials.append(bpy.data.materials.get("WaterPBR"))

    obj.instance_type = 'VERTS'
    sphere.parent = obj
    return obj


# ---------------------------------------------------------------------------
# Wheel OBJ import
# ---------------------------------------------------------------------------
def import_wheel_obj(filepath, material):
    """Import wheel OBJ with vertex colors."""
    verts = []
    normals = []
    colors = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v ') and not line.startswith('vn'):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(parts) >= 7:
                    colors.append((float(parts[4]), float(parts[5]), float(parts[6])))
            elif line.startswith('vn '):
                parts = line.split()
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                parts = line.split()[1:]
                face_verts = []
                for p in parts:
                    vi = int(p.split('//')[0]) - 1
                    face_verts.append(vi)
                faces.append(face_verts)

    mesh = bpy.data.meshes.new("WheelMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Apply vertex colors
    if colors:
        if not mesh.vertex_colors:
            mesh.vertex_colors.new(name="Col")
        color_layer = mesh.vertex_colors["Col"]
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vi = mesh.loops[li].vertex_index
                if vi < len(colors):
                    r, g, b = colors[vi]
                    color_layer.data[li].color = (r, g, b, 1.0)

    mesh.materials.append(material)

    obj = bpy.data.objects.new("Wheel", mesh)
    bpy.context.collection.objects.link(obj)

    # Also try loading .npy fallback colors (more reliable for older Blender)
    npy_path = filepath.replace('.obj', '').replace('wheel_', 'wheel_colors_') + '.npy'
    if not colors and os.path.exists(npy_path):
        npy_colors = np.load(npy_path)
        if not mesh.vertex_colors:
            mesh.vertex_colors.new(name="Col")
        color_layer = mesh.vertex_colors["Col"]
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vi = mesh.loops[li].vertex_index
                if vi < len(npy_colors):
                    r, g, b = npy_colors[vi]
                    color_layer.data[li].color = (r, g, b, 1.0)

    return obj


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def setup_camera(total_frames):
    cam_data = bpy.data.cameras.new(name="MainCamera")
    cam_data.lens = 35
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 4.0

    cam_obj = bpy.data.objects.new("MainCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Focus target at wheel center
    focus_empty = bpy.data.objects.new("FocusTarget", None)
    focus_empty.location = (0.5, 0.45, 0.5)
    bpy.context.collection.objects.link(focus_empty)
    cam_data.dof.focus_object = focus_empty

    # Camera outside the open front of the Cornell box (Z > 1),
    # looking in at the wheel and water.
    look_at = Vector((0.5, 0.45, 0.5))   # center of box
    cam_height = 0.6      # slightly above mid-height
    cam_z = 2.8           # far enough to frame the full box
    orbit_sweep = math.radians(8)   # subtle horizontal drift

    for frame in range(total_frames + 1):
        t = frame / max(total_frames, 1)
        # Small horizontal sweep centered on the box
        x_offset = 0.15 * math.sin(-orbit_sweep / 2 + orbit_sweep * t)
        pos = Vector((0.5 + x_offset, cam_height, cam_z))
        cam_obj.location = pos

        # Manual rotation: look at target with world-Y as up (no roll)
        fwd = (look_at - pos).normalized()
        world_up = Vector((0, 1, 0))
        right = fwd.cross(world_up).normalized()
        cam_up = right.cross(fwd).normalized()

        # Build rotation matrix (Blender camera: -Z is forward, Y is up)
        rot_mat = Matrix((
            (right.x,    right.y,    right.z),
            (cam_up.x,   cam_up.y,   cam_up.z),
            (-fwd.x,     -fwd.y,     -fwd.z),
        )).transposed()
        cam_obj.rotation_euler = rot_mat.to_euler()

        cam_obj.keyframe_insert(data_path="location", frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    return cam_obj


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
def setup_lighting():
    # Sun light shining down through the open ceiling
    sun_data = bpy.data.lights.new(name="Sun", type='SUN')
    sun_data.energy = 5.0
    sun_data.color = (1.0, 0.97, 0.92)
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    # Nearly straight down with slight angle for interesting shadows
    sun_obj.rotation_euler = (math.radians(70), 0, math.radians(15))
    bpy.context.collection.objects.link(sun_obj)

    # Subtle fill from the open front to soften shadows
    fill_data = bpy.data.lights.new(name="FrontFill", type='AREA')
    fill_data.energy = 15.0
    fill_data.color = (0.9, 0.92, 1.0)
    fill_data.size = 1.0
    fill_obj = bpy.data.objects.new("FrontFill", fill_data)
    fill_obj.location = (0.5, 0.6, 1.8)   # outside the box, in front
    fill_obj.rotation_euler = (math.radians(95), 0, 0)  # angled slightly into box
    bpy.context.collection.objects.link(fill_obj)

    # Nishita sky (procedural, no HDRI needed)
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new('ShaderNodeBackground')
    bg.location = (200, 0)
    bg.inputs['Strength'].default_value = 0.4

    sky = nodes.new('ShaderNodeTexSky')
    sky.location = (0, 0)
    # Blender 5.0 renamed NISHITA; pick best available physical sky
    sky_types = [item.identifier for item in sky.bl_rna.properties['sky_type'].enum_items]
    if 'NISHITA' in sky_types:
        sky.sky_type = 'NISHITA'
    elif 'MULTIPLE_SCATTERING' in sky_types:
        sky.sky_type = 'MULTIPLE_SCATTERING'
    else:
        sky.sky_type = 'HOSEK_WILKIE'
    if hasattr(sky, 'sun_elevation'):
        sky.sun_elevation = math.radians(30)
    if hasattr(sky, 'sun_rotation'):
        sky.sun_rotation = math.radians(-20)

    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)

    links.new(sky.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])


# ---------------------------------------------------------------------------
# Scene geometry — Cornell box matching sim domain [0,1]^3
# ---------------------------------------------------------------------------
def _make_wall_mat(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        set_principled(bsdf, "Base Color", (*color, 1.0))
        set_principled(bsdf, "Roughness", 0.9)
        set_principled(bsdf, "Specular", 0.0)
    return mat


def create_cornell_box():
    """Build a proper 3D Cornell box (cube with inward normals, front face removed).

    The box is slightly larger than the [0,1]^3 sim domain so nothing pokes through.
    Per-face materials: white floor/ceiling/back, red left, green right.
    """
    white = (0.73, 0.73, 0.73)
    red   = (0.65, 0.05, 0.05)
    green = (0.12, 0.45, 0.15)

    mat_white = _make_wall_mat("CornellWhite", white)
    mat_red   = _make_wall_mat("CornellRed",   red)
    mat_green = _make_wall_mat("CornellGreen", green)

    PAD = 0.05
    LO, HI = -PAD, 1.0 + PAD

    # 8 corners of the box
    verts = [
        (LO, LO, LO),  # 0: left-bottom-back
        (HI, LO, LO),  # 1: right-bottom-back
        (HI, HI, LO),  # 2: right-top-back
        (LO, HI, LO),  # 3: left-top-back
        (LO, LO, HI),  # 4: left-bottom-front
        (HI, LO, HI),  # 5: right-bottom-front
        (HI, HI, HI),  # 6: right-top-front
        (LO, HI, HI),  # 7: left-top-front
    ]

    # 4 quads — no front face (camera looks through) and no ceiling (sun comes in)
    # Winding order: normals point INWARD
    faces = [
        (0, 1, 2, 3),  # back wall   (Z = LO, normal +Z)  → white
        (4, 0, 3, 7),  # left wall   (X = LO, normal +X)  → red
        (1, 5, 6, 2),  # right wall  (X = HI, normal -X)  → green
        (4, 5, 1, 0),  # floor       (Y = LO, normal +Y)  → white
    ]

    mesh = bpy.data.meshes.new("CornellBox")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Assign materials: slot 0=white, 1=red, 2=green
    mesh.materials.append(mat_white)   # slot 0
    mesh.materials.append(mat_red)     # slot 1
    mesh.materials.append(mat_green)   # slot 2

    # Per-face material assignment
    # faces: 0=back(white), 1=left(red), 2=right(green), 3=floor(white)
    mat_indices = [0, 1, 2, 0]
    for i, fi in enumerate(mat_indices):
        mesh.polygons[i].material_index = fi

    obj = bpy.data.objects.new("CornellBox", mesh)
    bpy.context.collection.objects.link(obj)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------
def render_frames():
    # Resolve to absolute paths (Blender headless resolves relative paths to /)
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    print("=" * 60)
    print("Blender Cycles PBR — Phase 5 Water Wheel")
    print(f"  Input:      {args.input}")
    print(f"  Frames:     {args.frames}")
    print(f"  Output:     {args.output}")
    print(f"  Samples:    {args.samples}")
    print(f"  Resolution: {RES_X}x{RES_Y}")
    print(f"  Preview:    {args.preview}")
    print(f"  Caustics:   {not args.no_caustics}")
    print(f"  Blender:    {'.'.join(str(v) for v in BL_VERSION)}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    # Setup
    clear_scene()
    setup_cycles()

    water_mat = create_water_material()
    wood_mat = create_wood_material()

    setup_lighting()
    create_cornell_box()
    setup_camera(args.frames)

    # Create fluid geometry nodes modifier
    fluid_geonodes = None
    if not args.preview:
        fluid_geonodes = create_fluid_geonodes()

    # Load first frame to create persistent water object
    first_ply = os.path.join(args.input, "water_000000.ply")
    if not os.path.exists(first_ply):
        print(f"ERROR: Cannot find {first_ply}")
        return

    ply_data = load_ply(first_ply)
    print(f"  First frame: {len(ply_data['x'])} particles")

    if args.preview:
        water_obj = create_water_preview(ply_data)
    else:
        water_obj = create_water_pointcloud(ply_data)
        # Material must be in object slots for GeoNodes SetMaterial to work
        water_obj.data.materials.append(water_mat)
        # Add geometry nodes modifier for fluid meshing
        mod = water_obj.modifiers.new(name="FluidMesher", type='NODES')
        mod.node_group = fluid_geonodes

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = args.frames - 1

    # Per-frame render
    for frame in range(args.start, args.frames):
        ply_path = os.path.join(args.input, f"water_{frame:06d}.ply")
        obj_path = os.path.join(args.input, f"wheel_{frame:06d}.obj")

        if not os.path.exists(ply_path):
            print(f"  Skipping frame {frame}: {ply_path} not found")
            continue

        # Update water
        ply_data = load_ply(ply_path)
        if args.preview:
            # Rebuild point cloud for preview
            old_mesh = water_obj.data
            verts = list(zip(
                ply_data['x'].tolist(),
                ply_data['y'].tolist(),
                ply_data['z'].tolist()
            ))
            new_mesh = bpy.data.meshes.new(f"WaterPoints_{frame}")
            new_mesh.from_pydata(verts, [], [])
            new_mesh.update()
            water_obj.data = new_mesh
            bpy.data.meshes.remove(old_mesh)
        else:
            update_water_mesh(water_obj, ply_data)

        # Load wheel (delete previous)
        for obj in bpy.data.objects:
            if obj.name.startswith("Wheel"):
                bpy.data.objects.remove(obj, do_unlink=True)
        # Clean orphan wheel meshes
        for mesh in bpy.data.meshes:
            if mesh.name.startswith("WheelMesh") and mesh.users == 0:
                bpy.data.meshes.remove(mesh)

        if os.path.exists(obj_path):
            wheel_obj = import_wheel_obj(obj_path, wood_mat)
        else:
            print(f"  Warning: {obj_path} not found")

        # Set frame and render
        scene.frame_set(frame)
        out_path = os.path.join(args.output, f"frame_{frame:06d}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)

        print(f"  Rendered frame {frame}/{args.frames} → {out_path}")

        # Cleanup preview mesh data to avoid memory buildup
        if args.preview:
            for mesh in bpy.data.meshes:
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)

    print("\n" + "=" * 60)
    print(f"Rendering complete. {args.frames} frames in {args.output}/")
    print(f"Assemble video with:")
    print(f"  ffmpeg -framerate 30 -i {args.output}/frame_%06d.png \\")
    print(f"      -c:v libx264 -crf 18 -pix_fmt yuv420p {args.output}/water_wheel.mp4")
    print("=" * 60)


if __name__ == "__main__":
    render_frames()
