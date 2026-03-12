"""Convert particle NPZ files → continuous flood surface mesh.

Single water plane covering the domain with height varying by particle density.
Deeper where more particles (river channel), shallow flood level elsewhere.
Building footprint is cut out.

Usage:
    python mesh_particles.py --input ./export/river --output ./export/river/meshes \
        --meta ./export/river/meta.json
"""
import os
import sys
import json
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

p = argparse.ArgumentParser()
p.add_argument("--input", default="./export/river")
p.add_argument("--output", default="./export/river/meshes")
p.add_argument("--meta", default=None, help="Path to meta.json for building bounds")
p.add_argument("--frame", default=None, help="Comma-separated frames")
p.add_argument("--grid", type=int, default=200, help="XY grid resolution")
p.add_argument("--sigma", type=float, default=5.0, help="Density smoothing sigma")
p.add_argument("--margin", type=float, default=0.025,
               help="Extra margin around building cutout (normalized)")
args = p.parse_args()

os.makedirs(args.output, exist_ok=True)

# Frame list
if args.frame:
    frames = [int(f.strip()) for f in args.frame.split(",")]
else:
    files = sorted(f for f in os.listdir(args.input)
                   if f.startswith("water_") and f.endswith(".npz"))
    frames = [int(f.split("_")[1].split(".")[0]) for f in files]


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def create_flood_surface(positions, grid_res=200, sigma=5.0,
                          base_z=0.075, max_depth=0.02,
                          building_bounds=None, margin=0.025):
    """Single continuous flood surface with particle-driven depth variation.

    1. Compute particle density on XY grid
    2. Heavy Gaussian smooth → gentle height map
    3. Surface = base_z + density_fraction * max_depth
    4. Cut out building footprint
    5. Triangulate
    """
    dx = 1.0 / grid_res

    # Domain bounds (skip BOUND padding cells)
    pad = 0.025
    i_min = int(pad * grid_res)
    i_max = grid_res - i_min
    j_min = int(pad * grid_res)
    j_max = grid_res - j_min

    # Particle density
    gi_x = np.clip((positions[:, 0] * grid_res).astype(int), 0, grid_res - 1)
    gi_y = np.clip((positions[:, 1] * grid_res).astype(int), 0, grid_res - 1)

    density = np.zeros((grid_res, grid_res), dtype=np.float32)
    np.add.at(density, (gi_x, gi_y), 1.0)

    # Heavy Gaussian smooth for gentle variation
    density_smooth = gaussian_filter(density, sigma=sigma)

    # Normalize density to [0, 1]
    d_max = density_smooth.max()
    if d_max > 0:
        depth_frac = density_smooth / d_max
    else:
        depth_frac = density_smooth

    # Surface height: base flood + depth variation
    surface = base_z + depth_frac * max_depth

    # Building mask: True = inside building (no water)
    building_mask = np.zeros((grid_res, grid_res), dtype=bool)
    if building_bounds is not None:
        bx0 = building_bounds[0][0] - margin
        by0 = building_bounds[0][1] - margin
        bx1 = building_bounds[1][0] + margin
        by1 = building_bounds[1][1] + margin
        for i in range(i_min, i_max):
            cx = (i + 0.5) * dx
            if cx < bx0 or cx > bx1:
                continue
            for j in range(j_min, j_max):
                cy = (j + 0.5) * dx
                if cy >= by0 and cy <= by1:
                    building_mask[i, j] = True

    # Generate vertices
    vertex_index = np.full((grid_res, grid_res), -1, dtype=np.int32)
    verts = []

    for i in range(i_min, i_max):
        cx = (i + 0.5) * dx
        for j in range(j_min, j_max):
            if building_mask[i, j]:
                continue
            cy = (j + 0.5) * dx
            cz = surface[i, j]
            vertex_index[i, j] = len(verts)
            verts.append([cx, cy, cz])

    if len(verts) < 4:
        return None, None

    # Triangulate
    faces = []
    for i in range(i_min, i_max - 1):
        for j in range(j_min, j_max - 1):
            v00 = vertex_index[i, j]
            v10 = vertex_index[i + 1, j]
            v01 = vertex_index[i, j + 1]
            v11 = vertex_index[i + 1, j + 1]
            if v00 < 0 or v10 < 0 or v01 < 0 or v11 < 0:
                continue
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if not faces:
        return None, None

    return np.array(verts, dtype=np.float64), np.array(faces)


# Load building bounds
building_bounds = None
base_z = 0.075
max_depth = 0.02

if args.meta and os.path.exists(args.meta):
    with open(args.meta) as f:
        _meta = json.load(f)
    bz = _meta["mesh_bounds"]
    building_bounds = bz
    base_z = bz[0][2] + 0.005  # 5mm above building base = shallow flood
    max_depth = 0.025  # max 25mm additional depth = ~0.6m at 25x scale
    print(f"Building bounds: {bz}")
    print(f"Water Z: base={base_z:.4f}, max_depth={max_depth:.4f}")
    print(f"  → base={base_z*25:.2f}m, max surface={((base_z+max_depth)*25):.2f}m")

print(f"Meshing {len(frames)} frames (continuous flood surface): grid={args.grid}, sigma={args.sigma}")

for i, frame in enumerate(frames):
    npz_path = os.path.join(args.input, f"water_{frame:06d}.npz")
    if not os.path.exists(npz_path):
        print(f"  [{i+1}/{len(frames)}] Frame {frame}: MISSING")
        continue

    data = np.load(npz_path)
    positions = data["x"]

    verts, faces = create_flood_surface(
        positions, grid_res=args.grid, sigma=args.sigma,
        base_z=base_z, max_depth=max_depth,
        building_bounds=building_bounds, margin=args.margin)

    obj_path = os.path.join(args.output, f"water_{frame:06d}.obj")

    if verts is None:
        print(f"  [{i+1}/{len(frames)}] Frame {frame}: no mesh (empty)")
        continue

    write_obj(obj_path, verts, faces)
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    print(f"  [{i+1}/{len(frames)}] Frame {frame}: {len(positions):,} particles → "
          f"{len(verts):,} verts, {len(faces):,} tris, z=[{z_min:.4f}..{z_max:.4f}]")

print("Done!")
