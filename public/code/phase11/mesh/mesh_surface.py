"""PLY/NPZ particle -> OBJ mesh surface reconstruction with foam vertex colors.

Improvements over Phase 9:
- Higher grid resolution (350 vs 200) for finer surface detail
- Lower Gaussian sigma (1.2 vs 2.0) to preserve waves
- Foam attribute baked from NPZ particle data to per-vertex values
- Outputs OBJ + companion foam .npy per frame

Usage:
    python mesh_surface.py --input ./export/flood --output ./export/flood/meshes
"""
import os
import sys
import glob
import shutil
import subprocess
import argparse
import numpy as np


def has_splashsurf():
    return shutil.which("splashsurf") is not None


def mesh_splashsurf(ply_path, obj_path, particle_radius=0.004,
                    smoothing_length=3.5, cube_size=1.0,
                    aabb_min="0.07 0.07 0.0", aabb_max="0.93 0.93 0.5"):
    """Run SplashSurf on a single PLY file."""
    cmd = [
        "splashsurf", "reconstruct",
        ply_path,
        "-o", obj_path,
        f"--particle-radius={particle_radius}",
        f"--smoothing-length={smoothing_length}",
        f"--cube-size={cube_size}",
        "--normals=on",
        "--particle-aabb-min", *aabb_min.split(),
        "--particle-aabb-max", *aabb_max.split(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  SplashSurf error: {r.stderr[:200]}")
        return False
    return True


def bake_foam_to_mesh(npz_path, obj_path, foam_path,
                      clip_xy=(0.07, 0.93), clip_z=(0.0, 0.5)):
    """Interpolate foam from NPZ particles onto OBJ mesh vertices via KDTree."""
    from scipy.spatial import cKDTree

    data_npz = np.load(npz_path)
    if 'foam' not in data_npz or 'x' not in data_npz:
        return False

    particles = data_npz['x'].astype(np.float32)
    foam_data = data_npz['foam'].astype(np.float32)
    if foam_data.max() < 1e-6:
        return False

    # Read mesh vertices from OBJ
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v ') and not line.startswith('vn'):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts, dtype=np.float32)
    if len(verts) == 0:
        return False

    # Build KDTree from particles, query for each mesh vertex
    tree = cKDTree(particles)
    k = 8  # average over 8 nearest particles
    dists, idxs = tree.query(verts, k=k, workers=-1)

    # Inverse-distance weighted foam
    weights = 1.0 / np.maximum(dists, 1e-8)
    foam_vals = foam_data[idxs]  # (n_verts, k)
    vert_foam = (foam_vals * weights).sum(axis=1) / weights.sum(axis=1)
    vert_foam = np.clip(vert_foam, 0, 1).astype(np.float32)

    np.save(foam_path, vert_foam)
    return True


def load_ply(ply_path):
    """Load binary PLY with float32 xyz vertices."""
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        n_verts = 0
        for l in header.decode().split("\n"):
            if l.startswith("element vertex"):
                n_verts = int(l.split()[-1])
                break
        data = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(-1, 3)
    return data


def mesh_python_with_foam(npz_path, obj_path, foam_path,
                          grid_res=512, sigma=0.7, iso=0.5,
                          clip_xy=(0.07, 0.93), clip_z=(0.0, 1.0),
                          tight_bbox=True, padding=0.03):
    """Python fallback: density field + marching cubes + foam interpolation.

    Uses tight bounding box around actual particles for maximum effective
    resolution. With grid_res=512 and tight bbox, effective voxel size is
    much smaller than using the full domain.
    """
    from scipy.ndimage import gaussian_filter

    try:
        from skimage.measure import marching_cubes as mc_func
        mc_source = "skimage"
    except ImportError:
        mc_func = None
        try:
            import mcubes
            mc_source = "pymcubes"
        except ImportError:
            mc_source = "trimesh"

    # Load NPZ
    data_npz = np.load(npz_path)
    data = data_npz['x'].astype(np.float32)
    has_foam = 'foam' in data_npz
    foam_data = data_npz['foam'].astype(np.float32) if has_foam else None

    if len(data) == 0:
        return False

    # Clip to domain AABB
    mask = ((data[:, 0] >= clip_xy[0]) & (data[:, 0] <= clip_xy[1]) &
            (data[:, 1] >= clip_xy[0]) & (data[:, 1] <= clip_xy[1]) &
            (data[:, 2] >= clip_z[0]) & (data[:, 2] <= clip_z[1]))
    data = data[mask]
    if foam_data is not None:
        foam_data = foam_data[mask]
    if len(data) < 100:
        return False

    # Tight bounding box: focus resolution on where particles actually are
    if tight_bbox:
        lo = data.min(axis=0) - padding
        hi = data.max(axis=0) + padding
        # Clamp to domain
        lo = np.maximum(lo, [clip_xy[0], clip_xy[0], clip_z[0]])
        hi = np.minimum(hi, [clip_xy[1], clip_xy[1], clip_z[1]])
    else:
        lo = np.array([clip_xy[0], clip_xy[0], clip_z[0]], dtype=np.float32)
        hi = np.array([clip_xy[1], clip_xy[1], clip_z[1]], dtype=np.float32)

    span = (hi - lo).astype(np.float32)
    # Use anisotropic grid: same voxel size in all axes, based on longest span
    max_span = span.max()
    voxel_size = max_span / (grid_res - 1)
    grid_dims = np.maximum((span / voxel_size + 1).astype(int), 3)

    idx = ((data - lo) / voxel_size).astype(int)
    idx = np.clip(idx, 0, grid_dims - 1)

    density = np.zeros(tuple(grid_dims), dtype=np.float32)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    # Foam grid
    foam_grid = np.zeros(tuple(grid_dims), dtype=np.float32)
    if foam_data is not None and foam_data.max() > 0:
        np.add.at(foam_grid, (idx[:, 0], idx[:, 1], idx[:, 2]), foam_data)
        count_mask = density > 0
        foam_grid[count_mask] /= density[count_mask]

    # Smooth density
    density = gaussian_filter(density, sigma=sigma)
    if foam_data is not None:
        foam_grid = gaussian_filter(foam_grid, sigma=sigma * 0.7)

    # Adaptive iso level
    if density.max() < iso * 0.1:
        iso = density.max() * 0.3

    # Marching cubes
    if mc_source == "skimage":
        verts, faces, normals, _ = mc_func(density, level=iso)
    elif mc_source == "pymcubes":
        import mcubes
        verts, faces = mcubes.marching_cubes(density, iso)
        normals = np.zeros_like(verts)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
        fn /= np.maximum(fn_norm, 1e-10)
        for i in range(3):
            np.add.at(normals, faces[:, i], fn)
        n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(n_norm, 1e-10)
    else:
        import trimesh
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
            density > iso, pitch=1.0)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals)

    # Map back to sim coordinates
    verts_sim = verts * voxel_size + lo

    # Interpolate foam at vertex positions
    vert_foam = np.zeros(len(verts_sim), dtype=np.float32)
    if foam_data is not None and foam_grid.max() > 0:
        vert_idx = (verts_sim - lo) / voxel_size
        vert_idx = np.clip(vert_idx, 0, np.array(grid_dims) - 1)
        i0 = vert_idx.astype(int)
        i0 = np.clip(i0, 0, np.array(grid_dims) - 1)
        vert_foam = foam_grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        vert_foam = np.clip(vert_foam, 0, 1).astype(np.float32)

    # Write OBJ
    with open(obj_path, "w") as f:
        for v in verts_sim:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces + 1:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

    if foam_path is not None:
        np.save(foam_path, vert_foam)

    return True


def mesh_python(ply_path, obj_path, grid_res=350, sigma=1.2, iso=0.5,
                clip_xy=(0.07, 0.93), clip_z=(0.0, 1.0)):
    """Python fallback without foam (PLY input only)."""
    from scipy.ndimage import gaussian_filter

    try:
        from skimage.measure import marching_cubes as mc_func
        mc_source = "skimage"
    except ImportError:
        mc_func = None
        try:
            import mcubes
            mc_source = "pymcubes"
        except ImportError:
            mc_source = "trimesh"

    data = load_ply(ply_path)
    if len(data) == 0:
        return False

    mask = ((data[:, 0] >= clip_xy[0]) & (data[:, 0] <= clip_xy[1]) &
            (data[:, 1] >= clip_xy[0]) & (data[:, 1] <= clip_xy[1]) &
            (data[:, 2] >= clip_z[0]) & (data[:, 2] <= clip_z[1]))
    data = data[mask]
    if len(data) < 100:
        return False

    lo = np.array([clip_xy[0], clip_xy[0], clip_z[0]], dtype=np.float32)
    hi = np.array([clip_xy[1], clip_xy[1], clip_z[1]], dtype=np.float32)
    span = hi - lo
    idx = ((data - lo) / span * (grid_res - 1)).astype(int)
    idx = np.clip(idx, 0, grid_res - 1)

    density = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    density = gaussian_filter(density, sigma=sigma)

    if density.max() < iso * 0.1:
        iso = density.max() * 0.3

    if mc_source == "skimage":
        verts, faces, normals, _ = mc_func(density, level=iso)
    elif mc_source == "pymcubes":
        import mcubes
        verts, faces = mcubes.marching_cubes(density, iso)
        normals = np.zeros_like(verts)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
        fn /= np.maximum(fn_norm, 1e-10)
        for i in range(3):
            np.add.at(normals, faces[:, i], fn)
        n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(n_norm, 1e-10)
    else:
        import trimesh
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
            density > iso, pitch=1.0)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals)

    verts = verts / (grid_res - 1) * span + lo

    with open(obj_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces + 1:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with NPZ/PLY files")
    parser.add_argument("--output", required=True, help="Output directory for OBJ files")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=299)
    parser.add_argument("--grid-res", type=int, default=350)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--method", choices=["auto", "splashsurf", "python"],
                        default="auto")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    use_ss = args.method == "splashsurf" or (args.method == "auto" and has_splashsurf())
    method_name = "SplashSurf" if use_ss else "Python (scipy+marching cubes)"
    print(f"Surface reconstruction: {method_name}")
    print(f"  grid_res={args.grid_res}, sigma={args.sigma}")

    for i in range(args.start, args.end + 1):
        npz = os.path.join(args.input, f"water_{i:06d}.npz")
        ply = os.path.join(args.input, f"water_{i:06d}.ply")
        obj = os.path.join(args.output, f"water_{i:06d}.obj")
        foam_path = os.path.join(args.output, f"foam_{i:06d}.npy")

        if os.path.exists(obj) and os.path.getsize(obj) > 100:
            continue  # already done

        if use_ss:
            if not os.path.exists(ply):
                print(f"  [{i:4d}] SKIP — no PLY")
                continue
            ok = mesh_splashsurf(ply, obj)
            # Bake foam from NPZ onto SplashSurf mesh vertices
            if ok and os.path.exists(npz):
                bake_foam_to_mesh(npz, obj, foam_path)
        else:
            # Prefer NPZ (has foam data) over PLY
            if os.path.exists(npz):
                ok = mesh_python_with_foam(npz, obj, foam_path,
                                           grid_res=args.grid_res,
                                           sigma=args.sigma)
            elif os.path.exists(ply):
                ok = mesh_python(ply, obj, grid_res=args.grid_res,
                                 sigma=args.sigma)
            else:
                print(f"  [{i:4d}] SKIP — no data")
                continue

        if i % 30 == 0 or i < 5:
            sz = os.path.getsize(obj) / 1024 if ok and os.path.exists(obj) else 0
            foam_sz = os.path.getsize(foam_path) / 1024 if os.path.exists(foam_path) else 0
            print(f"  [{i:4d}] {'OK' if ok else 'FAIL'} — mesh {sz:.0f}KB, foam {foam_sz:.0f}KB")

    print("Done!")


if __name__ == "__main__":
    main()
