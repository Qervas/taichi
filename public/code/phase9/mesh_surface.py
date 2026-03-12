"""PLY particle → OBJ mesh surface reconstruction.

Uses SplashSurf if available, otherwise falls back to Python
(scipy gaussian density field + marching cubes via trimesh or skimage).

Usage:
    python mesh_surface.py --input ./export/flood --output ./export/flood/meshes
    python mesh_surface.py --input ./export/flood --output ./export/flood/meshes --start 0 --end 299
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


def mesh_splashsurf(ply_path, obj_path, particle_radius=0.005,
                    smoothing_length=4.0, cube_size=1.5,
                    aabb_min="0.07 0.07 0.0", aabb_max="0.93 0.93 1.0"):
    """Run SplashSurf on a single PLY file."""
    cmd = [
        "splashsurf", "reconstruct",
        "-i", ply_path,
        "-o", obj_path,
        f"--particle-radius={particle_radius}",
        f"--smoothing-length={smoothing_length}",
        f"--cube-size={cube_size}",
        "--normals=on",
        f"--particle-aabb-min", *aabb_min.split(),
        f"--particle-aabb-max", *aabb_max.split(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  SplashSurf error: {r.stderr[:200]}")
        return False
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


def mesh_python(ply_path, obj_path, grid_res=200, sigma=2.0, iso=0.5,
                clip_xy=(0.07, 0.93), clip_z=(0.0, 1.0)):
    """Python fallback: density field + marching cubes.

    Uses skimage if available, otherwise trimesh (which wraps scipy).
    Only needs scipy + numpy as hard deps.
    """
    from scipy.ndimage import gaussian_filter

    # Try skimage first, fall back to PyMCubes
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

    # Clip to AABB
    mask = ((data[:, 0] >= clip_xy[0]) & (data[:, 0] <= clip_xy[1]) &
            (data[:, 1] >= clip_xy[0]) & (data[:, 1] <= clip_xy[1]) &
            (data[:, 2] >= clip_z[0]) & (data[:, 2] <= clip_z[1]))
    data = data[mask]
    if len(data) < 100:
        return False

    # Bin into density grid
    lo = np.array([clip_xy[0], clip_xy[0], clip_z[0]], dtype=np.float32)
    hi = np.array([clip_xy[1], clip_xy[1], clip_z[1]], dtype=np.float32)
    span = hi - lo
    idx = ((data - lo) / span * (grid_res - 1)).astype(int)
    idx = np.clip(idx, 0, grid_res - 1)

    density = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    # Smooth
    density = gaussian_filter(density, sigma=sigma)

    # Adaptive iso level
    if density.max() < iso * 0.1:
        iso = density.max() * 0.3

    # Marching cubes
    if mc_source == "skimage":
        verts, faces, normals, _ = mc_func(density, level=iso)
    elif mc_source == "pymcubes":
        import mcubes
        verts, faces = mcubes.marching_cubes(density, iso)
        # Compute vertex normals from face normals
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
        # trimesh marching cubes (needs skimage internally)
        import trimesh
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
            density > iso, pitch=1.0)
        verts = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals)

    # Map back to sim coordinates
    verts = verts / (grid_res - 1) * span + lo

    # Write OBJ
    with open(obj_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces + 1:  # OBJ is 1-indexed
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with PLY files")
    parser.add_argument("--output", required=True, help="Output directory for OBJ files")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=299)
    parser.add_argument("--method", choices=["auto", "splashsurf", "python"],
                        default="auto")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    use_ss = args.method == "splashsurf" or (args.method == "auto" and has_splashsurf())
    method_name = "SplashSurf" if use_ss else "Python (scipy+marching cubes)"
    print(f"Surface reconstruction: {method_name}")

    for i in range(args.start, args.end + 1):
        ply = os.path.join(args.input, f"water_{i:06d}.ply")
        obj = os.path.join(args.output, f"water_{i:06d}.obj")

        if not os.path.exists(ply):
            print(f"  [{i:4d}] SKIP — no PLY")
            continue
        if os.path.exists(obj) and os.path.getsize(obj) > 100:
            continue  # already done

        if use_ss:
            ok = mesh_splashsurf(ply, obj)
        else:
            ok = mesh_python(ply, obj)

        if i % 30 == 0 or i < 5:
            sz = os.path.getsize(obj) / 1024 if ok and os.path.exists(obj) else 0
            print(f"  [{i:4d}] {'OK' if ok else 'FAIL'} — {sz:.0f} KB")

    print("Done!")


if __name__ == "__main__":
    main()
