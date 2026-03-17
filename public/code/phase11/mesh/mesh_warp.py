"""GPU-accelerated particle → mesh surface reconstruction using NVIDIA Warp.

Replaces SplashSurf (CPU) with a fully GPU pipeline:
  1. Load PLY/NPZ particle positions
  2. HashGrid neighbor search → density splatting (cubic spline kernel)
  3. Gaussian smoothing (separable 3-pass)
  4. MarchingCubes → triangle mesh
  5. Write OBJ with normals

Usage:
    python mesh_warp.py --input ./export/flood --output ./export/flood/meshes
"""
import os
import sys
import argparse
import time
import numpy as np

import warp as wp
wp.init()


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.func
def cubic_spline(q: float) -> float:
    """Normalized cubic spline kernel for q = r/h, returns 0 for q >= 1."""
    if q < 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    elif q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


@wp.kernel
def splat_density_normalized(
    positions: wp.array(dtype=wp.vec3),
    field: wp.array3d(dtype=float),
    n_particles: int,
    nx: int, ny: int, nz: int,
    origin_x: float, origin_y: float, origin_z: float,
    dx: float,
    h: float,
    norm: float,
):
    """Splat SPH density to grid: each particle deposits W(r, h) * norm.

    norm should be set so that the field value inside a uniform fluid ≈ 1.0.
    This makes the iso-threshold meaningful (0.5 = surface).
    """
    p = wp.tid()
    if p >= n_particles:
        return

    pos = positions[p]
    gx = (pos[0] - origin_x) / dx
    gy = (pos[1] - origin_y) / dx
    gz = (pos[2] - origin_z) / dx

    r_cells = int(wp.ceil(h / dx)) + 1
    ci = int(wp.floor(gx))
    cj = int(wp.floor(gy))
    ck = int(wp.floor(gz))

    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            for dk in range(-r_cells, r_cells + 1):
                ii = ci + di
                jj = cj + dj
                kk = ck + dk
                if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                    continue
                cx = (float(ii) + 0.5) * dx + origin_x
                cy = (float(jj) + 0.5) * dx + origin_y
                cz = (float(kk) + 0.5) * dx + origin_z
                ddx = pos[0] - cx
                ddy = pos[1] - cy
                ddz = pos[2] - cz
                dist = wp.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                q = dist / h
                w = cubic_spline(q)
                if w > 0.0:
                    wp.atomic_add(field, ii, jj, kk, w * norm)


@wp.kernel
def smooth_x(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    weights: wp.array(dtype=float),
    half_k: int,
):
    """1D Gaussian blur along X axis."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    val = float(0.0)
    for d in range(-half_k, half_k + 1):
        ii = wp.clamp(i + d, 0, nx - 1)
        val += src[ii, j, k] * weights[d + half_k]
    dst[i, j, k] = val


@wp.kernel
def smooth_y(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    weights: wp.array(dtype=float),
    half_k: int,
):
    """1D Gaussian blur along Y axis."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    val = float(0.0)
    for d in range(-half_k, half_k + 1):
        jj = wp.clamp(j + d, 0, ny - 1)
        val += src[i, jj, k] * weights[d + half_k]
    dst[i, j, k] = val


@wp.kernel
def smooth_z(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    weights: wp.array(dtype=float),
    half_k: int,
):
    """1D Gaussian blur along Z axis."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    val = float(0.0)
    for d in range(-half_k, half_k + 1):
        kk = wp.clamp(k + d, 0, nz - 1)
        val += src[i, j, kk] * weights[d + half_k]
    dst[i, j, k] = val


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def make_gaussian_weights(sigma, half_k):
    """Create normalized 1D Gaussian kernel."""
    x = np.arange(-half_k, half_k + 1, dtype=np.float32)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return w


def compute_normals(verts, faces):
    """Compute per-vertex normals from face normals (area-weighted)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-10)
    return normals


def write_obj(path, verts, faces, normals=None):
    """Write OBJ with optional normals."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for face in faces + 1:
                f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")
        else:
            for face in faces + 1:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")


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


def particles_to_mesh(positions_np, grid_res=250, sigma=1.0,
                      h_factor=3.5, iso=0.5,
                      clip_lo=(0.07, 0.07, 0.0),
                      clip_hi=(0.93, 0.93, 0.5),
                      device="cuda:0"):
    """Full GPU pipeline: particles → SPH splat → smooth → marching cubes → mesh.

    h_factor: kernel radius = h_factor * avg_particle_spacing
    """
    # Clip particles to AABB
    mask = ((positions_np[:, 0] >= clip_lo[0]) & (positions_np[:, 0] <= clip_hi[0]) &
            (positions_np[:, 1] >= clip_lo[1]) & (positions_np[:, 1] <= clip_hi[1]) &
            (positions_np[:, 2] >= clip_lo[2]) & (positions_np[:, 2] <= clip_hi[2]))
    positions_np = positions_np[mask]
    if len(positions_np) < 100:
        return None, None, None

    n_particles = len(positions_np)

    # Grid setup
    lo = np.array(clip_lo, dtype=np.float32)
    hi = np.array(clip_hi, dtype=np.float32)
    span = hi - lo
    dx = float(span.max() / grid_res)
    nx = int(np.ceil(span[0] / dx))
    ny = int(np.ceil(span[1] / dx))
    nz = int(np.ceil(span[2] / dx))

    # Estimate average particle spacing → kernel radius h
    # For a thin water layer, use the XY extent and particle count
    zmin, zmax = positions_np[:, 2].min(), positions_np[:, 2].max()
    water_volume = span[0] * span[1] * max(zmax - zmin, dx * 3)
    avg_spacing = (water_volume / n_particles) ** (1.0 / 3.0)
    h = float(h_factor * avg_spacing)

    # Normalization: in a uniform fluid with spacing `s`, the sum of
    # W(r, h) over all neighbors ≈ (4/3 π h³) / s³ * <W_avg>.
    # We want field ≈ 1.0 inside, so norm = 1 / expected_sum.
    # Empirically, norm = s³ / h³ works well.
    norm = float(avg_spacing ** 3 / h ** 3)

    # Upload particles to GPU
    positions_wp = wp.array(positions_np, dtype=wp.vec3, device=device)

    # SPH density splatting
    field = wp.zeros((nx, ny, nz), dtype=float, device=device)

    wp.launch(
        splat_density_normalized,
        dim=n_particles,
        inputs=[positions_wp, field, n_particles, nx, ny, nz,
                lo[0], lo[1], lo[2], dx, h, norm],
        device=device,
    )

    # Gaussian smooth (separable 3-pass)
    half_k = max(int(sigma * 2), 1)
    weights_np = make_gaussian_weights(sigma, half_k)
    weights_wp = wp.array(weights_np, dtype=float, device=device)
    tmp = wp.zeros((nx, ny, nz), dtype=float, device=device)

    wp.launch(smooth_x, dim=(nx, ny, nz),
              inputs=[field, tmp, nx, ny, nz, weights_wp, half_k], device=device)
    wp.launch(smooth_y, dim=(nx, ny, nz),
              inputs=[tmp, field, nx, ny, nz, weights_wp, half_k], device=device)
    wp.launch(smooth_z, dim=(nx, ny, nz),
              inputs=[field, tmp, nx, ny, nz, weights_wp, half_k], device=device)

    # Adaptive iso
    field_np = tmp.numpy()
    max_val = field_np.max()
    if max_val < iso * 0.1:
        iso = max_val * 0.3
    if max_val < 1e-6:
        return None, None, None

    # Upload smoothed field back (if we modified iso, use the tmp which has final smoothed data)
    field_gpu = wp.array(field_np, dtype=float, shape=(nx, ny, nz), device=device)

    # Marching cubes
    mc = wp.MarchingCubes(
        nx=nx, ny=ny, nz=nz,
        device=device,
    )
    mc.surface(field_gpu, iso)

    verts_np = mc.verts.numpy()  # grid-index space
    indices_np = mc.indices.numpy()

    if len(verts_np) == 0 or len(indices_np) == 0:
        return None, None, None

    faces_np = indices_np.reshape(-1, 3)

    # Convert from grid-index to world coordinates
    verts_world = verts_np * dx + lo

    # Compute normals
    normals = compute_normals(verts_world, faces_np)

    return verts_world, faces_np, normals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with NPZ/PLY files")
    parser.add_argument("--output", required=True, help="Output directory for OBJ files")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=299)
    parser.add_argument("--grid-res", type=int, default=250,
                        help="Grid resolution for density field")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (in grid cells)")
    parser.add_argument("--h-factor", type=float, default=3.5,
                        help="Kernel radius = h_factor * avg_particle_spacing")
    parser.add_argument("--iso", type=float, default=0.5,
                        help="Iso-surface threshold")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Warp GPU Surface Reconstruction")
    print(f"  grid_res={args.grid_res}, sigma={args.sigma}, iso={args.iso}")
    print(f"  device={args.device}")

    t0 = time.time()
    n_done = 0

    for i in range(args.start, args.end + 1):
        npz = os.path.join(args.input, f"water_{i:06d}.npz")
        ply = os.path.join(args.input, f"water_{i:06d}.ply")
        obj = os.path.join(args.output, f"water_{i:06d}.obj")

        if os.path.exists(obj) and os.path.getsize(obj) > 100:
            continue

        # Load particle positions
        if os.path.exists(npz):
            data = np.load(npz)
            positions = data['x'].astype(np.float32)
        elif os.path.exists(ply):
            positions = load_ply(ply)
        else:
            if i % 30 == 0:
                print(f"  [{i:4d}] SKIP — no data")
            continue

        t1 = time.time()
        verts, faces, normals = particles_to_mesh(
            positions,
            grid_res=args.grid_res,
            sigma=args.sigma,
            h_factor=args.h_factor,
            iso=args.iso,
            device=args.device,
        )

        if verts is None:
            if i % 30 == 0 or i < 5:
                print(f"  [{i:4d}] SKIP — too few particles")
            continue

        write_obj(obj, verts, faces, normals)
        dt = time.time() - t1
        n_done += 1

        if i % 30 == 0 or i < 5:
            sz = os.path.getsize(obj) / 1024
            print(f"  [{i:4d}] {len(verts):,} verts, {len(faces):,} tris — "
                  f"{sz:.0f}KB ({dt:.2f}s)")

    elapsed = time.time() - t0
    print(f"\nDone! {n_done} frames in {elapsed:.1f}s "
          f"({n_done/elapsed:.1f} fr/s)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()
