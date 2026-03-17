"""Fully GPU particle-to-mesh surface reconstruction using NVIDIA Warp.

Replaces SplashSurf with an all-GPU pipeline:
  1. Load PLY particle positions (binary little-endian, N x 3 float32)
  2. Splat particles to 3D density grid via Gaussian kernel (Warp kernel)
  3. Marching cubes isosurface extraction on GPU (wp.MarchingCubes)
  4. Export mesh as binary PLY

Usage:
    python gpu_mesh_warp.py --input export/flood --output export/flood/meshes_gpu
    python gpu_mesh_warp.py --input export/flood --output export/flood/meshes_gpu \
        --resolution 384 --radius 0.008 --iso 10.0

All heavy work (splatting, marching cubes) stays on GPU. CPU is only used for
PLY I/O and argument parsing.
"""

import os
import sys
import struct
import argparse
import time
import glob
import re
import json
import numpy as np

import warp as wp

wp.init()

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def splat_density_gaussian(
    positions: wp.array(dtype=wp.vec3),
    field: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    inv_dx: float,
    radius: float,
    inv_radius_sq: float,
):
    """Splat each particle onto the density grid with a Gaussian kernel.

    Each particle deposits exp(-||x_cell - x_particle||^2 / (2 * radius^2))
    to all grid cells within `radius` distance.  Atomic adds handle races.

    Coordinates are in normalized [0,1]^3 domain.  Grid cell (i,j,k)
    corresponds to world position ((i+0.5)/nx, (j+0.5)/ny, (k+0.5)/nz).
    """
    tid = wp.tid()
    pos = positions[tid]

    # Particle position in grid-index space (fractional)
    gx = pos[0] * float(nx)
    gy = pos[1] * float(ny)
    gz = pos[2] * float(nz)

    # How many grid cells the kernel radius spans
    r_cells_x = int(wp.ceil(radius * float(nx))) + 1
    r_cells_y = int(wp.ceil(radius * float(ny))) + 1
    r_cells_z = int(wp.ceil(radius * float(nz))) + 1

    ci = int(wp.floor(gx))
    cj = int(wp.floor(gy))
    ck = int(wp.floor(gz))

    for di in range(-r_cells_x, r_cells_x + 1):
        ii = ci + di
        if ii < 0 or ii >= nx:
            continue
        cx = (float(ii) + 0.5) / float(nx)
        ddx = pos[0] - cx

        for dj in range(-r_cells_y, r_cells_y + 1):
            jj = cj + dj
            if jj < 0 or jj >= ny:
                continue
            cy = (float(jj) + 0.5) / float(ny)
            ddy = pos[1] - cy

            for dk in range(-r_cells_z, r_cells_z + 1):
                kk = ck + dk
                if kk < 0 or kk >= nz:
                    continue
                cz = (float(kk) + 0.5) / float(nz)
                ddz = pos[2] - cz

                dist_sq = ddx * ddx + ddy * ddy + ddz * ddz
                # Gaussian: exp(-dist^2 / (2*r^2))  =  exp(-0.5 * dist^2 * inv_r^2)
                if dist_sq < radius * radius * 9.0:  # cut off at 3*sigma
                    w = wp.exp(-0.5 * dist_sq * inv_radius_sq)
                    wp.atomic_add(field, ii, jj, kk, w)


@wp.kernel
def splat_density_gaussian_fast(
    positions: wp.array(dtype=wp.vec3),
    field: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
    radius: float,
    inv_two_r_sq: float,
):
    """Fast Gaussian splat — uniform grid assumed (nx == ny == nz).

    Uses world-space distances directly. Grid cell centers at
    (i+0.5)*dx where dx = 1.0/N.
    """
    tid = wp.tid()
    pos = positions[tid]

    dx = 1.0 / float(nx)
    dy = 1.0 / float(ny)
    dz = 1.0 / float(nz)

    # Number of cells the kernel covers
    r_cells_x = int(wp.ceil(radius / dx)) + 1
    r_cells_y = int(wp.ceil(radius / dy)) + 1
    r_cells_z = int(wp.ceil(radius / dz)) + 1

    ci = int(wp.floor(pos[0] / dx))
    cj = int(wp.floor(pos[1] / dy))
    ck = int(wp.floor(pos[2] / dz))

    cutoff_sq = radius * radius * 9.0  # 3-sigma cutoff

    for di in range(-r_cells_x, r_cells_x + 1):
        ii = ci + di
        if ii < 0 or ii >= nx:
            continue
        cell_x = (float(ii) + 0.5) * dx
        ddx = pos[0] - cell_x

        for dj in range(-r_cells_y, r_cells_y + 1):
            jj = cj + dj
            if jj < 0 or jj >= ny:
                continue
            cell_y = (float(jj) + 0.5) * dy
            ddy = pos[1] - cell_y

            for dk in range(-r_cells_z, r_cells_z + 1):
                kk = ck + dk
                if kk < 0 or kk >= nz:
                    continue
                cell_z = (float(kk) + 0.5) * dz
                ddz = pos[2] - cell_z

                dist_sq = ddx * ddx + ddy * ddy + ddz * ddz
                if dist_sq < cutoff_sq:
                    w = wp.exp(-dist_sq * inv_two_r_sq)
                    wp.atomic_add(field, ii, jj, kk, w)


@wp.kernel
def scale_verts_to_world(
    verts: wp.array(dtype=wp.vec3),
    nx: int, ny: int, nz: int,
    inv_scale: float,
    off_x: float, off_y: float, off_z: float,
):
    """Transform marching-cubes vertices from grid-index space to native world coords.

    Grid-index → [0,1] normalized → native world:
      world = (grid_idx / N) * inv_scale + offset
    """
    tid = wp.tid()
    v = verts[tid]
    verts[tid] = wp.vec3(
        v[0] / float(nx) * inv_scale + off_x,
        v[1] / float(ny) * inv_scale + off_y,
        v[2] / float(nz) * inv_scale + off_z,
    )


# ---------------------------------------------------------------------------
# Surface detail enhancement kernels (run on GPU after marching cubes)
# ---------------------------------------------------------------------------

@wp.func
def warp_hash(x: float, y: float, z: float) -> float:
    """Simple GPU-friendly hash for noise generation."""
    ix = int(wp.floor(x)) * 73856093
    iy = int(wp.floor(y)) * 19349663
    iz = int(wp.floor(z)) * 83492791
    n = ix ^ iy ^ iz
    n = (n >> 13) ^ n
    n = n * (n * n * 15731 + 789221) + 1376312589
    return float(n & 0x7fffffff) / float(0x7fffffff)


@wp.func
def smooth_noise_3d(x: float, y: float, z: float) -> float:
    """Smoothed value noise with trilinear interpolation."""
    ix = int(wp.floor(x))
    iy = int(wp.floor(y))
    iz = int(wp.floor(z))
    fx = x - float(ix)
    fy = y - float(iy)
    fz = z - float(iz)

    # Smoothstep
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)

    # 8 corners
    c000 = warp_hash(float(ix), float(iy), float(iz))
    c100 = warp_hash(float(ix + 1), float(iy), float(iz))
    c010 = warp_hash(float(ix), float(iy + 1), float(iz))
    c110 = warp_hash(float(ix + 1), float(iy + 1), float(iz))
    c001 = warp_hash(float(ix), float(iy), float(iz + 1))
    c101 = warp_hash(float(ix + 1), float(iy), float(iz + 1))
    c011 = warp_hash(float(ix), float(iy + 1), float(iz + 1))
    c111 = warp_hash(float(ix + 1), float(iy + 1), float(iz + 1))

    # Trilinear
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def fbm_noise_4(x: float, y: float, z: float) -> float:
    """4-octave fBm noise (unrolled for Warp compatibility)."""
    v = smooth_noise_3d(x, y, z) * 1.0
    v += smooth_noise_3d(x * 2.0, y * 2.0, z * 2.0) * 0.5
    v += smooth_noise_3d(x * 4.0, y * 4.0, z * 4.0) * 0.25
    v += smooth_noise_3d(x * 8.0, y * 8.0, z * 8.0) * 0.125
    return v / 1.875


@wp.kernel
def compute_vertex_normals(
    verts: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    normals: wp.array(dtype=wp.vec3),
    n_tris: int,
):
    """Compute per-face normal and atomically accumulate to vertex normals."""
    tid = wp.tid()
    if tid >= n_tris:
        return
    i0 = indices[tid * 3]
    i1 = indices[tid * 3 + 1]
    i2 = indices[tid * 3 + 2]
    v0 = verts[i0]
    v1 = verts[i1]
    v2 = verts[i2]
    fn = wp.cross(v1 - v0, v2 - v0)
    wp.atomic_add(normals, i0, fn)
    wp.atomic_add(normals, i1, fn)
    wp.atomic_add(normals, i2, fn)


@wp.kernel
def displace_along_normal(
    verts: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    wave_scale: float,
    wave_amplitude: float,
    time_offset: float,
):
    """Displace vertices along their normal using 4-octave fBm noise.

    This adds wave-like surface detail to the coarse marching-cubes mesh.
    wave_scale: controls the spatial frequency (higher = finer waves)
    wave_amplitude: displacement magnitude in world units
    """
    tid = wp.tid()
    v = verts[tid]
    n = normals[tid]
    n_len = wp.length(n)
    if n_len < 1.0e-8:
        return
    n = n / n_len

    # 4-octave fBm noise at vertex position
    noise_val = fbm_noise_4(
        v[0] * wave_scale + time_offset,
        v[1] * wave_scale + 17.3,
        v[2] * wave_scale * 0.5 + 31.7,  # less Z variation (water is flatter)
    )
    # Center around 0 (-0.5 to 0.5 range)
    disp = (noise_val - 0.5) * 2.0 * wave_amplitude

    # Attenuate displacement near the bottom (only displace the surface)
    # Vertices with downward-facing normals get less displacement
    up_factor = wp.max(n[2], 0.0)  # Z component of normal (0=side, 1=top)
    disp = disp * (0.3 + 0.7 * up_factor)

    verts[tid] = v + n * disp


@wp.kernel
def normalize_normals(normals: wp.array(dtype=wp.vec3)):
    """Normalize accumulated vertex normals."""
    tid = wp.tid()
    n = normals[tid]
    n_len = wp.length(n)
    if n_len > 1.0e-8:
        normals[tid] = n / n_len


def enhance_surface_detail(verts_wp, indices_wp, n_verts, n_tris,
                           wave_scale=0.5, wave_amplitude=0.15,
                           octaves=4, frame=0, device="cuda:0"):
    """Add multi-scale wave displacement to the marching-cubes mesh.

    All computation on GPU. Modifies verts_wp in-place.

    Args:
        verts_wp: Warp array of vertices (world coords)
        indices_wp: Warp array of triangle indices
        n_verts: number of vertices
        n_tris: number of triangles
        wave_scale: spatial frequency of waves (higher = finer)
        wave_amplitude: max displacement in world units (meters)
        octaves: ignored (hardcoded 4 octaves for Warp compat)
        frame: frame number (for time-varying waves)
    """
    # Step 1: Compute vertex normals
    normals = wp.zeros(n_verts, dtype=wp.vec3, device=device)
    wp.launch(compute_vertex_normals, dim=n_tris,
              inputs=[verts_wp, indices_wp, normals, n_tris], device=device)
    wp.launch(normalize_normals, dim=n_verts,
              inputs=[normals], device=device)

    # Step 2: Displace along normals using 4-octave fBm
    time_offset = float(frame) * 0.1  # slow time evolution
    wp.launch(displace_along_normal, dim=n_verts,
              inputs=[verts_wp, normals, wave_scale, wave_amplitude,
                      time_offset],
              device=device)


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def load_ply_particles(ply_path):
    """Load binary little-endian PLY with float32 XYZ vertex positions.

    Returns numpy array of shape (N, 3), dtype float32.
    """
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

        n_verts = 0
        for line_str in header.decode("ascii", errors="replace").split("\n"):
            if line_str.startswith("element vertex"):
                n_verts = int(line_str.split()[-1])
                break

        if n_verts == 0:
            return np.zeros((0, 3), dtype=np.float32)

        raw = f.read(n_verts * 12)  # 3 x float32 = 12 bytes per vertex
        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3).copy()

    return data


def write_ply_mesh(path, verts, indices):
    """Write a triangle mesh as binary little-endian PLY.

    Args:
        path: output file path
        verts: (V, 3) float32 array of vertex positions
        indices: (T*3,) or (T, 3) int32 array of triangle indices
    """
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)

    n_verts = len(verts)
    n_faces = len(indices)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        # Vertex data: N x 3 float32
        f.write(verts.astype(np.float32).tobytes())
        # Face data: for each triangle, write (3, i0, i1, i2)
        face_data = np.empty((n_faces, 4), dtype=np.int32)
        face_data[:, 0] = 3
        face_data[:, 1:] = indices.astype(np.int32)
        # Write as uchar count + 3 x int32 per face
        for i in range(n_faces):
            f.write(struct.pack("<B", 3))
            f.write(struct.pack("<iii", int(face_data[i, 1]),
                                int(face_data[i, 2]),
                                int(face_data[i, 3])))


def write_ply_mesh_fast(path, verts, indices):
    """Write a triangle mesh as binary little-endian PLY — fast bulk write.

    Args:
        path: output file path
        verts: (V, 3) float32 array of vertex positions
        indices: (T*3,) or (T, 3) int32 array of triangle indices
    """
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)

    n_verts = len(verts)
    n_faces = len(indices)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        # Vertex block
        f.write(verts.astype(np.float32).tobytes())
        # Face block — build contiguous buffer:
        # Each face is 1 byte (count=3) + 3 x 4 bytes (int32) = 13 bytes
        face_buf = bytearray(n_faces * 13)
        idx32 = indices.astype(np.int32)
        for i in range(n_faces):
            off = i * 13
            face_buf[off] = 3
            face_buf[off + 1:off + 13] = struct.pack("<iii",
                                                      idx32[i, 0],
                                                      idx32[i, 1],
                                                      idx32[i, 2])
        f.write(face_buf)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def particles_to_mesh_gpu(positions_np, resolution=384, radius=0.008,
                          iso=10.0, inv_scale=1.0, inv_offset=(0.0, 0.0, 0.0),
                          device="cuda:0", enhance=True,
                          wave_scale=0.5, wave_amplitude=0.15,
                          wave_octaves=4, frame=0):
    """Full GPU pipeline: particles -> Gaussian splat -> marching cubes -> enhance -> mesh.

    Args:
        positions_np: (N, 3) float32 array in [0,1]^3 normalized coordinates
        resolution: grid resolution for density field (default 384)
        radius: Gaussian kernel radius in normalized coords (default 0.008)
        iso: isosurface threshold for marching cubes (default 10.0)
        inv_scale: scale factor for sim→world transform
        inv_offset: (x, y, z) offset for sim→world transform
        device: Warp device string
        enhance: whether to add surface wave detail (default True)
        wave_scale: spatial frequency of enhancement waves
        wave_amplitude: displacement magnitude in world units (meters)
        wave_octaves: fBm octaves for detail
        frame: frame number for time-varying waves

    Returns:
        (verts, indices) — numpy arrays, or (None, None) if extraction fails
        verts: (V, 3) float32 in native world coords
        indices: (T*3,) int32 triangle indices
    """
    n_particles = len(positions_np)
    if n_particles < 50:
        return None, None

    nx = ny = nz = resolution

    # Upload particles to GPU — minimize CPU-GPU transfer
    positions_wp = wp.array(positions_np.astype(np.float32), dtype=wp.vec3,
                            device=device)

    # Allocate density field on GPU
    field = wp.zeros((nx, ny, nz), dtype=float, device=device)

    # Precompute kernel constants
    inv_two_r_sq = 1.0 / (2.0 * radius * radius)

    # Gaussian splatting kernel
    wp.launch(
        splat_density_gaussian_fast,
        dim=n_particles,
        inputs=[positions_wp, field, nx, ny, nz, radius, inv_two_r_sq],
        device=device,
    )

    # Marching cubes on GPU
    max_verts = min(nx * ny * nz, 20_000_000)
    max_tris = min(nx * ny * nz * 2, 40_000_000)

    mc = wp.MarchingCubes(
        nx=nx, ny=ny, nz=nz,
        max_verts=max_verts,
        max_tris=max_tris,
        device=device,
    )

    mc.surface(field=field, threshold=iso)

    n_verts = mc.verts.shape[0]
    n_indices = mc.indices.shape[0]

    if n_verts == 0 or n_indices == 0:
        return None, None

    # Transform vertices: grid-index → [0,1] → native world coords (on GPU)
    wp.launch(
        scale_verts_to_world,
        dim=n_verts,
        inputs=[mc.verts, nx, ny, nz,
                float(inv_scale),
                float(inv_offset[0]), float(inv_offset[1]), float(inv_offset[2])],
        device=device,
    )

    # Surface detail enhancement (GPU) — adds wave displacement
    if enhance and n_verts > 100:
        n_tris = n_indices // 3
        enhance_surface_detail(
            mc.verts, mc.indices, n_verts, n_tris,
            wave_scale=wave_scale,
            wave_amplitude=wave_amplitude,
            octaves=wave_octaves,
            frame=frame,
            device=device,
        )

    # Transfer results to CPU (single transfer each)
    verts_np = mc.verts.numpy().astype(np.float32)
    indices_np = mc.indices.numpy().astype(np.int32)

    return verts_np, indices_np


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def find_water_plys(input_dir):
    """Find all water_NNNNNN.ply files in input directory, sorted by frame number."""
    pattern = os.path.join(input_dir, "water_*.ply")
    files = glob.glob(pattern)

    # Extract frame numbers and sort
    result = []
    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.match(r"water_(\d+)\.ply$", basename)
        if match:
            frame_num = int(match.group(1))
            result.append((frame_num, fpath))

    result.sort(key=lambda x: x[0])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GPU particle-to-mesh surface reconstruction using NVIDIA Warp"
    )
    parser.add_argument("--input", required=True,
                        help="Directory containing water_NNNNNN.ply files")
    parser.add_argument("--output", required=True,
                        help="Output directory for mesh PLY files")
    parser.add_argument("--resolution", type=int, default=384,
                        help="Density grid resolution (default: 384)")
    parser.add_argument("--radius", type=float, default=0.008,
                        help="Gaussian kernel radius in normalized coords (default: 0.008)")
    parser.add_argument("--iso", type=float, default=10.0,
                        help="Marching cubes iso-surface threshold (default: 10.0)")
    parser.add_argument("--device", default="cuda:0",
                        help="Warp device (default: cuda:0)")
    parser.add_argument("--start", type=int, default=None,
                        help="First frame number to process (default: all)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last frame number to process (default: all)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable surface detail enhancement")
    parser.add_argument("--wave-scale", type=float, default=0.5,
                        help="Wave spatial frequency (default: 0.5)")
    parser.add_argument("--wave-amp", type=float, default=0.15,
                        help="Wave displacement amplitude in world meters (default: 0.15)")
    parser.add_argument("--wave-octaves", type=int, default=4,
                        help="fBm octaves for wave detail (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load transform from meta.json (sim [0,1] → native world coords)
    meta_path = os.path.join(args.input, "meta.json")
    inv_scale = 1.0
    inv_offset = (0.0, 0.0, 0.0)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "transform" in meta:
            inv_scale = meta["transform"]["inv_scale"]
            inv_offset = tuple(meta["transform"]["inv_offset"])
            print(f"Transform from meta.json: scale={inv_scale:.4f}, "
                  f"offset=({inv_offset[0]:.2f}, {inv_offset[1]:.2f}, {inv_offset[2]:.2f})")
    else:
        print("WARNING: No meta.json found — output will be in [0,1] sim coords")

    print("=" * 60)
    print("GPU Mesh Reconstruction (NVIDIA Warp)")
    print("=" * 60)
    print(f"  Input:      {os.path.abspath(args.input)}")
    print(f"  Output:     {os.path.abspath(args.output)}")
    print(f"  Resolution: {args.resolution}^3")
    print(f"  Radius:     {args.radius}")
    print(f"  ISO:        {args.iso}")
    print(f"  Device:     {args.device}")
    print()

    # Find all input PLY files
    water_files = find_water_plys(args.input)
    if not water_files:
        print(f"ERROR: No water_NNNNNN.ply files found in {args.input}")
        sys.exit(1)

    # Filter by frame range
    if args.start is not None:
        water_files = [(n, p) for n, p in water_files if n >= args.start]
    if args.end is not None:
        water_files = [(n, p) for n, p in water_files if n <= args.end]

    print(f"Found {len(water_files)} PLY files to process")
    print()

    t_total = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0
    total_verts = 0
    total_tris = 0

    for frame_num, ply_path in water_files:
        out_path = os.path.join(args.output, f"water_{frame_num:06d}.ply")

        # Skip if output already exists
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            n_skipped += 1
            continue

        # Load particles from PLY
        t_frame = time.time()
        try:
            positions = load_ply_particles(ply_path)
        except Exception as e:
            print(f"  [{frame_num:06d}] ERROR loading PLY: {e}")
            n_failed += 1
            continue

        if len(positions) < 50:
            print(f"  [{frame_num:06d}] SKIP - only {len(positions)} particles")
            n_failed += 1
            continue

        t_load = time.time() - t_frame

        # GPU pipeline
        t_gpu = time.time()
        verts, indices = particles_to_mesh_gpu(
            positions,
            resolution=args.resolution,
            radius=args.radius,
            iso=args.iso,
            inv_scale=inv_scale,
            inv_offset=inv_offset,
            device=args.device,
            enhance=not args.no_enhance,
            wave_scale=args.wave_scale,
            wave_amplitude=args.wave_amp,
            wave_octaves=args.wave_octaves,
            frame=frame_num,
        )
        t_gpu = time.time() - t_gpu

        if verts is None or indices is None:
            print(f"  [{frame_num:06d}] SKIP - no surface extracted "
                  f"({len(positions):,} particles)")
            n_failed += 1
            continue

        # Write output PLY
        t_write = time.time()
        n_tris = len(indices) // 3
        write_ply_mesh_fast(out_path, verts, indices)
        t_write = time.time() - t_write

        dt_total = time.time() - t_frame
        n_done += 1
        total_verts += len(verts)
        total_tris += n_tris

        file_kb = os.path.getsize(out_path) / 1024.0

        print(f"  [{frame_num:06d}] {len(positions):>8,} particles -> "
              f"{len(verts):>7,} verts, {n_tris:>7,} tris  "
              f"({file_kb:>6.0f} KB)  "
              f"load={t_load:.2f}s gpu={t_gpu:.2f}s write={t_write:.2f}s "
              f"total={dt_total:.2f}s")

    elapsed = time.time() - t_total

    print()
    print("-" * 60)
    print(f"Completed: {n_done} frames meshed, {n_skipped} skipped, "
          f"{n_failed} failed")
    if n_done > 0:
        print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print(f"Throughput: {n_done / elapsed:.2f} frames/sec")
        print(f"Avg mesh:   {total_verts // n_done:,} verts, "
              f"{total_tris // n_done:,} tris per frame")
    print("-" * 60)


if __name__ == "__main__":
    main()
