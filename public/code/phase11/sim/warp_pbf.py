"""Phase 11 — Position-Based Fluids (PBF) solver using NVIDIA Warp.

Implements Macklin & Muller 2013 "Position Based Fluids" on GPU via warp-lang.
Designed as a drop-in alternative to the FLIP solver for flood simulation.

Pipeline per frame:
  1. Inject particles at left boundary
  2. Apply gravity → predict positions
  3. Build hash grid for neighbor search
  4. Iterative density constraint projection (Jacobi)
  5. Update velocities from position delta
  6. Apply XSPH viscosity
  7. Enforce domain boundaries + solid obstacle
  8. Export NPZ + PLY

Usage:
    python phases/phase11/warp_pbf.py
    python phases/phase11/warp_pbf.py --solid export/flood/solid.npy --meta export/flood/meta.json
"""
import os
import sys
import time
import json
import argparse
import numpy as np

import warp as wp

# ============================================================================
# Configuration
# ============================================================================
N_GRID = 192                 # neighbor search grid resolution
MAX_PARTICLES = 10_000_000
DT = 0.005                   # timestep per frame
N_SUBSTEPS = 2               # substeps per frame for stability
N_FRAMES = 300
KERNEL_RADIUS = 0.01         # SPH kernel support radius (h)
REST_DENSITY = 1000.0         # target density (kg/m^3)
SOLVER_ITERATIONS = 4         # Jacobi constraint iterations per substep
VISCOSITY = 0.01              # XSPH viscosity coefficient
EPSILON_LAMBDA = 100.0        # relaxation denominator for lambda (CFM)
CORR_K = 0.0001              # artificial pressure strength (tensile instability)
CORR_N = 4.0                 # artificial pressure exponent
CORR_DQ = 0.3                # artificial pressure reference distance (fraction of h)
GRAVITY = wp.vec3(0.0, 0.0, -9.81)

# Inflow parameters
INJECT_RATE = 15000           # particles per frame
INFLOW_VELOCITY = wp.vec3(2.0, 0.0, 0.0)
INFLOW_WIDTH = 0.06           # injection slab width in X
INFLOW_Y_LO = 0.05
INFLOW_Y_HI = 0.95
MAX_Z_FRAC = 0.55            # water level up to 55% of building height

# Domain
DOMAIN_LO = 0.0
DOMAIN_HI = 1.0
BOUND_CELLS = 3               # boundary padding in grid cells

# Export
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(SCRIPT_DIR, "export", "flood_warp")

# ============================================================================
# Precomputed kernel constants
# ============================================================================
PI = 3.14159265358979323846
H = KERNEL_RADIUS
H2 = H * H
H3 = H * H * H
H6 = H3 * H3
H9 = H6 * H3

# Poly6: W(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
POLY6_COEFF = 315.0 / (64.0 * PI * H9)
# Spiky gradient: grad W = -45 / (pi * h^6) * (h - r)^2 * (r_vec / r)
SPIKY_GRAD_COEFF = -45.0 / (PI * H6)


# ============================================================================
# Warp kernels
# ============================================================================

@wp.kernel
def predict_positions(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pos_pred: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    n_particles: int,
):
    """Apply gravity and predict new positions: x* = x + dt*v."""
    i = wp.tid()
    if i >= n_particles:
        return
    v = vel[i] + gravity * dt
    vel[i] = v
    pos_pred[i] = pos[i] + v * dt


@wp.kernel
def compute_lambda(
    pos_pred: wp.array(dtype=wp.vec3),
    lambdas: wp.array(dtype=float),
    grid: wp.uint64,
    n_particles: int,
    h: float,
    h2: float,
    poly6_coeff: float,
    spiky_grad_coeff: float,
    rest_density: float,
    epsilon_lambda: float,
):
    """Compute density constraint lambda_i for each particle."""
    i = wp.tid()
    if i >= n_particles:
        return

    pi_pos = pos_pred[i]

    # Accumulate density via Poly6 kernel (float() for dynamic loop mutation)
    rho_i = float(0.0)
    grad_sum_sq = float(0.0)    # sum of |grad_k C_i|^2 for k != i
    grad_i = wp.vec3(float(0.0), float(0.0), float(0.0))  # grad_i C_i (accumulated)

    # Self-contribution to density: poly6(0) = poly6_coeff * h^6
    rho_i = rho_i + poly6_coeff * h2 * h2 * h2

    # Neighbor query
    query = wp.hash_grid_query(grid, pi_pos, h)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        if idx == i:
            continue
        if idx >= n_particles:
            continue

        r_vec = pi_pos - pos_pred[idx]
        r2 = wp.dot(r_vec, r_vec)

        if r2 < h2 and r2 > 1.0e-12:
            r = wp.sqrt(r2)

            # Poly6 for density
            diff = h2 - r2
            rho_i = rho_i + poly6_coeff * diff * diff * diff

            # Spiky gradient for constraint gradient
            diff_r = h - r
            grad_w = spiky_grad_coeff * diff_r * diff_r * (r_vec / r)

            # grad_k C_i = (1/rho0) * grad W(pi - pj)  for k = j
            grad_k = grad_w / rest_density
            grad_sum_sq = grad_sum_sq + wp.dot(grad_k, grad_k)
            # grad_i C_i = sum of (1/rho0) * grad W(pi - pj)
            grad_i = grad_i + grad_w / rest_density

    # C_i = rho_i / rho0 - 1
    C_i = rho_i / rest_density - 1.0

    # Denominator: sum |grad_k C_i|^2 for all k (including i) + epsilon
    denom = grad_sum_sq + wp.dot(grad_i, grad_i) + epsilon_lambda

    # lambda_i = - C_i / denom
    lam = 0.0
    if C_i > 0.0:
        # Only correct when density exceeds rest density (incompressibility)
        lam = -C_i / denom

    lambdas[i] = lam


@wp.kernel
def compute_delta_pos(
    pos_pred: wp.array(dtype=wp.vec3),
    lambdas: wp.array(dtype=float),
    delta_pos: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    n_particles: int,
    h: float,
    h2: float,
    poly6_coeff: float,
    spiky_grad_coeff: float,
    rest_density: float,
    corr_k: float,
    corr_n: float,
    corr_dq_dist: float,
):
    """Compute position correction delta_p_i from constraint lambdas."""
    i = wp.tid()
    if i >= n_particles:
        return

    pi_pos = pos_pred[i]
    lam_i = lambdas[i]
    dp = wp.vec3(float(0.0), float(0.0), float(0.0))

    # W(corr_dq_dist) for artificial pressure
    dq2 = corr_dq_dist * corr_dq_dist
    w_dq = poly6_coeff * (h2 - dq2) * (h2 - dq2) * (h2 - dq2)
    if w_dq < 1.0e-30:
        w_dq = 1.0e-30

    query = wp.hash_grid_query(grid, pi_pos, h)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        if idx == i:
            continue
        if idx >= n_particles:
            continue

        r_vec = pi_pos - pos_pred[idx]
        r2 = wp.dot(r_vec, r_vec)

        if r2 < h2 and r2 > 1.0e-12:
            r = wp.sqrt(r2)

            # Artificial pressure (tensile instability correction)
            diff_poly = h2 - r2
            w_ij = poly6_coeff * diff_poly * diff_poly * diff_poly
            ratio = w_ij / w_dq
            # s_corr = -k * (W(r)/W(dq))^n
            s_corr = -corr_k * wp.pow(ratio, corr_n)

            # Spiky gradient
            diff_r = h - r
            grad_w = spiky_grad_coeff * diff_r * diff_r * (r_vec / r)

            dp = dp + (lam_i + lambdas[idx] + s_corr) * grad_w

    delta_pos[i] = dp / rest_density


@wp.kernel
def apply_delta_pos(
    pos_pred: wp.array(dtype=wp.vec3),
    delta_pos: wp.array(dtype=wp.vec3),
    n_particles: int,
):
    """Apply position correction: x* += delta_p."""
    i = wp.tid()
    if i >= n_particles:
        return
    pos_pred[i] = pos_pred[i] + delta_pos[i]


@wp.kernel
def update_velocity_and_position(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    pos_pred: wp.array(dtype=wp.vec3),
    dt_inv: float,
    n_particles: int,
):
    """Compute velocity from position delta and finalize position."""
    i = wp.tid()
    if i >= n_particles:
        return
    new_vel = (pos_pred[i] - pos[i]) * dt_inv
    vel[i] = new_vel
    pos[i] = pos_pred[i]


@wp.kernel
def apply_xsph_viscosity(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    vel_new: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    n_particles: int,
    h: float,
    h2: float,
    poly6_coeff: float,
    viscosity: float,
):
    """XSPH viscosity: v_i += c * sum_j (v_j - v_i) * W(r_ij)."""
    i = wp.tid()
    if i >= n_particles:
        return

    pi_pos = pos[i]
    vi = vel[i]
    dv = wp.vec3(float(0.0), float(0.0), float(0.0))

    query = wp.hash_grid_query(grid, pi_pos, h)
    idx = int(0)
    while wp.hash_grid_query_next(query, idx):
        if idx == i:
            continue
        if idx >= n_particles:
            continue

        r_vec = pi_pos - pos[idx]
        r2 = wp.dot(r_vec, r_vec)

        if r2 < h2:
            diff = h2 - r2
            w = poly6_coeff * diff * diff * diff
            dv = dv + (vel[idx] - vi) * w

    vel_new[i] = vi + viscosity * dv


@wp.kernel
def enforce_boundaries_and_solid(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    solid: wp.array(dtype=wp.int32, ndim=3),
    sdf: wp.array(dtype=wp.float32, ndim=3),
    sdf_nx: wp.array(dtype=wp.float32, ndim=3),
    sdf_ny: wp.array(dtype=wp.float32, ndim=3),
    sdf_nz: wp.array(dtype=wp.float32, ndim=3),
    n_grid: int,
    dx: float,
    lo: float,
    hi: float,
    floor_z: float,
    n_particles: int,
):
    """Clamp particles to domain and push out of solid obstacles."""
    i = wp.tid()
    if i >= n_particles:
        return

    p = pos[i]
    v = vel[i]

    # Domain boundaries with velocity zeroing
    eps = dx * 0.5

    if p[0] < lo + eps:
        p = wp.vec3(lo + eps, p[1], p[2])
        if v[0] < 0.0:
            v = wp.vec3(0.0, v[1], v[2])
    if p[0] > hi - eps:
        p = wp.vec3(hi - eps, p[1], p[2])
        if v[0] > 0.0:
            v = wp.vec3(0.0, v[1], v[2])
    if p[1] < lo + eps:
        p = wp.vec3(p[0], lo + eps, p[2])
        if v[1] < 0.0:
            v = wp.vec3(p[0], 0.0, v[2])
    if p[1] > hi - eps:
        p = wp.vec3(p[0], hi - eps, p[2])
        if v[1] > 0.0:
            v = wp.vec3(v[0], 0.0, v[2])
    if p[2] < floor_z + eps:
        p = wp.vec3(p[0], p[1], floor_z + eps)
        if v[2] < 0.0:
            v = wp.vec3(v[0], v[1], 0.0)
    if p[2] > hi - eps:
        p = wp.vec3(p[0], p[1], hi - eps)
        if v[2] > 0.0:
            v = wp.vec3(v[0], v[1], 0.0)

    # SDF obstacle push-out
    inv_dx = 1.0 / dx
    ci = wp.clamp(int(p[0] * inv_dx), 0, n_grid - 1)
    cj = wp.clamp(int(p[1] * inv_dx), 0, n_grid - 1)
    ck = wp.clamp(int(p[2] * inv_dx), 0, n_grid - 1)

    if sdf[ci, cj, ck] < 0.0:
        nx = sdf_nx[ci, cj, ck]
        ny = sdf_ny[ci, cj, ck]
        nz = sdf_nz[ci, cj, ck]
        push = -sdf[ci, cj, ck] + dx * 0.5
        p = p + wp.vec3(nx, ny, nz) * push
        # Zero velocity into the solid
        v_dot_n = v[0] * nx + v[1] * ny + v[2] * nz
        if v_dot_n < 0.0:
            v = v - wp.vec3(nx, ny, nz) * v_dot_n

    pos[i] = p
    vel[i] = v


@wp.kernel
def enforce_boundaries_and_solid_pred(
    pos_pred: wp.array(dtype=wp.vec3),
    solid: wp.array(dtype=wp.int32, ndim=3),
    sdf: wp.array(dtype=wp.float32, ndim=3),
    sdf_nx: wp.array(dtype=wp.float32, ndim=3),
    sdf_ny: wp.array(dtype=wp.float32, ndim=3),
    sdf_nz: wp.array(dtype=wp.float32, ndim=3),
    n_grid: int,
    dx: float,
    lo: float,
    hi: float,
    floor_z: float,
    n_particles: int,
):
    """Clamp predicted positions to domain and push out of solid obstacles."""
    i = wp.tid()
    if i >= n_particles:
        return

    p = pos_pred[i]
    eps = dx * 0.5

    if p[0] < lo + eps:
        p = wp.vec3(lo + eps, p[1], p[2])
    if p[0] > hi - eps:
        p = wp.vec3(hi - eps, p[1], p[2])
    if p[1] < lo + eps:
        p = wp.vec3(p[0], lo + eps, p[2])
    if p[1] > hi - eps:
        p = wp.vec3(p[0], hi - eps, p[2])
    if p[2] < floor_z + eps:
        p = wp.vec3(p[0], p[1], floor_z + eps)
    if p[2] > hi - eps:
        p = wp.vec3(p[0], p[1], hi - eps)

    # SDF obstacle push-out
    inv_dx = 1.0 / dx
    ci = wp.clamp(int(p[0] * inv_dx), 0, n_grid - 1)
    cj = wp.clamp(int(p[1] * inv_dx), 0, n_grid - 1)
    ck = wp.clamp(int(p[2] * inv_dx), 0, n_grid - 1)

    if sdf[ci, cj, ck] < 0.0:
        nx = sdf_nx[ci, cj, ck]
        ny = sdf_ny[ci, cj, ck]
        nz = sdf_nz[ci, cj, ck]
        push = -sdf[ci, cj, ck] + dx * 0.5
        p = p + wp.vec3(nx, ny, nz) * push

    pos_pred[i] = p


# ============================================================================
# PBF Solver class
# ============================================================================
class PBFSolver:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.n_grid = N_GRID
        self.dx = 1.0 / N_GRID
        self.max_particles = MAX_PARTICLES
        self.n_particles = 0

        # Allocate arrays on GPU
        self.pos = wp.zeros(MAX_PARTICLES, dtype=wp.vec3, device=device)
        self.vel = wp.zeros(MAX_PARTICLES, dtype=wp.vec3, device=device)
        self.pos_pred = wp.zeros(MAX_PARTICLES, dtype=wp.vec3, device=device)
        self.lambdas = wp.zeros(MAX_PARTICLES, dtype=float, device=device)
        self.delta_pos = wp.zeros(MAX_PARTICLES, dtype=wp.vec3, device=device)
        self.vel_new = wp.zeros(MAX_PARTICLES, dtype=wp.vec3, device=device)

        # Hash grid for neighbor search
        self.hash_grid = wp.HashGrid(N_GRID, N_GRID, N_GRID, device=device)

        # Solid obstacle arrays (initialized as empty)
        empty_3d = np.zeros((N_GRID, N_GRID, N_GRID), dtype=np.int32)
        empty_3d_f = np.zeros((N_GRID, N_GRID, N_GRID), dtype=np.float32)
        self.solid = wp.array(empty_3d, dtype=wp.int32, device=device)
        self.sdf = wp.array(empty_3d_f, dtype=wp.float32, device=device)
        self.sdf_nx = wp.array(empty_3d_f, dtype=wp.float32, device=device)
        self.sdf_ny = wp.array(empty_3d_f, dtype=wp.float32, device=device)
        self.sdf_nz = wp.array(empty_3d_f, dtype=wp.float32, device=device)

        self._solid_np = empty_3d

        # Floor Z (set after loading building)
        self.floor_z = BOUND_CELLS * self.dx

        # Kernel constants
        self.h = H
        self.h2 = H2
        self.poly6_coeff = POLY6_COEFF
        self.spiky_grad_coeff = SPIKY_GRAD_COEFF
        self.rest_density = REST_DENSITY
        self.epsilon_lambda = EPSILON_LAMBDA
        self.corr_k = CORR_K
        self.corr_n = CORR_N
        self.corr_dq_dist = CORR_DQ * H  # absolute distance for artificial pressure

        print(f"PBF Solver initialized on {device}")
        print(f"  Grid: {N_GRID}^3, dx={self.dx:.6f}")
        print(f"  Kernel radius h={H:.6f}")
        print(f"  Poly6 coeff={POLY6_COEFF:.2e}, Spiky grad coeff={SPIKY_GRAD_COEFF:.2e}")
        print(f"  Max particles: {MAX_PARTICLES:,}")

    def load_solid(self, solid_path, meta_path=None):
        """Load precomputed solid voxel grid and compute SDF via EDT."""
        from scipy.ndimage import distance_transform_edt

        solid_np = np.load(solid_path).astype(np.int32)
        n = solid_np.shape[0]
        assert solid_np.shape == (n, n, n), f"Expected cubic grid, got {solid_np.shape}"

        if n != self.n_grid:
            print(f"  WARNING: solid grid {n}^3 != solver grid {self.n_grid}^3")
            print(f"  Resampling solid grid to {self.n_grid}^3...")
            from scipy.ndimage import zoom
            scale = self.n_grid / n
            solid_np = zoom(solid_np.astype(np.float32), scale, order=0)
            solid_np = (solid_np > 0.5).astype(np.int32)
            n = self.n_grid

        self._solid_np = solid_np

        # Compute SDF from solid voxels via EDT
        thick = solid_np.astype(bool)
        dx = self.dx
        dist_in = distance_transform_edt(thick).astype(np.float32) * dx
        dist_out = distance_transform_edt(~thick).astype(np.float32) * dx
        sdf_np = np.where(thick, -dist_in, dist_out).astype(np.float32)

        # Compute normals from SDF gradient
        gx, gy, gz = np.gradient(sdf_np, dx)
        norms = np.sqrt(gx**2 + gy**2 + gz**2)
        norms = np.maximum(norms, 1e-8)
        sdf_nx_np = (gx / norms).astype(np.float32)
        sdf_ny_np = (gy / norms).astype(np.float32)
        sdf_nz_np = (gz / norms).astype(np.float32)

        # Upload to GPU
        self.solid = wp.array(solid_np, dtype=wp.int32, device=self.device)
        self.sdf = wp.array(sdf_np, dtype=wp.float32, device=self.device)
        self.sdf_nx = wp.array(sdf_nx_np, dtype=wp.float32, device=self.device)
        self.sdf_ny = wp.array(sdf_ny_np, dtype=wp.float32, device=self.device)
        self.sdf_nz = wp.array(sdf_nz_np, dtype=wp.float32, device=self.device)

        n_solid = int(solid_np.sum())
        print(f"  Loaded solid: {n_solid:,} voxels ({n}^3 grid)")
        print(f"  SDF range: [{sdf_np.min():.4f}, {sdf_np.max():.4f}]")

        # Load meta for building bounds if available
        if meta_path is not None and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "mesh_bounds" in meta:
                self.mesh_bounds = meta["mesh_bounds"]
                self.floor_z = self.mesh_bounds[0][2]
                building_height = self.mesh_bounds[1][2] - self.mesh_bounds[0][2]
                print(f"  Building bounds: Z=[{self.floor_z:.4f}, {self.mesh_bounds[1][2]:.4f}]")
                print(f"  Building height: {building_height:.4f}")
                return meta
        return None

    def inject_particles(self, z_lo, z_hi, count, xy_bounds, velocity):
        """Inject particles in the given region, avoiding solid voxels."""
        cur_n = self.n_particles
        room = self.max_particles - cur_n
        if room <= 0:
            return 0
        target = min(count, room)

        x_lo, x_hi, y_lo, y_hi = xy_bounds
        solid_np = self._solid_np
        n_grid = self.n_grid
        rng = np.random.default_rng()

        # Probe free-space fraction
        probe = 2000
        probe_pos = np.column_stack([
            rng.uniform(x_lo, x_hi, probe),
            rng.uniform(y_lo, y_hi, probe),
            rng.uniform(z_lo, z_hi, probe),
        ]).astype(np.float32)
        gi = np.clip((probe_pos * n_grid).astype(int), 0, n_grid - 1)
        free_frac = float((solid_np[gi[:, 0], gi[:, 1], gi[:, 2]] == 0).mean())
        if free_frac < 0.01:
            return 0

        oversample = min(int(target / free_frac * 1.3) + 200, room * 3)
        pos_candidates = np.column_stack([
            rng.uniform(x_lo, x_hi, oversample),
            rng.uniform(y_lo, y_hi, oversample),
            rng.uniform(z_lo, z_hi, oversample),
        ]).astype(np.float32)
        gi = np.clip((pos_candidates * n_grid).astype(int), 0, n_grid - 1)
        mask = solid_np[gi[:, 0], gi[:, 1], gi[:, 2]] == 0
        pos_candidates = pos_candidates[mask]
        if len(pos_candidates) > target:
            pos_candidates = pos_candidates[:target]
        actual = len(pos_candidates)
        if actual == 0:
            return 0

        # Create velocity array for new particles
        vel_inject = np.tile(np.array(velocity, dtype=np.float32), (actual, 1))

        # Upload new particles to GPU via staging buffers
        # Create CPU-side warp arrays for the new particle slice
        pos_stage = wp.array(pos_candidates, dtype=wp.vec3, device="cpu")
        vel_stage = wp.array(vel_inject, dtype=wp.vec3, device="cpu")

        # Copy into the correct offset of the GPU arrays
        wp.copy(dest=self.pos, src=pos_stage,
                dest_offset=cur_n, src_offset=0, count=actual)
        wp.copy(dest=self.vel, src=vel_stage,
                dest_offset=cur_n, src_offset=0, count=actual)

        self.n_particles = cur_n + actual
        return actual

    def substep(self, dt):
        """Run one PBF substep."""
        n = self.n_particles
        if n == 0:
            return
        dim = self.max_particles

        # 1. Predict positions: apply gravity, x* = x + dt*v
        wp.launch(predict_positions, dim=dim,
                  inputs=[self.pos, self.vel, self.pos_pred, GRAVITY, dt, n],
                  device=self.device)

        # 2. Enforce boundaries on predicted positions
        wp.launch(enforce_boundaries_and_solid_pred, dim=dim,
                  inputs=[self.pos_pred, self.solid, self.sdf,
                          self.sdf_nx, self.sdf_ny, self.sdf_nz,
                          self.n_grid, self.dx,
                          DOMAIN_LO, DOMAIN_HI, self.floor_z, n],
                  device=self.device)

        # 3. Build hash grid from predicted positions
        self.hash_grid.build(self.pos_pred, n)

        # 4. Iterative constraint solving
        grid_id = self.hash_grid.id
        for _ in range(SOLVER_ITERATIONS):
            # Compute lambda_i
            wp.launch(compute_lambda, dim=dim,
                      inputs=[self.pos_pred, self.lambdas, grid_id,
                              n, self.h, self.h2,
                              self.poly6_coeff, self.spiky_grad_coeff,
                              self.rest_density, self.epsilon_lambda],
                      device=self.device)

            # Compute delta position
            wp.launch(compute_delta_pos, dim=dim,
                      inputs=[self.pos_pred, self.lambdas, self.delta_pos,
                              grid_id, n, self.h, self.h2,
                              self.poly6_coeff, self.spiky_grad_coeff,
                              self.rest_density,
                              self.corr_k, self.corr_n, self.corr_dq_dist],
                      device=self.device)

            # Apply delta position
            wp.launch(apply_delta_pos, dim=dim,
                      inputs=[self.pos_pred, self.delta_pos, n],
                      device=self.device)

            # Re-enforce boundaries after correction
            wp.launch(enforce_boundaries_and_solid_pred, dim=dim,
                      inputs=[self.pos_pred, self.solid, self.sdf,
                              self.sdf_nx, self.sdf_ny, self.sdf_nz,
                              self.n_grid, self.dx,
                              DOMAIN_LO, DOMAIN_HI, self.floor_z, n],
                      device=self.device)

        # 5. Update velocity from position change: v = (x* - x) / dt
        dt_inv = 1.0 / dt if dt > 0 else 0.0
        wp.launch(update_velocity_and_position, dim=dim,
                  inputs=[self.pos, self.vel, self.pos_pred, dt_inv, n],
                  device=self.device)

        # 6. Rebuild hash grid on final positions for viscosity
        self.hash_grid.build(self.pos, n)

        # 7. XSPH viscosity
        wp.launch(apply_xsph_viscosity, dim=dim,
                  inputs=[self.pos, self.vel, self.vel_new, self.hash_grid.id,
                          n, self.h, self.h2, self.poly6_coeff, VISCOSITY],
                  device=self.device)
        # Swap velocity buffers
        self.vel, self.vel_new = self.vel_new, self.vel

        # 8. Final boundary enforcement
        wp.launch(enforce_boundaries_and_solid, dim=dim,
                  inputs=[self.pos, self.vel, self.solid, self.sdf,
                          self.sdf_nx, self.sdf_ny, self.sdf_nz,
                          self.n_grid, self.dx,
                          DOMAIN_LO, DOMAIN_HI, self.floor_z, n],
                  device=self.device)

    def step(self, frame_dt=None, n_substeps=None):
        """Run one frame (multiple substeps)."""
        if frame_dt is None:
            frame_dt = DT
        if n_substeps is None:
            n_substeps = N_SUBSTEPS
        dt_sub = frame_dt / n_substeps
        for _ in range(n_substeps):
            self.substep(dt_sub)

    def get_positions_numpy(self):
        """Return positions as numpy (N, 3) float32 array."""
        n = self.n_particles
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.pos.numpy()[:n].copy()

    def get_velocities_numpy(self):
        """Return velocities as numpy (N, 3) float32 array."""
        n = self.n_particles
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.vel.numpy()[:n].copy()

    def max_velocity(self):
        """Return max particle speed (on CPU for simplicity)."""
        n = self.n_particles
        if n == 0:
            return 0.0
        v_np = self.vel.numpy()[:n]
        speeds = np.linalg.norm(v_np, axis=1)
        return float(speeds.max())

    def get_water_level(self, percentile=90):
        """Return water level at given percentile."""
        n = self.n_particles
        if n == 0:
            return self.floor_z
        z = self.pos.numpy()[:n, 2]
        return float(np.percentile(z, percentile))

    def export_npz(self, path):
        """Export NPZ with x, v, foam arrays (compatible with FLIP solver)."""
        x = self.get_positions_numpy()
        v = self.get_velocities_numpy()
        foam = np.zeros(len(x), dtype=np.float32)
        np.savez_compressed(path, x=x, v=v, foam=foam)

    def export_ply(self, path):
        """Export binary PLY (float32 xyz) for SplashSurf."""
        x = self.get_positions_numpy()
        n = len(x)
        with open(path, "wb") as f:
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"end_header\n"
            )
            f.write(header.encode("ascii"))
            f.write(x.astype(np.float32).tobytes())


# ============================================================================
# Main simulation loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="PBF flood simulation (Warp)")
    parser.add_argument("--solid", type=str, default=None,
                        help="Path to solid.npy (building voxels)")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to meta.json (building bounds)")
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--export-dir", type=str, default=EXPORT_DIR)
    args = parser.parse_args()

    # Initialize Warp
    wp.init()

    # Create solver
    solver = PBFSolver(device=args.device)

    # Load solid obstacle if provided
    meta = None
    if args.solid is not None:
        meta = solver.load_solid(args.solid, args.meta)
    else:
        # Try default locations
        default_solid = os.path.join(SCRIPT_DIR, "export", "flood", "solid.npy")
        default_meta = os.path.join(SCRIPT_DIR, "export", "flood", "meta.json")
        if os.path.exists(default_solid):
            print(f"\nLoading solid from {default_solid}")
            meta = solver.load_solid(default_solid, default_meta)
        else:
            print("\nNo solid obstacle found — running with empty domain.")

    # Determine building geometry for inflow
    floor_z = solver.floor_z
    if hasattr(solver, 'mesh_bounds'):
        building_base_z = solver.mesh_bounds[0][2]
        building_top_z = solver.mesh_bounds[1][2]
        building_height = building_top_z - building_base_z
    else:
        building_base_z = floor_z
        building_height = 0.3
        building_top_z = building_base_z + building_height

    max_inject_z = building_base_z + building_height * MAX_Z_FRAC

    # Inflow geometry: narrow slab at left boundary
    bnd = BOUND_CELLS * solver.dx + solver.dx
    inflow_x_lo = bnd
    inflow_x_hi = bnd + INFLOW_WIDTH
    inflow_y_lo = INFLOW_Y_LO
    inflow_y_hi = INFLOW_Y_HI
    inflow_vel = (INFLOW_VELOCITY[0], INFLOW_VELOCITY[1], INFLOW_VELOCITY[2])

    # Export setup
    export_dir = args.export_dir
    os.makedirs(export_dir, exist_ok=True)

    # Save meta
    sim_meta = dict(
        solver="PBF-Warp",
        n_grid=N_GRID,
        kernel_radius=KERNEL_RADIUS,
        rest_density=REST_DENSITY,
        solver_iterations=SOLVER_ITERATIONS,
        viscosity=VISCOSITY,
        gravity=[GRAVITY[0], GRAVITY[1], GRAVITY[2]],
        dt=DT,
        n_substeps=N_SUBSTEPS,
        n_frames=args.frames,
        max_particles=MAX_PARTICLES,
        inject_rate=INJECT_RATE,
        inflow_velocity=[INFLOW_VELOCITY[0], INFLOW_VELOCITY[1], INFLOW_VELOCITY[2]],
        floor_z=floor_z,
        building_base_z=building_base_z,
        building_height=building_height,
    )
    if meta is not None:
        sim_meta["building_meta"] = meta
    with open(os.path.join(export_dir, "meta.json"), "w") as f:
        json.dump(sim_meta, f, indent=2)

    # If solid exists, copy to export dir
    if solver._solid_np is not None and solver._solid_np.sum() > 0:
        np.save(os.path.join(export_dir, "solid.npy"), solver._solid_np)

    # Print summary
    print("\n" + "=" * 60)
    print("PBF Flood Simulation (Warp)")
    print(f"  Grid: {N_GRID}^3, h={KERNEL_RADIUS}")
    print(f"  dt={DT}, substeps={N_SUBSTEPS}, sub-dt={DT/N_SUBSTEPS:.5f}")
    print(f"  Solver iterations: {SOLVER_ITERATIONS}")
    print(f"  Inject rate: {INJECT_RATE} p/frame")
    print(f"  Inflow X=[{inflow_x_lo:.4f}, {inflow_x_hi:.4f}], "
          f"Y=[{inflow_y_lo}, {inflow_y_hi}]")
    print(f"  Floor Z: {floor_z:.4f}, max flood Z: {max_inject_z:.4f}")
    print(f"  Frames: {args.frames}")
    print(f"  Export: {export_dir}")
    print("=" * 60)

    # Rising water level
    target_z_start = building_base_z
    target_z_end = max_inject_z
    target_z_per_frame = (target_z_end - target_z_start) / args.frames

    t0 = time.time()

    for frame in range(args.frames):
        # Rising water level
        target_z = target_z_start + target_z_per_frame * (frame + 1)
        z_lo = building_base_z
        z_hi = target_z

        # Inject particles
        rate = INJECT_RATE * 2 if frame < 5 else INJECT_RATE  # prefill boost
        added = solver.inject_particles(
            z_lo, z_hi, rate,
            xy_bounds=(inflow_x_lo, inflow_x_hi, inflow_y_lo, inflow_y_hi),
            velocity=inflow_vel,
        )

        # Step simulation
        solver.step(frame_dt=DT, n_substeps=N_SUBSTEPS)

        # Export
        npz_path = os.path.join(export_dir, f"water_{frame:06d}.npz")
        ply_path = os.path.join(export_dir, f"water_{frame:06d}.ply")
        solver.export_npz(npz_path)
        if solver.n_particles > 0:
            solver.export_ply(ply_path)

        # Progress reporting
        if frame % 10 == 0 or frame < 5:
            elapsed = time.time() - t0
            fps_sim = (frame + 1) / elapsed if elapsed > 0 else 0
            n_p = solver.n_particles
            v_max = solver.max_velocity()
            actual_z = solver.get_water_level() if n_p > 0 else floor_z
            print(f"  [{frame:4d}/{args.frames}] n={n_p:,} (+{added}) "
                  f"z={actual_z:.4f}/{target_z:.4f} "
                  f"v_max={v_max:.2f} | "
                  f"{elapsed:.0f}s ({fps_sim:.1f} fr/s)")

    elapsed = time.time() - t0
    n_p = solver.n_particles
    print(f"\nDone! {args.frames} frames, {n_p:,} particles")
    print(f"Total time: {elapsed:.0f}s ({elapsed/args.frames:.1f}s/frame)")
    print(f"Export: {export_dir}")


if __name__ == "__main__":
    main()
