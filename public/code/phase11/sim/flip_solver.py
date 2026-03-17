"""Phase 11 — FLIP (Fluid Implicit Particle) solver on MAC staggered grid.

SOTA water simulation pipeline:
- MAC staggered grid: u/v/w at cell faces
- MGPCG pressure Poisson solver with multigrid preconditioner
- FLIP/PIC blend (97% FLIP / 3% PIC) for low dissipation
- Trilinear P2G/G2P on staggered grid
- Velocity extrapolation for surface particles
- RK2 midpoint advection
- SDF-based solid obstacles
- Per-particle foam tagging

Reference: Bridson "Fluid Simulation for Computer Graphics" (2015),
           Zhu & Bridson 2005 (FLIP for fluids)
"""
import taichi as ti
import numpy as np
from mgpcg import MGPCGSolver, FLUID, AIR, SOLID


@ti.data_oriented
class FLIPSolver:
    def __init__(self, n_grid=192, max_p=10_000_000,
                 gravity=(0, 0, -9.81), dt=5e-3,
                 flip_ratio=0.97, pcg_max_iter=500, pcg_tol=1e-6,
                 foam_v_thresh=1.5, foam_decay=0.92,
                 floor_z=None):
        n = n_grid
        self.n = n
        self.dx = 1.0 / n
        self.inv_dx = float(n)
        self.dt = dt
        self.max_p = max_p
        self.flip_ratio = flip_ratio
        self.pcg_max_iter = pcg_max_iter
        self.pcg_tol = pcg_tol
        self.bound = 3
        self.floor_cell = int(floor_z * n) if floor_z is not None else self.bound
        self.grav = ti.Vector(list(gravity))
        self.foam_v_thresh = foam_v_thresh
        self.foam_decay = foam_decay

        # --- MAC staggered velocity fields ---
        # u at x-faces: shape (n+1, n, n)
        # v at y-faces: shape (n, n+1, n)
        # w at z-faces: shape (n, n, n+1)
        self.u = ti.field(float, shape=(n + 1, n, n))
        self.v = ti.field(float, shape=(n, n + 1, n))
        self.w = ti.field(float, shape=(n, n, n + 1))
        self.u_old = ti.field(float, shape=(n + 1, n, n))
        self.v_old = ti.field(float, shape=(n, n + 1, n))
        self.w_old = ti.field(float, shape=(n, n, n + 1))

        # P2G weight accumulators
        self.u_weight = ti.field(float, shape=(n + 1, n, n))
        self.v_weight = ti.field(float, shape=(n, n + 1, n))
        self.w_weight = ti.field(float, shape=(n, n, n + 1))

        # Valid flags for velocity extrapolation
        self.u_valid = ti.field(int, shape=(n + 1, n, n))
        self.v_valid = ti.field(int, shape=(n, n + 1, n))
        self.w_valid = ti.field(int, shape=(n, n, n + 1))
        self.u_valid_tmp = ti.field(int, shape=(n + 1, n, n))
        self.v_valid_tmp = ti.field(int, shape=(n, n + 1, n))
        self.w_valid_tmp = ti.field(int, shape=(n, n, n + 1))

        # Cell classification
        self.cell_type = ti.field(int, shape=(n, n, n))

        # SDF obstacle
        self.sdf = ti.field(float, shape=(n, n, n))
        self.sdf_n = ti.Vector.field(3, float, shape=(n, n, n))
        self.solid = ti.field(ti.i32, shape=(n, n, n))

        # Particles
        self.x = ti.Vector.field(3, float, shape=max_p)
        self.vel = ti.Vector.field(3, float, shape=max_p)
        self.foam = ti.field(float, shape=max_p)
        self.num_p = ti.field(int, shape=())

        # Pressure solver
        self.pressure = MGPCGSolver(n)

        # Scratch fields
        self.p_count = ti.field(int, shape=(n, n, n))
        self.v_max_field = ti.field(float, shape=())

    # ==================================================================
    # SDF loading (reused from Phase 11 MPM solver)
    # ==================================================================
    def load_solid_from_obj(self, obj_path, scale, center_xy=(0.5, 0.5)):
        """Load OBJ mesh, compute SDF via EDT. Returns (solid_np, bounds, transform)."""
        import trimesh
        from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt

        print(f"  Loading {obj_path}...")
        result = trimesh.load(obj_path, force='mesh', process=True)
        if isinstance(result, trimesh.Scene):
            meshes = [g for g in result.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = result
        print(f"  Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

        native_centroid = mesh.centroid.copy()
        native_max_ext = float(mesh.extents.max())
        native_bounds = mesh.bounds.copy()

        mesh.apply_translation(-mesh.centroid)
        mesh.apply_scale(scale / mesh.extents.max())

        ground_z = self.bound * self.dx
        half_z = mesh.extents[2] / 2
        placement = np.array([center_xy[0], center_xy[1], ground_z + half_z])
        mesh.apply_translation(placement)

        inv_scale = native_max_ext / scale
        inv_offset = native_centroid - placement * inv_scale

        print(f"  Placed at {placement.round(4)}, scale={scale:.3f}")
        print(f"  Bounds: {mesh.bounds[0].round(4)} -> {mesh.bounds[1].round(4)}")

        # SDF via voxelization + EDT
        print("  Computing SDF via EDT...")
        dx, n = self.dx, self.n
        vox = mesh.voxelized(pitch=dx)
        pts = vox.points
        gi = np.clip(np.floor(pts / dx).astype(int), 0, n - 1)

        surface_np = np.zeros((n, n, n), dtype=bool)
        surface_np[gi[:, 0], gi[:, 1], gi[:, 2]] = True
        # Fill interior so building is a solid volume, not a thin shell
        filled = binary_fill_holes(surface_np)
        # Dilate to thicken walls and seal small gaps
        thick = binary_dilation(filled, iterations=2)

        dist_in = distance_transform_edt(thick).astype(np.float32) * dx
        dist_out = distance_transform_edt(~thick).astype(np.float32) * dx
        sdf_np = np.where(thick, -dist_in, dist_out).astype(np.float32)

        gx, gy, gz = np.gradient(sdf_np, dx)
        normal_np = np.zeros((n, n, n, 3), dtype=np.float32)
        normal_np[..., 0] = gx
        normal_np[..., 1] = gy
        normal_np[..., 2] = gz
        norms = np.linalg.norm(normal_np, axis=-1, keepdims=True)
        normal_np /= np.maximum(norms, 1e-8)

        self.sdf.from_numpy(sdf_np)
        self.sdf_n.from_numpy(normal_np)
        solid_np = thick.astype(np.int32)
        self.solid.from_numpy(solid_np)
        self._solid_np = solid_np

        n_surface = int(surface_np.sum())
        n_thick = int(thick.sum())
        print(f"    Surface: {n_surface:,} voxels, after dilation: {n_thick:,}")
        print(f"    SDF range: [{sdf_np.min():.4f}, {sdf_np.max():.4f}]")

        transform = dict(
            inv_scale=float(inv_scale),
            inv_offset=inv_offset.tolist(),
            native_bounds=native_bounds.tolist(),
        )
        return solid_np, mesh.bounds, transform

    # ==================================================================
    # Cell classification
    # ==================================================================
    @ti.kernel
    def _count_particles(self):
        for i, j, k in self.p_count:
            self.p_count[i, j, k] = 0
        for p in range(self.num_p[None]):
            pos = self.x[p]
            ci = int(ti.floor(pos[0] * self.inv_dx))
            cj = int(ti.floor(pos[1] * self.inv_dx))
            ck = int(ti.floor(pos[2] * self.inv_dx))
            ci = ti.max(0, ti.min(self.n - 1, ci))
            cj = ti.max(0, ti.min(self.n - 1, cj))
            ck = ti.max(0, ti.min(self.n - 1, ck))
            ti.atomic_add(self.p_count[ci, cj, ck], 1)

    @ti.kernel
    def _classify_cells(self):
        for i, j, k in self.cell_type:
            # Default: AIR
            ct = AIR
            # Particles present → FLUID
            if self.p_count[i, j, k] > 0:
                ct = FLUID
            # SDF obstacle → SOLID (overrides FLUID)
            if self.solid[i, j, k] == 1:
                ct = SOLID
            # Domain walls (sides + floor) → SOLID
            if i < self.bound or i >= self.n - self.bound:
                ct = SOLID
            if j < self.bound or j >= self.n - self.bound:
                ct = SOLID
            if k < self.floor_cell:
                ct = SOLID
            # Top is OPEN (remains AIR/FLUID)
            self.cell_type[i, j, k] = ct

    def mark_cells(self):
        self._count_particles()
        self._classify_cells()

    # ==================================================================
    # P2G: splat particle velocities to MAC faces (trilinear)
    # ==================================================================
    @ti.kernel
    def _clear_grid(self):
        for i, j, k in ti.ndrange(self.n + 1, self.n, self.n):
            self.u[i, j, k] = 0.0
            self.u_weight[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.n, self.n + 1, self.n):
            self.v[i, j, k] = 0.0
            self.v_weight[i, j, k] = 0.0
        for i, j, k in ti.ndrange(self.n, self.n, self.n + 1):
            self.w[i, j, k] = 0.0
            self.w_weight[i, j, k] = 0.0

    @ti.kernel
    def _p2g(self):
        inv_dx = self.inv_dx
        n = self.n
        for p in range(self.num_p[None]):
            pos = self.x[p]
            vp = self.vel[p]

            # --- u: face at (i, j+0.5, k+0.5) ---
            gx = pos[0] * inv_dx
            gy = pos[1] * inv_dx - 0.5
            gz = pos[2] * inv_dx - 0.5
            bi = int(ti.floor(gx)); bj = int(ti.floor(gy)); bk = int(ti.floor(gz))
            fx = gx - bi; fy = gy - bj; fz = gz - bk
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ii, jj, kk = bi + di, bj + dj, bk + dk
                if 0 <= ii < n + 1 and 0 <= jj < n and 0 <= kk < n:
                    w = (1.0 - ti.abs(fx - di)) * (1.0 - ti.abs(fy - dj)) * (1.0 - ti.abs(fz - dk))
                    ti.atomic_add(self.u[ii, jj, kk], w * vp[0])
                    ti.atomic_add(self.u_weight[ii, jj, kk], w)

            # --- v: face at (i+0.5, j, k+0.5) ---
            gx = pos[0] * inv_dx - 0.5
            gy = pos[1] * inv_dx
            gz = pos[2] * inv_dx - 0.5
            bi = int(ti.floor(gx)); bj = int(ti.floor(gy)); bk = int(ti.floor(gz))
            fx = gx - bi; fy = gy - bj; fz = gz - bk
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ii, jj, kk = bi + di, bj + dj, bk + dk
                if 0 <= ii < n and 0 <= jj < n + 1 and 0 <= kk < n:
                    w = (1.0 - ti.abs(fx - di)) * (1.0 - ti.abs(fy - dj)) * (1.0 - ti.abs(fz - dk))
                    ti.atomic_add(self.v[ii, jj, kk], w * vp[1])
                    ti.atomic_add(self.v_weight[ii, jj, kk], w)

            # --- w: face at (i+0.5, j+0.5, k) ---
            gx = pos[0] * inv_dx - 0.5
            gy = pos[1] * inv_dx - 0.5
            gz = pos[2] * inv_dx
            bi = int(ti.floor(gx)); bj = int(ti.floor(gy)); bk = int(ti.floor(gz))
            fx = gx - bi; fy = gy - bj; fz = gz - bk
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                ii, jj, kk = bi + di, bj + dj, bk + dk
                if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n + 1:
                    w = (1.0 - ti.abs(fx - di)) * (1.0 - ti.abs(fy - dj)) * (1.0 - ti.abs(fz - dk))
                    ti.atomic_add(self.w[ii, jj, kk], w * vp[2])
                    ti.atomic_add(self.w_weight[ii, jj, kk], w)

    @ti.kernel
    def _normalize_grid(self):
        for i, j, k in ti.ndrange(self.n + 1, self.n, self.n):
            if self.u_weight[i, j, k] > 0.0:
                self.u[i, j, k] /= self.u_weight[i, j, k]
                self.u_valid[i, j, k] = 1
            else:
                self.u[i, j, k] = 0.0
                self.u_valid[i, j, k] = 0
        for i, j, k in ti.ndrange(self.n, self.n + 1, self.n):
            if self.v_weight[i, j, k] > 0.0:
                self.v[i, j, k] /= self.v_weight[i, j, k]
                self.v_valid[i, j, k] = 1
            else:
                self.v[i, j, k] = 0.0
                self.v_valid[i, j, k] = 0
        for i, j, k in ti.ndrange(self.n, self.n, self.n + 1):
            if self.w_weight[i, j, k] > 0.0:
                self.w[i, j, k] /= self.w_weight[i, j, k]
                self.w_valid[i, j, k] = 1
            else:
                self.w[i, j, k] = 0.0
                self.w_valid[i, j, k] = 0

    def p2g(self):
        self._clear_grid()
        self._p2g()
        self._normalize_grid()

    # ==================================================================
    # Save velocity (for FLIP delta computation)
    # ==================================================================
    @ti.kernel
    def _save_velocity(self):
        for i, j, k in ti.ndrange(self.n + 1, self.n, self.n):
            self.u_old[i, j, k] = self.u[i, j, k]
        for i, j, k in ti.ndrange(self.n, self.n + 1, self.n):
            self.v_old[i, j, k] = self.v[i, j, k]
        for i, j, k in ti.ndrange(self.n, self.n, self.n + 1):
            self.w_old[i, j, k] = self.w[i, j, k]

    # ==================================================================
    # Gravity (applied to w-faces only)
    # ==================================================================
    @ti.kernel
    def _add_gravity(self, dt: float):
        for i, j, k in ti.ndrange(self.n, self.n, self.n + 1):
            self.w[i, j, k] += dt * self.grav[2]

    # ==================================================================
    # Boundary enforcement: zero velocity at solid faces
    # ==================================================================
    @ti.kernel
    def _enforce_boundary(self):
        n = self.n
        # u-faces: between cells (i-1,j,k) and (i,j,k)
        for i, j, k in ti.ndrange(n + 1, n, n):
            is_solid = 0
            if i == 0 or i == n:
                is_solid = 1
            if i > 0 and self.cell_type[i - 1, j, k] == SOLID:
                is_solid = 1
            if i < n and self.cell_type[i, j, k] == SOLID:
                is_solid = 1
            if is_solid:
                self.u[i, j, k] = 0.0

        # v-faces: between cells (i,j-1,k) and (i,j,k)
        for i, j, k in ti.ndrange(n, n + 1, n):
            is_solid = 0
            if j == 0 or j == n:
                is_solid = 1
            if j > 0 and self.cell_type[i, j - 1, k] == SOLID:
                is_solid = 1
            if j < n and self.cell_type[i, j, k] == SOLID:
                is_solid = 1
            if is_solid:
                self.v[i, j, k] = 0.0

        # w-faces: between cells (i,j,k-1) and (i,j,k)
        for i, j, k in ti.ndrange(n, n, n + 1):
            is_solid = 0
            if k == 0 or k == n:
                is_solid = 1
            if k > 0 and self.cell_type[i, j, k - 1] == SOLID:
                is_solid = 1
            if k < n and self.cell_type[i, j, k] == SOLID:
                is_solid = 1
            if is_solid:
                self.w[i, j, k] = 0.0

    # ==================================================================
    # Velocity extrapolation into AIR cells (iterative averaging)
    # ==================================================================
    @ti.kernel
    def _extrapolate_u(self):
        n = self.n
        for i, j, k in ti.ndrange(n + 1, n, n):
            self.u_valid_tmp[i, j, k] = self.u_valid[i, j, k]
        for i, j, k in ti.ndrange(n + 1, n, n):
            if self.u_valid_tmp[i, j, k] == 0:
                s = 0.0; cnt = 0
                if i > 0     and self.u_valid_tmp[i - 1, j, k] == 1: s += self.u[i - 1, j, k]; cnt += 1
                if i < n     and self.u_valid_tmp[i + 1, j, k] == 1: s += self.u[i + 1, j, k]; cnt += 1
                if j > 0     and self.u_valid_tmp[i, j - 1, k] == 1: s += self.u[i, j - 1, k]; cnt += 1
                if j < n - 1 and self.u_valid_tmp[i, j + 1, k] == 1: s += self.u[i, j + 1, k]; cnt += 1
                if k > 0     and self.u_valid_tmp[i, j, k - 1] == 1: s += self.u[i, j, k - 1]; cnt += 1
                if k < n - 1 and self.u_valid_tmp[i, j, k + 1] == 1: s += self.u[i, j, k + 1]; cnt += 1
                if cnt > 0:
                    self.u[i, j, k] = s / cnt
                    self.u_valid[i, j, k] = 1

    @ti.kernel
    def _extrapolate_v(self):
        n = self.n
        for i, j, k in ti.ndrange(n, n + 1, n):
            self.v_valid_tmp[i, j, k] = self.v_valid[i, j, k]
        for i, j, k in ti.ndrange(n, n + 1, n):
            if self.v_valid_tmp[i, j, k] == 0:
                s = 0.0; cnt = 0
                if i > 0     and self.v_valid_tmp[i - 1, j, k] == 1: s += self.v[i - 1, j, k]; cnt += 1
                if i < n - 1 and self.v_valid_tmp[i + 1, j, k] == 1: s += self.v[i + 1, j, k]; cnt += 1
                if j > 0     and self.v_valid_tmp[i, j - 1, k] == 1: s += self.v[i, j - 1, k]; cnt += 1
                if j < n     and self.v_valid_tmp[i, j + 1, k] == 1: s += self.v[i, j + 1, k]; cnt += 1
                if k > 0     and self.v_valid_tmp[i, j, k - 1] == 1: s += self.v[i, j, k - 1]; cnt += 1
                if k < n - 1 and self.v_valid_tmp[i, j, k + 1] == 1: s += self.v[i, j, k + 1]; cnt += 1
                if cnt > 0:
                    self.v[i, j, k] = s / cnt
                    self.v_valid[i, j, k] = 1

    @ti.kernel
    def _extrapolate_w(self):
        n = self.n
        for i, j, k in ti.ndrange(n, n, n + 1):
            self.w_valid_tmp[i, j, k] = self.w_valid[i, j, k]
        for i, j, k in ti.ndrange(n, n, n + 1):
            if self.w_valid_tmp[i, j, k] == 0:
                s = 0.0; cnt = 0
                if i > 0     and self.w_valid_tmp[i - 1, j, k] == 1: s += self.w[i - 1, j, k]; cnt += 1
                if i < n - 1 and self.w_valid_tmp[i + 1, j, k] == 1: s += self.w[i + 1, j, k]; cnt += 1
                if j > 0     and self.w_valid_tmp[i, j - 1, k] == 1: s += self.w[i, j - 1, k]; cnt += 1
                if j < n - 1 and self.w_valid_tmp[i, j + 1, k] == 1: s += self.w[i, j + 1, k]; cnt += 1
                if k > 0     and self.w_valid_tmp[i, j, k - 1] == 1: s += self.w[i, j, k - 1]; cnt += 1
                if k < n     and self.w_valid_tmp[i, j, k + 1] == 1: s += self.w[i, j, k + 1]; cnt += 1
                if cnt > 0:
                    self.w[i, j, k] = s / cnt
                    self.w_valid[i, j, k] = 1

    def extrapolate_velocity(self, n_iters=10):
        for _ in range(n_iters):
            self._extrapolate_u()
            self._extrapolate_v()
            self._extrapolate_w()

    # ==================================================================
    # Pressure projection
    # ==================================================================
    @ti.kernel
    def _compute_divergence(self):
        """RHS = -(1/dx) * divergence(velocity) for FLUID cells."""
        inv_dx = self.inv_dx
        for i, j, k in self.pressure.b_vec:
            self.pressure.b_vec[i, j, k] = 0.0
        for i, j, k in self.cell_type:
            if self.cell_type[i, j, k] == FLUID:
                div = (self.u[i + 1, j, k] - self.u[i, j, k]
                     + self.v[i, j + 1, k] - self.v[i, j, k]
                     + self.w[i, j, k + 1] - self.w[i, j, k]) * inv_dx
                self.pressure.b_vec[i, j, k] = -div

    def solve_pressure(self, dt, verbose=False):
        """Build Laplacian and solve pressure Poisson equation."""
        scale_A = dt / (self.dx * self.dx)  # dt / (rho * dx^2), rho=1
        self.pressure.build_multigrid_hierarchy(self.cell_type, scale_A)
        self._compute_divergence()
        return self.pressure.solve(max_iters=self.pcg_max_iter,
                                   tol=self.pcg_tol, verbose=verbose)

    @ti.kernel
    def _apply_pressure(self, dt: float):
        """Subtract pressure gradient from face velocities."""
        n = self.n
        scale = dt / self.dx  # dt / (rho * dx), rho=1

        for i, j, k in ti.ndrange(n + 1, n, n):
            if 0 < i < n:
                left = self.cell_type[i - 1, j, k]
                right = self.cell_type[i, j, k]
                if (left == FLUID or right == FLUID) and left != SOLID and right != SOLID:
                    p_l = self.pressure.x_vec[i - 1, j, k] if left == FLUID else 0.0
                    p_r = self.pressure.x_vec[i, j, k] if right == FLUID else 0.0
                    self.u[i, j, k] -= scale * (p_r - p_l)

        for i, j, k in ti.ndrange(n, n + 1, n):
            if 0 < j < n:
                left = self.cell_type[i, j - 1, k]
                right = self.cell_type[i, j, k]
                if (left == FLUID or right == FLUID) and left != SOLID and right != SOLID:
                    p_l = self.pressure.x_vec[i, j - 1, k] if left == FLUID else 0.0
                    p_r = self.pressure.x_vec[i, j, k] if right == FLUID else 0.0
                    self.v[i, j, k] -= scale * (p_r - p_l)

        for i, j, k in ti.ndrange(n, n, n + 1):
            if 0 < k < n:
                left = self.cell_type[i, j, k - 1]
                right = self.cell_type[i, j, k]
                if (left == FLUID or right == FLUID) and left != SOLID and right != SOLID:
                    p_l = self.pressure.x_vec[i, j, k - 1] if left == FLUID else 0.0
                    p_r = self.pressure.x_vec[i, j, k] if right == FLUID else 0.0
                    self.w[i, j, k] -= scale * (p_r - p_l)

    # ==================================================================
    # G2P: FLIP velocity transfer
    # ==================================================================
    @ti.func
    def _sample_u_field(self, f: ti.template(), pos: ti.template()) -> float:
        """Trilinear sample of a u-type field (offset 0, 0.5, 0.5)."""
        n = self.n
        gx = pos[0] * self.inv_dx
        gy = pos[1] * self.inv_dx - 0.5
        gz = pos[2] * self.inv_dx - 0.5
        bi = int(ti.floor(gx)); bj = int(ti.floor(gy)); bk = int(ti.floor(gz))
        fx = gx - bi; fy = gy - bj; fz = gz - bk
        val = 0.0
        for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
            ii, jj, kk = bi + di, bj + dj, bk + dk
            if 0 <= ii < n + 1 and 0 <= jj < n and 0 <= kk < n:
                w = (1.0 - ti.abs(fx - di)) * (1.0 - ti.abs(fy - dj)) * (1.0 - ti.abs(fz - dk))
                val += w * f[ii, jj, kk]
        return val

    @ti.func
    def _sample_v_field(self, f: ti.template(), pos: ti.template()) -> float:
        """Trilinear sample of a v-type field (offset 0.5, 0, 0.5)."""
        n = self.n
        gx = pos[0] * self.inv_dx - 0.5
        gy = pos[1] * self.inv_dx
        gz = pos[2] * self.inv_dx - 0.5
        bi = int(ti.floor(gx)); bj = int(ti.floor(gy)); bk = int(ti.floor(gz))
        fx = gx - bi; fy = gy - bj; fz = gz - bk
        val = 0.0
        for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
            ii, jj, kk = bi + di, bj + dj, bk + dk
            if 0 <= ii < n and 0 <= jj < n + 1 and 0 <= kk < n:
                w = (1.0 - ti.abs(fx - di)) * (1.0 - ti.abs(fy - dj)) * (1.0 - ti.abs(fz - dk))
                val += w * f[ii, jj, kk]
        return val

    @ti.func
    def _sample_w_field(self, f: ti.template(), pos: ti.template()) -> float:
        """Trilinear sample of a w-type field (offset 0.5, 0.5, 0)."""
        n = self.n
        gx = pos[0] * self.inv_dx - 0.5
        gy = pos[1] * self.inv_dx - 0.5
        gz = pos[2] * self.inv_dx
        bi = int(ti.floor(gx)); bj = int(ti.floor(gy)); bk = int(ti.floor(gz))
        fx = gx - bi; fy = gy - bj; fz = gz - bk
        val = 0.0
        for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
            ii, jj, kk = bi + di, bj + dj, bk + dk
            if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n + 1:
                w = (1.0 - ti.abs(fx - di)) * (1.0 - ti.abs(fy - dj)) * (1.0 - ti.abs(fz - dk))
                val += w * f[ii, jj, kk]
        return val

    @ti.func
    def _interp_vel(self, pos: ti.template()) -> ti.Vector:
        return ti.Vector([
            self._sample_u_field(self.u, pos),
            self._sample_v_field(self.v, pos),
            self._sample_w_field(self.w, pos),
        ])

    @ti.func
    def _interp_vel_old(self, pos: ti.template()) -> ti.Vector:
        return ti.Vector([
            self._sample_u_field(self.u_old, pos),
            self._sample_v_field(self.v_old, pos),
            self._sample_w_field(self.w_old, pos),
        ])

    @ti.kernel
    def _g2p_flip(self):
        """FLIP: v_p = ratio*(v_p + v_new - v_old) + (1-ratio)*v_PIC."""
        ratio = self.flip_ratio
        for p in range(self.num_p[None]):
            pos = self.x[p]
            v_pic = self._interp_vel(pos)
            v_old_g = self._interp_vel_old(pos)
            v_flip = self.vel[p] + (v_pic - v_old_g)
            self.vel[p] = ratio * v_flip + (1.0 - ratio) * v_pic

    # ==================================================================
    # Advection: RK2 midpoint + foam tagging
    # ==================================================================
    @ti.kernel
    def _advect_rk2(self, dt: float):
        lo = self.dx * self.bound
        hi = 1.0 - lo
        n = self.n

        for p in range(self.num_p[None]):
            pos = self.x[p]
            vp = self.vel[p]

            # RK2: midpoint
            mid = pos + 0.5 * dt * vp
            for d in ti.static(range(3)):
                mid[d] = ti.max(lo, ti.min(hi, mid[d]))
            v_mid = self._interp_vel(mid)
            new_pos = pos + dt * v_mid
            for d in ti.static(range(3)):
                new_pos[d] = ti.max(lo, ti.min(hi, new_pos[d]))
            self.x[p] = new_pos

            # --- Foam tagging ---
            speed = vp.norm()
            foam_new = ti.min(1.0, ti.max(0.0, (speed - self.foam_v_thresh) / self.foam_v_thresh))
            ci = ti.max(1, ti.min(n - 2, int(ti.floor(new_pos[0] * self.inv_dx))))
            cj = ti.max(1, ti.min(n - 2, int(ti.floor(new_pos[1] * self.inv_dx))))
            ck = ti.max(1, ti.min(n - 2, int(ti.floor(new_pos[2] * self.inv_dx))))
            near_air = 0
            for di, dj, dk in ti.static(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                if self.cell_type[ci + di, cj + dj, ck + dk] == AIR:
                    near_air = 1
            foam_new *= ti.cast(near_air, float)
            self.foam[p] = self.foam[p] * self.foam_decay + foam_new * (1.0 - self.foam_decay)

    # ==================================================================
    # Push particles out of SDF solids
    # ==================================================================
    @ti.kernel
    def _push_out_of_solids(self):
        for p in range(self.num_p[None]):
            pos = self.x[p]
            ci = ti.max(0, ti.min(self.n - 1, int(ti.floor(pos[0] * self.inv_dx))))
            cj = ti.max(0, ti.min(self.n - 1, int(ti.floor(pos[1] * self.inv_dx))))
            ck = ti.max(0, ti.min(self.n - 1, int(ti.floor(pos[2] * self.inv_dx))))
            if self.sdf[ci, cj, ck] < 0.0:
                n_vec = self.sdf_n[ci, cj, ck]
                self.x[p] += (-self.sdf[ci, cj, ck] + self.dx * 0.5) * n_vec

    # ==================================================================
    # Max velocity (for CFL reporting)
    # ==================================================================
    @ti.kernel
    def _compute_v_max(self):
        self.v_max_field[None] = 0.0
        for p in range(self.num_p[None]):
            speed = self.vel[p].norm()
            ti.atomic_max(self.v_max_field[None], speed)

    def max_velocity(self):
        self._compute_v_max()
        return self.v_max_field[None]

    # ==================================================================
    # Water level query
    # ==================================================================
    def get_water_level(self, percentile=90):
        n = self.num_p[None]
        if n == 0:
            return self.floor_cell * self.dx
        z = self.x.to_numpy()[:n, 2]
        return float(np.percentile(z, percentile))

    # ==================================================================
    # Particle injection (SDF-aware)
    # ==================================================================
    def inject_uniform(self, z_lo, z_hi, count, xy_bounds=None,
                       velocity=(0.0, 0.0, 0.0)):
        """Add particles uniformly in [z_lo, z_hi]. Returns count added."""
        cur_n = self.num_p[None]
        room = self.max_p - cur_n
        if room <= 0:
            return 0
        target = min(count, room)

        bnd = self.bound * self.dx + self.dx
        if xy_bounds is not None:
            x_lo, x_hi, y_lo, y_hi = xy_bounds
            x_lo = max(x_lo, bnd); x_hi = min(x_hi, 1.0 - bnd)
            y_lo = max(y_lo, bnd); y_hi = min(y_hi, 1.0 - bnd)
        else:
            x_lo, x_hi = bnd, 1.0 - bnd
            y_lo, y_hi = bnd, 1.0 - bnd

        solid_np = self._solid_np
        rng = np.random.default_rng()

        # Probe free-space fraction
        probe = 2000
        probe_pos = np.column_stack([
            rng.uniform(x_lo, x_hi, probe),
            rng.uniform(y_lo, y_hi, probe),
            rng.uniform(z_lo, z_hi, probe),
        ]).astype(np.float32)
        gi = np.clip((probe_pos * self.n).astype(int), 0, self.n - 1)
        free_frac = float((solid_np[gi[:, 0], gi[:, 1], gi[:, 2]] == 0).mean())
        if free_frac < 0.01:
            return 0

        oversample = min(int(target / free_frac * 1.3) + 100, room * 3)
        pos = np.column_stack([
            rng.uniform(x_lo, x_hi, oversample),
            rng.uniform(y_lo, y_hi, oversample),
            rng.uniform(z_lo, z_hi, oversample),
        ]).astype(np.float32)
        gi = np.clip((pos * self.n).astype(int), 0, self.n - 1)
        pos = pos[solid_np[gi[:, 0], gi[:, 1], gi[:, 2]] == 0]
        if len(pos) > target:
            pos = pos[:target]
        actual = len(pos)
        if actual == 0:
            return 0

        x_np = self.x.to_numpy()
        x_np[cur_n:cur_n + actual] = pos
        self.x.from_numpy(x_np)
        self._init_inflow(cur_n, actual,
                          float(velocity[0]), float(velocity[1]), float(velocity[2]))
        self.num_p[None] = cur_n + actual
        return actual

    @ti.kernel
    def _init_inflow(self, start: int, count: int,
                     vx: float, vy: float, vz: float):
        for i in range(count):
            p = start + i
            self.vel[p] = ti.Vector([vx, vy, vz])
            self.foam[p] = 0.0

    # ==================================================================
    # Export
    # ==================================================================
    def export_frame(self, path):
        """Export positions, velocities, foam as NPZ."""
        n = self.num_p[None]
        x = self.x.to_numpy()[:n]
        v = self.vel.to_numpy()[:n]
        foam = self.foam.to_numpy()[:n]
        np.savez_compressed(path, x=x, v=v, foam=foam)

    # ==================================================================
    # Full substep pipeline
    # ==================================================================
    def substep(self, dt=None, verbose=False):
        """One FLIP substep. Returns PCG iteration count."""
        if dt is None:
            dt = self.dt

        # 1. Classify cells
        self.mark_cells()

        # 2. Transfer particle velocity to MAC grid
        self.p2g()

        # 3. Save pre-projection velocity (for FLIP delta)
        self._save_velocity()

        # 4. External forces (gravity)
        self._add_gravity(dt)

        # 5-6. Boundary + extrapolate
        self._enforce_boundary()
        self.extrapolate_velocity(n_iters=10)
        self._enforce_boundary()

        # 7-8. Pressure projection
        pcg_iters = self.solve_pressure(dt, verbose=verbose)
        self._apply_pressure(dt)

        # 9-10. Boundary + extrapolate (post-projection)
        self._enforce_boundary()
        self.extrapolate_velocity(n_iters=10)
        self._enforce_boundary()

        # 11. FLIP velocity transfer back to particles
        self._g2p_flip()

        # 12. Advect particles (RK2) + foam tagging
        self._advect_rk2(dt)

        # 13. Push particles out of solid obstacles
        self._push_out_of_solids()

        return pcg_iters

    def step(self, frame_dt=None, n_substeps=1, verbose=False):
        """Run one frame of simulation. Returns total PCG iterations."""
        if frame_dt is None:
            frame_dt = self.dt * n_substeps
        dt_sub = frame_dt / n_substeps

        total_iters = 0
        for s in range(n_substeps):
            iters = self.substep(dt=dt_sub, verbose=verbose and s == 0)
            total_iters += iters
        return total_iters
