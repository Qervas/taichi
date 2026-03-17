"""Phase 11 — Physically accurate MLS-MPM 3D water solver.

Improvements over Phase 9:
- Tait EOS (gamma=7) for near-incompressible water
- Explicit viscosity for wall boundary layers
- Smagorinsky SPS turbulence model
- Vorticity confinement for sub-grid swirl
- Partial-slip boundary conditions with wall friction
- Per-particle foam tagging
"""
import taichi as ti
import numpy as np


@ti.data_oriented
class MPMFluid:
    def __init__(self, n_grid=192, max_p=5_000_000,
                 gravity=(0, 0, -9.81),
                 tait_B=89.3, tait_gamma=7, mu=0.002,
                 smag_cs=0.12, vort_eps=0.8,
                 wall_friction=0.3, floor_friction=0.15,
                 foam_v_thresh=1.5, foam_decay=0.92,
                 dt=1e-4, floor_z=None):
        n = n_grid
        self.n = n
        self.dx = 1.0 / n
        self.inv_dx = float(n)
        self.bound = 3
        self.dt = dt
        self.max_p = max_p
        self.floor_cell = int(floor_z * n) if floor_z is not None else self.bound

        # Tait EOS: p = B * (J^(-gamma) - 1)
        self.B = tait_B
        self.gamma_f = float(tait_gamma)
        self.mu = mu  # explicit viscosity

        # Turbulence
        self.smag_cs = smag_cs
        self.vort_eps = vort_eps

        # Boundary friction
        self.wall_friction = wall_friction
        self.floor_friction = floor_friction

        # Foam
        self.foam_v_thresh = foam_v_thresh
        self.foam_decay = foam_decay

        # Particle volume & mass (spacing = dx/2 -> 8 per cell)
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol * 1.0  # rho = 1

        # Grid fields
        self.grid_v = ti.Vector.field(3, float, shape=(n, n, n))
        self.grid_m = ti.field(float, shape=(n, n, n))

        # Vorticity fields
        self.omega = ti.Vector.field(3, float, shape=(n, n, n))
        self.vort_mag = ti.field(float, shape=(n, n, n))

        # SDF boundary fields
        self.sdf = ti.field(float, shape=(n, n, n))
        self.sdf_n = ti.Vector.field(3, float, shape=(n, n, n))
        self.solid = ti.field(ti.i32, shape=(n, n, n))

        # Particle fields
        self.x = ti.Vector.field(3, float, shape=max_p)
        self.v = ti.Vector.field(3, float, shape=max_p)
        self.C = ti.Matrix.field(3, 3, float, shape=max_p)
        self.J = ti.field(float, shape=max_p)
        self.foam = ti.field(float, shape=max_p)

        # Particle count
        self.num_p = ti.field(int, shape=())
        self.grav = ti.Vector(list(gravity))

    # ------------------------------------------------------------------
    # SDF computation from mesh (same as Phase 9)
    # ------------------------------------------------------------------
    def load_solid(self, glb_path, center_xy, scale):
        """Compute SDF from GLB mesh. Returns (solid_np, bounds, transform)."""
        import trimesh

        print(f"  Loading {glb_path}...")
        scene = trimesh.load(glb_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        print(f"  Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot)

        native_centroid = mesh.centroid.copy()
        native_max_ext = float(mesh.extents.max())
        native_bounds = mesh.bounds.copy()

        mesh.apply_translation(-mesh.centroid)
        max_ext = mesh.extents.max()
        mesh.apply_scale(scale / max_ext)

        ground_z = self.bound * self.dx
        half_z = mesh.extents[2] / 2
        placement = np.array([center_xy[0], center_xy[1], ground_z + half_z])
        mesh.apply_translation(placement)

        inv_scale = native_max_ext / scale
        inv_offset = native_centroid - placement * inv_scale

        print(f"  Placed at {placement.round(4)}, scale={scale:.3f}")
        print(f"  Bounds: {mesh.bounds[0].round(4)} -> {mesh.bounds[1].round(4)}")

        self._compute_sdf(mesh)

        transform = dict(
            inv_scale=float(inv_scale),
            inv_offset=inv_offset.tolist(),
            native_bounds=native_bounds.tolist(),
        )
        return self._solid_np, mesh.bounds, transform

    def _compute_sdf(self, mesh):
        """EDT-based SDF from mesh surface."""
        from scipy.ndimage import binary_dilation, distance_transform_edt

        print("  Computing SDF via EDT...")
        dx = self.dx

        vox = mesh.voxelized(pitch=dx)
        pts = vox.points
        gi = np.clip(np.floor(pts / dx).astype(int), 0, self.n - 1)

        surface_np = np.zeros((self.n, self.n, self.n), dtype=bool)
        surface_np[gi[:, 0], gi[:, 1], gi[:, 2]] = True
        n_surface = int(surface_np.sum())

        thick = binary_dilation(surface_np, iterations=1)
        n_thick = int(thick.sum())

        dist_in = distance_transform_edt(thick).astype(np.float32) * dx
        dist_out = distance_transform_edt(~thick).astype(np.float32) * dx
        sdf_np = np.where(thick, -dist_in, dist_out).astype(np.float32)

        gx, gy, gz = np.gradient(sdf_np, dx)
        normal_np = np.zeros((self.n, self.n, self.n, 3), dtype=np.float32)
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

        print(f"    Surface: {n_surface:,} voxels, after dilation: {n_thick:,}")
        print(f"    SDF range: [{sdf_np.min():.4f}, {sdf_np.max():.4f}]")

    # ------------------------------------------------------------------
    # Water level query
    # ------------------------------------------------------------------
    def get_water_level(self, percentile=90):
        n = self.num_p[None]
        if n == 0:
            return self.floor_cell * self.dx
        z = self.x.to_numpy()[:n, 2]
        return float(np.percentile(z, percentile))

    # ------------------------------------------------------------------
    # Flood injection (SDF-aware)
    # ------------------------------------------------------------------
    def inject_uniform(self, z_lo, z_hi, count, xy_bounds=None,
                        velocity=(0.0, 0.0, 0.0)):
        """Add particles uniformly at [z_lo, z_hi] with optional velocity. Returns count added."""
        cur_n = self.num_p[None]
        room = self.max_p - cur_n
        if room <= 0:
            return 0

        target = min(count, room)
        bnd = self.bound * self.dx + self.dx
        if xy_bounds is not None:
            x_lo, x_hi, y_lo, y_hi = xy_bounds
            x_lo = max(x_lo, bnd)
            x_hi = min(x_hi, 1.0 - bnd)
            y_lo = max(y_lo, bnd)
            y_hi = min(y_hi, 1.0 - bnd)
        else:
            x_lo, x_hi = bnd, 1.0 - bnd
            y_lo, y_hi = bnd, 1.0 - bnd

        solid_np = self._solid_np

        rng = np.random.default_rng()
        probe = 2000
        probe_pos = np.zeros((probe, 3), dtype=np.float32)
        probe_pos[:, 0] = rng.uniform(x_lo, x_hi, probe).astype(np.float32)
        probe_pos[:, 1] = rng.uniform(y_lo, y_hi, probe).astype(np.float32)
        probe_pos[:, 2] = rng.uniform(z_lo, z_hi, probe).astype(np.float32)
        gi = np.clip((probe_pos * self.n).astype(int), 0, self.n - 1)
        free_frac = float((solid_np[gi[:, 0], gi[:, 1], gi[:, 2]] == 0).mean())

        if free_frac < 0.01:
            return 0

        oversample = int(target / free_frac * 1.3) + 100
        oversample = min(oversample, room * 3)

        pos = np.zeros((oversample, 3), dtype=np.float32)
        pos[:, 0] = rng.uniform(x_lo, x_hi, oversample).astype(np.float32)
        pos[:, 1] = rng.uniform(y_lo, y_hi, oversample).astype(np.float32)
        pos[:, 2] = rng.uniform(z_lo, z_hi, oversample).astype(np.float32)

        gi = np.clip((pos * self.n).astype(int), 0, self.n - 1)
        free = solid_np[gi[:, 0], gi[:, 1], gi[:, 2]] == 0
        pos = pos[free]

        if len(pos) > target:
            pos = pos[:target]
        actual = len(pos)

        if actual == 0:
            return 0

        x_np = self.x.to_numpy()
        x_np[cur_n:cur_n + actual] = pos
        self.x.from_numpy(x_np)

        self._init_inflow_fields(cur_n, actual,
                                 float(velocity[0]), float(velocity[1]), float(velocity[2]))
        self.num_p[None] = cur_n + actual
        return actual

    @ti.kernel
    def _init_inflow_fields(self, start: int, count: int,
                             vx: float, vy: float, vz: float):
        for i in range(count):
            p = start + i
            self.v[p] = ti.Vector([vx, vy, vz])
            self.C[p] = ti.Matrix.zero(float, 3, 3)
            self.J[p] = 1.0
            self.foam[p] = 0.0

    # ------------------------------------------------------------------
    # MLS-MPM kernels — Tait EOS + viscosity + SPS turbulence
    # ------------------------------------------------------------------
    @ti.kernel
    def p2g(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(float, 3)
            self.grid_m[I] = 0.0

        for p in range(self.num_p[None]):
            Xp = self.x[p]
            base = (Xp * self.inv_dx - 0.5).cast(int)
            fx = Xp * self.inv_dx - base.cast(float)

            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            # Update J (volume ratio)
            self.J[p] *= 1.0 + self.dt * self.C[p].trace()
            self.J[p] = ti.max(self.J[p], 0.5)  # clamp: max 2x compression

            J = self.J[p]

            # --- Tait EOS pressure ---
            # p = B * (J^(-gamma) - 1)
            J_neg_g = 1.0 / (J * J * J * J * J * J * J)  # J^(-7)
            pressure = self.B * (J_neg_g - 1.0)

            # Kirchhoff stress from pressure: P*F^T = -J*p*I
            stress = (-pressure * J) * ti.Matrix.identity(float, 3)

            # --- Viscosity + SPS turbulence ---
            Cp = self.C[p]
            S = 0.5 * (Cp + Cp.transpose())  # strain rate tensor
            S_sqr = (S @ S).trace()  # trace(S^2) = sum(S_ij^2)
            S_mag = ti.sqrt(2.0 * S_sqr + 1e-12)

            # Smagorinsky eddy viscosity
            mu_sgs = (self.smag_cs * self.dx) ** 2 * S_mag
            mu_eff = self.mu + mu_sgs

            # Add viscous + SPS stress: J * 2 * mu_eff * S
            stress += J * 2.0 * mu_eff * S

            # Scale stress for P2G
            stress = (-self.dt * self.p_vol * 4.0
                      * self.inv_dx * self.inv_dx) * stress

            affine = stress + self.p_mass * self.C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v[base + offset] += weight * (
                    self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def grid_op(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I]
                self.grid_v[I] += self.dt * self.grav

            # --- SDF boundary: partial-slip with friction ---
            sdf_val = self.sdf[I]
            if sdf_val < 0.0:
                self.grid_v[I] = ti.Vector.zero(float, 3)
            elif sdf_val < self.dx * 1.5:
                n = self.sdf_n[I]
                n_len = n.norm()
                if n_len > 1e-6:
                    n = n / n_len
                    v_n = self.grid_v[I].dot(n)
                    if v_n < 0.0:
                        # Remove normal component
                        v_t = self.grid_v[I] - v_n * n
                        # Apply wall friction to tangential
                        self.grid_v[I] = v_t * (1.0 - self.wall_friction)

            # Domain walls: no-penetration + floor friction
            for d in ti.static(range(3)):
                if I[d] < self.bound and self.grid_v[I][d] < 0:
                    self.grid_v[I][d] = 0
                if I[d] > self.n - self.bound and self.grid_v[I][d] > 0:
                    self.grid_v[I][d] = 0

            # Raised floor with friction
            if I[2] < self.floor_cell and self.grid_v[I][2] < 0:
                self.grid_v[I][2] = 0
                self.grid_v[I][0] *= (1.0 - self.floor_friction)
                self.grid_v[I][1] *= (1.0 - self.floor_friction)

    # ------------------------------------------------------------------
    # Vorticity confinement (two-pass)
    # ------------------------------------------------------------------
    @ti.kernel
    def compute_vorticity(self):
        """Pass 1: compute omega and |omega| on grid."""
        for I in ti.grouped(self.grid_m):
            i, j, k = I[0], I[1], I[2]
            if (1 <= i < self.n - 1 and
                1 <= j < self.n - 1 and
                1 <= k < self.n - 1 and
                self.grid_m[I] > 0):

                hdx = 0.5 * self.inv_dx  # 1/(2*dx)

                dvz_dy = (self.grid_v[i, j+1, k][2] - self.grid_v[i, j-1, k][2]) * hdx
                dvy_dz = (self.grid_v[i, j, k+1][1] - self.grid_v[i, j, k-1][1]) * hdx
                dvx_dz = (self.grid_v[i, j, k+1][0] - self.grid_v[i, j, k-1][0]) * hdx
                dvz_dx = (self.grid_v[i+1, j, k][2] - self.grid_v[i-1, j, k][2]) * hdx
                dvy_dx = (self.grid_v[i+1, j, k][1] - self.grid_v[i-1, j, k][1]) * hdx
                dvx_dy = (self.grid_v[i, j+1, k][0] - self.grid_v[i, j-1, k][0]) * hdx

                w = ti.Vector([dvz_dy - dvy_dz,
                               dvx_dz - dvz_dx,
                               dvy_dx - dvx_dy])
                self.omega[I] = w
                self.vort_mag[I] = w.norm()
            else:
                self.omega[I] = ti.Vector.zero(float, 3)
                self.vort_mag[I] = 0.0

    @ti.kernel
    def apply_vort_confinement(self):
        """Pass 2: apply confinement force f = eps * dx * (N x omega)."""
        for I in ti.grouped(self.grid_m):
            i, j, k = I[0], I[1], I[2]
            if (2 <= i < self.n - 2 and
                2 <= j < self.n - 2 and
                2 <= k < self.n - 2 and
                self.grid_m[I] > 0 and
                self.vort_mag[I] > 1e-6):

                hdx = 0.5 * self.inv_dx
                dw_dx = (self.vort_mag[i+1, j, k] - self.vort_mag[i-1, j, k]) * hdx
                dw_dy = (self.vort_mag[i, j+1, k] - self.vort_mag[i, j-1, k]) * hdx
                dw_dz = (self.vort_mag[i, j, k+1] - self.vort_mag[i, j, k-1]) * hdx

                N = ti.Vector([dw_dx, dw_dy, dw_dz])
                N_len = N.norm()
                if N_len > 1e-6:
                    N /= N_len
                    f = self.vort_eps * self.dx * N.cross(self.omega[I])
                    self.grid_v[I] += self.dt * f

    # ------------------------------------------------------------------
    # G2P with foam tagging
    # ------------------------------------------------------------------
    @ti.kernel
    def g2p(self):
        for p in range(self.num_p[None]):
            Xp = self.x[p]
            base = (Xp * self.inv_dx - 0.5).cast(int)
            fx = Xp * self.inv_dx - base.cast(float)

            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v = self.grid_v[base + offset]
                new_v += weight * g_v
                new_C += (4.0 * self.inv_dx * self.inv_dx
                          * weight * g_v.outer_product(dpos))

            self.v[p] = new_v
            self.x[p] += self.dt * new_v
            self.C[p] = new_C

            # --- Foam tagging ---
            speed = new_v.norm()
            foam_new = ti.max(0.0, (speed - self.foam_v_thresh) / self.foam_v_thresh)
            foam_new = ti.min(foam_new, 1.0)
            # Check if near surface: any neighbor node has zero mass
            empty_neighbors = 0
            for ii, jj, kk in ti.static(ti.ndrange(3, 3, 3)):
                nI = base + ti.Vector([ii, jj, kk])
                if self.grid_m[nI] < 1e-10:
                    empty_neighbors += 1
            near_surface = ti.cast(empty_neighbors > 3, float)
            foam_new *= near_surface
            # Blend with previous (persistence)
            self.foam[p] = self.foam[p] * self.foam_decay + foam_new * (1.0 - self.foam_decay)

            # Clamp to domain
            lo = self.dx * self.bound
            hi = 1.0 - self.dx * self.bound
            for d in ti.static(range(3)):
                self.x[p][d] = ti.max(lo, ti.min(hi, self.x[p][d]))

    # ------------------------------------------------------------------
    # Step & export
    # ------------------------------------------------------------------
    def substep(self):
        self.p2g()
        self.grid_op()
        self.compute_vorticity()
        self.apply_vort_confinement()
        self.g2p()

    def step(self, substeps=50):
        for _ in range(substeps):
            self.substep()

    def export_frame(self, path):
        """Export particle positions, velocities, and foam as NPZ."""
        n = self.num_p[None]
        x = self.x.to_numpy()[:n]
        v = self.v.to_numpy()[:n]
        foam = self.foam.to_numpy()[:n]
        np.savez_compressed(path, x=x, v=v, foam=foam)
