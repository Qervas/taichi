"""Phase 10 — MLS-MPM 3D water solver with SDF mesh collision.

Copy of phase 9 solver with added support for Z-up OBJ models
(skip Y→Z rotation when z_up=True in load_solid).
"""
import taichi as ti
import numpy as np


@ti.data_oriented
class MPMFluid:
    def __init__(self, n_grid=128, max_p=4_000_000, gravity=(0, 0, -5),
                 E=400.0, nu=0.2, dt=2e-4, floor_z=None):
        n = n_grid
        self.n = n
        self.dx = 1.0 / n
        self.inv_dx = float(n)
        self.bound = 3
        self.dt = dt
        self.max_p = max_p
        self.floor_cell = int(floor_z * n) if floor_z is not None else self.bound

        # Lamé parameter (mu=0 for water)
        self.la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # Particle volume & mass (spacing = dx/2 → 8 per cell)
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol * 1.0  # rho = 1

        # Grid fields
        self.grid_v = ti.Vector.field(3, float, shape=(n, n, n))
        self.grid_m = ti.field(float, shape=(n, n, n))

        # SDF boundary fields
        self.sdf = ti.field(float, shape=(n, n, n))
        self.sdf_n = ti.Vector.field(3, float, shape=(n, n, n))
        self.solid = ti.field(ti.i32, shape=(n, n, n))

        # Particle fields
        self.x = ti.Vector.field(3, float, shape=max_p)
        self.v = ti.Vector.field(3, float, shape=max_p)
        self.C = ti.Matrix.field(3, 3, float, shape=max_p)
        self.J = ti.field(float, shape=max_p)

        self.num_p = ti.field(int, shape=())
        self.grav = ti.Vector(list(gravity))

    # ------------------------------------------------------------------
    def load_solid(self, mesh_path, center_xy, scale, z_up=False):
        """Compute SDF from mesh and load boundary fields.

        Args:
            mesh_path: path to GLB or OBJ file
            center_xy: (x, y) placement in [0,1] domain
            scale: fraction of domain to fill
            z_up: if True, skip Y→Z rotation (OBJ already Z-up)

        Returns (solid_np, mesh_bounds, transform).
        """
        import trimesh

        print(f"  Loading {mesh_path}...")
        scene = trimesh.load(mesh_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        print(f"  Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

        # GLB is Y-up → rotate to Z-up. OBJ with z_up=True: skip.
        if not z_up:
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            mesh.apply_transform(rot)

        # Save native coords for inverse transform
        native_centroid = mesh.centroid.copy()
        native_max_ext = float(mesh.extents.max())
        native_bounds = mesh.bounds.copy()

        # Normalize: center, scale to fit domain
        mesh.apply_translation(-mesh.centroid)
        max_ext = mesh.extents.max()
        mesh.apply_scale(scale / max_ext)

        # Place: XY from config, Z so building sits on ground
        ground_z = self.bound * self.dx
        half_z = mesh.extents[2] / 2
        placement = np.array([center_xy[0], center_xy[1], ground_z + half_z])
        mesh.apply_translation(placement)

        # Inverse transform: sim_pos → native_pos
        inv_scale = native_max_ext / scale
        inv_offset = native_centroid - placement * inv_scale

        print(f"  Placed at {placement.round(4)}, scale={scale:.3f}")
        print(f"  Bounds: {mesh.bounds[0].round(4)} → {mesh.bounds[1].round(4)}")
        print(f"  Extents: {mesh.extents.round(4)}")
        print(f"  Native bounds: {native_bounds[0].round(4)} → {native_bounds[1].round(4)}")
        print(f"  Inverse transform: scale={inv_scale:.4f}, offset={inv_offset.round(4)}")

        # ----- SDF via surface voxelization + EDT -----
        print("  Computing SDF via EDT...")
        dx = self.dx
        from scipy.ndimage import binary_dilation, distance_transform_edt

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

        print(f"    Surface: {n_surface:,} voxels, after dilation: {n_thick:,} ({100*n_thick/self.n**3:.1f}%)")
        print(f"    SDF range: [{sdf_np.min():.4f}, {sdf_np.max():.4f}]")

        transform = dict(
            inv_scale=float(inv_scale),
            inv_offset=inv_offset.tolist(),
            native_bounds=native_bounds.tolist(),
        )
        return solid_np, mesh.bounds, transform

    # ------------------------------------------------------------------
    def get_water_level(self, percentile=90):
        n = self.num_p[None]
        if n == 0:
            return self.floor_cell * self.dx
        z = self.x.to_numpy()[:n, 2]
        return float(np.percentile(z, percentile))

    # ------------------------------------------------------------------
    def inject_uniform(self, z_lo, z_hi, count, xy_bounds=None):
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

        self._init_inflow_fields(cur_n, actual, 0.0, 0.0, 0.0)
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

    # ------------------------------------------------------------------
    # MLS-MPM kernels
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

            self.J[p] *= 1.0 + self.dt * self.C[p].trace()
            self.J[p] = ti.max(self.J[p], 0.05)

            stress = (self.la * self.J[p] * (self.J[p] - 1.0)
                      * ti.Matrix.identity(float, 3))
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
                        self.grid_v[I] -= v_n * n

            for d in ti.static(range(3)):
                if I[d] < self.bound and self.grid_v[I][d] < 0:
                    self.grid_v[I][d] = 0
                if I[d] > self.n - self.bound and self.grid_v[I][d] > 0:
                    self.grid_v[I][d] = 0
            if I[2] < self.floor_cell and self.grid_v[I][2] < 0:
                self.grid_v[I][2] = 0

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

            lo = self.dx * self.bound
            hi = 1.0 - self.dx * self.bound
            for d in ti.static(range(3)):
                self.x[p][d] = ti.max(lo, ti.min(hi, self.x[p][d]))

    # ------------------------------------------------------------------
    def substep(self):
        self.p2g()
        self.grid_op()
        self.g2p()

    def step(self, substeps=25):
        for _ in range(substeps):
            self.substep()

    def export_frame(self, path):
        n = self.num_p[None]
        x = self.x.to_numpy()[:n]
        v = self.v.to_numpy()[:n]
        np.savez_compressed(path, x=x, v=v)
