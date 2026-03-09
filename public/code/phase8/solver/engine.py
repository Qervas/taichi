"""
MLS-MPM Solver Engine — SOLIDS ONLY (building + cars + debris).

Water is handled by SWE. This solver handles:
  - Concrete (corotational elastic + Rankine fracture → rubble)
  - Cars (rigid-elastic, very stiff)
  - All solid-solid collisions through the shared grid

SWE forces are injected via swe_force_hook (set by HybridSolver).

    sim = Solver(mpm_origin, mpm_extent)
    sim.init()
    sim.swe_force_hook = coupling_kernel
    sim._substep()  # called by HybridSolver
"""

import sys
import os
import math
import numpy as np
import taichi as ti

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

from .fields import allocate, CONCRETE
from . import colliders

CAR = 2  # material ID for cars


class Solver:
    def __init__(self, mpm_origin=None, mpm_extent=None):
        # MPM sub-domain (auto-detected or manual)
        if mpm_origin is None:
            mpm_origin, mpm_extent = colliders.compute_mpm_zone()
        self.mpm_origin = tuple(mpm_origin)
        self.mpm_extent = tuple(mpm_extent)

        # Grid sizing: fit the extent with target dx
        max_ext = max(self.mpm_extent)
        self.n_grid = max(32, int(math.ceil(max_ext / C.MPM_DX)))
        self.domain = self.n_grid * C.MPM_DX
        self.dx = C.MPM_DX
        self.inv_dx = 1.0 / self.dx

        self.dt = C.MPM_DT
        self.substeps = C.MPM_SUBSTEPS
        self.gravity = C.GRAVITY
        self.export_dir = C.EXPORT_DIR
        self.frame_count = 0
        self.bound = C.BOUND
        self.floor_z_local = C.FLOOR_MARGIN - self.mpm_origin[2]

        # SWE coupling hook (set by HybridSolver)
        self.swe_force_hook = None

        self.F = None
        self.n_building = 0
        self.n_cars = 0
        self.n_particles = 0

        print(f"MPM: grid={self.n_grid}^3, dx={self.dx:.3f}m, "
              f"domain={self.domain:.1f}m, floor_z_local={self.floor_z_local:.1f}m")

    def init(self):
        """Seed solid particles and allocate fields."""
        building_pos, building_chunks = self._seed_building()
        car_pos = self._seed_cars()

        self.n_building = len(building_pos)
        self.n_cars = len(car_pos)
        self.n_particles = self.n_building + self.n_cars
        max_particles = self.n_particles + 20000

        print(f"  Building: {self.n_building:,} particles")
        print(f"  Cars:     {self.n_cars:,} particles")
        print(f"  Total:    {self.n_particles:,}")

        self.F = allocate(self.n_grid, max_particles)
        self.max_particles = max_particles

        self._init_particles(building_pos, building_chunks, car_pos)
        self._build_kernels()

    def _to_local(self, sim_x, sim_y, sim_z):
        """Convert sim-local coords to MPM-local coords."""
        return (sim_x - self.mpm_origin[0],
                sim_y - self.mpm_origin[1],
                sim_z - self.mpm_origin[2])

    def _to_sim(self, lx, ly, lz):
        """Convert MPM-local coords to sim-local coords."""
        return (lx + self.mpm_origin[0],
                ly + self.mpm_origin[1],
                lz + self.mpm_origin[2])

    # ------------------------------------------------------------------
    # Particle seeding
    # ------------------------------------------------------------------
    def _seed_building(self):
        spacing = self.dx * 0.5
        bld_lo_sim, bld_hi_sim = C.building_bounds_sim()

        # Convert to MPM-local coords
        lo = self._to_local(*bld_lo_sim)
        hi = self._to_local(*bld_hi_sim)

        floor_local = self.floor_z_local
        z_lo = floor_local
        z_hi = min(hi[2], floor_local + 20.0)

        xs = np.arange(lo[0] + spacing / 2, hi[0], spacing)
        ys = np.arange(lo[1] + spacing / 2, hi[1], spacing)
        zs = np.arange(z_lo + spacing / 2, z_hi, spacing)

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.array([], dtype=np.int32)

        positions = np.array(
            np.meshgrid(xs, ys, zs, indexing="ij")
        ).reshape(3, -1).T.astype(np.float32)

        # Voronoi chunks
        n_chunks = 60
        rng = np.random.default_rng(42)
        seeds = rng.uniform(
            [lo[0], lo[1], z_lo],
            [hi[0], hi[1], z_hi],
            size=(n_chunks, 3),
        ).astype(np.float32)
        from scipy.spatial import cKDTree
        tree = cKDTree(seeds)
        _, chunk_ids = tree.query(positions)

        # Save seeds for renderer (MPM-local coords)
        self.voronoi_seeds = seeds
        self.n_chunks = n_chunks

        print(f"  Building: X[{lo[0]:.1f},{hi[0]:.1f}] Y[{lo[1]:.1f},{hi[1]:.1f}] "
              f"Z[{z_lo:.1f},{z_hi:.1f}] (MPM-local), {n_chunks} chunks")
        return positions, chunk_ids.astype(np.int32)

    def _seed_cars(self):
        spacing = self.dx * 0.5
        car_boxes = C.car_boxes_sim()
        all_pos = []

        for center_sim, half, yaw in car_boxes:
            cx, cy, cz = self._to_local(*center_sim)
            # Simple AABB fill (ignoring yaw for particle seeding)
            r = max(half[0], half[1])
            xs = np.arange(cx - r + spacing / 2, cx + r, spacing)
            ys = np.arange(cy - r + spacing / 2, cy + r, spacing)
            zs = np.arange(cz - half[2] + spacing / 2, cz + half[2], spacing)

            if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
                continue

            grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
            all_pos.append(grid)

        if all_pos:
            positions = np.concatenate(all_pos, axis=0).astype(np.float32)
        else:
            positions = np.zeros((0, 3), dtype=np.float32)

        return positions

    # ------------------------------------------------------------------
    # GPU initialization
    # ------------------------------------------------------------------
    def _init_particles(self, building_pos, building_chunks, car_pos):
        F = self.F
        p_vol = (self.dx * 0.5) ** 3
        max_p = self.max_particles

        @ti.kernel
        def reset():
            for p in range(max_p):
                F["used"][p] = 0
                F["x"][p] = ti.Vector([0.0, 0.0, 0.0])
                F["v"][p] = ti.Vector([0.0, 0.0, 0.0])
                F["C"][p] = ti.Matrix.zero(float, 3, 3)
                F["dg"][p] = ti.Matrix.identity(float, 3)
                F["Jp"][p] = 1.0
                F["damage"][p] = 0.0
                F["material"][p] = CONCRETE
                F["mass"][p] = p_vol * C.CONCRETE["rho"]

        reset()

        # Upload building
        if self.n_building > 0:
            b_mass = p_vol * C.CONCRETE["rho"]
            @ti.kernel
            def upload_building(pos: ti.types.ndarray(), chunks: ti.types.ndarray(), n: int):
                for i in range(n):
                    F["x"][i] = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
                    F["used"][i] = 1
                    F["material"][i] = CONCRETE
                    F["mass"][i] = b_mass
                    F["chunk_id"][i] = chunks[i]
                    F["dg"][i] = ti.Matrix.identity(float, 3)

            upload_building(building_pos, building_chunks, self.n_building)

        # Upload cars
        if self.n_cars > 0:
            offset = self.n_building
            c_mass = p_vol * C.CAR["rho"]
            @ti.kernel
            def upload_cars(pos: ti.types.ndarray(), n: int):
                for i in range(n):
                    idx = offset + i
                    F["x"][idx] = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
                    F["used"][idx] = 1
                    F["material"][idx] = CAR
                    F["mass"][idx] = c_mass
                    F["dg"][idx] = ti.Matrix.identity(float, 3)

            upload_cars(car_pos, self.n_cars)

        F["n_particles"][None] = self.n_particles

    # ------------------------------------------------------------------
    # Build kernels
    # ------------------------------------------------------------------
    def _build_kernels(self):
        F = self.F
        n_grid = self.n_grid
        dx = self.dx
        inv_dx = self.inv_dx
        dt = self.dt
        gravity = self.gravity
        bound = self.bound
        floor_z_local = self.floor_z_local
        floor_grid_z = int(floor_z_local * inv_dx)

        p_vol = (dx * 0.5) ** 3
        stress_scale = p_vol * 4.0 * inv_dx * inv_dx

        # Concrete params
        E_c = C.CONCRETE["E"]
        nu_c = C.CONCRETE["nu"]
        mu_c = E_c / (2.0 * (1.0 + nu_c))
        la_c = E_c * nu_c / ((1.0 + nu_c) * (1.0 - 2.0 * nu_c))
        tensile_str = C.CONCRETE["tensile_strength"]
        damage_rate = C.CONCRETE["damage_rate"]
        softening_dmg = C.CONCRETE["softening_damage"]
        rubble_dmg = C.CONCRETE["rubble_damage"]
        E_rubble = C.CONCRETE["rubble_E"]
        vel_damp = C.VEL_DAMPING * dt

        # Car params
        E_car = C.CAR["E"]
        nu_car = C.CAR["nu"]
        mu_car = E_car / (2.0 * (1.0 + nu_car))
        la_car = E_car * nu_car / ((1.0 + nu_car) * (1.0 - 2.0 * nu_car))

        n_particles_ref = self.n_particles

        # Reference to SWE hook (called if set)
        solver_ref = self

        @ti.kernel
        def substep():
            # Reset grid
            for I in ti.grouped(F["grid_m"]):
                F["grid_v"][I] = ti.Vector([0.0, 0.0, 0.0])
                F["grid_m"][I] = 0.0

            # P2G
            for p in range(n_particles_ref):
                if F["used"][p] == 0:
                    continue
                Xp = F["x"][p] * inv_dx
                base = int(Xp - 0.5)
                fx = Xp - float(base)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

                mat_id = F["material"][p]

                # Update deformation gradient
                F["dg"][p] = (ti.Matrix.identity(float, 3) + dt * F["C"][p]) @ F["dg"][p]

                stress = ti.Matrix.zero(float, 3, 3)

                if mat_id == CONCRETE:
                    dmg = F["damage"][p]
                    if dmg < rubble_dmg:
                        dg = F["dg"][p]
                        U, sig, V = ti.svd(dg)
                        J = 1.0
                        for d in ti.static(range(3)):
                            J *= sig[d, d]

                        log_sig = ti.Vector([ti.log(ti.max(sig[i, i], 1e-6)) for i in range(3)])
                        tau_diag = 2.0 * mu_c * log_sig + la_c * log_sig.sum() * ti.Vector([1.0, 1.0, 1.0])

                        max_p = ti.max(tau_diag[0], ti.max(tau_diag[1], tau_diag[2]))
                        eff_str = tensile_str * ti.max(0.0, 1.0 - dmg / softening_dmg)
                        if max_p > eff_str:
                            F["damage"][p] = dmg + damage_rate
                            for d in ti.static(range(3)):
                                if tau_diag[d] > 0:
                                    tau_diag[d] = 0.0

                        PFt = U @ ti.Matrix([
                            [tau_diag[0], 0, 0],
                            [0, tau_diag[1], 0],
                            [0, 0, tau_diag[2]]
                        ]) @ U.transpose()

                        stress = -stress_scale * PFt
                    else:
                        # Rubble
                        dg_r = F["dg"][p]
                        U_r, sig_r, V_r = ti.svd(dg_r)
                        J_r = 1.0
                        for d in ti.static(range(3)):
                            J_r *= sig_r[d, d]
                        new_F_r = ti.Matrix.identity(float, 3)
                        new_F_r[0, 0] = J_r
                        F["dg"][p] = new_F_r
                        pressure = E_rubble * (1.0 - J_r)
                        stress = pressure * stress_scale * ti.Matrix.identity(float, 3)

                else:  # CAR — Neo-Hookean elastic (very stiff)
                    dg = F["dg"][p]
                    U, sig, V = ti.svd(dg)
                    J = 1.0
                    for d in ti.static(range(3)):
                        J *= sig[d, d]
                    # Neo-Hookean: stress = 2*mu*(F-R)*F^T + la*J*(J-1)*I
                    R = U @ V.transpose()
                    stress_tensor = 2.0 * mu_car * (dg - R) @ dg.transpose() + la_car * J * (J - 1.0) * ti.Matrix.identity(float, 3)
                    stress = -stress_scale * stress_tensor

                affine = stress * dt + F["mass"][p] * F["C"][p]

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (float(offset) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    idx = base + offset
                    if 0 <= idx[0] < n_grid and 0 <= idx[1] < n_grid and 0 <= idx[2] < n_grid:
                        F["grid_v"][idx] += weight * (F["mass"][p] * F["v"][p] + affine @ dpos)
                        F["grid_m"][idx] += weight * F["mass"][p]

            # Grid update
            for I in ti.grouped(F["grid_m"]):
                if F["grid_m"][I] > 0:
                    F["grid_v"][I] /= F["grid_m"][I]
                    F["grid_v"][I][2] -= gravity * dt

                    # NOTE: SWE coupling forces are applied AFTER this kernel
                    # via swe_force_hook (called by HybridSolver)

                    # Boundary conditions
                    for d in ti.static(range(3)):
                        if I[d] < bound and F["grid_v"][I][d] < 0:
                            F["grid_v"][I][d] = 0
                        if I[d] > n_grid - bound and F["grid_v"][I][d] > 0:
                            F["grid_v"][I][d] = 0

                    # Floor
                    if I[2] <= floor_grid_z and F["grid_v"][I][2] < 0:
                        F["grid_v"][I][2] = 0

            # G2P
            for p in range(n_particles_ref):
                if F["used"][p] == 0:
                    continue
                Xp = F["x"][p] * inv_dx
                base = int(Xp - 0.5)
                fx = Xp - float(base)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

                new_v = ti.Vector([0.0, 0.0, 0.0])
                new_C = ti.Matrix.zero(float, 3, 3)

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (float(offset) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    idx = base + offset
                    if 0 <= idx[0] < n_grid and 0 <= idx[1] < n_grid and 0 <= idx[2] < n_grid:
                        g_v = F["grid_v"][idx]
                        new_v += weight * g_v
                        new_C += 4.0 * inv_dx * inv_dx * weight * g_v.outer_product(dpos)

                # Damping for concrete
                if F["material"][p] == CONCRETE and F["damage"][p] < rubble_dmg:
                    new_v *= (1.0 - vel_damp)

                F["v"][p] = new_v
                F["C"][p] = new_C
                F["x"][p] += dt * new_v

        # Store — but we split into P2G+grid and G2P for coupling injection
        # Actually, for simplicity, the SWE hook runs between substep calls
        # The hook modifies grid_v AFTER grid update but BEFORE G2P
        # To do this cleanly, we split the substep:

        @ti.kernel
        def substep_p2g_and_grid():
            # Reset grid
            for I in ti.grouped(F["grid_m"]):
                F["grid_v"][I] = ti.Vector([0.0, 0.0, 0.0])
                F["grid_m"][I] = 0.0

            # P2G
            for p in range(n_particles_ref):
                if F["used"][p] == 0:
                    continue
                Xp = F["x"][p] * inv_dx
                base = int(Xp - 0.5)
                fx = Xp - float(base)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

                mat_id = F["material"][p]
                F["dg"][p] = (ti.Matrix.identity(float, 3) + dt * F["C"][p]) @ F["dg"][p]

                stress = ti.Matrix.zero(float, 3, 3)

                if mat_id == CONCRETE:
                    dmg = F["damage"][p]
                    if dmg < rubble_dmg:
                        dg = F["dg"][p]
                        U, sig, V = ti.svd(dg)
                        J = 1.0
                        for d in ti.static(range(3)):
                            J *= sig[d, d]
                        log_sig = ti.Vector([ti.log(ti.max(sig[i, i], 1e-6)) for i in range(3)])
                        tau_diag = 2.0 * mu_c * log_sig + la_c * log_sig.sum() * ti.Vector([1.0, 1.0, 1.0])

                        max_p = ti.max(tau_diag[0], ti.max(tau_diag[1], tau_diag[2]))
                        eff_str = tensile_str * ti.max(0.0, 1.0 - dmg / softening_dmg)
                        if max_p > eff_str:
                            F["damage"][p] = dmg + damage_rate
                            for d in ti.static(range(3)):
                                if tau_diag[d] > 0:
                                    tau_diag[d] = 0.0

                        PFt = U @ ti.Matrix([
                            [tau_diag[0], 0, 0],
                            [0, tau_diag[1], 0],
                            [0, 0, tau_diag[2]]
                        ]) @ U.transpose()
                        stress = -stress_scale * PFt
                    else:
                        dg_r = F["dg"][p]
                        U_r, sig_r, V_r = ti.svd(dg_r)
                        J_r = 1.0
                        for d in ti.static(range(3)):
                            J_r *= sig_r[d, d]
                        new_F_r = ti.Matrix.identity(float, 3)
                        new_F_r[0, 0] = J_r
                        F["dg"][p] = new_F_r
                        pressure = E_rubble * (1.0 - J_r)
                        stress = pressure * stress_scale * ti.Matrix.identity(float, 3)
                else:
                    dg = F["dg"][p]
                    U, sig, V = ti.svd(dg)
                    J = 1.0
                    for d in ti.static(range(3)):
                        J *= sig[d, d]
                    R = U @ V.transpose()
                    stress_tensor = 2.0 * mu_car * (dg - R) @ dg.transpose() + la_car * J * (J - 1.0) * ti.Matrix.identity(float, 3)
                    stress = -stress_scale * stress_tensor

                affine = stress * dt + F["mass"][p] * F["C"][p]

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (float(offset) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    idx = base + offset
                    if 0 <= idx[0] < n_grid and 0 <= idx[1] < n_grid and 0 <= idx[2] < n_grid:
                        F["grid_v"][idx] += weight * (F["mass"][p] * F["v"][p] + affine @ dpos)
                        F["grid_m"][idx] += weight * F["mass"][p]

            # Grid update (gravity + BCs)
            for I in ti.grouped(F["grid_m"]):
                if F["grid_m"][I] > 0:
                    F["grid_v"][I] /= F["grid_m"][I]
                    F["grid_v"][I][2] -= gravity * dt

                    for d in ti.static(range(3)):
                        if I[d] < bound and F["grid_v"][I][d] < 0:
                            F["grid_v"][I][d] = 0
                        if I[d] > n_grid - bound and F["grid_v"][I][d] > 0:
                            F["grid_v"][I][d] = 0

                    if I[2] <= floor_grid_z and F["grid_v"][I][2] < 0:
                        F["grid_v"][I][2] = 0

        @ti.kernel
        def substep_g2p():
            for p in range(n_particles_ref):
                if F["used"][p] == 0:
                    continue
                Xp = F["x"][p] * inv_dx
                base = int(Xp - 0.5)
                fx = Xp - float(base)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

                new_v = ti.Vector([0.0, 0.0, 0.0])
                new_C = ti.Matrix.zero(float, 3, 3)

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (float(offset) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    idx = base + offset
                    if 0 <= idx[0] < n_grid and 0 <= idx[1] < n_grid and 0 <= idx[2] < n_grid:
                        g_v = F["grid_v"][idx]
                        new_v += weight * g_v
                        new_C += 4.0 * inv_dx * inv_dx * weight * g_v.outer_product(dpos)

                if F["material"][p] == CONCRETE and F["damage"][p] < rubble_dmg:
                    new_v *= (1.0 - vel_damp)

                F["v"][p] = new_v
                F["C"][p] = new_C
                F["x"][p] += dt * new_v

                # Deactivate out-of-bounds particles
                margin = 2.0  # grid cells
                for d in ti.static(range(3)):
                    if F["x"][p][d] < -margin * dx or F["x"][p][d] > (n_grid + margin) * dx:
                        F["used"][p] = 0

        self._substep_p2g = substep_p2g_and_grid
        self._substep_g2p = substep_g2p

        # Full substep (with coupling hook)
        def full_substep():
            substep_p2g_and_grid()
            if solver_ref.swe_force_hook is not None:
                solver_ref.swe_force_hook(dt)
            substep_g2p()

        self._substep = full_substep

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_frame(self, frame_id, export_dir=None):
        out = export_dir or self.export_dir
        os.makedirs(out, exist_ok=True)

        x = self.F["x"].to_numpy()[:self.n_particles]
        mat = self.F["material"].to_numpy()[:self.n_particles]
        used = self.F["used"].to_numpy()[:self.n_particles]
        damage = self.F["damage"].to_numpy()[:self.n_particles]
        chunk_ids = self.F["chunk_id"].to_numpy()[:self.n_particles]

        mask_used = used == 1

        # Convert MPM-local → sim-local → Blender world coords
        ox_sim, oy_sim, oz_sim = self.mpm_origin
        ox_bl, oy_bl, oz_bl = C.SIM_ORIGIN

        # All solids
        mask_s = mask_used
        s_pos = x[mask_s]
        s_mat = mat[mask_s]
        s_dmg = damage[mask_s]
        s_chunk = chunk_ids[mask_s]

        if len(s_pos) > 0:
            # MPM-local → Blender world
            s_blender = s_pos.copy()
            s_blender[:, 0] += ox_sim + ox_bl
            s_blender[:, 1] += oy_sim + oy_bl
            s_blender[:, 2] += oz_sim + oz_bl
            self._write_ply(
                os.path.join(out, f"solid_{frame_id:06d}.ply"),
                s_blender, damage=s_dmg, material=s_mat, chunk_id=s_chunk
            )

    @staticmethod
    def _write_ply(path, positions, damage=None, material=None, chunk_id=None):
        n = len(positions)
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        cols = [positions]
        if damage is not None:
            header += "property float damage\n"
            cols.append(damage.reshape(-1, 1))
        if material is not None:
            header += "property float material\n"
            cols.append(material.astype(np.float32).reshape(-1, 1))
        if chunk_id is not None:
            header += "property float chunk_id\n"
            cols.append(chunk_id.astype(np.float32).reshape(-1, 1))
        header += "end_header\n"

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            arr = np.column_stack(cols).astype(np.float32)
            f.write(arr.tobytes())
