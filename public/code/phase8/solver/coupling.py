"""
SWE → MPM coupling: realistic flood forces + soaking damage.

Force terms (physically accurate, no artificial amplification):
  1. Buoyancy (Archimedes, vertical)
  2. Hydrodynamic drag (velocity blending, horizontal)
  3. Wall pressure (hydrostatic + dynamic, horizontal — real ρgh + ½ρv²)

Soaking damage (the main destruction mechanism):
  Prolonged submersion weakens concrete over time. This models water
  infiltration, foundation erosion, and material degradation. The base
  of the building slowly loses strength → upper floors collapse under
  gravity → progressive/pancake collapse (not explosion).
"""

import taichi as ti

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def build_coupling_kernel(mpm_grid_v, mpm_grid_m, swe_h, swe_hu, swe_hv,
                          swe_wall, n_grid_mpm, dx_mpm, mpm_origin,
                          swe_nx, swe_ny, swe_dx, floor_z):
    """Build a Taichi kernel that applies SWE forces to MPM grid nodes."""
    g = C.GRAVITY
    rho_w = C.COUPLING["rho_water"]
    C_drag = C.COUPLING["C_drag"]
    wall_mult = C.COUPLING["wall_pressure_mult"]
    rho_solid = C.CONCRETE["rho"]
    eps_h = 1e-6

    ox, oy, oz = mpm_origin

    @ti.kernel
    def apply_swe_forces(dt: float):
        for I in ti.grouped(mpm_grid_m):
            if mpm_grid_m[I] <= 0.0:
                continue

            gx = I[0] * dx_mpm + ox
            gy = I[1] * dx_mpm + oy
            gz = I[2] * dx_mpm + oz

            fi = gx / swe_dx - 0.5
            fj = gy / swe_dx - 0.5
            ic = int(ti.round(fi))
            jc = int(ti.round(fj))
            ic = ti.max(0, ti.min(ic, swe_nx - 1))
            jc = ti.max(0, ti.min(jc, swe_ny - 1))

            at_wall = swe_wall[ic, jc] == 1

            # --- Gather water state ---
            h_interp = 0.0
            hu_interp = 0.0
            hv_interp = 0.0

            if not at_wall:
                # Standard bilinear interpolation
                i0 = int(ti.floor(fi))
                j0 = int(ti.floor(fj))
                i1 = i0 + 1
                j1 = j0 + 1
                i0 = ti.max(0, ti.min(i0, swe_nx - 1))
                i1 = ti.max(0, ti.min(i1, swe_nx - 1))
                j0 = ti.max(0, ti.min(j0, swe_ny - 1))
                j1 = ti.max(0, ti.min(j1, swe_ny - 1))
                wx = ti.max(0.0, ti.min(1.0, fi - float(i0)))
                wy = ti.max(0.0, ti.min(1.0, fj - float(j0)))

                h_interp = (swe_h[i0, j0] * (1 - wx) * (1 - wy) +
                            swe_h[i1, j0] * wx * (1 - wy) +
                            swe_h[i0, j1] * (1 - wx) * wy +
                            swe_h[i1, j1] * wx * wy)
                hu_interp = (swe_hu[i0, j0] * (1 - wx) * (1 - wy) +
                             swe_hu[i1, j0] * wx * (1 - wy) +
                             swe_hu[i0, j1] * (1 - wx) * wy +
                             swe_hu[i1, j1] * wx * wy)
                hv_interp = (swe_hv[i0, j0] * (1 - wx) * (1 - wy) +
                             swe_hv[i0, j1] * (1 - wx) * wy +
                             swe_hv[i1, j0] * wx * (1 - wy) +
                             swe_hv[i1, j1] * wx * wy)

            # --- Forces for non-wall cells: buoyancy + drag ---
            if not at_wall and h_interp > eps_h:
                water_z = floor_z + h_interp
                if gz < water_z:
                    submersion_frac = ti.min((water_z - gz) / h_interp, 1.0)

                    # 1. Buoyancy
                    a_buoy = (rho_w / rho_solid) * g * submersion_frac
                    mpm_grid_v[I][2] += a_buoy * dt

                    # 2. Drag
                    u_water = hu_interp / h_interp
                    v_water = hv_interp / h_interp
                    alpha = C_drag * submersion_frac
                    blend = ti.min(alpha * dt, 0.2)
                    mpm_grid_v[I][0] += blend * (u_water - mpm_grid_v[I][0])
                    mpm_grid_v[I][1] += blend * (v_water - mpm_grid_v[I][1])

            # --- Forces for wall cells: real hydrostatic + dynamic pressure ---
            if at_wall:
                pressure_ax = 0.0
                pressure_ay = 0.0
                best_h = 0.0
                best_hu = 0.0
                best_hv = 0.0

                # Neighbor (-1, 0) → force pushes in +X
                ni = ic - 1
                if 0 <= ni < swe_nx:
                    if swe_wall[ni, jc] == 0 and swe_h[ni, jc] > eps_h:
                        h_adj = swe_h[ni, jc]
                        local_h = ti.max(0.0, (floor_z + h_adj) - gz)
                        hydro = rho_w * g * local_h
                        u_a = swe_hu[ni, jc] / h_adj
                        v_a = swe_hv[ni, jc] / h_adj
                        dyn = 0.5 * rho_w * (u_a * u_a + v_a * v_a)
                        pressure_ax += wall_mult * (hydro + dyn)
                        if h_adj > best_h:
                            best_h = h_adj
                            best_hu = swe_hu[ni, jc]
                            best_hv = swe_hv[ni, jc]

                # Neighbor (+1, 0) → force pushes in -X
                ni = ic + 1
                if 0 <= ni < swe_nx:
                    if swe_wall[ni, jc] == 0 and swe_h[ni, jc] > eps_h:
                        h_adj = swe_h[ni, jc]
                        local_h = ti.max(0.0, (floor_z + h_adj) - gz)
                        hydro = rho_w * g * local_h
                        u_a = swe_hu[ni, jc] / h_adj
                        v_a = swe_hv[ni, jc] / h_adj
                        dyn = 0.5 * rho_w * (u_a * u_a + v_a * v_a)
                        pressure_ax -= wall_mult * (hydro + dyn)
                        if h_adj > best_h:
                            best_h = h_adj
                            best_hu = swe_hu[ni, jc]
                            best_hv = swe_hv[ni, jc]

                # Neighbor (0, -1) → force pushes in +Y
                nj = jc - 1
                if 0 <= nj < swe_ny:
                    if swe_wall[ic, nj] == 0 and swe_h[ic, nj] > eps_h:
                        h_adj = swe_h[ic, nj]
                        local_h = ti.max(0.0, (floor_z + h_adj) - gz)
                        hydro = rho_w * g * local_h
                        u_a = swe_hu[ic, nj] / h_adj
                        v_a = swe_hv[ic, nj] / h_adj
                        dyn = 0.5 * rho_w * (u_a * u_a + v_a * v_a)
                        pressure_ay += wall_mult * (hydro + dyn)
                        if h_adj > best_h:
                            best_h = h_adj
                            best_hu = swe_hu[ic, nj]
                            best_hv = swe_hv[ic, nj]

                # Neighbor (0, +1) → force pushes in -Y
                nj = jc + 1
                if 0 <= nj < swe_ny:
                    if swe_wall[ic, nj] == 0 and swe_h[ic, nj] > eps_h:
                        h_adj = swe_h[ic, nj]
                        local_h = ti.max(0.0, (floor_z + h_adj) - gz)
                        hydro = rho_w * g * local_h
                        u_a = swe_hu[ic, nj] / h_adj
                        v_a = swe_hv[ic, nj] / h_adj
                        dyn = 0.5 * rho_w * (u_a * u_a + v_a * v_a)
                        pressure_ay -= wall_mult * (hydro + dyn)
                        if h_adj > best_h:
                            best_h = h_adj
                            best_hu = swe_hu[ic, nj]
                            best_hv = swe_hv[ic, nj]

                # Apply wall pressure — clamped to physical range
                max_dv = 5.0  # m/s per substep (no explosive ejection)
                dvx = ti.max(-max_dv, ti.min(max_dv, (pressure_ax / rho_solid) * dt))
                dvy = ti.max(-max_dv, ti.min(max_dv, (pressure_ay / rho_solid) * dt))
                mpm_grid_v[I][0] += dvx
                mpm_grid_v[I][1] += dvy

                # Buoyancy + drag at wall using best adjacent water
                if best_h > eps_h:
                    water_z = floor_z + best_h
                    if gz < water_z:
                        sub_frac = ti.min((water_z - gz) / best_h, 1.0)
                        mpm_grid_v[I][2] += (rho_w / rho_solid) * g * sub_frac * dt

                        u_w = best_hu / best_h
                        v_w = best_hv / best_h
                        blend = ti.min(C_drag * sub_frac * dt, 0.2)
                        mpm_grid_v[I][0] += blend * (u_w - mpm_grid_v[I][0])
                        mpm_grid_v[I][1] += blend * (v_w - mpm_grid_v[I][1])

    return apply_swe_forces


def build_soaking_kernel(particle_x, particle_damage, particle_material,
                         particle_used, n_particles,
                         swe_h, swe_wall, swe_nx, swe_ny, swe_dx,
                         mpm_origin, floor_z):
    """Build a kernel that applies soaking damage to submerged concrete.

    This is the main destruction mechanism: water floods the ground floor
    of the building. ALL concrete below the flood line gets soaked, not
    just the outer wall — water enters through doors, windows, gaps.
    When the base weakens enough → gravity-driven collapse.

    Uses a two-pass approach:
    1. Find max water height adjacent to the building (flood level)
    2. Apply soaking to ALL concrete particles below that level
    """
    soaking_rate = C.COUPLING["soaking_rate"]
    rubble_dmg = C.CONCRETE["rubble_damage"]
    CONCRETE_MAT = 1  # material ID

    ox, oy, oz = mpm_origin

    # Scalar field for the effective flood level (computed per frame)
    flood_level = ti.field(float, ())

    @ti.kernel
    def compute_flood_level():
        """Find the max water height adjacent to any wall cell.
        This is the flood level that water reaches inside the building."""
        flood_level[None] = 0.0
        for i, j in swe_h:
            if swe_wall[i, j] == 1:
                # Check if any neighbor has water
                for di, dj in ti.static([(-1,0),(1,0),(0,-1),(0,1)]):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < swe_nx and 0 <= nj < swe_ny:
                        if swe_wall[ni, nj] == 0 and swe_h[ni, nj] > 0.1:
                            ti.atomic_max(flood_level[None], swe_h[ni, nj])

    @ti.kernel
    def apply_soaking(dt_frame: float):
        """Apply soaking damage for one frame's worth of time."""
        fl = flood_level[None]
        water_z = floor_z + fl

        for p in range(n_particles):
            if particle_used[p] == 0:
                continue
            if particle_material[p] != CONCRETE_MAT:
                continue
            if particle_damage[p] >= rubble_dmg:
                continue
            if fl < 0.1:
                continue

            # Particle Z in sim-local coords
            pz = particle_x[p][2] + oz

            # ALL concrete below flood level gets soaked
            if pz < water_z:
                # Deeper = faster soaking (more water pressure, more erosion)
                sub_frac = ti.min((water_z - pz) / fl, 1.0)
                particle_damage[p] += soaking_rate * sub_frac * dt_frame

    def soaking_step(dt_frame):
        compute_flood_level()
        apply_soaking(dt_frame)

    return soaking_step
