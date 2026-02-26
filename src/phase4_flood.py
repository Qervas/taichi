import time
import math
import taichi as ti

ti.init(arch=ti.cuda)

# =============================================================================
# Grid: 128x64x64 (X=flow direction, Y=height, Z=river width)
# Domain: X in [0, 2.0], Y in [0, 1.0], Z in [0, 1.0]
# Uniform spacing dx = 1/64
# =============================================================================
n_grid_x, n_grid_y, n_grid_z = 128, 64, 64
dx = 1.0 / 64
inv_dx = 64.0
dt = 8e-5                 # CFL-safe for E_concrete=10000 (c=64.5, CFL_dt=1.21e-4)
SUBSTEPS = 80             # 80 * 8e-5 = 6.4e-3 s simulated per frame
GRAVITY = 9.8
bound = 3

# Material IDs
WATER = 0
CONCRETE = 1

# Shared particle volume (8 particles per cell, 2x2x2 sub-grid)
p_vol = (dx * 0.5) ** 2   # convention consistent with phase2/phase3

# =============================================================================
# Water material (weakly compressible EOS)
#
# Two-sided EOS:  stress = -4 * inv_dx^2 * p_vol * E_water * (J - 1)
# Resists both compression (J<1) and expansion (J>1).
#
# E_water = 400 keeps J within +/- 1% under hydrostatic load:
#   max pressure = rho*g*h = 1.0*9.8*0.3125 = 3.06
#   J deviation  = 3.06/400 = 0.0077  (J in [0.992, 1.008])
#
# CFL check: c = sqrt(E/rho) = 20.0, CFL_dt = 0.5*dx/c = 3.9e-4 >> 8e-5
# =============================================================================
E_water = 400.0
rho_water = 1.0
p_mass_water = p_vol * rho_water

# =============================================================================
# Concrete material (corotational elastic + Rankine tensile failure)
#
# Real concrete properties (scaled for simulation domain):
#   - Quasi-rigid under service loads (E=10000)
#   - Fails in TENSION, not shear (Rankine criterion, not Drucker-Prager)
#   - Brittle: once a crack initiates, it propagates rapidly
#   - Damage softens stiffness exponentially, causing stress redistribution
#     to neighboring particles -> cascade crack propagation
#
# CFL check: c = sqrt(E/rho) = sqrt(10000/2.4) = 64.5
#             CFL_dt = 0.5*dx/c = 1.21e-4, with dt=8e-5 margin = 1.5x
# =============================================================================
E_concrete = 10000.0
nu_concrete = 0.2
mu_concrete = E_concrete / (2.0 * (1.0 + nu_concrete))          # 4166.7
la_concrete = E_concrete * nu_concrete / (
    (1.0 + nu_concrete) * (1.0 - 2.0 * nu_concrete))            # 2777.8
rho_concrete = 2.4
p_mass_concrete = p_vol * rho_concrete

# Rankine tensile failure parameters
# Concrete is STRONG — real bridge piers withstand enormous sustained
# water pressure. The pillar must hold for seconds of sustained flood
# before cracking begins, then fail progressively (not instantly).
TENSILE_STRENGTH = 1500.0  # very high — pillar resists initial impact
DAMAGE_RATE = 0.3          # slow crack propagation (real concrete is tough)
SOFTENING_DAMAGE = 0.5     # more damage needed before stiffness drops
RUBBLE_DAMAGE = 1.5        # harder to fully disintegrate

# Rubble physics: broken concrete is still a HARD SOLID. It doesn't dissolve.
# When chunks break off and hit the ground, they bounce and stack.
# We model rubble with a J-based EOS (like heavy water) — resists compression,
# prevents interpenetration, gives solid-like collisions with ground and water.
E_RUBBLE = 1000.0           # rubble EOS stiffness (stiffer than water, hard chunks)

# =============================================================================
# Foundation anchoring
#
# Real bridge pillars are anchored to bedrock through pile foundations
# (driven piles, drilled shafts, or caissons extending 10-30m deep).
# We model this by pinning the bottom FOUNDATION_ROWS cell-rows of each
# pillar. Pinned particles have their position blended toward their
# initial anchor position, with a health value (scour_hp) that degrades
# under sustained water flow (simulating scour erosion).
# =============================================================================
FOUNDATION_ROWS = 3        # bottom 3 cell-rows per pillar are pinned

# =============================================================================
# Scour erosion parameters
#
# Scour is the #1 cause of bridge failure worldwide (FHWA, AASHTO).
# Flowing water erodes the riverbed around pile foundations,
# progressively undermining the pillar's support.
#
# Implementation: since grid velocity inside the pillar is dominated by
# concrete (nearly stationary), we probe upstream of the pillar face to
# detect water flow. The scour rate scales with water speed past the face.
#
# Timeline target: ~5s sim-time for full foundation erosion at flood speed.
# =============================================================================
SCOUR_RATE = 5e-5            # hp reduction per substep per unit excess speed
SCOUR_VELOCITY_THRESH = 0.5  # minimum water speed to cause erosion
SCOUR_PROBE_CELLS = 5        # probe this many cells upstream of pillar face

# =============================================================================
# Flood parameters
#
# Mega-disaster flood: water level ABOVE pillar tops (overtopping).
# The upstream river continuously feeds the flood through particle
# recycling + river slope body force (simulates riverbed gradient).
# =============================================================================
FLOOD_VX_INIT = 5.0         # initial flood velocity (adjustable at runtime)

# These are ti.field so kernels read updated values when user adjusts flux.
# Terminal velocity = river_accel / damping_rate ≈ flood_vx
flood_vx = ti.field(dtype=ti.f32, shape=())
river_accel = ti.field(dtype=ti.f32, shape=())

# =============================================================================
# Geometry (cell coordinates)
# =============================================================================

# Water: starts at half pillar height — river recycling builds the flood up
w_cell_lo = (4, 3, 4)
w_cell_hi = (48, 17, 60)
n_water_cells = ((w_cell_hi[0] - w_cell_lo[0])
                 * (w_cell_hi[1] - w_cell_lo[1])
                 * (w_cell_hi[2] - w_cell_lo[2]))
n_water = n_water_cells * 8

# Bridge pillars: row across Z at x ~ [1.12, 1.22]
pillar_x = (72, 78)       # 6 cells thick in flow direction
pillar_y = (3, 32)        # 29 cells tall (riverbed to above waterline)
pillar_z_list = [          # 5 pillars, 4 cells wide each
    (10, 14),
    (20, 24),
    (30, 34),
    (40, 44),
    (50, 54),
]

pillar_data = []
n_concrete = 0
for z_lo, z_hi in pillar_z_list:
    clo = (pillar_x[0], pillar_y[0], z_lo)
    chi = (pillar_x[1], pillar_y[1], z_hi)
    nc = (chi[0] - clo[0]) * (chi[1] - clo[1]) * (chi[2] - clo[2])
    pillar_data.append((clo, chi, nc))
    n_concrete += nc * 8

n_particles = n_water + n_concrete

# Foundation Y boundary: cells below this are pinned
foundation_y_hi = pillar_y[0] + FOUNDATION_ROWS  # cell index 6

# Recycling boundaries (downstream -> upstream inlet)
RECYCLE_X = (n_grid_x - 8) * dx       # ~1.875
CEILING_Y = 0.75                       # recycle water that splashes too high
INLET_X_LO = 4 * dx
INLET_X_HI = 20 * dx                  # wider inlet band for dense water column
INLET_Y_LO = 3 * dx
INLET_Y_HI = 17 * dx                  # match initial water height

# =============================================================================
# Particle fields
# =============================================================================
pos = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
vel = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
J = ti.field(dtype=ti.f32, shape=n_particles)
material = ti.field(dtype=ti.i32, shape=n_particles)
F_def = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
color_f = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
damage = ti.field(dtype=ti.f32, shape=n_particles)

# Foundation fields
is_pinned = ti.field(dtype=ti.i32, shape=n_particles)
anchor_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
scour_hp = ti.field(dtype=ti.f32, shape=n_particles)

# 3D grid (non-cubic)
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid_x, n_grid_y, n_grid_z))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid_x, n_grid_y, n_grid_z))


# =============================================================================
# Kernels
# =============================================================================

@ti.func
def quadratic_kernel(d: ti.f32) -> ti.f32:
    abs_d = ti.abs(d)
    w = 0.0
    if abs_d < 0.5:
        w = 0.75 - abs_d * abs_d
    elif abs_d < 1.5:
        t = 1.5 - abs_d
        w = 0.5 * t * t
    return w


@ti.func
def rankine_damage_check(tau_diag: ti.types.vector(3, ti.f32),
                          current_damage: ti.f32) -> ti.f32:
    """Rankine tensile failure criterion.

    Concrete fails in tension, not shear. When the max principal
    Kirchhoff stress exceeds the (damage-softened) tensile strength,
    damage accumulates. This softens the particle's stiffness, forcing
    neighboring particles to carry more load — driving crack propagation.

    Returns: damage increment for this substep.
    """
    delta_damage = 0.0
    # Effective tensile strength decreases exponentially with damage
    eff_ft = TENSILE_STRENGTH * ti.exp(-current_damage / SOFTENING_DAMAGE)
    max_principal = ti.max(tau_diag[0], ti.max(tau_diag[1], tau_diag[2]))
    if max_principal > eff_ft:
        overstress = max_principal - eff_ft
        delta_damage = DAMAGE_RATE * overstress * dt
    return delta_damage


@ti.kernel
def clear_grid():
    for i, j, k in grid_v:
        grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        grid_m[i, j, k] = 0.0


@ti.kernel
def p2g():
    for pid in range(n_particles):
        xp = pos[pid]
        vp = vel[pid]
        Cp = C[pid]
        mat = material[pid]

        mass = p_mass_water
        affine = ti.Matrix.zero(ti.f32, 3, 3)

        if mat == WATER:
            # Two-sided EOS: stress = -4 * inv_dx^2 * p_vol * E * (J - 1)
            Jp = J[pid]
            stress = -dt * 4.0 * inv_dx * inv_dx * p_vol * E_water * (Jp - 1.0)
            affine = stress * ti.Matrix.identity(ti.f32, 3) + p_mass_water * Cp
        else:
            # Corotational elastic with Rankine tensile failure
            mass = p_mass_concrete
            F = F_def[pid]
            U, sig_mat, V = ti.svd(F)

            sig_vec = ti.Vector([ti.max(sig_mat[0, 0], 0.05),
                                 ti.max(sig_mat[1, 1], 0.05),
                                 ti.max(sig_mat[2, 2], 0.05)])

            # Damage-dependent stiffness (exponential softening)
            dmg = damage[pid]
            stiffness_factor = ti.exp(-dmg / SOFTENING_DAMAGE)
            is_rubble = 0
            if dmg >= RUBBLE_DAMAGE:
                stiffness_factor = 0.0
                is_rubble = 1

            eff_mu = mu_concrete * stiffness_factor
            eff_la = la_concrete * stiffness_factor

            # Hencky strain -> Kirchhoff stress
            log_sig = ti.Vector([ti.log(sig_vec[0]),
                                 ti.log(sig_vec[1]),
                                 ti.log(sig_vec[2])])
            tr_log = log_sig.sum()
            tau_diag = (2.0 * eff_mu * log_sig
                        + eff_la * tr_log * ti.Vector([1.0, 1.0, 1.0]))

            # Rankine tensile failure check (uses full-stiffness stress
            # for damage computation, not the softened one, to avoid
            # underestimating stress in already-damaged particles)
            tau_diag_full = (2.0 * mu_concrete * log_sig
                             + la_concrete * tr_log * ti.Vector([1.0, 1.0, 1.0]))
            delta_dmg = rankine_damage_check(tau_diag_full, dmg)
            damage[pid] += delta_dmg

            if is_rubble == 1:
                # Rubble: broken concrete is still HARD. Uses J-based EOS
                # (like heavy water) for solid-like collision with ground,
                # other rubble, and water. Prevents "dissolving into ground."
                Jp = J[pid]
                stress_r = -dt * 4.0 * inv_dx * inv_dx * p_vol * E_RUBBLE * (Jp - 1.0)
                affine = stress_r * ti.Matrix.identity(ti.f32, 3) + p_mass_concrete * Cp
                F_def[pid] = ti.Matrix.identity(ti.f32, 3)
            else:
                # Elastic stress contribution
                tau_mat = ti.Matrix([[tau_diag[0], 0.0, 0.0],
                                     [0.0, tau_diag[1], 0.0],
                                     [0.0, 0.0, tau_diag[2]]])
                tau = U @ tau_mat @ U.transpose()

                Jc = sig_vec[0] * sig_vec[1] * sig_vec[2]
                stress_contrib = -dt * 4.0 * inv_dx * inv_dx * p_vol / Jc * tau
                affine = stress_contrib + p_mass_concrete * Cp

        base = ti.cast(xp * inv_dx - 0.5, ti.i32)
        fx = xp * inv_dx - base.cast(ti.f32)

        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    ni = base[0] + di
                    nj = base[1] + dj
                    nk = base[2] + dk
                    if 0 <= ni < n_grid_x and 0 <= nj < n_grid_y and 0 <= nk < n_grid_z:
                        d_gc = ti.Vector([ti.cast(di, ti.f32) - fx[0],
                                          ti.cast(dj, ti.f32) - fx[1],
                                          ti.cast(dk, ti.f32) - fx[2]])
                        w = (quadratic_kernel(d_gc[0])
                             * quadratic_kernel(d_gc[1])
                             * quadratic_kernel(d_gc[2]))
                        dpos = d_gc * dx
                        momentum = w * (mass * vp + affine @ dpos)
                        ti.atomic_add(grid_v[ni, nj, nk], momentum)
                        ti.atomic_add(grid_m[ni, nj, nk], w * mass)


@ti.kernel
def grid_ops():
    """Grid operations with physically meaningful boundaries.

    Floor  = riverbed (solid, no-penetration)
    Z sides = river banks (solid, free-slip)
    X-     = upstream inflow (constant flood velocity)
    X+, Y+ = open (no reflection — water exits freely)
    """
    for i, j, k in grid_v:
        if grid_m[i, j, k] > 1e-6:
            grid_v[i, j, k] /= grid_m[i, j, k]

            # Gravity (vertical) + river slope (horizontal)
            # River slope force drives sustained flood flow — equivalent
            # to gravity component along the riverbed gradient.
            grid_v[i, j, k][1] -= GRAVITY * dt
            grid_v[i, j, k][0] += river_accel[None] * dt

            # Floor / riverbed (solid)
            if j < bound and grid_v[i, j, k][1] < 0.0:
                grid_v[i, j, k][1] = 0.0

            # River banks (Z sides, free-slip)
            if k < bound and grid_v[i, j, k][2] < 0.0:
                grid_v[i, j, k][2] = 0.0
            if k >= n_grid_z - bound and grid_v[i, j, k][2] > 0.0:
                grid_v[i, j, k][2] = 0.0

            # Upstream wall (prevent backflow at boundary)
            if i < bound and grid_v[i, j, k][0] < 0.0:
                grid_v[i, j, k][0] = 0.0

            # Downstream (X+) and Top (Y+) = OPEN — no reflection
        else:
            grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def g2p():
    for pid in range(n_particles):
        xp = pos[pid]
        mat = material[pid]

        new_v = ti.Vector([0.0, 0.0, 0.0])
        new_C = ti.Matrix.zero(ti.f32, 3, 3)

        base = ti.cast(xp * inv_dx - 0.5, ti.i32)
        fx = xp * inv_dx - base.cast(ti.f32)

        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    ni = base[0] + di
                    nj = base[1] + dj
                    nk = base[2] + dk
                    if 0 <= ni < n_grid_x and 0 <= nj < n_grid_y and 0 <= nk < n_grid_z:
                        d_gc = ti.Vector([ti.cast(di, ti.f32) - fx[0],
                                          ti.cast(dj, ti.f32) - fx[1],
                                          ti.cast(dk, ti.f32) - fx[2]])
                        w = (quadratic_kernel(d_gc[0])
                             * quadratic_kernel(d_gc[1])
                             * quadratic_kernel(d_gc[2]))
                        dpos = d_gc * dx
                        gv = grid_v[ni, nj, nk]
                        new_v += w * gv
                        new_C += 4.0 * inv_dx * inv_dx * w * gv.outer_product(dpos)

        vel[pid] = new_v * (1.0 - 2.0 * dt)
        C[pid] = new_C * 0.999

        if mat == WATER:
            J[pid] *= 1.0 + dt * new_C.trace()
            J[pid] = ti.max(0.8, ti.min(1.2, J[pid]))
        elif damage[pid] >= RUBBLE_DAMAGE:
            # Rubble: update J (like water) for EOS pressure.
            # F stays at identity (no elastic memory). J tracks volume.
            J[pid] *= 1.0 + dt * new_C.trace()
            J[pid] = ti.max(0.8, ti.min(1.2, J[pid]))
        else:
            # Intact/damaged concrete: update deformation gradient F
            F_old = F_def[pid]
            F_def[pid] = (ti.Matrix.identity(ti.f32, 3) + dt * new_C) @ F_old

            # Concrete damage coloring:
            #   intact (grey) -> cracking (reddish) -> rubble (dark grey)
            dmg = ti.min(damage[pid] / RUBBLE_DAMAGE, 1.0)
            if dmg < 0.3:
                # Intact: light grey
                t = dmg / 0.3
                color_f[pid] = ti.Vector([0.75 - 0.10 * t,
                                          0.75 - 0.15 * t,
                                          0.73 - 0.18 * t])
            elif dmg < 0.7:
                # Cracking: grey -> reddish-brown (stress cracks visible)
                t = (dmg - 0.3) / 0.4
                color_f[pid] = ti.Vector([0.65 + 0.15 * t,
                                          0.60 - 0.25 * t,
                                          0.55 - 0.25 * t])
            else:
                # Rubble: dark reddish -> grey rubble
                t = (dmg - 0.7) / 0.3
                color_f[pid] = ti.Vector([0.80 * (1.0 - t) + 0.40 * t,
                                          0.35 * (1.0 - t) + 0.38 * t,
                                          0.30 * (1.0 - t) + 0.38 * t])

            # Foundation pinning: blend position toward anchor
            if is_pinned[pid] == 1:
                hp = scour_hp[pid]
                if hp > 0.01:
                    ap = anchor_pos[pid]
                    # Clamp blend to [0,1] — body particles have hp=999
                    blend = ti.min(hp, 1.0)
                    new_pos = blend * ap + (1.0 - blend) * (xp + new_v * dt)
                    pos[pid] = new_pos
                    vel[pid] = vel[pid] * (1.0 - blend)
                    C[pid] = C[pid] * (1.0 - blend)
                    F_def[pid] = (blend * ti.Matrix.identity(ti.f32, 3)
                                  + (1.0 - blend) * F_def[pid])
                else:
                    # Fully scoured: release from foundation
                    is_pinned[pid] = 0

        # Advect (skip for pinned particles — handled above)
        if not (mat == CONCRETE and is_pinned[pid] == 1 and scour_hp[pid] > 0.01):
            pos[pid] += new_v * dt

        # Clamp to valid grid range (prevent out-of-bounds interpolation)
        lo = (bound + 0.5) * dx
        pos[pid][0] = ti.max(lo, ti.min((n_grid_x - bound - 0.5) * dx, pos[pid][0]))
        pos[pid][1] = ti.max(lo, ti.min((n_grid_y - bound - 0.5) * dx, pos[pid][1]))
        pos[pid][2] = ti.max(lo, ti.min((n_grid_z - bound - 0.5) * dx, pos[pid][2]))


@ti.kernel
def scour_update():
    """Erode foundation particles based on water velocity UPSTREAM of pillar.

    Grid velocity inside the pillar is dominated by concrete (near zero).
    To detect water flow, we probe at a point UPSTREAM of the pillar face
    (shifted -X by SCOUR_PROBE_CELLS). This samples the approaching water
    velocity, which drives scour erosion.

    Physical basis: HEC-18 (FHWA) scour equations relate local scour depth
    to approach velocity, water depth, and pier geometry.
    """
    # Probe X position: upstream of pillar face
    probe_x = (72 - SCOUR_PROBE_CELLS) * dx   # 5 cells upstream of pillar face

    for pid in range(n_particles):
        if (material[pid] == CONCRETE and is_pinned[pid] == 1
                and scour_hp[pid] > 0.0 and scour_hp[pid] <= 1.0):
            xp = pos[pid]
            # Probe point: upstream face exterior, same Y and Z as particle
            probe = ti.Vector([probe_x, xp[1], xp[2]])
            base = ti.cast(probe * inv_dx - 0.5, ti.i32)
            fx = probe * inv_dx - base.cast(ti.f32)

            local_v = ti.Vector([0.0, 0.0, 0.0])
            total_w = 0.0
            for di in range(3):
                for dj in range(3):
                    for dk in range(3):
                        ni = base[0] + di
                        nj = base[1] + dj
                        nk = base[2] + dk
                        if 0 <= ni < n_grid_x and 0 <= nj < n_grid_y and 0 <= nk < n_grid_z:
                            d_gc = ti.Vector([ti.cast(di, ti.f32) - fx[0],
                                              ti.cast(dj, ti.f32) - fx[1],
                                              ti.cast(dk, ti.f32) - fx[2]])
                            w = (quadratic_kernel(d_gc[0])
                                 * quadratic_kernel(d_gc[1])
                                 * quadratic_kernel(d_gc[2]))
                            local_v += w * grid_v[ni, nj, nk]
                            total_w += w

            if total_w > 1e-6:
                speed = local_v.norm()
                if speed > SCOUR_VELOCITY_THRESH:
                    scour_hp[pid] -= SCOUR_RATE * (speed - SCOUR_VELOCITY_THRESH)
                    scour_hp[pid] = ti.max(0.0, scour_hp[pid])


@ti.kernel
def recycle_particles():
    """Teleport downstream water particles back to upstream inlet.

    Conserves total particle count = conserves mass. Creates infinite flood.
    Particles fill the full water column height at the wider inlet zone,
    creating a dense, sustained flood wall.
    """
    for pid in range(n_particles):
        if material[pid] == WATER and (pos[pid][0] > RECYCLE_X
                                          or pos[pid][1] > CEILING_Y):
            pos[pid][0] = INLET_X_LO + ti.random() * (INLET_X_HI - INLET_X_LO)
            pos[pid][1] = INLET_Y_LO + ti.random() * (INLET_Y_HI - INLET_Y_LO)
            vel[pid] = ti.Vector([flood_vx[None], 0.0, 0.0])
            C[pid] = ti.Matrix.zero(ti.f32, 3, 3)
            J[pid] = 1.0


@ti.kernel
def init_block(offset: ti.i32,
               clo_x: ti.i32, clo_y: ti.i32, clo_z: ti.i32,
               chi_x: ti.i32, chi_y: ti.i32, chi_z: ti.i32,
               mat: ti.i32, cr: ti.f32, cg: ti.f32, cb: ti.f32,
               init_vx: ti.f32, init_vy: ti.f32, init_vz: ti.f32,
               pin_y_hi: ti.i32):
    """Initialize a rectangular block of particles.

    For concrete particles with cj < pin_y_hi, mark as pinned (foundation).
    """
    ny = chi_y - clo_y
    nz = chi_z - clo_z
    for ci in range(clo_x, chi_x):
        for cj in range(clo_y, chi_y):
            for ck in range(clo_z, chi_z):
                for si in range(2):
                    for sj in range(2):
                        for sk in range(2):
                            idx = offset + (((ci - clo_x) * ny * nz * 8)
                                            + ((cj - clo_y) * nz * 8)
                                            + ((ck - clo_z) * 8)
                                            + si * 4 + sj * 2 + sk)
                            p = ti.Vector([(ci + 0.25 + 0.5 * si) * dx,
                                           (cj + 0.25 + 0.5 * sj) * dx,
                                           (ck + 0.25 + 0.5 * sk) * dx])
                            pos[idx] = p
                            vel[idx] = ti.Vector([init_vx, init_vy, init_vz])
                            C[idx] = ti.Matrix.zero(ti.f32, 3, 3)
                            J[idx] = 1.0
                            material[idx] = mat
                            F_def[idx] = ti.Matrix.identity(ti.f32, 3)
                            color_f[idx] = ti.Vector([cr, cg, cb])
                            damage[idx] = 0.0
                            # Foundation anchoring
                            if mat == CONCRETE:
                                # ALL concrete particles are pinned (quasi-rigid body).
                                # Foundation rows get scour_hp=1.0 (erodible later).
                                # Body rows get scour_hp=999.0 (invincible for now).
                                is_pinned[idx] = 1
                                anchor_pos[idx] = p
                                if cj < pin_y_hi:
                                    scour_hp[idx] = 1.0   # foundation: scourable
                                else:
                                    scour_hp[idx] = 999.0  # body: invincible
                            else:
                                is_pinned[idx] = 0
                                anchor_pos[idx] = p
                                scour_hp[idx] = 0.0


def init_scene():
    """Initialize flood water (flowing) + 5 concrete bridge pillars with foundations."""
    # Set adjustable flood parameters
    flood_vx[None] = FLOOD_VX_INIT
    river_accel[None] = FLOOD_VX_INIT * 2.0   # terminal velocity ≈ accel / damping_rate

    # Water — starts with flood velocity, no foundation pinning
    init_block(0,
               w_cell_lo[0], w_cell_lo[1], w_cell_lo[2],
               w_cell_hi[0], w_cell_hi[1], w_cell_hi[2],
               WATER, 0.4, 0.7, 1.0,
               FLOOD_VX_INIT, 0.0, 0.0,
               0)    # pin_y_hi=0 -> no pinning for water

    # Concrete pillars — stationary, with foundation anchoring
    offset = n_water
    for clo, chi, nc in pillar_data:
        init_block(offset,
                   clo[0], clo[1], clo[2], chi[0], chi[1], chi[2],
                   CONCRETE, 0.75, 0.75, 0.73,
                   0.0, 0.0, 0.0,
                   foundation_y_hi)
        offset += nc * 8


def substep():
    clear_grid()
    p2g()
    grid_ops()
    g2p()
    scour_update()
    recycle_particles()


def main():
    window = ti.ui.Window("Phase 4: Flood vs Concrete Bridge", (1280, 720), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.6, 0.7, 2.0)
    camera.lookat(1.1, 0.15, 0.5)
    camera.fov(55)

    init_scene()

    n_foundation = FOUNDATION_ROWS * (pillar_x[1] - pillar_x[0]) * len(pillar_z_list) * 4 * 8
    water_h = (w_cell_hi[1] - w_cell_lo[1]) * dx
    pillar_h = (pillar_y[1] - pillar_y[0]) * dx
    print("=" * 60)
    print("Phase 4: Flood vs Concrete Bridge (MLS-MPM)")
    print("=" * 60)
    print(f"Water:      {n_water:,} particles  (depth={water_h:.2f})")
    print(f"Concrete:   {n_concrete:,} particles (height={pillar_h:.2f}, E={E_concrete})")
    print(f"Foundation: {n_foundation:,} pinned ({FOUNDATION_ROWS} rows)")
    print(f"Total:      {n_particles:,} particles")
    print(f"Water overtops pillars by {water_h - pillar_h:.3f} units")
    print(f"Failure:    Rankine tensile ft={TENSILE_STRENGTH}, rubble at damage={RUBBLE_DAMAGE}")
    print()
    print("Controls:")
    print("  SPACE     Play / Pause")
    print("  UP/DOWN   Increase / Decrease flood velocity (+/-0.5)")
    print("  R         Reset simulation")
    print("  RMB drag  Orbit camera")
    print("  ESC       Quit")
    print("=" * 60)

    paused = True
    frame = 0
    t0 = time.perf_counter()

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == ti.ui.SPACE:
                paused = not paused
            elif window.event.key == "r":
                init_scene()
                paused = True
                frame = 0
            elif window.event.key == ti.ui.UP:
                v = min(flood_vx[None] + 0.5, 10.0)
                flood_vx[None] = v
                river_accel[None] = v * 2.0
                print(f"  >> Flood Vx = {v:.1f}  (slope accel = {v*2:.1f})")
            elif window.event.key == ti.ui.DOWN:
                v = max(flood_vx[None] - 0.5, 0.0)
                flood_vx[None] = v
                river_accel[None] = v * 2.0
                print(f"  >> Flood Vx = {v:.1f}  (slope accel = {v*2:.1f})")

        if not paused:
            for _ in range(SUBSTEPS):
                substep()
            frame += 1

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(1.0, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(1.0, 1.5, 1.5), color=(0.5, 0.5, 0.5))

        scene.particles(pos, radius=0.003, per_vertex_color=color_f)

        canvas.scene(scene)
        window.show()

        if not paused and frame % 60 == 0 and frame > 0:
            dt_wall = time.perf_counter() - t0
            fps = 60.0 / dt_wall
            t0 = time.perf_counter()
            sim_time = frame * SUBSTEPS * dt
            print(f"frame={frame}  sim_t={sim_time:.2f}s  fps={fps:.1f}")


if __name__ == "__main__":
    main()
