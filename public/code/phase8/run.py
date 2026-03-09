"""
Phase 8 — Building Destruction + Flood (Multi-Material MLS-MPM)

Water + concrete fracture + car coupling in one unified MPM solver.
Water uses EOS pressure (Phase 5), concrete uses corotational elastic +
Rankine tensile failure + damage softening (Phase 4), car uses kinematic
grid-velocity override.

Usage:
    python run.py                                   # interactive GGUI
    python run.py --export 900 --export-dir ./export  # headless export
    python run.py --export 30 --n_grid 64             # quick test
"""

import argparse
import json
import os
import time

import numpy as np
import taichi as ti

parser = argparse.ArgumentParser(description="Phase 8 — Building Destruction + Flood")
parser.add_argument("--export", type=int, default=0, help="Export N frames headless (0=interactive)")
parser.add_argument("--export-dir", default="./export", help="Output directory for exported frames")
parser.add_argument("--n_grid", type=int, default=128, help="Grid resolution (64=test, 128=production)")
parser.add_argument("--building", default=None, help="Path to building_particles.npy (None=box fallback)")
args = parser.parse_args()

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
dim = 3
n_grid = args.n_grid
DOMAIN = 60.0                   # 60m physical domain
dx = DOMAIN / n_grid            # 0.47m at n_grid=128
inv_dx = 1.0 / dx
dt = 8e-5                       # CFL safe for concrete c=64.5 m/s
SUBSTEPS = 50                   # per frame at 30fps
GRAVITY = 9.8
bound = 3

# Material IDs
WATER = 0
CONCRETE = 1

# ---------------------------------------------------------------------------
# Water material (weakly compressible EOS)
# ---------------------------------------------------------------------------
E_water = 400.0
rho_water = 1.0
p_vol = (dx * 0.5) ** 2
p_mass_water = p_vol * rho_water

# ---------------------------------------------------------------------------
# Concrete material (corotational elastic + Rankine tensile failure)
# ---------------------------------------------------------------------------
E_concrete = 10000.0
nu_concrete = 0.2
mu_concrete = E_concrete / (2.0 * (1.0 + nu_concrete))          # 4166.7
la_concrete = E_concrete * nu_concrete / (
    (1.0 + nu_concrete) * (1.0 - 2.0 * nu_concrete))            # 2777.8
rho_concrete = 2.4
p_mass_concrete = p_vol * rho_concrete

# Rankine tensile failure (lower strength for dramatic fracture at flood scale)
TENSILE_STRENGTH = 500.0
DAMAGE_RATE = 0.5
SOFTENING_DAMAGE = 0.5
RUBBLE_DAMAGE = 1.5
E_RUBBLE = 1000.0

# Velocity damping (prevents numerical explosion in concrete)
VEL_DAMPING = 2.0 * dt   # ~1.6e-4 per substep (from Phase 4)

# ---------------------------------------------------------------------------
# Car parameters (oriented AABB, kinematic coupling)
# ---------------------------------------------------------------------------
CAR_HALF = ti.Vector([2.25, 1.0, 0.75])  # half extents: 4.5×2.0×1.5m
CAR_MASS = 1500.0                          # kg

# ---------------------------------------------------------------------------
# Water inflow / dam-break parameters
# ---------------------------------------------------------------------------
FLOOD_VX = 5.0          # initial velocity toward building
FLOOR_Y = (bound + 1) * dx      # floor level (above boundary cells)
# Dam-break: water starts as tall column directly upstream of building
SOURCE_X_LO = (bound + 1) * dx  # ~4m from domain edge
SOURCE_X_HI = 24.0              # dam right up to building face (building at x=25)
SOURCE_Y_LO = FLOOR_Y
SOURCE_Y_HI = FLOOR_Y + 15.0   # 15m tall water column (overtops building)
SOURCE_Z_LO = 8.0
SOURCE_Z_HI = 52.0
RECYCLE_X = DOMAIN - 5.0 * dx
CEILING_Y = DOMAIN - 5.0 * dx
RIVER_ACCEL = 3.0               # sustained slope-driven acceleration (m/s^2)

# ---------------------------------------------------------------------------
# Generate building particles (box fallback or loaded from .npy)
# ---------------------------------------------------------------------------
def generate_box_building():
    """Create a simple rectangular building from grid particles for testing.
    Building: ~25×40×20m centered in domain.
    """
    spacing = dx * 0.5   # denser particle packing for structural integrity
    # Solid block building: 12×12×10m sitting on the floor
    bx0, bx1 = 25.0, 37.0
    by0, by1 = FLOOR_Y, FLOOR_Y + 12.0
    bz0, bz1 = 25.0, 35.0

    xs = np.arange(bx0 + spacing / 2, bx1, spacing)
    ys = np.arange(by0 + spacing / 2, by1, spacing)
    zs = np.arange(bz0 + spacing / 2, bz1, spacing)

    # Solid block (every position inside is a particle)
    positions = []
    for x in xs:
        for y in ys:
            for z in zs:
                positions.append([x, y, z])

    positions = np.array(positions, dtype=np.float32)
    print(f"  Box building: {len(positions)} particles, "
          f"bounds [{bx0},{by0},{bz0}]-[{bx1},{by1},{bz1}]m")

    # Assign Voronoi chunk IDs (random seed points)
    n_chunks = 30
    rng = np.random.default_rng(42)
    seeds = rng.uniform([bx0, by0, bz0], [bx1, by1, bz1], size=(n_chunks, 3)).astype(np.float32)
    from scipy.spatial import cKDTree
    tree = cKDTree(seeds)
    _, chunk_ids = tree.query(positions)

    return positions, chunk_ids.astype(np.int32)


def load_building_particles():
    """Load building particles from .npy files or generate box fallback."""
    if args.building and os.path.exists(args.building):
        positions = np.load(args.building).astype(np.float32)
        chunk_path = args.building.replace("building_particles", "building_chunk_ids")
        if os.path.exists(chunk_path):
            chunk_ids = np.load(chunk_path).astype(np.int32)
        else:
            chunk_ids = np.zeros(len(positions), dtype=np.int32)
        print(f"  Loaded building: {len(positions)} particles from {args.building}")
        return positions, chunk_ids
    else:
        print("  No building .npy provided, using box fallback")
        return generate_box_building()


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------
building_pos, building_chunks = load_building_particles()
n_building = len(building_pos)

# Water particles: fill source region
water_spacing = dx * 0.5
wx = np.arange(SOURCE_X_LO + water_spacing / 2, SOURCE_X_HI, water_spacing)
wy = np.arange(SOURCE_Y_LO + water_spacing / 2, SOURCE_Y_HI, water_spacing)
wz = np.arange(SOURCE_Z_LO + water_spacing / 2, SOURCE_Z_HI, water_spacing)
water_grid = np.array(np.meshgrid(wx, wy, wz, indexing='ij')).reshape(3, -1).T.astype(np.float32)
n_water = len(water_grid)

# Extra buffer for recycled particles
n_particles = n_water + n_building
MAX_PARTICLES = n_particles + 50000  # headroom

print(f"\nPhase 8 — Building Destruction + Flood")
print(f"  Domain:    {DOMAIN}m, grid {n_grid}^3, dx={dx:.3f}m")
print(f"  Water:     {n_water:,} particles")
print(f"  Building:  {n_building:,} particles")
print(f"  Total:     {n_particles:,} particles (max {MAX_PARTICLES:,})")
print(f"  dt={dt:.1e}, substeps={SUBSTEPS}, flood_vx={FLOOD_VX}")
print()

# ---------------------------------------------------------------------------
# Taichi fields
# ---------------------------------------------------------------------------
F_x = ti.Vector.field(dim, float, MAX_PARTICLES)
F_v = ti.Vector.field(dim, float, MAX_PARTICLES)
F_C = ti.Matrix.field(dim, dim, float, MAX_PARTICLES)
F_dg = ti.Matrix.field(3, 3, float, MAX_PARTICLES)   # deformation gradient
F_Jp = ti.field(float, MAX_PARTICLES)                 # J for water/rubble EOS
F_material = ti.field(int, MAX_PARTICLES)
F_damage = ti.field(float, MAX_PARTICLES)
F_chunk_id = ti.field(int, MAX_PARTICLES)
F_mass = ti.field(float, MAX_PARTICLES)
F_used = ti.field(int, MAX_PARTICLES)

F_grid_v = ti.Vector.field(dim, float, (n_grid, n_grid, n_grid))
F_grid_m = ti.field(float, (n_grid, n_grid, n_grid))

F_colors = ti.Vector.field(4, float, MAX_PARTICLES)

# Car state (position, velocity, yaw)
car_pos = ti.Vector.field(3, float, ())
car_vel = ti.Vector.field(3, float, ())
car_yaw = ti.field(float, ())

# Runtime-adjustable flood speed
flood_vx = ti.field(float, ())

neighbour = (3,) * dim


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
@ti.kernel
def set_all_unused():
    for p in range(MAX_PARTICLES):
        F_used[p] = 0
        F_x[p] = ti.Vector([999999.0, 999999.0, 999999.0])
        F_Jp[p] = 1.0
        F_dg[p] = ti.Matrix.identity(float, 3)
        F_C[p] = ti.Matrix.zero(float, 3, 3)
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])
        F_damage[p] = 0.0
        F_chunk_id[p] = 0
        F_material[p] = WATER
        F_mass[p] = p_mass_water
        F_colors[p] = ti.Vector([0.08, 0.50, 0.80, 1.0])


@ti.kernel
def init_water_particles(positions: ti.types.ndarray()):
    for i in range(positions.shape[0]):
        F_x[i] = ti.Vector([positions[i, 0], positions[i, 1], positions[i, 2]])
        F_v[i] = ti.Vector([flood_vx[None], 0.0, 0.0])
        F_Jp[i] = 1.0
        F_dg[i] = ti.Matrix.identity(float, 3)
        F_C[i] = ti.Matrix.zero(float, 3, 3)
        F_material[i] = WATER
        F_mass[i] = p_mass_water
        F_used[i] = 1
        F_damage[i] = 0.0
        F_chunk_id[i] = 0
        F_colors[i] = ti.Vector([0.08, 0.50, 0.80, 1.0])


@ti.kernel
def init_building_particles(offset: int,
                            positions: ti.types.ndarray(),
                            chunk_ids: ti.types.ndarray()):
    for i in range(positions.shape[0]):
        idx = offset + i
        F_x[idx] = ti.Vector([positions[i, 0], positions[i, 1], positions[i, 2]])
        F_v[idx] = ti.Vector([0.0, 0.0, 0.0])
        F_Jp[idx] = 1.0
        F_dg[idx] = ti.Matrix.identity(float, 3)
        F_C[idx] = ti.Matrix.zero(float, 3, 3)
        F_material[idx] = CONCRETE
        F_mass[idx] = p_mass_concrete
        F_used[idx] = 1
        F_damage[idx] = 0.0
        F_chunk_id[idx] = chunk_ids[i]
        # Concrete color: light grey
        F_colors[idx] = ti.Vector([0.75, 0.75, 0.73, 1.0])


def init_scene():
    set_all_unused()
    flood_vx[None] = FLOOD_VX
    init_water_particles(water_grid)
    init_building_particles(n_water, building_pos, building_chunks)

    # Car initial position: between water source and building
    car_pos[None] = ti.Vector([30.0, FLOOR_Y + CAR_HALF[1], 22.0])
    car_vel[None] = ti.Vector([0.0, 0.0, 0.0])
    car_yaw[None] = 0.0


# ---------------------------------------------------------------------------
# Rankine tensile failure criterion
# ---------------------------------------------------------------------------
@ti.func
def rankine_damage_check(tau_diag: ti.types.vector(3, float),
                          current_damage: float) -> float:
    delta_damage = 0.0
    eff_ft = TENSILE_STRENGTH * ti.exp(-current_damage / SOFTENING_DAMAGE)
    max_principal = ti.max(tau_diag[0], ti.max(tau_diag[1], tau_diag[2]))
    if max_principal > eff_ft:
        overstress = max_principal - eff_ft
        delta_damage = DAMAGE_RATE * overstress * dt
    return delta_damage


# ---------------------------------------------------------------------------
# Car collision test
# ---------------------------------------------------------------------------
@ti.func
def is_in_car(pos: ti.types.vector(3, float)) -> int:
    """Test if a grid node is inside the car AABB (axis-aligned, no yaw for now)."""
    cp = car_pos[None]
    inside = 1
    for d in ti.static(range(3)):
        if ti.abs(pos[d] - cp[d]) > CAR_HALF[d]:
            inside = 0
    return inside


# ---------------------------------------------------------------------------
# Substep kernel — unified multi-material MPM
# ---------------------------------------------------------------------------
@ti.kernel
def substep():
    # --- Clear grid ---
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0.0

    # --- P2G ---
    for p in range(MAX_PARTICLES):
        if F_used[p] == 0:
            continue
        xp = F_x[p] / dx
        base = int(xp - 0.5)
        fx = xp - base
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1) ** 2,
             0.5 * (fx - 0.5) ** 2]

        mat = F_material[p]
        mass = F_mass[p]
        affine = ti.Matrix.zero(float, 3, 3)

        if mat == WATER:
            # Water: EOS pressure, mu=0
            # Update J via deformation gradient trace
            F_Jp[p] *= 1.0 + dt * F_C[p].trace()
            F_Jp[p] = ti.max(0.8, ti.min(1.2, F_Jp[p]))
            Jp = F_Jp[p]
            stress = -dt * 4.0 * inv_dx * inv_dx * p_vol * E_water * (Jp - 1.0)
            affine = stress * ti.Matrix.identity(float, 3) + mass * F_C[p]

        elif mat == CONCRETE:
            dmg = F_damage[p]
            is_rubble = 0
            if dmg >= RUBBLE_DAMAGE:
                is_rubble = 1

            if is_rubble == 1:
                # Rubble: J-based EOS (like heavy water)
                F_Jp[p] *= 1.0 + dt * F_C[p].trace()
                F_Jp[p] = ti.max(0.8, ti.min(1.2, F_Jp[p]))
                Jp = F_Jp[p]
                stress_r = -dt * 4.0 * inv_dx * inv_dx * p_vol * E_RUBBLE * (Jp - 1.0)
                affine = stress_r * ti.Matrix.identity(float, 3) + mass * F_C[p]
                F_dg[p] = ti.Matrix.identity(float, 3)
            else:
                # Corotational elastic + Rankine failure
                F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]
                F_old = F_dg[p]
                U, sig_mat, V = ti.svd(F_old)

                sig_vec = ti.Vector([ti.max(sig_mat[0, 0], 0.05),
                                     ti.max(sig_mat[1, 1], 0.05),
                                     ti.max(sig_mat[2, 2], 0.05)])

                # Damage-dependent stiffness
                stiffness_factor = ti.exp(-dmg / SOFTENING_DAMAGE)

                eff_mu = mu_concrete * stiffness_factor
                eff_la = la_concrete * stiffness_factor

                # Hencky strain -> Kirchhoff stress
                log_sig = ti.Vector([ti.log(sig_vec[0]),
                                     ti.log(sig_vec[1]),
                                     ti.log(sig_vec[2])])
                tr_log = log_sig.sum()
                tau_diag = (2.0 * eff_mu * log_sig
                            + eff_la * tr_log * ti.Vector([1.0, 1.0, 1.0]))

                # Rankine check with FULL stiffness stress
                tau_diag_full = (2.0 * mu_concrete * log_sig
                                 + la_concrete * tr_log * ti.Vector([1.0, 1.0, 1.0]))
                delta_dmg = rankine_damage_check(tau_diag_full, dmg)
                F_damage[p] += delta_dmg

                # Stress contribution
                tau_mat = ti.Matrix([[tau_diag[0], 0.0, 0.0],
                                     [0.0, tau_diag[1], 0.0],
                                     [0.0, 0.0, tau_diag[2]]])
                tau = U @ tau_mat @ U.transpose()
                Jc = sig_vec[0] * sig_vec[1] * sig_vec[2]
                stress_contrib = -dt * 4.0 * inv_dx * inv_dx * p_vol / Jc * tau
                affine = stress_contrib + mass * F_C[p]

                # Update concrete color based on damage
                dmg_frac = ti.min(F_damage[p] / RUBBLE_DAMAGE, 1.0)
                if dmg_frac < 0.3:
                    t = dmg_frac / 0.3
                    F_colors[p] = ti.Vector([0.75 - 0.10 * t,
                                             0.75 - 0.15 * t,
                                             0.73 - 0.18 * t, 1.0])
                elif dmg_frac < 0.7:
                    t = (dmg_frac - 0.3) / 0.4
                    F_colors[p] = ti.Vector([0.65 + 0.15 * t,
                                             0.60 - 0.25 * t,
                                             0.55 - 0.25 * t, 1.0])
                else:
                    t = (dmg_frac - 0.7) / 0.3
                    F_colors[p] = ti.Vector([0.80 * (1.0 - t) + 0.40 * t,
                                             0.35 * (1.0 - t) + 0.38 * t,
                                             0.30 * (1.0 - t) + 0.38 * t, 1.0])

        # Scatter to grid
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * mass

    # --- Grid update ---
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]

        # Gravity + river slope acceleration (pushes water downstream)
        # Only apply slope if below terminal velocity to prevent runaway
        F_grid_v[I][1] -= GRAVITY * dt
        if F_grid_v[I][0] < flood_vx[None] * 2.0:
            F_grid_v[I][0] += RIVER_ACCEL * dt

        # Boundary conditions — per-axis, with X+ OPEN for water exit
        # Floor (Y-)
        if I[1] < bound and F_grid_v[I][1] < 0:
            F_grid_v[I][1] = 0.0
        # Ceiling (Y+)
        if I[1] > n_grid - bound and F_grid_v[I][1] > 0:
            F_grid_v[I][1] = 0.0
        # Upstream wall (X-): no-penetration
        if I[0] < bound and F_grid_v[I][0] < 0:
            F_grid_v[I][0] = 0.0
        # Downstream (X+): OPEN — water exits freely (no clamping)
        # Z walls: no-penetration
        if I[2] < bound and F_grid_v[I][2] < 0:
            F_grid_v[I][2] = 0.0
        if I[2] > n_grid - bound and F_grid_v[I][2] > 0:
            F_grid_v[I][2] = 0.0

        # Car kinematic coupling
        node_pos = I.cast(float) * dx
        if is_in_car(node_pos):
            F_grid_v[I] = car_vel[None]

    # --- G2P ---
    for p in range(MAX_PARTICLES):
        if F_used[p] == 0:
            continue
        xp = F_x[p] / dx
        base = int(xp - 0.5)
        fx = xp - base
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1) ** 2,
             0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4.0 * inv_dx * inv_dx * weight * g_v.outer_product(dpos)
        # Velocity damping (concrete only — stabilizes structure)
        if F_material[p] == CONCRETE:
            F_v[p] = new_v * (1.0 - VEL_DAMPING)
            F_C[p] = new_C * 0.999
        else:
            F_v[p] = new_v
            F_C[p] = new_C
        F_x[p] += dt * F_v[p]

        # Clamp to domain (X+ open — no upper clamp on X)
        lo = (bound + 0.5) * dx
        hi_y = (n_grid - bound - 0.5) * dx
        hi_z = (n_grid - bound - 0.5) * dx
        F_x[p][0] = ti.max(lo, F_x[p][0])  # X-: wall, X+: open
        F_x[p][1] = ti.max(lo, ti.min(hi_y, F_x[p][1]))
        F_x[p][2] = ti.max(lo, ti.min(hi_z, F_x[p][2]))


# ---------------------------------------------------------------------------
# Water particle recycling (downstream → upstream source)
# ---------------------------------------------------------------------------
@ti.kernel
def recycle_water():
    for p in range(MAX_PARTICLES):
        if F_used[p] == 0 or F_material[p] != WATER:
            continue
        if F_x[p][0] > RECYCLE_X or F_x[p][1] > CEILING_Y:
            F_x[p][0] = SOURCE_X_LO + ti.random() * (SOURCE_X_HI - SOURCE_X_LO)
            F_x[p][1] = SOURCE_Y_LO + ti.random() * (SOURCE_Y_HI - SOURCE_Y_LO)
            F_x[p][2] = SOURCE_Z_LO + ti.random() * (SOURCE_Z_HI - SOURCE_Z_LO)
            F_v[p] = ti.Vector([flood_vx[None], 0.0, 0.0])
            F_C[p] = ti.Matrix.zero(float, 3, 3)
            F_Jp[p] = 1.0


# ---------------------------------------------------------------------------
# Car dynamics (simple momentum transfer from grid)
# ---------------------------------------------------------------------------
@ti.kernel
def update_car():
    """Sum momentum from grid nodes inside car AABB → integrate car motion."""
    total_force = ti.Vector([0.0, 0.0, 0.0])
    n_nodes = 0
    cp = car_pos[None]
    cv = car_vel[None]

    for I in ti.grouped(F_grid_m):
        node_pos = I.cast(float) * dx
        if is_in_car(node_pos) and F_grid_m[I] > 0:
            # Force = grid momentum that differs from car velocity
            dv = F_grid_v[I] - cv
            total_force += F_grid_m[I] * dv / dt
            n_nodes += 1

    # Simple Euler integration with damping
    if n_nodes > 0:
        accel = total_force / CAR_MASS
        car_vel[None] += accel * dt * 0.1  # scale down coupling
    car_vel[None][1] -= GRAVITY * dt  # gravity on car

    # Ground collision (floor at FLOOR_Y)
    floor = (bound + 1.0) * dx
    if car_pos[None][1] < floor + CAR_HALF[1] and car_vel[None][1] < 0:
        car_vel[None][1] = 0.0
        car_pos[None][1] = floor + CAR_HALF[1]

    # Friction
    car_vel[None] *= 0.999

    car_pos[None] += car_vel[None] * dt

    # Clamp car to domain
    lo = (bound + 1.0) * dx + CAR_HALF[0]
    hi = (n_grid - bound - 1.0) * dx - CAR_HALF[0]
    car_pos[None][0] = ti.max(lo, ti.min(hi, car_pos[None][0]))
    car_pos[None][2] = ti.max(lo, ti.min(hi, car_pos[None][2]))


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_frame(frame, out_dir):
    """Export water PLY, solid PLY, car NPZ for one frame."""
    pos_np = F_x.to_numpy()[:n_particles]
    vel_np = F_v.to_numpy()[:n_particles]
    used_np = F_used.to_numpy()[:n_particles]
    mat_np = F_material.to_numpy()[:n_particles]
    dmg_np = F_damage.to_numpy()[:n_particles]
    chunk_np = F_chunk_id.to_numpy()[:n_particles]

    # Water particles
    water_mask = (used_np == 1) & (mat_np == WATER)
    w_pos = pos_np[water_mask]
    w_vel = vel_np[water_mask]
    if len(w_pos) > 0:
        writer = ti.tools.PLYWriter(num_vertices=len(w_pos))
        writer.add_vertex_pos(w_pos[:, 0], w_pos[:, 1], w_pos[:, 2])
        writer.add_vertex_channel("vx", "float", w_vel[:, 0])
        writer.add_vertex_channel("vy", "float", w_vel[:, 1])
        writer.add_vertex_channel("vz", "float", w_vel[:, 2])
        writer.export_frame(frame, os.path.join(out_dir, "water"))

    # Solid particles (concrete: intact + rubble)
    solid_mask = (used_np == 1) & (mat_np == CONCRETE)
    s_pos = pos_np[solid_mask]
    s_dmg = dmg_np[solid_mask]
    s_chunk = chunk_np[solid_mask]
    if len(s_pos) > 0:
        # Encode damage as red channel, chunk_id as green for rendering
        dmg_norm = np.clip(s_dmg / RUBBLE_DAMAGE, 0, 1).astype(np.float32)
        chunk_norm = (s_chunk % 256).astype(np.float32) / 255.0
        writer = ti.tools.PLYWriter(num_vertices=len(s_pos))
        writer.add_vertex_pos(s_pos[:, 0], s_pos[:, 1], s_pos[:, 2])
        writer.add_vertex_channel("damage", "float", dmg_norm)
        writer.add_vertex_channel("chunk_id", "float", chunk_norm)
        writer.export_frame(frame, os.path.join(out_dir, "solid"))

    # Car
    cp = car_pos.to_numpy()
    cv = car_vel.to_numpy()
    cy = car_yaw.to_numpy()
    np.savez(os.path.join(out_dir, f"car_{frame:06d}.npz"),
             position=cp, velocity=cv, yaw=cy)


def export_scene_meta(out_dir):
    """Write scene metadata JSON."""
    meta = {
        "domain": DOMAIN,
        "n_grid": n_grid,
        "dx": dx,
        "dt": dt,
        "substeps": SUBSTEPS,
        "fps": 30,
        "n_water": n_water,
        "n_building": n_building,
        "n_particles": n_particles,
        "car_half_extents": CAR_HALF.to_list(),
        "flood_vx": FLOOD_VX,
    }
    with open(os.path.join(out_dir, "scene_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_scene()

    # --- Headless export mode ---
    if args.export > 0:
        out = os.path.abspath(args.export_dir)
        os.makedirs(out, exist_ok=True)
        export_scene_meta(out)
        print(f"\nExporting {args.export} frames to {out}/")
        t0 = time.perf_counter()

        for frame in range(args.export):
            for _ in range(SUBSTEPS):
                substep()
                update_car()
            recycle_water()
            export_frame(frame, out)

            if frame % 10 == 0:
                elapsed = time.perf_counter() - t0
                sim_time = (frame + 1) * SUBSTEPS * dt
                print(f"  frame {frame}/{args.export}  sim_t={sim_time:.2f}s  "
                      f"wall={elapsed:.1f}s")

        print(f"\nExport complete. {args.export} frames in {out}/")
        return

    # --- Interactive GGUI ---
    window = ti.ui.Window("Phase 8 — Building Destruction + Flood",
                          (1280, 720), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Camera positioned to see the building from the flood direction
    camera.position(0.0, 25.0, 30.0)
    camera.lookat(30.0, 15.0, 30.0)
    camera.fov(55)

    paused = True
    frame = 0

    print("Controls: SPACE=pause  R=reset  UP/DOWN=flood speed  RMB=orbit  ESC=quit")

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == ti.ui.SPACE:
                paused = not paused
            elif window.event.key == "r":
                init_scene()
                frame = 0
            elif window.event.key == ti.ui.UP:
                v = min(flood_vx[None] + 1.0, 20.0)
                flood_vx[None] = v
                print(f"  >> Flood Vx = {v:.1f}")
            elif window.event.key == ti.ui.DOWN:
                v = max(flood_vx[None] - 1.0, 0.0)
                flood_vx[None] = v
                print(f"  >> Flood Vx = {v:.1f}")

        if not paused:
            for _ in range(SUBSTEPS):
                substep()
                update_car()
            recycle_water()
            frame += 1

        camera.track_user_inputs(window, movement_speed=0.5, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(30.0, 50.0, 30.0), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(0.0, 30.0, 60.0), color=(0.3, 0.35, 0.4))

        scene.particles(F_x, per_vertex_color=F_colors, radius=dx * 0.3)

        canvas.scene(scene)

        with gui.sub_window("Phase 8", 0.02, 0.02, 0.22, 0.20) as w:
            w.text(f"Frame: {frame}")
            w.text(f"Particles: {n_particles:,}")
            w.text(f"Flood Vx: {flood_vx[None]:.1f} m/s")
            w.text("SPACE: pause/play")
            w.text("R: reset")
            w.text("UP/DOWN: flood speed")

        window.show()

        if not paused and frame % 60 == 0 and frame > 0:
            sim_time = frame * SUBSTEPS * dt
            print(f"  frame={frame}  sim_t={sim_time:.2f}s")


if __name__ == "__main__":
    main()
