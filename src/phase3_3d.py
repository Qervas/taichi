import time
import math
import taichi as ti

ti.init(arch=ti.cuda)

# =========================
# Config — normalized coordinates [0,1]³
# =========================
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4
SUBSTEPS = 50
GRAVITY = 9.8
bound = 3

# Material IDs
WATER = 0
SAND = 1

# Shared particle volume
p_vol = (dx * 0.5) ** 2

# Water material (weakly compressible EOS)
E = 0.1                # bulk modulus (scaled for inv_dx² discretization)
p_rho = 1.0
p_mass_water = p_vol * p_rho

# Structure material (Drucker-Prager elastoplastic)
# --- Tune COHESION to control shattering ---
#   COHESION=0.05  → very strong (holds against water)
#   COHESION=0.01  → moderate (shatters under sustained pressure)
#   COHESION=0.003 → weak (shatters quickly)
E_sand = 400.0
nu_sand = 0.2
mu_sand = E_sand / (2.0 * (1.0 + nu_sand))
la_sand = E_sand * nu_sand / ((1.0 + nu_sand) * (1.0 - 2.0 * nu_sand))
rho_sand = 2.0
p_mass_sand = p_vol * rho_sand
friction_angle = 45.0
alpha_dp = math.sqrt(2.0 / 3.0) * 2.0 * math.sin(math.radians(friction_angle)) / (
    3.0 - math.sin(math.radians(friction_angle))
)
COHESION = 0.01
SHATTER_DAMAGE = 1.0   # accumulated yield strain for full cohesion loss

# =========================
# Geometry (cell coordinates for precision)
# =========================

# Water block — wide river dam break
w_cell_lo = (3, 3, 3)      # ≈ (0.05, 0.05, 0.05)
w_cell_hi = (20, 35, 60)   # ≈ (0.31, 0.55, 0.94)
n_water_cells = ((w_cell_hi[0] - w_cell_lo[0])
                 * (w_cell_hi[1] - w_cell_lo[1])
                 * (w_cell_hi[2] - w_cell_lo[2]))
n_water = n_water_cells * 8

# 4 bridge pillars — rectangular columns blocking the water
pillar_x = (30, 35)     # 5 cells thick, x ≈ [0.47, 0.55]
pillar_y = (3, 30)      # 27 cells tall, y ≈ [0.05, 0.47]
pillar_z_list = [        # each 5 cells wide, evenly spaced
    (7, 12),             # z ≈ [0.11, 0.19]
    (21, 26),            # z ≈ [0.33, 0.41]
    (35, 40),            # z ≈ [0.55, 0.63]
    (49, 54),            # z ≈ [0.77, 0.84]
]

pillar_data = []
n_sand = 0
for z_lo, z_hi in pillar_z_list:
    clo = (pillar_x[0], pillar_y[0], z_lo)
    chi = (pillar_x[1], pillar_y[1], z_hi)
    nc = (chi[0] - clo[0]) * (chi[1] - clo[1]) * (chi[2] - clo[2])
    pillar_data.append((clo, chi, nc))
    n_sand += nc * 8
n_pillars = len(pillar_z_list)

n_particles = n_water + n_sand

# =========================
# Particle fields
# =========================
pos = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
vel = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
J = ti.field(dtype=ti.f32, shape=n_particles)
material = ti.field(dtype=ti.i32, shape=n_particles)
F_def = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
color_f = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
damage = ti.field(dtype=ti.f32, shape=n_particles)

# 3D collocated grid
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))


# =========================
# Kernels
# =========================

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
def sand_projection(sig_vec: ti.types.vector(3, ti.f32), cohesion: ti.f32):
    """Drucker-Prager return mapping. Returns (projected_sig, delta_gamma)."""
    epsilon = ti.Vector([ti.log(ti.max(sig_vec[0], 1e-10)),
                         ti.log(ti.max(sig_vec[1], 1e-10)),
                         ti.log(ti.max(sig_vec[2], 1e-10))])
    tr = epsilon.sum()

    new_sig = sig_vec
    yielded = 0.0

    epsilon_hat = epsilon - (tr / 3.0) * ti.Vector([1.0, 1.0, 1.0])
    epsilon_hat_norm = ti.sqrt(epsilon_hat.dot(epsilon_hat) + 1e-20)
    delta_gamma = (epsilon_hat_norm
                   + (3.0 * la_sand + 2.0 * mu_sand) / (2.0 * mu_sand) * tr * alpha_dp
                   - cohesion)

    if tr >= 0.0 and delta_gamma > 0.0:
        # Tensile failure → shatter (F reset to identity)
        new_sig = ti.Vector([1.0, 1.0, 1.0])
        yielded = delta_gamma
    elif delta_gamma > 0.0:
        # Compressive yield — project back onto yield surface
        epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
        new_sig = ti.Vector([ti.exp(epsilon[0]),
                             ti.exp(epsilon[1]),
                             ti.exp(epsilon[2])])
        yielded = delta_gamma

    return new_sig, yielded


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
            Jp = J[pid]
            stress = -dt * 4.0 * inv_dx * inv_dx * p_vol * E * (Jp - 1.0)
            affine = stress * ti.Matrix.identity(ti.f32, 3) + p_mass_water * Cp
        else:
            # Structure: SVD-based Drucker-Prager with damage
            mass = p_mass_sand
            F = F_def[pid]
            U, sig_mat, V = ti.svd(F)

            # Clamp singular values to prevent degenerate stress
            sig_vec = ti.Vector([ti.max(sig_mat[0, 0], 0.05),
                                 ti.max(sig_mat[1, 1], 0.05),
                                 ti.max(sig_mat[2, 2], 0.05)])

            # Damage-dependent cohesion: degrades from COHESION → 0
            eff_cohesion = ti.max(0.0, COHESION * (1.0 - damage[pid] / SHATTER_DAMAGE))
            sig_vec, delta_g = sand_projection(sig_vec, eff_cohesion)

            # Accumulate damage from yield events
            damage[pid] += delta_g

            # Reconstruct F with projected singular values
            sig_new = ti.Matrix([[sig_vec[0], 0.0, 0.0],
                                 [0.0, sig_vec[1], 0.0],
                                 [0.0, 0.0, sig_vec[2]]])
            F_def[pid] = U @ sig_new @ V.transpose()

            # Hencky-strain Kirchhoff stress
            log_sig = ti.Vector([ti.log(sig_vec[0]),
                                 ti.log(sig_vec[1]),
                                 ti.log(sig_vec[2])])
            tr_log = log_sig.sum()
            tau_diag = 2.0 * mu_sand * log_sig + la_sand * tr_log * ti.Vector([1.0, 1.0, 1.0])
            tau_mat = ti.Matrix([[tau_diag[0], 0.0, 0.0],
                                 [0.0, tau_diag[1], 0.0],
                                 [0.0, 0.0, tau_diag[2]]])
            tau = U @ tau_mat @ U.transpose()

            Jc = sig_vec[0] * sig_vec[1] * sig_vec[2]
            stress_contrib = -dt * 4.0 * inv_dx * inv_dx * p_vol / Jc * tau
            affine = stress_contrib + p_mass_sand * Cp

        base = ti.cast(xp * inv_dx - 0.5, ti.i32)
        fx = xp * inv_dx - base.cast(ti.f32)

        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    ni = base[0] + di
                    nj = base[1] + dj
                    nk = base[2] + dk
                    if 0 <= ni < n_grid and 0 <= nj < n_grid and 0 <= nk < n_grid:
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
    for i, j, k in grid_v:
        if grid_m[i, j, k] > 1e-6:
            grid_v[i, j, k] /= grid_m[i, j, k]

            # Gravity (Y is up)
            grid_v[i, j, k][1] -= GRAVITY * dt

            # Free-slip boundary on all 6 faces
            if i < bound and grid_v[i, j, k][0] < 0.0:
                grid_v[i, j, k][0] = 0.0
            if i >= n_grid - bound and grid_v[i, j, k][0] > 0.0:
                grid_v[i, j, k][0] = 0.0
            if j < bound and grid_v[i, j, k][1] < 0.0:
                grid_v[i, j, k][1] = 0.0
            if j >= n_grid - bound and grid_v[i, j, k][1] > 0.0:
                grid_v[i, j, k][1] = 0.0
            if k < bound and grid_v[i, j, k][2] < 0.0:
                grid_v[i, j, k][2] = 0.0
            if k >= n_grid - bound and grid_v[i, j, k][2] > 0.0:
                grid_v[i, j, k][2] = 0.0
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
                    if 0 <= ni < n_grid and 0 <= nj < n_grid and 0 <= nk < n_grid:
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
        else:
            # Update deformation gradient
            F_old = F_def[pid]
            F_def[pid] = (ti.Matrix.identity(ti.f32, 3) + dt * new_C) @ F_old

            # Damage coloring: brown (intact) → dark red (cracking) → grey (shattered)
            dmg = ti.min(damage[pid] / SHATTER_DAMAGE, 1.0)
            if dmg < 0.5:
                t = dmg * 2.0
                color_f[pid] = ti.Vector([0.72 * (1.0 - t) + 0.65 * t,
                                          0.53 * (1.0 - t) + 0.20 * t,
                                          0.35 * (1.0 - t) + 0.10 * t])
            else:
                t = (dmg - 0.5) * 2.0
                color_f[pid] = ti.Vector([0.65 * (1.0 - t) + 0.40 * t,
                                          0.20 * (1.0 - t) + 0.38 * t,
                                          0.10 * (1.0 - t) + 0.38 * t])

        # Advect
        pos[pid] += new_v * dt

        # Clamp to domain
        lo = bound * dx
        hi = 1.0 - bound * dx
        pos[pid] = ti.max(lo, ti.min(hi, pos[pid]))


@ti.kernel
def init_block(offset: ti.i32,
               clo_x: ti.i32, clo_y: ti.i32, clo_z: ti.i32,
               chi_x: ti.i32, chi_y: ti.i32, chi_z: ti.i32,
               mat: ti.i32, cr: ti.f32, cg: ti.f32, cb: ti.f32):
    """Initialize a rectangular block of particles."""
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
                            pos[idx] = ti.Vector([(ci + 0.25 + 0.5 * si) * dx,
                                                   (cj + 0.25 + 0.5 * sj) * dx,
                                                   (ck + 0.25 + 0.5 * sk) * dx])
                            vel[idx] = ti.Vector([0.0, 0.0, 0.0])
                            C[idx] = ti.Matrix.zero(ti.f32, 3, 3)
                            J[idx] = 1.0
                            material[idx] = mat
                            F_def[idx] = ti.Matrix.identity(ti.f32, 3)
                            color_f[idx] = ti.Vector([cr, cg, cb])
                            damage[idx] = 0.0


def init_scene():
    """Initialize water + 4 bridge pillars."""
    # Water
    init_block(0, w_cell_lo[0], w_cell_lo[1], w_cell_lo[2],
               w_cell_hi[0], w_cell_hi[1], w_cell_hi[2],
               WATER, 0.4, 0.7, 1.0)
    # Pillars
    offset = n_water
    for clo, chi, nc in pillar_data:
        init_block(offset, clo[0], clo[1], clo[2], chi[0], chi[1], chi[2],
                   SAND, 0.72, 0.53, 0.35)
        offset += nc * 8


def substep():
    clear_grid()
    p2g()
    grid_ops()
    g2p()


def main():
    window = ti.ui.Window("Phase 3: 3D Dam Break + Bridge Pillars", (1280, 720), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.0, 0.8, 1.8)
    camera.lookat(0.4, 0.2, 0.5)
    camera.fov(55)

    init_scene()
    print(f"Water: {n_water} particles")
    print(f"Pillars: {n_sand} particles ({n_pillars} pillars, {n_sand // n_pillars} each)")
    print(f"Total: {n_particles} particles")
    print(f"Grid: {n_grid}³, dt={dt}, substeps={SUBSTEPS}")
    print(f"Structure: E={E_sand}, friction={friction_angle}°, cohesion={COHESION}")
    print(f"Shatter damage={SHATTER_DAMAGE}")
    print("Controls: SPACE=play/pause, R=reset, ESC=quit")
    print("Camera: hold RMB to orbit, scroll to zoom")

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

        if not paused:
            for _ in range(SUBSTEPS):
                substep()
            frame += 1

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

        scene.particles(pos, radius=0.003, per_vertex_color=color_f)

        canvas.scene(scene)
        window.show()

        if not paused and frame % 60 == 0 and frame > 0:
            dt_wall = time.perf_counter() - t0
            fps = 60.0 / dt_wall
            t0 = time.perf_counter()
            print(f"frame={frame} fps={fps:.1f}")


if __name__ == "__main__":
    main()
