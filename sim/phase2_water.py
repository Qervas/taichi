import time
import taichi as ti

ti.init(arch=ti.cuda)

# =========================
# Config
# =========================
NX, NY = 128, 128
DT = 1.0 / 60.0
SUBSTEPS = 24          # more substeps for two-sided EOS CFL stability
SUB_DT = DT / SUBSTEPS

# Gravity in grid-units/s².
GRAVITY = 600.0

# =========================
# Material: weakly compressible water
#
# Two-sided EOS (same as official Taichi mpm88/mpm99):
#   pressure = K * (1 - J)
# Resists both compression AND expansion.
# Surface particles (J>1) feel tension → pulled back → no free-fall.
# This prevents wall circulation and preserves volume naturally.
# =========================
BULK_MODULUS = 4e5     # lower than before — two-sided EOS has stricter CFL

# Particle seeding: 4 per cell (2×2 jittered)
PARTICLES_PER_CELL = 4
p_vol = 1.0 / PARTICLES_PER_CELL       # initial volume per particle (dx²/4)
p_rho = 1.0                             # reference density
p_mass = p_vol * p_rho                  # mass per particle

# =========================
# Particles
#
# Each particle carries:
#   pos  — position in grid coordinates
#   vel  — velocity (grid-units/s)
#   C    — affine velocity gradient (2×2 matrix)
#   J    — volume ratio = det(deformation gradient F)
#
# C is the MLS-MPM version of APIC's B matrix. It captures the local
# velocity gradient, preserving angular momentum during transfer.
#
# J tracks how much a particle's neighborhood has compressed or expanded.
# J=1 means original volume, J<1 means compressed, J>1 means expanded.
# This single scalar replaces the entire pressure solve from our old code.
# =========================
MAX_PARTICLES = 300_000
num_particles = ti.field(dtype=ti.i32, shape=())
pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=MAX_PARTICLES)
J = ti.field(dtype=ti.f32, shape=MAX_PARTICLES)

# =========================
# Collocated grid
#
# Unlike Phase 1's MAC staggered grid (u at vertical faces, v at horizontal faces),
# MLS-MPM uses a simple collocated grid: both velocity components at each node.
# Grid nodes live at integer coordinates (0,0) to (NX-1, NY-1).
#
# Each node accumulates momentum and mass during P2G,
# then we divide to get velocity, add gravity, enforce boundaries.
# That's it — no pressure solve, no divergence, no cell classification.
# =========================
grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(NX, NY))
grid_m = ti.field(dtype=ti.f32, shape=(NX, NY))

# Rendering buffer (4x upscaled)
img = ti.Vector.field(3, dtype=ti.f32, shape=(NX * 4, NY * 4))


# =========================
# Quadratic B-spline kernel (same as before)
# Support width = 3 cells, C1 continuous.
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


@ti.kernel
def clear_grid():
    for i, j in grid_v:
        grid_v[i, j] = ti.Vector([0.0, 0.0])
        grid_m[i, j] = 0.0


# =========================
# P2G (Particle to Grid) — the heart of MLS-MPM
#
# For each particle, we scatter to the 3×3 grid neighborhood:
#   momentum_i += w * (m * v  +  affine @ dpos)
#   mass_i     += w * m
#
# The affine matrix combines two things:
#   1. APIC velocity extrapolation: m * C * dpos  (preserves angular momentum)
#   2. Stress force: dt * (-4 * V₀ * K * (J-1)) * I * dpos  (equation of state)
#
# This is the key MLS-MPM insight: stress forces are folded directly into
# the P2G scatter, eliminating the need for a separate pressure solve.
# =========================
@ti.kernel
def p2g(dt: ti.f32):
    for pid in range(num_particles[None]):
        xp = pos[pid]
        vp = vel[pid]
        Cp = C[pid]
        Jp = J[pid]

        # Two-sided EOS: resists BOTH compression (J<1) AND expansion (J>1).
        # Same as official mpm88.py — surface particles feel tension pulling
        # them back, preventing free-fall and wall circulation.
        # Also naturally conserves volume (expansion is resisted).
        stress = -4.0 * p_vol * BULK_MODULUS * (Jp - 1.0)

        # Affine = APIC momentum + stress
        affine = p_mass * Cp + dt * stress * ti.Matrix.identity(ti.f32, 2)

        base = ti.cast(ti.floor(xp - 0.5), ti.i32)
        for di in range(3):
            for dj in range(3):
                ni = base[0] + di
                nj = base[1] + dj
                if 0 <= ni < NX and 0 <= nj < NY:
                    dpos = ti.Vector([ti.cast(ni, ti.f32),
                                      ti.cast(nj, ti.f32)]) - xp
                    w = quadratic_kernel(dpos[0]) * quadratic_kernel(dpos[1])
                    momentum = w * (p_mass * vp + affine @ dpos)
                    ti.atomic_add(grid_v[ni, nj], momentum)
                    ti.atomic_add(grid_m[ni, nj], w * p_mass)


# =========================
# Grid operations: normalize, gravity, boundary
#
# This replaces ALL of the old code's:
#   normalize_grid + add_gravity + mark_cell_type + apply_boundary +
#   compute_divergence + pressure_zero + 120×GS_red + 120×GS_black + project
# with a single simple kernel.
# =========================
@ti.kernel
def grid_ops(dt: ti.f32):
    for i, j in grid_v:
        if grid_m[i, j] > 1e-6:
            # Momentum → velocity
            grid_v[i, j] /= grid_m[i, j]

            # Gravity
            grid_v[i, j][1] -= GRAVITY * dt

            # Free-slip boundary: zero normal velocity near walls
            # "bound" matches the kernel support so particles never
            # interact with nodes outside the domain.
            bound = 3
            if i < bound and grid_v[i, j][0] < 0.0:
                grid_v[i, j][0] = 0.0
            if i >= NX - bound and grid_v[i, j][0] > 0.0:
                grid_v[i, j][0] = 0.0
            if j < bound and grid_v[i, j][1] < 0.0:
                grid_v[i, j][1] = 0.0
            if j >= NY - bound and grid_v[i, j][1] > 0.0:
                grid_v[i, j][1] = 0.0


# =========================
# G2P (Grid to Particle)
#
# Gather velocity and reconstruct the affine matrix C.
# Then update J (volume ratio) using the velocity divergence:
#   J_new = J_old * (1 + dt * trace(C))
#
# trace(C) = ∂vx/∂x + ∂vy/∂y = divergence of velocity.
# If flow diverges (trace > 0), volume grows (J increases).
# If flow converges (trace < 0), volume shrinks (J decreases).
#
# The factor 4.0 in the C computation comes from the quadratic B-spline
# inertia tensor inverse: D⁻¹ = 4/dx² * I = 4*I for dx=1.
# =========================
@ti.kernel
def g2p(dt: ti.f32):
    for pid in range(num_particles[None]):
        xp = pos[pid]

        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix.zero(ti.f32, 2, 2)

        base = ti.cast(ti.floor(xp - 0.5), ti.i32)
        for di in range(3):
            for dj in range(3):
                ni = base[0] + di
                nj = base[1] + dj
                if 0 <= ni < NX and 0 <= nj < NY:
                    dpos = ti.Vector([ti.cast(ni, ti.f32),
                                      ti.cast(nj, ti.f32)]) - xp
                    w = quadratic_kernel(dpos[0]) * quadratic_kernel(dpos[1])
                    gv = grid_v[ni, nj]
                    new_v += w * gv
                    new_C += 4.0 * w * gv.outer_product(dpos)

        # Particle-level damping to control energy (mpm88 has none, but splashes
        # too high). Applied on particles, not grid, to avoid wall artifacts.
        vel[pid] = new_v * (1.0 - 2.0 * dt)
        C[pid] = new_C * 0.999

        # Incremental J update — two-sided EOS naturally conserves volume
        # (expansion is resisted), so no density correction needed.
        J[pid] *= 1.0 + dt * new_C.trace()

        # Advect
        pos[pid] += new_v * dt

        # Clamp inside domain (matching the boundary "bound" zone)
        lo = 3.0
        hi_x = ti.cast(NX, ti.f32) - 3.0
        hi_y = ti.cast(NY, ti.f32) - 3.0
        pos[pid][0] = ti.max(lo, ti.min(hi_x, pos[pid][0]))
        pos[pid][1] = ti.max(lo, ti.min(hi_y, pos[pid][1]))


# =========================
# Scene: dam break (left 40%, bottom 80%)
# =========================
@ti.kernel
def init_dam_break():
    water_w = ti.cast(NX * 0.4, ti.i32)
    water_h = ti.cast(NY * 0.8, ti.i32)

    count = 0
    for i in range(3, water_w):
        for j in range(3, water_h):
            for si in range(2):
                for sj in range(2):
                    idx = ((i - 3) * (water_h - 3) * 4 +
                           (j - 3) * 4 +
                           si * 2 + sj)
                    if idx < MAX_PARTICLES:
                        px = i + 0.25 + 0.5 * si
                        py = j + 0.25 + 0.5 * sj
                        pos[idx] = ti.Vector([px, py])
                        vel[idx] = ti.Vector([0.0, 0.0])
                        C[idx] = ti.Matrix.zero(ti.f32, 2, 2)
                        J[idx] = 1.0
                        ti.atomic_max(count, idx + 1)

    num_particles[None] = count


# =========================
# Rendering
# =========================
@ti.kernel
def build_image():
    SCALE = 4
    for i, j in img:
        img[i, j] = ti.Vector([0.02, 0.02, 0.05])

    for pid in range(num_particles[None]):
        px = ti.cast(pos[pid][0] * SCALE, ti.i32)
        py = ti.cast(pos[pid][1] * SCALE, ti.i32)
        if 0 <= px < NX * SCALE and 0 <= py < NY * SCALE:
            img[px, py] = ti.Vector([0.4, 0.7, 1.0])


# =========================
# Substep: the entire pipeline in 3 calls
#
# Compare to the old FLIP/APIC which needed:
#   clear → p2g → normalize → gravity → mark_cells → boundary →
#   save_velocity → divergence → pressure_zero → 120×GS → project →
#   boundary → g2p → advect
# =========================
def substep(dt):
    clear_grid()
    p2g(dt)
    grid_ops(dt)
    g2p(dt)


def main():
    gui = ti.GUI("Phase 2: MLS-MPM Water (Dam Break)", (NX * 4, NY * 4))
    init_dam_break()
    print(f"Initialized {num_particles[None]} particles")
    print("Controls: SPACE = start/pause, R = reset, ESC = quit")

    paused = True
    frame = 0
    t0 = time.perf_counter()

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
            elif e.key == "r":
                init_dam_break()
                paused = True
                frame = 0

        if not paused:
            for _ in range(SUBSTEPS):
                substep(SUB_DT)
            frame += 1

        build_image()
        gui.set_image(img)
        gui.show()

        if not paused and frame % 60 == 0 and frame > 0:
            dt_wall = time.perf_counter() - t0
            fps = 60.0 / dt_wall
            t0 = time.perf_counter()
            print(f"frame={frame} fps={fps:.1f} particles={num_particles[None]}")


if __name__ == "__main__":
    main()
