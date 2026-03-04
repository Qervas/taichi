"""
Phase 5 — Water Wheel  (Hu et al. MLS-MPM + grid-velocity rigid coupling)

Physics : MLS-MPM verbatim from demo_mpm3d_ggui.py (Hu et al. SIGGRAPH 2018)
Coupling: Option B — grid-velocity override for a kinematic rigid wheel.
          After P2G solve, every grid node inside the wheel gets
          v = ω × r  (rigid-body velocity), so water responds to the
          rotating wheel.  The wheel itself spins at a fixed ω.

Controls: SPACE=pause  R=reset  RMB=orbit  ESC=quit
"""

import taichi as ti

ti.init(arch=ti.gpu)

# ---------------------------------------------------------------------------
# Simulation parameters  (Hu et al. default 3-D settings)
# ---------------------------------------------------------------------------
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
n_particles = n_grid ** dim // 2 ** (dim - 1)
dx          = 1.0 / n_grid
p_rho       = 1.0
p_vol       = (dx * 0.5) ** 2
p_mass      = p_vol * p_rho
GRAVITY     = 9.8
bound       = 3
E           = 400.0
nu          = 0.2
mu_0        = E / (2 * (1 + nu))
lambda_0    = E * nu / ((1 + nu) * (1 - 2 * nu))

WATER = 0

# ---------------------------------------------------------------------------
# Water wheel parameters
# ---------------------------------------------------------------------------
PI           = 3.14159265358979

WHEEL_CX     = 0.50    # center X  (wheel axis runs along X)
WHEEL_CY     = 0.52    # center Y  →  bottom at y=0.24, ~3 paddles in water
WHEEL_CZ     = 0.50    # center Z
WHEEL_R_OUT  = 0.28    # outer radius
WHEEL_R_HUB  = 0.05    # hub / axle radius
WHEEL_HALF_W = 0.12    # half-extent along X axis
WHEEL_N_PAD  = 8       # number of paddles
WHEEL_PAD_HT = 0.05    # paddle half-thickness — wider = more grid nodes per paddle
WHEEL_OMEGA  = 5.5     # angular velocity  rad/s  (≈ sqrt(g/R), particles centrifuge off)

WHEEL_LINE_SEG = 24                                          # circle resolution
# Layout: 4 circles × SEG×2  +  4 spoke sets × N_PAD×2  +  axle 2
#   front/back outer rim, front/back hub,
#   front spokes, back spokes, hub laterals, rim laterals, axle
_N_WHL_VERTS   = WHEEL_LINE_SEG * 8 + WHEEL_N_PAD * 8 + 2

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------
F_x         = ti.Vector.field(dim, float, n_particles)
F_v         = ti.Vector.field(dim, float, n_particles)
F_C         = ti.Matrix.field(dim, dim, float, n_particles)
F_dg        = ti.Matrix.field(3, 3, float, n_particles)
F_Jp        = ti.field(float, n_particles)
F_materials = ti.field(int,   n_particles)
F_colors    = ti.Vector.field(4, float, n_particles)
F_used      = ti.field(int,   n_particles)

F_grid_v    = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m    = ti.field(float,            (n_grid,) * dim)

neighbour   = (3,) * dim

wheel_angle      = ti.field(float, ())
wheel_cy         = ti.field(float, ())   # runtime-adjustable wheel Y position
wheel_line_verts = ti.Vector.field(3, float, _N_WHL_VERTS)

# ---------------------------------------------------------------------------
# Wheel rigid-body geometry
# ---------------------------------------------------------------------------
@ti.func
def is_in_wheel(pos):
    """1 if the grid node at `pos` is inside the rigid wheel solid."""
    dy = pos[1] - wheel_cy[None]
    dz = pos[2] - WHEEL_CZ
    in_axial = ti.abs(pos[0] - WHEEL_CX) < WHEEL_HALF_W
    in_solid  = 0
    if in_axial:
        # Hub (central cylinder)
        if ti.sqrt(dy * dy + dz * dz) < WHEEL_R_HUB:
            in_solid = 1
        # Paddles (radial slabs)
        for k in ti.static(range(WHEEL_N_PAD)):
            pa   = wheel_angle[None] + k * (2.0 * PI / WHEEL_N_PAD)
            rd_y = ti.cos(pa);  rd_z = ti.sin(pa)   # radial direction of paddle
            ds   =  dy * rd_y + dz * rd_z            # distance ALONG  paddle
            dn   = -dy * rd_z + dz * rd_y            # distance ACROSS paddle face
            if ti.abs(dn) < WHEEL_PAD_HT and WHEEL_R_HUB < ds < WHEEL_R_OUT:
                in_solid = 1
    return in_solid


# ---------------------------------------------------------------------------
# Substep  — verbatim Hu et al. physics + wheel grid override
# ---------------------------------------------------------------------------
@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):

    # --- clear grid ---
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0

    # --- P2G ---
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp   = F_x[p] / dx
        base = int(Xp - 0.5)
        fx   = Xp - base
        w    = [0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2]

        F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]
        h = ti.exp(10 * (1.0 - F_Jp[p]))
        mu, la = mu_0 * h, lambda_0 * h
        if F_materials[p] == WATER:
            mu = 0.0

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig     = sig[d, d]
            F_Jp[p]    *= sig[d, d] / new_sig
            sig[d, d]   = new_sig
            J          *= new_sig

        if F_materials[p] == WATER:
            new_F        = ti.Matrix.identity(float, 3)
            new_F[0, 0]  = J
            F_dg[p]      = new_F

        stress = (2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose()
                  + ti.Matrix.identity(float, 3) * la * J * (J - 1))
        stress = (-dt * p_vol * 4 / dx ** 2) * stress
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos   = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass

    # --- grid update + wheel override ---
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        cond = (I < bound) & (F_grid_v[I] < 0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
        # Rigid wheel — impose v = ω × r
        # ω = (WHEEL_OMEGA, 0, 0)  →  v = (0, -ω·dz, ω·dy)
        node_pos = I.cast(float) * dx
        if is_in_wheel(node_pos):
            dy = node_pos[1] - wheel_cy[None]
            dz = node_pos[2] - WHEEL_CZ
            F_grid_v[I] = ti.Vector([0.0, -WHEEL_OMEGA * dz, WHEEL_OMEGA * dy])

    # --- G2P ---
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp   = F_x[p] / dx
        base = int(Xp - 0.5)
        fx   = Xp - base
        w    = [0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos   = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v    = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        F_v[p]  = new_v
        F_x[p] += dt * F_v[p]
        F_C[p]  = new_C

    # --- Eject particles trapped inside the wheel solid ---
    # Without this, particles co-rotate exactly with the paddle and never escape.
    # We push each trapped particle to just outside the nearest paddle face,
    # preserving its current (wheel) velocity so it flies off naturally.
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        if ti.abs(F_x[p][0] - WHEEL_CX) >= WHEEL_HALF_W:
            continue
        dy = F_x[p][1] - wheel_cy[None]
        dz = F_x[p][2] - WHEEL_CZ
        r  = ti.sqrt(dy * dy + dz * dz)
        # Hub ejection
        if r < WHEEL_R_HUB and r > 1e-6:
            scale = (WHEEL_R_HUB + dx) / r
            F_x[p][1] = wheel_cy[None] + dy * scale
            F_x[p][2] = WHEEL_CZ       + dz * scale
        elif r > 1e-6:
            # Paddle ejection: find if inside any paddle, push out the narrow face
            for k in ti.static(range(WHEEL_N_PAD)):
                pa   = wheel_angle[None] + k * (2.0 * PI / WHEEL_N_PAD)
                rd_y = ti.cos(pa);  rd_z = ti.sin(pa)
                ds   =  dy * rd_y + dz * rd_z   # along paddle (radial)
                dn   = -dy * rd_z + dz * rd_y   # across paddle face
                if ti.abs(dn) < WHEEL_PAD_HT and WHEEL_R_HUB < ds < WHEEL_R_OUT:
                    # Normal direction out of the paddle face  (-rd_z, rd_y) in YZ
                    sign  = 1.0 if dn >= 0.0 else -1.0
                    push  = sign * (WHEEL_PAD_HT + dx - ti.abs(dn))
                    F_x[p][1] += push * (-rd_z)
                    F_x[p][2] += push * ( rd_y)

    # Advance wheel angle (runs once, outside parallel loops)
    wheel_angle[None] += WHEEL_OMEGA * dt


# ---------------------------------------------------------------------------
# Initialisation helpers  (from demo_mpm3d_ggui.py)
# ---------------------------------------------------------------------------
@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        F_x[p]    = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p]   = 1
        F_dg[p]   = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p]    = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p]    = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def init_cube_vol(first_par: int, last_par: int,
                  x0: float, y0: float, z0: float,
                  xs: float, ys: float, zs: float,
                  material: int):
    for i in range(first_par, last_par):
        F_x[i]         = ti.Vector([ti.random()*xs + x0,
                                    ti.random()*ys + y0,
                                    ti.random()*zs + z0])
        F_Jp[i]        = 1
        F_dg[i]        = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i]         = ti.Vector([0.0, 0.0, 0.0])
        F_materials[i] = material
        F_used[i]      = 1


@ti.kernel
def set_colors():
    for i in range(n_particles):
        F_colors[i] = ti.Vector([0.08, 0.50, 0.80, 1.0])


def init_scene():
    set_all_unused()
    init_cube_vol(0, n_particles,
                  0.05, 0.05, 0.05,
                  0.90, 0.35, 0.90,
                  WATER)
    set_colors()
    wheel_angle[None] = 0.0
    wheel_cy[None]    = WHEEL_CY


# ---------------------------------------------------------------------------
# Wheel wireframe  (updated every frame from current wheel_angle)
# ---------------------------------------------------------------------------
@ti.kernel
def update_wheel_lines():
    angle = wheel_angle[None]
    cx = WHEEL_CX;  cy = wheel_cy[None];  cz = WHEEL_CZ
    xf = cx - WHEEL_HALF_W   # front face
    xb = cx + WHEEL_HALF_W   # back  face

    # Outer rim — front (o0) and back (o1)
    for i in range(WHEEL_LINE_SEG):
        t0 = i       * (2.0 * PI / WHEEL_LINE_SEG)
        t1 = (i + 1) * (2.0 * PI / WHEEL_LINE_SEG)
        y0 = cy + WHEEL_R_OUT * ti.cos(t0);  z0 = cz + WHEEL_R_OUT * ti.sin(t0)
        y1 = cy + WHEEL_R_OUT * ti.cos(t1);  z1 = cz + WHEEL_R_OUT * ti.sin(t1)
        o0 = WHEEL_LINE_SEG * 0
        wheel_line_verts[o0 + i*2    ] = [xf, y0, z0]
        wheel_line_verts[o0 + i*2 + 1] = [xf, y1, z1]
        o1 = WHEEL_LINE_SEG * 2
        wheel_line_verts[o1 + i*2    ] = [xb, y0, z0]
        wheel_line_verts[o1 + i*2 + 1] = [xb, y1, z1]

    # Hub circle — front (o2) and back (o3)
    for i in range(WHEEL_LINE_SEG):
        t0 = i       * (2.0 * PI / WHEEL_LINE_SEG)
        t1 = (i + 1) * (2.0 * PI / WHEEL_LINE_SEG)
        y0 = cy + WHEEL_R_HUB * ti.cos(t0);  z0 = cz + WHEEL_R_HUB * ti.sin(t0)
        y1 = cy + WHEEL_R_HUB * ti.cos(t1);  z1 = cz + WHEEL_R_HUB * ti.sin(t1)
        o2 = WHEEL_LINE_SEG * 4
        wheel_line_verts[o2 + i*2    ] = [xf, y0, z0]
        wheel_line_verts[o2 + i*2 + 1] = [xf, y1, z1]
        o3 = WHEEL_LINE_SEG * 6
        wheel_line_verts[o3 + i*2    ] = [xb, y0, z0]
        wheel_line_verts[o3 + i*2 + 1] = [xb, y1, z1]

    # Paddle spokes — front (o4) and back (o5)
    # Hub laterals: front-hub → back-hub (o6)
    # Rim laterals: front-rim → back-rim (o7)   ← the missing connections
    for k in range(WHEEL_N_PAD):
        pa  = angle + k * (2.0 * PI / WHEEL_N_PAD)
        yh  = cy + WHEEL_R_HUB * ti.cos(pa);  zh = cz + WHEEL_R_HUB * ti.sin(pa)
        yr  = cy + WHEEL_R_OUT * ti.cos(pa);  zr = cz + WHEEL_R_OUT * ti.sin(pa)
        o4  = WHEEL_LINE_SEG * 8
        wheel_line_verts[o4 + k*2    ] = [xf, yh, zh]
        wheel_line_verts[o4 + k*2 + 1] = [xf, yr, zr]
        o5  = WHEEL_LINE_SEG * 8 + WHEEL_N_PAD * 2
        wheel_line_verts[o5 + k*2    ] = [xb, yh, zh]
        wheel_line_verts[o5 + k*2 + 1] = [xb, yr, zr]
        # Hub lateral (paddle board inner edge)
        o6  = WHEEL_LINE_SEG * 8 + WHEEL_N_PAD * 4
        wheel_line_verts[o6 + k*2    ] = [xf, yh, zh]
        wheel_line_verts[o6 + k*2 + 1] = [xb, yh, zh]
        # Rim lateral (paddle board outer edge)
        o7  = WHEEL_LINE_SEG * 8 + WHEEL_N_PAD * 6
        wheel_line_verts[o7 + k*2    ] = [xf, yr, zr]
        wheel_line_verts[o7 + k*2 + 1] = [xb, yr, zr]

    # Axle (single segment connecting front to back hub centres)
    o8 = WHEEL_LINE_SEG * 8 + WHEEL_N_PAD * 8
    wheel_line_verts[o8    ] = [xf, cy, cz]
    wheel_line_verts[o8 + 1] = [xb, cy, cz]


# ---------------------------------------------------------------------------
# Floor / sky geometry  (unchanged from baseline)
# ---------------------------------------------------------------------------
FLOOR_GRID_N     = 11
_n_floor_verts   = FLOOR_GRID_N * 4
floor_line_verts = ti.Vector.field(3, float, _n_floor_verts)
floor_quad_verts   = ti.Vector.field(3, float, 4)
floor_quad_norms   = ti.Vector.field(3, float, 4)
floor_quad_indices = ti.field(int, 6)
sky_verts   = ti.Vector.field(3, float, 4)
sky_norms   = ti.Vector.field(3, float, 4)
sky_indices = ti.field(int, 6)


@ti.kernel
def init_floor_geometry():
    FY = 0.005
    floor_quad_verts[0] = [0.0, FY, 0.0];  floor_quad_verts[1] = [1.0, FY, 0.0]
    floor_quad_verts[2] = [1.0, FY, 1.0];  floor_quad_verts[3] = [0.0, FY, 1.0]
    for i in ti.static(range(4)):
        floor_quad_norms[i] = [0.0, 1.0, 0.0]
    floor_quad_indices[0] = 0; floor_quad_indices[1] = 1; floor_quad_indices[2] = 2
    floor_quad_indices[3] = 0; floor_quad_indices[4] = 2; floor_quad_indices[5] = 3

    for i in range(FLOOR_GRID_N):
        t  = ti.cast(i, float) / (FLOOR_GRID_N - 1)
        LY = FY + 0.001
        floor_line_verts[i * 2    ] = [0.0, LY, t]
        floor_line_verts[i * 2 + 1] = [1.0, LY, t]
        floor_line_verts[FLOOR_GRID_N * 2 + i * 2    ] = [t, LY, 0.0]
        floor_line_verts[FLOOR_GRID_N * 2 + i * 2 + 1] = [t, LY, 1.0]

    SZ = -0.8
    sky_verts[0] = [-3.0, -2.0, SZ];  sky_verts[1] = [ 4.0, -2.0, SZ]
    sky_verts[2] = [ 4.0,  4.0, SZ];  sky_verts[3] = [-3.0,  4.0, SZ]
    for i in ti.static(range(4)):
        sky_norms[i] = [0.0, 0.0, 1.0]
    sky_indices[0] = 0; sky_indices[1] = 1; sky_indices[2] = 2
    sky_indices[3] = 0; sky_indices[4] = 2; sky_indices[5] = 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    init_scene()
    init_floor_geometry()
    update_wheel_lines()

    window = ti.ui.Window("Phase 5 — Water Wheel  [SPACE pause  R reset  RMB orbit]",
                          (1280, 720), vsync=True)
    canvas = window.get_canvas()
    gui    = window.get_gui()
    scene  = window.get_scene()
    camera = ti.ui.Camera()

    # Side-front angle so the wheel reads as a circle, not edge-on
    camera.position(1.4, 1.0, 1.5)
    camera.lookat(0.5, 0.4, 0.5)
    camera.fov(55)

    paused = False
    grav   = [0.0, -GRAVITY, 0.0]

    print(f"\nParticles : {n_particles:,}  (all water)")
    print(f"Wheel     : center=({WHEEL_CX},{WHEEL_CY},{WHEEL_CZ})  "
          f"R={WHEEL_R_OUT}  ω={WHEEL_OMEGA} rad/s")
    print("Controls  : SPACE=pause  R=reset  RMB=orbit  ESC=quit\n")

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == ti.ui.SPACE:
                paused = not paused
            elif window.event.key == "r":
                init_scene()

        if not paused:
            for _ in range(steps):
                substep(*grav)

        update_wheel_lines()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # Sky
        scene.ambient_light((0.65, 0.75, 0.88))
        scene.mesh(sky_verts, indices=sky_indices,
                   normals=sky_norms, color=(0.68, 0.84, 0.96))

        # Floor
        scene.mesh(floor_quad_verts, indices=floor_quad_indices,
                   normals=floor_quad_norms, color=(0.48, 0.42, 0.34))
        scene.lines(floor_line_verts, color=(0.28, 0.22, 0.16), width=1.5)

        # Lights
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.6, 0.6, 0.6))
        scene.point_light(pos=(0.0, 2.0, 0.5), color=(0.3, 0.35, 0.4))

        # Water particles
        scene.particles(F_x, per_vertex_color=F_colors, radius=0.004)

        # Wheel wireframe
        scene.lines(wheel_line_verts, color=(0.70, 0.50, 0.20), width=2.0)

        canvas.scene(scene)

        with gui.sub_window("Water Wheel", 0.02, 0.02, 0.22, 0.14) as w:
            w.text("SPACE: pause/play")
            w.text("R: reset")
            w.text(f"omega: {WHEEL_OMEGA:.1f} rad/s")
            wheel_cy[None] = w.slider_float("Wheel Y", wheel_cy[None], 0.30, 0.85)

        window.show()


if __name__ == "__main__":
    main()
