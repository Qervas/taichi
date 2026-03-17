"""
Phase 5 — Water Wheel  (Hu et al. MLS-MPM + grid-velocity rigid coupling)

Physics : MLS-MPM verbatim from demo_mpm3d_ggui.py (Hu et al. SIGGRAPH 2018)
Coupling: Option B — grid-velocity override for a kinematic rigid wheel.
          After P2G solve, every grid node inside the wheel gets
          v = ω × r  (rigid-body velocity), so water responds to the
          rotating wheel.  The wheel itself spins at a fixed ω.

Controls: SPACE=pause  R=reset  RMB=orbit  ESC=quit
"""

import argparse
import os

import taichi as ti

parser = argparse.ArgumentParser(description="Phase 5 — Water Wheel (MLS-MPM)")
parser.add_argument("--export", type=int, default=0, help="Export N frames headless (0=disabled)")
parser.add_argument("--export-dir", default="./export", help="Output directory for exported frames")
args = parser.parse_args()

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
WHEEL_PAD_HT = 0.025   # paddle half-thickness — thinner blades (~1.6 grid cells thick)
WHEEL_OMEGA  = -5.5    # angular velocity  rad/s  (negative = clockwise from front view)

WHEEL_SHROUD_R  = 0.16   # inner radius of side shrouds (hub is 0.05, outer is 0.28)
WHEEL_SHROUD_HT = 0.015  # half-thickness of shroud boards in X direction

WHEEL_LINE_SEG = 24                                          # circle resolution
# Solid mesh: hub(SEG) + paddles(N_PAD×6) + shrouds(N_PAD×2)
WHEEL_N_FACES  = WHEEL_LINE_SEG + WHEEL_N_PAD * 6 + WHEEL_N_PAD * 2  # 88
WHEEL_NV       = WHEEL_N_FACES * 4                           # 352 verts (flat shading)
WHEEL_NI       = WHEEL_N_FACES * 6                           # 528 indices
_FG_HUB    = 0                                               # face-group base offsets
_FG_PAD    = WHEEL_LINE_SEG
_FG_SHROUD = WHEEL_LINE_SEG + WHEEL_N_PAD * 6

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
wheel_mesh_verts  = ti.Vector.field(3, float, WHEEL_NV)
wheel_mesh_norms  = ti.Vector.field(3, float, WHEEL_NV)
wheel_mesh_colors = ti.Vector.field(3, float, WHEEL_NV)
wheel_mesh_idx    = ti.field(int, WHEEL_NI)

# ---------------------------------------------------------------------------
# Frame export (PLY water + OBJ wheel)
# ---------------------------------------------------------------------------
def export_water_ply(frame, out_dir):
    """Export active water particles as binary PLY with velocity channels."""
    import numpy as np
    pos  = F_x.to_numpy()       # (n_particles, 3)
    used = F_used.to_numpy()    # (n_particles,)
    vel  = F_v.to_numpy()       # (n_particles, 3)
    mask = used == 1
    active_pos = pos[mask]
    active_vel = vel[mask]
    writer = ti.tools.PLYWriter(num_vertices=len(active_pos))
    writer.add_vertex_pos(active_pos[:, 0], active_pos[:, 1], active_pos[:, 2])
    writer.add_vertex_channel("vx", "float", active_vel[:, 0])
    writer.add_vertex_channel("vy", "float", active_vel[:, 1])
    writer.add_vertex_channel("vz", "float", active_vel[:, 2])
    writer.export_frame(frame, os.path.join(out_dir, "water"))


def export_wheel_obj(frame, out_dir):
    """Export wheel mesh as OBJ with vertex colors (v x y z r g b)."""
    import numpy as np
    verts   = wheel_mesh_verts.to_numpy()    # (352, 3)
    norms   = wheel_mesh_norms.to_numpy()    # (352, 3)
    colors  = wheel_mesh_colors.to_numpy()   # (352, 3)
    indices = wheel_mesh_idx.to_numpy()      # (528,)
    path = os.path.join(out_dir, f"wheel_{frame:06d}.obj")
    with open(path, "w") as f:
        for i in range(len(verts)):
            f.write(f"v {verts[i,0]:.6f} {verts[i,1]:.6f} {verts[i,2]:.6f} "
                    f"{colors[i,0]:.4f} {colors[i,1]:.4f} {colors[i,2]:.4f}\n")
        for i in range(len(norms)):
            f.write(f"vn {norms[i,0]:.6f} {norms[i,1]:.6f} {norms[i,2]:.6f}\n")
        for i in range(0, len(indices), 3):
            a, b, c = indices[i] + 1, indices[i+1] + 1, indices[i+2] + 1
            f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
    # Also save colors as .npy for Blender fallback
    np.save(os.path.join(out_dir, f"wheel_colors_{frame:06d}.npy"), colors)


# ---------------------------------------------------------------------------
# Procedural wood colours
# ---------------------------------------------------------------------------
@ti.func
def wood_endgrain(r_frac):
    """End-grain rings for disc faces.  r_frac: 0 = hub, 1 = rim."""
    base_r = 0.56 + 0.04 * r_frac
    base_g = 0.36 + 0.02 * r_frac
    base_b = 0.19 + 0.01 * r_frac
    ring = 0.5 + 0.5 * ti.sin(r_frac * 50.0)
    d = ring * 0.06
    return ti.Vector([base_r - d, base_g - d * 0.8, base_b - d * 0.5])


@ti.func
def wood_facegrain(u):
    """Face-grain stripes for rim / hub / paddle sides.  u: 0-1 along grain."""
    base = ti.Vector([0.54, 0.35, 0.18])
    grain = 0.5 + 0.5 * ti.sin(u * 60.0)
    d = grain * 0.05
    return ti.Vector([base[0] - d, base[1] - d * 0.8, base[2] - d * 0.5])


# ---------------------------------------------------------------------------
# Wheel rigid-body geometry
# ---------------------------------------------------------------------------
@ti.func
def is_in_wheel(pos):
    """1 if the grid node at `pos` is inside the rigid wheel solid."""
    dy = pos[1] - wheel_cy[None]
    dz = pos[2] - WHEEL_CZ
    in_axial = ti.abs(pos[0] - WHEEL_CX) < WHEEL_HALF_W
    r = ti.sqrt(dy * dy + dz * dz)
    in_solid  = 0
    if in_axial:
        # Hub (central cylinder)
        if r < WHEEL_R_HUB:
            in_solid = 1
        # Paddles (radial slabs)
        for k in ti.static(range(WHEEL_N_PAD)):
            pa   = wheel_angle[None] + k * (2.0 * PI / WHEEL_N_PAD)
            rd_y = ti.cos(pa);  rd_z = ti.sin(pa)   # radial direction of paddle
            ds   =  dy * rd_y + dz * rd_z            # distance ALONG  paddle
            dn   = -dy * rd_z + dz * rd_y            # distance ACROSS paddle face
            if ti.abs(dn) < WHEEL_PAD_HT and WHEEL_R_HUB < ds < WHEEL_R_OUT:
                in_solid = 1
    # Side shrouds — thin annular discs at wheel edges
    if WHEEL_SHROUD_R < r < WHEEL_R_OUT:
        if ti.abs(pos[0] - (WHEEL_CX - WHEEL_HALF_W)) < WHEEL_SHROUD_HT:
            in_solid = 1
        if ti.abs(pos[0] - (WHEEL_CX + WHEEL_HALF_W)) < WHEEL_SHROUD_HT:
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
            # Shroud ejection: push particles out of side disc boards
            if WHEEL_SHROUD_R < r < WHEEL_R_OUT:
                xf_pos = WHEEL_CX - WHEEL_HALF_W
                xb_pos = WHEEL_CX + WHEEL_HALF_W
                if ti.abs(F_x[p][0] - xf_pos) < WHEEL_SHROUD_HT:
                    sign = 1.0 if (F_x[p][0] - xf_pos) >= 0.0 else -1.0
                    F_x[p][0] = xf_pos + sign * (WHEEL_SHROUD_HT + dx)
                if ti.abs(F_x[p][0] - xb_pos) < WHEEL_SHROUD_HT:
                    sign = 1.0 if (F_x[p][0] - xb_pos) >= 0.0 else -1.0
                    F_x[p][0] = xb_pos + sign * (WHEEL_SHROUD_HT + dx)

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
# Wheel solid mesh  (rebuilt every frame from current wheel_angle / wheel_cy)
# ---------------------------------------------------------------------------
@ti.kernel
def init_wheel_idx():
    """One-time: fill index buffer — 2 triangles per quad."""
    for f in range(WHEEL_N_FACES):
        v0 = f * 4
        i0 = f * 6
        wheel_mesh_idx[i0 + 0] = v0;      wheel_mesh_idx[i0 + 1] = v0 + 1;  wheel_mesh_idx[i0 + 2] = v0 + 2
        wheel_mesh_idx[i0 + 3] = v0;      wheel_mesh_idx[i0 + 4] = v0 + 2;  wheel_mesh_idx[i0 + 5] = v0 + 3


@ti.kernel
def update_wheel_mesh():
    """Per-frame: rebuild vertex positions, normals, and wood colours."""
    ang = wheel_angle[None]
    cx = WHEEL_CX;  cy = wheel_cy[None];  cz = WHEEL_CZ
    xf = cx - WHEEL_HALF_W
    xb = cx + WHEEL_HALF_W

    # ---- Hub cylinder ----
    for i in range(WHEEL_LINE_SEG):
        f = _FG_HUB + i;  v = f * 4
        t0 = float(i) * (2.0 * PI / WHEEL_LINE_SEG)
        t1 = float(i + 1) * (2.0 * PI / WHEEL_LINE_SEG)
        tmid = 0.5 * (t0 + t1)
        wheel_mesh_verts[v + 0] = ti.Vector([xf, cy + WHEEL_R_HUB * ti.cos(t0), cz + WHEEL_R_HUB * ti.sin(t0)])
        wheel_mesh_verts[v + 1] = ti.Vector([xb, cy + WHEEL_R_HUB * ti.cos(t0), cz + WHEEL_R_HUB * ti.sin(t0)])
        wheel_mesh_verts[v + 2] = ti.Vector([xb, cy + WHEEL_R_HUB * ti.cos(t1), cz + WHEEL_R_HUB * ti.sin(t1)])
        wheel_mesh_verts[v + 3] = ti.Vector([xf, cy + WHEEL_R_HUB * ti.cos(t1), cz + WHEEL_R_HUB * ti.sin(t1)])
        n = ti.Vector([0.0, ti.cos(tmid), ti.sin(tmid)])
        u0 = float(i) / WHEEL_LINE_SEG
        u1 = float(i + 1) / WHEEL_LINE_SEG
        for j in ti.static(range(4)):
            wheel_mesh_norms[v + j] = n
        wheel_mesh_colors[v + 0] = wood_facegrain(u0)
        wheel_mesh_colors[v + 1] = wood_facegrain(u0)
        wheel_mesh_colors[v + 2] = wood_facegrain(u1)
        wheel_mesh_colors[v + 3] = wood_facegrain(u1)

    # ---- Paddles (6 faces each) ----
    for k in range(WHEEL_N_PAD):
        pa   = ang + float(k) * (2.0 * PI / WHEEL_N_PAD)
        rd_y = ti.cos(pa);   rd_z = ti.sin(pa)     # radial direction
        nd_y = -ti.sin(pa);  nd_z = ti.cos(pa)     # normal direction
        # 8 corners  (inner/outer × minus/plus normal × front/back)
        ihm_y = cy + WHEEL_R_HUB * rd_y - WHEEL_PAD_HT * nd_y
        ihm_z = cz + WHEEL_R_HUB * rd_z - WHEEL_PAD_HT * nd_z
        ihp_y = cy + WHEEL_R_HUB * rd_y + WHEEL_PAD_HT * nd_y
        ihp_z = cz + WHEEL_R_HUB * rd_z + WHEEL_PAD_HT * nd_z
        ohm_y = cy + WHEEL_R_OUT * rd_y - WHEEL_PAD_HT * nd_y
        ohm_z = cz + WHEEL_R_OUT * rd_z - WHEEL_PAD_HT * nd_z
        ohp_y = cy + WHEEL_R_OUT * rd_y + WHEEL_PAD_HT * nd_y
        ohp_z = cz + WHEEL_R_OUT * rd_z + WHEEL_PAD_HT * nd_z

        fb = _FG_PAD + k * 6                       # first face of this paddle

        # Face 0 — front  (x = xf)
        v = (fb + 0) * 4
        wheel_mesh_verts[v+0] = ti.Vector([xf, ihm_y, ihm_z])
        wheel_mesh_verts[v+1] = ti.Vector([xf, ohm_y, ohm_z])
        wheel_mesh_verts[v+2] = ti.Vector([xf, ohp_y, ohp_z])
        wheel_mesh_verts[v+3] = ti.Vector([xf, ihp_y, ihp_z])
        nf = ti.Vector([-1.0, 0.0, 0.0])
        for j in ti.static(range(4)):
            wheel_mesh_norms[v+j] = nf
            wheel_mesh_colors[v+j] = wood_endgrain(0.5)

        # Face 1 — back  (x = xb)
        v = (fb + 1) * 4
        wheel_mesh_verts[v+0] = ti.Vector([xb, ihm_y, ihm_z])
        wheel_mesh_verts[v+1] = ti.Vector([xb, ihp_y, ihp_z])
        wheel_mesh_verts[v+2] = ti.Vector([xb, ohp_y, ohp_z])
        wheel_mesh_verts[v+3] = ti.Vector([xb, ohm_y, ohm_z])
        nb = ti.Vector([1.0, 0.0, 0.0])
        for j in ti.static(range(4)):
            wheel_mesh_norms[v+j] = nb
            wheel_mesh_colors[v+j] = wood_endgrain(0.5)

        # Face 2 — +normal side  (dn = +PAD_HT, the "scooping" face)
        v = (fb + 2) * 4
        wheel_mesh_verts[v+0] = ti.Vector([xf, ihp_y, ihp_z])
        wheel_mesh_verts[v+1] = ti.Vector([xf, ohp_y, ohp_z])
        wheel_mesh_verts[v+2] = ti.Vector([xb, ohp_y, ohp_z])
        wheel_mesh_verts[v+3] = ti.Vector([xb, ihp_y, ihp_z])
        nn = ti.Vector([0.0, nd_y, nd_z])
        for j in ti.static(range(4)):
            wheel_mesh_norms[v+j] = nn
        wheel_mesh_colors[v+0] = wood_facegrain(0.0)
        wheel_mesh_colors[v+1] = wood_facegrain(1.0)
        wheel_mesh_colors[v+2] = wood_facegrain(1.0)
        wheel_mesh_colors[v+3] = wood_facegrain(0.0)

        # Face 3 — -normal side
        v = (fb + 3) * 4
        wheel_mesh_verts[v+0] = ti.Vector([xf, ihm_y, ihm_z])
        wheel_mesh_verts[v+1] = ti.Vector([xb, ihm_y, ihm_z])
        wheel_mesh_verts[v+2] = ti.Vector([xb, ohm_y, ohm_z])
        wheel_mesh_verts[v+3] = ti.Vector([xf, ohm_y, ohm_z])
        nm = ti.Vector([0.0, -nd_y, -nd_z])
        for j in ti.static(range(4)):
            wheel_mesh_norms[v+j] = nm
        wheel_mesh_colors[v+0] = wood_facegrain(0.0)
        wheel_mesh_colors[v+1] = wood_facegrain(0.0)
        wheel_mesh_colors[v+2] = wood_facegrain(1.0)
        wheel_mesh_colors[v+3] = wood_facegrain(1.0)

        # Face 4 — outer tip  (ds = R_OUT)
        v = (fb + 4) * 4
        wheel_mesh_verts[v+0] = ti.Vector([xf, ohm_y, ohm_z])
        wheel_mesh_verts[v+1] = ti.Vector([xb, ohm_y, ohm_z])
        wheel_mesh_verts[v+2] = ti.Vector([xb, ohp_y, ohp_z])
        wheel_mesh_verts[v+3] = ti.Vector([xf, ohp_y, ohp_z])
        no = ti.Vector([0.0, rd_y, rd_z])
        for j in ti.static(range(4)):
            wheel_mesh_norms[v+j] = no
            wheel_mesh_colors[v+j] = wood_facegrain(0.5)

        # Face 5 — inner tip  (ds = R_HUB)
        v = (fb + 5) * 4
        wheel_mesh_verts[v+0] = ti.Vector([xf, ihp_y, ihp_z])
        wheel_mesh_verts[v+1] = ti.Vector([xb, ihp_y, ihp_z])
        wheel_mesh_verts[v+2] = ti.Vector([xb, ihm_y, ihm_z])
        wheel_mesh_verts[v+3] = ti.Vector([xf, ihm_y, ihm_z])
        ni = ti.Vector([0.0, -rd_y, -rd_z])
        for j in ti.static(range(4)):
            wheel_mesh_norms[v+j] = ni
            wheel_mesh_colors[v+j] = wood_facegrain(0.5)

    # ---- Side shrouds (bucket walls) ----
    for k in range(WHEEL_N_PAD):
        a0 = ang + float(k)     * (2.0 * PI / WHEEL_N_PAD)
        a1 = ang + float(k + 1) * (2.0 * PI / WHEEL_N_PAD)

        # Front shroud (x = xf)
        fb = _FG_SHROUD + k * 2;  v = fb * 4
        wheel_mesh_verts[v + 0] = ti.Vector([xf, cy + WHEEL_SHROUD_R * ti.cos(a0), cz + WHEEL_SHROUD_R * ti.sin(a0)])
        wheel_mesh_verts[v + 1] = ti.Vector([xf, cy + WHEEL_R_OUT    * ti.cos(a0), cz + WHEEL_R_OUT    * ti.sin(a0)])
        wheel_mesh_verts[v + 2] = ti.Vector([xf, cy + WHEEL_R_OUT    * ti.cos(a1), cz + WHEEL_R_OUT    * ti.sin(a1)])
        wheel_mesh_verts[v + 3] = ti.Vector([xf, cy + WHEEL_SHROUD_R * ti.cos(a1), cz + WHEEL_SHROUD_R * ti.sin(a1)])
        nf = ti.Vector([-1.0, 0.0, 0.0])
        cf = wood_endgrain(0.7)
        for j in ti.static(range(4)):
            wheel_mesh_norms[v + j]  = nf
            wheel_mesh_colors[v + j] = cf

        # Back shroud (x = xb)
        fb = _FG_SHROUD + k * 2 + 1;  v = fb * 4
        wheel_mesh_verts[v + 0] = ti.Vector([xb, cy + WHEEL_SHROUD_R * ti.cos(a0), cz + WHEEL_SHROUD_R * ti.sin(a0)])
        wheel_mesh_verts[v + 1] = ti.Vector([xb, cy + WHEEL_SHROUD_R * ti.cos(a1), cz + WHEEL_SHROUD_R * ti.sin(a1)])
        wheel_mesh_verts[v + 2] = ti.Vector([xb, cy + WHEEL_R_OUT    * ti.cos(a1), cz + WHEEL_R_OUT    * ti.sin(a1)])
        wheel_mesh_verts[v + 3] = ti.Vector([xb, cy + WHEEL_R_OUT    * ti.cos(a0), cz + WHEEL_R_OUT    * ti.sin(a0)])
        nb = ti.Vector([1.0, 0.0, 0.0])
        cb = wood_endgrain(0.7)
        for j in ti.static(range(4)):
            wheel_mesh_norms[v + j]  = nb
            wheel_mesh_colors[v + j] = cb


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
    init_wheel_idx()
    update_wheel_mesh()

    # --- Headless export mode ---
    if args.export > 0:
        out = args.export_dir
        os.makedirs(out, exist_ok=True)
        print(f"\nExporting {args.export} frames to {out}/")
        print(f"  Particles: {n_particles:,}  Steps/frame: {steps}")
        for frame in range(args.export):
            for _ in range(steps):
                substep(0, -GRAVITY, 0)
            update_wheel_mesh()
            export_water_ply(frame, out)
            export_wheel_obj(frame, out)
            if frame % 10 == 0:
                print(f"  frame {frame}/{args.export}")
        print("Export complete.")
        return

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

        update_wheel_mesh()

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
        scene.point_light(pos=(0.5, 0.3, -0.5), color=(0.25, 0.25, 0.25))

        # Water particles
        scene.particles(F_x, per_vertex_color=F_colors, radius=0.004)

        # Solid wooden wheel (rim + hub + paddles, no disc faces)
        scene.mesh(wheel_mesh_verts, indices=wheel_mesh_idx,
                   normals=wheel_mesh_norms, per_vertex_color=wheel_mesh_colors)

        canvas.scene(scene)

        with gui.sub_window("Water Wheel", 0.02, 0.02, 0.22, 0.16) as w:
            w.text("SPACE: pause/play")
            w.text("R: reset")
            w.text(f"omega: {WHEEL_OMEGA:.1f} rad/s")
            wheel_cy[None] = w.slider_float("Wheel Y", wheel_cy[None], 0.30, 0.85)
            w.text(f"Wheel Y = {wheel_cy[None]:.3f}")

        window.show()


if __name__ == "__main__":
    main()
