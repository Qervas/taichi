"""
Phase 6 — GPU-Accelerated Shallow Water Flood Simulator
    Taichi HLLC+MUSCL SWE solver  +  moderngl PBR ocean rendering

Usage:
    python phases/phase6/run.py                     # default dam_break
    python phases/phase6/run.py --scene urban        # urban flood
    python phases/phase6/run.py --scene valley
    python phases/phase6/run.py --scene levee
    python phases/phase6/run.py --scene circular
    python phases/phase6/run.py --scene thacker
    python phases/phase6/run.py --test               # Stoker benchmark

Controls:
    1-6     Switch scene
    SPACE   Pause / resume
    R       Reset current scene
    T       Cycle storm intensity (0.0 / 0.5 / 0.8 / 1.0)
    F1/F2/F3 Ocean preset (Calm / Medium / Storm)
    Mouse   Orbit (LMB drag), zoom (scroll)
    ESC     Quit
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

# ═══════════════════════════════════════════════════════════════════════
# Taichi init
# ═══════════════════════════════════════════════════════════════════════
ti.init(arch=ti.gpu)

# Grid dimensions (will be set by load_scene)
MAX_NX, MAX_NY = 512, 512

# ═══════════════════════════════════════════════════════════════════════
# SWE fields — allocated at max size, NX/NY track active region
# ═══════════════════════════════════════════════════════════════════════
NX_f = ti.field(int, shape=())
NY_f = ti.field(int, shape=())
DX_f = ti.field(float, shape=())
G_f = ti.field(float, shape=())
DT_f = ti.field(float, shape=())
T_f = ti.field(float, shape=())  # simulation time

h = ti.field(float, shape=(MAX_NX, MAX_NY))
hu = ti.field(float, shape=(MAX_NX, MAX_NY))
hv = ti.field(float, shape=(MAX_NX, MAX_NY))
z_bed = ti.field(float, shape=(MAX_NX, MAX_NY))
is_wall = ti.field(int, shape=(MAX_NX, MAX_NY))
n_manning = ti.field(float, shape=(MAX_NX, MAX_NY))

# MUSCL-reconstructed left/right states at cell interfaces
# x-interfaces: (NX+1) faces
hL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
hR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
huL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
huR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
hvL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
hvR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
zL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
zR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))

# y-interfaces: (NY+1) faces
hL_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
hR_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
huL_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
huR_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
hvL_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
hvR_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
zL_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
zR_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))

# Intercell fluxes
Fh_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
Fhu_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
Fhv_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
Fh_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
Fhu_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))
Fhv_y = ti.field(float, shape=(MAX_NX, MAX_NY + 1))

# CFL wave speed tracking
max_wavespeed = ti.field(float, shape=())

EPS_H = 1e-6  # dry tolerance


# ═══════════════════════════════════════════════════════════════════════
# MUSCL slope limiter (minmod)
# ═══════════════════════════════════════════════════════════════════════
@ti.func
def minmod(a: float, b: float) -> float:
    result = 0.0
    if a * b > 0.0:
        if ti.abs(a) < ti.abs(b):
            result = a
        else:
            result = b
    return result


# ═══════════════════════════════════════════════════════════════════════
# MUSCL reconstruction — x-direction interfaces
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def muscl_reconstruct_x():
    nx = NX_f[None]
    ny = NY_f[None]
    for i, j in ti.ndrange((1, nx), ny):
        # Left state (from cell i-1)
        h_c = h[i - 1, j]
        hu_c = hu[i - 1, j]
        hv_c = hv[i - 1, j]
        z_c = z_bed[i - 1, j]

        dh_l = 0.0
        dhu_l = 0.0
        dhv_l = 0.0
        if i >= 2:
            dh_l = minmod(h[i - 1, j] - h[i - 2, j], h[i, j] - h[i - 1, j])
            dhu_l = minmod(hu[i - 1, j] - hu[i - 2, j], hu[i, j] - hu[i - 1, j])
            dhv_l = minmod(hv[i - 1, j] - hv[i - 2, j], hv[i, j] - hv[i - 1, j])

        hL_x[i, j] = h_c + 0.5 * dh_l
        huL_x[i, j] = hu_c + 0.5 * dhu_l
        hvL_x[i, j] = hv_c + 0.5 * dhv_l
        zL_x[i, j] = z_c

        # Right state (from cell i)
        h_c2 = h[i, j]
        hu_c2 = hu[i, j]
        hv_c2 = hv[i, j]
        z_c2 = z_bed[i, j]

        dh_r = 0.0
        dhu_r = 0.0
        dhv_r = 0.0
        if i < nx - 1:
            dh_r = minmod(h[i, j] - h[i - 1, j], h[i + 1, j] - h[i, j])
            dhu_r = minmod(hu[i, j] - hu[i - 1, j], hu[i + 1, j] - hu[i, j])
            dhv_r = minmod(hv[i, j] - hv[i - 1, j], hv[i + 1, j] - hv[i, j])

        hR_x[i, j] = h_c2 - 0.5 * dh_r
        huR_x[i, j] = hu_c2 - 0.5 * dhu_r
        hvR_x[i, j] = hv_c2 - 0.5 * dhv_r
        zR_x[i, j] = z_c2

    # Boundary faces (i=0 and i=nx): copy from adjacent cell
    for j in range(ny):
        hL_x[0, j] = h[0, j]
        huL_x[0, j] = hu[0, j]
        hvL_x[0, j] = hv[0, j]
        zL_x[0, j] = z_bed[0, j]
        hR_x[0, j] = h[0, j]
        huR_x[0, j] = hu[0, j]
        hvR_x[0, j] = hv[0, j]
        zR_x[0, j] = z_bed[0, j]

        hL_x[nx, j] = h[nx - 1, j]
        huL_x[nx, j] = hu[nx - 1, j]
        hvL_x[nx, j] = hv[nx - 1, j]
        zL_x[nx, j] = z_bed[nx - 1, j]
        hR_x[nx, j] = h[nx - 1, j]
        huR_x[nx, j] = hu[nx - 1, j]
        hvR_x[nx, j] = hv[nx - 1, j]
        zR_x[nx, j] = z_bed[nx - 1, j]


# ═══════════════════════════════════════════════════════════════════════
# MUSCL reconstruction — y-direction interfaces
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def muscl_reconstruct_y():
    nx = NX_f[None]
    ny = NY_f[None]
    for i, j in ti.ndrange(nx, (1, ny)):
        h_c = h[i, j - 1]
        hu_c = hu[i, j - 1]
        hv_c = hv[i, j - 1]
        z_c = z_bed[i, j - 1]

        dh_l = 0.0
        dhu_l = 0.0
        dhv_l = 0.0
        if j >= 2:
            dh_l = minmod(h[i, j - 1] - h[i, j - 2], h[i, j] - h[i, j - 1])
            dhu_l = minmod(hu[i, j - 1] - hu[i, j - 2], hu[i, j] - hu[i, j - 1])
            dhv_l = minmod(hv[i, j - 1] - hv[i, j - 2], hv[i, j] - hv[i, j - 1])

        hL_y[i, j] = h_c + 0.5 * dh_l
        huL_y[i, j] = hu_c + 0.5 * dhu_l
        hvL_y[i, j] = hv_c + 0.5 * dhv_l
        zL_y[i, j] = z_c

        h_c2 = h[i, j]
        hu_c2 = hu[i, j]
        hv_c2 = hv[i, j]
        z_c2 = z_bed[i, j]

        dh_r = 0.0
        dhu_r = 0.0
        dhv_r = 0.0
        if j < ny - 1:
            dh_r = minmod(h[i, j] - h[i, j - 1], h[i, j + 1] - h[i, j])
            dhu_r = minmod(hu[i, j] - hu[i, j - 1], hu[i, j + 1] - hu[i, j])
            dhv_r = minmod(hv[i, j] - hv[i, j - 1], hv[i, j + 1] - hv[i, j])

        hR_y[i, j] = h_c2 - 0.5 * dh_r
        huR_y[i, j] = hu_c2 - 0.5 * dhu_r
        hvR_y[i, j] = hv_c2 - 0.5 * dhv_r
        zR_y[i, j] = z_c2

    for i in range(nx):
        hL_y[i, 0] = h[i, 0]
        huL_y[i, 0] = hu[i, 0]
        hvL_y[i, 0] = hv[i, 0]
        zL_y[i, 0] = z_bed[i, 0]
        hR_y[i, 0] = h[i, 0]
        huR_y[i, 0] = hu[i, 0]
        hvR_y[i, 0] = hv[i, 0]
        zR_y[i, 0] = z_bed[i, 0]

        hL_y[i, ny] = h[i, ny - 1]
        huL_y[i, ny] = hu[i, ny - 1]
        hvL_y[i, ny] = hv[i, ny - 1]
        zL_y[i, ny] = z_bed[i, ny - 1]
        hR_y[i, ny] = h[i, ny - 1]
        huR_y[i, ny] = hu[i, ny - 1]
        hvR_y[i, ny] = hv[i, ny - 1]
        zR_y[i, ny] = z_bed[i, ny - 1]


# ═══════════════════════════════════════════════════════════════════════
# Hydrostatic reconstruction (Audusse et al. 2004)
# ═══════════════════════════════════════════════════════════════════════
@ti.func
def hydrostatic_recon(hL: float, hR: float, zL: float, zR: float):
    """Modify h at interface for well-balanced scheme on variable bathymetry."""
    z_star = ti.max(zL, zR)
    hL_star = ti.max(hL + zL - z_star, 0.0)
    hR_star = ti.max(hR + zR - z_star, 0.0)
    return hL_star, hR_star


# ═══════════════════════════════════════════════════════════════════════
# HLLC Riemann solver for x-direction fluxes
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def hllc_flux_x():
    nx = NX_f[None]
    ny = NY_f[None]
    g = G_f[None]
    for i, j in ti.ndrange((0, nx + 1), ny):
        hl = hL_x[i, j]
        hr = hR_x[i, j]
        zl = zL_x[i, j]
        zr = zR_x[i, j]

        # Hydrostatic reconstruction
        hl_s, hr_s = hydrostatic_recon(hl, hr, zl, zr)

        # Velocities (safe division)
        ul = 0.0
        vl = 0.0
        if hl_s > EPS_H:
            ul = huL_x[i, j] / hl
            vl = hvL_x[i, j] / hl

        ur = 0.0
        vr = 0.0
        if hr_s > EPS_H:
            ur = huR_x[i, j] / hr
            vr = hvR_x[i, j] / hr

        # Wave speeds
        cl = ti.sqrt(g * hl_s) if hl_s > EPS_H else 0.0
        cr = ti.sqrt(g * hr_s) if hr_s > EPS_H else 0.0

        sL = ti.min(ul - cl, ur - cr)
        sR = ti.max(ul + cl, ur + cr)

        # Track max wave speed for CFL
        local_max = ti.max(ti.abs(sL), ti.abs(sR))
        ti.atomic_max(max_wavespeed[None], local_max)

        # HLLC middle wave speed
        denom = hl_s * (ul - sL) - hr_s * (ur - sR)
        s_star = 0.0
        if ti.abs(denom) > 1e-12:
            s_star = (hl_s * ul * (ul - sL) - hr_s * ur * (ur - sR)
                      + 0.5 * g * (hl_s * hl_s - hr_s * hr_s)) / denom

        # Left flux
        fh_l = hl_s * ul
        fhu_l = hl_s * ul * ul + 0.5 * g * hl_s * hl_s
        fhv_l = hl_s * ul * vl

        # Right flux
        fh_r = hr_s * ur
        fhu_r = hr_s * ur * ur + 0.5 * g * hr_s * hr_s
        fhv_r = hr_s * ur * vr

        # Choose flux region
        fh_out = 0.0
        fhu_out = 0.0
        fhv_out = 0.0

        if sL >= 0.0:
            fh_out = fh_l
            fhu_out = fhu_l
            fhv_out = fhv_l
        elif sR <= 0.0:
            fh_out = fh_r
            fhu_out = fhu_r
            fhv_out = fhv_r
        elif s_star >= 0.0:
            # Left star state
            coeff = (sL - ul) / (sL - s_star + 1e-12)
            h_star_l = hl_s * coeff
            fh_out = fh_l + sL * (h_star_l - hl_s)
            fhu_out = fhu_l + sL * (h_star_l * s_star - hl_s * ul)
            fhv_out = fhv_l + sL * (h_star_l * vl - hl_s * vl)
        else:
            # Right star state
            coeff = (sR - ur) / (sR - s_star + 1e-12)
            h_star_r = hr_s * coeff
            fh_out = fh_r + sR * (h_star_r - hr_s)
            fhu_out = fhu_r + sR * (h_star_r * s_star - hr_s * ur)
            fhv_out = fhv_r + sR * (h_star_r * vr - hr_s * vr)

        Fh_x[i, j] = fh_out
        Fhu_x[i, j] = fhu_out
        Fhv_x[i, j] = fhv_out


# ═══════════════════════════════════════════════════════════════════════
# HLLC Riemann solver for y-direction fluxes (rotated)
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def hllc_flux_y():
    nx = NX_f[None]
    ny = NY_f[None]
    g = G_f[None]
    for i, j in ti.ndrange(nx, (0, ny + 1)):
        hl = hL_y[i, j]
        hr = hR_y[i, j]
        zl = zL_y[i, j]
        zr = zR_y[i, j]

        hl_s, hr_s = hydrostatic_recon(hl, hr, zl, zr)

        # For y-flux, normal velocity is v, tangential is u
        vl = 0.0
        ul_tang = 0.0
        if hl_s > EPS_H:
            vl = hvL_y[i, j] / hl
            ul_tang = huL_y[i, j] / hl

        vr = 0.0
        ur_tang = 0.0
        if hr_s > EPS_H:
            vr = hvR_y[i, j] / hr
            ur_tang = huR_y[i, j] / hr

        cl = ti.sqrt(g * hl_s) if hl_s > EPS_H else 0.0
        cr = ti.sqrt(g * hr_s) if hr_s > EPS_H else 0.0

        sL = ti.min(vl - cl, vr - cr)
        sR = ti.max(vl + cl, vr + cr)

        local_max = ti.max(ti.abs(sL), ti.abs(sR))
        ti.atomic_max(max_wavespeed[None], local_max)

        denom = hl_s * (vl - sL) - hr_s * (vr - sR)
        s_star = 0.0
        if ti.abs(denom) > 1e-12:
            s_star = (hl_s * vl * (vl - sL) - hr_s * vr * (vr - sR)
                      + 0.5 * g * (hl_s * hl_s - hr_s * hr_s)) / denom

        # Fluxes in y: (h*v, h*u*v, h*v*v + 0.5*g*h^2)
        fh_l = hl_s * vl
        fhu_l = hl_s * ul_tang * vl
        fhv_l = hl_s * vl * vl + 0.5 * g * hl_s * hl_s

        fh_r = hr_s * vr
        fhu_r = hr_s * ur_tang * vr
        fhv_r = hr_s * vr * vr + 0.5 * g * hr_s * hr_s

        fh_out = 0.0
        fhu_out = 0.0
        fhv_out = 0.0

        if sL >= 0.0:
            fh_out = fh_l
            fhu_out = fhu_l
            fhv_out = fhv_l
        elif sR <= 0.0:
            fh_out = fh_r
            fhu_out = fhu_r
            fhv_out = fhv_r
        elif s_star >= 0.0:
            coeff = (sL - vl) / (sL - s_star + 1e-12)
            h_star_l = hl_s * coeff
            fh_out = fh_l + sL * (h_star_l - hl_s)
            fhu_out = fhu_l + sL * (h_star_l * ul_tang - hl_s * ul_tang)
            fhv_out = fhv_l + sL * (h_star_l * s_star - hl_s * vl)
        else:
            coeff = (sR - vr) / (sR - s_star + 1e-12)
            h_star_r = hr_s * coeff
            fh_out = fh_r + sR * (h_star_r - hr_s)
            fhu_out = fhu_r + sR * (h_star_r * ur_tang - hr_s * ur_tang)
            fhv_out = fhv_r + sR * (h_star_r * s_star - hr_s * vr)

        Fh_y[i, j] = fh_out
        Fhu_y[i, j] = fhu_out
        Fhv_y[i, j] = fhv_out


# ═══════════════════════════════════════════════════════════════════════
# Time update + source terms (bed slope + Manning friction)
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def update_conserved():
    nx = NX_f[None]
    ny = NY_f[None]
    g = G_f[None]
    dt = DT_f[None]
    dx = DX_f[None]
    for i, j in ti.ndrange(nx, ny):
        if is_wall[i, j] == 1:
            h[i, j] = 0.0
            hu[i, j] = 0.0
            hv[i, j] = 0.0
            continue

        # Flux divergence
        dh = -(Fh_x[i + 1, j] - Fh_x[i, j]) / dx \
             - (Fh_y[i, j + 1] - Fh_y[i, j]) / dx
        dhu = -(Fhu_x[i + 1, j] - Fhu_x[i, j]) / dx \
              - (Fhu_y[i, j + 1] - Fhu_y[i, j]) / dx
        dhv = -(Fhv_x[i + 1, j] - Fhv_x[i, j]) / dx \
              - (Fhv_y[i, j + 1] - Fhv_y[i, j]) / dx

        # Bed slope source (hydrostatic reconstruction centered form)
        h_here = h[i, j]
        if h_here > EPS_H:
            # x-direction: pressure difference from bed steps
            dzL = 0.0
            dzR = 0.0
            if i > 0:
                dzL = z_bed[i, j] - z_bed[i - 1, j]
            if i < nx - 1:
                dzR = z_bed[i + 1, j] - z_bed[i, j]
            sx = -g * h_here * (dzL + dzR) / (2.0 * dx)

            dzD = 0.0
            dzU = 0.0
            if j > 0:
                dzD = z_bed[i, j] - z_bed[i, j - 1]
            if j < ny - 1:
                dzU = z_bed[i, j + 1] - z_bed[i, j]
            sy = -g * h_here * (dzD + dzU) / (2.0 * dx)

            dhu += sx
            dhv += sy

        # Update
        h_new = h_here + dt * dh
        hu_new = hu[i, j] + dt * dhu
        hv_new = hv[i, j] + dt * dhv

        # Enforce non-negative depth
        if h_new < 0.0:
            h_new = 0.0
            hu_new = 0.0
            hv_new = 0.0

        # Implicit Manning friction: hu_new / (1 + dt * friction_coeff)
        if h_new > EPS_H:
            n_m = n_manning[i, j]
            if n_m > 0.0:
                vel_mag = ti.sqrt(hu_new * hu_new + hv_new * hv_new) / h_new
                Cf = g * n_m * n_m / (h_new ** (1.0 / 3.0) + 1e-8)
                denom = 1.0 + dt * Cf * vel_mag / (h_new + 1e-8)
                hu_new /= denom
                hv_new /= denom
        else:
            hu_new = 0.0
            hv_new = 0.0

        h[i, j] = h_new
        hu[i, j] = hu_new
        hv[i, j] = hv_new


# ═══════════════════════════════════════════════════════════════════════
# Reflecting boundary conditions
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def apply_bc():
    nx = NX_f[None]
    ny = NY_f[None]
    # x boundaries: reflect hu
    for j in range(ny):
        h[0, j] = h[1, j]
        hu[0, j] = -hu[1, j]
        hv[0, j] = hv[1, j]

        h[nx - 1, j] = h[nx - 2, j]
        hu[nx - 1, j] = -hu[nx - 2, j]
        hv[nx - 1, j] = hv[nx - 2, j]

    # y boundaries: reflect hv
    for i in range(nx):
        h[i, 0] = h[i, 1]
        hu[i, 0] = hu[i, 1]
        hv[i, 0] = -hv[i, 1]

        h[i, ny - 1] = h[i, ny - 2]
        hu[i, ny - 1] = hu[i, ny - 2]
        hv[i, ny - 1] = -hv[i, ny - 2]


# ═══════════════════════════════════════════════════════════════════════
# CFL-adaptive timestep
# ═══════════════════════════════════════════════════════════════════════
def compute_dt(cfl=0.4):
    s = max_wavespeed[None]
    dx = DX_f[None]
    if s < 1e-10:
        return 0.01
    dt = cfl * dx / s
    return min(dt, 1.0)  # cap at 1s


def swe_step():
    """One full SWE timestep: reconstruct → flux → update → BC."""
    max_wavespeed[None] = 0.0

    apply_bc()
    muscl_reconstruct_x()
    muscl_reconstruct_y()
    hllc_flux_x()
    hllc_flux_y()

    dt = compute_dt()
    DT_f[None] = dt
    update_conserved()
    T_f[None] += dt
    return dt


# ═══════════════════════════════════════════════════════════════════════
# Scene loading
# ═══════════════════════════════════════════════════════════════════════
def _pad(arr, target_shape):
    """Pad 2D array to target_shape with zeros."""
    out = np.zeros(target_shape, dtype=arr.dtype)
    out[:arr.shape[0], :arr.shape[1]] = arr
    return out


def load_scene(scene_dict):
    """Upload scene data to Taichi fields."""
    sd = scene_dict
    nx, ny = sd['z_bed'].shape
    assert nx <= MAX_NX and ny <= MAX_NY, f"Scene {nx}x{ny} exceeds max {MAX_NX}x{MAX_NY}"

    NX_f[None] = nx
    NY_f[None] = ny
    DX_f[None] = sd['dx']
    G_f[None] = sd['g']
    T_f[None] = 0.0

    shape = (MAX_NX, MAX_NY)
    h.from_numpy(_pad(sd['h0'], shape))
    hu.from_numpy(_pad(sd['hu0'], shape))
    hv.from_numpy(_pad(sd['hv0'], shape))
    z_bed.from_numpy(_pad(sd['z_bed'], shape))
    is_wall.from_numpy(_pad(sd['is_wall'].astype(np.int32), shape))
    n_manning.from_numpy(_pad(sd['n_manning'], shape))


# ═══════════════════════════════════════════════════════════════════════
# Stoker dam-break analytical solution (for validation)
# ═══════════════════════════════════════════════════════════════════════
def stoker_1d(h_left, h_right, g, x_dam, x_arr, t):
    """Stoker analytical solution for 1D dam break at time t."""
    if t <= 0:
        h_exact = np.where(x_arr < x_dam, h_left, h_right)
        u_exact = np.zeros_like(x_arr)
        return h_exact, u_exact

    cL = np.sqrt(g * h_left)
    # For dry-bed (h_right=0): exact rarefaction solution
    h_exact = np.zeros_like(x_arr, dtype=np.float64)
    u_exact = np.zeros_like(x_arr, dtype=np.float64)

    for k, x in enumerate(x_arr):
        xi = (x - x_dam) / t
        if xi <= -cL:
            # Undisturbed left state
            h_exact[k] = h_left
            u_exact[k] = 0.0
        elif xi <= 2 * cL:
            # Rarefaction fan
            h_exact[k] = (2 * cL - xi) ** 2 / (9 * g)
            u_exact[k] = 2.0 / 3.0 * (cL + xi)
        else:
            # Dry right state
            h_exact[k] = 0.0
            u_exact[k] = 0.0

    return h_exact, u_exact


def run_stoker_test():
    """Run Stoker dam-break benchmark and report L1 error."""
    from terrain import scene_dam_break
    nx, ny = 512, 32  # thin strip for 1D comparison
    sd = scene_dam_break(nx, ny)
    load_scene(sd)

    dx = sd['dx']
    g = sd['g']
    x_dam = 0.3 * nx * dx
    target_t = 1.0  # compare at t=1s

    # Run simulation
    elapsed = 0.0
    steps = 0
    while elapsed < target_t:
        dt = swe_step()
        elapsed += dt
        steps += 1

    print(f"Stoker test: {steps} steps, t={elapsed:.4f}s")

    # Extract 1D slice at mid-y
    h_np = h.to_numpy()[:nx, :ny]
    mid_j = ny // 2
    h_sim = h_np[:, mid_j]
    x_arr = (np.arange(nx) + 0.5) * dx

    h_exact, _ = stoker_1d(2.0, 0.0, g, x_dam, x_arr, elapsed)

    # L1 error (normalized by initial volume)
    l1 = np.mean(np.abs(h_sim - h_exact)) / 2.0
    print(f"Stoker L1 error: {l1 * 100:.2f}%")
    if l1 < 0.05:
        print("PASS (< 5%)")
    else:
        print("FAIL (>= 5%)")
    return l1


# ═══════════════════════════════════════════════════════════════════════
# Orbit camera
# ═══════════════════════════════════════════════════════════════════════
class OrbitCamera:
    def __init__(self, target, radius, azimuth=-45.0, elevation=35.0,
                 fov=55.0, near=0.1, far=1000.0):
        self.target = np.array(target, dtype=np.float64)
        self.radius = float(radius)
        self.azimuth = float(azimuth)    # degrees
        self.elevation = float(elevation)  # degrees
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)

    def position(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        x = self.target[0] + self.radius * math.cos(el) * math.sin(az)
        y = self.target[1] + self.radius * math.sin(el)
        z = self.target[2] + self.radius * math.cos(el) * math.cos(az)
        return np.array([x, y, z], dtype=np.float64)

    def view_matrix(self):
        eye = self.position()
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1, 0, 0], dtype=np.float64)
        else:
            right /= rn
        up = np.cross(right, fwd)

        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -fwd
        mat[0, 3] = -np.dot(right, eye)
        mat[1, 3] = -np.dot(up, eye)
        mat[2, 3] = np.dot(fwd, eye)
        return mat

    def proj_matrix(self, aspect):
        f = 1.0 / math.tan(math.radians(self.fov) / 2)
        n, far = self.near, self.far
        mat = np.zeros((4, 4), dtype=np.float64)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = -(far + n) / (far - n)
        mat[2, 3] = -2 * far * n / (far - n)
        mat[3, 2] = -1
        return mat

    def rot_state(self, dx, dy):
        self.azimuth += dx * 0.3
        self.elevation = max(-89, min(89, self.elevation + dy * 0.3))

    def zoom_state(self, delta):
        self.radius *= 0.9 ** delta
        self.radius = max(0.5, self.radius)


# ═══════════════════════════════════════════════════════════════════════
# moderngl_window application
# ═══════════════════════════════════════════════════════════════════════
def run_app():
    import moderngl
    import moderngl_window

    # Import here to avoid circular issues when running --test
    from terrain import SCENES, get_scene
    from renderer import FloodRenderer

    class FloodApp(moderngl_window.WindowConfig):
        gl_version = (4, 3)
        window_size = (1920, 1080)
        title = "Phase 6 — SWE Flood Simulator"
        resizable = True
        vsync = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Load initial scene (dam_break)
            self.scene_idx = 0
            self.scene_data = SCENES[self.scene_idx][1]()
            load_scene(self.scene_data)
            sd = self.scene_data
            nx, ny = sd['z_bed'].shape

            # Renderer
            self.renderer = FloodRenderer(
                self.ctx, nx, ny, sd['dx'], sd['z_bed'],
                window_size=self.window_size,
            )

            # Camera — cinematic: low angle, telephoto compression
            cx = nx * sd['dx'] / 2
            cy = ny * sd['dx'] / 2
            D = max(nx, ny) * sd['dx']
            self.camera = OrbitCamera(
                target=(cx, 0, cy),
                radius=D * 0.8,
                azimuth=-45,
                elevation=20,
                fov=40,
                near=0.1,
                far=D * 10,
            )

            # Sun direction — golden-hour (lower sun angle)
            self.sun_dir = np.array([0.5, 0.45, 0.3], dtype=np.float64)
            self.sun_dir /= np.linalg.norm(self.sun_dir)

            # State
            self.paused = False
            self.sim_time = 0.0
            self.frame_count = 0
            self.fps_timer = time.perf_counter()
            self.fps_value = 0.0
            self.substeps_per_frame = 4

            # Storm / rain
            self.storm_intensity = 0.7
            self.rain_intensity = 0.5
            self.ocean_presets = [
                {
                    "name": "Calm",
                    "storm": 0.30,
                    "rain": 0.00,
                    "swell": 0.95,
                    "chop": 0.70,
                    "foam": 0.65,
                },
                {
                    "name": "Medium",
                    "storm": 0.55,
                    "rain": 0.10,
                    "swell": 1.10,
                    "chop": 1.00,
                    "foam": 1.00,
                },
                {
                    "name": "Storm",
                    "storm": 0.95,
                    "rain": 0.70,
                    "swell": 1.35,
                    "chop": 1.35,
                    "foam": 1.30,
                },
            ]
            self.ocean_preset_idx = 1
            self.ocean_preset_name = "Medium"
            self.ocean_swell_gain = 1.0
            self.ocean_chop_gain = 1.0
            self.ocean_foam_gain = 1.0
            self._apply_scene_visual_preset()

            # Mouse state
            self.mouse_pressed = False
            self.last_mx = 0
            self.last_my = 0

        def _reload_scene(self, idx):
            idx = idx % len(SCENES)
            self.scene_idx = idx
            self.scene_data = SCENES[idx][1]()
            load_scene(self.scene_data)
            sd = self.scene_data
            nx, ny = sd['z_bed'].shape

            # Rebuild renderer with new grid
            self.renderer = FloodRenderer(
                self.ctx, nx, ny, sd['dx'], sd['z_bed'],
                window_size=self.window_size,
            )

            cx = nx * sd['dx'] / 2
            cy = ny * sd['dx'] / 2
            D = max(nx, ny) * sd['dx']
            self.camera.target = np.array([cx, 0, cy], dtype=np.float64)
            self.camera.radius = D * 0.8
            self.camera.far = D * 10

            self.sim_time = 0.0
            self.frame_count = 0
            self._apply_scene_visual_preset()
            print(f"Scene: {sd['name']}")

        def _set_ocean_preset(self, idx, announce=True):
            idx = idx % len(self.ocean_presets)
            self.ocean_preset_idx = idx
            preset = self.ocean_presets[idx]
            self.ocean_preset_name = preset["name"]
            self.storm_intensity = float(preset["storm"])
            self.rain_intensity = float(preset["rain"])
            self.ocean_swell_gain = float(preset["swell"])
            self.ocean_chop_gain = float(preset["chop"])
            self.ocean_foam_gain = float(preset["foam"])
            if announce:
                print(
                    f"Ocean preset: {self.ocean_preset_name} | "
                    f"storm={self.storm_intensity:.2f}, rain={self.rain_intensity:.2f}, "
                    f"swell={self.ocean_swell_gain:.2f}, chop={self.ocean_chop_gain:.2f}, "
                    f"foam={self.ocean_foam_gain:.2f}"
                )

        def _apply_scene_visual_preset(self):
            # Scene 1 (Dam Break): calmer open-water look, no rain artifacts.
            if self.scene_idx == 0:
                self._set_ocean_preset(0, announce=False)
            else:
                self._set_ocean_preset(1, announce=False)

        def on_render(self, time_val: float, frame_time: float):
            self.ctx.clear(0.1, 0.1, 0.1)
            self.ctx.enable(moderngl.DEPTH_TEST)

            # Physics substeps
            if not self.paused:
                for _ in range(self.substeps_per_frame):
                    swe_step()
                self.sim_time = T_f[None]

            # Export fields to numpy
            nx = NX_f[None]
            ny = NY_f[None]
            h_np = h.to_numpy()[:nx, :ny]
            hu_np = hu.to_numpy()[:nx, :ny]
            hv_np = hv.to_numpy()[:nx, :ny]

            # Camera matrices
            aspect = self.window_size[0] / self.window_size[1]
            view = self.camera.view_matrix()
            proj = self.camera.proj_matrix(aspect)
            cam_pos = self.camera.position()

            # Render
            self.renderer.render(
                h_np, hu_np, hv_np,
                view, proj, cam_pos,
                self.sun_dir, self.sim_time,
                storm=self.storm_intensity,
                rain=self.rain_intensity,
                wall_time=time_val,
                swell_gain=self.ocean_swell_gain,
                chop_gain=self.ocean_chop_gain,
                foam_gain=self.ocean_foam_gain,
            )

            # FPS tracking
            self.frame_count += 1
            now = time.perf_counter()
            if now - self.fps_timer > 1.0:
                self.fps_value = self.frame_count / (now - self.fps_timer)
                self.frame_count = 0
                self.fps_timer = now

            # Stats overlay (title bar)
            vol = float(np.sum(h_np)) * self.scene_data['dx'] ** 2
            max_h = float(np.max(h_np))
            vel_np = np.sqrt(hu_np**2 + hv_np**2)
            mask = h_np > 1e-4
            max_v = float(np.max(vel_np[mask] / h_np[mask])) if np.any(mask) else 0.0
            self.wnd.title = (
                f"Phase 6 — {self.scene_data['name']} | "
                f"Ocean={self.ocean_preset_name} | "
                f"t={self.sim_time:.2f}s | "
                f"Vol={vol:.0f}m³ | "
                f"h_max={max_h:.2f}m | "
                f"v_max={max_v:.1f}m/s | "
                f"FPS={self.fps_value:.0f}"
            )

        def on_key_event(self, key, action, modifiers):
            keys = self.wnd.keys
            if action != keys.ACTION_PRESS:
                return

            if key == keys.SPACE:
                self.paused = not self.paused
                print("Paused" if self.paused else "Running")
            elif key == keys.R:
                self._reload_scene(self.scene_idx)
                print("Reset")
            elif key == keys.ESCAPE:
                self.wnd.close()
            # Number keys 1-6 for scene switching
            elif key == keys.NUMBER_1:
                self._reload_scene(0)
            elif key == keys.NUMBER_2:
                self._reload_scene(1)
            elif key == keys.NUMBER_3:
                self._reload_scene(2)
            elif key == keys.NUMBER_4:
                self._reload_scene(3)
            elif key == keys.NUMBER_5:
                self._reload_scene(4)
            elif key == keys.NUMBER_6:
                self._reload_scene(5)
            elif key == keys.F1:
                self._set_ocean_preset(0)
            elif key == keys.F2:
                self._set_ocean_preset(1)
            elif key == keys.F3:
                self._set_ocean_preset(2)
            elif key == keys.T:
                levels = [0.0, 0.5, 0.8, 1.0]
                cur = self.storm_intensity
                # Find next level
                idx = 0
                for i, lv in enumerate(levels):
                    if abs(cur - lv) < 0.01:
                        idx = (i + 1) % len(levels)
                        break
                self.storm_intensity = levels[idx]
                self.rain_intensity = self.storm_intensity * 0.7
                self.ocean_preset_name = "Custom"
                print(f"Storm: {self.storm_intensity:.1f}, Rain: {self.rain_intensity:.2f}")

        def on_mouse_drag_event(self, x, y, dx, dy):
            self.camera.rot_state(dx, -dy)

        def on_mouse_scroll_event(self, x_offset, y_offset):
            self.camera.zoom_state(y_offset)

        def on_resize(self, width, height):
            self.window_size = (width, height)
            if hasattr(self, 'renderer'):
                self.renderer.resize(width, height)

    moderngl_window.run_window_config(FloodApp)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    # Strip --test before moderngl_window grabs sys.argv
    if "--test" in sys.argv:
        run_stoker_test()
        return

    run_app()


if __name__ == "__main__":
    main()
