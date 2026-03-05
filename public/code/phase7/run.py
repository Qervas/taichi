"""
Phase 7 — Shenzhen Flood Simulation Driver

Headless SWE solver with two-way car coupling and per-frame export.
Reuses Phase 6 HLLC+MUSCL SWE core, adds car coupling and continuous
flood inflow boundary.

Usage:
    python phases/phase7/run.py --cars 30 --duration 60 --export_dir ./export
    python phases/phase7/run.py --area "shenzhen_futian" --cars 5 --duration 10 --export_dir ./test_export
    python phases/phase7/run.py --city_data ./city_export --cars 30 --duration 60

No rendering — headless simulation only, output goes to export/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

# ═══════════════════════════════════════════════════════════════════════
# Taichi init
# ═══════════════════════════════════════════════════════════════════════
ti.init(arch=ti.gpu)

# Grid dimensions
MAX_NX, MAX_NY = 512, 512

# ═══════════════════════════════════════════════════════════════════════
# SWE fields (same layout as Phase 6)
# ═══════════════════════════════════════════════════════════════════════
NX_f = ti.field(int, shape=())
NY_f = ti.field(int, shape=())
DX_f = ti.field(float, shape=())
G_f = ti.field(float, shape=())
DT_f = ti.field(float, shape=())
T_f = ti.field(float, shape=())

h = ti.field(float, shape=(MAX_NX, MAX_NY))
hu = ti.field(float, shape=(MAX_NX, MAX_NY))
hv = ti.field(float, shape=(MAX_NX, MAX_NY))
z_bed = ti.field(float, shape=(MAX_NX, MAX_NY))
is_wall = ti.field(int, shape=(MAX_NX, MAX_NY))
n_manning = ti.field(float, shape=(MAX_NX, MAX_NY))

# MUSCL-reconstructed states — x-interfaces
hL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
hR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
huL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
huR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
hvL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
hvR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
zL_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))
zR_x = ti.field(float, shape=(MAX_NX + 1, MAX_NY))

# y-interfaces
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

max_wavespeed = ti.field(float, shape=())
EPS_H = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# SWE kernels (copied from Phase 6 run.py)
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


@ti.kernel
def muscl_reconstruct_x():
    nx = NX_f[None]
    ny = NY_f[None]
    for i, j in ti.ndrange((1, nx), ny):
        h_c = h[i - 1, j]
        hu_c = hu[i - 1, j]
        hv_c = hv[i - 1, j]
        z_c = z_bed[i - 1, j]
        dh_l = 0.0; dhu_l = 0.0; dhv_l = 0.0
        if i >= 2:
            dh_l = minmod(h[i - 1, j] - h[i - 2, j], h[i, j] - h[i - 1, j])
            dhu_l = minmod(hu[i - 1, j] - hu[i - 2, j], hu[i, j] - hu[i - 1, j])
            dhv_l = minmod(hv[i - 1, j] - hv[i - 2, j], hv[i, j] - hv[i - 1, j])
        hL_x[i, j] = h_c + 0.5 * dh_l
        huL_x[i, j] = hu_c + 0.5 * dhu_l
        hvL_x[i, j] = hv_c + 0.5 * dhv_l
        zL_x[i, j] = z_c

        h_c2 = h[i, j]; hu_c2 = hu[i, j]; hv_c2 = hv[i, j]; z_c2 = z_bed[i, j]
        dh_r = 0.0; dhu_r = 0.0; dhv_r = 0.0
        if i < nx - 1:
            dh_r = minmod(h[i, j] - h[i - 1, j], h[i + 1, j] - h[i, j])
            dhu_r = minmod(hu[i, j] - hu[i - 1, j], hu[i + 1, j] - hu[i, j])
            dhv_r = minmod(hv[i, j] - hv[i - 1, j], hv[i + 1, j] - hv[i, j])
        hR_x[i, j] = h_c2 - 0.5 * dh_r
        huR_x[i, j] = hu_c2 - 0.5 * dhu_r
        hvR_x[i, j] = hv_c2 - 0.5 * dhv_r
        zR_x[i, j] = z_c2

    for j in range(ny):
        hL_x[0, j] = h[0, j]; huL_x[0, j] = hu[0, j]; hvL_x[0, j] = hv[0, j]; zL_x[0, j] = z_bed[0, j]
        hR_x[0, j] = h[0, j]; huR_x[0, j] = hu[0, j]; hvR_x[0, j] = hv[0, j]; zR_x[0, j] = z_bed[0, j]
        nx_ = NX_f[None]
        hL_x[nx_, j] = h[nx_ - 1, j]; huL_x[nx_, j] = hu[nx_ - 1, j]
        hvL_x[nx_, j] = hv[nx_ - 1, j]; zL_x[nx_, j] = z_bed[nx_ - 1, j]
        hR_x[nx_, j] = h[nx_ - 1, j]; huR_x[nx_, j] = hu[nx_ - 1, j]
        hvR_x[nx_, j] = hv[nx_ - 1, j]; zR_x[nx_, j] = z_bed[nx_ - 1, j]


@ti.kernel
def muscl_reconstruct_y():
    nx = NX_f[None]
    ny = NY_f[None]
    for i, j in ti.ndrange(nx, (1, ny)):
        h_c = h[i, j - 1]; hu_c = hu[i, j - 1]; hv_c = hv[i, j - 1]; z_c = z_bed[i, j - 1]
        dh_l = 0.0; dhu_l = 0.0; dhv_l = 0.0
        if j >= 2:
            dh_l = minmod(h[i, j - 1] - h[i, j - 2], h[i, j] - h[i, j - 1])
            dhu_l = minmod(hu[i, j - 1] - hu[i, j - 2], hu[i, j] - hu[i, j - 1])
            dhv_l = minmod(hv[i, j - 1] - hv[i, j - 2], hv[i, j] - hv[i, j - 1])
        hL_y[i, j] = h_c + 0.5 * dh_l
        huL_y[i, j] = hu_c + 0.5 * dhu_l
        hvL_y[i, j] = hv_c + 0.5 * dhv_l
        zL_y[i, j] = z_c

        h_c2 = h[i, j]; hu_c2 = hu[i, j]; hv_c2 = hv[i, j]; z_c2 = z_bed[i, j]
        dh_r = 0.0; dhu_r = 0.0; dhv_r = 0.0
        if j < ny - 1:
            dh_r = minmod(h[i, j] - h[i, j - 1], h[i, j + 1] - h[i, j])
            dhu_r = minmod(hu[i, j] - hu[i, j - 1], hu[i, j + 1] - hu[i, j])
            dhv_r = minmod(hv[i, j] - hv[i, j - 1], hv[i, j + 1] - hv[i, j])
        hR_y[i, j] = h_c2 - 0.5 * dh_r
        huR_y[i, j] = hu_c2 - 0.5 * dhu_r
        hvR_y[i, j] = hv_c2 - 0.5 * dhv_r
        zR_y[i, j] = z_c2

    for i in range(nx):
        hL_y[i, 0] = h[i, 0]; huL_y[i, 0] = hu[i, 0]; hvL_y[i, 0] = hv[i, 0]; zL_y[i, 0] = z_bed[i, 0]
        hR_y[i, 0] = h[i, 0]; huR_y[i, 0] = hu[i, 0]; hvR_y[i, 0] = hv[i, 0]; zR_y[i, 0] = z_bed[i, 0]
        ny_ = NY_f[None]
        hL_y[i, ny_] = h[i, ny_ - 1]; huL_y[i, ny_] = hu[i, ny_ - 1]
        hvL_y[i, ny_] = hv[i, ny_ - 1]; zL_y[i, ny_] = z_bed[i, ny_ - 1]
        hR_y[i, ny_] = h[i, ny_ - 1]; huR_y[i, ny_] = hu[i, ny_ - 1]
        hvR_y[i, ny_] = hv[i, ny_ - 1]; zR_y[i, ny_] = z_bed[i, ny_ - 1]


@ti.func
def hydrostatic_recon(hL: float, hR: float, zL: float, zR: float):
    z_star = ti.max(zL, zR)
    hL_star = ti.max(hL + zL - z_star, 0.0)
    hR_star = ti.max(hR + zR - z_star, 0.0)
    return hL_star, hR_star


@ti.kernel
def hllc_flux_x():
    nx = NX_f[None]; ny = NY_f[None]; g = G_f[None]
    for i, j in ti.ndrange((0, nx + 1), ny):
        hl = hL_x[i, j]; hr = hR_x[i, j]
        zl = zL_x[i, j]; zr = zR_x[i, j]
        hl_s, hr_s = hydrostatic_recon(hl, hr, zl, zr)

        ul = 0.0; vl = 0.0
        if hl_s > EPS_H:
            ul = huL_x[i, j] / hl; vl = hvL_x[i, j] / hl
        ur = 0.0; vr = 0.0
        if hr_s > EPS_H:
            ur = huR_x[i, j] / hr; vr = hvR_x[i, j] / hr

        cl = ti.sqrt(g * hl_s) if hl_s > EPS_H else 0.0
        cr = ti.sqrt(g * hr_s) if hr_s > EPS_H else 0.0
        sL = ti.min(ul - cl, ur - cr)
        sR = ti.max(ul + cl, ur + cr)
        ti.atomic_max(max_wavespeed[None], ti.max(ti.abs(sL), ti.abs(sR)))

        denom = hl_s * (ul - sL) - hr_s * (ur - sR)
        s_star = 0.0
        if ti.abs(denom) > 1e-12:
            s_star = (hl_s * ul * (ul - sL) - hr_s * ur * (ur - sR)
                      + 0.5 * g * (hl_s * hl_s - hr_s * hr_s)) / denom

        fh_l = hl_s * ul; fhu_l = hl_s * ul * ul + 0.5 * g * hl_s * hl_s; fhv_l = hl_s * ul * vl
        fh_r = hr_s * ur; fhu_r = hr_s * ur * ur + 0.5 * g * hr_s * hr_s; fhv_r = hr_s * ur * vr

        fh_out = 0.0; fhu_out = 0.0; fhv_out = 0.0
        if sL >= 0.0:
            fh_out = fh_l; fhu_out = fhu_l; fhv_out = fhv_l
        elif sR <= 0.0:
            fh_out = fh_r; fhu_out = fhu_r; fhv_out = fhv_r
        elif s_star >= 0.0:
            coeff = (sL - ul) / (sL - s_star + 1e-12)
            h_star_l = hl_s * coeff
            fh_out = fh_l + sL * (h_star_l - hl_s)
            fhu_out = fhu_l + sL * (h_star_l * s_star - hl_s * ul)
            fhv_out = fhv_l + sL * (h_star_l * vl - hl_s * vl)
        else:
            coeff = (sR - ur) / (sR - s_star + 1e-12)
            h_star_r = hr_s * coeff
            fh_out = fh_r + sR * (h_star_r - hr_s)
            fhu_out = fhu_r + sR * (h_star_r * s_star - hr_s * ur)
            fhv_out = fhv_r + sR * (h_star_r * vr - hr_s * vr)

        Fh_x[i, j] = fh_out; Fhu_x[i, j] = fhu_out; Fhv_x[i, j] = fhv_out


@ti.kernel
def hllc_flux_y():
    nx = NX_f[None]; ny = NY_f[None]; g = G_f[None]
    for i, j in ti.ndrange(nx, (0, ny + 1)):
        hl = hL_y[i, j]; hr = hR_y[i, j]
        zl = zL_y[i, j]; zr = zR_y[i, j]
        hl_s, hr_s = hydrostatic_recon(hl, hr, zl, zr)

        vl = 0.0; ul_tang = 0.0
        if hl_s > EPS_H:
            vl = hvL_y[i, j] / hl; ul_tang = huL_y[i, j] / hl
        vr = 0.0; ur_tang = 0.0
        if hr_s > EPS_H:
            vr = hvR_y[i, j] / hr; ur_tang = huR_y[i, j] / hr

        cl = ti.sqrt(g * hl_s) if hl_s > EPS_H else 0.0
        cr = ti.sqrt(g * hr_s) if hr_s > EPS_H else 0.0
        sL = ti.min(vl - cl, vr - cr)
        sR = ti.max(vl + cl, vr + cr)
        ti.atomic_max(max_wavespeed[None], ti.max(ti.abs(sL), ti.abs(sR)))

        denom = hl_s * (vl - sL) - hr_s * (vr - sR)
        s_star = 0.0
        if ti.abs(denom) > 1e-12:
            s_star = (hl_s * vl * (vl - sL) - hr_s * vr * (vr - sR)
                      + 0.5 * g * (hl_s * hl_s - hr_s * hr_s)) / denom

        fh_l = hl_s * vl; fhu_l = hl_s * ul_tang * vl; fhv_l = hl_s * vl * vl + 0.5 * g * hl_s * hl_s
        fh_r = hr_s * vr; fhu_r = hr_s * ur_tang * vr; fhv_r = hr_s * vr * vr + 0.5 * g * hr_s * hr_s

        fh_out = 0.0; fhu_out = 0.0; fhv_out = 0.0
        if sL >= 0.0:
            fh_out = fh_l; fhu_out = fhu_l; fhv_out = fhv_l
        elif sR <= 0.0:
            fh_out = fh_r; fhu_out = fhu_r; fhv_out = fhv_r
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

        Fh_y[i, j] = fh_out; Fhu_y[i, j] = fhu_out; Fhv_y[i, j] = fhv_out


@ti.kernel
def update_conserved():
    nx = NX_f[None]; ny = NY_f[None]; g = G_f[None]; dt = DT_f[None]; dx = DX_f[None]
    for i, j in ti.ndrange(nx, ny):
        if is_wall[i, j] == 1:
            h[i, j] = 0.0; hu[i, j] = 0.0; hv[i, j] = 0.0
            continue

        dh = -(Fh_x[i + 1, j] - Fh_x[i, j]) / dx - (Fh_y[i, j + 1] - Fh_y[i, j]) / dx
        dhu = -(Fhu_x[i + 1, j] - Fhu_x[i, j]) / dx - (Fhu_y[i, j + 1] - Fhu_y[i, j]) / dx
        dhv = -(Fhv_x[i + 1, j] - Fhv_x[i, j]) / dx - (Fhv_y[i, j + 1] - Fhv_y[i, j]) / dx

        h_here = h[i, j]
        if h_here > EPS_H:
            dzL = 0.0; dzR = 0.0
            if i > 0: dzL = z_bed[i, j] - z_bed[i - 1, j]
            if i < nx - 1: dzR = z_bed[i + 1, j] - z_bed[i, j]
            sx = -g * h_here * (dzL + dzR) / (2.0 * dx)
            dzD = 0.0; dzU = 0.0
            if j > 0: dzD = z_bed[i, j] - z_bed[i, j - 1]
            if j < ny - 1: dzU = z_bed[i, j + 1] - z_bed[i, j]
            sy = -g * h_here * (dzD + dzU) / (2.0 * dx)
            dhu += sx; dhv += sy

        h_new = h_here + dt * dh
        hu_new = hu[i, j] + dt * dhu
        hv_new = hv[i, j] + dt * dhv

        if h_new < 0.0:
            h_new = 0.0; hu_new = 0.0; hv_new = 0.0

        if h_new > EPS_H:
            n_m = n_manning[i, j]
            if n_m > 0.0:
                vel_mag = ti.sqrt(hu_new * hu_new + hv_new * hv_new) / h_new
                Cf = g * n_m * n_m / (h_new ** (1.0 / 3.0) + 1e-8)
                denom = 1.0 + dt * Cf * vel_mag / (h_new + 1e-8)
                hu_new /= denom; hv_new /= denom
        else:
            hu_new = 0.0; hv_new = 0.0

        h[i, j] = h_new; hu[i, j] = hu_new; hv[i, j] = hv_new


@ti.kernel
def apply_bc():
    """Open/outflow BCs — water flows off all edges freely (no wall reflection)."""
    nx = NX_f[None]; ny = NY_f[None]
    for j in range(ny):
        # Left: keep inflow side reflective (inflow applied separately)
        h[0, j] = h[1, j]; hu[0, j] = hu[1, j]; hv[0, j] = hv[1, j]
        # Right: open outflow
        h[nx - 1, j] = h[nx - 2, j]; hu[nx - 1, j] = hu[nx - 2, j]; hv[nx - 1, j] = hv[nx - 2, j]
    for i in range(nx):
        # Bottom: open outflow
        h[i, 0] = h[i, 1]; hu[i, 0] = hu[i, 1]; hv[i, 0] = hv[i, 1]
        # Top: open outflow
        h[i, ny - 1] = h[i, ny - 2]; hu[i, ny - 1] = hu[i, ny - 2]; hv[i, ny - 1] = hv[i, ny - 2]


# ═══════════════════════════════════════════════════════════════════════
# Flood inflow boundary
# ═══════════════════════════════════════════════════════════════════════
@ti.kernel
def apply_inflow(inflow_h: float, inflow_vel: float, inflow_width: int):
    """Apply continuous inflow on the left (i=1) boundary.

    Simulates river overflow pouring into the city.
    """
    ny = NY_f[None]
    # Inflow strip: left edge, central band
    y_start = ny // 4
    y_end = ny * 3 // 4

    for j in range(y_start, y_end):
        for i in range(1, 1 + inflow_width):
            if is_wall[i, j] == 0:
                h[i, j] = ti.max(h[i, j], inflow_h)
                hu[i, j] = h[i, j] * inflow_vel  # positive x velocity
                # Slight spread
                hv[i, j] = 0.0


# ═══════════════════════════════════════════════════════════════════════
# CFL timestep
# ═══════════════════════════════════════════════════════════════════════
def compute_dt(cfl=0.4):
    s = max_wavespeed[None]
    dx = DX_f[None]
    if s < 1e-10:
        return 0.01
    dt = cfl * dx / s
    return min(dt, 0.5)


def swe_step():
    """One full SWE timestep."""
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


def load_scene_from_dir(city_dir):
    """Load scene from city_data.py export directory."""
    scene = {
        'z_bed': np.load(os.path.join(city_dir, 'z_bed.npy')),
        'h0': np.load(os.path.join(city_dir, 'h0.npy')),
        'hu0': np.load(os.path.join(city_dir, 'hu0.npy')),
        'hv0': np.load(os.path.join(city_dir, 'hv0.npy')),
        'is_wall': np.load(os.path.join(city_dir, 'is_wall.npy')),
        'n_manning': np.load(os.path.join(city_dir, 'n_manning.npy')),
    }

    meta_path = os.path.join(city_dir, 'scene_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    scene['dx'] = meta['dx']
    scene['g'] = meta['g']
    scene['name'] = meta['name']
    scene['domain_size'] = meta['domain_size']

    # Load car data
    car_pos_path = os.path.join(city_dir, 'car_positions.npy')
    car_yaw_path = os.path.join(city_dir, 'car_yaws.npy')
    if os.path.exists(car_pos_path):
        scene['car_positions'] = np.load(car_pos_path)
        scene['car_yaws'] = np.load(car_yaw_path)
    else:
        scene['car_positions'] = np.zeros((0, 2), dtype=np.float32)
        scene['car_yaws'] = np.zeros(0, dtype=np.float32)

    return scene


# ═══════════════════════════════════════════════════════════════════════
# Main simulation loop
# ═══════════════════════════════════════════════════════════════════════
def run_simulation(args):
    from car_coupling import (
        init_cars, water_to_car_forces, integrate_cars,
        car_to_water_source, car_collisions, get_car_state, n_cars_f as car_n_cars_f
    )
    from export import export_frame, write_scene_metadata

    # Load city scene
    if args.city_data and os.path.isdir(args.city_data):
        print(f"Loading city data from {args.city_data}...")
        scene = load_scene_from_dir(args.city_data)
    else:
        # Generate synthetic scene
        from city_data import generate_synthetic_scene
        print("Generating synthetic city scene...")
        scene = generate_synthetic_scene(
            nx=args.grid, ny=args.grid,
            domain_size=args.domain,
            n_cars=args.cars,
        )

    # Load SWE fields
    load_scene(scene)
    nx = NX_f[None]
    ny = NY_f[None]
    dx = DX_f[None]
    domain_size = scene.get('domain_size', nx * dx)

    print(f"Scene: {scene['name']}")
    print(f"Grid: {nx}x{ny}, dx={dx:.2f}m, domain={domain_size:.0f}m")

    # Initialize cars
    n_cars = min(args.cars, len(scene.get('car_positions', [])))
    if n_cars > 0:
        car_positions = scene['car_positions'][:n_cars]
        car_yaws = scene['car_yaws'][:n_cars]
        init_cars(car_positions, car_yaws)
        print(f"Cars: {n_cars}")
    else:
        n_cars = 0
        car_n_cars_f[None] = 0
        print("Cars: 0 (no car coupling)")

    # Setup export (skip entirely in preview mode)
    export_dir = os.path.abspath(args.export_dir)
    if not args.preview:
        os.makedirs(export_dir, exist_ok=True)

        # Write scene metadata
        z_bed_np = z_bed.to_numpy()[:nx, :ny]
        meta = {
            'name': scene['name'],
            'nx': nx, 'ny': ny,
            'dx': float(dx),
            'domain_size': float(domain_size),
            'n_cars': n_cars,
            'fps': args.fps,
            'duration': args.duration,
        }
        if args.city_data and os.path.isdir(args.city_data):
            city_meta_path = os.path.join(args.city_data, 'scene_meta.json')
            if os.path.exists(city_meta_path):
                with open(city_meta_path) as f:
                    city_meta = json.load(f)
                meta['building_objs'] = city_meta.get('building_objs', [])
                meta['city_data_dir'] = os.path.abspath(args.city_data)

        write_scene_metadata(export_dir, meta)

    # Simulation parameters
    fps = args.fps
    duration = args.duration
    total_frames = int(duration * fps)
    frame_dt = 1.0 / fps
    sim_time = 0.0
    frame_idx = 0

    # Real-time preview GUI
    gui = None
    record_dir = None
    if args.preview:
        gui = ti.GUI("Phase 7 — Shenzhen Flood Preview", res=(nx, ny), fast_gui=False)
        print("  Preview window opened (close window or press ESC to stop)")
        if args.record:
            record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_record")
            os.makedirs(record_dir, exist_ok=True)
            print(f"  Recording frames to {record_dir}/")

    # Inflow parameters
    inflow_h = args.inflow_h
    inflow_vel = args.inflow_vel
    inflow_width = max(1, int(5.0 / dx))  # ~5m inflow strip

    print(f"\nSimulation: {duration}s at {fps}fps = {total_frames} frames")
    print(f"Inflow: h={inflow_h:.1f}m, v={inflow_vel:.1f}m/s")
    print(f"Export: {export_dir}")
    print("=" * 60)

    t_start = time.perf_counter()
    next_frame_time = 0.0
    total_steps = 0

    while sim_time < duration and frame_idx < total_frames:
        # Substep until next frame
        while sim_time < next_frame_time:
            # SWE step
            dt = swe_step()

            # Flood inflow
            if inflow_h > 0:
                apply_inflow(inflow_h, inflow_vel, inflow_width)

            # Car coupling
            if n_cars > 0:
                water_to_car_forces(h, hu, hv, z_bed, is_wall, nx, ny, dx)
                integrate_cars(dt, domain_size)
                car_to_water_source(h, hu, hv, is_wall, nx, ny, dx, dt)
                car_collisions(is_wall, nx, ny, dx)

            sim_time += dt
            total_steps += 1

        # Export to disk (skip in preview mode)
        if not args.preview:
            h_np = h.to_numpy()[:nx, :ny]
            hu_np = hu.to_numpy()[:nx, :ny]
            hv_np = hv.to_numpy()[:nx, :ny]
            z_bed_np = z_bed.to_numpy()[:nx, :ny]

            if n_cars > 0:
                car_state = get_car_state(n_cars)
            else:
                car_state = None

            export_frame(
                export_dir, frame_idx,
                h_np, hu_np, hv_np, z_bed_np,
                dx, car_state,
            )

        # Progress report (every 30 frames for preview, every 30 for export)
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            if args.preview:
                # Minimal stats without h_np (avoid extra GPU read)
                elapsed = time.perf_counter() - t_start
                fps_actual = (frame_idx + 1) / max(elapsed, 1e-6)
                print(f"  Frame {frame_idx:5d}/{total_frames} | "
                      f"t={sim_time:.2f}s | steps={total_steps} | "
                      f"fps={fps_actual:.1f}")
            else:
                elapsed = time.perf_counter() - t_start
                vol = float(np.sum(h_np)) * dx * dx
                max_h = float(np.max(h_np))
                fps_actual = (frame_idx + 1) / max(elapsed, 1e-6)
                eta = (total_frames - frame_idx - 1) / max(fps_actual, 1e-6)
                print(f"  Frame {frame_idx:5d}/{total_frames} | "
                      f"t={sim_time:.2f}s | "
                      f"steps={total_steps} | "
                      f"vol={vol:.0f}m³ | "
                      f"h_max={max_h:.2f}m | "
                      f"fps={fps_actual:.1f} | "
                      f"ETA={eta:.0f}s")

        # Real-time preview: show water depth as blue on dark ground
        if gui is not None:
            if not gui.running:
                print("  Preview closed by user.")
                break
            # Read only what we need for display
            h_np = h.to_numpy()[:nx, :ny]
            hu_np = hu.to_numpy()[:nx, :ny]
            hv_np = hv.to_numpy()[:nx, :ny]

            img = np.zeros((nx, ny, 3), dtype=np.float32)
            # Ground: dark gray
            img[:, :, 0] = 0.15
            img[:, :, 1] = 0.13
            img[:, :, 2] = 0.12
            # Buildings: white (cache wall_np, only read once)
            if frame_idx == 0:
                _wall_np_cache = is_wall.to_numpy()[:nx, :ny]
            wall_np = _wall_np_cache
            img[wall_np > 0.5, 0] = 0.6
            img[wall_np > 0.5, 1] = 0.6
            img[wall_np > 0.5, 2] = 0.6
            # Water: blue intensity by depth
            wet = h_np > 0.05
            depth_norm = np.clip(h_np / 5.0, 0, 1)
            speed = np.zeros_like(h_np)
            hmask = h_np > 0.01
            speed[hmask] = np.sqrt(hu_np[hmask]**2 + hv_np[hmask]**2) / h_np[hmask]
            speed_norm = np.clip(speed / 5.0, 0, 1)
            img[wet, 0] = 0.05 + 0.9 * speed_norm[wet]
            img[wet, 1] = 0.15 + 0.3 * depth_norm[wet] + 0.55 * speed_norm[wet]
            img[wet, 2] = 0.4 + 0.5 * depth_norm[wet]
            # Car positions as red dots
            if n_cars > 0:
                car_state_preview = get_car_state(n_cars)
                for c in range(n_cars):
                    cx, cy = car_state_preview[0][c]
                    ci, cj = int(cx / dx), int(cy / dx)
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            pi, pj = ci + di, cj + dj
                            if 0 <= pi < nx and 0 <= pj < ny:
                                img[pi, pj] = [1.0, 0.2, 0.1]
            gui.set_image(img)
            gui.show()

            # Save frame for video recording
            if record_dir is not None:
                out_path = os.path.join(record_dir, f"frame_{frame_idx:06d}.png")
                ti.tools.imwrite(img, out_path)

        frame_idx += 1
        next_frame_time = frame_idx * frame_dt

    elapsed = time.perf_counter() - t_start
    print("=" * 60)
    print(f"Simulation complete: {frame_idx} frames, {total_steps} substeps, {elapsed:.1f}s wall time")
    if not args.preview:
        print(f"Export directory: {export_dir}")

    # Assemble recorded frames into video
    if record_dir is not None:
        import subprocess
        video_path = os.path.join(os.path.dirname(record_dir), "preview_recording.mp4")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(record_dir, "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            video_path,
        ]
        print(f"\nAssembling video: {video_path}")
        subprocess.run(cmd, check=True)
        print(f"Video saved: {video_path}")
        # Clean up frames
        import glob
        for f in glob.glob(os.path.join(record_dir, "frame_*.png")):
            os.remove(f)
        os.rmdir(record_dir)
        print(f"Cleaned up {frame_idx} frame files.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 7 — Shenzhen Flood Simulation")
    parser.add_argument("--city_data", type=str, default=None,
                        help="Path to city_data.py export directory")
    parser.add_argument("--area", type=str, default="shenzhen_futian",
                        help="Predefined area name (for synthetic fallback)")
    parser.add_argument("--grid", type=int, default=512,
                        help="Grid resolution NxN")
    parser.add_argument("--domain", type=float, default=1000.0,
                        help="Domain size in meters (for synthetic mode)")
    parser.add_argument("--cars", type=int, default=30,
                        help="Number of cars")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--fps", type=int, default=30,
                        help="Export frames per second")
    parser.add_argument("--export_dir", type=str, default="./export",
                        help="Output directory for exported frames")
    parser.add_argument("--inflow_h", type=float, default=3.0,
                        help="Inflow water depth (m)")
    parser.add_argument("--inflow_vel", type=float, default=2.0,
                        help="Inflow velocity (m/s)")
    parser.add_argument("--preview", action="store_true",
                        help="Open real-time Taichi GUI to visualize simulation")
    parser.add_argument("--record", action="store_true",
                        help="Record preview frames and assemble video (requires --preview)")
    args = parser.parse_args()

    run_simulation(args)


if __name__ == "__main__":
    main()
