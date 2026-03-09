"""
2D Shallow Water Equations solver (HLLC + MUSCL).
Ported from Phase 6, wrapped in a clean class.

Handles all water physics: flood propagation, obstacle reflection,
Manning friction, bed slope. Provides h(x,y,t) and velocity fields
for coupling with MPM solids.
"""

import numpy as np
import taichi as ti

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


@ti.func
def minmod(a: float, b: float) -> float:
    result = 0.0
    if a * b > 0.0:
        if ti.abs(a) < ti.abs(b):
            result = a
        else:
            result = b
    return result


@ti.data_oriented
class SWESolver:
    def __init__(self):
        self.nx = C.SWE_NX
        self.ny = C.SWE_NY
        self.dx = C.SWE_DX
        self.g = C.GRAVITY
        self.cfl = C.SWE_CFL
        self.manning = C.SWE_MANNING
        self.eps_h = 1e-6
        self.time = 0.0

        # SWE grid origin in sim-local coords (same as full domain origin)
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Inflow config
        fl = C.SWE_FLOOD
        self.inflow_depth = fl["inflow_depth"]
        self.inflow_vel = fl["inflow_velocity"]
        self.ramp_time = fl["ramp_time"]

        self._allocate()
        self._build_kernels()

        print(f"SWE: {self.nx}x{self.ny} grid, dx={self.dx:.3f}m")

    def _allocate(self):
        nx, ny = self.nx, self.ny
        # Conserved variables
        self.h = ti.field(float, (nx, ny))
        self.hu = ti.field(float, (nx, ny))
        self.hv = ti.field(float, (nx, ny))
        self.z_bed = ti.field(float, (nx, ny))
        self.is_wall = ti.field(int, (nx, ny))

        # MUSCL reconstruction at x-interfaces (nx+1, ny)
        self.hL_x = ti.field(float, (nx + 1, ny))
        self.hR_x = ti.field(float, (nx + 1, ny))
        self.huL_x = ti.field(float, (nx + 1, ny))
        self.huR_x = ti.field(float, (nx + 1, ny))
        self.hvL_x = ti.field(float, (nx + 1, ny))
        self.hvR_x = ti.field(float, (nx + 1, ny))
        self.zL_x = ti.field(float, (nx + 1, ny))
        self.zR_x = ti.field(float, (nx + 1, ny))

        # MUSCL reconstruction at y-interfaces (nx, ny+1)
        self.hL_y = ti.field(float, (nx, ny + 1))
        self.hR_y = ti.field(float, (nx, ny + 1))
        self.huL_y = ti.field(float, (nx, ny + 1))
        self.huR_y = ti.field(float, (nx, ny + 1))
        self.hvL_y = ti.field(float, (nx, ny + 1))
        self.hvR_y = ti.field(float, (nx, ny + 1))
        self.zL_y = ti.field(float, (nx, ny + 1))
        self.zR_y = ti.field(float, (nx, ny + 1))

        # Fluxes at x-interfaces
        self.Fh_x = ti.field(float, (nx + 1, ny))
        self.Fhu_x = ti.field(float, (nx + 1, ny))
        self.Fhv_x = ti.field(float, (nx + 1, ny))

        # Fluxes at y-interfaces
        self.Fh_y = ti.field(float, (nx, ny + 1))
        self.Fhu_y = ti.field(float, (nx, ny + 1))
        self.Fhv_y = ti.field(float, (nx, ny + 1))

        # CFL tracking
        self.max_wavespeed = ti.field(float, ())

    def init(self, building_lo_sim, building_hi_sim, car_boxes_sim=None):
        """Initialize: set walls from building/car footprints, zero water."""
        self._zero_fields()

        # Mark building footprint as wall
        bld_lo = building_lo_sim
        bld_hi = building_hi_sim
        self._set_wall_rect(
            bld_lo[0], bld_lo[1], bld_hi[0], bld_hi[1]
        )

        # Mark car footprints as wall (optional)
        if car_boxes_sim:
            for center, half, yaw in car_boxes_sim:
                r = max(half[0], half[1])
                self._set_wall_rect(
                    center[0] - r, center[1] - r,
                    center[0] + r, center[1] + r,
                )

    @ti.kernel
    def _zero_fields(self):
        for i, j in self.h:
            self.h[i, j] = 0.0
            self.hu[i, j] = 0.0
            self.hv[i, j] = 0.0
            self.z_bed[i, j] = 0.0
            self.is_wall[i, j] = 0

    def _set_wall_rect(self, x_lo, y_lo, x_hi, y_hi):
        """Mark SWE cells within a rectangle as walls."""
        dx = self.dx
        i0 = max(0, int(x_lo / dx))
        i1 = min(self.nx, int(x_hi / dx) + 1)
        j0 = max(0, int(y_lo / dx))
        j1 = min(self.ny, int(y_hi / dx) + 1)
        self._mark_wall(i0, i1, j0, j1)

    @ti.kernel
    def _mark_wall(self, i0: int, i1: int, j0: int, j1: int):
        for i, j in self.is_wall:
            if i0 <= i < i1 and j0 <= j < j1:
                self.is_wall[i, j] = 1

    def step(self) -> float:
        """Advance one SWE timestep. Returns dt used."""
        dt = self._compute_dt()
        ramp = min(1.0, self.time / max(self.ramp_time, 0.01))
        self._apply_bc(ramp * self.inflow_depth, ramp * self.inflow_vel)
        self._muscl_x()
        self._muscl_y()
        self.max_wavespeed[None] = 0.0
        self._hllc_flux_x()
        self._hllc_flux_y()
        self._update_conserved(dt)
        self.time += dt
        return dt

    def _compute_dt(self) -> float:
        s = self.max_wavespeed[None]
        if s < 1e-10:
            return min(0.01, self.cfl * self.dx / max(self.inflow_vel, 1.0))
        dt = self.cfl * self.dx / s
        return min(dt, 0.1)

    def query_numpy(self):
        """Return h, hu, hv as numpy arrays."""
        return {
            "h": self.h.to_numpy(),
            "hu": self.hu.to_numpy(),
            "hv": self.hv.to_numpy(),
        }

    def _build_kernels(self):
        """Build all Taichi kernels."""
        h_f = self.h
        hu_f = self.hu
        hv_f = self.hv
        z_bed_f = self.z_bed
        is_wall_f = self.is_wall
        nx = self.nx
        ny = self.ny
        dx = self.dx
        g = self.g
        eps_h = self.eps_h
        manning = self.manning

        # Reconstruction fields
        hL_x, hR_x = self.hL_x, self.hR_x
        huL_x, huR_x = self.huL_x, self.huR_x
        hvL_x, hvR_x = self.hvL_x, self.hvR_x
        zL_x, zR_x = self.zL_x, self.zR_x

        hL_y, hR_y = self.hL_y, self.hR_y
        huL_y, huR_y = self.huL_y, self.huR_y
        hvL_y, hvR_y = self.hvL_y, self.hvR_y
        zL_y, zR_y = self.zL_y, self.zR_y

        Fh_x, Fhu_x, Fhv_x = self.Fh_x, self.Fhu_x, self.Fhv_x
        Fh_y, Fhu_y, Fhv_y = self.Fh_y, self.Fhu_y, self.Fhv_y
        max_ws = self.max_wavespeed

        @ti.kernel
        def apply_bc(inflow_h: float, inflow_v: float):
            # Y=0 boundary: inflow
            for i in range(nx):
                h_f[i, 0] = inflow_h
                hu_f[i, 0] = 0.0
                hv_f[i, 0] = inflow_h * inflow_v  # +Y direction

            # Y=ny-1: open outflow (copy from interior)
            for i in range(nx):
                h_f[i, ny - 1] = h_f[i, ny - 2]
                hu_f[i, ny - 1] = hu_f[i, ny - 2]
                hv_f[i, ny - 1] = hv_f[i, ny - 2]

            # X boundaries: reflecting
            for j in range(ny):
                h_f[0, j] = h_f[1, j]
                hu_f[0, j] = -hu_f[1, j]
                hv_f[0, j] = hv_f[1, j]

                h_f[nx - 1, j] = h_f[nx - 2, j]
                hu_f[nx - 1, j] = -hu_f[nx - 2, j]
                hv_f[nx - 1, j] = hv_f[nx - 2, j]

            # Enforce walls
            for i, j in is_wall_f:
                if is_wall_f[i, j] == 1:
                    h_f[i, j] = 0.0
                    hu_f[i, j] = 0.0
                    hv_f[i, j] = 0.0

        @ti.kernel
        def muscl_x():
            for i, j in ti.ndrange((1, nx), ny):
                # Left state at interface i (between cell i-1 and i)
                sl_h = minmod(h_f[i - 1, j] - (h_f[i - 2, j] if i >= 2 else h_f[i - 1, j]),
                              h_f[i, j] - h_f[i - 1, j])
                hL_x[i, j] = h_f[i - 1, j] + 0.5 * sl_h

                sl_hu = minmod(hu_f[i - 1, j] - (hu_f[i - 2, j] if i >= 2 else hu_f[i - 1, j]),
                               hu_f[i, j] - hu_f[i - 1, j])
                huL_x[i, j] = hu_f[i - 1, j] + 0.5 * sl_hu

                sl_hv = minmod(hv_f[i - 1, j] - (hv_f[i - 2, j] if i >= 2 else hv_f[i - 1, j]),
                               hv_f[i, j] - hv_f[i - 1, j])
                hvL_x[i, j] = hv_f[i - 1, j] + 0.5 * sl_hv

                zL_x[i, j] = z_bed_f[i - 1, j]

                # Right state
                sr_h = minmod(h_f[i, j] - h_f[i - 1, j],
                              (h_f[i + 1, j] if i < nx - 1 else h_f[i, j]) - h_f[i, j])
                hR_x[i, j] = h_f[i, j] - 0.5 * sr_h

                sr_hu = minmod(hu_f[i, j] - hu_f[i - 1, j],
                               (hu_f[i + 1, j] if i < nx - 1 else hu_f[i, j]) - hu_f[i, j])
                huR_x[i, j] = hu_f[i, j] - 0.5 * sr_hu

                sr_hv = minmod(hv_f[i, j] - hv_f[i - 1, j],
                               (hv_f[i + 1, j] if i < nx - 1 else hv_f[i, j]) - hv_f[i, j])
                hvR_x[i, j] = hv_f[i, j] - 0.5 * sr_hv

                zR_x[i, j] = z_bed_f[i, j]

        @ti.kernel
        def muscl_y():
            for i, j in ti.ndrange(nx, (1, ny)):
                sl_h = minmod(h_f[i, j - 1] - (h_f[i, j - 2] if j >= 2 else h_f[i, j - 1]),
                              h_f[i, j] - h_f[i, j - 1])
                hL_y[i, j] = h_f[i, j - 1] + 0.5 * sl_h

                sl_hu = minmod(hu_f[i, j - 1] - (hu_f[i, j - 2] if j >= 2 else hu_f[i, j - 1]),
                               hu_f[i, j] - hu_f[i, j - 1])
                huL_y[i, j] = hu_f[i, j - 1] + 0.5 * sl_hu

                sl_hv = minmod(hv_f[i, j - 1] - (hv_f[i, j - 2] if j >= 2 else hv_f[i, j - 1]),
                               hv_f[i, j] - hv_f[i, j - 1])
                hvL_y[i, j] = hv_f[i, j - 1] + 0.5 * sl_hv

                zL_y[i, j] = z_bed_f[i, j - 1]

                sr_h = minmod(h_f[i, j] - h_f[i, j - 1],
                              (h_f[i, j + 1] if j < ny - 1 else h_f[i, j]) - h_f[i, j])
                hR_y[i, j] = h_f[i, j] - 0.5 * sr_h

                sr_hu = minmod(hu_f[i, j] - hu_f[i, j - 1],
                               (hu_f[i, j + 1] if j < ny - 1 else hu_f[i, j]) - hu_f[i, j])
                huR_y[i, j] = hu_f[i, j] - 0.5 * sr_hu

                sr_hv = minmod(hv_f[i, j] - hv_f[i, j - 1],
                               (hv_f[i, j + 1] if j < ny - 1 else hv_f[i, j]) - hv_f[i, j])
                hvR_y[i, j] = hv_f[i, j] - 0.5 * sr_hv

                zR_y[i, j] = z_bed_f[i, j]

        @ti.kernel
        def hllc_flux_x():
            for i, j in ti.ndrange((1, nx), ny):
                h_L = ti.max(hL_x[i, j], 0.0)
                h_R = ti.max(hR_x[i, j], 0.0)

                # Hydrostatic reconstruction
                z_star = ti.max(zL_x[i, j], zR_x[i, j])
                h_sL = ti.max(h_L + zL_x[i, j] - z_star, 0.0)
                h_sR = ti.max(h_R + zR_x[i, j] - z_star, 0.0)

                u_L = huL_x[i, j] / h_L if h_L > eps_h else 0.0
                u_R = huR_x[i, j] / h_R if h_R > eps_h else 0.0
                v_L = hvL_x[i, j] / h_L if h_L > eps_h else 0.0
                v_R = hvR_x[i, j] / h_R if h_R > eps_h else 0.0

                c_L = ti.sqrt(g * h_sL) if h_sL > eps_h else 0.0
                c_R = ti.sqrt(g * h_sR) if h_sR > eps_h else 0.0

                sL = ti.min(u_L - c_L, u_R - c_R)
                sR = ti.max(u_L + c_L, u_R + c_R)

                ti.atomic_max(max_ws, ti.max(ti.abs(sL), ti.abs(sR)))

                if sL >= 0.0:
                    Fh_x[i, j] = h_sL * u_L
                    Fhu_x[i, j] = h_sL * u_L * u_L + 0.5 * g * h_sL * h_sL
                    Fhv_x[i, j] = h_sL * u_L * v_L
                elif sR <= 0.0:
                    Fh_x[i, j] = h_sR * u_R
                    Fhu_x[i, j] = h_sR * u_R * u_R + 0.5 * g * h_sR * h_sR
                    Fhv_x[i, j] = h_sR * u_R * v_R
                else:
                    denom = sR - sL
                    if ti.abs(denom) < 1e-12:
                        denom = 1e-12
                    fL_h = h_sL * u_L
                    fR_h = h_sR * u_R
                    Fh_x[i, j] = (sR * fL_h - sL * fR_h + sL * sR * (h_sR - h_sL)) / denom

                    fL_hu = h_sL * u_L * u_L + 0.5 * g * h_sL * h_sL
                    fR_hu = h_sR * u_R * u_R + 0.5 * g * h_sR * h_sR
                    Fhu_x[i, j] = (sR * fL_hu - sL * fR_hu + sL * sR * (h_sR * u_R - h_sL * u_L)) / denom

                    fL_hv = h_sL * u_L * v_L
                    fR_hv = h_sR * u_R * v_R
                    Fhv_x[i, j] = (sR * fL_hv - sL * fR_hv + sL * sR * (h_sR * v_R - h_sL * v_L)) / denom

        @ti.kernel
        def hllc_flux_y():
            for i, j in ti.ndrange(nx, (1, ny)):
                h_L = ti.max(hL_y[i, j], 0.0)
                h_R = ti.max(hR_y[i, j], 0.0)

                z_star = ti.max(zL_y[i, j], zR_y[i, j])
                h_sL = ti.max(h_L + zL_y[i, j] - z_star, 0.0)
                h_sR = ti.max(h_R + zR_y[i, j] - z_star, 0.0)

                # Y-direction: normal velocity is v, tangential is u
                v_L = hvL_y[i, j] / h_L if h_L > eps_h else 0.0
                v_R = hvR_y[i, j] / h_R if h_R > eps_h else 0.0
                u_L = huL_y[i, j] / h_L if h_L > eps_h else 0.0
                u_R = huR_y[i, j] / h_R if h_R > eps_h else 0.0

                c_L = ti.sqrt(g * h_sL) if h_sL > eps_h else 0.0
                c_R = ti.sqrt(g * h_sR) if h_sR > eps_h else 0.0

                sL = ti.min(v_L - c_L, v_R - c_R)
                sR = ti.max(v_L + c_L, v_R + c_R)

                ti.atomic_max(max_ws, ti.max(ti.abs(sL), ti.abs(sR)))

                if sL >= 0.0:
                    Fh_y[i, j] = h_sL * v_L
                    Fhu_y[i, j] = h_sL * v_L * u_L
                    Fhv_y[i, j] = h_sL * v_L * v_L + 0.5 * g * h_sL * h_sL
                elif sR <= 0.0:
                    Fh_y[i, j] = h_sR * v_R
                    Fhu_y[i, j] = h_sR * v_R * u_R
                    Fhv_y[i, j] = h_sR * v_R * v_R + 0.5 * g * h_sR * h_sR
                else:
                    denom = sR - sL
                    if ti.abs(denom) < 1e-12:
                        denom = 1e-12
                    fL_h = h_sL * v_L
                    fR_h = h_sR * v_R
                    Fh_y[i, j] = (sR * fL_h - sL * fR_h + sL * sR * (h_sR - h_sL)) / denom

                    fL_hu = h_sL * v_L * u_L
                    fR_hu = h_sR * v_R * u_R
                    Fhu_y[i, j] = (sR * fL_hu - sL * fR_hu + sL * sR * (h_sR * u_R - h_sL * u_L)) / denom

                    fL_hv = h_sL * v_L * v_L + 0.5 * g * h_sL * h_sL
                    fR_hv = h_sR * v_R * v_R + 0.5 * g * h_sR * h_sR
                    Fhv_y[i, j] = (sR * fL_hv - sL * fR_hv + sL * sR * (h_sR * v_R - h_sL * v_L)) / denom

        @ti.kernel
        def update_conserved(dt: float):
            for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
                if is_wall_f[i, j] == 1:
                    continue

                # Flux divergence
                h_new = h_f[i, j] - dt / dx * (Fh_x[i + 1, j] - Fh_x[i, j]) \
                                   - dt / dx * (Fh_y[i, j + 1] - Fh_y[i, j])
                hu_new = hu_f[i, j] - dt / dx * (Fhu_x[i + 1, j] - Fhu_x[i, j]) \
                                     - dt / dx * (Fhu_y[i, j + 1] - Fhu_y[i, j])
                hv_new = hv_f[i, j] - dt / dx * (Fhv_x[i + 1, j] - Fhv_x[i, j]) \
                                     - dt / dx * (Fhv_y[i, j + 1] - Fhv_y[i, j])

                # Bed slope source
                hc = ti.max(h_new, 0.0)
                if hc > eps_h:
                    dz_dx = (z_bed_f[i + 1, j] - z_bed_f[i - 1, j]) / (2.0 * dx)
                    dz_dy = (z_bed_f[i, j + 1] - z_bed_f[i, j - 1]) / (2.0 * dx)
                    hu_new -= dt * g * hc * dz_dx
                    hv_new -= dt * g * hc * dz_dy

                # Manning friction (implicit)
                if hc > eps_h and manning > 0.0:
                    uc = hu_new / hc
                    vc = hv_new / hc
                    spd = ti.sqrt(uc * uc + vc * vc)
                    Cf = g * manning * manning / ti.max(ti.pow(hc, 1.0 / 3.0), 0.01)
                    damp = 1.0 / (1.0 + dt * Cf * spd)
                    hu_new *= damp
                    hv_new *= damp

                h_f[i, j] = ti.max(h_new, 0.0)
                hu_f[i, j] = hu_new if h_f[i, j] > eps_h else 0.0
                hv_f[i, j] = hv_new if h_f[i, j] > eps_h else 0.0

        # Store kernel references
        self._apply_bc = apply_bc
        self._muscl_x = muscl_x
        self._muscl_y = muscl_y
        self._hllc_flux_x = hllc_flux_x
        self._hllc_flux_y = hllc_flux_y
        self._update_conserved = update_conserved
