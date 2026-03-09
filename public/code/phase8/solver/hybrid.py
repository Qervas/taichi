"""
Hybrid SWE + MPM Orchestrator.

Ties together:
  - SWE (2D shallow water) for ALL water physics
  - MPM (3D solids only) for building fracture, car collision, debris
  - Coupling: SWE pressure/drag forces on MPM grid nodes

Usage:
    from solver.hybrid import HybridSolver
    sim = HybridSolver()
    sim.init()
    sim.step()
    sim.export_frame(0)
"""

import os
import math
import json
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

from .swe import SWESolver
from .engine import Solver as MPMSolver
from .colliders import compute_mpm_zone, building_aabb_sim, car_boxes_sim
from .coupling import build_coupling_kernel, build_soaking_kernel


class HybridSolver:
    def __init__(self, export_dir=None):
        self.export_dir = export_dir or C.EXPORT_DIR
        self.frame_count = 0
        self.time = 0.0

        # Auto-detect MPM zone
        mpm_origin, mpm_extent = compute_mpm_zone()
        print(f"MPM zone: origin=({mpm_origin[0]:.1f}, {mpm_origin[1]:.1f}, {mpm_origin[2]:.1f})")
        print(f"  extent=({mpm_extent[0]:.1f}, {mpm_extent[1]:.1f}, {mpm_extent[2]:.1f})")

        # Create solvers
        self.swe = SWESolver()
        self.mpm = MPMSolver(mpm_origin=mpm_origin, mpm_extent=mpm_extent)

    def init(self):
        """Initialize both solvers and build coupling kernel."""
        # Init SWE with obstacle walls
        bld_lo, bld_hi = building_aabb_sim()
        cars = car_boxes_sim()
        self.swe.init(bld_lo, bld_hi, cars)

        # Init MPM (building + car particles)
        self.mpm.init()

        # Build coupling kernel
        self._apply_swe_forces = build_coupling_kernel(
            mpm_grid_v=self.mpm.F["grid_v"],
            mpm_grid_m=self.mpm.F["grid_m"],
            swe_h=self.swe.h,
            swe_hu=self.swe.hu,
            swe_hv=self.swe.hv,
            swe_wall=self.swe.is_wall,
            n_grid_mpm=self.mpm.n_grid,
            dx_mpm=self.mpm.dx,
            mpm_origin=self.mpm.mpm_origin,
            swe_nx=self.swe.nx,
            swe_ny=self.swe.ny,
            swe_dx=self.swe.dx,
            floor_z=C.FLOOR_MARGIN,
        )

        # Inject coupling into MPM substep
        self.mpm.swe_force_hook = self._apply_swe_forces

        # Build soaking kernel (progressive water damage to submerged concrete)
        self._apply_soaking = build_soaking_kernel(
            particle_x=self.mpm.F["x"],
            particle_damage=self.mpm.F["damage"],
            particle_material=self.mpm.F["material"],
            particle_used=self.mpm.F["used"],
            n_particles=self.mpm.n_particles,
            swe_h=self.swe.h,
            swe_wall=self.swe.is_wall,
            swe_nx=self.swe.nx,
            swe_ny=self.swe.ny,
            swe_dx=self.swe.dx,
            mpm_origin=self.mpm.mpm_origin,
            floor_z=C.FLOOR_MARGIN,
        )

        print(f"\nHybrid solver ready. SWE {self.swe.nx}x{self.swe.ny} + MPM {self.mpm.n_grid}^3")
        print(f"  Soaking rate: {C.COUPLING['soaking_rate']} dmg/s")

    def step(self):
        """Advance one frame. SWE and MPM are subcycled."""
        dt_mpm = self.mpm.dt
        n_substeps = self.mpm.substeps

        frame_time = dt_mpm * n_substeps
        swe_elapsed = 0.0

        # Advance SWE to cover the frame time
        while swe_elapsed < frame_time - 1e-12:
            dt_swe = self.swe.step()
            swe_elapsed += dt_swe

        # Apply soaking damage (once per frame, accumulates over time)
        # This is the main destruction mechanism: water weakens the base
        self._apply_soaking(frame_time)

        # Advance MPM substeps (SWE fields frozen for this frame)
        for _ in range(n_substeps):
            self.mpm._substep()

        self.time += frame_time
        self.frame_count += 1

    def export_frame(self, frame_id, export_dir=None):
        """Export SWE height field + MPM solids."""
        out = export_dir or self.export_dir
        os.makedirs(out, exist_ok=True)

        # SWE: export h, hu, hv as NPZ
        swe_data = self.swe.query_numpy()
        np.savez(os.path.join(out, f"swe_{frame_id:06d}.npz"), **swe_data)

        # MPM: export solid particles as PLY
        self.mpm.export_frame(frame_id, export_dir=out)

        # Scene metadata (first frame only)
        if frame_id == 0:
            meta = {
                "type": "hybrid_swe_mpm",
                "swe_nx": self.swe.nx, "swe_ny": self.swe.ny,
                "swe_dx": self.swe.dx,
                "swe_origin": [0.0, 0.0],
                "sim_origin": list(C.SIM_ORIGIN),
                "floor_z_sim": C.FLOOR_MARGIN,
                "mpm_origin": list(self.mpm.mpm_origin),
                "mpm_extent": list(self.mpm.mpm_extent),
                "mpm_dx": self.mpm.dx,
                "mpm_n_grid": self.mpm.n_grid,
                "fps": C.FPS,
                "domain": C.SIM_DOMAIN,
                "n_chunks": self.mpm.n_chunks,
                "voronoi_seeds_mpm_local": self.mpm.voronoi_seeds.tolist(),
            }
            with open(os.path.join(out, "scene_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
