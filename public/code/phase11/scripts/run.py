"""Phase 11 — FLIP flood simulation: directional inflow from left boundary.

SOTA pipeline: FLIP solver on MAC staggered grid with MGPCG pressure solve.
50x fewer substeps than MLS-MPM (1 vs 50) thanks to implicit pressure projection.

Usage:
    python phases/phase11/run.py
"""
import taichi as ti
ti.init(arch=ti.gpu)

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from flip_solver import FLIPSolver

dt_sub = C.FRAME_DT / C.FLIP_SUBSTEPS
cfl_est = C.FLOOD["inflow_velocity"] * dt_sub * C.N_GRID

print("=" * 60)
print("Phase 11 — FLIP Flood Simulation")
print(f"  Grid: {C.N_GRID}^3 = {C.N_GRID**3:,} cells")
print(f"  Gravity: {C.GRAVITY}")
print(f"  FLIP/PIC ratio: {C.FLIP_RATIO}")
print(f"  Frame dt: {C.FRAME_DT}, substeps: {C.FLIP_SUBSTEPS}, "
      f"sub-dt: {dt_sub:.4f}")
print(f"  CFL estimate: {cfl_est:.2f} (limit ~5 for FLIP)")
print(f"  PCG: max_iter={C.PCG_MAX_ITER}, tol={C.PCG_TOL}")
print(f"  Inflow: {C.FLOOD['inject_rate']} p/frame, "
      f"v={C.FLOOD['inflow_velocity']:.1f}, "
      f"width={C.FLOOD['inflow_width']:.3f}")
print("=" * 60)

sim = FLIPSolver(
    n_grid=C.N_GRID,
    max_p=C.MAX_PARTICLES,
    gravity=C.GRAVITY,
    dt=dt_sub,
    flip_ratio=C.FLIP_RATIO,
    pcg_max_iter=C.PCG_MAX_ITER,
    pcg_tol=C.PCG_TOL,
    foam_v_thresh=C.FOAM_V_THRESH,
    foam_decay=C.FOAM_DECAY,
)

# ---------------------------------------------------------------------------
# Load building
# ---------------------------------------------------------------------------
models_dir = os.path.expanduser("~/Downloads/models")
obj_path = os.path.join(models_dir, C.BUILDING_OBJ)
print(f"\nVoxelizing building: {C.BUILDING_OBJ}")

solid_np, mesh_bounds, transform = sim.load_solid_from_obj(obj_path, C.BUILDING_SCALE)
building_base_z = mesh_bounds[0][2]
building_height = mesh_bounds[1][2] - mesh_bounds[0][2]

sim.floor_cell = int(building_base_z * sim.n)
print(f"  Floor at Z={building_base_z:.4f} (cell {sim.floor_cell})")

max_inject_z = building_base_z + building_height * C.FLOOD["max_z_frac"]
print(f"  Building height: {building_height:.4f}, max flood Z: {max_inject_z:.4f}")

# ---------------------------------------------------------------------------
# Inflow geometry: narrow slab at left boundary
# ---------------------------------------------------------------------------
bnd = sim.bound * sim.dx + sim.dx
inflow_x_lo = bnd
inflow_x_hi = bnd + C.FLOOD["inflow_width"]
inflow_y_lo = 0.05
inflow_y_hi = 0.95
inflow_vx = C.FLOOD["inflow_velocity"]

print(f"\n  Inflow slab: X=[{inflow_x_lo:.4f}, {inflow_x_hi:.4f}], "
      f"Y=[{inflow_y_lo}, {inflow_y_hi}]")
print(f"  Inflow velocity: ({inflow_vx:.1f}, 0, 0)")

# ---------------------------------------------------------------------------
# Export setup
# ---------------------------------------------------------------------------
export_dir = os.path.join(C.EXPORT_DIR, "flood")
os.makedirs(export_dir, exist_ok=True)

meta = dict(
    solver="FLIP",
    n_grid=C.N_GRID,
    gravity=list(C.GRAVITY),
    frame_dt=C.FRAME_DT,
    flip_substeps=C.FLIP_SUBSTEPS,
    flip_ratio=C.FLIP_RATIO,
    pcg_max_iter=C.PCG_MAX_ITER,
    pcg_tol=C.PCG_TOL,
    fps=C.FPS,
    n_frames=C.N_FRAMES,
    building_obj=C.BUILDING_OBJ,
    building_scale=C.BUILDING_SCALE,
    mesh_bounds=mesh_bounds.tolist(),
    flood_type="inundation",
    flood_inject_rate=C.FLOOD["inject_rate"],
    flood_inflow_velocity=C.FLOOD["inflow_velocity"],
    flood_inflow_width=C.FLOOD["inflow_width"],
    transform=transform,
)
with open(os.path.join(export_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
np.save(os.path.join(export_dir, "solid.npy"), solid_np)

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
inject_rate = C.FLOOD["inject_rate"]
prefill_frames = C.FLOOD["prefill_frames"]
prefill_rate = C.FLOOD["prefill_rate"]

target_z_start = building_base_z
target_z_end = max_inject_z
target_z_per_frame = (target_z_end - target_z_start) / C.N_FRAMES

t0 = time.time()
print(f"\nRunning FLIP simulation...")
print(f"  Inflow Z: {target_z_start:.4f} -> {target_z_end:.4f} over {C.N_FRAMES} frames")

for frame in range(C.N_FRAMES):
    # Rising water level at the inflow boundary
    target_z = target_z_start + target_z_per_frame * (frame + 1)
    rate = prefill_rate if frame < prefill_frames else inject_rate

    z_lo = building_base_z
    z_hi = target_z

    added = sim.inject_uniform(
        z_lo, z_hi, rate,
        xy_bounds=(inflow_x_lo, inflow_x_hi, inflow_y_lo, inflow_y_hi),
        velocity=(inflow_vx, 0.0, 0.0),
    )

    # Run substeps
    verbose = (frame < 3)
    pcg_iters = sim.step(frame_dt=C.FRAME_DT, n_substeps=C.FLIP_SUBSTEPS,
                         verbose=verbose)

    # Export NPZ + PLY
    out_path = os.path.join(export_dir, f"water_{frame:06d}.npz")
    sim.export_frame(out_path)

    n_p = sim.num_p[None]
    if n_p > 0:
        x_np = sim.x.to_numpy()[:n_p]
        ply_path = os.path.join(export_dir, f"water_{frame:06d}.ply")
        with open(ply_path, "wb") as f:
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n_p}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"end_header\n"
            )
            f.write(header.encode("ascii"))
            f.write(x_np.astype(np.float32).tobytes())

    if frame % 30 == 0 or frame < 5:
        elapsed = time.time() - t0
        fps_sim = (frame + 1) / elapsed if elapsed > 0 else 0
        actual_z = sim.get_water_level(percentile=90) if n_p > 0 else building_base_z
        v_max = sim.max_velocity() if n_p > 0 else 0
        cfl = v_max * dt_sub / sim.dx
        print(f"  [{frame:4d}/{C.N_FRAMES}] n={n_p:,} (+{added}) "
              f"z={actual_z:.4f} target={target_z:.4f} "
              f"v_max={v_max:.2f} CFL={cfl:.2f} "
              f"PCG={pcg_iters} | {elapsed:.0f}s ({fps_sim:.1f} fr/s)")

elapsed = time.time() - t0
print(f"\nDone! {C.N_FRAMES} frames in {elapsed:.0f}s "
      f"({elapsed/C.N_FRAMES:.1f}s/frame)")
