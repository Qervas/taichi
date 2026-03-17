"""Rising tide / bathtub fill simulation using FLIP solver.

Water rises uniformly from the floor — no directional inflow.
Building sits in container, water level increases steadily over 300 frames.

Usage: CUDA_VISIBLE_DEVICES=0 python run_rising_tide.py [--n_grid 192] [--frames 300]
"""
import os, sys, time, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import taichi as ti
ti.init(arch=ti.gpu)

from flip_solver import FLIPSolver

# --- Config ---
parser = argparse.ArgumentParser()
parser.add_argument("--n_grid", type=int, default=192)
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--inject_rate", type=int, default=20000)
parser.add_argument("--max_z_frac", type=float, default=0.60,
                    help="Water rises to this fraction of building height")
args = parser.parse_args()

N_GRID = args.n_grid
N_FRAMES = args.frames
INJECT_RATE = args.inject_rate
MAX_Z_FRAC = args.max_z_frac
FRAME_DT = 5e-3
FLIP_SUBSTEPS = 1
BUILDING_OBJ = "6_4_cluster_texture.obj"
BUILDING_SCALE = 0.40

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "export", "rising_tide")
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- Solver ---
dt_sub = FRAME_DT / FLIP_SUBSTEPS

sim = FLIPSolver(
    n_grid=N_GRID,
    max_p=10_000_000,
    gravity=(0, 0, -9.81),
    dt=dt_sub,
    flip_ratio=0.97,
    pcg_max_iter=500,
    pcg_tol=1e-6,
    foam_v_thresh=1.5,
    foam_decay=0.92,
)

# --- Building ---
models_dir = os.path.expanduser("~/Downloads/models")
obj_path = os.path.join(models_dir, BUILDING_OBJ)
print(f"Loading building: {BUILDING_OBJ}")
solid_np, mesh_bounds, transform = sim.load_solid_from_obj(obj_path, BUILDING_SCALE)
building_base_z = mesh_bounds[0][2]
building_height = mesh_bounds[1][2] - mesh_bounds[0][2]
sim.floor_cell = int(building_base_z * sim.n)

max_water_z = building_base_z + building_height * MAX_Z_FRAC
print(f"  Building base: {building_base_z:.4f}, height: {building_height:.4f}")
print(f"  Target water level: {max_water_z:.4f} ({MAX_Z_FRAC*100:.0f}% of building)")

# --- Injection: uniform across entire floor ---
bnd = sim.bound * sim.dx + sim.dx
# Full XY domain (minus boundary cells)
inject_xy = (bnd, 1.0 - bnd, bnd, 1.0 - bnd)

# Water level rises linearly over N_FRAMES
z_per_frame = (max_water_z - building_base_z) / N_FRAMES

# --- Meta ---
meta = dict(
    solver="FLIP", variant="rising_tide", n_grid=N_GRID,
    gravity=[0, 0, -9.81], frame_dt=FRAME_DT,
    flip_substeps=FLIP_SUBSTEPS, flip_ratio=0.97,
    fps=30, n_frames=N_FRAMES,
    building_obj=BUILDING_OBJ, building_scale=BUILDING_SCALE,
    mesh_bounds=mesh_bounds.tolist(), transform=transform,
    flood_inject_rate=INJECT_RATE,
    flood_inflow_velocity=0.0,
    mode="rising_tide",
    max_z_frac=MAX_Z_FRAC,
)
with open(os.path.join(EXPORT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
np.save(os.path.join(EXPORT_DIR, "solid.npy"), solid_np)

# --- Simulation ---
print(f"\nRunning rising tide: {N_FRAMES} frames, {INJECT_RATE} particles/frame")
print(f"  Grid: {N_GRID}^3, substeps: {FLIP_SUBSTEPS}")

t0 = time.time()
for frame in range(N_FRAMES):
    # Target water level for this frame
    target_z = building_base_z + z_per_frame * (frame + 1)

    # Inject particles in thin layer at current water surface
    # This creates a gentle rising effect rather than splashing
    current_level = sim.get_water_level(percentile=85) if sim.num_p[None] > 100 else building_base_z
    inject_z_lo = current_level - 0.01  # slightly below surface
    inject_z_hi = current_level + 0.02  # slightly above

    # Clamp to valid range
    inject_z_lo = max(inject_z_lo, building_base_z)
    inject_z_hi = min(inject_z_hi, target_z)

    # Inject rate: boost early frames to fill faster
    rate = INJECT_RATE * 3 if frame < 10 else INJECT_RATE

    if inject_z_hi > inject_z_lo + 0.001:
        added = sim.inject_uniform(
            inject_z_lo, inject_z_hi, rate,
            xy_bounds=inject_xy,
            velocity=(0.0, 0.0, 0.0),  # no horizontal flow!
        )
    else:
        added = 0

    # Step simulation
    pcg_iters = sim.step(frame_dt=FRAME_DT, n_substeps=FLIP_SUBSTEPS,
                         verbose=(frame < 3))

    # Export
    n_p = sim.num_p[None]
    out_npz = os.path.join(EXPORT_DIR, f"water_{frame:06d}.npz")
    sim.export_frame(out_npz)

    if n_p > 0:
        x_np = sim.x.to_numpy()[:n_p]
        ply_path = os.path.join(EXPORT_DIR, f"water_{frame:06d}.ply")
        with open(ply_path, "wb") as f:
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {n_p}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"end_header\n"
            )
            f.write(header.encode("ascii"))
            f.write(x_np.astype(np.float32).tobytes())

    if frame % 20 == 0 or frame < 5:
        elapsed = time.time() - t0
        fps_sim = (frame + 1) / elapsed if elapsed > 0 else 0
        wl = sim.get_water_level(percentile=90) if n_p > 100 else 0
        v_max = sim.max_velocity() if n_p > 0 else 0
        cfl = v_max * dt_sub / sim.dx
        print(f"  [{frame:4d}/{N_FRAMES}] n={n_p:,} (+{added}) "
              f"water_z={wl:.4f}/{target_z:.4f} "
              f"v_max={v_max:.2f} CFL={cfl:.2f} "
              f"PCG={pcg_iters} | {elapsed:.0f}s ({fps_sim:.1f} fr/s)")

elapsed = time.time() - t0
print(f"\nDone! {N_FRAMES} frames in {elapsed:.0f}s ({elapsed/N_FRAMES:.1f}s/frame)")
