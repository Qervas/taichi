"""Phase 9 — Rising flood simulation with real-time 3D preview.

Water rises uniformly around the building. Taichi GUI shows particles live.
Press SPACE to pause/resume, ESC to quit.
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
from solver import MPMFluid

print("=" * 60)
print("Phase 9 — 3D MLS-MPM Rising Flood")
print(f"  Grid: {C.N_GRID}³ = {C.N_GRID**3:,} cells")
print(f"  Gravity: {C.GRAVITY}")
print(f"  dt={C.DT}, substeps={C.SUBSTEPS}, E={C.WATER_E}")
print(f"  Flood: {C.FLOOD['inject_rate']} particles/frame")
print("=" * 60)

# Create solver
sim = MPMFluid(
    n_grid=C.N_GRID,
    max_p=C.MAX_PARTICLES,
    gravity=C.GRAVITY,
    E=C.WATER_E,
    nu=C.WATER_NU,
    dt=C.DT,
)

# Load building as solid obstacle
print("\nVoxelizing building...")
solid_np, mesh_bounds, transform = sim.load_solid(
    C.BUILDING["glb"],
    C.BUILDING["center_xy"],
    C.BUILDING["scale"],
)
building_base_z = mesh_bounds[0][2]
sim.floor_cell = int(building_base_z * C.N_GRID)
print(f"  Floor at Z={building_base_z:.4f} (cell {sim.floor_cell})")

# Injection XY bounds: wide area (most of the domain) for natural flood look
# Use domain [0.05, 0.95] — wide flood, not just building footprint
inject_xy = (0.05, 0.95, 0.05, 0.95)

# Compute max injection Z from building height
building_height = mesh_bounds[1][2] - mesh_bounds[0][2]
max_inject_z = building_base_z + building_height * C.FLOOD["max_z_frac"]
print(f"  Injection XY: [{inject_xy[0]:.3f}, {inject_xy[1]:.3f}] x "
      f"[{inject_xy[2]:.3f}, {inject_xy[3]:.3f}]")
print(f"  Building height: {building_height:.4f}, max flood Z: {max_inject_z:.4f}")

# Export directory
export_dir = os.path.join(C.EXPORT_DIR, "flood")
os.makedirs(export_dir, exist_ok=True)

# Save metadata (including inverse transform for renderer)
meta = dict(
    n_grid=C.N_GRID,
    gravity=list(C.GRAVITY),
    dt=C.DT,
    substeps=C.SUBSTEPS,
    fps=C.FPS,
    n_frames=C.N_FRAMES,
    building_glb=os.path.basename(C.BUILDING["glb"]),
    building_center_xy=list(C.BUILDING["center_xy"]),
    building_scale=C.BUILDING["scale"],
    mesh_bounds=mesh_bounds.tolist(),
    flood_inject_rate=C.FLOOD["inject_rate"],
    flood_z_thickness=C.FLOOD["z_thickness"],
    transform=transform,
)
with open(os.path.join(export_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

np.save(os.path.join(export_dir, "solid.npy"), solid_np)

# --- Optional Taichi 3D visualization ---
USE_GUI = "--headless" not in sys.argv

# Simulation state
inject_rate = C.FLOOD["inject_rate"]
z_thickness = C.FLOOD["z_thickness"]
prefill_frames = C.FLOOD["prefill_frames"]
prefill_rate = C.FLOOD["prefill_rate"]
prefill_thickness = C.FLOOD["prefill_thickness"]

# Target water level rises linearly from floor to max over N_FRAMES
target_z_start = building_base_z
target_z_end = max_inject_z
target_z_per_frame = (target_z_end - target_z_start) / C.N_FRAMES

frame = 0
t0 = time.time()

print(f"\nRunning {'(headless)' if not USE_GUI else '(GUI)'}...")
print(f"  Target Z: {target_z_start:.4f} → {target_z_end:.4f} over {C.N_FRAMES} frames")
print(f"  Rise rate: {target_z_per_frame:.6f} / frame")

for frame in range(C.N_FRAMES):
    # Target water level for this frame (linear ramp)
    target_z = target_z_start + target_z_per_frame * (frame + 1)

    # Get actual water surface level (90th percentile of existing particles)
    if frame == 0:
        actual_z = building_base_z
    else:
        actual_z = sim.get_water_level(percentile=90)

    # Injection Z: at the actual water surface, up to the target
    # This prevents injection from getting ahead of the real water level
    inject_z = min(target_z, actual_z + z_thickness * 3)

    # Pre-fill phase: heavy injection to establish connected base layer
    if frame < prefill_frames:
        rate = prefill_rate
        thick = prefill_thickness
    else:
        rate = inject_rate
        thick = z_thickness

    z_lo = max(inject_z - thick, building_base_z)
    z_hi = inject_z + thick
    added = sim.inject_uniform(z_lo, z_hi, rate, xy_bounds=inject_xy)

    # Step physics
    sim.step(C.SUBSTEPS)

    # Export
    out_path = os.path.join(export_dir, f"water_{frame:06d}.npz")
    sim.export_frame(out_path)

    # Also export PLY for SplashSurf
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
        n_p = sim.num_p[None]
        elapsed = time.time() - t0
        fps_sim = (frame + 1) / elapsed if elapsed > 0 else 0
        print(f"  [{frame:4d}/{C.N_FRAMES}] n={n_p:,} (+{added}) "
              f"actual_z={actual_z:.4f} target_z={target_z:.4f} "
              f"inject_z={inject_z:.4f} | {elapsed:.0f}s ({fps_sim:.1f} fr/s)")

elapsed = time.time() - t0
print(f"\nDone! {C.N_FRAMES} frames in {elapsed:.0f}s")
