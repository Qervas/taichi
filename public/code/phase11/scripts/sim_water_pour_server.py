"""Water pour simulation — tube from above filling a box.

MLS-MPM on Taichi CUDA (A100). Injects particles every frame.
Exports normalized [0,1] coords for gpu_mesh_warp.py pipeline.

Usage:
    python sim_water_pour_server.py [--frames 200] [--grid 192]
"""
import taichi as ti
import numpy as np
import os
import time
import json
import math
import sys

ti.init(arch=ti.cuda)

# ── Config ───────────────────────────────────────────────────────
n_grid = 256
dim = 3
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4
substeps = 30
E = 400.0
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * 1.0
bound = 3
gravity = ti.Vector([0.0, 0.0, -9.8])
neighbour = (3, 3, 3)

# Dam break: solid water block, no injection
N = 500_000

# ── Fields ───────────────────────────────────────────────────────
F_x = ti.Vector.field(dim, float, N)
F_v = ti.Vector.field(dim, float, N)
F_C = ti.Matrix.field(dim, dim, float, N)
F_Jp = ti.field(float, N)
F_used = ti.field(int, N)
F_grid_v = ti.Vector.field(dim, float, (n_grid, n_grid, n_grid))
F_grid_m = ti.field(float, (n_grid, n_grid, n_grid))


@ti.kernel
def init_all():
    for p in F_used:
        F_used[p] = 0
        F_x[p] = ti.Vector([0.5, 0.5, 0.5])
        F_Jp[p] = 1.0
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])
        F_C[p] = ti.Matrix.zero(float, 3, 3)


@ti.kernel
def activate(start: int, count: int):
    for i in range(count):
        idx = start + i
        if idx < N:
            F_used[idx] = 1


@ti.kernel
def substep():
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.Vector([0.0, 0.0, 0.0])
        F_grid_m[I] = 0.0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        J = F_Jp[p]
        stress = -4.0 * E * dx * (J - 1.0) * ti.Matrix.identity(float, 3)
        affine = stress + p_mass * F_C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * gravity
        cond = (I < bound) & (F_grid_v[I] < 0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset.cast(float) - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        # Clamp positions
        for d in ti.static(range(3)):
            F_x[p][d] = ti.max(F_x[p][d], dx * 2.0)
            F_x[p][d] = ti.min(F_x[p][d], 1.0 - dx * 2.0)
        F_C[p] = new_C
        F_Jp[p] *= 1.0 + dt * new_C.trace()
        F_Jp[p] = ti.max(F_Jp[p], 0.05)


def write_ply(path, pos):
    n = len(pos)
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
           f"property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, 'wb') as f:
        f.write(hdr.encode('ascii'))
        f.write(pos.astype(np.float32).tobytes())


def main():
    frames = int(sys.argv[sys.argv.index('--frames') + 1]) if '--frames' in sys.argv else 250
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export', 'water_pour')
    os.makedirs(out_dir, exist_ok=True)
    domain = 2.0

    with open(os.path.join(out_dir, "meta.json"), 'w') as f:
        json.dump({"domain_size": domain, "n_grid": n_grid,
                    "transform": {"inv_scale": domain, "inv_offset": [0, 0, 0]}}, f)

    print(f"Dam Break: {n_grid}³ grid | {N:,} max particles | {frames} frames", flush=True)
    print(f"  Substeps: {substeps}, dt={dt}", flush=True)

    # Generate dam: water block on left side, tall
    # x: [0.05, 0.4], y: [0.15, 0.85], z: [0.05, 0.75]
    rng = np.random.RandomState(42)
    # Random fill to target count
    n_actual = min(N, 500_000)
    dam_x = rng.uniform(0.05, 0.40, n_actual).astype(np.float32)
    dam_y = rng.uniform(0.15, 0.85, n_actual).astype(np.float32)
    dam_z = rng.uniform(0.05, 0.75, n_actual).astype(np.float32)
    positions = np.stack([dam_x, dam_y, dam_z], axis=1)
    print(f"  Generated {n_actual:,} particles", flush=True)

    init_all()
    pos_np = F_x.to_numpy()
    pos_np[:n_actual] = positions
    F_x.from_numpy(pos_np)
    activate(0, n_actual)

    t_total = time.time()
    for frame in range(frames):
        t0 = time.time()

        for _ in range(substeps):
            substep()

        # Export in normalized [0,1] coords
        pos = F_x.to_numpy()
        used = F_used.to_numpy()
        active_pos = pos[used == 1]
        ply = os.path.join(out_dir, f"water_{frame:06d}.ply")
        write_ply(ply, active_pos)

        ms = (time.time() - t0) * 1000
        if frame % 20 == 0 or frame < 5:
            print(f"  [{frame:3d}/{frames}] {len(active_pos):>8,} ptcls  {ms:.0f}ms", flush=True)

    total = time.time() - t_total
    print(f"\nDone! {frames} frames in {total:.1f}s ({total/frames*1000:.0f}ms/frame)", flush=True)
    print(f"Output: {out_dir}", flush=True)


if __name__ == '__main__':
    main()
