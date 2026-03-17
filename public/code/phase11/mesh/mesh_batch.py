"""Batch mesh all flood frames using mesh_python_with_foam with adaptive grid_res.

Adaptive grid_res based on particle count:
  < 500K  -> 350
  500K-1.5M -> 300
  1.5M-3M -> 250
  > 3M    -> 200

Usage:
    python mesh_batch.py
    python mesh_batch.py --input ./export/flood --output ./export/flood/meshes --start 0 --end 299
"""
import os
import sys
import time
import argparse
import numpy as np

# Allow importing mesh_surface from the phase11 directory
PHASE11_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PHASE11_DIR)

from mesh_surface import mesh_python_with_foam


def adaptive_grid_res(n_particles):
    """Return grid_res based on particle count."""
    if n_particles < 500_000:
        return 350
    elif n_particles < 1_500_000:
        return 300
    elif n_particles < 3_000_000:
        return 250
    else:
        return 200


def main():
    parser = argparse.ArgumentParser(description="Batch mesh flood frames with adaptive grid_res")
    parser.add_argument("--input", default="./export/flood",
                        help="Directory with NPZ files (default: ./export/flood)")
    parser.add_argument("--output", default="./export/flood/meshes",
                        help="Output directory for OBJ/foam files (default: ./export/flood/meshes)")
    parser.add_argument("--start", type=int, default=0, help="First frame (default: 0)")
    parser.add_argument("--end", type=int, default=299, help="Last frame (default: 299)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    total = args.end - args.start + 1
    succeeded = 0
    skipped = 0
    failed = 0

    print(f"Batch meshing frames {args.start}-{args.end} ({total} frames)")
    print(f"  Input:  {os.path.abspath(args.input)}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"  sigma=0.7, tight_bbox=False, adaptive grid_res")
    print()

    t_start = time.time()

    for i in range(args.start, args.end + 1):
        npz_path = os.path.join(args.input, f"water_{i:06d}.npz")
        obj_path = os.path.join(args.output, f"water_{i:06d}.obj")
        foam_path = os.path.join(args.output, f"foam_{i:06d}.npy")

        # Skip if output already exists and has content
        if os.path.exists(obj_path) and os.path.getsize(obj_path) > 100:
            skipped += 1
            if i % 10 == 0:
                print(f"  [{i:4d}/{args.end}] SKIP (already exists)")
            continue

        # Check input exists
        if not os.path.exists(npz_path):
            failed += 1
            print(f"  [{i:4d}/{args.end}] FAIL - no NPZ file")
            continue

        # Load particle count for adaptive grid_res
        try:
            data = np.load(npz_path)
            n_particles = len(data['x'])
        except Exception as e:
            failed += 1
            print(f"  [{i:4d}/{args.end}] FAIL - cannot read NPZ: {e}")
            continue

        grid_res = adaptive_grid_res(n_particles)

        t_frame = time.time()
        try:
            ok = mesh_python_with_foam(
                npz_path, obj_path, foam_path,
                grid_res=grid_res,
                sigma=0.7,
                tight_bbox=False,
            )
        except Exception as e:
            ok = False
            print(f"  [{i:4d}/{args.end}] FAIL - meshing error: {e}")

        dt = time.time() - t_frame

        if ok:
            succeeded += 1
            if i % 10 == 0:
                obj_kb = os.path.getsize(obj_path) / 1024
                print(f"  [{i:4d}/{args.end}] OK  {n_particles:>8,} particles, "
                      f"grid_res={grid_res}, {obj_kb:.0f}KB, {dt:.1f}s")
        else:
            failed += 1
            if i % 10 == 0:
                print(f"  [{i:4d}/{args.end}] FAIL  {n_particles:>8,} particles, grid_res={grid_res}")

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")


if __name__ == "__main__":
    main()
