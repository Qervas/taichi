"""Generate 2D height maps from PLY particle data.

Bins particles into a 2D XY grid, takes max(Z) per cell,
smooths gaps, and saves as 32-bit EXR images.

Usage:
    python gen_heightmaps.py --input ./export/flood --output ./export/flood/heightmaps
    python gen_heightmaps.py --input ./export/flood --output ./export/flood/heightmaps --start 0 --end 299
"""
import os
import sys
import glob
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

# OpenEXR via imageio or raw numpy
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


def load_ply(ply_path):
    """Load binary PLY with float32 xyz vertices."""
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        n_verts = 0
        for l in header.decode().split("\n"):
            if l.startswith("element vertex"):
                n_verts = int(l.split()[-1])
                break
        data = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(-1, 3)
    return data


def save_exr_numpy(path, data):
    """Save float32 2D array as raw binary (numpy .npy fallback)."""
    np.save(path, data)


def save_exr(path, data):
    """Save float32 2D array as EXR if available, else .npy."""
    if HAS_OPENEXR:
        h, w = data.shape
        header = OpenEXR.Header(w, h)
        header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        out = OpenEXR.OutputFile(path, header)
        out.writePixels({'Y': data.astype(np.float32).tobytes()})
        out.close()
    else:
        # Fallback: save as .npy (Blender can't read this directly,
        # but we'll load it in the render script)
        npy_path = path.replace('.exr', '.npy')
        np.save(npy_path, data.astype(np.float32))
        return npy_path
    return path


def particles_to_heightmap(particles, grid_res=512,
                            clip_xy=(0.05, 0.95), clip_z=(0.0, 1.0),
                            sigma=1.5, fill_radius=3):
    """Convert 3D particle positions to 2D height map.

    Args:
        particles: (N, 3) float32 array
        grid_res: output resolution (grid_res x grid_res)
        clip_xy: (lo, hi) XY clipping in sim coords
        clip_z: (lo, hi) Z clipping
        sigma: gaussian smoothing sigma
        fill_radius: max filter radius to fill small gaps

    Returns:
        heightmap: (grid_res, grid_res) float32, values in sim Z coords
        valid_mask: (grid_res, grid_res) bool, cells with particles
    """
    if len(particles) == 0:
        return np.zeros((grid_res, grid_res), dtype=np.float32), \
               np.zeros((grid_res, grid_res), dtype=bool)

    # Clip to bounds
    mask = ((particles[:, 0] >= clip_xy[0]) & (particles[:, 0] <= clip_xy[1]) &
            (particles[:, 1] >= clip_xy[0]) & (particles[:, 1] <= clip_xy[1]) &
            (particles[:, 2] >= clip_z[0]) & (particles[:, 2] <= clip_z[1]))
    pts = particles[mask]

    if len(pts) < 10:
        return np.zeros((grid_res, grid_res), dtype=np.float32), \
               np.zeros((grid_res, grid_res), dtype=bool)

    # Bin into 2D grid — take max Z per cell
    lo = clip_xy[0]
    hi = clip_xy[1]
    span = hi - lo

    ix = np.clip(((pts[:, 0] - lo) / span * (grid_res - 1)).astype(int), 0, grid_res - 1)
    iy = np.clip(((pts[:, 1] - lo) / span * (grid_res - 1)).astype(int), 0, grid_res - 1)

    heightmap = np.full((grid_res, grid_res), -1e10, dtype=np.float32)
    np.maximum.at(heightmap, (ix, iy), pts[:, 2])

    valid = heightmap > -1e9
    # Fill small gaps with max filter (expand nearby water into empty cells)
    if fill_radius > 0:
        filled = maximum_filter(heightmap * valid, size=fill_radius)
        filled_valid = maximum_filter(valid.astype(np.float32), size=fill_radius) > 0
        heightmap = np.where(valid, heightmap, np.where(filled_valid, filled, 0.0))
        valid = valid | filled_valid

    # Smooth
    if sigma > 0:
        # Only smooth valid regions
        h_smooth = gaussian_filter(heightmap * valid, sigma=sigma)
        w_smooth = gaussian_filter(valid.astype(np.float32), sigma=sigma)
        heightmap = np.where(w_smooth > 0.01, h_smooth / w_smooth, 0.0)

    # Zero out invalid
    heightmap = heightmap * valid

    return heightmap.astype(np.float32), valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with PLY files")
    parser.add_argument("--output", required=True, help="Output directory for height maps")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=299)
    parser.add_argument("--res", type=int, default=512, help="Height map resolution")
    parser.add_argument("--sigma", type=float, default=1.5, help="Smoothing sigma")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Generating height maps: res={args.res}, sigma={args.sigma}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")

    for i in range(args.start, args.end + 1):
        ply = os.path.join(args.input, f"water_{i:06d}.ply")
        out = os.path.join(args.output, f"height_{i:06d}.npy")

        if not os.path.exists(ply):
            if i % 30 == 0:
                print(f"  [{i:4d}] SKIP — no PLY")
            continue

        if os.path.exists(out):
            continue

        pts = load_ply(ply)
        hmap, valid = particles_to_heightmap(pts, grid_res=args.res, sigma=args.sigma)

        np.save(out, hmap)

        if i % 30 == 0 or i < 5:
            n_valid = int(valid.sum())
            z_range = (hmap[valid].min(), hmap[valid].max()) if n_valid > 0 else (0, 0)
            print(f"  [{i:4d}] {n_valid:,}/{args.res**2:,} cells, "
                  f"Z=[{z_range[0]:.4f}, {z_range[1]:.4f}]")

    print("Done!")


if __name__ == "__main__":
    main()
