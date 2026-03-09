"""
Collision geometry — reads building/car bounds from config,
provides grid-level collision queries for the solver.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def building_aabb_sim():
    """Building axis-aligned bounding box in sim-local coordinates.
    Returns (min_xyz, max_xyz) tuples."""
    return C.building_bounds_sim()


def car_boxes_sim(n_cars=10):
    """Car oriented bounding boxes in sim-local coordinates.
    Returns list of (center, half_extents, yaw_rad)."""
    return C.car_boxes_sim(n_cars)


def all_collider_aabbs(n_cars=10):
    """All static collider AABBs for quick grid-level checks.
    Returns list of (min_xyz, max_xyz) tuples."""
    boxes = []

    # Building
    bmin, bmax = building_aabb_sim()
    boxes.append((bmin, bmax))

    # Cars (approximate as AABB ignoring yaw for simplicity)
    for center, half, yaw in car_boxes_sim(n_cars):
        # Conservative AABB (max of rotated extents)
        import math
        r = max(half[0], half[1])  # max horizontal half-extent
        boxes.append((
            (center[0] - r, center[1] - r, center[2] - half[2]),
            (center[0] + r, center[1] + r, center[2] + half[2]),
        ))

    return boxes


def compute_mpm_zone(n_cars=10, margin=None):
    """Auto-detect MPM zone from all solid colliders.
    Returns (origin, extent) in sim-local coords where:
      origin = (x_lo, y_lo, z_lo)
      extent = (size_x, size_y, size_z)
    """
    if margin is None:
        margin = C.MPM_MARGIN
    boxes = all_collider_aabbs(n_cars)
    lo = [min(b[0][d] for b in boxes) - margin for d in range(3)]
    hi = [max(b[1][d] for b in boxes) + margin for d in range(3)]
    # Floor clamp — don't go below ground
    lo[2] = max(lo[2], C.FLOOR_MARGIN - 2.0)
    origin = tuple(lo)
    extent = tuple(hi[d] - lo[d] for d in range(3))
    return origin, extent
