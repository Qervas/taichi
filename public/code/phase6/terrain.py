"""
Phase 6 — Terrain scene generators for SWE flood simulator.

Each scene function returns a dict:
    z_bed    : np.ndarray (NX, NY)  bed elevation [m]
    h0       : np.ndarray (NX, NY)  initial water depth [m]
    hu0      : np.ndarray (NX, NY)  initial x-momentum [m^2/s]
    hv0      : np.ndarray (NX, NY)  initial y-momentum [m^2/s]
    is_wall  : np.ndarray (NX, NY)  bool — solid wall mask
    n_manning: np.ndarray (NX, NY)  Manning roughness [s/m^{1/3}]
    dx       : float                cell size [m]
    g        : float                gravity [m/s^2]
    name     : str
"""

import numpy as np


def _perlin_2d(nx, ny, scale=4.0, seed=0):
    """Simple 2D value noise (bilinear interpolated random grid)."""
    rng = np.random.default_rng(seed)
    gx = int(np.ceil(scale)) + 2
    gy = int(np.ceil(scale)) + 2
    grid = rng.standard_normal((gx, gy))

    xs = np.linspace(0, scale, nx, endpoint=False)
    ys = np.linspace(0, scale, ny, endpoint=False)
    xi = np.floor(xs).astype(int)
    yi = np.floor(ys).astype(int)
    xf = xs - xi
    yf = ys - yi

    # bilinear interp
    v00 = grid[xi[:, None], yi[None, :]]
    v10 = grid[xi[:, None] + 1, yi[None, :]]
    v01 = grid[xi[:, None], yi[None, :] + 1]
    v11 = grid[xi[:, None] + 1, yi[None, :] + 1]

    xf2 = xf[:, None]
    yf2 = yf[None, :]
    return (v00 * (1 - xf2) * (1 - yf2) +
            v10 * xf2 * (1 - yf2) +
            v01 * (1 - xf2) * yf2 +
            v11 * xf2 * yf2)


def scene_dam_break(nx=512, ny=512):
    """Classic dam break: flat bed, wall of water on the left."""
    dx = 50.0 / nx  # 50m domain
    z_bed = np.zeros((nx, ny), dtype=np.float32)
    h0 = np.zeros((nx, ny), dtype=np.float32)

    # Left 30% has 2m water, right is dry
    wall_i = int(0.3 * nx)
    h0[:wall_i, :] = 2.0

    is_wall = np.zeros((nx, ny), dtype=bool)
    # Boundary walls (1-cell thick)
    is_wall[0, :] = True
    is_wall[-1, :] = True
    is_wall[:, 0] = True
    is_wall[:, -1] = True

    n_manning = np.full((nx, ny), 0.025, dtype=np.float32)

    return dict(z_bed=z_bed, h0=h0, hu0=np.zeros_like(h0),
                hv0=np.zeros_like(h0), is_wall=is_wall,
                n_manning=n_manning, dx=dx, g=9.81, name="Dam Break")


def scene_urban_flood(nx=512, ny=512):
    """Cinematic flood through a dense city grid — 500m domain."""
    dx = 500.0 / nx  # 500m domain (~0.98 m/cell at 512)
    z_bed = np.full((nx, ny), 0.0, dtype=np.float32)

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    # Gentle slope: higher at top-left, lower at bottom-right
    z_bed += (1.0 - X) * 1.5 + (1.0 - Y) * 1.0

    # Micro-noise for ground texture
    z_bed += _perlin_2d(nx, ny, scale=8.0, seed=42) * 0.1

    is_wall = np.zeros((nx, ny), dtype=bool)
    # Boundary walls
    is_wall[0, :] = True
    is_wall[-1, :] = True
    is_wall[:, 0] = True
    is_wall[:, -1] = True

    rng = np.random.default_rng(123)
    domain_m = 500.0

    # ─── River channel through the middle (Y direction ~0.45-0.55) ───
    river_center = 0.50
    river_half = 0.04  # ~40m wide
    river_dist = np.abs(Y - river_center)
    river_mask = river_dist < river_half
    # Carve river bed 2m below surface
    z_bed[river_mask] -= 2.0

    # ─── Downtown grid: streets every 40-60m, buildings 8-20m tall ───
    # In normalized coords, 50m = 50/500 = 0.10
    street_w_norm = 12.0 / domain_m   # ~12m streets
    block_spacing = 50.0 / domain_m   # ~50m block spacing

    # Place buildings in a grid, skip river zone and reservoir zone
    bx_start = 0.15
    by_start = 0.10
    bx = bx_start
    block_id = 0
    while bx < 0.90:
        by = by_start
        bx_size = rng.uniform(0.04, 0.07)  # 20-35m building width
        while by < 0.90:
            by_size = rng.uniform(0.04, 0.07)

            # Skip if overlapping river channel
            by_mid = by + by_size / 2
            if abs(by_mid - river_center) < river_half + 0.03:
                by += by_size + street_w_norm
                continue

            # Skip reservoir zone (top-left)
            if bx < 0.22 and by < 0.30:
                by += by_size + street_w_norm
                continue

            # Skip some blocks for parks/plazas
            block_id += 1
            if block_id % 7 == 0:
                by += by_size + street_w_norm
                continue

            # Convert to grid indices
            x0 = int(bx * nx)
            x1 = int((bx + bx_size) * nx)
            y0 = int(by * ny)
            y1 = int((by + by_size) * ny)
            x0 = max(2, min(x0, nx - 2))
            x1 = max(2, min(x1, nx - 2))
            y0 = max(2, min(y0, ny - 2))
            y1 = max(2, min(y1, ny - 2))

            if x1 > x0 and y1 > y0:
                building_h = rng.uniform(8.0, 20.0)
                z_bed[x0:x1, y0:y1] += building_h
                is_wall[x0:x1, y0:y1] = True

            by += by_size + street_w_norm + rng.uniform(0.0, 0.02)

        bx += bx_size + street_w_norm + rng.uniform(0.0, 0.02)

    # ─── Flood: reservoir on top-left side, 4-5m deep ───
    h0 = np.zeros((nx, ny), dtype=np.float32)
    res_x = int(0.18 * nx)
    res_y = int(0.25 * ny)
    h0[:res_x, :res_y] = 4.5

    n_manning = np.full((nx, ny), 0.015, dtype=np.float32)  # smooth streets
    n_manning[river_mask] = 0.025  # rougher river bed
    n_manning[is_wall] = 0.10  # buildings

    return dict(z_bed=z_bed, h0=h0, hu0=np.zeros_like(h0),
                hv0=np.zeros_like(h0), is_wall=is_wall,
                n_manning=n_manning, dx=dx, g=9.81, name="Urban Flood")


def scene_valley(nx=512, ny=512):
    """River valley with sloped banks — water flows downstream."""
    dx = 200.0 / nx  # 200m domain
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    # V-shaped valley cross-section (along Y) + gentle downstream slope (along X)
    valley_center = 0.5
    valley_half = 0.25
    dist_from_center = np.abs(Y - valley_center)
    bank_height = np.clip((dist_from_center - valley_half) / 0.2, 0, 1) * 8.0
    downstream_slope = (1.0 - X) * 2.0

    z_bed = (downstream_slope + bank_height).astype(np.float32)
    z_bed += _perlin_2d(nx, ny, scale=6.0, seed=7) * 0.3

    # Initial water: pool at upstream end (high X values)
    h0 = np.zeros((nx, ny), dtype=np.float32)
    pool_region = X > 0.7
    water_level = 4.0
    depth = water_level - z_bed
    h0[pool_region] = np.maximum(depth[pool_region], 0.0)

    is_wall = np.zeros((nx, ny), dtype=bool)
    is_wall[0, :] = True
    is_wall[-1, :] = True
    is_wall[:, 0] = True
    is_wall[:, -1] = True

    n_manning = np.full((nx, ny), 0.035, dtype=np.float32)

    return dict(z_bed=z_bed, h0=h0, hu0=np.zeros_like(h0),
                hv0=np.zeros_like(h0), is_wall=is_wall,
                n_manning=n_manning, dx=dx, g=9.81, name="Valley Flood")


def scene_levee_breach(nx=512, ny=512):
    """Levee with a breach — water pours through a gap into floodplain."""
    dx = 100.0 / nx
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    z_bed = np.zeros((nx, ny), dtype=np.float32)
    # Slight floodplain slope
    z_bed += (1.0 - Y) * 0.2

    # Levee: ridge across the middle (Y direction)
    levee_center = 0.4
    levee_width = 0.04
    levee_mask = np.abs(Y - levee_center) < levee_width
    z_bed[levee_mask] = 3.0

    is_wall = np.zeros((nx, ny), dtype=bool)
    is_wall[0, :] = True
    is_wall[-1, :] = True
    is_wall[:, 0] = True
    is_wall[:, -1] = True

    # Breach: gap in the levee near center-X
    breach_x0 = int(0.45 * nx)
    breach_x1 = int(0.55 * nx)
    breach_y0 = int((levee_center - levee_width) * ny)
    breach_y1 = int((levee_center + levee_width) * ny)
    z_bed[breach_x0:breach_x1, breach_y0:breach_y1] = 0.0

    # Water behind the levee
    h0 = np.zeros((nx, ny), dtype=np.float32)
    behind_levee = Y < levee_center - levee_width
    water_level = 2.5
    depth = water_level - z_bed
    h0[behind_levee] = np.maximum(depth[behind_levee], 0.0)

    n_manning = np.full((nx, ny), 0.030, dtype=np.float32)

    return dict(z_bed=z_bed, h0=h0, hu0=np.zeros_like(h0),
                hv0=np.zeros_like(h0), is_wall=is_wall,
                n_manning=n_manning, dx=dx, g=9.81, name="Levee Breach")


def scene_circular_dam_break(nx=512, ny=512):
    """Circular dam break — radially symmetric flood wave."""
    dx = 50.0 / nx
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    z_bed = np.zeros((nx, ny), dtype=np.float32)
    # Gentle bowl shape
    R = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    z_bed += R * 0.2

    # Circular water column in center
    h0 = np.zeros((nx, ny), dtype=np.float32)
    h0[R < 0.15] = 3.0

    is_wall = np.zeros((nx, ny), dtype=bool)
    is_wall[0, :] = True
    is_wall[-1, :] = True
    is_wall[:, 0] = True
    is_wall[:, -1] = True

    n_manning = np.full((nx, ny), 0.020, dtype=np.float32)

    return dict(z_bed=z_bed, h0=h0, hu0=np.zeros_like(h0),
                hv0=np.zeros_like(h0), is_wall=is_wall,
                n_manning=n_manning, dx=dx, g=9.81, name="Circular Dam Break")


def scene_thacker(nx=256, ny=256):
    """Thacker's planar surface in a paraboloid — analytical benchmark.

    The exact solution is a planar water surface oscillating in a
    parabolic bowl.  h0 = max(eta(x,y,0) - z_bed, 0).

    Bowl: z = a*(x^2 + y^2) / L^2
    Analytical period T = 2*pi / omega, omega = sqrt(8*g*a) / L
    """
    L = 5000.0  # domain half-width [m]
    dx = 2 * L / nx
    a = 1.0  # bowl depth parameter [m]
    g = 9.81
    h_0 = 10.0  # mean water depth [m]

    xs = np.linspace(-L, L, nx)
    ys = np.linspace(-L, L, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    R2 = X**2 + Y**2
    z_bed = (a * R2 / L**2).astype(np.float32)

    # Initial water surface: tilted plane
    eta0 = h_0 - a * R2 / L**2 + a * X / L * 0.5
    h0 = np.maximum(eta0 - z_bed, 0.0).astype(np.float32)

    is_wall = np.zeros((nx, ny), dtype=bool)
    n_manning = np.full((nx, ny), 0.0, dtype=np.float32)  # frictionless for benchmark

    return dict(z_bed=z_bed, h0=h0, hu0=np.zeros_like(h0),
                hv0=np.zeros_like(h0), is_wall=is_wall,
                n_manning=n_manning, dx=dx, g=g, name="Thacker Benchmark")


# Scene registry — ordered for keyboard switching (1-6)
SCENES = [
    ("dam_break", scene_dam_break),
    ("urban", scene_urban_flood),
    ("valley", scene_valley),
    ("levee", scene_levee_breach),
    ("circular", scene_circular_dam_break),
    ("thacker", scene_thacker),
]


def get_scene(name, nx=512, ny=512):
    """Look up scene by name or index."""
    for sname, sfunc in SCENES:
        if sname == name:
            return sfunc(nx, ny)
    # Try as integer index
    try:
        idx = int(name)
        if 0 <= idx < len(SCENES):
            return SCENES[idx][1](nx, ny)
    except ValueError:
        pass
    raise ValueError(f"Unknown scene: {name}. Available: {[s[0] for s in SCENES]}")
