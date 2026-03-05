"""
Phase 7 — Real City Data Pipeline (Shenzhen)

Downloads and processes OSM building footprints + DEM elevation into
SWE-compatible arrays, exports building OBJ meshes and car spawn points.

Usage:
    python phases/phase7/city_data.py --area "shenzhen_futian"
    python phases/phase7/city_data.py --bbox 114.05,22.52,114.07,22.54
    python phases/phase7/city_data.py --area "shenzhen_futian" --grid 512 --output ./city_export

Dependencies:
    pip install osmnx rasterio shapely pyproj
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Predefined city areas
# ═══════════════════════════════════════════════════════════════════════
AREAS = {
    "shenzhen_futian": {
        "bbox": (114.055, 22.525, 114.065, 22.535),
        "name": "Shenzhen Futian CBD",
    },
    "shenzhen_nanshan": {
        "bbox": (113.925, 22.515, 113.935, 22.525),
        "name": "Shenzhen Nanshan",
    },
    "shenzhen_luohu": {
        "bbox": (114.105, 22.545, 114.115, 22.555),
        "name": "Shenzhen Luohu",
    },
}

# Default building height when no data is available
DEFAULT_BUILDING_HEIGHT = 12.0
DEFAULT_FLOOR_HEIGHT = 3.0

# Manning roughness values by surface type
MANNING_ROAD = 0.015
MANNING_GRASS = 0.025
MANNING_CONCRETE = 0.012
MANNING_BUILDING = 0.10


# ═══════════════════════════════════════════════════════════════════════
# Building height extraction
# ═══════════════════════════════════════════════════════════════════════
def _get_building_height(tags):
    """Extract building height from OSM tags."""
    if "height" in tags:
        try:
            h = float(str(tags["height"]).replace("m", "").strip())
            return h
        except (ValueError, TypeError):
            pass
    if "building:levels" in tags:
        try:
            levels = int(float(str(tags["building:levels"])))
            return levels * DEFAULT_FLOOR_HEIGHT
        except (ValueError, TypeError):
            pass
    return DEFAULT_BUILDING_HEIGHT


# ═══════════════════════════════════════════════════════════════════════
# OSM data fetching
# ═══════════════════════════════════════════════════════════════════════
def fetch_buildings(bbox):
    """Fetch building footprints from OSM within bounding box.

    Returns list of dicts: {polygon: shapely.Polygon (UTM), height: float, id: int}
    """
    import osmnx as ox
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    west, south, east, north = bbox
    print(f"  Fetching buildings in ({west:.4f},{south:.4f})–({east:.4f},{north:.4f})...")

    gdf = ox.features_from_bbox(bbox=(north, south, east, west),
                                tags={"building": True})

    # Project to UTM for metric coordinates
    # Determine UTM zone from center of bbox
    center_lon = (west + east) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if (south + north) / 2 >= 0 else "south"
    epsg_utm = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)

    buildings = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        elif geom.geom_type == "Polygon":
            polys = [geom]
        else:
            continue

        height = _get_building_height(row)

        for poly in polys:
            utm_poly = shapely_transform(transformer.transform, poly)
            buildings.append({
                "polygon": utm_poly,
                "height": height,
                "id": len(buildings),
            })

    print(f"  Found {len(buildings)} building footprints")
    return buildings, epsg_utm


def fetch_roads(bbox):
    """Fetch road centerlines from OSM for car spawning."""
    import osmnx as ox
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    west, south, east, north = bbox

    center_lon = (west + east) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if (south + north) / 2 >= 0 else "south"
    epsg_utm = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)

    print(f"  Fetching roads...")
    try:
        G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False)
        lines = []
        for _, row in edges.iterrows():
            geom = row.geometry
            if geom is not None:
                utm_geom = shapely_transform(transformer.transform, geom)
                lines.append(utm_geom)
        print(f"  Found {len(lines)} road segments")
        return lines
    except Exception as e:
        print(f"  Warning: Could not fetch roads: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# DEM elevation (fallback: synthetic terrain if SRTM unavailable)
# ═══════════════════════════════════════════════════════════════════════
def fetch_dem(bbox, epsg_utm, nx, ny, domain_size):
    """Try to fetch DEM elevation. Falls back to flat + gentle slope."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject, Resampling

        # Try SRTM via elevation package or local file
        print("  Attempting DEM fetch (SRTM)...")
        # For now, use synthetic terrain as SRTM requires auth
        raise ImportError("Using synthetic DEM")

    except (ImportError, Exception) as e:
        print(f"  Using synthetic DEM: {e}")
        # Gentle slope: higher in NW corner (flood source), lower in SE
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        # Base elevation: Shenzhen is ~5-30m above sea level
        z = 5.0 + (1.0 - X) * 3.0 + (1.0 - Y) * 2.0

        # Add some micro-terrain noise
        rng = np.random.default_rng(42)
        # Simple value noise
        scale = 8
        gx = scale + 2
        gy = scale + 2
        grid = rng.standard_normal((gx, gy))
        xi = np.floor(xs * scale).astype(int)
        yi = np.floor(ys * scale).astype(int)
        xf = xs * scale - xi
        yf = ys * scale - yi
        v00 = grid[xi[:, None], yi[None, :]]
        v10 = grid[xi[:, None] + 1, yi[None, :]]
        v01 = grid[xi[:, None], yi[None, :] + 1]
        v11 = grid[xi[:, None] + 1, yi[None, :] + 1]
        noise = (v00 * (1 - xf[:, None]) * (1 - yf[None, :]) +
                 v10 * xf[:, None] * (1 - yf[None, :]) +
                 v01 * (1 - xf[:, None]) * yf[None, :] +
                 v11 * xf[:, None] * yf[None, :])
        z += noise * 0.3

        return z.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Rasterize buildings onto grid
# ═══════════════════════════════════════════════════════════════════════
def rasterize_buildings(buildings, origin_x, origin_y, nx, ny, dx):
    """Rasterize building footprints onto the SWE grid.

    Returns:
        z_building: height added to z_bed at building cells
        is_building: boolean mask of building cells
    """
    from shapely.geometry import box as shapely_box

    z_building = np.zeros((nx, ny), dtype=np.float32)
    is_building = np.zeros((nx, ny), dtype=bool)

    for bldg in buildings:
        poly = bldg["polygon"]
        height = bldg["height"]

        # Get bounding box of polygon in grid coords
        minx, miny, maxx, maxy = poly.bounds
        i0 = max(0, int((minx - origin_x) / dx))
        i1 = min(nx, int((maxx - origin_x) / dx) + 1)
        j0 = max(0, int((miny - origin_y) / dx))
        j1 = min(ny, int((maxy - origin_y) / dx) + 1)

        for i in range(i0, i1):
            for j in range(j0, j1):
                cx = origin_x + (i + 0.5) * dx
                cy = origin_y + (j + 0.5) * dx
                from shapely.geometry import Point
                if poly.contains(Point(cx, cy)):
                    z_building[i, j] = max(z_building[i, j], height)
                    is_building[i, j] = True

    return z_building, is_building


# ═══════════════════════════════════════════════════════════════════════
# Export buildings to OBJ (batched)
# ═══════════════════════════════════════════════════════════════════════
def export_buildings_obj(buildings, origin_x, origin_y, output_dir, batch_size=50):
    """Export building footprints as extruded OBJ meshes for Blender import.

    Returns list of OBJ file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    obj_paths = []

    for batch_idx in range(0, len(buildings), batch_size):
        batch = buildings[batch_idx:batch_idx + batch_size]
        obj_path = os.path.join(output_dir, f"buildings_{batch_idx:04d}.obj")

        with open(obj_path, "w") as f:
            f.write(f"# Phase 7 — Shenzhen buildings batch {batch_idx}\n")
            vert_offset = 0

            for bldg in batch:
                poly = bldg["polygon"]
                height = bldg["height"]
                bldg_id = bldg["id"]

                # Get exterior ring coordinates (in UTM, relative to origin)
                coords = list(poly.exterior.coords)
                if len(coords) < 4:
                    continue

                # Remove closing vertex if it duplicates the first
                if coords[-1] == coords[0]:
                    coords = coords[:-1]

                n_verts = len(coords)
                f.write(f"o building_{bldg_id}\n")

                # Bottom vertices
                for x, y in coords:
                    rx, ry = x - origin_x, y - origin_y
                    f.write(f"v {rx:.3f} 0.0 {ry:.3f}\n")

                # Top vertices
                for x, y in coords:
                    rx, ry = x - origin_x, y - origin_y
                    f.write(f"v {rx:.3f} {height:.3f} {ry:.3f}\n")

                # UV coordinates for texture mapping
                for i in range(n_verts):
                    u = i / n_verts
                    f.write(f"vt {u:.4f} 0.0\n")  # bottom
                for i in range(n_verts):
                    u = i / n_verts
                    f.write(f"vt {u:.4f} 1.0\n")  # top

                # Faces — walls (quads)
                base = vert_offset + 1  # OBJ is 1-indexed
                for i in range(n_verts):
                    i_next = (i + 1) % n_verts
                    b0 = base + i
                    b1 = base + i_next
                    t0 = base + n_verts + i
                    t1 = base + n_verts + i_next
                    f.write(f"f {b0} {b1} {t1} {t0}\n")

                # Top face (fan triangulation)
                top_base = base + n_verts
                for i in range(1, n_verts - 1):
                    f.write(f"f {top_base} {top_base + i} {top_base + i + 1}\n")

                vert_offset += 2 * n_verts

        obj_paths.append(obj_path)
        print(f"  Exported {obj_path} ({len(batch)} buildings)")

    return obj_paths


# ═══════════════════════════════════════════════════════════════════════
# Car spawn on road centerlines
# ═══════════════════════════════════════════════════════════════════════
def spawn_cars_on_roads(roads, origin_x, origin_y, domain_size, n_cars=30, seed=123):
    """Place cars along road centerlines.

    Returns:
        positions: (N, 2) car positions in grid coordinates [meters]
        yaws: (N,) yaw angles [radians]
    """
    rng = np.random.default_rng(seed)

    if not roads:
        # Fallback: random placement on a grid pattern
        print(f"  No roads available, placing {n_cars} cars randomly on grid")
        positions = rng.uniform(0.1 * domain_size, 0.9 * domain_size, size=(n_cars, 2))
        yaws = rng.uniform(0, 2 * np.pi, size=n_cars)
        return positions.astype(np.float32), yaws.astype(np.float32)

    # Collect candidate points from road segments
    candidate_points = []
    candidate_yaws = []

    for line in roads:
        coords = list(line.coords)
        for k in range(len(coords) - 1):
            x0, y0 = coords[k][0] - origin_x, coords[k][1] - origin_y
            x1, y1 = coords[k + 1][0] - origin_x, coords[k + 1][1] - origin_y

            # Check if segment is within domain
            if not (0 < x0 < domain_size and 0 < y0 < domain_size):
                continue
            if not (0 < x1 < domain_size and 0 < y1 < domain_size):
                continue

            seg_len = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            if seg_len < 5.0:  # skip very short segments
                continue

            yaw = np.arctan2(y1 - y0, x1 - x0)

            # Place cars every ~30m along the segment
            n_on_seg = max(1, int(seg_len / 30.0))
            for t_idx in range(n_on_seg):
                t = (t_idx + 0.5) / n_on_seg
                cx = x0 + t * (x1 - x0) + rng.uniform(-2, 2)
                cy = y0 + t * (y1 - y0) + rng.uniform(-2, 2)
                candidate_points.append([cx, cy])
                candidate_yaws.append(yaw + rng.uniform(-0.1, 0.1))

    if len(candidate_points) == 0:
        print(f"  No valid road segments, placing cars randomly")
        positions = rng.uniform(0.1 * domain_size, 0.9 * domain_size, size=(n_cars, 2))
        yaws = rng.uniform(0, 2 * np.pi, size=n_cars)
        return positions.astype(np.float32), yaws.astype(np.float32)

    # Randomly sample n_cars from candidates
    candidate_points = np.array(candidate_points)
    candidate_yaws = np.array(candidate_yaws)
    n_avail = len(candidate_points)

    if n_avail >= n_cars:
        indices = rng.choice(n_avail, size=n_cars, replace=False)
    else:
        indices = rng.choice(n_avail, size=n_cars, replace=True)

    positions = candidate_points[indices].astype(np.float32)
    yaws = candidate_yaws[indices].astype(np.float32)

    print(f"  Spawned {n_cars} cars on roads")
    return positions, yaws


# ═══════════════════════════════════════════════════════════════════════
# Synthetic scene (no network access needed)
# ═══════════════════════════════════════════════════════════════════════
def generate_synthetic_scene(nx=512, ny=512, domain_size=1000.0, n_cars=30, seed=42):
    """Generate a synthetic Shenzhen-like city scene without OSM access.

    Produces the same output format as the OSM pipeline for offline testing.
    """
    print("Generating synthetic Shenzhen city scene...")
    rng = np.random.default_rng(seed)
    dx = domain_size / nx

    # --- Terrain ---
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Base elevation: gentle slope NW → SE
    z_bed = 5.0 + (1.0 - X) * 3.0 + (1.0 - Y) * 2.0

    # Micro-terrain noise
    scale = 8
    gx = scale + 2
    grid = rng.standard_normal((gx, gx))
    xi = np.floor(xs * scale).astype(int)
    yi = np.floor(ys * scale).astype(int)
    xf = xs * scale - xi
    yf = ys * scale - yi
    v00 = grid[xi[:, None], yi[None, :]]
    v10 = grid[xi[:, None] + 1, yi[None, :]]
    v01 = grid[xi[:, None], yi[None, :] + 1]
    v11 = grid[xi[:, None] + 1, yi[None, :] + 1]
    noise = (v00 * (1 - xf[:, None]) * (1 - yf[None, :]) +
             v10 * xf[:, None] * (1 - yf[None, :]) +
             v01 * (1 - xf[:, None]) * yf[None, :] +
             v11 * xf[:, None] * yf[None, :])
    z_bed += noise * 0.3
    z_bed = z_bed.astype(np.float32)

    # --- River channel through the domain (wide, smooth banks) ---
    river_center_y = 0.5
    river_half = 0.08  # ~80m half-width = 160m total river
    river_dist = np.abs(Y - river_center_y)
    river_mask = river_dist < river_half
    # Smooth parabolic cross-section instead of sharp trench
    river_depth = 2.0 * np.maximum(0, 1.0 - (river_dist / river_half) ** 2)
    z_bed -= river_depth  # gradual slope into river, no sharp wall

    # --- Street grid and buildings ---
    is_wall = np.zeros((nx, ny), dtype=bool)
    n_manning = np.full((nx, ny), MANNING_ROAD, dtype=np.float32)
    n_manning[river_mask] = MANNING_GRASS

    # Building grid: blocks every ~50-80m
    street_w = 15.0  # street width in meters
    street_w_cells = max(2, int(street_w / dx))

    buildings_info = []  # for OBJ export
    block_spacing = 60.0  # meters between block starts
    block_spacing_cells = int(block_spacing / dx)

    bx = int(0.12 * nx)
    block_id = 0
    while bx < int(0.92 * nx):
        by = int(0.08 * ny)
        bx_size = int(rng.uniform(20, 40) / dx)  # building width in cells

        while by < int(0.92 * ny):
            by_size = int(rng.uniform(20, 40) / dx)

            # Skip river zone (wide buffer so no wall effect)
            by_center = (by + by_size / 2) / ny
            if abs(by_center - river_center_y) < river_half + 0.02:
                by += by_size + street_w_cells
                continue

            # Skip upstream reservoir zone
            bx_norm = bx / nx
            by_norm = by / ny
            if bx_norm < 0.20 and by_norm < 0.25:
                by += by_size + street_w_cells
                continue

            # Occasional parks
            block_id += 1
            if block_id % 8 == 0:
                by += by_size + street_w_cells
                continue

            x0 = max(2, min(bx, nx - 2))
            x1 = max(2, min(bx + bx_size, nx - 2))
            y0 = max(2, min(by, ny - 2))
            y1 = max(2, min(by + by_size, ny - 2))

            if x1 > x0 + 2 and y1 > y0 + 2:
                height = rng.uniform(10.0, 60.0)
                # Taller buildings in CBD area (center)
                dist_center = np.sqrt((bx_norm - 0.5)**2 + (by_norm - 0.5)**2)
                if dist_center < 0.2:
                    height = rng.uniform(30.0, 80.0)

                z_bed[x0:x1, y0:y1] += height
                is_wall[x0:x1, y0:y1] = True
                n_manning[x0:x1, y0:y1] = MANNING_BUILDING

                # Store for OBJ export
                cx = x0 * dx
                cy = y0 * dx
                w = (x1 - x0) * dx
                d = (y1 - y0) * dx
                buildings_info.append({
                    "x": cx, "y": cy, "w": w, "d": d,
                    "height": height, "id": block_id,
                })

            by += by_size + street_w_cells + int(rng.uniform(0, 8))
        bx += bx_size + street_w_cells + int(rng.uniform(0, 8))

    # No boundary walls — water flows off edges naturally via open BCs

    # --- Initial water: upstream flood reservoir ---
    h0 = np.zeros((nx, ny), dtype=np.float32)
    # Reservoir in NW corner (high elevation side)
    res_x = int(0.15 * nx)
    res_y = int(0.20 * ny)
    h0[:res_x, :res_y] = 5.0

    # Some water in the river
    h0[river_mask] = np.maximum(1.0, h0[river_mask])

    # --- Car spawning on streets ---
    car_positions = []
    car_yaws = []
    # Place cars along streets (gaps between buildings)
    for _ in range(n_cars * 5):  # oversample, then pick
        cx = rng.uniform(0.15, 0.85) * domain_size
        cy = rng.uniform(0.15, 0.85) * domain_size
        ci = int(cx / dx)
        cj = int(cy / dx)
        if 0 <= ci < nx and 0 <= cj < ny and not is_wall[ci, cj]:
            car_positions.append([cx, cy])
            car_yaws.append(rng.uniform(0, 2 * np.pi))
            if len(car_positions) >= n_cars:
                break

    # Pad if not enough
    while len(car_positions) < n_cars:
        car_positions.append([rng.uniform(100, 900), rng.uniform(100, 900)])
        car_yaws.append(rng.uniform(0, 2 * np.pi))

    car_positions = np.array(car_positions[:n_cars], dtype=np.float32)
    car_yaws = np.array(car_yaws[:n_cars], dtype=np.float32)

    print(f"  Grid: {nx}x{ny}, dx={dx:.2f}m, domain={domain_size:.0f}m")
    print(f"  Buildings: {len(buildings_info)}")
    print(f"  Cars: {n_cars}")

    return {
        "z_bed": z_bed,
        "h0": h0,
        "hu0": np.zeros((nx, ny), dtype=np.float32),
        "hv0": np.zeros((nx, ny), dtype=np.float32),
        "is_wall": is_wall,
        "n_manning": n_manning,
        "dx": dx,
        "g": 9.81,
        "name": "Shenzhen Synthetic",
        "car_positions": car_positions,
        "car_yaws": car_yaws,
        "buildings_info": buildings_info,
        "domain_size": domain_size,
    }


# ═══════════════════════════════════════════════════════════════════════
# Export synthetic buildings to OBJ
# ═══════════════════════════════════════════════════════════════════════
def export_synthetic_buildings_obj(buildings_info, output_dir, batch_size=50):
    """Export synthetic box buildings as OBJ files."""
    os.makedirs(output_dir, exist_ok=True)
    obj_paths = []

    for batch_idx in range(0, len(buildings_info), batch_size):
        batch = buildings_info[batch_idx:batch_idx + batch_size]
        obj_path = os.path.join(output_dir, f"buildings_{batch_idx:04d}.obj")

        with open(obj_path, "w") as f:
            f.write(f"# Phase 7 — Synthetic buildings batch {batch_idx}\n")
            vert_offset = 0

            for bldg in batch:
                x, y, w, d = bldg["x"], bldg["y"], bldg["w"], bldg["d"]
                height = bldg["height"]
                bid = bldg["id"]

                f.write(f"o building_{bid}\n")

                # 8 vertices of a box
                # Bottom: 4 corners
                f.write(f"v {x:.3f} 0.0 {y:.3f}\n")       # 0: front-left-bottom
                f.write(f"v {x+w:.3f} 0.0 {y:.3f}\n")     # 1: front-right-bottom
                f.write(f"v {x+w:.3f} 0.0 {y+d:.3f}\n")   # 2: back-right-bottom
                f.write(f"v {x:.3f} 0.0 {y+d:.3f}\n")     # 3: back-left-bottom
                # Top: 4 corners
                f.write(f"v {x:.3f} {height:.3f} {y:.3f}\n")       # 4
                f.write(f"v {x+w:.3f} {height:.3f} {y:.3f}\n")     # 5
                f.write(f"v {x+w:.3f} {height:.3f} {y+d:.3f}\n")   # 6
                f.write(f"v {x:.3f} {height:.3f} {y+d:.3f}\n")     # 7

                # UV coordinates
                for u, v in [(0,0),(1,0),(1,1),(0,1),(0,0),(1,0),(1,1),(0,1)]:
                    f.write(f"vt {u} {v}\n")

                b = vert_offset + 1  # OBJ 1-indexed
                # Front wall
                f.write(f"f {b} {b+1} {b+5} {b+4}\n")
                # Right wall
                f.write(f"f {b+1} {b+2} {b+6} {b+5}\n")
                # Back wall
                f.write(f"f {b+2} {b+3} {b+7} {b+6}\n")
                # Left wall
                f.write(f"f {b+3} {b} {b+4} {b+7}\n")
                # Top face
                f.write(f"f {b+4} {b+5} {b+6} {b+7}\n")

                vert_offset += 8

        obj_paths.append(obj_path)
        print(f"  Exported {obj_path} ({len(batch)} buildings)")

    return obj_paths


# ═══════════════════════════════════════════════════════════════════════
# Full OSM pipeline
# ═══════════════════════════════════════════════════════════════════════
def generate_osm_scene(bbox, nx=512, ny=512, n_cars=30):
    """Full pipeline: fetch OSM data, rasterize, export.

    Returns same dict format as generate_synthetic_scene().
    """
    from pyproj import Transformer

    west, south, east, north = bbox
    print(f"Generating scene from OSM: ({west:.4f},{south:.4f})–({east:.4f},{north:.4f})")

    # Fetch data
    buildings, epsg_utm = fetch_buildings(bbox)
    roads = fetch_roads(bbox)

    # Compute UTM origin and domain size
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
    x_min, y_min = transformer.transform(west, south)
    x_max, y_max = transformer.transform(east, north)

    domain_x = x_max - x_min
    domain_y = y_max - y_min
    domain_size = max(domain_x, domain_y)
    dx = domain_size / nx

    print(f"  Domain: {domain_x:.0f}m x {domain_y:.0f}m, dx={dx:.2f}m")

    # DEM
    z_bed = fetch_dem(bbox, epsg_utm, nx, ny, domain_size)

    # Rasterize buildings
    z_bldg, is_building = rasterize_buildings(buildings, x_min, y_min, nx, ny, dx)
    z_bed += z_bldg

    is_wall = is_building.copy()
    is_wall[0, :] = True
    is_wall[-1, :] = True
    is_wall[:, 0] = True
    is_wall[:, -1] = True

    # Manning roughness
    n_manning = np.full((nx, ny), MANNING_ROAD, dtype=np.float32)
    n_manning[is_building] = MANNING_BUILDING

    # Initial water: upstream reservoir
    h0 = np.zeros((nx, ny), dtype=np.float32)
    res_x = int(0.15 * nx)
    res_y = int(0.20 * ny)
    h0[:res_x, :res_y] = 5.0

    # Spawn cars
    car_positions, car_yaws = spawn_cars_on_roads(
        roads, x_min, y_min, domain_size, n_cars=n_cars
    )

    return {
        "z_bed": z_bed,
        "h0": h0,
        "hu0": np.zeros((nx, ny), dtype=np.float32),
        "hv0": np.zeros((nx, ny), dtype=np.float32),
        "is_wall": is_wall,
        "n_manning": n_manning,
        "dx": dx,
        "g": 9.81,
        "name": f"Shenzhen OSM ({west:.3f},{south:.3f})",
        "car_positions": car_positions,
        "car_yaws": car_yaws,
        "buildings": buildings,
        "origin_x": x_min,
        "origin_y": y_min,
        "domain_size": domain_size,
    }


# ═══════════════════════════════════════════════════════════════════════
# Save scene data
# ═══════════════════════════════════════════════════════════════════════
def save_scene(scene, output_dir):
    """Save scene arrays and metadata to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Save numpy arrays
    np.save(os.path.join(output_dir, "z_bed.npy"), scene["z_bed"])
    np.save(os.path.join(output_dir, "h0.npy"), scene["h0"])
    np.save(os.path.join(output_dir, "hu0.npy"), scene["hu0"])
    np.save(os.path.join(output_dir, "hv0.npy"), scene["hv0"])
    np.save(os.path.join(output_dir, "is_wall.npy"), scene["is_wall"])
    np.save(os.path.join(output_dir, "n_manning.npy"), scene["n_manning"])
    np.save(os.path.join(output_dir, "car_positions.npy"), scene["car_positions"])
    np.save(os.path.join(output_dir, "car_yaws.npy"), scene["car_yaws"])

    # Export building OBJs
    buildings_dir = os.path.join(output_dir, "buildings")
    if "buildings_info" in scene:
        obj_paths = export_synthetic_buildings_obj(
            scene["buildings_info"], buildings_dir
        )
    elif "buildings" in scene:
        obj_paths = export_buildings_obj(
            scene["buildings"], scene["origin_x"], scene["origin_y"], buildings_dir
        )
    else:
        obj_paths = []

    # Save metadata
    meta = {
        "name": scene["name"],
        "dx": float(scene["dx"]),
        "g": float(scene["g"]),
        "nx": int(scene["z_bed"].shape[0]),
        "ny": int(scene["z_bed"].shape[1]),
        "domain_size": float(scene.get("domain_size", scene["z_bed"].shape[0] * scene["dx"])),
        "n_cars": int(len(scene["car_positions"])),
        "building_objs": [os.path.relpath(p, output_dir) for p in obj_paths],
    }
    with open(os.path.join(output_dir, "scene_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nScene saved to {output_dir}/")
    print(f"  z_bed.npy, h0.npy, is_wall.npy, n_manning.npy")
    print(f"  car_positions.npy ({meta['n_cars']} cars)")
    print(f"  {len(obj_paths)} building OBJ files")
    print(f"  scene_meta.json")

    return meta


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 7 — City Data Pipeline")
    parser.add_argument("--area", default="shenzhen_futian",
                        help=f"Predefined area name: {list(AREAS.keys())}")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Custom bbox: west,south,east,north (WGS84)")
    parser.add_argument("--grid", type=int, default=512,
                        help="Grid resolution NxN (default: 512)")
    parser.add_argument("--domain", type=float, default=1000.0,
                        help="Domain size in meters (for synthetic mode)")
    parser.add_argument("--cars", type=int, default=30,
                        help="Number of cars to spawn")
    parser.add_argument("--output", default="./city_export",
                        help="Output directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic scene (no OSM fetch)")
    args = parser.parse_args()

    nx = ny = args.grid

    if args.synthetic or args.bbox is None:
        # Use synthetic scene (works offline)
        scene = generate_synthetic_scene(
            nx=nx, ny=ny,
            domain_size=args.domain,
            n_cars=args.cars,
        )
    else:
        # Parse bbox
        if args.bbox:
            parts = [float(x) for x in args.bbox.split(",")]
            bbox = tuple(parts)
        else:
            if args.area not in AREAS:
                print(f"Unknown area: {args.area}. Available: {list(AREAS.keys())}")
                sys.exit(1)
            bbox = AREAS[args.area]["bbox"]

        try:
            scene = generate_osm_scene(bbox, nx=nx, ny=ny, n_cars=args.cars)
        except ImportError as e:
            print(f"OSM dependencies not available ({e}), falling back to synthetic")
            scene = generate_synthetic_scene(
                nx=nx, ny=ny, domain_size=args.domain, n_cars=args.cars
            )

    save_scene(scene, args.output)


if __name__ == "__main__":
    main()
