"""
Phase 8 — Scene Configuration (single source of truth)

Everything about the scene lives here: asset paths, building/car layout,
simulation domain, physics parameters. Both solver and renderer import this.

Coordinates are in Blender world space (Z-up, meters).
The solver uses its own internal coordinate mapping defined here.
"""

import math

# ===================================================================
# Asset Paths
# ===================================================================
BUILDING_FBX = r"C:\Users\djmax\Downloads\building_residential\source\232323.fbx"
BUILDING_TEX = r"C:\Users\djmax\Downloads\building_residential\textures"
CAR_GLB = r"C:\Users\djmax\Downloads\ferrari-portofino-2018-wwwvecarzcom\source\2018_ferrari_portofino.glb"

# ===================================================================
# Building Layout (Blender world coords, Z-up)
# Measured from FBX import — treat as constants
# ===================================================================
BUILDING_CENTER = (51.1, 13.4, 19.6)      # (x, y, z)
BUILDING_DIMS   = (49.1, 88.3, 44.7)      # (width_x, depth_y, height_z)
GROUND_Z        = -2.72                     # building base / ground level

# Derived bounds
BUILDING_MIN = tuple(c - d / 2 for c, d in zip(BUILDING_CENTER, BUILDING_DIMS))
BUILDING_MAX = tuple(c + d / 2 for c, d in zip(BUILDING_CENTER, BUILDING_DIMS))

# ===================================================================
# Car Layout
# ===================================================================
CAR_LENGTH = 4.5  # Ferrari Portofino, scaled

def car_spots(n_cars=10, seed=42):
    """Deterministic parking spots: (x, y, yaw_degrees).
    All coords in Blender world space."""
    import random
    rng = random.Random(seed)

    cx, cy, _ = BUILDING_CENTER
    dx, dy, _ = BUILDING_DIMS
    front_y = cy - dy / 2
    left_x  = cx - dx / 2
    right_x = cx + dx / 2

    spots = []
    # Front row: 6 cars
    for i in range(6):
        spots.append((cx - 18 + i * 7.0, front_y - 5.0, 90 + rng.uniform(-3, 3)))
    # Left side: 2 cars
    for i in range(2):
        spots.append((left_x - 5.0, cy - 10 + i * 8.0, rng.uniform(-3, 3)))
    # Right side: 2 cars
    for i in range(2):
        spots.append((right_x + 5.0, cy - 5 + i * 8.0, 180 + rng.uniform(-3, 3)))

    return spots[:n_cars]

# ===================================================================
# Simulation Domain
# ===================================================================
# The solver works in its own coordinate system.
# We define the sim domain to enclose the building + a flood region.
#
# Blender world:  building at X~[26..76], Y~[-31..58], Z~[-3..42]
# Sim domain:     a cube centered on the building, large enough for
#                 water inflow from one side.
#
# Mapping: sim_coord = blender_coord - SIM_ORIGIN
# (simple translation, no rotation — both are Z-up in the solver now)

SIM_DOMAIN   = 130.0   # meters, cube side length
FLOOR_MARGIN = 10.0    # meters below ground reserved for boundary zone
BOUND        = 3       # grid cells of boundary padding

SIM_ORIGIN   = (        # bottom-left-back corner in Blender coords
    BUILDING_CENTER[0] - SIM_DOMAIN / 2,
    BUILDING_CENTER[1] - SIM_DOMAIN / 2,
    GROUND_Z - FLOOR_MARGIN,
)

N_GRID       = 128       # grid resolution (64=test, 128=production)
DX           = SIM_DOMAIN / N_GRID

# ===================================================================
# Physics Parameters
# ===================================================================
FPS          = 30
GRAVITY      = 9.8

# --- SWE (2D shallow water for ALL water physics) ---
SWE_NX       = 256       # SWE grid cells in X
SWE_NY       = 256       # SWE grid cells in Y
SWE_DX       = SIM_DOMAIN / SWE_NX
SWE_CFL      = 0.4
SWE_MANNING  = 0.025     # Manning roughness (concrete/asphalt)

SWE_FLOOD = dict(
    inflow_depth    = 5.0,    # meters — deep flash flood
    inflow_velocity = 8.0,    # m/s toward building (+Y direction)
    ramp_time       = 1.5,    # seconds to reach full inflow
)

# --- MPM (3D solids ONLY — building, cars, debris) ---
MPM_MARGIN   = 15.0      # meters padding around solid bounding boxes
MPM_DX       = 1.0       # target MPM cell size (meters)
MPM_DT       = 1e-3      # MPM substep timestep
MPM_SUBSTEPS = 30        # per frame

# Concrete (corotational elastic + Rankine failure)
CONCRETE = dict(
    E       = 10000.0,
    nu      = 0.2,
    rho     = 2.4,        # normalized (2.4x water density)
    tensile_strength = 500.0,  # moderate: intact concrete resists, soaked doesn't
    damage_rate      = 0.2,    # progressive fracture once failure starts
    softening_damage = 0.5,    # damage level where strength halves
    rubble_damage    = 1.0,
    rubble_E         = 500.0,
)

# Car (rigid-elastic, very stiff)
CAR = dict(
    E       = 50000.0,    # very stiff (effectively rigid)
    nu      = 0.3,
    rho     = 3.0,        # normalized
)

# Velocity damping for concrete (prevents numerical explosion)
VEL_DAMPING = 1.0  # gentler damping so rubble falls naturally

# SWE→MPM coupling
COUPLING = dict(
    rho_water = 1.0,      # normalized (matches MPM density scale)
    C_drag    = 2.0,      # moderate drag on submerged solids
    wall_pressure_mult = 2.0,  # mild amplification (accounts for wave impact
                               # and turbulence that SWE can't resolve)
    # Soaking: prolonged submersion weakens concrete. Accelerated for
    # cinematic purposes (real erosion takes hours/days, not seconds).
    # The MECHANISM is real: water weakens base → gravity collapses upper floors.
    soaking_rate = 0.25,  # damage/second for fully submerged concrete
)

# ===================================================================
# Coordinate Helpers
# ===================================================================
def blender_to_sim(bx, by, bz):
    """Convert Blender world coords → sim-local coords."""
    return (bx - SIM_ORIGIN[0], by - SIM_ORIGIN[1], bz - SIM_ORIGIN[2])

def sim_to_blender(sx, sy, sz):
    """Convert sim-local coords → Blender world coords."""
    return (sx + SIM_ORIGIN[0], sy + SIM_ORIGIN[1], sz + SIM_ORIGIN[2])

def building_bounds_sim():
    """Building AABB in sim-local coords."""
    lo = blender_to_sim(*BUILDING_MIN)
    hi = blender_to_sim(*BUILDING_MAX)
    return lo, hi

def car_boxes_sim(n_cars=10):
    """Car AABBs in sim-local coords. Returns list of (center, half_extents, yaw)."""
    half = (CAR_LENGTH / 2, 1.0, 0.75)  # half extents: length/2, width/2, height/2
    boxes = []
    for x, y, yaw in car_spots(n_cars):
        cx, cy, cz = blender_to_sim(x, y, GROUND_Z + half[2])
        boxes.append(((cx, cy, cz), half, math.radians(yaw)))
    return boxes

# ===================================================================
# Export
# ===================================================================
EXPORT_DIR = "./export"
