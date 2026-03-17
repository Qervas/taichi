"""Phase 11 — FLIP flood simulation config.

SOTA pipeline:
- FLIP/PIC solver on MAC staggered grid
- MGPCG pressure Poisson solver (multigrid preconditioned)
- SplashSurf surface reconstruction
- Multi-GPU Blender Cycles rendering
"""
import os

# Grid — normalized [0,1] domain
N_GRID = 192          # 192^3 cells, dx ≈ 5.2mm
BOUND = 3

# Physics — FLIP solver
GRAVITY = (0.0, 0.0, -9.81)       # real Earth gravity
FRAME_DT = 5e-3                    # physical time per animation frame
FLIP_SUBSTEPS = 1                  # substeps per frame (CFL~1.9 at v=2, very safe for FLIP)
FLIP_RATIO = 0.97                  # 97% FLIP / 3% PIC (low dissipation)

# Pressure solver
PCG_MAX_ITER = 500
PCG_TOL = 1e-6

# Time
FPS = 30
N_FRAMES = 300

# Building — Phase 10 model (loaded from server ~/Downloads/models/)
BUILDING_OBJ = "6_4_cluster_texture.obj"
BUILDING_SCALE = 0.40       # fraction of [0,1] domain occupied by longest axis

# Inundation flood — directional inflow from left boundary (+X direction)
FLOOD = dict(
    inject_rate=15000,           # particles per frame (steady state)
    inflow_width=0.06,           # width of injection slab in X
    inflow_velocity=2.0,         # horizontal velocity in +X (normalized domain units/s)
    max_z_frac=0.55,             # water to 55% of building height
    prefill_frames=5,            # fast fill at start
    prefill_rate=30000,          # higher rate during prefill
)

# Foam detection
FOAM_V_THRESH = 1.5        # min |v| for foam generation
FOAM_DECAY = 0.92           # foam persistence between substeps

# Export
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")
MAX_PARTICLES = 10_000_000

# Surface reconstruction (for mesh_surface.py)
MESH_GRID_RES = 512         # high-res density field
MESH_SIGMA = 0.7            # tight kernel — less blobby
MESH_SMOOTH_ITER = 3
MESH_SMOOTH_FACTOR = 0.5
