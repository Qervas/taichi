"""Phase 10 — 3D MLS-MPM flood with reconstructed building."""
import os

# Assets
MODELS_DIR = os.path.expanduser("~/Downloads/models")
BUILDING_OBJ = os.path.join(MODELS_DIR, "0_1_cluster_texture.obj")

# Grid
N_GRID = 128
BOUND = 3

# Physics
GRAVITY = (0.0, 0.0, -5.0)
WATER_E = 400.0
WATER_NU = 0.2
DT = 2e-4
SUBSTEPS = 25

# Time
FPS = 30
N_FRAMES = 300

# Scene layout — building centered in [0, 1] domain
BUILDING = dict(
    obj=BUILDING_OBJ,
    center_xy=(0.50, 0.50),
    scale=0.38,
    z_up=True,
)

# Rising flood
FLOOD = dict(
    inject_rate=10000,
    z_thickness=0.008,
    max_z_frac=0.85,
    prefill_frames=25,
    prefill_rate=20000,
    prefill_thickness=0.025,
)

# Export
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")
MAX_PARTICLES = 4_000_000
