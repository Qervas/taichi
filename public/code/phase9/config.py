"""Phase 9 — 3D MLS-MPM flood simulation config."""
import os

# Assets
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
MODEL_1 = os.path.join(ASSETS_DIR, "1.glb")
MODEL_1_RENDER = os.path.join(ASSETS_DIR, "1_noground.glb")  # ground removed for rendering
MODEL_2 = os.path.join(ASSETS_DIR, "2.glb")

# Grid
N_GRID = 128          # 128 test, 192 production
BOUND = 3             # boundary padding (grid cells)

# Physics
GRAVITY = (0.0, 0.0, -5.0)   # Z-up
WATER_E = 400.0               # bulk stiffness
WATER_NU = 0.2
DT = 2e-4                     # substep timestep
SUBSTEPS = 25                  # per frame

# Time
FPS = 30
N_FRAMES = 300                 # rising flood — longer sim

# Scene layout — all in normalized [0, 1] domain
# Building: model 1 (wooden house with trees) — centered in domain
BUILDING = dict(
    glb=MODEL_1,
    center_xy=(0.50, 0.50),    # center of domain
    scale=0.38,
)

# Rising flood — uniform rise across most of the domain
FLOOD = dict(
    inject_rate=10000,         # particles per frame (gradual phase)
    z_thickness=0.008,         # injection layer thickness
    max_z_frac=0.85,           # water to ~85% of building height (near roof)
    # Pre-fill phase: thick initial layer so water is connected from frame 0
    prefill_frames=25,         # frames of heavy injection
    prefill_rate=20000,        # particles/frame during pre-fill
    prefill_thickness=0.025,   # thicker layer during pre-fill
)

# Export
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")
MAX_PARTICLES = 4_000_000
