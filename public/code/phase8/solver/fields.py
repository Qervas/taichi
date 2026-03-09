"""
Taichi field allocations — all simulation state lives here.

Separated so that kernels.py and engine.py can both import fields
without circular dependencies.
"""

import taichi as ti

# Material IDs
WATER    = 0
CONCRETE = 1


def allocate(n_grid, max_particles):
    """Allocate all Taichi fields. Returns a namespace dict."""
    dim = 3
    F = {}

    # Per-particle
    F["x"]        = ti.Vector.field(dim, float, max_particles)
    F["v"]        = ti.Vector.field(dim, float, max_particles)
    F["C"]        = ti.Matrix.field(dim, dim, float, max_particles)
    F["dg"]       = ti.Matrix.field(dim, dim, float, max_particles)  # deformation gradient
    F["Jp"]       = ti.field(float, max_particles)                   # J for EOS
    F["material"] = ti.field(int, max_particles)
    F["damage"]   = ti.field(float, max_particles)
    F["chunk_id"] = ti.field(int, max_particles)
    F["mass"]     = ti.field(float, max_particles)
    F["used"]     = ti.field(int, max_particles)

    # Grid
    F["grid_v"] = ti.Vector.field(dim, float, (n_grid, n_grid, n_grid))
    F["grid_m"] = ti.field(float, (n_grid, n_grid, n_grid))

    # Scalars
    F["n_particles"] = ti.field(int, ())
    F["flood_vx"]    = ti.field(float, ())

    return F
