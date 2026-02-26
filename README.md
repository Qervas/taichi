# MLS-MPM Fluid & Structure Simulation

Real-time material point method simulations built with [Taichi](https://github.com/taichi-dev/taichi). Implements the **MLS-MPM** algorithm (Hu et al., SIGGRAPH 2018) for both fluid dynamics and solid mechanics with fracture.

<p align="center">
  <b>2D Dam Break</b> &nbsp;|&nbsp; <b>3D Bridge Destruction</b>
</p>

## Simulations

### Phase 2 — 2D Water (MLS-MPM)

A 2D dam-break simulation using weakly compressible MLS-MPM with equation-of-state pressure.

- **Grid**: 128x128 collocated grid
- **Particles**: ~50K water particles (4 per cell)
- **Pressure model**: Two-sided EOS — `p = K(1 - J)` — resists both compression and expansion
- **Transfer**: APIC with affine matrix C for angular momentum conservation

```bash
python src/phase2_water.py
```

### Phase 3 — 3D Water + Breakable Bridge Pillars

A 3D dam-break where a wall of water impacts four bridge pillars, causing progressive fracture and structural collapse.

- **Grid**: 64x64x64
- **Particles**: 248K water + 21.6K solid (270K total)
- **Water**: Weakly compressible EOS
- **Solid**: Drucker-Prager elastoplasticity with SVD-based return mapping
- **Fracture**: Damage accumulation degrades cohesion, leading to cascading failure and shattering
- **Coupling**: Automatic two-phase coupling through shared grid — no explicit coupling terms needed

```bash
python src/phase3_3d.py
```

## Key Concepts

| Concept | Description |
|---|---|
| **MLS-MPM** | Moving Least Squares Material Point Method — fuses force and velocity scattering into a single P2G loop, eliminating the pressure solver entirely |
| **APIC** | Affine Particle-In-Cell — C matrix preserves angular momentum and doubles as the velocity gradient |
| **EOS** | Equation of State — each particle independently computes pressure from volume ratio J, replacing global Poisson solves |
| **Drucker-Prager** | Yield criterion for granular/rock-like materials — controls when the solid transitions from elastic to plastic deformation |
| **Damage Model** | Accumulated yield strain degrades cohesion over time, creating progressive crack propagation and realistic shattering |

## Controls

| Key | Action |
|---|---|
| `SPACE` | Play / Pause |
| `R` | Reset simulation |
| `ESC` | Quit |
| RMB drag (3D only) | Orbit camera |

## Requirements

- Python 3.8+
- [Taichi](https://github.com/taichi-dev/taichi) >= 1.6.0
- CUDA-capable GPU (recommended)

```bash
pip install taichi
```

## References

- Hu et al., *"A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling"*, ACM Transactions on Graphics (SIGGRAPH), 2018
- Klar et al., *"Drucker-Prager Elastoplasticity for Sand Animation"*, ACM Transactions on Graphics (SIGGRAPH), 2016
- Stomakhin et al., *"A Material Point Method for Snow Simulation"*, ACM Transactions on Graphics (SIGGRAPH), 2013
