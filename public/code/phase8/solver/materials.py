"""
Constitutive models — material math only, no Taichi fields.

These are helper functions called from within Taichi kernels.
They take material parameters and deformation state, return stress.
"""

import taichi as ti


@ti.func
def water_pressure(Jp: float, E: float) -> float:
    """Weakly compressible EOS: pressure from volume ratio."""
    return E * (1.0 - Jp)


@ti.func
def hencky_stress(dg: ti.Matrix, mu: float, la: float):
    """Corotational Hencky strain → Kirchhoff stress."""
    U, sig, V = ti.svd(dg)
    # Log strain
    log_sig = ti.Vector([ti.log(max(sig[i, i], 1e-6)) for i in range(3)])
    # Kirchhoff stress in rotated frame
    tau_diag = 2.0 * mu * log_sig + la * log_sig.sum() * ti.Vector([1.0, 1.0, 1.0])
    return U, sig, V, tau_diag, log_sig


@ti.func
def rankine_failure(tau_diag: ti.Matrix, log_sig: ti.Matrix,
                    tensile_strength: float, damage: float,
                    damage_rate: float, softening_damage: float):
    """Rankine tensile failure criterion. Returns (new_tau_diag, new_damage, failed)."""
    new_damage = damage
    failed = 0
    new_tau = tau_diag

    max_principal = ti.max(tau_diag[0], ti.max(tau_diag[1], tau_diag[2]))

    # Softening: reduce strength as damage accumulates
    effective_strength = tensile_strength * ti.max(0.0, 1.0 - damage / softening_damage)

    if max_principal > effective_strength:
        new_damage = damage + damage_rate
        failed = 1
        # Clamp tensile stress to zero on failed axes
        for i in ti.static(range(3)):
            if new_tau[i] > 0:
                new_tau[i] = 0.0

    return new_tau, new_damage, failed
