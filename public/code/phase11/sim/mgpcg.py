"""Multigrid-Preconditioned Conjugate Gradient (MGPCG) pressure solver.

Solves the pressure Poisson equation on a 3D grid:
    A * p = b
where A is the Laplacian with free-surface (Dirichlet p=0 at AIR)
and solid-wall (Neumann dp/dn=0 at SOLID) boundary conditions.

Architecture:
- 4-level V-cycle multigrid preconditioner
- Red-black Gauss-Seidel smoother
- Compressed symmetric Laplacian (Adiag + Ax + Ay + Az)

Reference: Power-PIC (g1n0st, SIGGRAPH 2022), LHCSim (Robslhc)
"""
import taichi as ti

FLUID = 0
AIR = 1
SOLID = 2


@ti.data_oriented
class MGPCGSolver:
    def __init__(self, n, n_mg_levels=4, pre_post_smooth=2, bottom_smooth=10):
        self.n = n
        self.n_mg = n_mg_levels
        self.pre_post_smooth = pre_post_smooth
        self.bottom_smooth = bottom_smooth

        # Per-level fields
        self.r = []       # residual
        self.z = []       # preconditioned residual / correction
        self.Adiag = []   # diagonal coefficient
        self.Ax = []      # off-diagonal +x coupling
        self.Ay = []      # off-diagonal +y coupling
        self.Az = []      # off-diagonal +z coupling
        self.grid_type = []

        for l in range(n_mg_levels):
            nl = n >> l  # n, n/2, n/4, n/8
            self.r.append(ti.field(float, shape=(nl, nl, nl)))
            self.z.append(ti.field(float, shape=(nl, nl, nl)))
            self.Adiag.append(ti.field(float, shape=(nl, nl, nl)))
            self.Ax.append(ti.field(float, shape=(nl, nl, nl)))
            self.Ay.append(ti.field(float, shape=(nl, nl, nl)))
            self.Az.append(ti.field(float, shape=(nl, nl, nl)))
            self.grid_type.append(ti.field(int, shape=(nl, nl, nl)))

        # Top-level CG fields
        self.p_vec = ti.field(float, shape=(n, n, n))   # search direction
        self.Ap = ti.field(float, shape=(n, n, n))       # A * p
        self.x_vec = ti.field(float, shape=(n, n, n))    # solution (pressure)
        self.b_vec = ti.field(float, shape=(n, n, n))    # RHS (divergence)

        # Scalar accumulators
        self.sum = ti.field(float, shape=())

    # ------------------------------------------------------------------
    # Laplacian helpers — fields passed as templates, not indexed by l
    # ------------------------------------------------------------------
    @ti.func
    def _nb_sum(self, Ax: ti.template(), Ay: ti.template(), Az: ti.template(),
                z: ti.template(), I: ti.template(), nl: int) -> float:
        """Off-diagonal contribution: sum of A(I,J)*z(J) for neighbors J."""
        ret = 0.0
        if I[0] > 0:
            ret += Ax[I[0] - 1, I[1], I[2]] * z[I[0] - 1, I[1], I[2]]
        if I[0] < nl - 1:
            ret += Ax[I] * z[I[0] + 1, I[1], I[2]]
        if I[1] > 0:
            ret += Ay[I[0], I[1] - 1, I[2]] * z[I[0], I[1] - 1, I[2]]
        if I[1] < nl - 1:
            ret += Ay[I] * z[I[0], I[1] + 1, I[2]]
        if I[2] > 0:
            ret += Az[I[0], I[1], I[2] - 1] * z[I[0], I[1], I[2] - 1]
        if I[2] < nl - 1:
            ret += Az[I] * z[I[0], I[1], I[2] + 1]
        return ret

    # ------------------------------------------------------------------
    # Build LHS (Laplacian coefficients) for level 0
    # ------------------------------------------------------------------
    @ti.kernel
    def _build_A_level0(self, cell_type: ti.template(),
                        Adiag: ti.template(), Ax: ti.template(),
                        Ay: ti.template(), Az: ti.template(),
                        n: int, scale_A: float):
        for i, j, k in Adiag:
            Adiag[i, j, k] = 0.0
            Ax[i, j, k] = 0.0
            Ay[i, j, k] = 0.0
            Az[i, j, k] = 0.0

        for i, j, k in cell_type:
            if cell_type[i, j, k] == FLUID:
                if i + 1 < n:
                    if cell_type[i + 1, j, k] == FLUID:
                        Adiag[i, j, k] += scale_A
                        Ax[i, j, k] = -scale_A
                    elif cell_type[i + 1, j, k] == AIR:
                        Adiag[i, j, k] += scale_A
                if i - 1 >= 0:
                    if cell_type[i - 1, j, k] != SOLID:
                        Adiag[i, j, k] += scale_A
                if j + 1 < n:
                    if cell_type[i, j + 1, k] == FLUID:
                        Adiag[i, j, k] += scale_A
                        Ay[i, j, k] = -scale_A
                    elif cell_type[i, j + 1, k] == AIR:
                        Adiag[i, j, k] += scale_A
                if j - 1 >= 0:
                    if cell_type[i, j - 1, k] != SOLID:
                        Adiag[i, j, k] += scale_A
                if k + 1 < n:
                    if cell_type[i, j, k + 1] == FLUID:
                        Adiag[i, j, k] += scale_A
                        Az[i, j, k] = -scale_A
                    elif cell_type[i, j, k + 1] == AIR:
                        Adiag[i, j, k] += scale_A
                if k - 1 >= 0:
                    if cell_type[i, j, k - 1] != SOLID:
                        Adiag[i, j, k] += scale_A

    # ------------------------------------------------------------------
    # Multigrid coarsening — all fields passed as templates
    # ------------------------------------------------------------------
    @ti.kernel
    def _coarsen_grid_type(self, fine_type: ti.template(),
                           coarse_type: ti.template(), nl: int):
        for i, j, k in ti.ndrange(nl, nl, nl):
            has_fluid = 0
            has_air = 0
            for di, dj, dk in ti.static(ti.ndrange(2, 2, 2)):
                fi, fj, fk = i * 2 + di, j * 2 + dj, k * 2 + dk
                ct = fine_type[fi, fj, fk]
                if ct == FLUID:
                    has_fluid = 1
                elif ct == AIR:
                    has_air = 1
            if has_fluid:
                coarse_type[i, j, k] = FLUID
            elif has_air:
                coarse_type[i, j, k] = AIR
            else:
                coarse_type[i, j, k] = SOLID

    @ti.kernel
    def _build_A_coarse(self, gt: ti.template(),
                        Adiag: ti.template(), Ax: ti.template(),
                        Ay: ti.template(), Az: ti.template(),
                        nl: int, scale_A: float):
        for i, j, k in ti.ndrange(nl, nl, nl):
            Adiag[i, j, k] = 0.0
            Ax[i, j, k] = 0.0
            Ay[i, j, k] = 0.0
            Az[i, j, k] = 0.0

        for i, j, k in ti.ndrange(nl, nl, nl):
            if gt[i, j, k] == FLUID:
                if i + 1 < nl:
                    if gt[i + 1, j, k] == FLUID:
                        Adiag[i, j, k] += scale_A
                        Ax[i, j, k] = -scale_A
                    elif gt[i + 1, j, k] == AIR:
                        Adiag[i, j, k] += scale_A
                if i - 1 >= 0:
                    if gt[i - 1, j, k] != SOLID:
                        Adiag[i, j, k] += scale_A
                if j + 1 < nl:
                    if gt[i, j + 1, k] == FLUID:
                        Adiag[i, j, k] += scale_A
                        Ay[i, j, k] = -scale_A
                    elif gt[i, j + 1, k] == AIR:
                        Adiag[i, j, k] += scale_A
                if j - 1 >= 0:
                    if gt[i, j - 1, k] != SOLID:
                        Adiag[i, j, k] += scale_A
                if k + 1 < nl:
                    if gt[i, j, k + 1] == FLUID:
                        Adiag[i, j, k] += scale_A
                        Az[i, j, k] = -scale_A
                    elif gt[i, j, k + 1] == AIR:
                        Adiag[i, j, k] += scale_A
                if k - 1 >= 0:
                    if gt[i, j, k - 1] != SOLID:
                        Adiag[i, j, k] += scale_A

    def build_multigrid_hierarchy(self, cell_type, scale_A):
        """Build Laplacian at all multigrid levels."""
        self.grid_type[0].copy_from(cell_type)
        self._build_A_level0(cell_type, self.Adiag[0], self.Ax[0],
                             self.Ay[0], self.Az[0], self.n, scale_A)

        for l in range(self.n_mg - 1):
            nl_coarse = self.n >> (l + 1)
            self._coarsen_grid_type(self.grid_type[l], self.grid_type[l + 1],
                                    nl_coarse)
            coarse_scale = scale_A * (4.0 ** (l + 1))
            self._build_A_coarse(self.grid_type[l + 1],
                                 self.Adiag[l + 1], self.Ax[l + 1],
                                 self.Ay[l + 1], self.Az[l + 1],
                                 nl_coarse, coarse_scale)

    # ------------------------------------------------------------------
    # Smoother: Red-Black Gauss-Seidel
    # ------------------------------------------------------------------
    @ti.kernel
    def _smooth(self, gt: ti.template(), Adiag: ti.template(),
                Ax: ti.template(), Ay: ti.template(), Az: ti.template(),
                r: ti.template(), z: ti.template(), nl: int, phase: int):
        for i, j, k in ti.ndrange(nl, nl, nl):
            if (i + j + k) & 1 == phase and gt[i, j, k] == FLUID:
                diag = Adiag[i, j, k]
                if diag > 0.0:
                    I = ti.Vector([i, j, k])
                    z[i, j, k] = (r[i, j, k]
                                  - self._nb_sum(Ax, Ay, Az, z, I, nl)) / diag

    def smooth(self, l, n_iters):
        nl = self.n >> l
        for _ in range(n_iters):
            self._smooth(self.grid_type[l], self.Adiag[l],
                         self.Ax[l], self.Ay[l], self.Az[l],
                         self.r[l], self.z[l], nl, 0)
            self._smooth(self.grid_type[l], self.Adiag[l],
                         self.Ax[l], self.Ay[l], self.Az[l],
                         self.r[l], self.z[l], nl, 1)

    # ------------------------------------------------------------------
    # Restriction (fine → coarse)
    # ------------------------------------------------------------------
    @ti.kernel
    def _restrict(self, gt_f: ti.template(), Adiag_f: ti.template(),
                  Ax_f: ti.template(), Ay_f: ti.template(), Az_f: ti.template(),
                  r_f: ti.template(), z_f: ti.template(),
                  r_c: ti.template(),
                  nl_f: int, nl_c: int):
        for i, j, k in ti.ndrange(nl_c, nl_c, nl_c):
            r_c[i, j, k] = 0.0

        for i, j, k in ti.ndrange(nl_f, nl_f, nl_f):
            if gt_f[i, j, k] == FLUID:
                I = ti.Vector([i, j, k])
                res = r_f[i, j, k] - (Adiag_f[i, j, k] * z_f[i, j, k]
                                       + self._nb_sum(Ax_f, Ay_f, Az_f,
                                                      z_f, I, nl_f))
                ci, cj, ck = i // 2, j // 2, k // 2
                r_c[ci, cj, ck] += res * 0.125

    # ------------------------------------------------------------------
    # Prolongation (coarse → fine)
    # ------------------------------------------------------------------
    @ti.kernel
    def _prolongate(self, gt_f: ti.template(), z_f: ti.template(),
                    z_c: ti.template(), nl_f: int):
        for i, j, k in ti.ndrange(nl_f, nl_f, nl_f):
            if gt_f[i, j, k] == FLUID:
                z_f[i, j, k] += z_c[i // 2, j // 2, k // 2]

    # ------------------------------------------------------------------
    # V-Cycle
    # ------------------------------------------------------------------
    @ti.kernel
    def _clear_field(self, f: ti.template(), nl: int):
        for i, j, k in ti.ndrange(nl, nl, nl):
            f[i, j, k] = 0.0

    def v_cycle(self):
        """One full V-cycle: precondition r → z."""
        # Downward
        for l in range(self.n_mg - 1):
            nl = self.n >> l
            nl_c = self.n >> (l + 1)
            self._clear_field(self.z[l], nl)
            self.smooth(l, self.pre_post_smooth)
            self._restrict(self.grid_type[l], self.Adiag[l],
                           self.Ax[l], self.Ay[l], self.Az[l],
                           self.r[l], self.z[l],
                           self.r[l + 1], nl, nl_c)

        # Bottom solve
        nl_bot = self.n >> (self.n_mg - 1)
        self._clear_field(self.z[self.n_mg - 1], nl_bot)
        self.smooth(self.n_mg - 1, self.bottom_smooth)

        # Upward
        for l in range(self.n_mg - 2, -1, -1):
            nl = self.n >> l
            self._prolongate(self.grid_type[l], self.z[l],
                             self.z[l + 1], nl)
            self.smooth(l, self.pre_post_smooth)

    # ------------------------------------------------------------------
    # CG operations (level 0 only — use Python literal indices)
    # ------------------------------------------------------------------
    @ti.kernel
    def _compute_Ap(self, gt0: ti.template(), Adiag0: ti.template(),
                    Ax0: ti.template(), Ay0: ti.template(), Az0: ti.template(),
                    n: int):
        for i, j, k in self.Ap:
            if gt0[i, j, k] == FLUID:
                I = ti.Vector([i, j, k])
                self.Ap[i, j, k] = (Adiag0[i, j, k] * self.p_vec[i, j, k]
                                     + self._nb_sum(Ax0, Ay0, Az0,
                                                    self.p_vec, I, n))
            else:
                self.Ap[i, j, k] = 0.0

    @ti.kernel
    def _dot(self, a: ti.template(), b: ti.template()):
        self.sum[None] = 0.0
        for I in ti.grouped(a):
            self.sum[None] += a[I] * b[I]

    @ti.kernel
    def _update_x_and_r(self, r0: ti.template(), alpha: float):
        for I in ti.grouped(self.x_vec):
            self.x_vec[I] += alpha * self.p_vec[I]
            r0[I] -= alpha * self.Ap[I]

    @ti.kernel
    def _update_p(self, z0: ti.template(), beta: float):
        for I in ti.grouped(self.p_vec):
            self.p_vec[I] = z0[I] + beta * self.p_vec[I]

    @ti.kernel
    def _init_cg(self, r0: ti.template()):
        for I in ti.grouped(self.x_vec):
            self.x_vec[I] = 0.0
            r0[I] = self.b_vec[I]

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------
    def solve(self, max_iters=500, tol=1e-6, verbose=False):
        """Solve Ax = b using MGPCG. Returns iteration count."""
        r0 = self.r[0]
        z0 = self.z[0]

        self._init_cg(r0)

        self._dot(self.b_vec, self.b_vec)
        init_rTr = self.sum[None]
        if init_rTr < 1e-20:
            return 0

        self.v_cycle()
        self.p_vec.copy_from(z0)

        self._dot(z0, r0)
        old_zTr = self.sum[None]

        for it in range(max_iters):
            self._compute_Ap(self.grid_type[0], self.Adiag[0],
                             self.Ax[0], self.Ay[0], self.Az[0], self.n)

            self._dot(self.p_vec, self.Ap)
            pTAp = self.sum[None]
            if abs(pTAp) < 1e-20:
                break

            alpha = old_zTr / pTAp
            self._update_x_and_r(r0, alpha)

            self._dot(r0, r0)
            rTr = self.sum[None]

            if verbose and (it % 50 == 0 or it < 5):
                print(f"    PCG iter {it}: |r|² = {rTr:.2e}")

            if rTr < init_rTr * tol * tol:
                if verbose:
                    print(f"    PCG converged at iter {it}: "
                          f"|r|²/|b|² = {rTr / init_rTr:.2e}")
                return it + 1

            self.v_cycle()

            self._dot(z0, r0)
            new_zTr = self.sum[None]

            beta = new_zTr / max(old_zTr, 1e-20)
            old_zTr = new_zTr

            self._update_p(z0, beta)

        if verbose:
            print(f"    PCG: max iters ({max_iters}) reached")
        return max_iters
