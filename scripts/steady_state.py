"""
Steady-state solver for the multi-stage roundabout production model
described in `model01.rtf` (closed economy, real version, log-linear
productivity).

Setting
-------
N stages, one firm per stage, in a sequential chain:
    y_n = (z_n L_n)^{alpha_n} * y_{n-1}^{1-alpha_n},   alpha_1 = 1
with stage productivity that "matures" with time:
    z_n(t_n) = t_n ** zeta_n.

Households are a representative agent with CRRA preferences. In a
deterministic steady state the Euler equation pins down  r = rho.
Production then determines the rest in closed form (Antras 2023):
    t_n^* = alpha_n * zeta_n / r,
    L_n   given by eq. (20),
    w     given by eq. (21),
    K^d   given by eq. (23).

Government runs a balanced budget G = T (lump-sum), so the goods-market
clearing reads  y_N = C + G  (intermediate inputs are embedded in y_N
through the chain and are not double-counted).

Notation matches the working notes:
    alpha[n]   labor (value-added) intensity at stage n     [alpha_1 = 1]
    zeta[n]    time intensity at stage n                    [in (0,1)]
    rho        household discount rate                      [= r in SS]
    L          aggregate labor endowment
    g_share    G / y_N (government share of final output)
    t[n]       optimal length of stage n
    L_n        labor at stage n (stationary cross-section)
    w          equilibrium wage
    y_N        final-good output
    K_d        aggregate working-capital demand
    G, C       government spending and consumption

Alternative household closure: Antras-Caballero perpetual-youth savers
with K^s = wL/(rho - r). In that case r is determined by capital-market
clearing K^d(r) = K^s(r). See `solve_steady_state_perpetual_youth`.

Phase-2(a) monetary block (CIA / liquidity constraint):
    naive money rule  dot M / M = vartheta  =>  pi = vartheta,
    Fisher relation in steady state  r = (1 - chi) rho - chi pi.
With the household side now pinning r through Fisher, the production
block reacts to monetary policy via t_n^* = alpha_n zeta_n / r.
See `solve_steady_state_monetary` and `monetary_sweep`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelParameters:
    """Parameters for the real-side multi-stage roundabout economy."""

    N: int = 10
    alpha: Optional[np.ndarray] = None
    zeta:  Optional[np.ndarray] = None
    rho:   float = 0.05
    L:     float = 1.0
    g_share: float = 0.0   # G / y_N (lump-sum financed)

    def __post_init__(self):
        if self.alpha is None:
            a = np.full(self.N, 0.5)
            a[0] = 1.0
            self.alpha = a
        if self.zeta is None:
            self.zeta = np.full(self.N, 0.3)

        self.alpha = np.asarray(self.alpha, dtype=float)
        self.zeta  = np.asarray(self.zeta,  dtype=float)

        if self.alpha.shape != (self.N,) or self.zeta.shape != (self.N,):
            raise ValueError("alpha and zeta must each have length N")
        if not np.isclose(self.alpha[0], 1.0):
            raise ValueError("alpha_1 must equal 1 (stage 1 uses only labor)")
        if not np.all((self.zeta > 0) & (self.zeta < 1)):
            raise ValueError("zeta_n must lie in (0, 1)")
        if not np.all((self.alpha[1:] > 0) & (self.alpha[1:] < 1)):
            raise ValueError("alpha_n must lie in (0, 1) for n > 1")
        if not (0.0 <= self.g_share < 1.0):
            raise ValueError("g_share must lie in [0, 1)")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if self.L <= 0.0:
            raise ValueError("L must be positive")


def beta_weights(alpha: np.ndarray) -> np.ndarray:
    """beta_n = prod_{m > n}(1 - alpha_m), with beta_N = 1."""
    N = len(alpha)
    beta = np.ones(N)
    for n in range(N - 2, -1, -1):
        beta[n] = (1.0 - alpha[n + 1]) * beta[n + 1]
    return beta


# ---------------------------------------------------------------------------
# Steady state container
# ---------------------------------------------------------------------------

@dataclass
class SteadyState:
    params: ModelParameters
    r:    float
    t:    np.ndarray   # stage durations t_n^*
    L_n:  np.ndarray   # labor at each stage
    z:    np.ndarray   # productivity z_n(t_n^*)
    y:    np.ndarray   # output at each stage
    p:    np.ndarray   # stage prices, p_N = 1 (numéraire)
    w:    float
    y_N:  float
    K_d:  float
    G:    float
    C:    float

    def report(self) -> str:
        p = self.params
        out = []
        out.append(
            f"Steady state — N={p.N}, rho={p.rho}, L={p.L}, g_share={p.g_share}"
        )
        out.append(f"  r            = {self.r:.6f}")
        out.append(f"  w            = {self.w:.6f}")
        out.append(f"  y_N          = {self.y_N:.6f}")
        out.append(f"  K_d          = {self.K_d:.6f}")
        out.append(f"  G            = {self.G:.6f}")
        out.append(f"  C            = {self.C:.6f}")
        out.append(f"  sum(L_n)     = {self.L_n.sum():.6f}    (target L = {p.L})")
        out.append(
            f"  wL + r*K_d   = {self.w * p.L + self.r * self.K_d:.6f}    "
            f"(target y_N = {self.y_N:.6f})"
        )
        out.append("")
        out.append(f"  Stage detail:")
        out.append(
            f"    n   alpha   zeta   t_n*       L_n            z_n           "
            f"y_n            p_n"
        )
        for n in range(p.N):
            pn = "n/a" if not np.isfinite(self.p[n]) else f"{self.p[n]:.6e}"
            out.append(
                f"    {n+1:<3d} {p.alpha[n]:.3f}   {p.zeta[n]:.3f}   "
                f"{self.t[n]:8.4f}   {self.L_n[n]:.6e}   "
                f"{self.z[n]:.6e}   {self.y[n]:.6e}   {pn}"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Solver — representative-agent closure
# ---------------------------------------------------------------------------

def solve_steady_state(params: ModelParameters) -> SteadyState:
    """
    Real-side steady state with a representative-agent CRRA household.

    Steady-state Euler:  r = rho.
    Production block then yields {t_n^*, L_n, w, y_N, K^d} in closed form.
    """
    alpha, zeta = params.alpha, params.zeta
    rho, L, N = params.rho, params.L, params.N
    beta = beta_weights(alpha)

    r = rho                                    # Euler equation in SS

    t = (alpha * zeta) / r                     # eq. (6) under z_n = t_n^{zeta_n}
    T = np.cumsum(t[::-1])[::-1]               # T_n = sum_{m >= n} t_m

    # Labor allocation across stages — eq. (20)
    raw   = alpha * beta * np.exp(-r * T)
    share = raw / raw.sum()
    L_n   = share * L

    z = t ** zeta                              # productivity at optimal length

    # Equilibrium wage — eq. (21)
    log_w = np.sum(alpha * beta * (np.log(alpha * beta) + np.log(z) - r * T))
    w = float(np.exp(log_w))

    # Stage outputs — eq. (1) iterated forward
    y = np.empty(N)
    y[0] = z[0] * L_n[0]                       # alpha_1 = 1
    for n in range(1, N):
        y[n] = (z[n] * L_n[n]) ** alpha[n] * y[n - 1] ** (1.0 - alpha[n])
    y_N = float(y[N - 1])

    # Stage prices: p_N = 1, recurse backward from
    #   p_{n-1} y_{n-1} e^{r t_n} = (1 - alpha_n) p_n y_n
    p = np.empty(N)
    p[N - 1] = 1.0
    for n in range(N - 1, 0, -1):
        p[n - 1] = ((1.0 - alpha[n]) * p[n] * y[n]
                    / (y[n - 1] * np.exp(r * t[n])))

    # Working-capital demand — eq. (23)
    K_d = float(np.sum(w * L_n * (np.exp(r * T) - 1.0)) / r)

    # Government / goods market
    G = params.g_share * y_N
    C = y_N - G

    return SteadyState(
        params=params, r=float(r), t=t, L_n=L_n, z=z, y=y, p=p,
        w=w, y_N=y_N, K_d=K_d, G=G, C=C,
    )


# ---------------------------------------------------------------------------
# Solver — Antràs–Caballero perpetual-youth closure (alternative)
# ---------------------------------------------------------------------------

def solve_steady_state_perpetual_youth(params: ModelParameters) -> SteadyState:
    """
    Real-side steady state with Antras-Caballero perpetual-youth savers.

    Capital supply  K^s = wL/(rho - r)  must equal capital demand K^d.
    In the log-linear case, this delivers a closed form for r:
        r = rho * S / (1 + S),   where  S = K^d / (wL).
    Note: both the labor share L_n/L and the cumulative interest e^{rT_n}
    are independent of r in the log-linear case, so S is a constant.
    """
    alpha, zeta = params.alpha, params.zeta
    rho, L, N = params.rho, params.L, params.N
    beta = beta_weights(alpha)

    # rT_n is independent of r in the log-linear case:
    rT = np.cumsum((alpha * zeta)[::-1])[::-1]
    expRT = np.exp(rT)

    # Labor share (also independent of r here):
    raw   = alpha * beta * np.exp(-rT)
    share = raw / raw.sum()

    # K^d/(wL) reduces to a constant S in the log-linear case.
    S = float(np.sum(share * (expRT - 1.0)))

    # Capital-market clearing  K^d/(wL) = 1/(rho - r)  =>  r = rho S /(1+S)
    r = rho * S / (1.0 + S)
    if not (0.0 < r < rho):
        raise RuntimeError(f"Implied r={r} outside (0, rho={rho})")

    # Now reconstruct the rest at this r (same steps as the rep-agent solver):
    t   = (alpha * zeta) / r
    T   = rT / r
    L_n = share * L
    z   = t ** zeta

    log_w = np.sum(alpha * beta * (np.log(alpha * beta) + np.log(z) - r * T))
    w = float(np.exp(log_w))

    y = np.empty(N)
    y[0] = z[0] * L_n[0]
    for n in range(1, N):
        y[n] = (z[n] * L_n[n]) ** alpha[n] * y[n - 1] ** (1.0 - alpha[n])
    y_N = float(y[N - 1])

    p = np.empty(N)
    p[N - 1] = 1.0
    for n in range(N - 1, 0, -1):
        p[n - 1] = ((1.0 - alpha[n]) * p[n] * y[n]
                    / (y[n - 1] * np.exp(r * t[n])))

    K_d = float(np.sum(w * L_n * (np.exp(r * T) - 1.0)) / r)
    K_s = w * L / (rho - r)
    assert np.isclose(K_d, K_s, rtol=1e-10), (K_d, K_s)

    G = params.g_share * y_N
    C = y_N - G

    return SteadyState(
        params=params, r=float(r), t=t, L_n=L_n, z=z, y=y, p=p,
        w=w, y_N=y_N, K_d=K_d, G=G, C=C,
    )


# ---------------------------------------------------------------------------
# Solver — perpetual-youth + monetary block (Phase 2a)
# ---------------------------------------------------------------------------

@dataclass
class MonetaryParameters:
    """Parameters for the Phase-2(a) monetary block."""
    chi:   float = 0.5    # liquidity-constraint elasticity, in (0, 1)
    theta: float = 0.0    # nominal money growth rate (= pi under naive rule)

    def __post_init__(self):
        if not (0.0 < self.chi < 1.0):
            raise ValueError("chi must lie in (0, 1)")


def solve_steady_state_monetary(
    params: ModelParameters,
    mon: MonetaryParameters,
) -> SteadyState:
    """
    Steady state with naive money rule and CIA-induced Fisher relation.

    Naive money rule:        pi = theta
    Fisher (CIA) in SS:      r  = (1 - chi) * rho - chi * pi

    The Fisher relation pins r on the household side. The real production
    block then responds via t_n^* = alpha_n * zeta_n / r in the log-linear
    case. K^s = wL/(rho - r) is reported alongside K^d for reference; the
    perpetual-youth savings stock matches K^d only at the special theta
    that solves the original Antras-Caballero closure.
    """
    alpha, zeta = params.alpha, params.zeta
    rho, L, N = params.rho, params.L, params.N
    beta = beta_weights(alpha)

    pi = mon.theta
    r  = (1.0 - mon.chi) * rho - mon.chi * pi
    if r <= 0.0:
        raise ValueError(
            f"Fisher implies r={r:.4f} <= 0; reduce theta or raise chi"
        )
    if r >= rho:
        raise ValueError(
            f"Fisher implies r={r:.4f} >= rho={rho:.4f}; capital supply ill-defined"
        )

    t = (alpha * zeta) / r                     # eq. (6) under z_n = t_n^{zeta_n}
    T = np.cumsum(t[::-1])[::-1]

    raw   = alpha * beta * np.exp(-r * T)
    share = raw / raw.sum()
    L_n   = share * L

    z = t ** zeta
    log_w = np.sum(alpha * beta * (np.log(alpha * beta) + np.log(z) - r * T))
    w = float(np.exp(log_w))

    y = np.empty(N)
    y[0] = z[0] * L_n[0]
    for n in range(1, N):
        y[n] = (z[n] * L_n[n]) ** alpha[n] * y[n - 1] ** (1.0 - alpha[n])
    y_N = float(y[N - 1])

    p = np.empty(N)
    p[N - 1] = 1.0
    for n in range(N - 1, 0, -1):
        p[n - 1] = ((1.0 - alpha[n]) * p[n] * y[n]
                    / (y[n - 1] * np.exp(r * t[n])))

    K_d = float(np.sum(w * L_n * (np.exp(r * T) - 1.0)) / r)
    G = params.g_share * y_N
    C = y_N - G

    return SteadyState(
        params=params, r=float(r), t=t, L_n=L_n, z=z, y=y, p=p,
        w=w, y_N=y_N, K_d=K_d, G=G, C=C,
    )


def monetary_sweep(
    params: ModelParameters,
    chi: float,
    theta_grid,
):
    """
    Comparative statics: sweep theta over `theta_grid` and return a list of
    (theta, pi, ss) tuples. Useful for the table of Phase 2(a) results.
    """
    out = []
    for theta in theta_grid:
        ss = solve_steady_state_monetary(
            params, MonetaryParameters(chi=chi, theta=float(theta))
        )
        out.append((float(theta), float(theta), ss))   # pi = theta (naive rule)
    return out


# ---------------------------------------------------------------------------
# Solver — Phase 2(d): CIA-Fisher household + financial wedge that closes
#                      K^d = K^s on top of the monetary block.
# ---------------------------------------------------------------------------

def solve_steady_state_friction(
    params: ModelParameters,
    mon: MonetaryParameters,
):
    """
    Phase 2(d) — close the K^d=K^s gap with a Wicksellian / external-finance
    wedge phi on top of the Phase 2(a) monetary block.

    Households earn the Fisher rate r_H = (1 - chi) rho - chi pi (with the
    naive rule pi = theta). Firms borrow at r_F = r_H + phi. The wedge phi
    adjusts so that capital-market clearing K^d(r_F) = K^s(r_F, r_H) holds,
    where  K^s = w(r_F) L / (rho - r_H).

    In the log-linear case  K^d/(wL) = C / r_F  with C = sum_n share_n * (e^{rT_n} - 1)
    independent of r, and  K^s/(wL) = 1/(rho - r_H), so clearing collapses to

        r_F = C * (rho - r_H).

    Returns
    -------
    (ss, r_H, phi) : tuple
        ss   : SteadyState evaluated at r = r_F.
        r_H  : household-side Fisher rate.
        phi  : wedge r_F - r_H. Sign convention: phi > 0 is the textbook
               external-finance premium (firms pay above household savers);
               phi < 0 indicates a savings-glut regime (equilibrium r_F is
               below the Fisher r_H, requiring a transfer the other way).
    """
    alpha, zeta = params.alpha, params.zeta
    rho, L, N = params.rho, params.L, params.N
    beta = beta_weights(alpha)

    pi = mon.theta
    r_H = (1.0 - mon.chi) * rho - mon.chi * pi
    if not (0.0 < r_H < rho):
        raise ValueError(
            f"Fisher r_H={r_H:.4f} outside (0, rho={rho:.4f}); "
            f"adjust theta or chi"
        )

    # K^d/(wL) factor C, independent of r in the log-linear case.
    rT_const = np.cumsum((alpha * zeta)[::-1])[::-1]   # r * T_n
    raw   = alpha * beta * np.exp(-rT_const)
    share = raw / raw.sum()
    C_const = float(np.sum(share * (np.exp(rT_const) - 1.0)))

    # Capital-market clearing pins r_F.
    r_F = C_const * (rho - r_H)
    if r_F <= 0.0:
        raise ValueError(f"clearing implies r_F={r_F} <= 0")

    # Production block at r_F (same template as the other solvers).
    t = (alpha * zeta) / r_F
    T = rT_const / r_F
    L_n = share * L
    z = t ** zeta

    log_w = np.sum(alpha * beta * (np.log(alpha * beta) + np.log(z) - r_F * T))
    w = float(np.exp(log_w))

    y = np.empty(N)
    y[0] = z[0] * L_n[0]
    for n in range(1, N):
        y[n] = (z[n] * L_n[n]) ** alpha[n] * y[n - 1] ** (1.0 - alpha[n])
    y_N = float(y[N - 1])

    p = np.empty(N)
    p[N - 1] = 1.0
    for n in range(N - 1, 0, -1):
        p[n - 1] = ((1.0 - alpha[n]) * p[n] * y[n]
                    / (y[n - 1] * np.exp(r_F * t[n])))

    K_d = float(np.sum(w * L_n * (np.exp(r_F * T) - 1.0)) / r_F)
    K_s = w * L / (rho - r_H)
    assert np.isclose(K_d, K_s, rtol=1e-10), (K_d, K_s)

    G = params.g_share * y_N
    C_cons = y_N - G

    ss = SteadyState(
        params=params, r=float(r_F), t=t, L_n=L_n, z=z, y=y, p=p,
        w=w, y_N=y_N, K_d=K_d, G=G, C=C_cons,
    )
    return ss, float(r_H), float(r_F - r_H)


def friction_sweep(
    params: ModelParameters,
    chi: float,
    theta_grid,
):
    """
    Phase 2(d) comparative statics: sweep theta and return
    [(theta, r_H, r_F, phi, ss), ...].
    """
    out = []
    for theta in theta_grid:
        ss, r_H, phi = solve_steady_state_friction(
            params, MonetaryParameters(chi=chi, theta=float(theta))
        )
        out.append((float(theta), r_H, ss.r, phi, ss))
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(ss: SteadyState, *, closure: str = "rep_agent",
           mon: Optional[MonetaryParameters] = None,
           tol: float = 1e-8) -> dict:
    """
    Return residuals on the equilibrium conditions. Each should be ~0.

    Conditions checked
    ------------------
    FOC_t           : alpha_n * z_n'(t_n) / z_n(t_n) - r = 0  for all n
    labor_clearing  : sum_n L_n = L
    national_id     : y_N = w*L + r*K^d
    goods_clearing  : y_N = C + G
    closure         : r = rho                            (rep-agent)
                      K^d = wL/(rho - r)                 (perpetual-youth)
                      r = (1 - chi) rho - chi theta      (monetary)
    """
    p = ss.params
    res = {}
    # Log-linear z_n => z_n'/z_n = zeta_n / t_n
    res["FOC_t"]          = float(np.max(np.abs(p.alpha * p.zeta / ss.t - ss.r)))
    res["labor_clearing"] = float(abs(ss.L_n.sum() - p.L))
    res["national_id"]    = float(abs(ss.y_N - ss.w * p.L - ss.r * ss.K_d))
    res["goods_clearing"] = float(abs(ss.y_N - ss.C - ss.G))
    if closure == "rep_agent":
        res["euler"] = float(abs(ss.r - p.rho))
    elif closure == "perpetual_youth":
        K_s = ss.w * p.L / (p.rho - ss.r)
        res["capital_clearing"] = float(abs(ss.K_d - K_s))
    elif closure == "monetary":
        if mon is None:
            raise ValueError("monetary closure requires `mon` argument")
        r_fisher = (1.0 - mon.chi) * p.rho - mon.chi * mon.theta
        res["fisher"] = float(abs(ss.r - r_fisher))
    elif closure == "friction":
        # ss.r holds the firm rate r_F; r_H comes from Fisher.
        if mon is None:
            raise ValueError("friction closure requires `mon` argument")
        r_H = (1.0 - mon.chi) * p.rho - mon.chi * mon.theta
        K_s = ss.w * p.L / (p.rho - r_H)
        res["fisher_r_H"]      = 0.0   # r_H constructed exactly from inputs
        res["capital_clearing"] = float(abs(ss.K_d - K_s))
    else:
        raise ValueError(f"unknown closure: {closure}")
    res["passed"] = all(v < tol for k, v in res.items() if k != "passed")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_check(name: str, residuals: dict) -> None:
    print(f"[{name}] equilibrium residuals")
    for k, v in residuals.items():
        if k == "passed":
            continue
        print(f"  {k:18s} = {v:.3e}")
    print(f"  passed             = {residuals['passed']}")


def main() -> None:
    # Phase-2 baseline calibration: alpha_n = 0.67 for n >= 2 (homogeneous).
    params = ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05,
        L=1.0,
        g_share=0.20,
    )

    print("=" * 78)
    print("Closure 1: representative-agent CRRA  (r = rho in steady state)")
    print("=" * 78)
    ss_ra = solve_steady_state(params)
    print(ss_ra.report())
    print()
    _print_check("rep_agent", verify(ss_ra, closure="rep_agent"))

    print()
    print("=" * 78)
    print("Closure 2: Antras-Caballero perpetual-youth savers  "
          "(K^d = K^s pins down r)")
    print("=" * 78)
    ss_py = solve_steady_state_perpetual_youth(params)
    print(ss_py.report())
    print()
    _print_check("perpetual_youth", verify(ss_py, closure="perpetual_youth"))

    # ----------------------------------------------------------------------
    # Phase 2(a) — perpetual-youth household with CIA + naive money rule.
    # Fisher: r = (1 - chi) rho - chi pi, with pi = theta (= dot M / M).
    # ----------------------------------------------------------------------
    chi = 0.5
    print()
    print("=" * 78)
    print(f"Closure 3: monetary block — naive rule, CIA  "
          f"(chi={chi}, theta=0)")
    print("=" * 78)
    mon0 = MonetaryParameters(chi=chi, theta=0.0)
    ss_m0 = solve_steady_state_monetary(params, mon0)
    print(ss_m0.report())
    print()
    _print_check("monetary", verify(ss_m0, closure="monetary", mon=mon0))

    # Comparative statics over money growth theta.
    theta_grid = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    sweep = monetary_sweep(params, chi=chi, theta_grid=theta_grid)

    print()
    print("=" * 78)
    print(f"Comparative statics: monetary closure, chi={chi}, rho={params.rho}")
    print("=" * 78)
    header = (
        f"  {'theta':>7s} {'pi':>7s} {'r':>9s} {'w':>9s} {'y_N':>9s} "
        f"{'K_d':>9s} {'K_s':>9s} {'t_1*':>9s} {'t_n*(n>=2)':>10s}"
    )
    print(header)
    for theta, pi, ss in sweep:
        K_s = ss.w * params.L / (params.rho - ss.r)
        print(
            f"  {theta:7.4f} {pi:7.4f} {ss.r:9.5f} {ss.w:9.5f} {ss.y_N:9.5f} "
            f"{ss.K_d:9.5f} {K_s:9.5f} {ss.t[0]:9.4f} {ss.t[1]:10.4f}"
        )

    # Verify the sweep cleanly satisfies the equilibrium conditions.
    print()
    print(f"  All sweep points pass equilibrium checks (tol=1e-8): "
          f"{all(verify(ss, closure='monetary', mon=MonetaryParameters(chi=chi, theta=th))['passed'] for th, _, ss in sweep)}")

    # ----------------------------------------------------------------------
    # Phase 2(d) — close the K^d=K^s gap with a financial wedge phi.
    # Households earn r_H (Fisher); firms pay r_F = r_H + phi; phi clears
    # the asset market. Fisher pi = theta still holds.
    # ----------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"Closure 4: friction — CIA-Fisher household + asset-market wedge phi  "
          f"(chi={chi}, theta=0)")
    print("=" * 78)
    mon0 = MonetaryParameters(chi=chi, theta=0.0)
    ss_f0, r_H0, phi0 = solve_steady_state_friction(params, mon0)
    print(ss_f0.report())
    print(f"  r_H (Fisher) = {r_H0:.6f}")
    print(f"  r_F (firm)   = {ss_f0.r:.6f}")
    print(f"  phi          = {phi0:+.6f}")
    print()
    _print_check("friction", verify(ss_f0, closure="friction", mon=mon0))

    fric_sweep = friction_sweep(params, chi=chi, theta_grid=theta_grid)

    print()
    print("=" * 78)
    print(f"Comparative statics: friction closure, chi={chi}, rho={params.rho}")
    print("=" * 78)
    header = (
        f"  {'theta':>7s} {'pi':>7s} {'r_H':>9s} {'r_F':>9s} {'phi':>9s} "
        f"{'w':>9s} {'y_N':>9s} {'K_d=K_s':>9s} {'t_1*':>9s} {'t_n*(n>=2)':>10s}"
    )
    print(header)
    for theta, r_H, r_F, phi, ss in fric_sweep:
        print(
            f"  {theta:7.4f} {theta:7.4f} {r_H:9.5f} {r_F:9.5f} {phi:+9.5f} "
            f"{ss.w:9.5f} {ss.y_N:9.5f} {ss.K_d:9.5f} {ss.t[0]:9.4f} "
            f"{ss.t[1]:10.4f}"
        )

    print()
    print(f"  All friction sweep points pass equilibrium checks (tol=1e-8): "
          f"{all(verify(ss, closure='friction', mon=MonetaryParameters(chi=chi, theta=th))['passed'] for th, _, _, _, ss in fric_sweep)}")

    # ----------------------------------------------------------------------
    # Side-by-side: Phase 2(a) vs Phase 2(d) at each theta.
    # ----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Phase 2(a) vs Phase 2(d) — same theta grid, contrasting closures")
    print("=" * 78)
    print(
        f"  {'theta':>7s} | {'r (2a)':>9s} {'w (2a)':>9s} {'y_N (2a)':>9s} "
        f"| {'r_F (2d)':>9s} {'w (2d)':>9s} {'y_N (2d)':>9s}"
    )
    for (theta, _, ss_m), (_, _, _, _, ss_f) in zip(sweep, fric_sweep):
        print(
            f"  {theta:7.4f} | {ss_m.r:9.5f} {ss_m.w:9.5f} {ss_m.y_N:9.5f} "
            f"| {ss_f.r:9.5f} {ss_f.w:9.5f} {ss_f.y_N:9.5f}"
        )


if __name__ == "__main__":
    main()
