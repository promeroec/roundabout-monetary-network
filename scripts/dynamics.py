"""
Phase 2(b) — Sectoral TFP shocks and the production-network linearisation
of the multi-stage roundabout model.

Setting
-------
Hicks-neutral sector-specific TFP at each stage:
    y_n = lambda_n (z_n L_n)^{alpha_n} y_{n-1}^{1-alpha_n}.
At the deterministic steady state lambda_n = 1.

In the log-linear case z_n(t_n) = t_n^{zeta_n} the FOC
    alpha_n z_n'(t_n) / z_n(t_n) = r
is invariant to lambda_n (the TFP factor cancels in z_n'/z_n). So
optimal stage durations t_n^*, the labour share L_n / L, and the
working-capital factor C in K^d/(wL) = C/r are all unchanged by TFP
shocks. Only the levels of (y_n, w, K^d) move.

Linearisation around steady state (in log-deviations) gives
    d log y_N = sum_n beta_n * d log lambda_n,
where beta_n = prod_{m>n}(1 - alpha_m). The vector
    w_n^Hulten = beta_n
collects the **Hulten elasticities** — the gross-output (revenue)
shares of stage n in y_N. They sum to >1 because of intermediate-
input double-counting (sum_n beta_n ≈ 1.49 at the benchmark). The
same elasticities apply to log w and log K^d (in this Cobb-Douglas
chain TFP shocks do not reallocate labour or shift the capital share
to first order).

Cost-share input-output matrix:
    A[n, m] = 1 - alpha_n   if m = n - 1,
              0             otherwise.
The Leontief inverse [I - A]^{-1} is lower-triangular with entries
    L[n, m] = beta_m / beta_n   for m <= n
(beta_N = 1). In particular L[N, m] = beta_m, which is exactly the
network-propagated elasticity of y_N to a Hicks-neutral TFP shock
at stage m.

Forward-looking Leontief (model01.rtf):
    L^{FL} = [I - sum_{s>=0} beta_disc^s E_t(A_{t+s})]^{-1}.
With time-invariant A this simplifies to
    L^{FL} = [I - A / (1 - beta_disc)]^{-1},
which amplifies network propagation as beta_disc -> 1.

Dynamic shock process (per stage):
    log lambda_{n,t} = rho_lambda * log lambda_{n,t-1} + eta_{n,t},
    eta_{n,t} ~ N(0, sigma^2)  i.i.d.
Linearised IRF of log y_N to a unit eta innovation at stage n:
    d log y_N at horizon h = beta_n * rho_lambda^h
(maturation lags T_{n+1} are abstracted from in this first-order pass).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from steady_state import (
    ModelParameters,
    MonetaryParameters,
    SteadyState,
    beta_weights,
    solve_steady_state_friction,
    solve_steady_state_monetary,
)


# ---------------------------------------------------------------------------
# Network objects
# ---------------------------------------------------------------------------

def io_matrix(params: ModelParameters) -> np.ndarray:
    """Cost-share IO matrix A. Bidiagonal: A[n, n-1] = 1 - alpha_n."""
    N = params.N
    A = np.zeros((N, N))
    for n in range(1, N):
        A[n, n - 1] = 1.0 - params.alpha[n]
    return A


def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """Standard Leontief inverse [I - A]^{-1}."""
    return np.linalg.inv(np.eye(A.shape[0]) - A)


def forward_looking_leontief(A: np.ndarray, beta_disc: float) -> np.ndarray:
    """
    Forward-looking Leontief inverse (model01.rtf):
        L^{FL} = [I - sum_{s>=0} beta_disc^s A]^{-1} = [I - A/(1-beta_disc)]^{-1}
    when A is time-invariant. beta_disc in [0, 1).
    """
    if not (0.0 <= beta_disc < 1.0):
        raise ValueError("beta_disc must lie in [0, 1)")
    return np.linalg.inv(np.eye(A.shape[0]) - A / (1.0 - beta_disc))


def domar_weights(params: ModelParameters) -> np.ndarray:
    """Hulten elasticity vector w_n = beta_n = d log y_N / d log lambda_n
    under the Hicks-neutral TFP specification y_n = lambda_n (z_n L_n)^a y_{n-1}^{1-a}.
    These are the gross-output (revenue) shares of stage n in y_N and sum to >1."""
    return beta_weights(params.alpha)


# ---------------------------------------------------------------------------
# TFP shock process and impulse responses
# ---------------------------------------------------------------------------

@dataclass
class TfpParameters:
    """AR(1) sector-TFP shock process (one per stage)."""
    rho_lambda: float = 0.9    # persistence
    sigma:      float = 0.01   # std dev of innovations

    def __post_init__(self):
        if not (0.0 <= self.rho_lambda < 1.0):
            raise ValueError("rho_lambda must lie in [0, 1)")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")


def tfp_irf(
    params: ModelParameters,
    tfp: TfpParameters,
    horizons: np.ndarray,
) -> np.ndarray:
    """
    Linearised IRF of log y_N to a unit eta innovation at each stage.

    Returns
    -------
    irf : (H, N) array
        irf[h, n] = response of log y_N at horizon h to a unit eta
        innovation at stage n. Convention: maturation lag T_{n+1}
        ignored (instantaneous network propagation, only AR(1) decay).
    """
    horizons = np.asarray(horizons)
    w = domar_weights(params)
    rho_h = tfp.rho_lambda ** horizons.astype(float)
    return np.outer(rho_h, w)


def tfp_simulate(
    params: ModelParameters,
    tfp: TfpParameters,
    T: int,
    seed: Optional[int] = None,
):
    """
    Simulate T periods of sectoral TFP and the implied log y_N path.

    Returns
    -------
    log_lambda : (T, N) array
        Log-TFP per stage and period.
    log_yN : (T,) array
        Log-deviation of final-good output, log_lambda @ Domar weights.
    """
    rng = np.random.default_rng(seed)
    N = params.N
    log_lambda = np.zeros((T, N))
    log_lambda[0] = rng.normal(0.0, tfp.sigma, N)
    for t in range(1, T):
        log_lambda[t] = (
            tfp.rho_lambda * log_lambda[t - 1]
            + rng.normal(0.0, tfp.sigma, N)
        )
    log_yN = log_lambda @ domar_weights(params)
    return log_lambda, log_yN


def downstream_lag(t_star: np.ndarray) -> np.ndarray:
    """
    Maturation lag T_lag[n] = sum_{m > n} t_star[m]:  number of calendar
    periods between a stage-n productivity innovation and its first
    impact on log y_N (the "downstream-only" convention used in
    STEADY_STATE_RESULTS.md).  T_lag[N-1] = 0 (final stage has no
    downstream wait).
    """
    N = len(t_star)
    T_lag = np.zeros(N)
    for n in range(N - 2, -1, -1):
        T_lag[n] = T_lag[n + 1] + t_star[n + 1]
    return T_lag


def tfp_irf_lagged(
    params: ModelParameters,
    tfp: TfpParameters,
    horizons: np.ndarray,
    t_star: np.ndarray,
) -> np.ndarray:
    """
    Lagged stage-level IRF of log y_N to a unit eta_n innovation at t=0:

        IRF[h, n] = 0                                  if h < T_lag[n]
                  = beta_n * rho_lambda^{h - T_lag[n]} otherwise.

    The horizon argument is treated as an integer calendar period; the
    fractional t_star[n] = alpha_n*zeta_n / r is rounded to nearest
    integer when building the lag.  Otherwise identical to tfp_irf().
    """
    horizons = np.asarray(horizons).astype(float)
    weights = domar_weights(params)
    lag = np.rint(downstream_lag(t_star)).astype(int)
    H = len(horizons)
    N = params.N
    irf = np.zeros((H, N))
    for n in range(N):
        active = horizons >= lag[n]
        irf[active, n] = weights[n] * tfp.rho_lambda ** (horizons[active] - lag[n])
    return irf


def tfp_simulate_lagged(
    params: ModelParameters,
    tfp: TfpParameters,
    t_star: np.ndarray,
    T: int,
    seed: Optional[int] = None,
):
    """
    Simulate T periods of sectoral TFP with maturation lags.

        log y_N(t) = sum_n beta_n * log lambda_n(t - T_lag[n])

    where log lambda_n is an AR(1) process with persistence rho_lambda
    and innovation std sigma.  Simulation length T must exceed
    max(T_lag) so the start-up transient does not dominate.  Returns
    only the post-warmup window of length T.
    """
    rng = np.random.default_rng(seed)
    N = params.N
    weights = domar_weights(params)
    lag = np.rint(downstream_lag(t_star)).astype(int)
    L = int(lag.max())
    T_total = T + L
    log_lambda = np.zeros((T_total, N))
    log_lambda[0] = rng.normal(0.0, tfp.sigma, N)
    for t in range(1, T_total):
        log_lambda[t] = (
            tfp.rho_lambda * log_lambda[t - 1]
            + rng.normal(0.0, tfp.sigma, N)
        )
    log_yN = np.zeros(T)
    for t in range(T):
        for n in range(N):
            log_yN[t] += weights[n] * log_lambda[t + L - lag[n], n]
    return log_lambda[L:], log_yN


def cumulative_variance_share(
    params: ModelParameters,
    tfp: TfpParameters,
    t_star: np.ndarray,
    horizons: np.ndarray,
) -> dict:
    """
    Cumulative share of unconditional Var(log y_N) that has *propagated*
    by horizon h, broken down by stage of origin.

    Var(log y_N | h) = sum_n beta_n^2 *
        sigma^2 * (1 - rho^{2(h - T_lag[n] + 1)}_+) / (1 - rho^2),
    with the truncation max(0, h - T_lag[n] + 1) so stages whose lag
    exceeds h contribute zero.

    Returns a dict with stage-level contributions and the share of the
    full unconditional variance accounted for by horizon h.
    """
    horizons = np.asarray(horizons)
    weights = domar_weights(params)
    lag = np.rint(downstream_lag(t_star)).astype(int)
    rho2 = tfp.rho_lambda ** 2
    var_eta = tfp.sigma ** 2
    full = (weights ** 2).sum() * var_eta / (1 - rho2)

    H, N = len(horizons), params.N
    contrib = np.zeros((H, N))
    for hi, h in enumerate(horizons):
        for n in range(N):
            d = h - lag[n] + 1
            if d <= 0:
                contrib[hi, n] = 0.0
            else:
                contrib[hi, n] = (weights[n] ** 2) * var_eta * (1 - rho2 ** d) / (1 - rho2)
    return {
        "horizons": horizons,
        "stage_contrib": contrib,
        "stage_share":   contrib / full,
        "total_share":   contrib.sum(axis=1) / full,
        "full_var":      full,
    }


def variance_decomposition(params: ModelParameters, tfp: TfpParameters) -> dict:
    """
    Unconditional variance contribution of each stage's TFP innovations to
    log y_N. With i.i.d. innovations across stages and AR(1) persistence,
        Var(log y_N) = (sum_n beta_n^2) * sigma^2 / (1 - rho^2).
    Returns the share contributed by each stage and the absolute variance.
    """
    w = domar_weights(params)
    var_eta = tfp.sigma ** 2 / (1.0 - tfp.rho_lambda ** 2)
    contrib = (w ** 2) * var_eta
    total = float(contrib.sum())
    return {
        "stage_contrib": contrib,
        "stage_share":   contrib / total,
        "total_var":     total,
        "total_std":     float(np.sqrt(total)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _baseline_params() -> ModelParameters:
    return ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05,
        L=1.0,
        g_share=0.20,
    )


def main() -> None:
    params = _baseline_params()
    A = io_matrix(params)
    L = leontief_inverse(A)
    w = domar_weights(params)

    print("=" * 78)
    print("Phase 2(b) — production network and sectoral TFP shocks")
    print("=" * 78)
    print()
    print("Cost-share IO matrix A (bidiagonal, A[n, n-1] = 1 - alpha_n):")
    print(np.array_str(A, precision=4, suppress_small=True))
    print()
    print("Leontief inverse L = [I - A]^{-1}:")
    print(np.array_str(L, precision=4, suppress_small=True))
    print()
    print("Hulten elasticities beta_n  (= d log y_N / d log lambda_n):")
    print(f"  {'stage':>5s}  {'alpha_n':>8s}  {'beta_n':>10s}  "
          f"{'w_n=beta_n':>15s}")
    beta = beta_weights(params.alpha)
    for n in range(params.N):
        print(f"  {n+1:5d}  {params.alpha[n]:8.4f}  {beta[n]:10.4e}  "
              f"{w[n]:15.6e}")
    print(f"  {'sum':>5s}  {'':>8s}  {'':>10s}  {w.sum():15.6e}  "
          f"(>1 due to intermediate double-counting)")

    # ---------------------------------------------------------------------
    # Forward-looking Leontief at several discount factors.
    # ---------------------------------------------------------------------
    print()
    print("Forward-looking Leontief L^FL = [I - A/(1-beta_disc)]^{-1}.")
    print("Reporting last row L^FL[N, :] (final-stage exposure to upstream shocks):")
    print(f"  {'beta_disc':>10s}  {'L^FL[N, 1]':>14s}  {'L^FL[N, 5]':>14s}  "
          f"{'L^FL[N, 9]':>14s}  {'L^FL[N, 10]':>14s}")
    for bd in [0.0, 0.25, 0.5, 0.75, 0.90]:
        Lfl = forward_looking_leontief(A, bd)
        print(f"  {bd:10.2f}  {Lfl[-1, 0]:14.4e}  {Lfl[-1, 4]:14.4e}  "
              f"{Lfl[-1, 8]:14.4e}  {Lfl[-1, 9]:14.4e}")

    # ---------------------------------------------------------------------
    # IRFs for an AR(1) TFP innovation at each stage.
    # ---------------------------------------------------------------------
    tfp = TfpParameters(rho_lambda=0.9, sigma=0.01)
    horizons = np.arange(0, 13)
    irf = tfp_irf(params, tfp, horizons)
    print()
    print(f"IRF of log y_N to a unit eta innovation, rho_lambda={tfp.rho_lambda}")
    print("  (% response = irf * 100; maturation lag abstracted away)")
    print(f"  {'h':>3s}  {'shock at 1':>12s}  {'shock at 5':>12s}  "
          f"{'shock at 9':>12s}  {'shock at 10':>12s}")
    for h_idx in range(len(horizons)):
        print(f"  {horizons[h_idx]:3d}  "
              f"{irf[h_idx, 0]:12.4e}  {irf[h_idx, 4]:12.4e}  "
              f"{irf[h_idx, 8]:12.4e}  {irf[h_idx, 9]:12.4e}")

    # ---------------------------------------------------------------------
    # Variance decomposition (i.i.d. innovations across stages).
    # ---------------------------------------------------------------------
    vd = variance_decomposition(params, tfp)
    print()
    print(f"Variance decomposition of log y_N "
          f"(sigma={tfp.sigma}, rho_lambda={tfp.rho_lambda}):")
    print(f"  total std(log y_N) = {vd['total_std']:.4e}")
    print(f"  {'stage':>5s}  {'contribution':>14s}  {'share':>8s}")
    for n in range(params.N):
        print(f"  {n+1:5d}  {vd['stage_contrib'][n]:14.4e}  "
              f"{100*vd['stage_share'][n]:7.3f}%")

    # ---------------------------------------------------------------------
    # Phase 2(b'): maturation lag in the IRFs.
    # ---------------------------------------------------------------------
    chi = 0.5
    # Reuse the Phase 2(a) θ=0 steady state for the t_n* used in the lag.
    ss_2a = solve_steady_state_monetary(params, MonetaryParameters(chi=chi, theta=0.0))
    lag = downstream_lag(ss_2a.t)
    print()
    print("Phase 2(b'): downstream maturation lag T_lag[n] = sum_{m>n} t_m*")
    print(f"  (Phase 2(a) θ=0 steady state, t_1*={ss_2a.t[0]:.2f}, "
          f"t_n*(n>=2)={ss_2a.t[1]:.2f})")
    print(f"  {'stage n':>8s}  {'T_lag[n]':>10s}")
    for n in range(params.N):
        print(f"  {n+1:8d}  {lag[n]:10.2f}")

    horizons_lag = np.array([0, 8, 16, 24, 48, 72, 96])
    irf_lag = tfp_irf_lagged(params, tfp, horizons_lag, ss_2a.t)
    print()
    print(f"Lagged IRF of log y_N at horizon h to a unit eta innovation, "
          f"rho_lambda={tfp.rho_lambda}")
    print(f"  {'h':>3s}  {'shock at 1':>12s}  {'shock at 5':>12s}  "
          f"{'shock at 9':>12s}  {'shock at 10':>12s}")
    for hi, h in enumerate(horizons_lag):
        print(f"  {h:3d}  "
              f"{irf_lag[hi, 0]:12.4e}  {irf_lag[hi, 4]:12.4e}  "
              f"{irf_lag[hi, 8]:12.4e}  {irf_lag[hi, 9]:12.4e}")

    # Cumulative variance share by horizon (started from rest at t=0).
    cv = cumulative_variance_share(params, tfp, ss_2a.t, horizons_lag)
    print()
    print(f"Fraction of steady-state Var(log y_N) realized by horizon h "
          f"(system started from rest at t=0, σ={tfp.sigma}):")
    print(f"  {'h':>3s}  {'total share':>11s}  {'stage 7':>9s}  "
          f"{'stage 8':>9s}  {'stage 9':>9s}  {'stage 10':>9s}")
    for hi, h in enumerate(horizons_lag):
        s = cv["stage_share"][hi]
        print(f"  {h:3d}  {cv['total_share'][hi]:11.4f}  "
              f"{s[6]:9.4f}  {s[7]:9.4f}  {s[8]:9.4f}  {s[9]:9.4f}")

    # ---------------------------------------------------------------------
    # Phase 2(b'): lag × θ interaction.  Sweep θ in {0, 0.02, 0.04} and
    # report T_lag[1] (longest), T_lag[8] (mid-stage), T_lag[9] (one-stage),
    # plus the cumulative variance share at h = 4, 8, 24 horizons.
    # ---------------------------------------------------------------------
    horizons_x = np.array([4, 8, 16, 24, 48, 96])
    print()
    print("Lag × θ interaction:  T_lag[n] = α_n ζ_n / r summed downstream "
          "scales as 1/r.")
    print(f"  {'θ':>6s}  {'r':>7s}  {'t_n*(n>=2)':>10s}  {'T_lag[1]':>9s}  "
          f"{'T_lag[5]':>9s}  {'T_lag[9]':>9s}  " +
          "  ".join(f"share@h={h}" for h in horizons_x))
    for theta in [0.00, 0.01, 0.02, 0.03, 0.04]:
        ss = solve_steady_state_monetary(params, MonetaryParameters(chi=chi, theta=theta))
        lag_th = downstream_lag(ss.t)
        cv_th = cumulative_variance_share(params, tfp, ss.t, horizons_x)
        line = (f"  {theta:6.3f}  {ss.r:7.4f}  {ss.t[1]:10.3f}  "
                f"{lag_th[0]:9.2f}  {lag_th[4]:9.2f}  {lag_th[8]:9.2f}  ")
        line += "  ".join(f"{cv_th['total_share'][hi]:10.4f}"
                          for hi in range(len(horizons_x)))
        print(line)

    # ---------------------------------------------------------------------
    # Stochastic simulation under a Phase 2(a) baseline steady state.
    # ---------------------------------------------------------------------
    log_lam, log_yN = tfp_simulate(params, tfp, T=400, seed=42)
    print()
    print(f"Simulated path (T=400, seed=42):")
    print(f"  empirical std(log y_N)        = {log_yN.std():.4e}")
    print(f"  empirical std(log lambda_10)  = {log_lam[:, -1].std():.4e}")
    print(f"  empirical std(log lambda_1)   = {log_lam[:, 0].std():.4e}")

    # Anchor levels using the Phase 2(a) monetary closure at theta=0.
    chi = 0.5
    ss = solve_steady_state_monetary(params, MonetaryParameters(chi=chi, theta=0.0))
    print()
    print(f"Anchored to Phase 2(a) baseline (chi={chi}, theta=0): "
          f"y_N(SS) = {ss.y_N:.4f}")
    print(f"  ⇒ simulated y_N range: "
          f"[{ss.y_N * np.exp(log_yN.min()):.4f}, "
          f"{ss.y_N * np.exp(log_yN.max()):.4f}]  "
          f"(SS = {ss.y_N:.4f})")


if __name__ == "__main__":
    main()
