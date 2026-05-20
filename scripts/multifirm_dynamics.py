"""
Phase (c) — Firm-level TFP shocks and aggregate volatility on the
multi-firm chain network.

Setting
-------
Firm-level Hicks-neutral TFP shifters lambda_{i,n} enter the production
function as
    y_{i,n} = lambda_{i,n} (z_{n}(t_n) L_{i,n})^{alpha_n}
              * prod_j x^{(n-1,n)}_{ij}^{(1-alpha_n) a^{(n-1,n)}_{ij}}.
In the log-linear case the time-FOC is invariant to lambda, so optimal
durations t_n^* and the cost-share matrix A are unchanged.  A first-order
expansion gives Hulten's identity for log final output:

    d log y_N = sum_{i,n} v_{i,n} d log lambda_{i,n}
              = sum_{i,n} d_{i,n} d log lambda_{i,n},

where d_{i,n} = v_{i,n} is the firm's gross-output share in y_N
(Hulten's theorem under Hicks-neutral TFP).  The chain (M=1) recovers
d_n = beta_n.

Process
-------
Each firm has an AR(1) log-TFP:
    log lambda_{i,n,t} = rho_lambda * log lambda_{i,n,t-1} + eta_{i,n,t},
    eta_{i,n,t} ~ N(0, sigma^2)  iid across firms and stages.

IRF at horizon h to a unit eta innovation at firm (i,n):
    IRF_{i,n}(h) = d_{i,n} * rho_lambda^h.
(Maturation lags are abstracted from at this first-order pass, as in
Phase 2(b) — the same caveat applies here.)

Aggregate variance under iid firm-level innovations:
    Var(log y_N) = (sigma^2 / (1 - rho_lambda^2)) * sum_{i,n} d_{i,n}^2.

Granularity
-----------
Define the network "concentration index"
    G(net) = sum_{i,n} d_{i,n}^2 = sum_n sum_i v_{i,n}^2.
On the chain (uniform sourcing) sum_i v_{i,n}^2 = beta_n^2 / M_n exactly,
so G(chain, M) = sum_n beta_n^2 / M_n  =>  G ~ 1/M.  This is the
Gabaix-style 1/sqrt(M) volatility decay.

Under common Pareto-out-degree at every stage with shape gamma:
sum_i v_{i,n}^2 = beta_n^2 * ||w^{(n)}||^2 where w^{(n)} is the
Pareto-distributed supplier-popularity vector at stage n.  As M grows,
||w^{(n)}||^2 stays bounded away from zero whenever gamma <= 2 (infinite
variance), so aggregate volatility does not vanish — the
Acemoglu-Carvalho-Ozdaglar-Tahbaz-Salehi (2012) granularity result
adapted to a multi-stage chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from steady_state import ModelParameters, MonetaryParameters, beta_weights
from dynamics import downstream_lag
from network import (
    FirmNetwork,
    chain_network,
    firm_domar_weights,
    influence_vector,
    multistage_random,
    pareto_outdegree,
    stage_aggregate_domar,
    stage_hhi,
    stage_influence_total,
)


# ---------------------------------------------------------------------------
# Firm-level TFP process
# ---------------------------------------------------------------------------

@dataclass
class FirmTfpParameters:
    """AR(1) firm-level TFP shock process."""
    rho_lambda: float = 0.9
    sigma:      float = 0.01

    def __post_init__(self):
        if not (0.0 <= self.rho_lambda < 1.0):
            raise ValueError("rho_lambda must lie in [0, 1)")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")


def firm_irf(
    net: FirmNetwork,
    tfp: FirmTfpParameters,
    horizons: np.ndarray,
    d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    IRF of log y_N to a unit eta innovation at each firm.

    Returns
    -------
    irf : (H, F) array
        irf[h, f] = response at horizon h to a unit innovation at firm f
        (stacked stage-by-stage).
    """
    if d is None:
        d = firm_domar_weights(net)
    horizons = np.asarray(horizons).astype(float)
    rho_h = tfp.rho_lambda ** horizons
    return np.outer(rho_h, d)


def variance_decomposition(
    net: FirmNetwork,
    tfp: FirmTfpParameters,
    d: Optional[np.ndarray] = None,
) -> dict:
    """
    Unconditional variance contribution of each firm's iid TFP shock to
    log y_N.  Reports the firm-level contributions and stage rollups.
    """
    if d is None:
        d = firm_domar_weights(net)
    var_eta = tfp.sigma ** 2 / (1.0 - tfp.rho_lambda ** 2)
    contrib = (d ** 2) * var_eta
    total = float(contrib.sum())
    stage_contrib = np.array(
        [contrib[net.stage_slice(n)].sum() for n in range(net.N)]
    )
    return {
        "firm_contrib":  contrib,
        "firm_share":    contrib / total,
        "stage_contrib": stage_contrib,
        "stage_share":   stage_contrib / total,
        "total_var":     total,
        "total_std":     float(np.sqrt(total)),
        "G_index":       float((d ** 2).sum()),  # sum of squared Domar weights
    }


def firm_irf_lagged(
    net: FirmNetwork,
    tfp: FirmTfpParameters,
    horizons: np.ndarray,
    t_star: np.ndarray,
    d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Lagged firm-level IRF of log y_N to a unit eta_{i,n} innovation:

        IRF[h, (i,n)] = 0                            if h < T_lag[n]
                      = d_{i,n} * rho^{h - T_lag[n]} otherwise.

    Within-stage homogeneity means T_lag is stage-level (same lag for
    every firm at stage n).
    """
    if d is None:
        d = firm_domar_weights(net)
    horizons = np.asarray(horizons).astype(float)
    lag_stage = np.rint(downstream_lag(t_star)).astype(int)
    H, F = len(horizons), net.total_firms
    irf = np.zeros((H, F))
    for n in range(net.N):
        sl = net.stage_slice(n)
        active = horizons >= lag_stage[n]
        if not active.any():
            continue
        decay = tfp.rho_lambda ** (horizons[active] - lag_stage[n])
        irf[np.ix_(active, np.arange(sl.start, sl.stop))] = (
            np.outer(decay, d[sl])
        )
    return irf


def cumulative_variance_share_firm(
    net: FirmNetwork,
    tfp: FirmTfpParameters,
    t_star: np.ndarray,
    horizons: np.ndarray,
    d: Optional[np.ndarray] = None,
) -> dict:
    """
    Fraction of steady-state Var(log y_N) realized by horizon h, broken
    down by stage of origin (firm-level).  Same identity as the
    single-firm case but Domar weights are summed within stage:

        Var_{stage n}(h) = sum_{i in n} d_{i,n}^2 *
                           sigma^2 * (1 - rho^{2(h - T_lag[n] + 1)}_+) /
                           (1 - rho^2).
    """
    if d is None:
        d = firm_domar_weights(net)
    horizons = np.asarray(horizons)
    lag = np.rint(downstream_lag(t_star)).astype(int)
    rho2 = tfp.rho_lambda ** 2
    var_eta = tfp.sigma ** 2
    sumsq_stage = np.array(
        [(d[net.stage_slice(n)] ** 2).sum() for n in range(net.N)]
    )
    full = float(sumsq_stage.sum() * var_eta / (1 - rho2))

    H, N = len(horizons), net.N
    contrib = np.zeros((H, N))
    for hi, h in enumerate(horizons):
        for n in range(N):
            d_step = h - lag[n] + 1
            if d_step <= 0:
                continue
            contrib[hi, n] = (
                sumsq_stage[n] * var_eta * (1 - rho2 ** d_step) / (1 - rho2)
            )
    return {
        "horizons":      horizons,
        "stage_contrib": contrib,
        "stage_share":   contrib / full,
        "total_share":   contrib.sum(axis=1) / full,
        "full_var":      full,
    }


def firm_simulate_lagged(
    net: FirmNetwork,
    tfp: FirmTfpParameters,
    t_star: np.ndarray,
    T: int,
    seed: Optional[int] = None,
    d: Optional[np.ndarray] = None,
):
    """
    Simulate T post-warmup periods of firm-level TFP with maturation lags:

        log y_N(t) = sum_{i,n} d_{i,n} * log lambda_{i,n}(t - T_lag[n]).

    Returns (log_lambda[T,F], log_yN[T]).
    """
    if d is None:
        d = firm_domar_weights(net)
    rng = np.random.default_rng(seed)
    F, N = net.total_firms, net.N
    lag = np.rint(downstream_lag(t_star)).astype(int)
    L = int(lag.max())
    T_total = T + L
    log_lambda = np.zeros((T_total, F))
    log_lambda[0] = rng.normal(0.0, tfp.sigma, F)
    for t in range(1, T_total):
        log_lambda[t] = (
            tfp.rho_lambda * log_lambda[t - 1]
            + rng.normal(0.0, tfp.sigma, F)
        )
    log_yN = np.zeros(T)
    for t in range(T):
        for n in range(N):
            sl = net.stage_slice(n)
            log_yN[t] += d[sl] @ log_lambda[t + L - lag[n], sl]
    return log_lambda[L:], log_yN


def simulate(
    net: FirmNetwork,
    tfp: FirmTfpParameters,
    T: int,
    seed: Optional[int] = None,
    d: Optional[np.ndarray] = None,
):
    """
    Simulate T periods of firm-level TFP and the implied log y_N path.

    Returns
    -------
    log_lambda : (T, F) array  — firm-level log-TFP per period
    log_yN     : (T,) array    — aggregate log-deviation of final output
    """
    if d is None:
        d = firm_domar_weights(net)
    rng = np.random.default_rng(seed)
    F = net.total_firms
    log_lambda = np.zeros((T, F))
    log_lambda[0] = rng.normal(0.0, tfp.sigma, F)
    for t in range(1, T):
        log_lambda[t] = (
            tfp.rho_lambda * log_lambda[t - 1]
            + rng.normal(0.0, tfp.sigma, F)
        )
    log_yN = log_lambda @ d
    return log_lambda, log_yN


# ---------------------------------------------------------------------------
# Granularity scaling experiment
# ---------------------------------------------------------------------------

def granularity_curve(
    alpha: np.ndarray,
    M_grid: np.ndarray,
    network_kind: str,
    *,
    k: Optional[int] = None,
    pareto_shape: Optional[float] = None,
    pareto_common: bool = True,
    seeds: Optional[np.ndarray] = None,
) -> dict:
    """
    For each M in M_grid, build a network of the requested kind and
    return the concentration index G = sum_{i,n} d_{i,n}^2.

    For the random and Pareto kinds, average over `seeds` realizations
    (default: seeds=range(50) for sampling stability).
    """
    if seeds is None:
        seeds = np.arange(50)
    G_mean = np.empty(len(M_grid))
    G_std  = np.empty(len(M_grid))
    for idx, M in enumerate(M_grid):
        M_vec = np.full(len(alpha), int(M))
        if network_kind == "chain":
            net = chain_network(M_vec, alpha)
            v = influence_vector(net)
            d = firm_domar_weights(net, v)
            G_mean[idx] = float((d ** 2).sum())
            G_std[idx]  = 0.0
        else:
            G_runs = []
            for seed in seeds:
                if network_kind == "random":
                    if k is None:
                        raise ValueError("kind='random' requires k=")
                    net = multistage_random(M_vec, alpha, k=int(k),
                                            seed=int(seed))
                elif network_kind == "pareto":
                    if pareto_shape is None:
                        raise ValueError("kind='pareto' requires pareto_shape=")
                    net = pareto_outdegree(M_vec, alpha, shape=pareto_shape,
                                           common=pareto_common, seed=int(seed))
                else:
                    raise ValueError(f"unknown network_kind: {network_kind}")
                v = influence_vector(net)
                d = firm_domar_weights(net, v)
                G_runs.append(float((d ** 2).sum()))
            G_runs = np.array(G_runs)
            G_mean[idx] = float(G_runs.mean())
            G_std[idx]  = float(G_runs.std())
    return {"M": np.asarray(M_grid), "G_mean": G_mean, "G_std": G_std}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _baseline_alpha(N: int = 10) -> np.ndarray:
    a = np.full(N, 0.67)
    a[0] = 1.0
    return a


def main() -> None:
    from steady_state import solve_steady_state_monetary, MonetaryParameters
    N, M = 10, 20
    alpha = _baseline_alpha(N)
    M_vec = np.full(N, M)
    tfp = FirmTfpParameters(rho_lambda=0.9, sigma=0.01)

    chain = chain_network(M_vec, alpha)
    rand4 = multistage_random(M_vec, alpha, k=4, seed=42)
    par15 = pareto_outdegree(M_vec, alpha, shape=1.5, common=True, seed=42)

    nets = [
        ("chain",                              chain),
        ("multistage_random(k=4, seed=42)",    rand4),
        ("pareto_outdegree(shape=1.5, seed=42)", par15),
    ]

    print("=" * 80)
    print(f"Phase (c) — firm-level TFP shocks, N={N}, M={M}, "
          f"rho_lambda={tfp.rho_lambda}, sigma={tfp.sigma}")
    print("=" * 80)

    # ---------------------------------------------------------------------
    # Domar concentration & variance decomposition for each network.
    # ---------------------------------------------------------------------
    print()
    print(f"  {'network':40s}  {'G=sum d^2':>11s}  "
          f"{'std(log y_N)':>12s}  {'top1 d_{i,n}':>12s}  "
          f"{'top10 share':>11s}")
    for label, net in nets:
        d = firm_domar_weights(net)
        vd = variance_decomposition(net, tfp, d)
        order = np.argsort(d)[::-1]
        top1 = d[order[0]]
        top10_share = float(d[order[:10]].sum())
        print(f"  {label:40s}  {vd['G_index']:11.4e}  "
              f"{vd['total_std']:12.4e}  {top1:12.4e}  "
              f"{top10_share:11.4f}")

    # Reference: M=1 chain ("aggregate-only" Phase 2(b) baseline).
    chain_M1 = chain_network(np.ones(N, dtype=int), alpha)
    d_M1 = firm_domar_weights(chain_M1)
    vd_M1 = variance_decomposition(chain_M1, tfp, d_M1)
    print(f"  {'chain M=1 (Phase 2b reference)':40s}  "
          f"{vd_M1['G_index']:11.4e}  {vd_M1['total_std']:12.4e}  "
          f"{d_M1.max():12.4e}  {1.0:11.4f}")

    # ---------------------------------------------------------------------
    # Stage rollup of variance contribution under each network.
    # ---------------------------------------------------------------------
    print()
    print("  Variance share of log y_N by stage:")
    print(f"    {'n':>3s}  {'chain':>10s}  {'random k=4':>10s}  "
          f"{'pareto 1.5':>10s}")
    vds = [variance_decomposition(net, tfp) for _, net in nets]
    for n in range(N):
        print(
            f"    {n+1:3d}  "
            f"{100*vds[0]['stage_share'][n]:9.4f}%  "
            f"{100*vds[1]['stage_share'][n]:9.4f}%  "
            f"{100*vds[2]['stage_share'][n]:9.4f}%"
        )

    # ---------------------------------------------------------------------
    # IRFs to a unit innovation at the largest-Domar firm in each network.
    # ---------------------------------------------------------------------
    print()
    print(f"  IRF of log y_N to unit eta at the *most-influential* firm:")
    horizons = np.array([0, 1, 3, 6, 12, 24])
    print(f"    {'horizon':>7s}  " + "  ".join(
        f"{label.split('(')[0]:>14s}" for label, _ in nets
    ))
    for h in horizons:
        row = [f"    {h:7d}"]
        for label, net in nets:
            d = firm_domar_weights(net)
            top = d.max()
            row.append(f"{top * tfp.rho_lambda ** h:14.4e}")
        print("  ".join(row))

    # ---------------------------------------------------------------------
    # Granularity scaling: sum d^2 vs M.
    # ---------------------------------------------------------------------
    print()
    print("  Granularity scaling (G = sum d^2 vs M, average over 50 seeds):")
    M_grid = np.array([5, 10, 20, 50, 100, 200])
    chain_curve  = granularity_curve(alpha, M_grid, "chain")
    rand_curve   = granularity_curve(alpha, M_grid, "random", k=4,
                                     seeds=np.arange(50))
    par_curve    = granularity_curve(alpha, M_grid, "pareto",
                                     pareto_shape=1.5, pareto_common=True,
                                     seeds=np.arange(50))

    print(f"    {'M':>5s}  {'G(chain)':>11s}  "
          f"{'G(random k=4) mean ± std':>30s}  "
          f"{'G(pareto 1.5) mean ± std':>30s}")
    for idx, M in enumerate(M_grid):
        print(
            f"    {M:5d}  {chain_curve['G_mean'][idx]:11.4e}  "
            f"  {rand_curve['G_mean'][idx]:11.4e} ± "
            f"{rand_curve['G_std'][idx]:9.2e}     "
            f"{par_curve['G_mean'][idx]:11.4e} ± "
            f"{par_curve['G_std'][idx]:9.2e}"
        )

    # Effective decay rate G(M) ~ M^{-p}: estimate p by log-log slope.
    def slope(curve):
        return float(np.polyfit(np.log(M_grid), np.log(curve['G_mean']), 1)[0])

    print()
    print(f"    log-log slope (G ~ M^p) — chain:  {slope(chain_curve):.3f}  "
          f"(theory: -1.000)")
    print(f"    log-log slope (G ~ M^p) — random: {slope(rand_curve):.3f}")
    print(f"    log-log slope (G ~ M^p) — pareto: {slope(par_curve):.3f}")

    # ---------------------------------------------------------------------
    # Pareto-shape sweep at fixed M=20: how granularity depends on gamma.
    # ---------------------------------------------------------------------
    print()
    print("  Pareto-shape sweep at M=20, mean over 200 seeds:")
    print(f"    {'gamma':>6s}  {'G mean':>11s}  {'G std':>11s}  "
          f"{'G/G(chain)':>11s}  {'top1 d':>11s}  {'std(log yN)':>12s}")
    G_chain_M20 = float((firm_domar_weights(chain) ** 2).sum())
    for gamma in [3.0, 2.5, 2.0, 1.5, 1.2, 1.05]:
        Gs = []
        top1s = []
        for seed in range(200):
            net = pareto_outdegree(np.full(N, 20), alpha,
                                   shape=gamma, common=True, seed=seed)
            d = firm_domar_weights(net)
            Gs.append(float((d ** 2).sum()))
            top1s.append(float(d.max()))
        Gs = np.array(Gs); top1s = np.array(top1s)
        std_yN = np.sqrt(Gs.mean() * tfp.sigma ** 2 / (1 - tfp.rho_lambda ** 2))
        print(f"    {gamma:6.2f}  {Gs.mean():11.4e}  {Gs.std():11.4e}  "
              f"{Gs.mean()/G_chain_M20:11.3f}  {top1s.mean():11.4e}  "
              f"{std_yN:12.4e}")

    # ---------------------------------------------------------------------
    # Stochastic simulation under each network (T = 400, seed = 42).
    # ---------------------------------------------------------------------
    print()
    print("  Stochastic simulation (T = 400, seed = 42):")
    print(f"    {'network':40s}  {'std(log y_N) emp':>16s}  "
          f"{'std(log y_N) thy':>16s}")
    for label, net in nets:
        _, log_yN = simulate(net, tfp, T=400, seed=42)
        thy = variance_decomposition(net, tfp)["total_std"]
        print(f"    {label:40s}  {log_yN.std():16.4e}  {thy:16.4e}")

    # ---------------------------------------------------------------------
    # Phase 2(b'): lagged firm-level IRFs.  Verify the cumulative-variance
    # identity matches the single-firm dynamics.py result on the chain.
    # ---------------------------------------------------------------------
    ss_2a = solve_steady_state_monetary(
        ModelParameters(N=N, alpha=alpha, zeta=np.full(N, 0.3),
                        rho=0.05, L=1.0, g_share=0.20),
        MonetaryParameters(chi=0.5, theta=0.0),
    )
    horizons = np.array([0, 8, 16, 24, 48, 72, 96])
    print()
    print(f"  Lagged firm-level cumulative variance share (Phase 2(a) θ=0 anchor)"
          f", t_n*(n>=2)={ss_2a.t[1]:.2f}:")
    print(f"    {'network':40s}  " +
          "  ".join(f"h={h:>3d}" for h in horizons))
    for label, net in nets:
        cv = cumulative_variance_share_firm(net, tfp, ss_2a.t, horizons)
        row = f"    {label:40s}  " + "  ".join(
            f"{cv['total_share'][hi]:5.3f}"
            for hi in range(len(horizons))
        )
        print(row)

    # Verify chain firm-level total share matches dynamics.py
    from dynamics import (
        cumulative_variance_share as cvs_stage,
        TfpParameters as StageTfp,
    )
    cvs_ref = cvs_stage(
        ModelParameters(N=N, alpha=alpha, zeta=np.full(N, 0.3),
                        rho=0.05, L=1.0, g_share=0.20),
        StageTfp(rho_lambda=tfp.rho_lambda, sigma=tfp.sigma),
        ss_2a.t, horizons,
    )
    cv_chain = cumulative_variance_share_firm(chain, tfp, ss_2a.t, horizons)
    err = float(np.max(np.abs(cv_chain["total_share"] - cvs_ref["total_share"])))
    print(f"    chain vs dynamics.py max|Δshare|: {err:.3e}  "
          f"({'matches' if err < 1e-12 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
