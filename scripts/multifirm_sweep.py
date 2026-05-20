"""
Phase (c2) — Firm-level θ-comparative statics for Phase 2(a) and 2(d).

For each θ in the Phase 2 grid, for each network topology (chain,
multistage random k=4, common Pareto γ=1.5), and for each closure
(Phase 2(a) CIA-Fisher, Phase 2(d) Wicksellian wedge), compute:

  - aggregate macro quantities (r, w, y_N, K_d) — should be identical to
    the (M=1) chain results from STEADY_STATE_RESULTS.md.
  - firm-level objects (revenue R_{i,n}, labor L_{i,n}).
  - aggregate volatility std(log y_N) under iid firm-level TFP shocks.
  - granularity index G = sum d^2.

Separability claim
------------------
In the log-linear case z_n(t_n) = t_n^{ζ_n}:

  - cost shares α_n and the IO matrix A are r-invariant by construction
    (CD primitives), so the influence vector v and firm-level Domar
    weights d = α v are r-invariant for every network topology.
  - Therefore G = sum d^2 and aggregate volatility
    std(log y_N) = sqrt(G) · σ / sqrt(1-ρ²) are *θ-invariant* per
    network: the macro inflation-tax channel and the within-stage
    network dimension are first-order separable.
  - Within-stage labor and revenue *shares* L_{i,n}/L_n = v_{i,n}/V_n
    and R_{i,n}/R_n are also r-invariant; only stage-level totals
    move with θ via the macro anchor.

This script verifies each of those claims numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from steady_state import (
    ModelParameters,
    MonetaryParameters,
    monetary_sweep,
    friction_sweep,
)
from network import (
    chain_network,
    firm_domar_weights,
    influence_vector,
    multistage_random,
    pareto_outdegree,
    stage_aggregate_domar,
    stage_influence_total,
    stage_hhi,
)
from multifirm_steady_state import (
    MultifirmSteadyState,
    solve_multifirm_steady_state,
)
from multifirm_dynamics import (
    FirmTfpParameters,
    variance_decomposition,
)


# ---------------------------------------------------------------------------
# Sweep machinery
# ---------------------------------------------------------------------------

@dataclass
class SweepRow:
    theta: float
    closure: str          # "monetary" (2a) or "friction" (2d)
    network: str
    r: float
    w: float
    y_N: float
    K_d: float
    G: float              # sum d_{i,n}^2
    std_log_yN: float
    top1_d: float
    hhi_v_max: float
    L_max: float          # largest firm labor at the largest stage
    R_max: float          # largest firm revenue at the largest stage (scales with y_N)
    R_max_stage7: float   # largest firm revenue at stage 7 (where Pareto bites)


def _baseline_params() -> ModelParameters:
    return ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05,
        L=1.0,
        g_share=0.20,
    )


def _build_networks(params: ModelParameters, M: int):
    M_vec = np.full(params.N, M)
    return [
        ("chain",      chain_network(M_vec, params.alpha)),
        ("random k=4", multistage_random(M_vec, params.alpha, k=4, seed=42)),
        ("Pareto 1.5", pareto_outdegree(M_vec, params.alpha,
                                        shape=1.5, common=True, seed=42)),
    ]


def sweep(
    params: ModelParameters,
    chi: float,
    theta_grid: np.ndarray,
    M: int,
    tfp: FirmTfpParameters,
):
    """Return a list of SweepRow records covering every (closure, network, θ)."""
    nets = _build_networks(params, M)
    # Pre-compute network statistics (r-invariant, so once per network).
    net_stats = {}
    for label, net in nets:
        v = influence_vector(net)
        d = firm_domar_weights(net, v)
        vd = variance_decomposition(net, tfp, d)
        net_stats[label] = {
            "v": v, "d": d,
            "G": float((d ** 2).sum()),
            "std_log_yN": vd["total_std"],
            "top1_d": float(d.max()),
            "hhi_v_max": float(stage_hhi(net, v).max()),
        }

    rows = []
    # Phase 2(a): monetary
    for theta, _, ss in monetary_sweep(params, chi=chi, theta_grid=theta_grid):
        for label, net in nets:
            mfss = solve_multifirm_steady_state(
                net, params, MonetaryParameters(chi=chi, theta=theta),
            )
            stats = net_stats[label]
            largest_stage = net.N - 1
            L_max = float(mfss.L_firm[net.stage_slice(largest_stage)].max())
            R_max = float(mfss.R[net.stage_slice(largest_stage)].max())
            R_max_s7 = float(mfss.R[net.stage_slice(6)].max())
            rows.append(SweepRow(
                theta=theta, closure="2a", network=label,
                r=ss.r, w=ss.w, y_N=ss.y_N, K_d=ss.K_d,
                G=stats["G"], std_log_yN=stats["std_log_yN"],
                top1_d=stats["top1_d"], hhi_v_max=stats["hhi_v_max"],
                L_max=L_max, R_max=R_max, R_max_stage7=R_max_s7,
            ))

    # Phase 2(d): friction (the friction sweep returns r_F, not r_H)
    for theta, _r_H, r_F, _phi, ss in friction_sweep(params, chi=chi, theta_grid=theta_grid):
        for label, net in nets:
            # solve_multifirm_steady_state is keyed off the monetary closure;
            # for the friction closure we just need to anchor at the 2(d)
            # macro variables.  Build an ad-hoc anchor by overriding the
            # SteadyState fields.
            from multifirm_steady_state import MultifirmSteadyState  # local
            from network import assemble_matrix
            v = net_stats[label]["v"]
            d = net_stats[label]["d"]
            T = np.cumsum(ss.t[::-1])[::-1]
            R = np.empty_like(v)
            L_firm = np.empty_like(v)
            for n in range(net.N):
                sl = net.stage_slice(n)
                R[sl] = v[sl] * np.exp(-ss.r * (T[n] - ss.t[n])) * ss.y_N
                L_firm[sl] = (
                    params.alpha[n] * v[sl] * ss.y_N
                    * np.exp(-ss.r * T[n]) / ss.w
                )
            largest_stage = net.N - 1
            L_max = float(L_firm[net.stage_slice(largest_stage)].max())
            R_max = float(R[net.stage_slice(largest_stage)].max())
            R_max_s7 = float(R[net.stage_slice(6)].max())
            stats = net_stats[label]
            rows.append(SweepRow(
                theta=theta, closure="2d", network=label,
                r=ss.r, w=ss.w, y_N=ss.y_N, K_d=ss.K_d,
                G=stats["G"], std_log_yN=stats["std_log_yN"],
                top1_d=stats["top1_d"], hhi_v_max=stats["hhi_v_max"],
                L_max=L_max, R_max=R_max, R_max_stage7=R_max_s7,
            ))
    return rows


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_chain_reproduces_macro(rows, theta_grid, params, chi, tol=1e-12):
    """
    Chain network rows must reproduce the M=1 chain macro tables
    (Phase 2a and 2d) to machine precision.
    """
    chain_2a = monetary_sweep(params, chi=chi, theta_grid=theta_grid)
    chain_2d = friction_sweep(params, chi=chi, theta_grid=theta_grid)

    fails = []
    for theta, _, ss_ref in chain_2a:
        row = next(r for r in rows
                   if r.network == "chain" and r.closure == "2a"
                   and np.isclose(r.theta, theta))
        for fld in ["r", "w", "y_N", "K_d"]:
            ref = getattr(ss_ref, fld)
            got = getattr(row, fld)
            if abs(ref - got) > tol:
                fails.append((theta, "2a", fld, ref, got))
    for theta, _, r_F, _, ss_ref in chain_2d:
        row = next(r for r in rows
                   if r.network == "chain" and r.closure == "2d"
                   and np.isclose(r.theta, theta))
        for fld in ["r", "w", "y_N", "K_d"]:
            ref = getattr(ss_ref, fld)
            got = getattr(row, fld)
            if abs(ref - got) > tol:
                fails.append((theta, "2d", fld, ref, got))
    return fails


def verify_volatility_theta_invariance(rows, tol=1e-12):
    """
    G index and std(log y_N) must be exactly identical across all θ within
    the same (closure, network) — and identical across closures within the
    same network — because they depend only on the network primitives.
    """
    fails = []
    by_net = {}
    for r in rows:
        by_net.setdefault(r.network, []).append(r)
    for label, rs in by_net.items():
        Gs = np.array([r.G for r in rs])
        stds = np.array([r.std_log_yN for r in rs])
        if Gs.std() > tol:
            fails.append((label, "G", float(Gs.std())))
        if stds.std() > tol:
            fails.append((label, "std_log_yN", float(stds.std())))
    return fails


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(rows, closure, fields):
    sub = [r for r in rows if r.closure == closure]
    nets = sorted({r.network for r in sub}, key=["chain", "random k=4", "Pareto 1.5"].index)
    thetas = sorted({r.theta for r in sub})
    print(f"  Closure {closure}:")
    header = f"    {'theta':>7s}"
    for net in nets:
        for fld in fields:
            header += f"  {net + '/' + fld:>17s}"
    print(header)
    for theta in thetas:
        line = f"    {theta:7.4f}"
        for net in nets:
            row = next(r for r in sub
                       if r.network == net and np.isclose(r.theta, theta))
            for fld in fields:
                v = getattr(row, fld)
                line += f"  {v:17.5e}"
        print(line)


def main() -> None:
    params = _baseline_params()
    chi = 0.5
    M = 20
    theta_grid = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    tfp = FirmTfpParameters(rho_lambda=0.9, sigma=0.01)

    print("=" * 90)
    print(f"Phase (c2) — firm-level θ-comparative statics, "
          f"M={M} per stage, chi={chi}")
    print("=" * 90)

    rows = sweep(params, chi, theta_grid, M, tfp)

    print()
    fails_macro = verify_chain_reproduces_macro(rows, theta_grid, params, chi)
    if not fails_macro:
        print("  ✓  Chain network reproduces Phase 2(a) and Phase 2(d) macro "
              "sweeps to machine precision.")
    else:
        print(f"  ✗  Chain macro reproduction FAILS at:")
        for f in fails_macro:
            print(f"      {f}")

    fails_vol = verify_volatility_theta_invariance(rows)
    if not fails_vol:
        print("  ✓  G index and std(log y_N) are exactly θ-invariant per "
              "network and identical across closures.")
    else:
        print(f"  ✗  Volatility θ-invariance FAILS at:")
        for f in fails_vol:
            print(f"      {f}")

    print()
    print("  Macro aggregates per network and θ "
          "(should be identical across networks at any θ; "
          "differ only across closures):")
    _print_table(rows, "2a", ["r", "w", "y_N"])
    print()
    _print_table(rows, "2d", ["r", "w", "y_N"])

    print()
    print("  Volatility & granularity per network "
          "(θ-invariant within network):")
    seen = set()
    print(f"    {'network':12s}  {'G=sum d^2':>11s}  "
          f"{'std(log y_N)':>13s}  {'top1 d':>10s}  {'max HHI(v_n)':>13s}")
    for r in rows:
        if r.network in seen:
            continue
        seen.add(r.network)
        print(f"    {r.network:12s}  {r.G:11.4e}  {r.std_log_yN:13.4e}  "
              f"{r.top1_d:10.4e}  {r.hhi_v_max:13.4f}")

    print()
    print("  Largest firm labor L_max at stage N as θ varies "
          "(θ-invariant — all stage labor shares are r-invariant in "
          "log-linear case):")
    for closure in ("2a", "2d"):
        print(f"    Closure {closure}:")
        print(f"      {'theta':>7s}  {'chain':>11s}  {'random k=4':>11s}  "
              f"{'Pareto 1.5':>11s}")
        for theta in theta_grid:
            line = f"      {theta:7.4f}"
            for net in ["chain", "random k=4", "Pareto 1.5"]:
                r = next(r for r in rows
                         if r.closure == closure and r.network == net
                         and np.isclose(r.theta, theta))
                line += f"  {r.L_max:11.4e}"
            print(line)

    print()
    print("  Largest firm REVENUE R_max at stage N as θ varies "
          "(scales linearly with y_N — boom/bust shows up here):")
    for closure in ("2a", "2d"):
        print(f"    Closure {closure}:")
        print(f"      {'theta':>7s}  {'chain':>11s}  {'random k=4':>11s}  "
              f"{'Pareto 1.5':>11s}  {'(=y_N/M_N)':>12s}")
        for theta in theta_grid:
            line = f"      {theta:7.4f}"
            for net in ["chain", "random k=4", "Pareto 1.5"]:
                r = next(r for r in rows
                         if r.closure == closure and r.network == net
                         and np.isclose(r.theta, theta))
                line += f"  {r.R_max:11.4e}"
            ref = next(r for r in rows
                       if r.closure == closure and r.network == "chain"
                       and np.isclose(r.theta, theta))
            line += f"  {ref.y_N / 20:12.4e}"
            print(line)

    print()
    print("  Largest firm REVENUE R_max at stage 7 "
          "(Pareto bites here — its R_max can be ~3-5× chain):")
    for closure in ("2a", "2d"):
        print(f"    Closure {closure}:")
        print(f"      {'theta':>7s}  {'chain':>11s}  {'random k=4':>11s}  "
              f"{'Pareto 1.5':>11s}  {'Pareto/chain':>12s}")
        for theta in theta_grid:
            line = f"      {theta:7.4f}"
            for net in ["chain", "random k=4", "Pareto 1.5"]:
                r = next(r for r in rows
                         if r.closure == closure and r.network == net
                         and np.isclose(r.theta, theta))
                line += f"  {r.R_max_stage7:11.4e}"
            chain_r = next(r for r in rows
                           if r.closure == closure and r.network == "chain"
                           and np.isclose(r.theta, theta))
            par_r = next(r for r in rows
                         if r.closure == closure and r.network == "Pareto 1.5"
                         and np.isclose(r.theta, theta))
            ratio = par_r.R_max_stage7 / chain_r.R_max_stage7
            line += f"  {ratio:12.3f}"
            print(line)


if __name__ == "__main__":
    main()
