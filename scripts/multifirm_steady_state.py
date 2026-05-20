"""
Phase (c) — Firm-level steady state of the multi-firm chain network.

Anchor
------
Aggregate quantities (r, w, y_N, K_d, t_n^*) are taken from the Phase 2(a)
monetary closure at theta = 0 (see steady_state.py).  Under our
final-good aggregator (CD over stage-N firms with uniform 1/M_N
weights) and homogeneous within-stage parameters, every realization of
sourcing weights gives the same stage-aggregate influence V_n = beta_n,
so aggregate macro variables are network-invariant.  The new content
of Phase (c) is the within-stage distribution of:

    v_{i,n}        firm-level influence
    d_{i,n}        firm-level Domar weight = v_{i,n} * alpha_n
    R_{i,n}        firm revenue share (= v_{i,n}, in units of y_N)
    L_{i,n}        firm labor (proportional to v_{i,n} within a stage)

Firm labor is recovered from the labor FOC at firm (i,n):
    w * L_{i,n} * exp(r * t_n^*) = alpha_n * R_{i,n}.
Revenue at firm (i,n) inherits the chain's cumulative discount: the
influence vector v aggregates to V_n = beta_n stage-by-stage, but actual
revenue flows are discounted by exp(-r*(T_n - t_n^*)) where T_n =
sum_{m >= n} t_m^*, so

    R_{i,n} = v_{i,n} * exp(-r * (T_n - t_n^*)) * y_N,
    L_{i,n} = alpha_n * v_{i,n} * y_N * exp(-r * T_n) / w.

Summing over i within stage n recovers eq. (20) of model01.rtf:
sum_i L_{i,n} = alpha_n * beta_n * y_N * exp(-r * T_n) / w = L_n.

Verification: under the chain network L_{i,n} is constant within each
stage and the stage sums match `solve_steady_state_monetary` exactly.
Under random / Pareto sourcing, stage sums are identical but firm-level
labor inherits the influence-vector dispersion.
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
    solve_steady_state_monetary,
)
from network import (
    FirmNetwork,
    assemble_matrix,
    chain_network,
    final_demand_vector,
    firm_domar_weights,
    influence_vector,
    multistage_random,
    pareto_outdegree,
    stage_aggregate_domar,
    stage_hhi,
    stage_influence_total,
)


@dataclass
class MultifirmSteadyState:
    """Firm-level steady state stacked in (stage, firm) order."""

    network: FirmNetwork
    aggregate: SteadyState              # macro anchor (Phase 2a)
    A: np.ndarray                       # block IO matrix (cost shares)
    v: np.ndarray                       # influence vector
    d: np.ndarray                       # firm Domar weights
    R: np.ndarray                       # firm revenue (units of y_N)
    L_firm: np.ndarray                  # firm labor

    def stage_view(self, n: int):
        return self.network.stage_slice(n)

    def stage_stats(self) -> dict:
        """Per-stage summary statistics on v, d, L_firm."""
        N = self.network.N
        out = {
            "stage":       np.arange(1, N + 1),
            "M_n":         self.network.M.copy(),
            "V_n":         stage_influence_total(self.network, self.v),
            "D_n":         stage_aggregate_domar(self.network, self.d),
            "L_n":         np.array(
                [self.L_firm[self.stage_view(n)].sum() for n in range(N)]
            ),
            "hhi_v":       stage_hhi(self.network, self.v),
            "max_share_v": np.array([
                (self.v[self.stage_view(n)].max() / self.v[self.stage_view(n)].sum())
                if self.v[self.stage_view(n)].sum() > 0 else np.nan
                for n in range(N)
            ]),
            "min_share_v": np.array([
                (self.v[self.stage_view(n)].min() / self.v[self.stage_view(n)].sum())
                if self.v[self.stage_view(n)].sum() > 0 else np.nan
                for n in range(N)
            ]),
        }
        return out


def solve_multifirm_steady_state(
    network: FirmNetwork,
    params: ModelParameters,
    mon: MonetaryParameters,
) -> MultifirmSteadyState:
    """Anchor at Phase 2(a) and dress with firm-level objects."""
    if network.N != params.N:
        raise ValueError("network.N must equal params.N")
    if not np.allclose(network.alpha, params.alpha):
        raise ValueError("network.alpha must equal params.alpha")

    ss = solve_steady_state_monetary(params, mon)

    A = assemble_matrix(network)
    v = influence_vector(network, A)
    d = firm_domar_weights(network, v)

    # Cumulative discount T_n = sum_{m >= n} t_m^*
    T = np.cumsum(ss.t[::-1])[::-1]

    R = np.empty_like(v)
    L_firm = np.empty_like(v)
    for n in range(network.N):
        sl = network.stage_slice(n)
        # Revenue carries the chain's downstream discount e^{-r (T_n - t_n)}.
        R[sl] = v[sl] * np.exp(-ss.r * (T[n] - ss.t[n])) * ss.y_N
        # Labor FOC: w L_{i,n} e^{r t_n} = alpha_n R_{i,n}
        L_firm[sl] = (
            params.alpha[n] * v[sl] * ss.y_N
            * np.exp(-ss.r * T[n]) / ss.w
        )

    return MultifirmSteadyState(
        network=network, aggregate=ss, A=A, v=v, d=d, R=R, L_firm=L_firm,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_chain_match(mfss: MultifirmSteadyState, tol: float = 1e-12) -> dict:
    """
    Sanity check: under the chain network, the firm-level steady state
    must replicate the aggregate Phase 2(a) numbers exactly.  Stage-aggregate
    Domar weights must equal alpha_n * beta_n.  Stage labor must equal the
    eq.-(20) chain solution.
    """
    p = mfss.network
    ss = mfss.aggregate
    alpha = mfss.network.alpha
    beta = beta_weights(alpha)

    V = stage_influence_total(p, mfss.v)
    D = stage_aggregate_domar(p, mfss.d)
    L_stage = np.array(
        [mfss.L_firm[p.stage_slice(n)].sum() for n in range(p.N)]
    )

    res = {
        "domar_sum":        float(abs(mfss.d.sum() - 1.0)),
        "stage_V_vs_beta":  float(np.max(np.abs(V - beta))),
        "stage_D_vs_alpha_beta": float(np.max(np.abs(D - alpha * beta))),
        "stage_labor_vs_aggregate": float(np.max(np.abs(L_stage - ss.L_n))),
        "total_labor":      float(L_stage.sum()),
    }
    res["passed"] = all(v < tol for k, v in res.items()
                        if k not in ("passed", "total_labor"))
    return res


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


def _print_stage_table(mfss: MultifirmSteadyState, label: str) -> None:
    s = mfss.stage_stats()
    p = mfss.network
    print(f"  Network: {label}")
    print(
        f"    {'n':>3s}  {'M_n':>4s}  {'V_n':>10s}  {'D_n=sum d_{i,n}':>16s}  "
        f"{'HHI(v_n)':>10s}  {'min sh':>8s}  {'max sh':>8s}  {'L_n':>10s}"
    )
    for n in range(p.N):
        print(
            f"    {n+1:3d}  {p.M[n]:4d}  {s['V_n'][n]:10.4e}  "
            f"{s['D_n'][n]:16.4e}  {s['hhi_v'][n]:10.4f}  "
            f"{s['min_share_v'][n]:8.4f}  {s['max_share_v'][n]:8.4f}  "
            f"{s['L_n'][n]:10.4e}"
        )


def main() -> None:
    params = _baseline_params()
    mon = MonetaryParameters(chi=0.5, theta=0.0)
    M_vec = np.full(params.N, 20)

    print("=" * 80)
    print("Phase (c) — multi-firm steady state, anchor: Phase 2(a) at theta = 0")
    print("=" * 80)
    print()

    chain = chain_network(M_vec, params.alpha)
    rand4 = multistage_random(M_vec, params.alpha, k=4, seed=42)
    par15 = pareto_outdegree(M_vec, params.alpha, shape=1.5, common=True, seed=42)

    for net, label in [(chain, "chain"),
                       (rand4, "multistage_random(k=4, seed=42)"),
                       (par15, "pareto_outdegree(shape=1.5, common, seed=42)")]:
        mfss = solve_multifirm_steady_state(net, params, mon)
        _print_stage_table(mfss, label)
        print()
        if label == "chain":
            res = verify_chain_match(mfss)
            print(f"    chain regression test:")
            for k, v in res.items():
                print(f"      {k:30s} = {v}")
            print()


if __name__ == "__main__":
    main()
