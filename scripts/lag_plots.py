"""
Phase 2(b') — Plots for the lagged IRF extension.

Outputs (PDF, in Figures/):
  lag_irfs.pdf     Three panels: (1) lagged vs instantaneous IRFs for
                   selected stages at θ=0; (2) cumulative variance share
                   by horizon for θ ∈ {0, 0.02, 0.04}; (3) T_lag[n] vs
                   stage for the same θ grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from steady_state import (
    ModelParameters,
    MonetaryParameters,
    solve_steady_state_monetary,
)
from dynamics import (
    TfpParameters,
    cumulative_variance_share,
    domar_weights,
    downstream_lag,
    tfp_irf,
    tfp_irf_lagged,
)


FIG_DIR = Path(__file__).resolve().parents[1] / "Figures"


def _baseline() -> ModelParameters:
    return ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05, L=1.0, g_share=0.20,
    )


def plot_lag_irfs() -> None:
    params = _baseline()
    chi = 0.5
    tfp = TfpParameters(rho_lambda=0.9, sigma=0.01)

    ss0 = solve_steady_state_monetary(params, MonetaryParameters(chi=chi, theta=0.0))
    horizons = np.arange(0, 200)
    stages_to_plot = [0, 4, 8, 9]   # stages 1, 5, 9, 10
    colors = {0: "tab:gray", 4: "tab:blue", 8: "tab:orange", 9: "tab:red"}
    labels = {0: "stage 1", 4: "stage 5", 8: "stage 9", 9: "stage 10"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                             constrained_layout=True)
    axL, axM, axR = axes

    # Panel L: lagged vs instantaneous IRFs at θ=0
    irf_inst = tfp_irf(params, tfp, horizons)
    irf_lag = tfp_irf_lagged(params, tfp, horizons, ss0.t)
    for n in stages_to_plot:
        axL.plot(horizons, irf_inst[:, n], ":", color=colors[n], alpha=0.6,
                 label=f"{labels[n]} (instant)" if n == 9 else None)
        axL.plot(horizons, irf_lag[:, n], "-", color=colors[n], linewidth=1.6,
                 label=f"{labels[n]} (lagged)")
    axL.set_xlabel(r"horizon $h$ (periods)")
    axL.set_ylabel(r"$d\log y_N$ / $d\eta_n$")
    axL.set_title(r"IRFs at $\vartheta=0$ — lagged (solid) vs instantaneous (dotted)")
    axL.set_yscale("symlog", linthresh=1e-5)
    axL.legend(fontsize=9, loc="upper right")
    axL.grid(True, which="both", alpha=0.3)

    # Panel M: cumulative variance share by horizon, θ-sweep
    horizons_cv = np.arange(0, 401)
    for theta, color in [(0.00, "tab:blue"),
                         (0.02, "tab:purple"),
                         (0.04, "tab:red")]:
        ss = solve_steady_state_monetary(params, MonetaryParameters(chi=chi, theta=theta))
        cv = cumulative_variance_share(params, tfp, ss.t, horizons_cv)
        axM.plot(horizons_cv, cv["total_share"], color=color,
                 label=rf"$\vartheta={theta:.2f}$ ($r={ss.r:.3f}$, "
                       rf"$t_n^*={ss.t[1]:.1f}$)",
                 linewidth=1.6)
    axM.axhline(1.0, ls=":", color="gray", alpha=0.5)
    axM.set_xlabel(r"horizon $h$ (periods)")
    axM.set_ylabel(r"share of steady-state Var$(\log y_N)$ realised")
    axM.set_title(r"Cumulative variance share — boom periods propagate slower")
    axM.set_xscale("log")
    axM.legend(fontsize=9, loc="lower right")
    axM.grid(True, which="both", alpha=0.3)

    # Panel R: T_lag[n] vs stage n for θ ∈ {0, 0.02, 0.04}
    stages = np.arange(1, params.N + 1)
    for theta, color in [(0.00, "tab:blue"),
                         (0.02, "tab:purple"),
                         (0.04, "tab:red")]:
        ss = solve_steady_state_monetary(params, MonetaryParameters(chi=chi, theta=theta))
        T_lag = downstream_lag(ss.t)
        axR.plot(stages, T_lag, "o-", color=color,
                 label=rf"$\vartheta={theta:.2f}$",
                 linewidth=1.6, markersize=4)
    axR.set_xlabel("stage $n$")
    axR.set_ylabel(r"$T_{n+1}^{\mathrm{lag}} = \sum_{m>n} t_m^*$ (periods)")
    axR.set_title(r"Maturation lag scales as $1/r$")
    axR.legend(fontsize=9, loc="upper right")
    axR.grid(True, alpha=0.3)
    axR.set_xticks(stages)

    fig.suptitle(
        r"Phase 2(b') — embedding maturation lag $T^{\mathrm{lag}}_{n+1}$ in the IRFs",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "lag_irfs.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plot_lag_irfs()
    print(f"Wrote: {FIG_DIR}/lag_irfs.pdf")


if __name__ == "__main__":
    main()
