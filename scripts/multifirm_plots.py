"""
Phase (c) — Visualisations of the multi-firm chain network.

Outputs (PDF, in Figures/):
  io_matrix.pdf           Block-IO heatmap for chain / random / Pareto.
  firm_domar.pdf          Within-stage Domar weight distribution at stages
                          7, 8, 9, 10 across the three networks.
  granularity.pdf         G = sum d^2 vs M (log-log) for the three
                          networks; secondary panel: Pareto-shape sweep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from network import (
    assemble_matrix,
    chain_network,
    firm_domar_weights,
    multistage_random,
    pareto_outdegree,
)
from multifirm_dynamics import (
    FirmTfpParameters,
    granularity_curve,
)
from multifirm_sweep import sweep
from steady_state import ModelParameters


FIG_DIR = Path(__file__).resolve().parents[1] / "Figures"


def _alpha(N: int = 10) -> np.ndarray:
    a = np.full(N, 0.67)
    a[0] = 1.0
    return a


# ---------------------------------------------------------------------------
# Figure 1 — IO matrix heatmaps
# ---------------------------------------------------------------------------

def plot_io_matrix() -> None:
    N, M = 10, 20
    alpha = _alpha(N)
    M_vec = np.full(N, M)

    nets = [
        ("chain (uniform)",                chain_network(M_vec, alpha)),
        ("multistage random (k = 4)",      multistage_random(M_vec, alpha, k=4, seed=42)),
        ("Pareto out-degree (shape = 1.5)", pareto_outdegree(M_vec, alpha,
                                                              shape=1.5,
                                                              common=True,
                                                              seed=42)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for ax, (label, net) in zip(axes, nets):
        A = assemble_matrix(net)
        # Show log-scale to make Pareto block heterogeneity legible.
        with np.errstate(divide="ignore"):
            logA = np.where(A > 0, np.log10(A), np.nan)
        im = ax.imshow(logA, cmap="viridis", aspect="auto",
                       interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("supplier (firm, stacked stage 1 → 10)")
        ax.set_ylabel("buyer (firm, stacked stage 1 → 10)")
        # Stage gridlines
        offs = net.stage_offsets()
        for o in offs[1:-1]:
            ax.axhline(o - 0.5, color="white", linewidth=0.4, alpha=0.5)
            ax.axvline(o - 0.5, color="white", linewidth=0.4, alpha=0.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\log_{10}\, A_{ij}$", fontsize=9)
    fig.suptitle(
        r"Block input-output matrix $A$ "
        r"$\,(N=10,\;M=20$ per stage; sourcing restricted to stage $n-1)$",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "io_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Within-stage Domar weight distributions (stages 7-10)
# ---------------------------------------------------------------------------

def plot_firm_domar() -> None:
    N, M = 10, 20
    alpha = _alpha(N)
    M_vec = np.full(N, M)

    nets = [
        ("chain",     chain_network(M_vec, alpha),                  "tab:blue"),
        ("random k=4", multistage_random(M_vec, alpha, k=4, seed=42),
                                                                    "tab:green"),
        ("Pareto 1.5", pareto_outdegree(M_vec, alpha, shape=1.5,
                                        common=True, seed=42),      "tab:red"),
    ]

    stages_to_plot = [6, 7, 8, 9]   # stages 7..10 (zero-indexed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, n in zip(axes.flat, stages_to_plot):
        bar_x = np.arange(M)
        width = 0.27
        for k, (label, net, color) in enumerate(nets):
            d = firm_domar_weights(net)
            ds = d[net.stage_slice(n)]
            ax.bar(bar_x + (k - 1) * width, ds, width=width, label=label,
                   color=color, alpha=0.8)
        ax.set_title(f"stage n = {n + 1}", fontsize=11)
        ax.set_xlabel("firm index $i$")
        ax.set_ylabel(r"$d_{i,n} = v_{i,n}$")
        ax.set_xticks(bar_x[::2])
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Firm-level Domar weights within selected stages "
        r"$(M=20,\;\alpha_{n\geq 2}=0.67)$",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "firm_domar.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Granularity scaling
# ---------------------------------------------------------------------------

def plot_granularity() -> None:
    N = 10
    alpha = _alpha(N)
    tfp = FirmTfpParameters(rho_lambda=0.9, sigma=0.01)

    M_grid = np.array([5, 10, 20, 50, 100, 200, 400])
    chain  = granularity_curve(alpha, M_grid, "chain")
    rand4  = granularity_curve(alpha, M_grid, "random", k=4,
                               seeds=np.arange(50))
    par15  = granularity_curve(alpha, M_grid, "pareto",
                               pareto_shape=1.5, pareto_common=True,
                               seeds=np.arange(100))
    par12  = granularity_curve(alpha, M_grid, "pareto",
                               pareto_shape=1.2, pareto_common=True,
                               seeds=np.arange(100))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 4.6), constrained_layout=True
    )

    # Left: granularity scaling, log-log.
    def plot_curve(ax, curve, label, color, marker):
        ax.plot(curve["M"], curve["G_mean"], marker=marker, color=color,
                label=label)
        if curve["G_std"].any():
            ax.fill_between(curve["M"],
                            curve["G_mean"] - curve["G_std"],
                            np.maximum(curve["G_mean"] - curve["G_std"],
                                       1e-6),
                            alpha=0.0)
            ax.errorbar(curve["M"], curve["G_mean"], yerr=curve["G_std"],
                        fmt="none", color=color, alpha=0.5, capsize=3)

    plot_curve(ax_left, chain,  "chain",                    "tab:blue",   "o")
    plot_curve(ax_left, rand4,  "random (k=4)",             "tab:green",  "s")
    plot_curve(ax_left, par15,  "Pareto, shape = 1.5",      "tab:red",    "v")
    plot_curve(ax_left, par12,  "Pareto, shape = 1.2",      "tab:purple", "^")

    # Reference 1/M line through chain
    M_ref = np.array(M_grid, dtype=float)
    ax_left.plot(M_ref, chain["G_mean"][0] * (M_grid[0] / M_ref),
                 ls=":", color="gray", label=r"$\propto 1/M$")
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("firms per stage $M$")
    ax_left.set_ylabel(r"$G(M) = \sum_{i,n} d_{i,n}^2$")
    ax_left.set_title("Granularity scaling")
    ax_left.legend(fontsize=9, loc="best")
    ax_left.grid(True, which="both", alpha=0.3)

    # Right: Pareto-shape sweep at fixed M.
    M_fixed = 50
    M_vec = np.full(N, M_fixed)
    G_chain_fixed = float(
        (firm_domar_weights(chain_network(M_vec, alpha)) ** 2).sum()
    )
    gammas = np.array([3.0, 2.5, 2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05])
    G_means, G_stds, top1_means = [], [], []
    seeds = range(200)
    for g in gammas:
        Gs, top1 = [], []
        for s in seeds:
            net = pareto_outdegree(M_vec, alpha, shape=float(g),
                                   common=True, seed=int(s))
            d = firm_domar_weights(net)
            Gs.append(float((d ** 2).sum()))
            top1.append(float(d.max()))
        G_means.append(np.mean(Gs))
        G_stds.append(np.std(Gs))
        top1_means.append(np.mean(top1))
    G_means = np.asarray(G_means)
    G_stds  = np.asarray(G_stds)
    top1_means = np.asarray(top1_means)

    ax_right.plot(gammas, G_means / G_chain_fixed, "o-", color="tab:red",
                  label=r"$G(\gamma) / G(\mathrm{chain})$")
    ax_right.fill_between(gammas,
                          (G_means - G_stds) / G_chain_fixed,
                          (G_means + G_stds) / G_chain_fixed,
                          color="tab:red", alpha=0.15)
    ax_right.axhline(1.0, ls=":", color="gray", label="chain reference")
    ax_right.invert_xaxis()
    ax_right.set_xlabel(r"Pareto shape $\gamma$ (heavier tail $\to$)")
    ax_right.set_ylabel(r"$G(\gamma)\,/\,G(\mathrm{chain})$")
    ax_right.set_title(rf"Pareto-shape sweep at $M={M_fixed}$ (mean ± std, 200 seeds)")
    ax_right.legend(fontsize=9, loc="upper left")
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(
        "Granularity in the multi-firm chain — chain decay "
        r"$1/M$, fat-tailed networks slower (with realisation noise)",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "granularity.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Boom/bust × granularity (Phase c2)
# ---------------------------------------------------------------------------

def plot_boom_bust_granularity() -> None:
    """
    Three panels documenting the separability of the macro and network
    dimensions in the log-linear case.

    Left:  std(log y_N) vs θ for chain / random / Pareto under both
           closures.  Each network gives a flat line (θ-invariant); the
           lines stratify by network with chain at the bottom.
    Mid:   y_N vs θ — the macro boom/bust curves from Phase 2(a)/2(d),
           identical across networks.
    Right: R_max at stage 7 vs θ for each network and closure.  Levels
           track y_N (boom/bust); the across-network ratio is constant
           in θ (network-determined HHI).
    """
    params = ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05, L=1.0, g_share=0.20,
    )
    chi = 0.5
    M = 20
    theta_grid = np.linspace(0.0, 0.04, 21)
    tfp = FirmTfpParameters(rho_lambda=0.9, sigma=0.01)

    rows = sweep(params, chi, theta_grid, M, tfp)

    nets = ["chain", "random k=4", "Pareto 1.5"]
    colors = {"chain": "tab:blue", "random k=4": "tab:green",
              "Pareto 1.5": "tab:red"}
    style = {"2a": "-", "2d": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                             constrained_layout=True)
    axL, axM, axR = axes

    # Left panel: std(log y_N) vs theta
    for closure in ("2a", "2d"):
        for net in nets:
            xs = sorted(theta_grid)
            ys = [next(r for r in rows
                       if r.closure == closure and r.network == net
                       and np.isclose(r.theta, t)).std_log_yN for t in xs]
            label = f"{net} ({closure})"
            axL.plot(xs, ys, style[closure], color=colors[net], label=label,
                     linewidth=1.4)
    axL.set_xlabel(r"money growth $\vartheta$")
    axL.set_ylabel(r"std$(\log y_N)$ — aggregate volatility")
    axL.set_title(r"Volatility is $\vartheta$-invariant per network")
    axL.set_ylim(bottom=0.0)
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=8, loc="upper right", ncol=2)

    # Middle panel: y_N vs theta (boom/bust macro, identical across nets)
    for closure in ("2a", "2d"):
        xs = sorted(theta_grid)
        ys = [next(r for r in rows
                   if r.closure == closure and r.network == "chain"
                   and np.isclose(r.theta, t)).y_N for t in xs]
        label = "boom (Phase 2a)" if closure == "2a" else "bust (Phase 2d)"
        clr  = "tab:blue" if closure == "2a" else "tab:red"
        axM.plot(xs, ys, "-", color=clr, label=label, linewidth=2)
    axM.set_xlabel(r"money growth $\vartheta$")
    axM.set_ylabel(r"$y_N$ (final-good output)")
    axM.set_title(r"Macro boom/bust — network-invariant")
    axM.grid(True, alpha=0.3)
    axM.legend(fontsize=9)

    # Right panel: R_max at stage 7 vs theta — levels boom/bust, ratios fixed
    for closure in ("2a", "2d"):
        for net in nets:
            xs = sorted(theta_grid)
            ys = [next(r for r in rows
                       if r.closure == closure and r.network == net
                       and np.isclose(r.theta, t)).R_max_stage7 for t in xs]
            label = f"{net} ({closure})"
            axR.plot(xs, ys, style[closure], color=colors[net], label=label,
                     linewidth=1.4)
    axR.set_xlabel(r"money growth $\vartheta$")
    axR.set_ylabel(r"$\max_i R_{i,7}$  (largest firm revenue at stage 7)")
    axR.set_title(r"Firm-level boom/bust × network HHI")
    axR.set_yscale("log")
    axR.grid(True, alpha=0.3, which="both")
    axR.legend(fontsize=8, loc="lower right", ncol=2)

    fig.suptitle(
        "Phase (c2) — separability: macro inflation-tax channel and "
        "network granularity decouple in the log-linear case",
        fontsize=12,
    )
    fig.savefig(FIG_DIR / "boom_bust_granularity.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plot_io_matrix()
    plot_firm_domar()
    plot_granularity()
    plot_boom_bust_granularity()
    print(f"Wrote: {FIG_DIR}/io_matrix.pdf")
    print(f"Wrote: {FIG_DIR}/firm_domar.pdf")
    print(f"Wrote: {FIG_DIR}/granularity.pdf")
    print(f"Wrote: {FIG_DIR}/boom_bust_granularity.pdf")


if __name__ == "__main__":
    main()
