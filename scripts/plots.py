"""
scripts/plots.py — Phase 2 boom-vs-bust comparative statics.

Plots Phase 2(a) (CIA-Fisher household, naive money rule, no
asset-market clearing — the "boom" regime where lower r lengthens
chains and raises output) against Phase 2(d) (same household block
plus a wedge that enforces K^d = K^s on top — the "bust" regime
where the inflation tax on savers shrinks the capital pool and
forces a higher firm rate r_F).

The comparison is done at a *common* monetary-policy stance theta:
2(a) is what a naive textbook reading would predict; 2(d) is what
actually obtains once the asset market has to clear. The gap
between them is the "missing output" — Austrian malinvestment in
narrative terms — that opens up as theta rises beyond the
Wicksellian-neutrality point theta* at which the wedge phi crosses
zero.

Outputs (PDF vector format, into Figures/):
  boom_bust_output.pdf    — y_N vs theta in both regimes
  boom_bust_rates.pdf     — r (2a) vs r_F, r_H (2d)
  boom_bust_chains_w.pdf  — stage durations and real wage
  capital_market.pdf      — K_d, K_s, and the wedge phi
  boom_bust_summary.pdf   — 2x2 dashboard combining the above
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np

from steady_state import (
    ModelParameters,
    friction_sweep,
    monetary_sweep,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THETA_GRID = np.linspace(0.0, 0.045, 46)
CHI = 0.5
COLOR_BOOM = "#1f77b4"   # Phase 2(a)
COLOR_BUST = "#d62728"   # Phase 2(d)
LW = 2.0


def _baseline_params() -> ModelParameters:
    return ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05,
        L=1.0,
        g_share=0.20,
    )


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Run sweeps and pull arrays
# ---------------------------------------------------------------------------

def _run_sweeps(params: ModelParameters):
    sweep_a = monetary_sweep(params, CHI, THETA_GRID)
    sweep_d = friction_sweep(params, CHI, THETA_GRID)
    out = {
        "theta": THETA_GRID,
        "r_a":  np.array([ss.r   for _, _, ss in sweep_a]),
        "w_a":  np.array([ss.w   for _, _, ss in sweep_a]),
        "yN_a": np.array([ss.y_N for _, _, ss in sweep_a]),
        "Kd_a": np.array([ss.K_d for _, _, ss in sweep_a]),
        "t2_a": np.array([ss.t[1] for _, _, ss in sweep_a]),
        "rH_d": np.array([rH for _, rH, _, _, _ in sweep_d]),
        "rF_d": np.array([ss.r for _, _, _, _, ss in sweep_d]),
        "phi_d": np.array([phi for _, _, _, phi, _ in sweep_d]),
        "w_d":  np.array([ss.w   for _, _, _, _, ss in sweep_d]),
        "yN_d": np.array([ss.y_N for _, _, _, _, ss in sweep_d]),
        "Kd_d": np.array([ss.K_d for _, _, _, _, ss in sweep_d]),
        "t2_d": np.array([ss.t[1] for _, _, _, _, ss in sweep_d]),
    }
    out["Ks_a"] = out["w_a"] * params.L / (params.rho - out["r_a"])
    # theta* where the wedge crosses zero (Wicksellian-neutral monetary stance)
    out["theta_star"] = float(THETA_GRID[np.abs(out["phi_d"]).argmin()])
    return out


# ---------------------------------------------------------------------------
# Common annotations
# ---------------------------------------------------------------------------

def _shade_regimes(ax, theta_star, ymin=None, ymax=None):
    if ymin is None:
        ymin, ymax = ax.get_ylim()
    ax.axvspan(THETA_GRID[0], theta_star, alpha=0.06, color=COLOR_BOOM, zorder=0)
    ax.axvspan(theta_star, THETA_GRID[-1], alpha=0.06, color=COLOR_BUST, zorder=0)
    ax.axvline(theta_star, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(THETA_GRID[0], THETA_GRID[-1])
    ax.set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def _fig_output(d, figdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d["theta"], d["yN_a"], color=COLOR_BOOM, linewidth=LW,
            label="Boom — Phase 2(a): naive Fisher,  $K^s \\neq K^d$")
    ax.plot(d["theta"], d["yN_d"], color=COLOR_BUST, linewidth=LW,
            label="Bust — Phase 2(d): wedge clears $K^s = K^d$")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.text(d["theta_star"], ymin + 0.02 * (ymax - ymin),
            r"$\vartheta^* \approx %.3f$" % d["theta_star"],
            ha="center", va="bottom", fontsize=9, color="gray")
    ax.set_xlabel(r"Money growth rate  $\vartheta$  (= $\pi$ under naive rule)")
    ax.set_ylabel(r"Final-good output  $y_N$")
    ax.set_title("Same monetary expansion, opposite output response")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "boom_bust_output.pdf"))
    plt.close(fig)


def _fig_rates(d, figdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d["theta"], d["r_a"], color=COLOR_BOOM, linewidth=LW,
            label=r"$r$  (Boom 2a; = Fisher rate)")
    ax.plot(d["theta"], d["rF_d"], color=COLOR_BUST, linewidth=LW,
            label=r"$r_F$  (Bust 2d; firm rate)")
    ax.plot(d["theta"], d["rH_d"], color=COLOR_BUST, linewidth=1.2, linestyle="--",
            label=r"$r_H$  (Bust 2d; household Fisher rate)")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"Money growth rate  $\vartheta$")
    ax.set_ylabel("Interest rate")
    ax.set_title("Production-side interest rate diverges across closures")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "boom_bust_rates.pdf"))
    plt.close(fig)


def _fig_chains_wage(d, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(d["theta"], d["t2_a"], color=COLOR_BOOM, linewidth=LW, label="Boom (2a)")
    ax.plot(d["theta"], d["t2_d"], color=COLOR_BUST, linewidth=LW, label="Bust (2d)")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"$t_n^*$  ($n \geq 2$)")
    ax.set_title("Optimal stage duration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(d["theta"], d["w_a"], color=COLOR_BOOM, linewidth=LW, label="Boom (2a)")
    ax.plot(d["theta"], d["w_d"], color=COLOR_BUST, linewidth=LW, label="Bust (2d)")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"Real wage  $w$")
    ax.set_title("Real wage")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "boom_bust_chains_w.pdf"))
    plt.close(fig)


def _fig_capital_market(d, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(d["theta"], d["Kd_a"], color=COLOR_BOOM, linewidth=LW,
            label=r"$K^d$  (Phase 2a)")
    ax.plot(d["theta"], d["Ks_a"], color=COLOR_BOOM, linewidth=1.5, linestyle="--",
            label=r"$K^s$  (Phase 2a)")
    ax.plot(d["theta"], d["Kd_d"], color=COLOR_BUST, linewidth=LW,
            label=r"$K^d = K^s$  (Phase 2d)")
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"Capital  (log scale)")
    ax.set_title("Capital demand and supply")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(d["theta"], d["phi_d"], color=COLOR_BUST, linewidth=LW)
    ax.fill_between(d["theta"], 0, d["phi_d"], where=(d["phi_d"] > 0),
                    alpha=0.25, color=COLOR_BUST,
                    label=r"Financing pressure  $\varphi > 0$")
    ax.fill_between(d["theta"], d["phi_d"], 0, where=(d["phi_d"] < 0),
                    alpha=0.25, color=COLOR_BOOM,
                    label=r"Saving glut  $\varphi < 0$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(d["theta_star"], color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(THETA_GRID[0], THETA_GRID[-1])
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"Wedge  $\varphi = r_F - r_H$")
    ax.set_title("Phase 2(d) wedge")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "capital_market.pdf"))
    plt.close(fig)


def _fig_summary(d, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(d["theta"], d["yN_a"], color=COLOR_BOOM, linewidth=LW, label="Boom (2a)")
    ax.plot(d["theta"], d["yN_d"], color=COLOR_BUST, linewidth=LW, label="Bust (2d)")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.set_ylabel(r"$y_N$")
    ax.set_title("Output")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(d["theta"], d["r_a"], color=COLOR_BOOM, linewidth=LW, label=r"$r$ (2a)")
    ax.plot(d["theta"], d["rF_d"], color=COLOR_BUST, linewidth=LW, label=r"$r_F$ (2d)")
    ax.plot(d["theta"], d["rH_d"], color=COLOR_BUST, linewidth=1.2, linestyle="--",
            label=r"$r_H$ (2d)")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Interest rate")
    ax.set_title("Interest rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(d["theta"], d["t2_a"], color=COLOR_BOOM, linewidth=LW, label="Boom (2a)")
    ax.plot(d["theta"], d["t2_d"], color=COLOR_BUST, linewidth=LW, label="Bust (2d)")
    ymin, ymax = ax.get_ylim()
    _shade_regimes(ax, d["theta_star"], ymin, ymax)
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"$t_n^*$  ($n \geq 2$)")
    ax.set_title("Stage duration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(d["theta"], d["phi_d"], color=COLOR_BUST, linewidth=LW)
    ax.fill_between(d["theta"], 0, d["phi_d"], where=(d["phi_d"] > 0),
                    alpha=0.25, color=COLOR_BUST,
                    label=r"$\varphi > 0$")
    ax.fill_between(d["theta"], d["phi_d"], 0, where=(d["phi_d"] < 0),
                    alpha=0.25, color=COLOR_BOOM,
                    label=r"$\varphi < 0$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(d["theta_star"], color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(THETA_GRID[0], THETA_GRID[-1])
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"$\varphi = r_F - r_H$")
    ax.set_title("Wedge (Phase 2d)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        r"Boom (Phase 2a) vs Bust (Phase 2d): "
        r"comparative statics over $\vartheta$  ($\rho=0.05$, $\chi=0.5$)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(figdir, "boom_bust_summary.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    params = _baseline_params()
    d = _run_sweeps(params)
    figdir = os.path.join(_project_root(), "Figures")
    os.makedirs(figdir, exist_ok=True)

    _fig_output(d, figdir)
    _fig_rates(d, figdir)
    _fig_chains_wage(d, figdir)
    _fig_capital_market(d, figdir)
    _fig_summary(d, figdir)

    print(f"theta_star (wedge crosses zero): {d['theta_star']:.4f}")
    print(f"Saved 5 PDF figures to {figdir}/")
    for name in [
        "boom_bust_output.pdf",
        "boom_bust_rates.pdf",
        "boom_bust_chains_w.pdf",
        "capital_market.pdf",
        "boom_bust_summary.pdf",
    ]:
        path = os.path.join(figdir, name)
        size = os.path.getsize(path)
        print(f"  {name:30s}  {size:>8d} bytes")


if __name__ == "__main__":
    main()
