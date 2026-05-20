"""
scripts/rho_sweep.py — Preference-driven (rho) comparative statics
                       under the monetary closure with theta = 0.

Companion to scripts/plots.py (theta-sweep, boom-vs-bust). Sweeps the
household discount rate rho on a fine grid, holding theta = 0 and all
other primitives fixed, and verifies the closed-form scalings of
Proposition rho_scalings under the CIA-Fisher closure  r = (1-chi)*rho:

    r*(rho)      = (1 - chi) * rho,
    t_n*(rho)    = alpha_n * zeta_n / ((1 - chi) * rho),
    w(rho)       proportional to rho ^ (-bar_zeta),
    y_N(rho)     = (1 + Psi) * w(rho) * L,
    K^d(rho)     = Psi * w(rho) * L / ((1 - chi) * rho),

with bar_zeta = sum_n zeta_n * alpha_n * beta_n  (Domar-weighted
average time intensity) and Psi the parametric working-capital wedge.

Two figures are written to Figures/:

  rho_sweep_verification.pdf   2x3 dashboard. Each panel overlays the
                               numerical solver output (solid markers)
                               on the closed-form prediction (dashed
                               line). The agreement is machine-precision
                               in the log-linear case.

  rho_theta_wedge.pdf          (rho, theta) phase diagram for the
                               Wicksellian wedge phi(rho, theta)
                               = chi*(rho + theta) under the CIA-Fisher
                               closure. The Friedman locus theta = -rho
                               is the line where the wedge vanishes;
                               positive phi marks credit-induced
                               (unsustainable) booms, zero phi marks
                               Wicksellian neutrality.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from steady_state import (
    ModelParameters,
    MonetaryParameters,
    beta_weights,
    solve_steady_state_monetary,
    verify,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RHO_GRID = np.linspace(0.010, 0.060, 51)
RHO_BENCH = 0.05
CHI = 0.5
COLOR_SUST = "#2ca02c"      # green — sustainable / voluntary saving
COLOR_NUM  = "#2ca02c"
COLOR_BOOM = "#1f77b4"
COLOR_BUST = "#d62728"
LW = 2.0


def _baseline_params(rho: float) -> ModelParameters:
    return ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=rho,
        L=1.0,
        g_share=0.20,
    )


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Run the sweep and collect arrays
# ---------------------------------------------------------------------------

def _run_rho_sweep(rho_grid: np.ndarray):
    rs    = np.empty_like(rho_grid)
    t2s   = np.empty_like(rho_grid)
    t1s   = np.empty_like(rho_grid)
    ws    = np.empty_like(rho_grid)
    yNs   = np.empty_like(rho_grid)
    Ks    = np.empty_like(rho_grid)
    resid = np.empty_like(rho_grid)
    mon   = MonetaryParameters(chi=CHI, theta=0.0)
    for i, rho in enumerate(rho_grid):
        params = _baseline_params(float(rho))
        ss = solve_steady_state_monetary(params, mon)
        rs[i]  = ss.r
        t1s[i] = ss.t[0]
        t2s[i] = ss.t[1]
        ws[i]  = ss.w
        yNs[i] = ss.y_N
        Ks[i]  = ss.K_d
        r = verify(ss, closure="monetary", mon=mon)
        resid[i] = max(v for k, v in r.items() if k != "passed")
    return dict(rho=rho_grid, r=rs, t1=t1s, t2=t2s, w=ws,
                y_N=yNs, K=Ks, resid=resid)


def _closed_form_predictions(rho_grid: np.ndarray, params0: ModelParameters):
    """Compute closed-form r*, t_n*, w, y_N, K^d at every rho on the grid
    under the monetary closure with theta = 0:  r = (1 - chi) * rho.

    The proportionality constant for w is calibrated to match the numerical
    value at the benchmark rho.
    """
    alpha = params0.alpha
    zeta  = params0.zeta
    beta  = beta_weights(alpha)
    L     = params0.L

    # Parametric working-capital wedge Psi in the log-linear case.
    rT_const = np.cumsum((alpha * zeta)[::-1])[::-1]
    raw   = alpha * beta * np.exp(-rT_const)
    share = raw / raw.sum()
    Psi   = float(np.sum(share * (np.exp(rT_const) - 1.0)))

    # Domar-weighted average time intensity.
    bar_zeta = float(np.sum(zeta * alpha * beta))

    # Closed form for r* under the monetary closure with theta = 0.
    r_cf  = (1.0 - CHI) * rho_grid
    t1_cf = (alpha[0] * zeta[0]) / r_cf
    t2_cf = (alpha[1] * zeta[1]) / r_cf

    # Proportionality constant for w: pin from a single point under the
    # monetary closure at theta = 0.
    params_bench = ModelParameters(
        N=params0.N, alpha=alpha.copy(), zeta=zeta.copy(),
        rho=RHO_BENCH, L=L, g_share=params0.g_share,
    )
    ss_bench  = solve_steady_state_monetary(
        params_bench, MonetaryParameters(chi=CHI, theta=0.0)
    )
    w_const   = ss_bench.w * (RHO_BENCH ** bar_zeta)
    w_cf      = w_const * (rho_grid ** (-bar_zeta))

    yN_cf = (1.0 + Psi) * w_cf * L
    K_cf  = Psi * w_cf * L / r_cf
    return dict(r=r_cf, t1=t1_cf, t2=t2_cf, w=w_cf, y_N=yN_cf, K=K_cf,
                Psi=Psi, bar_zeta=bar_zeta)


# ---------------------------------------------------------------------------
# Figure 1 — six-panel verification
# ---------------------------------------------------------------------------

def _fig_verification(num, cf, figdir, params0):
    rho   = num["rho"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))

    panels = [
        ("r",   r"$r^*$",                  axes[0, 0]),
        ("t2",  r"$t_n^*$  ($n\geq 2$)",   axes[0, 1]),
        ("w",   r"$w$",                    axes[0, 2]),
        ("y_N", r"$y_N$",                  axes[1, 0]),
        ("K",   r"$K^d$",                  axes[1, 1]),
    ]
    for key, ylabel, ax in panels:
        ax.plot(rho, cf[key], color="k", linestyle="--", linewidth=1.2,
                label="closed form")
        ax.plot(rho, num[key], color=COLOR_SUST, linewidth=0,
                marker="o", markersize=3.5, alpha=0.85,
                label="numerical")
        ax.axvline(RHO_BENCH, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if key in ("t2", "K"):
            ax.set_yscale("log")
        if key == "r":
            ax.legend(loc="best", fontsize=9)

    # Sixth panel: residual diagnostic.
    ax = axes[1, 2]
    ax.semilogy(rho, np.maximum(num["resid"], 1e-18),
                color=COLOR_SUST, linewidth=1.2, marker="o", markersize=3)
    ax.axhline(1e-15, color="k", linestyle="--", linewidth=0.8,
               label=r"$10^{-15}$")
    ax.axvline(RHO_BENCH, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"Max equilibrium residual")
    ax.set_title("Solver residual (log scale)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    bar_zeta = cf["bar_zeta"]
    Psi      = cf["Psi"]
    fig.suptitle(
        r"Patience sweep under the monetary closure at $\theta = 0$  "
        r"($r = (1-\chi)\rho$, $\chi = %.2f$).  "
        r"$\Psi = %.4f$,  $\bar\zeta = %.3f$.  "
        r"Closed-form (dashed) vs numerical (markers)."
        % (CHI, Psi, bar_zeta),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(figdir, "rho_sweep_verification.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — (rho, theta) phase diagram of the Wicksellian wedge
# ---------------------------------------------------------------------------

def _fig_wedge_plane(num, cf, figdir, params0):
    """Wicksellian wedge phi(rho, theta) = chi*(rho + theta) under the
    CIA-Fisher closure. The Friedman locus theta = -rho is the line
    where phi vanishes."""
    rho_axis   = np.linspace(0.012, 0.058, 90)
    theta_axis = np.linspace(-0.060, 0.045, 90)
    R, T       = np.meshgrid(rho_axis, theta_axis, indexing="ij")
    chi        = CHI

    # Closed-form CIA-Fisher wedge.
    phi = chi * (R + T)

    # Cross-check markers — solve the monetary block and recover
    # phi = rho - r at sample points inside the admissible domain.
    sample_rho   = np.array([0.020, 0.030, 0.040, 0.050])
    sample_theta = np.array([-0.040, -0.020, 0.000, 0.020, 0.040])
    sample_phi   = np.full((len(sample_rho), len(sample_theta)), np.nan)
    for i, rho_i in enumerate(sample_rho):
        params = _baseline_params(float(rho_i))
        for j, th_j in enumerate(sample_theta):
            mon = MonetaryParameters(chi=chi, theta=float(th_j))
            try:
                ss = solve_steady_state_monetary(params, mon)
                sample_phi[i, j] = rho_i - ss.r
            except (ValueError, RuntimeError):
                pass

    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    levels = np.linspace(-0.030, 0.030, 13)
    cs = ax.contourf(R, T, phi, levels=levels, cmap="RdBu_r", extend="both")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label(r"$\varphi(\rho,\theta) = \chi(\rho + \theta)$")

    # Friedman locus: theta = -rho.
    ax.plot(rho_axis, -rho_axis, color="k", linewidth=2,
            label=r"Friedman locus  $\theta = -\rho$")
    # Benchmark point.
    ax.plot([RHO_BENCH], [0.0], "k*", markersize=12, label="Benchmark")

    # Cross-check markers — solver-recovered wedge at sample points.
    for i, rho_i in enumerate(sample_rho):
        for j, th_j in enumerate(sample_theta):
            if np.isfinite(sample_phi[i, j]):
                ax.plot(rho_i, th_j, "ko", markersize=4, alpha=0.65)

    ax.set_xlabel(r"$\rho$  (household discount rate)")
    ax.set_ylabel(r"$\theta$  (nominal money growth)")
    ax.set_title(
        r"Wicksellian wedge $\varphi(\rho,\theta) = \chi(\rho + \theta)$"
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(rho_axis.min(), rho_axis.max())
    ax.set_ylim(theta_axis.min(), theta_axis.max())
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "rho_theta_wedge.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Elasticity diagnostic (printed table)
# ---------------------------------------------------------------------------

def _elasticity_table(num, cf):
    """Numerical log-log slopes from finite differences vs analytic
    elasticities. Should match to ~1e-3."""
    rho = num["rho"]
    log_rho = np.log(rho)

    def slope(y):
        ly = np.log(y)
        return float(np.polyfit(log_rho, ly, 1)[0])

    print()
    print("Elasticities  d log(quantity) / d log(rho):")
    print("                  numerical    analytic")
    print(f"  r*           {slope(num['r']):+.5f}    {+1.0:+.5f}")
    print(f"  t_n*(n>=2)   {slope(num['t2']):+.5f}    {-1.0:+.5f}")
    print(f"  w            {slope(num['w']):+.5f}    {-cf['bar_zeta']:+.5f}")
    print(f"  y_N          {slope(num['y_N']):+.5f}    {-cf['bar_zeta']:+.5f}")
    print(f"  K*           {slope(num['K']):+.5f}    "
          f"{-(1 + cf['bar_zeta']):+.5f}")
    print(f"  max equil. residual across the sweep: "
          f"{num['resid'].max():.3e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    params0 = _baseline_params(RHO_BENCH)
    num     = _run_rho_sweep(RHO_GRID)
    cf      = _closed_form_predictions(RHO_GRID, params0)

    figdir = os.path.join(_project_root(), "Figures")
    os.makedirs(figdir, exist_ok=True)

    _fig_verification(num, cf, figdir, params0)
    _fig_wedge_plane(num, cf, figdir, params0)

    _elasticity_table(num, cf)

    print()
    print(f"Closed-form Psi     = {cf['Psi']:.6f}")
    print(f"Closed-form bar_zeta= {cf['bar_zeta']:.6f}")
    print(f"Friedman locus theta = -rho;  at rho=0.05 this is theta = -0.050")
    print(f"CIA-Fisher wedge phi(rho, theta) = chi*(rho+theta)")
    print()
    print(f"Saved 2 PDF figures to {figdir}/")
    for name in ["rho_sweep_verification.pdf", "rho_theta_wedge.pdf"]:
        path = os.path.join(figdir, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {name:32s}  {size:>8d} bytes")


if __name__ == "__main__":
    main()
