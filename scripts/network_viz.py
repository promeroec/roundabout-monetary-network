"""
Production-network visualisation across three monetary regimes.

Renders one figure with three stacked panels — steady-state, boom,
bust — sharing the same Pareto out-degree network (gamma = 1.5,
M = 20 firms per stage, N = 10 stages). The macro inputs come from:

  steady-state : Phase 1 perpetual-youth (no monetary friction)
                 r ~ 0.0125, t_n*(n>=2) ~ 16
  boom         : Phase 2(a) at theta = 0.04
                 r = 0.005, t_n*(n>=2) = 40 (chain x3 vs SS)
  bust         : Phase 2(d) at theta = 0.04
                 r_F = 0.015, t_n*(n>=2) = 13 (chain shrinks)

Because the log-linear case is separable (Phase c2), the network
*topology* is identical across regimes. What changes:
  - x-axis: cumulative production time sum_{m<=n} t_m^*, so the
    chain physically lengthens in the boom and contracts in the bust.
  - node area: proportional to firm revenue R_{i,n}.
  - macro numbers in the panel titles.

Outputs Figures/network_three_regimes.pdf.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from network import (
    influence_vector,
    pareto_outdegree,
)
from steady_state import (
    ModelParameters,
    MonetaryParameters,
    solve_steady_state_friction,
    solve_steady_state_monetary,
    solve_steady_state_perpetual_youth,
)


FIG_DIR = Path(__file__).resolve().parents[1] / "Figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params() -> ModelParameters:
    return ModelParameters(
        N=10,
        alpha=np.array([1.0] + [0.67] * 9),
        zeta=np.full(10, 0.3),
        rho=0.05, L=1.0, g_share=0.20,
    )


def _firm_revenue(net, ss):
    """R_{i,n} = v_{i,n} * exp(-r * (T_n - t_n)) * y_N."""
    v = influence_vector(net)
    T = np.cumsum(ss.t[::-1])[::-1]   # T_n = sum_{m>=n} t_m
    R = np.empty_like(v)
    for n in range(net.N):
        sl = net.stage_slice(n)
        R[sl] = v[sl] * np.exp(-ss.r * (T[n] - ss.t[n])) * ss.y_N
    return v, R


def _stage_color(n: int, N: int) -> tuple:
    return plt.get_cmap("viridis")(0.10 + 0.78 * n / max(N - 1, 1))


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def _draw_panel(
    ax,
    net,
    ss,
    title: str,
    R_norm: float,
    edge_threshold: float = 0.04,
):
    """Draw the network on `ax`. `R_norm` normalises node area across panels."""
    N = net.N
    M = net.M
    off = net.stage_offsets()

    # Stage delivery times: x_n = sum_{m<=n} t_m^*.
    x_stage = np.cumsum(ss.t)

    # Influence and revenue.
    v, R = _firm_revenue(net, ss)

    # Within-stage layout: rank firms by v (descending) and place on a
    # uniform vertical grid in [-0.5, 0.5].
    coord_x = np.empty_like(v)
    coord_y = np.empty_like(v)
    for n in range(N):
        sl = net.stage_slice(n)
        coord_x[sl] = x_stage[n]
        order = np.argsort(-v[sl])
        if M[n] > 1:
            for k, idx in enumerate(order):
                coord_y[off[n] + idx] = 0.5 - k / (M[n] - 1)
        else:
            coord_y[sl] = 0.0

    # ---------- Edges ----------
    segments = []
    seg_alphas = []
    seg_widths = []
    for n in range(1, N):
        S = net.sourcing[n]                          # M_n x M_{n-1}
        for i in range(M[n]):
            for j in range(M[n - 1]):
                w_ij = float(S[i, j])
                if w_ij < edge_threshold:
                    continue
                src = off[n - 1] + j
                dst = off[n] + i
                segments.append([(coord_x[src], coord_y[src]),
                                 (coord_x[dst], coord_y[dst])])
                seg_alphas.append(min(0.55, 0.10 + 1.5 * w_ij))
                seg_widths.append(0.30 + 1.8 * w_ij)

    if segments:
        seg_colors = np.zeros((len(segments), 4))
        seg_colors[:, :3] = 0.30                     # dark gray
        seg_colors[:, 3] = np.clip(seg_alphas, 0.0, 1.0)
        lc = LineCollection(
            segments, colors=seg_colors, linewidths=seg_widths, zorder=1,
        )
        ax.add_collection(lc)

    # ---------- Nodes ----------
    # Area scales with revenue, normalised by the max revenue across panels
    # so the three panels are visually comparable. Use sqrt for legibility:
    # area is set linearly in scatter `s`, so this gives radius ~ R^{1/2}.
    sizes = 18.0 + 950.0 * (R / R_norm)

    for n in range(N):
        sl = net.stage_slice(n)
        ax.scatter(
            coord_x[sl], coord_y[sl],
            s=sizes[sl], c=[_stage_color(n, N)] * M[n],
            edgecolors="black", linewidths=0.4, zorder=3,
        )

    # Stage labels along the bottom.
    for n in range(N):
        ax.text(
            x_stage[n], -0.66, f"$n={n + 1}$",
            fontsize=8, ha="center", va="top", color="0.25",
        )

    ax.set_ylim(-0.78, 0.78)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, loc="left")
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(out_path: Path | None = None) -> Path:
    params = _params()
    chi = 0.5
    M = 20
    M_vec = np.full(params.N, M)

    # Three regimes -----------------------------------------------------
    ss_steady = solve_steady_state_perpetual_youth(params)

    mon_b = MonetaryParameters(chi=chi, theta=0.04)
    ss_boom = solve_steady_state_monetary(params, mon_b)

    mon_d = MonetaryParameters(chi=chi, theta=0.04)
    ss_bust, r_H_d, phi_d = solve_steady_state_friction(params, mon_d)

    # Single network — separability makes it shared across regimes.
    net = pareto_outdegree(
        M_vec, params.alpha, shape=1.5, common=True, seed=42,
    )

    # Common x-scale (boom is the longest); common revenue scale (boom y_N).
    chain_len_max = float(np.cumsum(ss_boom.t)[-1])
    R_norm = float(
        max(
            _firm_revenue(net, ss_steady)[1].max(),
            _firm_revenue(net, ss_boom)[1].max(),
            _firm_revenue(net, ss_bust)[1].max(),
        )
    )

    # Build figure ------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(14.0, 12.5),
    )
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.99,
                        hspace=0.55)

    def _macro_title(label: str, ss, *, extra: str = "") -> str:
        chain_len = float(np.cumsum(ss.t)[-1])
        head = (
            f"$r$ = {ss.r:.4f}    $w$ = {ss.w:.3f}    "
            f"$y_N$ = {ss.y_N:.3f}    "
            f"$t^*_{{n\\geq 2}}$ = {ss.t[1]:.2f}    "
            f"chain length = {chain_len:.0f}"
        )
        return f"{label}\n{head}" + (f"    {extra}" if extra else "")

    panels = [
        (axes[0], ss_steady, _macro_title(
            "Steady state — perpetual-youth, no monetary friction",
            ss_steady,
        )),
        (axes[1], ss_boom, _macro_title(
            "Boom — Phase 2(a) at $\\vartheta = 0.04$ "
            "(naive Fisher, no asset-market clearing)",
            ss_boom,
        )),
        (axes[2], ss_bust, _macro_title(
            "Bust — Phase 2(d) at $\\vartheta = 0.04$ "
            "(wedge $\\varphi$ enforces $K^d = K^s$)",
            ss_bust,
            extra=(
                f"$r_H$ = {r_H_d:.4f}    "
                f"$\\varphi$ = {phi_d:+.4f}"
            ),
        )),
    ]

    for ax, ss, title in panels:
        _draw_panel(ax, net, ss, title, R_norm=R_norm)
        ax.set_xlim(-0.02 * chain_len_max, 1.02 * chain_len_max)
    axes[-1].set_xlabel(
        "cumulative production time $\\sum_{m \\leq n} t_m^*$",
        fontsize=10,
    )

    # Stage-colour legend strip beneath the bottom panel.
    N = params.N
    color_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=_stage_color(n, N),
               markeredgecolor="black", markersize=7,
               label=f"$n={n + 1}$")
        for n in range(N)
    ]
    size_levels = [0.10, 0.40, 0.90]   # fraction of R_norm
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="0.7", markeredgecolor="black",
               markersize=np.sqrt(18.0 + 950.0 * f),
               label=f"$R/R_{{\\max}}$ = {f:.2f}")
        for f in size_levels
    ]
    leg1 = fig.legend(
        handles=color_handles, loc="lower center",
        bbox_to_anchor=(0.30, 0.005), fontsize=8,
        title="stage colour", title_fontsize=9, frameon=False, ncol=10,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=size_handles, loc="lower center",
        bbox_to_anchor=(0.78, 0.005), fontsize=8,
        title="node area $\\propto R_{i,n}$",
        title_fontsize=9, frameon=False, ncol=3,
    )

    fig.suptitle(
        "Production network across monetary regimes "
        "— Pareto $\\gamma = 1.5$, $N = 10$, $M = 20$/stage "
        "(topology regime-invariant; x-axis = $\\sum_{m \\leq n} t_m^*$; "
        "node area $\\propto R_{i,n}$)",
        fontsize=11.5,
    )

    if out_path is None:
        out_path = FIG_DIR / "network_three_regimes.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    out = make_figure()
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
