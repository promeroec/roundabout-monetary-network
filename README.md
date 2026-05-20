# Roundabout-Production Network with Monetary Frictions

Replication code for the Handbook chapter

> Pedro Romero, *A Roundabout-Production DSGE Model with Monetary Frictions and Sectoral Networks*, Universidad San Francisco de Quito.

The chapter develops a closed-economy DSGE model in which:

- production is sequential and "roundabout" in the sense of Antràs (2023), with stage-specific maturation functions $z_n(t_n)$ and an optimal length $t_n^\ast$ pinned down by $\alpha_n\, z_n'(t_n^\ast)/z_n(t_n^\ast) = r$;
- capital is supplied à la Antràs–Caballero (2009, 2010), giving $K^s = wL/(\rho - r)$;
- households have CRRA preferences and a cash-in-advance / liquidity constraint that delivers a steady-state Fisher relation $r = (1-\chi)\rho - \chi\pi$;
- a central bank chooses nominal money growth $\theta$ (alternatively a Taylor rule); the government runs a balanced budget;
- sectoral TFP shocks $\lambda_t = \varrho_\lambda \lambda_{t-1} + \eta_t$ propagate through a forward-looking Leontief inverse.

This repository contains the Python code that produces every numerical result and every figure in the chapter.

## Repository layout

```
roundabout-monetary-network/
├── README.md
├── LICENSE                MIT
├── CITATION.cff           machine-readable citation metadata
├── requirements.txt       Python dependencies
├── scripts/               all numerical code
└── Figures/               PDF output (versioned for convenience)
```

## File map: scripts → chapter sections

Each script is a standalone entry point (`python scripts/<name>.py`) and writes its outputs into `Figures/`.

| Script | Produces | Chapter section |
| --- | --- | --- |
| `scripts/steady_state.py` | Closed-form steady state under both household closures; equilibrium residual table | §6.1 Implementation, §6.2 Equilibrium residuals |
| `scripts/rho_sweep.py` | Patience ($\rho$) comparative statics; verifies Proposition `rho_scalings` | §6.3 Closed-form scalings |
| `scripts/plots.py` | Boom-vs-bust comparison across the monetary stance $\theta$ | §3 Steady-state comparative statics, §6.4 Wedge decomposition |
| `scripts/dynamics.py` | Sectoral TFP shocks; forward-looking Leontief inverse; IRFs and variance decompositions | §4 Sectoral TFP shocks and the production network |
| `scripts/lag_plots.py` | Maturation-lag IRFs (lagged vs. instantaneous responses) | §4.4 Maturation lag |
| `scripts/network.py` | Multi-firm block-IO network; firm-level Domar weights; granularity index | §5 Multi-firm extension and granularity |
| `scripts/network_viz.py` | Network visualisations across the three monetary regimes | §5 Multi-firm extension and granularity |
| `scripts/multifirm_steady_state.py` | Firm-level steady-state anchor for the multi-firm chain | §5.1 Influence and Domar weights at the firm level |
| `scripts/multifirm_sweep.py` | Firm-level $\theta$-comparative statics across network topologies | §5.2 Granularity index, §5.3 Separability |
| `scripts/multifirm_dynamics.py` | Firm-level TFP shocks; Hulten decomposition; recovers $r=\rho$ benchmark | §6.5 Stochastic dynamics |
| `scripts/multifirm_plots.py` | Plots for the multi-firm exercises (IO matrix, Domar distribution, granularity) | §5 Multi-firm extension and granularity |

## Installation

Python 3.10 or newer. Standard scientific stack only — no compiled extensions, no proprietary dependencies.

```bash
git clone https://github.com/promeroec/roundabout-monetary-network.git
cd roundabout-monetary-network
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (pinned in `requirements.txt`):

- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`

## Reproducing the results

From the repository root:

```bash
# Steady state and equilibrium residuals (§6.1–6.2)
python scripts/steady_state.py

# Comparative statics — patience and inflation booms (§3, §6.3)
python scripts/rho_sweep.py
python scripts/plots.py

# Sectoral TFP, network propagation (§4)
python scripts/dynamics.py
python scripts/lag_plots.py

# Multi-firm extension and granularity (§5)
python scripts/network.py
python scripts/multifirm_steady_state.py
python scripts/multifirm_sweep.py
python scripts/multifirm_dynamics.py
python scripts/multifirm_plots.py
python scripts/network_viz.py
```

Each script regenerates the figures it owns into `Figures/`. Running the full set takes a few minutes on a modern laptop; no GPU, no parallelism, no random-seed configuration is required (the calibration is deterministic; stochastic IRFs use seeded draws documented in the scripts themselves).

## Verification

The log-linear case $z_n(t_n) = t_n^{\zeta_n}$ admits closed-form steady-state expressions: $t_n^\ast = \alpha_n \zeta_n / r$, with $r t_n^\ast$ independent of $r$. `scripts/steady_state.py` checks the numerical solver against these analytic identities; `scripts/rho_sweep.py` checks the patience scalings of Proposition `rho_scalings` across a fine grid. Maximum equilibrium residuals are reported in §6.2 of the chapter.

## Citation

If you use this code, please cite both the chapter and this software release. The Zenodo DOI for the archived snapshot is shown on the GitHub repository page; `CITATION.cff` gives machine-readable metadata.

```bibtex
@incollection{romero2026roundabout,
  author    = {Pedro Romero},
  title     = {A Roundabout-Production {DSGE} Model with Monetary Frictions and Sectoral Networks},
  booktitle = {[Handbook title — forthcoming]},
  year      = {2026},
  publisher = {[Publisher]},
}
```

## License

MIT — see `LICENSE`.

## Contact

Pedro Romero, Universidad San Francisco de Quito · promero@usfq.edu.ec
