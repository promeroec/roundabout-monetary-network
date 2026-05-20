"""
Phase (c) — Multi-firm chain network with input-output matrix
a^{(n-1,n)}_{ij}.

Network specification
---------------------
- Stages n = 1, ..., N with M_n firms each.
- Sourcing is restricted to stage n-1 (multi-firm chain).
- For each downstream firm i at stage n > 1, the cost shares
  a^{(n-1,n)}_{ij} satisfy  sum_j a^{(n-1,n)}_{ij} = 1.
- The block IO matrix A is laid out with firms stacked stage-by-stage:
  rows/cols = (stage 1 firms, stage 2 firms, ..., stage N firms). The
  off-diagonal block (n, n-1) carries entries (1 - alpha_n)·a^{(n-1,n)}_{ij};
  all other blocks are zero. A is therefore strictly block lower-triangular
  in stage-index order.

Influence vector
----------------
Final-good aggregator: log y_N = (1/M_N) * sum_{i in I_N} log y_{i,N}
(Cobb-Douglas with uniform weights). The associated final-demand vector
e_F has 1/M_N at every stage-N position and 0 elsewhere; the influence
vector solves
    (I - A^T) v = e_F,
so v_{i,n} is firm (i,n)'s gross-output (revenue) share in y_N. Under
the Hicks-neutral TFP specification y_{i,n} = lambda_{i,n} * f(...),
the firm-level Domar weight is d_{i,n} = v_{i,n} (Hulten's theorem),
and they sum to sum_n beta_n > 1 because intermediate inputs are
double-counted in gross output.

Chain limit
-----------
Under uniform sourcing (a^{(n-1,n)}_{ij} = 1/M_{n-1} for all i,j), symmetry
gives V_n := sum_i v_{i,n} = beta_n where beta_n = prod_{m>n}(1 - alpha_m),
matching the original (M=1) chain. Stage-aggregate Domar weights
sum_i d_{i,n} = beta_n exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FirmNetwork:
    """Multi-firm chain network with sourcing restricted to stage n-1."""

    M: np.ndarray                       # firms per stage, shape (N,)
    alpha: np.ndarray                   # stage-level alpha_n, shape (N,)
    sourcing: List[Optional[np.ndarray]] = field(default_factory=list)
    # sourcing[n] is the M_n x M_{n-1} matrix a^{(n-1,n)}; sourcing[0] = None.
    name: str = ""

    @property
    def N(self) -> int:
        return len(self.M)

    @property
    def total_firms(self) -> int:
        return int(self.M.sum())

    def stage_offsets(self) -> np.ndarray:
        """offsets[n] = first row/col index of stage n in the stacked layout."""
        return np.concatenate(([0], np.cumsum(self.M))).astype(int)

    def stage_slice(self, n: int) -> slice:
        off = self.stage_offsets()
        return slice(int(off[n]), int(off[n + 1]))

    def validate(self, tol: float = 1e-12) -> None:
        if self.alpha.shape != (self.N,):
            raise ValueError("alpha must have length N")
        if not np.isclose(self.alpha[0], 1.0):
            raise ValueError("alpha_1 must equal 1 (stage 1 has no intermediates)")
        if len(self.sourcing) != self.N:
            raise ValueError("sourcing list must have length N")
        if self.sourcing[0] is not None:
            raise ValueError("sourcing[0] must be None (stage 1 has no inputs)")
        for n in range(1, self.N):
            S = self.sourcing[n]
            if S is None:
                raise ValueError(f"sourcing[{n}] is None")
            if S.shape != (self.M[n], self.M[n - 1]):
                raise ValueError(
                    f"sourcing[{n}] has shape {S.shape}, expected "
                    f"({self.M[n]}, {self.M[n - 1]})"
                )
            row_sums = S.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=tol):
                raise ValueError(
                    f"sourcing[{n}] row sums deviate from 1 (max abs dev "
                    f"{np.max(np.abs(row_sums - 1.0)):.3e})"
                )
            if (S < 0).any():
                raise ValueError(f"sourcing[{n}] contains negative entries")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def chain_network(M, alpha) -> FirmNetwork:
    """Uniform sourcing: a^{(n-1,n)}_{ij} = 1/M_{n-1}.  Recovers the chain."""
    M = np.asarray(M, dtype=int)
    alpha = np.asarray(alpha, dtype=float)
    N = len(M)
    sourcing: List[Optional[np.ndarray]] = [None]
    for n in range(1, N):
        sourcing.append(np.full((M[n], M[n - 1]), 1.0 / M[n - 1]))
    net = FirmNetwork(M=M, alpha=alpha, sourcing=sourcing, name="chain")
    net.validate()
    return net


def multistage_random(M, alpha, k: int, seed: Optional[int] = None) -> FirmNetwork:
    """
    Each downstream firm picks k suppliers uniformly at random from
    stage n-1 (without replacement, equal weight 1/k each). Recovers
    the chain when k = M[n-1].
    """
    M = np.asarray(M, dtype=int)
    alpha = np.asarray(alpha, dtype=float)
    N = len(M)
    rng = np.random.default_rng(seed)
    sourcing: List[Optional[np.ndarray]] = [None]
    for n in range(1, N):
        if k > M[n - 1]:
            raise ValueError(
                f"k={k} exceeds upstream firm count M[{n-1}]={M[n-1]}"
            )
        S = np.zeros((M[n], M[n - 1]))
        for i in range(M[n]):
            picks = rng.choice(M[n - 1], size=k, replace=False)
            S[i, picks] = 1.0 / k
        sourcing.append(S)
    net = FirmNetwork(M=M, alpha=alpha, sourcing=sourcing,
                      name=f"random_k={k}")
    net.validate()
    return net


def pareto_outdegree(
    M,
    alpha,
    shape: float = 1.5,
    common: bool = True,
    seed: Optional[int] = None,
) -> FirmNetwork:
    """
    Pareto-distributed supplier popularity.

    For each upstream stage n-1, draw weights w_j ~ Pareto(shape) (Lomax
    tail with index `shape`, normalized to sum to 1). Each downstream
    firm at stage n sources from upstream with weights w_j.

    common=True : all buyers at stage n share the same supplier mix
                  (one popular supplier feeds many buyers — the
                  Acemoglu-Carvalho-Ozdaglar-Tahbaz-Salehi setup).
    common=False: each buyer draws its own iid Pareto weights.

    Pareto with shape <= 2 has infinite variance; under common=True the
    HHI of supplier popularity does not vanish as M grows, so aggregate
    granularity does not vanish either.
    """
    M = np.asarray(M, dtype=int)
    alpha = np.asarray(alpha, dtype=float)
    N = len(M)
    if shape <= 0:
        raise ValueError("Pareto shape must be positive")
    rng = np.random.default_rng(seed)
    sourcing: List[Optional[np.ndarray]] = [None]
    for n in range(1, N):
        if common:
            w = rng.pareto(shape, size=M[n - 1]) + 1.0
            w = w / w.sum()
            S = np.tile(w, (M[n], 1))
        else:
            S = np.zeros((M[n], M[n - 1]))
            for i in range(M[n]):
                w = rng.pareto(shape, size=M[n - 1]) + 1.0
                S[i] = w / w.sum()
        sourcing.append(S)
    net = FirmNetwork(M=M, alpha=alpha, sourcing=sourcing,
                      name=f"pareto_shape={shape}_{'common' if common else 'iid'}")
    net.validate()
    return net


# ---------------------------------------------------------------------------
# Block IO matrix and influence-vector solver
# ---------------------------------------------------------------------------

def assemble_matrix(net: FirmNetwork) -> np.ndarray:
    """Block IO matrix A (cost-share form).  A[(i',n+1),(j,n)] = (1-alpha_{n+1})·a."""
    F = net.total_firms
    off = net.stage_offsets()
    A = np.zeros((F, F))
    for n in range(1, net.N):
        block = (1.0 - net.alpha[n]) * net.sourcing[n]
        A[off[n]:off[n + 1], off[n - 1]:off[n]] = block
    return A


def final_demand_vector(net: FirmNetwork) -> np.ndarray:
    """Uniform final demand on stage-N firms: 1/M_N each."""
    e_F = np.zeros(net.total_firms)
    e_F[net.stage_slice(net.N - 1)] = 1.0 / net.M[net.N - 1]
    return e_F


def influence_vector(
    net: FirmNetwork,
    A: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    v = (I - A^T)^{-1} e_F. Exploits the strictly-block-lower-triangular
    structure of A (sourcing only from stage n-1) to back-substitute
    stage by stage:  v^{(N)} = e_F^{(N)};
                     v^{(n)} = (1-alpha_{n+1}) sourcing[n+1].T @ v^{(n+1)}.
    The dense `A` argument is accepted for API compatibility but not used.
    Cost: O(N * M^2)  vs  O((N*M)^3)  for the full solve.
    """
    del A  # unused; recursion is cheaper than a full linear solve
    off = net.stage_offsets()
    v = np.zeros(net.total_firms)
    # Stage N: uniform final demand 1/M_N.
    v[net.stage_slice(net.N - 1)] = 1.0 / net.M[net.N - 1]
    # Walk upstream: v^{(n)} = (1 - alpha_{n+1}) * sourcing[n+1].T @ v^{(n+1)}.
    for n in range(net.N - 2, -1, -1):
        v_up = (1.0 - net.alpha[n + 1]) * net.sourcing[n + 1].T @ v[
            off[n + 1]:off[n + 2]
        ]
        v[off[n]:off[n + 1]] = v_up
    return v


def firm_domar_weights(
    net: FirmNetwork,
    v: Optional[np.ndarray] = None,
) -> np.ndarray:
    """d_{i,n} = v_{i,n} under the Hicks-neutral TFP specification.
    Equals the firm's gross-output share in y_N (Hulten's theorem).
    Sum equals sum_n beta_n > 1 due to intermediate double-counting."""
    if v is None:
        v = influence_vector(net)
    return v.copy()


def stage_aggregate_domar(net: FirmNetwork, d: np.ndarray) -> np.ndarray:
    """sum_i d_{i,n} per stage. On the chain: equals beta_n."""
    return np.array([d[net.stage_slice(n)].sum() for n in range(net.N)])


def stage_influence_total(net: FirmNetwork, v: np.ndarray) -> np.ndarray:
    """sum_i v_{i,n} per stage. On the chain: equals beta_n."""
    return np.array([v[net.stage_slice(n)].sum() for n in range(net.N)])


def stage_hhi(net: FirmNetwork, x: np.ndarray) -> np.ndarray:
    """
    Within-stage Herfindahl: sum_i (x_{i,n} / X_n)^2 where X_n = sum_i x_{i,n}.
    Equals 1/M_n under uniform shares; approaches 1 under perfect concentration.
    """
    out = np.empty(net.N)
    for n in range(net.N):
        sl = net.stage_slice(n)
        s = x[sl].sum()
        if s <= 0:
            out[n] = np.nan
        else:
            out[n] = float(((x[sl] / s) ** 2).sum())
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _baseline_alpha(N: int = 10) -> np.ndarray:
    a = np.full(N, 0.67)
    a[0] = 1.0
    return a


def main() -> None:
    N, M = 10, 20
    M_vec = np.full(N, M)
    alpha = _baseline_alpha(N)

    nets = [
        chain_network(M_vec, alpha),
        multistage_random(M_vec, alpha, k=4, seed=42),
        pareto_outdegree(M_vec, alpha, shape=1.5, common=True, seed=42),
    ]

    print("=" * 78)
    print(f"Phase (c) — multi-firm chain network, N={N}, M={M}")
    print("=" * 78)
    print()

    # beta_n for the chain reference (same alpha).
    beta = np.empty(N)
    beta[-1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = (1.0 - alpha[n + 1]) * beta[n + 1]

    for net in nets:
        A = assemble_matrix(net)
        v = influence_vector(net, A)
        d = firm_domar_weights(net, v)
        V = stage_influence_total(net, v)
        D = stage_aggregate_domar(net, d)
        H = stage_hhi(net, v)

        print(f"  Network: {net.name}")
        print(f"    total firms    = {net.total_firms}")
        print(f"    A.shape        = {A.shape}")
        print(f"    sum d_{{i,n}}    = {d.sum():.6f}   (target sum_n beta_n = {beta.sum():.6f})")
        print(f"    max|V_n - beta_n|  = {np.max(np.abs(V - beta)):.3e}")
        print(f"    max|D_n - beta_n| = "
              f"{np.max(np.abs(D - beta)):.3e}")
        print(f"    stage HHI of v_{{i,n}}: "
              f"min={H.min():.4f}  max={H.max():.4f}   "
              f"(uniform = 1/M = {1.0 / M:.4f})")
        print()


if __name__ == "__main__":
    main()
