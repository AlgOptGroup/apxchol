# Latest benchmarks

Comparative benchmarks for Laplacian / SDDM linear-system solvers, regenerated
from a fresh local run. Charts in `figures/`, raw data in `results.csv`, summary
in `summary.md`.

## Protocol

- **Tolerance**: true relative residual `‖b − L x‖ / ‖b‖ ≤ 1e-8` against the
  **original** operator (no preconditioned-norm shortcuts, no scoring against a
  perturbed system). The runner accepts a cell as `complete` at `rel_res ≤ 10·tol`
  (1e-7), but every `complete` cell in the committed store is in fact ≤ 1e-8
  (largest 9.996e-9).
- **Reps**: 3, median — except the **CMG** cells, which are single-shot (`repeat=1`).
- **Threads**: 16 physical cores (the `t16` figures), pinned, run without contention.
- **Machine**: AMD Ryzen 9 7945HX (16C/32T, Zen 4), 128 GB RAM, NVIDIA RTX 4090 Laptop
  GPU (16 GB), Linux, GCC. Boost-on, so the same config re-runs within about ±20–30%;
  read ratios well above that band, not individual cells.
- **De-singularization (the fairness model)**: grid/SuiteSparse Laplacians are
  **singular** (a constant null vector per connected component). Under the **current**
  protocol each solver removes that null space the way its own machinery prefers and is
  scored on the *true* residual against the **unmodified singular `L`** — so the
  multigrid solvers no longer floor at ~1e-4 from a pin-vs-score mismatch:
  - **apxchol** — native rank-aware solve (mean-centring; no operator change).
  - **BoomerAMG** — symmetric **Dirichlet pin**, one *provably-safe*
    (DFS-tree-leaf, never a cut vertex) node per connected component → SPD, solver
    run with its default config.
  - **AMGCL** — symmetric **Dirichlet pin** (one provably-safe node per connected
    component → SPD → AMGCL defaults, incl. the *direct* coarse solve), same as the
    other algebraic multigrids.
  - **ParAC** — cannot take a singular operator either, but is **no longer `ε·I`
    regularized**: its graph driver reads the same **per-component Dirichlet pin** as the
    multigrids and its physics driver the literal per-component split, both on a shared
    pin-zeroed RHS, so its residual is scored on `‖b − L x‖` against the **original
    singular `L`** like every in-house solver.
  - **CMG** — the only solver still on the old `ε·I` regularization
    (`A = L + ε I`, `ε = 1e-6·mean|diag|`), scored on `L + εI`. Read its bars as
    "same matrix, lightly regularized".
- **Which protocol each cell was measured under (mixed).** The committed store holds
  1048 cells over two protocols, each cell carrying its own provenance note. The pin
  protocol above is current and covers essentially the whole GPU axis (237 of 249 GPU
  cells) plus part of the CPU axis. The majority — 697 cells, almost
  all CPU (260 grids, 104 IPM, 321 SuiteSparse) — predate it and come from the earlier
  **unified regularization**, where *every* solver ran on the same `L + ε·I` operator and
  was scored on it. That protocol is internally consistent; re-running solvers under both
  moved iteration counts by only ~10% at unchanged setup cost, i.e. inside the session
  variance above. See `../README.md` for the full breakdown.

## Solvers

| Solver | Type | Source |
|---|---|---|
| **apxchol** | Our approximate-Cholesky preconditioner + PCG (C++, 16T); the headline series picks the best IS-selector (bg/luby/root/bk) per matrix | this repo |
| **RCHOL / pRCHOL** | Randomized Cholesky + PCG (serial / parallel factor) | [ut-padas/rchol](https://github.com/ut-padas/rchol) |
| **BoomerAMG** | Classical algebraic multigrid + PCG (C, 16T) | [Hypre](https://github.com/hypre-space/hypre) |
| **AMGCL** | Smoothed-aggregation AMG + PCG (C++, 16T); Dirichlet-pin de-singularization, default config | [ddemidov/amgcl](https://github.com/ddemidov/amgcl) |
| **CMG (MATLAB)** † | Combinatorial multigrid + PCG (canonical Koutis MEX, MATLAB R2026a in the `matlab-deps` container; serial) | [ikoutis/cmg-solver](https://github.com/ikoutis/cmg-solver) |
| **ParAC** | Parallel randomized Cholesky + PCG (C++/MKL, 16T, AMD-reordered) | [Tianyu-Liang/Parallel-Randomized-Cholesky](https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky) |
| **AC / AC2** † | ApproxChol — the Julia *reference* implementation of our method (serial); AC2 = `(split=2, merge=2)` oversampling | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) |

## Caveats (read before comparing)

- **† Cross-language wall-times are not speed baselines.** **CMG** runs as the
  canonical Koutis MEX under **MATLAB R2026a** (in the `matlab-deps` container), serial,
  on the full matrix set; **AC/AC2** are serial Julia. Their MATLAB-`pcg` / Julia
  wall-clock is not comparable to the 16-thread C++ solvers — their **iteration count**
  is the meaningful (algorithmic-quality) signal.
- **ParAC** requires an AMD-reordered input (fill-reducing); its reorder time is
  included in its setup. It reads the per-component Dirichlet pin (graph driver) or the
  per-component split (physics driver), **not** `ε·I`, and its convergence is measured
  against the original singular `L` to the same true 1e-8.
- **CHOLMOD** (direct supernodal Cholesky) is out of scope — the suite compares
  preconditioned iterative methods.
- Grids here are **κ=100 weighted** (anisotropic), not uniform Poisson.

## Headline finding

On structured grids the **algebraic multigrids (AMGCL, then BoomerAMG) lead** the
comparison; the Cholesky-type solvers (apxchol, RCHOL, ParAC) cluster behind, with
apxchol competitive among them. On **IPM** (SDDM) matrices **BoomerAMG leads** and apxchol is
within ~30%. On the **scale-free / social graphs** (com-Amazon, coAuthorsDBLP)
**apxchol is fastest** — the regime where the Cholesky-type structure pays off.

> **Note (provenance):** the charts and `results.csv` mix the two protocols described
> above — the GPU axis is essentially all pin-protocol, the bulk of the CPU cells still
> carry the earlier unified-`reg_rel` provenance. The 5 large **social giants**
> (as-Skitter, coPapersDBLP, com-LiveJournal, com-Orkut, com-Youtube) are measured on
> **both** axes (9 GPU cells each, pin protocol); what is outstanding for them is the
> **CPU** side — 117 of their 139 CPU cells predate the pin protocol and have not been
> re-run, so their CPU panels may shift.
