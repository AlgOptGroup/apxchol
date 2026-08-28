# Benchmarks

Comparative benchmarks for Laplacian / SDDM linear-system solvers, comparing our
approximate-Cholesky preconditioner against third-party direct, randomized, and
multigrid solvers.

The benchmark's defining feature is a **fair comparison**: every solver attacks
the *same* operator and is judged by the *same* true residual (see
[Protocol](#protocol)). The latest results live in [`latest/`](latest/) and are
committed so the charts render on GitHub — see [`latest/README.md`](latest/README.md)
for methodology and [`latest/summary.md`](latest/summary.md) for the tables.
CPU scaling and apxchol-only historical comparisons from CSCS Daint are kept
separately in [`daint/`](daint/); they are not mixed into the laptop competitor
bars.

## Solvers

| Solver | Type | Threads | Source |
|---|---|---|---|
| **apxchol** | Approximate Cholesky + PCG. `apxchol/bg` is the declared headline default; selectors are compared in a separate compact ablation | 16 (C++) | this repo (`src/`, `include/apxchol/`) |
| ↳ IS selectors | bg=block-greedy, greedy=fixed-priority greedy MIS, bk=Baumann-Kyng. Historical `luby` cells are read as `greedy`; redundant `root` is retired. | 16 (C++) | this repo |
| **RCHOL** | Randomized Cholesky + **their own** PCG (`util/pcg.cpp`, MKL ILP64; serial factorization; x86 only) | serial factor | [ut-padas/rchol](https://github.com/ut-padas/rchol) |
| **BoomerAMG** | Classical *algebraic* multigrid + PCG | 16 (C, OpenMP) | [Hypre](https://github.com/hypre-space/hypre) |
| **AMGCL** | Smoothed-aggregation *algebraic* multigrid + PCG; Dirichlet-pin de-singularization, default config | 16 (C++) | [ddemidov/amgcl](https://github.com/ddemidov/amgcl) |
| **ParAC** | Parallel randomized Cholesky + PCG (AMD-reordered, MKL) | 16 (C++) | [Tianyu-Liang/Parallel-Randomized-Cholesky](https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky) |
| **CMG (MATLAB)** † | *Combinatorial* multigrid + PCG | serial (MATLAB) | [ikoutis/cmg-solver](https://github.com/ikoutis/cmg-solver) |
| **AC / AC2** | ApproxChol — Julia *reference* of our method | serial (Julia) | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) |

BoomerAMG (algebraic multigrid) and CMG (combinatorial multigrid, the closest
cousin to our approximate-Cholesky — both from the Spielman–Teng / support-theory
line) are the two multigrids kept.

**Status notes:**
- **AMGCL** (smoothed-aggregation AMG) is de-singularized with the same symmetric
  **Dirichlet pin** as the other algebraic multigrids (→ SPD → AMGCL defaults, *direct*
  coarse solve) and run with its default config.
- **†** `CMG` runs as the **canonical MATLAB CMG** (Koutis `cmg-solver`, MEX
  recompiled for R2026a, in the `matlab-deps` container). Its MATLAB-`pcg`
  wall-time isn't cross-language-comparable to the C++ solvers, so its
  **iteration count** is the comparable signal (use it, not the wall-time).
- **CHOLMOD** (direct supernodal Cholesky) is out of scope — the suite compares
  preconditioned iterative methods. `CG/LDLT [Eigen]` and the apxchol IS/storage
  ablations (`bk/greedy`, alternative storage) exist in the binary but are not in the headline set.

## Whose solve loop produced each number

**The RHS, the de-singularization and the final residual are the harness's, identical for
every solver in a cell. The iteration is the vendor's.** We do not re-implement a solver's
Krylov loop and report the result as theirs. Per solver:

| Solver | Setup we time | The iteration that produced `iters`/`solve_s` | Ours in the loop |
|---|---|---|---|
| **apxchol** | our factorization | our PCG | everything — it is our method |
| **RCHOL / pRCHOL** | their `rchol(A,G)` / `rchol(A,G,perm,nt)` | **their `util/pcg.cpp`** (MKL `mkl_sparse_d_mv` / `d_trsv`), the loop their `ex_laplace.cpp` drives | nothing. AMD reorder (`rchol`) is ours, on their own README's recommendation; pRCHOL uses **their** `reorder()` with the ordering `rchol` returns, as in their `ex_laplace_parallel.cpp`. **x86 only** — see the note below |
| **BoomerAMG** | `HYPRE_ParCSRPCGSetup` (which builds the AMG hierarchy — exactly once) | `HYPRE_ParCSRPCGSolve` with `HYPRE_BoomerAMGSolve` as preconditioner — hypre's iteration, hypre's count | the Dirichlet pin, the IJ marshalling, the residual re-grade |
| **AMGCL** | their `make_solver<amg<SA,spai0>, solver::cg>` ctor | their `solve(rhs,sol)` | the pin and the Eigen→CRS marshalling. **We chose `cg` over their runtime default `bicgstab`** — the operator is SPD, their own docs recommend it there, and it keeps the Krylov method uniform across the table |
| **ParAC** | their `experiment/driver` binary, their documented CLI | their driver's own loop, their printed timings/iterations | nothing in the iteration. Its input is built by **their** `write_graph.jl`, so in graph mode the ParAC row is same-operator / **different RHS** |
| **CMG** | `cmg_sdd(A)` | MATLAB's built-in `pcg` — CMG's own documented usage (`matlab/cmg/README.txt`) | the RHS build and the re-grade. Its MATLAB wall-time is not cross-language comparable; **the iteration count is the signal** |
| **AC / AC2** | `approxchol_lap` / `approxchol_sddm` closure | **their** `pcg`, including their `stag_test` | the `tol_eff` rescale for the sddm augmented basis. `AC2` = their `approxchol_lap` with `ApproxCholParams(:deg,5,2,2)`, **not** their separate `approxchol_lap2` |
| `cg` / `icc` / `ldlt` / `cholmod` | — | Eigen's / SuiteSparse's own | nothing |

**Timing boundary.** Parsing/assembling the common operator and benchmark-only
validation/cleanup are excluded. `setup_s` includes every solver-required input
conversion, grounding, reorder, upload, hierarchy/factor construction and analysis;
`solve_s` includes per-RHS workspace/upload, the solver's iteration, and returning
the solution to the caller (including GPU D2H and RCHOL unpermutation). We use a
solver's supplied adapter/setup routine whenever it has one: AMGCL's `zero_copy`
CRS adapter, Hypre's IJ API, RCHOL's own `reorder` and `pcg`, and ParAC's own
`write_graph.jl` plus instrumented native driver intervals. AC/AC2 and CMG expose
no assembled-operator adapter, so the minimal required operator conversion is in
their setup interval. Split-component preparation is timed explicitly; validation
or destructor time is never inferred from a wall-time remainder.

**Documented calibrations and patches — these must stay visible wherever the table is
published** (see [THE GRADING RULE](#grading-rule) for why a calibration, not a looser mark):

| Solver | Their stop test | What we pass / change | Tolerance actually used |
|---|---|---|---|
| ParAC-CPU | **absolute**, `sqrt(dpar[4]) > sqrt(dpar[0])`, on the recurrence residual | `PARAC_REL_TOL` calibrated from one probe run (`_calibrate_rel_tol`); recorded per cell as `parac_rel_tol` | `rel_tol = (τ·r_recur,0/R_0)²`, per matrix, so their **printed true** residual lands under 1e-8 |
| ParAC-GPU | **preconditioned**-residual ratio `√⟨r,M⁻¹r⟩/√⟨r₀,M⁻¹r₀⟩` — optimistic by ~10× | its existing CLI tolerance is calibrated from one probe (`_calibrate_tol_gpu`); the upstream stopping test and recurrence are untouched | calibrated per matrix so its printed true relative residual lands under 1e-8 |
| AC (`sddm` path) | `‖r‖/‖b‖` against the **augmented** RHS `[b; −Σb]` — a different basis | `tol_eff = tol·‖b‖/‖b_aug‖` (`bench_laplacians.jl`) | `tol_eff`, so the residual in *our* basis is 1e-8 |
| AC / AC2 | `‖r‖/nb < tol` **plus `stag_test = 5`** — a stagnation early exit can return a non-converged `x` | nothing: their exit is theirs to keep | 1e-8; a stagnation exit is caught by our re-grade and lands as `not_converged` |
| BoomerAMG | 2-norm **recurrence** residual (`TwoNorm=1`, `StopCrit=0`), on the **Dirichlet-pinned** subsystem — and it is pinned only when the matrix is *declared* singular | nothing. Recurrence drift was tested and ruled out: `HYPRE_PCGSetRecomputeResidual(1)` leaves iterations and residual bit-identical on iter0040 / grid_2000 / com-Amazon, so the knob was not kept. The "pin-vs-score gap" that used to be listed here was **our misclassification of the IPM matrices, now CLOSED — see below** | 1e-8, on hypre's system (pinned only where `--class laplacian`) |
| RCHOL / pRCHOL | `dnrm2(r) > dnrm2(b)·tol` (`pcg.cpp:82`), recurrence | nothing — **semantically identical to ours**, so adopting their loop imports no quirk | 1e-8 |

**CLOSED (2026-08-21): the BoomerAMG "pin-vs-score gap" on the IPM matrices was OUR
misclassification, and no tolerance could ever have fixed it.** This paragraph used to say the
fix was to *calibrate the tolerance handed to hypre for the pin-to-original transfer*, the same
shape as ParAC's calibration. That was **unattainable**, and the reason matters.

`iter0020`, `iter0030` and `iter0040` are **full-rank SDDM**, not singular Laplacians. They
carry a uniform `+1e-6` diagonal regularization, so their row sums never vanish. The benchmark
classified them by the ratio `max|rowsum| / max|diag| < 1e-10` (`is_laplacian_operator`, now
deleted), and that ratio's numerator is pinned at ~1e-6 by the shift while only its
**denominator** moves as the barrier tightens — so it slid under the threshold mid-family:

| matrix | `max‖rowsum‖ / max‖diag‖` | old verdict | truth |
|---|---|---|---|
| `iter0010` | 4.396623e-10 | SDDM | SDDM — correct, but only **4.4×** from flipping |
| `iter0020` | 9.988164e-12 | LAPLACIAN | **wrong** |
| `iter0030` | 8.242149e-12 | LAPLACIAN | **wrong** |
| `iter0040` | 1.116573e-11 | LAPLACIAN | **wrong** |
| `ecology1` | 8.881784e-17 | LAPLACIAN | correct, 6 orders of margin |

So BoomerAMG and AMGCL were **Dirichlet-pinned on a matrix with no nullspace**, and then scored
against the unpinned original. The 1.71e-8 was not a gap to be closed by asking hypre for a
tighter tolerance — it was a **floor**. All of the residual sits in the single pinned row, and
its size is set by the shift, not by the solver: `1e-6 × 524288 × mean(x) = 1.70e-8`, which no
number of iterations and no `--tol` can move. A calibration can only rescale a tolerance; it
cannot lift a floor.

**Measured before/after on `iter0040`** (CPU, 16T, `--tol 1e-8`, `--repeat 3`, true relative
residual re-graded against the original `L`):

| solver | before: iters / rel_res | after: iters / rel_res |
|---|---|---|
| `apxchol_v1 bg+tree[vec_pool]` | 44 / 9.873e-09 ✅ | 45 / 8.399e-09 ✅ |
| `hypre_boomeramg` | 10 / **1.714e-08** ❌ `not_converged` | 10 / **2.059e-09** ✅ |
| `amgcl` | 19 / **1.840e-08** ❌ `not_converged` | 19 / **7.003e-09** ✅ |

Same iteration counts; only the system changed. BoomerAMG converges in the iterations it always
took, and on total wall clock it now **beats** apxchol on this cell. That is the correct
outcome: the cell was ours to lose and we were only holding it because a heuristic of ours had
crippled two competitors.

The matrix class is now **declared**, not detected — `--class laplacian|sddm`, required with
`--kind operator`, carried in the registry beside `kind` (`runner_common.py`), and **asserted**
against `apxchol::scan_operator`'s per-row census so a mis-declaration is a hard error rather
than a silent one. The scan tests each row against *its own* diagonal, which is why a uniform
shift is visible to it (an excess on every row) and invisible to a global ratio.

This also **invalidates every stored cell on `iter0020`/`iter0030`/`iter0040`**, and as a
**re-run**, not a re-grade: the same flag drove `center_if_laplacian` inside `make_rhs`, so the
right-hand side itself changed (on `iter0040`, `sum(b)` 1.694e-14 → 1.815e-10). See the rule in
`stale_cells.py`.

**RCHOL is x86-only.** Their `util/pcg.hpp` hard-includes `mkl_spblas.h`/`mkl.h` (having
first forced `MKL_INT = size_t`, which is why the build links MKL's **ILP64** interface, as
their own Makefile does). On a machine without MKL — aarch64: Grace / GH200 / Daint — their
shipped solve cannot run, and substituting one of ours is exactly what we stopped doing. The
`rchol`/`rchol_par` rows there are **factor-only**: `rchol()`'s time and fill-in are real,
the solve columns are the `n/a` sentinel, and the solver name carries
*"factor only; upstream solve needs MKL (x86)"*. The price of faithfulness on Grace is four
full competitors (BoomerAMG, AMGCL, AC, AC2) instead of six.

**`solve_rss_mb` is not reported for RCHOL/pRCHOL.** Their `create_sparse` (`pcg.cpp:33-53`)
allocates a second full copy of `A` and of `G` on every construction and never frees it — the
`delete[]` at `pcg.cpp:53` is commented out upstream — so the number would measure their leak
rather than the memory a solve holds. The binary emits `-1` (unmeasured) and the runner drops
the metric rather than storing a misleading value.

## Protocol

- **Series rule — one series per (solver, configuration); no series is a minimum over
  configurations.** A chart series or a table column is exactly one solver at exactly
  one configuration. Cross-method headline charts use one apxchol baseline **declared
  a priori** — `apxchol/bg`, `fair_charts.APX_DEFAULT` — while keeping competitor
  configurations separate. The selector spread (`bg`, `greedy`, `bk`) appears only in
  dedicated compact ablations. No chart substitutes the per-matrix fastest selector.

  This is enforced mechanically: `fair_charts.LABELS` and `gpu_charts.LABELS` pass
  a runtime injectivity check even under `python -O`, so a re-collapse fails loudly
  instead of silently reinstating a minimum.

  *Why it matters.* Until 2026-08-21 all four apxchol selectors mapped to one
  `apxchol` label and `_pick`/`gpu_charts.load` kept the fastest of the four per
  matrix, while every competitor was charted at its single configuration — best-of-4
  for us against best-of-1 for them. Measured over the 27 CPU matrices in the store,
  that minimum was worth a **geomean 5.6 % (max 23.4 %)** against apxchol's own
  declared default, and **9.7 % (max 2.33×)** on the GPU axis; it also shrank the
  `10× apxchol` cap that every timed-out competitor bar is clamped to. Collapsing the
  competitors instead was rejected for the same reason dual-toolchain "report best"
  was: RCHOL/pRCHOL/ParAC-CPU redraw their RNG every process, so a min over their
  configurations is a min over random draws.

- **Status rule — status never selects a representative.** The chart thread count is
  fixed a priori (t16, with a t1 fallback for historical serial cells). There must then
  be exactly one cell for that `(solver, configuration)` series; duplicates are an
  error. A `complete` cell is therefore never ranked against a `timeout` /
  `not_converged` / `failed` cell.

  Schema 2 timeout cells persist the exact wall-clock cap as top-level
  `timeout_cap_s`. Charts draw a numerical `≥cap` only from that field. They never
  reconstruct a bound from current apxchol timings: old schema-1 timeout cells render
  as unbounded `Timeout` and `stale_cells.py` schedules them for rerun.

  Check both invariants with `PYTHONPATH=benchmarks python3 benchmarks/dev/audit_series_rule.py`.

- **Matrix interpretation (what system each file defines).** A `.mtx` file can hold
  either of two different things, and they define two different linear systems. Every
  entry in the registry (`benchmarks/runner_common.py`) **declares** which, and the
  benchmark binary **requires** the declaration on the command line (`--kind`). It is
  never auto-detected: a value-carrying file can perfectly well be a graph
  (`kron_g500-logn16` stores integer *edge weights*), so any heuristic would be a
  silent decision about which problem the suite reports on. An undeclared matrix is a
  hard error, not a default.

  | kind | what the file holds | the system we solve |
  |---|---|---|
  | `graph` | adjacency / `pattern` matrix (no operator diagonal) | `L = D − A`, assembled from `|value|` (unit weights for a `pattern` file; a self-loop on the file's diagonal is dropped). That **is** the system a graph file defines, so it is not a deviation. |
  | `operator` | already-assembled Laplacian / SDDM operator | the published matrix itself — **solved as it stands**, diagonal included, signs untouched. |

  | matrix | kind | | matrix | kind |
  |---|---|---|---|---|
  | `grid_*`, `grid3d_*` (generated) | `graph` | | `com-Amazon` | `graph` |
  | `parabolic_fem` | `operator` | | `coAuthorsDBLP` | `graph` |
  | `apache2` | `operator` | | `kron_g500-logn16` | `graph` |
  | `ecology1` | `operator` | | `com-Youtube` | `graph` |
  | `G3_circuit` | `operator` | | `coPapersDBLP` | `graph` |
  | `thermal2` | `operator` | | `as-Skitter` | `graph` |
  | `iter0010…0040` (IPM) | `operator` | | `com-LiveJournal` | `graph` |
  | | | | `com-Orkut` | `graph` |

  The five SuiteSparse `operator` files carry a stored positive diagonal on every row;
  the `graph` files carry no usable diagonal at all (`kron_g500` has one on 327 of
  65536 rows — self-loops, not an operator). ParAC's own benchmark splits its matrices
  the same way ("physics matrices: used as-is" vs "graph matrices: build Laplacian from
  adjacency"), and four of our five operator files are on its physics list.

  Every solver in a cell receives the **same** matrix: the in-process ones directly,
  the external ones (ParAC, AC/AC2, CMG) through `--dump-mtx`, which writes exactly the
  operator the binary assembled. Each run prints one `[matrix] …` line saying how the
  file was read, and each stored cell repeats it under `matrix_meta`. Solvers that
  cannot take a given matrix produce an **`n/a`** cell (`iters = rel_res = −1`), never a
  silent failure or a missing row — for example ParAC's *graph* mode on an `operator`
  matrix (graph mode grounds a Laplacian; the matrix goes through ParAC's *physics*
  path instead), or AC/AC2 **and CMG** on an operator carrying positive off-diagonals —
  `parabolic_fem` (524 288 of them) and `thermal2` (420). Laplacians.jl needs
  non-negative edge weights, and it solves the matrix it factorizes, so lumping them
  would make it converge on a different matrix than it is scored against; CMG says so
  itself (`cmg_precondition` returns an empty preconditioner: *"The current version of
  CMG does not support positive off-diagonals"*). Every such cell carries the reason
  under `matrix_meta.na_reason` / `matrix_meta.cmg_na_reason`. `n/a` is
  reserved for *cannot take this matrix*: a solver that runs and misses the tolerance is
  `not_converged` — AC/AC2 do reach `apache2` and `G3_circuit` through
  `approxchol_sddm`, but floor at 6.5e-6 / 1.4e-7 because the ground-vertex
  augmentation gives those two negative-weight edges where their row sums go negative.

  Whether the assembled operator is **singular** is a separate, downstream fact, read off
  the matrix rather than declared: a graph always yields a singular Laplacian, and an
  `operator` file may be one (`ecology1` is published as an exact Laplacian) or full-rank
  SDDM (`apache2`, `G3_circuit`, `thermal2`, IPM). Only a singular operator is grounded
  and only its solution and residual are mean-centred.

- **De-singularization (the fairness model).** Grid / SuiteSparse Laplacians are
  **singular** (a constant null vector per connected component). Under the **current**
  protocol they are benchmarked on the **original singular `L`** — each solver removes
  that null space the way its own machinery prefers and is scored on the *true*
  residual `‖b − L x‖` (so the multigrid solvers no longer floor at ~1e-4 from a
  pin-vs-score mismatch):
  **apxchol** native rank-aware solve (mean-centring); **BoomerAMG** a
  provably-safe (DFS-tree-leaf) Dirichlet pin, one node per connected component → SPD;
  **AMGCL** the same Dirichlet pin, run with its default config.
  **ParAC grounds itself**, and is given the input its own scripts build (see
  [`patches/parac/README.md`](patches/parac/README.md) for the measurements behind
  this). A `kind=graph` matrix goes to its **graph** driver as the **pure singular
  `L`**, one connected component at a time: graph mode generates its own zero-sum
  RHS, which is consistent for a connected Laplacian, so it solves the very `L` we
  report on and its printed `‖b − L·x‖ / ‖b‖` is against that `L`. We do **not**
  hand it a Dirichlet-pinned matrix — its RHS generator knows nothing about the
  pin, and the residual against the original `L` then floors at ~1e-3 (com-Amazon
  1.4e-3, coAuthorsDBLP 2.0e-3) no matter how tight the tolerance. Published
  operators declared `class=laplacian` take this same graph route (`ecology1` is
  the current example), because physics mode would trim a real vertex. A
  `class=sddm` matrix goes to its **physics** driver as the **published operator**,
  AMD-reordered and then augmented with the ground row/column exactly as ParAC's
  own `write_graph.jl physics_produce` does; physics mode's trim removes that
  appended node, so what it solves is the published operator itself. Each
  matrix runs **one** mode and the other cell is `n/a` with the reason recorded.
  ParAC's stopping test is absolute (`‖r‖` vs `sqrt(rel_tol)`) and on the
  recurrence residual, so the tolerance is **calibrated from one probe run** —
  configuration, not a patched convergence test — until its own printed residual
  lands under 1e-8. The same competitor wall-clock cap applies to ParAC:
  on **com-Orkut** it exceeds the shared ~20 min competitor cap during setup and is
  recorded as `timeout`, like rchol_par/rchol on the same matrix. **CMG** keeps the `ε·I`
  regularization (`A = L + ε·I`, `ε = 1e-6·mean|diag|`), scored on `L + εI`, on the
  **singular** families only; on a `kind=operator` matrix it now reads the dump *as an
  operator* — diagonal included — and solves it unpinned and unshifted, so it is measured
  on the same published matrix as everything else in the cell. (It used to rebuild the
  diagonal as the degree there, i.e. solve a different system; `parabolic_fem` /
  `thermal2` were "complete" only because of that rebuild — against the real operator
  CMG declines them, see the n/a note below.)
- **Which protocol each committed cell was measured under (mixed).** The store holds
  **1048** cells recorded over two protocols; every cell carries its own provenance note.
  The pin protocol above is the **current** one and covers **essentially the whole GPU
  axis** (237 of the 249 GPU cells) plus part of the CPU axis — 216 cells under the plain
  pin note (183 GPU + 33 CPU) plus the 40 re-run ParAC cells scored against the original
  `L`, and the 54 ParAC-GPU cells run by the same drivers. The **majority of the cells —
  697, and almost all of them CPU (260 grids, 104 IPM, 321 SuiteSparse; 12 GPU) — predate
  it** and come from the earlier
  **unified regularization**: *every* solver on the same `L + ε·I` operator
  (`ε = 1e-6·mean|diag|`), scored on that operator. That protocol is *internally consistent*
  (one matrix, one score, all solvers) — it answers "same regularized matrix" where the
  current one answers "same singular matrix". They are not interchangeable in principle, but
  re-running solvers under both moved **iteration counts by only ~10%** at identical setup
  cost, which is inside the session-variance envelope (see [Machine](#machine)) — so read
  cross-protocol cells as comparable to within that band, not as exact. The remaining cells
  are CMG (27, `ε·I`, scored on `L + εI`) and an older ParAC set (14).
- **All third-party backends read `.mtx`.** Synthetic grids are dumped once via
  `--dump-mtx` and fed to the Julia / MATLAB / external solvers, so they consume the
  *exact* same matrix the C++ solvers build (no per-backend grid generators — e.g.
  3D grids work without a Julia `grid3d` generator).
- <a name="grading-rule"></a>**THE GRADING RULE (tolerance).** A cell is `complete` **iff its
  TRUE relative residual against the defining operator — `‖b − A x‖ / ‖b‖`, recomputed by
  the harness from the returned `x`, never a solver's own recurrence estimate — is at or
  below *exactly* the requested `tol` (1e-8).** One rule, every solver, ours included: no
  grace factor, no per-solver pass mark. It is implemented in one place,
  `runner_common.classify()`, and the chart filters use the same mark.

  Until 2026-08-20 this band was **10·tol** for every solver routed through `classify()`
  (apxchol, BoomerAMG, AMGCL, RCHOL, CMG, AC/AC2) while `parac_runner` held ParAC to `tol`
  — one competitor graded ten times harder than everything else, including us. That
  asymmetry is gone.

  When a solver's **own** stopping test is optimistic (a preconditioned-residual ratio, an
  absolute test, a recurrence estimate), the answer is **never** to relax this grade. It is,
  in order of preference:
  1. **Calibrate what we hand their loop** so their true residual lands at `tol` — ParAC-CPU
     and ParAC-GPU (`PARAC_REL_TOL` from a one-run probe, `parac_runner._calibrate_rel_tol`),
     AC-`sddm` (`tol_eff = tol·‖b‖/‖b_aug‖` for the augmented basis). BoomerAMG needs no
     calibration: recurrence drift was ruled out, and its former apparent gap was the
     now-fixed matrix-classification bug documented below.
  2. **Patch their convergence test** to the true residual when no parameter reaches it —
     `gpu_rchol`'s CUDA PCG, patched in `benchmarks/CMakeLists.txt` and documented in
     [`patches/`](patches/), with the reason.
  3. Otherwise the cell is **`not_converged`, carrying its true residual**. That is an
     honest result, not a harness failure.

  Every calibration and patch is listed in [Whose solve loop produced each
  number](#whose-solve-loop-produced-each-number) with the tolerance actually passed.
- **Reps**: 3, median — except the **CMG** cells, which are single-shot (`repeat=1`,
  one MATLAB run per matrix). **Threads**: 16 physical cores, pinned, run without contention.
- **Metrics**: setup_s, solve_s, total_s, PCG iterations, rel_res, µs/nnz.
  `setup_s` includes every solver-specific transformation from the shared Eigen
  operator (grounding, format conversion and host/device upload) as well as
  preconditioner construction. `solve_s` includes per-RHS workspace allocation,
  upload and the iterative solve. Common input parsing/operator assembly and the
  harness's post-solve true-residual grading are outside `total_s` for every row.
- **Crash artifacts**: the shared shell harness disables core dumps for every
  solver by default; a third-party crash is still recorded in its cell and logs,
  but cannot leave a factor-sized core in the campaign directory. Set
  `APXCHOL_BENCH_COREDUMP=1` only for an intentional diagnostic reproduction.
- **Disconnected matrices (multi-component, e.g. the social giants `as-Skitter`,
  `kron_g500`, `thermal2`).** Solvers handle the per-component null space two ways.
  **apxchol** (native mean-centring) and **AMGCL** (+pin, its default) solve the **whole**
  disconnected operator in one shot — faster than paying per-component build overhead.
  **BoomerAMG** (and coarse-grounded AMGCL) instead **split**: decompose into connected
  components, solve each independently (per-component Dirichlet pin → SPD), and recombine
  the cell as **setup_s, solve_s = SUM** over components (plus the union-find decomposition
  + sub-matrix extraction + recombine overhead, charged to **setup**); **iters = MAX** over
  components (the bottleneck block — summing would let hundreds of trivial singleton specks
  swamp the count); **rel_res = √(Σ‖res_c‖²)/‖b‖** (valid because `L` is block-diagonal
  across components, so per-component residuals are independent); **peak RSS / VRAM = MAX**
  over components (peak resident during the largest sub-solve). A connected matrix is a
  single component, so the split path is a no-op.
- **CSV schema**: `solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz`.

### Machine

Every committed number was measured on one machine: **AMD Ryzen 9 7945HX** (16 cores /
32 threads, Zen 4), **128 GB** RAM, **NVIDIA RTX 4090 Laptop GPU (16 GB)**, Linux.

**Which toolchain produced a cell is recorded in the cell**, not in this paragraph.
Every cell written from 2026-08-21 on carries `compiler`, `compiler_version`,
`openmp_runtime` and `arch_flags` under `provenance` (plus `cuda_host_compiler` on a
CUDA build), and each of those comes from the artefact that actually ran:

- **Solvers running in our binary** (apxchol, RCHOL/pRCHOL, BoomerAMG, AMGCL) — the
  binary prints one `BUILD_META …` line as the first thing `main()` does
  (`emit_build_meta`, `benchmarks/src/benchmark.cpp`) and the runner lifts it
  (`runner_common.parse_build_meta`). The compiler names itself through its own
  predefined macros, the OpenMP runtime is probed with `dlsym` in the running
  process, and the tuning flags are compiled in by `benchmarks/CMakeLists.txt`.
  Nothing is inferred from which build directory was invoked: a stale binary in
  `build-clang/` is still a gcc binary, and this is the one way to say so.
- **External solvers** (ParAC's CPU/GPU drivers, the CMG MEX) — read off the ELF of
  the file that ran (`.comment` producers + `DT_NEEDED`, `runner_common.binary_toolchain`),
  not copied from the build script that is supposed to have produced it.
- **AC/AC2** — julia JIT: version + libLLVM + `--cpu-target`, asked of the julia that
  runs them. `arch_flags` is not recoverable from an ELF, so it reads `unknown` for the
  external binaries rather than being guessed.

Cells written **before** that date carry none of these fields — **all 826 of them**:
616 in `results/cells` plus 210 in `results/scaling_cells`, whose provenance keys were
exactly `git_sha, boost, boost_expected, note, repeat, tier, timestamp, source`. For
those, the toolchain is genuinely unrecorded — GCC on this machine, with no version
pinned down. (An earlier draft of this paragraph said "all 616", counting only the main
store; the thread-scaling cells are just as unstamped and are just as much part of the
published figures.)
Runs are **boost-on** (no frequency lock), which on this laptop gives a **±20–30%
session-to-session spread** on the same commit and config — so treat only ratios well
above that band as signal, not individual cells or small deltas. The locked-frequency
harness for paired A/B work is in `benchmarks/dev/` (see [Timing harness](#timing-harness)).

### Workloads (current run)

- **Grids** — 2D square (`grid_500…grid_5000`, κ=100 *weighted*, not uniform Poisson)
  and 3D 7-point (`grid3d_100…grid3d_250`).
- **IPM** — interior-point normal-equation matrices (`iter0010…0040`).
- **SuiteSparse** — planar/PDE (`parabolic_fem`, `apache2`, `ecology1`, `G3_circuit`,
  `thermal2`) + smaller scale-free/social (`com-Amazon`, `coAuthorsDBLP`, `kron_g500-logn16`).
  The charts split SuiteSparse into **`_small`** (these) and **`_giants`** (the large
  social graphs) so the giants don't flatten the FEM matrices.

Both **CPU and GPU** axes are run. The 5 social **giants** (`as-Skitter`,
`coPapersDBLP`, `com-LiveJournal`, `com-Orkut`, `com-Youtube`) are covered on **both**
axes — 9 GPU cells each, essentially all under the current pin protocol. What is still
outstanding for them is the **CPU** side: 117 of their 139 CPU cells carry the earlier
unified-`reg_rel` provenance and have not been re-run under the pin protocol, so the
CPU `_giants` panels mix protocols (see the provenance bullet above).

### Residual mean-centring is not a thumb on the scale (measured, 2026-08-21)

`center_if_laplacian` mean-centres the scored residual and not only the solution,
and `‖r − mean(r)‖ ≤ ‖r‖` always — so in principle the harness could report a
residual *better* than the truth but never worse, a one-directional channel that
would bite hardest exactly where margins are thinnest.

It was measured rather than argued about: **ratio 1.000000000000 on 16/16 real
cells, largest effect anywhere 9.4e-7 relative.** The reason is structural, not
lucky — `‖L·1‖` is exactly 0 on every graph matrix, so there is nothing to centre
away. Removing the centring would invalidate zero cells and could not change any
published verdict.

The probe used to establish this was deliberately **not** kept: it was env-gated
instrumentation for an effect proven to be exactly zero, and half of it measured
the Laplacian-vs-SDDM sniff that no longer exists. If it needs re-checking, the
one-line test is whether `L * VectorXd::Ones(n)` is nonzero for the matrix in
question — for a declared `class=sddm` operator it is, and for those the residual
is not centred at all.

## Running

**Re-running only what a change invalidated.** Each cell records the `git_sha` it
was produced at, and the sweep is resume-safe (a cell with a terminal status is
skipped). So a code change that alters what a solver measures does not cost a
full sweep:

```sh
python3 benchmarks/stale_cells.py            # report: total, stale, reusable
python3 benchmarks/stale_cells.py --delete   # drop the stale cells
python3 benchmarks/sweep_fair.py ...         # refills exactly those gaps
```

`stale_cells.py` carries one rule per invalidating commit (which solvers or which
matrix kinds it changed). **Add a rule whenever a change alters what is measured
or which matrix is solved** — a stale cell is indistinguishable from a fresh one
in the tables, and mixing the two is how a corrected defect quietly survives in
a published number. The stale cells are committed, so `--delete` is recoverable
with `git checkout -- results/cells`.


**Input matrices.** Grid instances are generated in-process. The SuiteSparse
matrices come from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu);
`./scripts/download_graphs.sh` (repo root) fetches every one used by the sweep
into `data/matrices/`. The `ipm_*` cells use a ladder of LP interior-point
Schur-complement matrices (`data/ipm/iterNNNN/matrix.mtx`) that is not
redistributed with the repo — without the files those cells error out (the
sweep records the failure and continues); contact the authors for the dataset
or point `data/ipm/` at your own IPM sequence (Matrix Market, one directory
per iterate).

**Prerequisites.** Only Eigen is mandatory; every competitor is behind a CMake gate and
is silently skipped when its dependency is missing (configure prints which):

| Solver / axis | Needs |
| --- | --- |
| `rchol_par` (pRCHOL) | system **METIS** (`libmetis` + `metis.h`) |
| `cholmod` | system **CHOLMOD** (SuiteSparse) |
| `amgcl`, `amgcl_cuda` | **Boost** headers (AMGCL only; the apxchol core is Boost-free) |
| `rchol` / `rchol_par` **solve** (their `util/pcg.cpp`), ParAC CPU driver | **MKL** (`-DMKL_ROOT=…`, `MKLROOT`, or `paths_local.cmake`); **ILP64** interface, x86 only. Without it those two rows are factor-only — see [Whose solve loop](#whose-solve-loop-produced-each-number) |
| GPU axis | **CUDA** + `-DAPXCHOL_USE_CUDA=ON`; Hypre-GPU also needs `-DBENCH_HYPRE_USE_CUDA=ON`, ParAC needs `-DBUILD_GPU_RCHOL=ON` |
| `cmg`, `parac`, `ac`/`ac2` | out-of-tree tools — see [Local configuration](#local-configuration-out-of-tree-tools) |

BoomerAMG (Hypre) and AMGCL are pulled in by FetchContent, so they need network
access on the first configure.

```bash
# Build (root library + benchmarks are separate CMake projects)
cmake -B benchmarks/build -S benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc) benchmark

# One solver on one matrix (original singular L; BoomerAMG/AMGCL self-ground via the pin)
benchmarks/build/benchmark --graph grid --n 2000 \
    --solver hypre_boomeramg --threads 16 --tol 1e-8 --repeat 3 --csv

# Dump a matrix to .mtx (for external solvers / ParAC)
benchmarks/build/benchmark --graph grid3d --n 100 --dump-mtx /tmp/g.mtx --solver none

# The single fair CPU sweep — ONE runner for grids + SuiteSparse + IPM, and for
# EVERY solver in the comparison: apxchol + the C++ competitors in-process, ParAC
# (graph+physics, parac_runner.py), AC/AC2 (Julia) and CMG (MATLAB container,
# cmg_matlab_runner.py). Original singular L; each solver self-grounds
# (--only <ids> refreshes a subset). Opt out individually with --no-parac /
# --no-julia / --no-cmg (--parac-only runs only ParAC). One-time setup for AC/AC2
# and for ParAC's own input producer:
#   julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
python3 benchmarks/sweep_fair.py
# ParAC alone (their write_graph.jl producer + driver), standalone:
python3 benchmarks/parac_runner.py --device cpu  # [--only id1,id2]
# CMG alone (canonical MATLAB CMG in the matlab-deps container):
python3 benchmarks/cmg_matlab_runner.py
# CPU charts + tables, then the thread-scaling charts:
PYTHONPATH=benchmarks python3 benchmarks/fair_charts.py --out benchmarks/latest
python3 benchmarks/thread_scaling.py
# Cholesky-family fill (one-time, factor-only — measures factor density, no solve):
python3 benchmarks/fill_pass.py
python3 benchmarks/fill_chart.py --out benchmarks/latest/figures
# Per-(selector, matrix) SpTRSV back-solve level counts + the selector_levels figure:
python3 benchmarks/selector_levels.py
# SpTRSV level structure per matrix (level counts / work concentration):
python3 benchmarks/level_stats.py
```

For the **GPU axis** (same runner, same per-cell store — see
[Results](#results)) build the benchmark with CUDA, then run the fair
sweep with `--device gpu` (apxchol uses the GPU-resident PCG) and re-render:

```bash
cmake -B benchmarks/build-cuda -S benchmarks -DCMAKE_BUILD_TYPE=Release \
      -DAPXCHOL_USE_CUDA=ON -DBENCH_HYPRE_USE_CUDA=ON -DBUILD_GPU_RCHOL=ON
cmake --build benchmarks/build-cuda -j$(nproc) benchmark
# BENCH_HYPRE_USE_CUDA (Hypre GPU backend) and BUILD_GPU_RCHOL (FetchContents the
# pinned ParAC commit and applies the benchmark-only patch stack) both default
# OFF. A CUDA build with CPU-only Hypre rejects `hypre_boomeramg_gpu` instead of
# emitting a CPU result under a GPU label; BUILD_META records hypre_cuda=on|off.
# GPU cells written into the SAME results/cells store (device=gpu), resume-safe.
# ParAC-GPU (both CUDA drivers; needs the ParAC repo + drivers built) runs
# in-process as part of the sweep, or alone via parac_runner.py --device gpu:
python3 benchmarks/sweep_fair.py --device gpu
# render GPU-only + combined CPU-vs-GPU figures from the store:
python3 benchmarks/gpu_charts.py --root results/cells --out benchmarks/latest/figures
PYTHONPATH=benchmarks python3 benchmarks/combined_charts.py --out benchmarks/latest/figures
```

ParAC is charted as **two drivers on both devices** — graph (Laplacian) and
physics (SDDM). The same self-contained binary runs both: on GPU via driver.cu /
driver_physics.cu, on CPU via one extra argv (`is_graph=0`). **Each matrix runs
exactly one of them**, chosen by operator class: `class=laplacian` → graph
(pure `L`, its own zero-sum RHS), `class=sddm` → physics (published operator plus
ParAC's ground-node augmentation, which the mode's trim removes). This routes the
published `kind=operator,class=laplacian` matrix `ecology1` through graph mode.
The other cell is `n/a` and carries the reason, so a chart shows a deliberate gap rather
than a solver that silently vanished. Running the *other* mode is not a second
data point but a wrong answer: physics on an un-augmented operator deletes a real
degree of freedom (apache2 scores 3.1e-3 against the published matrix while ParAC
prints 8.8e-9; G3_circuit does not converge at all). Note: ParAC's GPU SpTRSV is extremely
sensitive to the elimination ordering — its required nnz-sort is a **random
permutation then a sort by column-nnz** (`parac_nnz_sort.jl`, matching their
`write_graph.jl`). A deterministic degree-sort instead builds a very deep
elimination tree and makes the cuSPARSE level-set SpTRSV ~1000× slower
(1.5 s/iter vs ~1 ms/iter on a 250k-node grid). Setup is counted as
`reorder + post-parse adapter + complete factor setup + solver analysis`; solve
includes RHS work, PCG and returning `x` to host. The original kernel/conversion/
SpSV timers remain diagnostics and are not summed. Disk I/O, residual validation
and cleanup are excluded, matching the in-process solvers.

**Binary solver names**: `apxchol_v1` (headline/default
`--v1-configs bg+tree[vec_pool_aos]`),
`rchol`, `rchol_par`, `hypre_boomeramg`, `amgcl`, `cholmod`, `cg`,
`ldlt`. GPU variants (CUDA build): `hypre_boomeramg_gpu`,
`amgcl_cuda`, plus apxchol's GPU-resident PCG (the default on CUDA builds).
External: `ac`/`ac2` (Julia), `cmg` (canonical MATLAB CMG via `cmg_matlab_runner.py` + the matlab-deps container), `parac` (driver via `parac_runner.py`).

**Key flags**: `--reg-rel <c>` (SDDM regularization), `--dump-mtx <path>`,
`--threads`, `--tol`, `--repeat`, `--mtx <path>`, `--graph {grid,grid3d,checkerboard,erdos}`.

### Local configuration (out-of-tree tools)

Some competitors live outside this repository, so their paths are **not** hardcoded:

| What | Environment variable | `paths_local.py` name |
| --- | --- | --- |
| ParAC CPU driver (`experiment/driver`) | `APXCHOL_PARAC_DRIVER` | `PARAC_CPU_DRIVER` |
| ParAC AMD-reorder cache directory | `APXCHOL_PARAC_REORDER_DIR` | `PARAC_REORD` |
| ParAC's own `cpu_implementation/write_graph.jl` (the input producer the runner calls; **derived from the driver's checkout** unless the two live apart) | `APXCHOL_PARAC_WRITE_GRAPH` | `PARAC_WRITE_GRAPH` |
| Runtime libs the ParAC driver needs | `APXCHOL_PARAC_LDLIB` | `PARAC_LDLIB` |
| MATLAB install tree (CMG) | `APXCHOL_MATLAB_ROOT` | `MATLAB` |
| `cmg-solver` checkout (CMG) | `APXCHOL_CMG_SOLVER` | `CMG_SOLVER` |

Set the environment variables, or drop the same names into a **gitignored**
`benchmarks/paths_local.py` (assignments there win over the environment). Leaving
them unset is fine until a ParAC/CMG cell actually runs; the runner then fails with
a message naming the variable to set. Everything else — the repo root, the benchmark
binaries, the cell store — is derived from the source tree, so no configuration is
needed for the in-tree solvers.

**MKL** (optional; it backs RCHOL's own PCG — their `util/pcg.cpp`, which every
`rchol`/`rchol_par` solve number now comes from — and the ParAC CPU driver) is found
via `-DMKL_ROOT=/path/to/oneapi/mkl/<version>`, the `MKLROOT` environment variable,
or a gitignored `benchmarks/paths_local.cmake` containing
`set(MKL_ROOT "..." CACHE PATH "")`. Two interface layers are used, deliberately:
`rchol_lib` links **ILP64** (`mkl_intel_ilp64`) because their `pcg.hpp` forces
`MKL_INT = size_t` before including `mkl_types.h` — their own Makefile does the same —
while the out-of-tree ParAC/`gpu_rchol` CPU driver links **LP64**, MKL's default `int`.
Without MKL, configure prints `MKL not found ...`, the ParAC CPU driver is skipped, and
`rchol`/`rchol_par` degrade to factor-only rows (their solve is x86-only).

### Timing harness

`benchmarks/dev/` holds the thermal-stable timing harness used for perf A/B work:
`bench_stable_setup.sh` (disable boost / pin the governor; needs root),
`bench_stable.sh` (warmup + N-rep median with cooldown gating), `bench_stable_all.sh`
(the standard workload set, CPU + CUDA), `bench_reference_table.sh` (the cross-solver
reference table) and `bench_stable_teardown.sh` (restore the CPU state).

## Results

`benchmarks/latest/` holds the committed charts/tables. The **comparison** figures all
read the one committed per-cell store (`results/cells`): CPU figures from
`fair_charts.py`, GPU figures from `gpu_charts.py` (device=gpu cells), and the
CPU-vs-GPU overlays from `combined_charts.py`. The remaining figures read their own
stores, which are **gitignored but regenerable** by the commands in [Running](#running):
fill from `results/fill_cells` (`fill_pass.py` → `fill_chart.py`), thread scaling from
`results/scaling_cells` (`thread_scaling.py`), and the SpTRSV level-count chart from
`results/selector_levels.csv` (`selector_levels.py`). Raw data:
[`latest/results.csv`](latest/results.csv). Note the `[svec]` rows there (and the matching
cells) are **historical**: that storage backend has since been removed from the library and
can no longer be re-measured.

Each headline section below is shown three ways: a **combined** CPU-vs-GPU overlay
(solid = CPU t16, `///` = GPU; one colour per method), then the **CPU-only** and
**GPU-only** views with the full solver set for that device. CPU and GPU wall times
are different hardware, so the combined view is for per-method "what does the GPU
buy", not a single ranking. Every solver attacks the original singular `L`
(de-singularized per the [Protocol](#protocol); every solver — the in-house ones *and*
ParAC, which now uses a per-component-consistent pin-zeroed RHS — scores `‖b−Lx‖` against
the original L, only CMG uses the `εI`-regularized variant) to true 1e-8. On apxchol, the GPU bar is the **GPU-resident PCG** (
all PCG vectors stay on device; since 2026-08-18 our own SpMV / vector kernels, no cuBLAS / cuSPARSE), which is 2-4.5× faster per iter
than the host-PCG hybrid (measured on an RTX 4090 Laptop at locked clocks).

GitHub markdown has no tabs, so each metric uses collapsible `<details>` blocks —
click to switch between the **Combined**, **CPU**, and **GPU** views. Bars are
**sorted fastest→slowest within each matrix group**; matrices appear in the same
nnz-ascending order in every figure (for grids only the largest 2D and largest 3D
grid are shown per-matrix; the full ladder is in the scaling chart). Non-converged
runs are **excluded** from the time/iteration charts (only in the accuracy chart).

### Setup + solve time (t16)

<details open><summary><b>Combined — CPU vs GPU per method</b></summary>

Per method, the **GPU bar is on the left (outlined in black)** and the **CPU bar on
the right (plain)**; each bar stacks setup (solid) + solve (`///`). Total time is just
the bar height. (Grids reduce to the largest 2D + largest 3D here — grouped bars get
cramped past a handful; the full grid ladder is in the heatmaps below.)

![Grids 2D combined breakdown](latest/figures/combined_breakdown_grids_2d.png)
![Grids 3D combined breakdown](latest/figures/combined_breakdown_grids_3d.png)
![IPM combined breakdown](latest/figures/combined_breakdown_ipm.png)
![SuiteSparse small combined breakdown](latest/figures/combined_breakdown_suitesparse_small.png)
![SuiteSparse giants combined breakdown](latest/figures/combined_breakdown_suitesparse_giants.png)
![SuiteSparse XL giants combined breakdown](latest/figures/combined_breakdown_suitesparse_giants_xl.png)

(Grids show the largest **two** 2D + **two** 3D sizes — the very largest, grid_5000 /
grid3d_250, OOM Hypre-GPU, so the next size down is included to keep a Hypre-GPU bar.
The social giants split into mid — com-Youtube / as-Skitter / coPapersDBLP — and **XL**
— com-LiveJournal / com-Orkut — since the two heaviest dwarf the rest on a shared axis.)

The setup-only and solve-only cuts are **heatmaps** (matrix × method, every measured
grid): colour = time relative to the best solver in that matrix (per column, green =
fastest, **log** scale), each cell annotated with the absolute time over the ratio.

| setup only | solve only |
|---|---|
| ![grids setup](latest/figures/combined_setup_grids.png) | ![grids solve](latest/figures/combined_solve_grids.png) |
| ![ipm setup](latest/figures/combined_setup_ipm.png) | ![ipm solve](latest/figures/combined_solve_ipm.png) |
| ![ss setup](latest/figures/combined_setup_suitesparse.png) | ![ss solve](latest/figures/combined_solve_suitesparse.png) |

Heatmap cell legend: a value/×best is shown for solvers that completed; **`Timeout`** =
hit the *total* wall-clock cap, so there's no setup/solve breakdown — the **total**
heatmap fairly clamps it to `≥cap`, the setup/solve cuts just mark it `Timeout`, and in
the bar charts it appears **last** as a hatched red-edged `≥cap` bar; **`OOM`** = ran out
of memory (RAM = host, GPU = device); a **blank/white** cell = never run.
</details>

**Solver × matrix overview** — one grid per family, colour = total time relative to
the best solver in that matrix (per column; green = winner, **log** scale so the slow
tier doesn't saturate to one red wall), each cell annotated with absolute total (s)
over the ratio. The at-a-glance "who wins where" companion to the per-matrix grouped
bars. The headline **combined** grid puts every (solver, device) on its own row, so a
GPU variant beating its CPU twin is visible and the per-column winner is taken across
both devices; CPU-only and GPU-only variants are in the dropdown.

![Grids overview](latest/figures/combined_overview_grids.png)
![IPM overview](latest/figures/combined_overview_ipm.png)
![SuiteSparse overview](latest/figures/combined_overview_suitesparse.png)

<details><summary><b>CPU-only / GPU-only overview</b></summary>

![Grids overview CPU](latest/figures/combined_overview_cpu_grids.png)
![Grids overview GPU](latest/figures/combined_overview_gpu_grids.png)
![IPM overview CPU](latest/figures/combined_overview_cpu_ipm.png)
![IPM overview GPU](latest/figures/combined_overview_gpu_ipm.png)
![SuiteSparse overview CPU](latest/figures/combined_overview_cpu_suitesparse.png)
![SuiteSparse overview GPU](latest/figures/combined_overview_gpu_suitesparse.png)
</details>

<details><summary><b>CPU only</b> (the combined chart, CPU rows only)</summary>

![Grids 2D CPU breakdown](latest/figures/combined_breakdown_cpu_grids_2d.png)
![Grids 3D CPU breakdown](latest/figures/combined_breakdown_cpu_grids_3d.png)
![IPM CPU breakdown](latest/figures/combined_breakdown_cpu_ipm.png)
![SuiteSparse small CPU breakdown](latest/figures/combined_breakdown_cpu_suitesparse_small.png)
![SuiteSparse giants CPU breakdown](latest/figures/combined_breakdown_cpu_suitesparse_giants.png)
![SuiteSparse XL giants CPU breakdown](latest/figures/combined_breakdown_cpu_suitesparse_giants_xl.png)
</details>

<details><summary><b>GPU only</b> (the combined chart, GPU rows only)</summary>

![Grids 2D GPU breakdown](latest/figures/combined_breakdown_gpu_grids_2d.png)
![Grids 3D GPU breakdown](latest/figures/combined_breakdown_gpu_grids_3d.png)
![IPM GPU breakdown](latest/figures/combined_breakdown_gpu_ipm.png)
![SuiteSparse small GPU breakdown](latest/figures/combined_breakdown_gpu_suitesparse_small.png)
![SuiteSparse giants GPU breakdown](latest/figures/combined_breakdown_gpu_suitesparse_giants.png)
![SuiteSparse XL giants GPU breakdown](latest/figures/combined_breakdown_gpu_suitesparse_giants_xl.png)
</details>

### Memory — peak and solve-held RSS (t16)

Two **heatmaps** per family (matrix × method): colour = resident memory relative to the
**least** in that matrix (per column, green = lightest, **log** scale), each cell
annotated in GB over the ratio. **Peak RSS** is the high-water mark over the whole run
(setup-dominated — the factorization/coarsening peak); **solve-held RSS** is the memory
still resident during the PCG solve (the operator + factor that has to stay live). `OOM`
marks a solver that exceeded the host memory cap (e.g. serial RCHOL's natural-order
fill on com-Orkut hits ~110 GB); a blank cell = never run.

These are **host** RSS and show **CPU solvers only** — a GPU solver's host RSS is not its
real footprint (that lives in **VRAM**, charted separately just below), so including GPU
rows here would misleadingly paint a GPU competitor as memory-light right up until it hits
a GPU-OOM. GPU device-memory pressure shows both in the VRAM heatmap and as `OOM GPU` in
the time heatmaps.

| peak RSS (setup-dominated) | solve-held RSS |
|---|---|
| ![grids peak rss](latest/figures/combined_rss_peak_grids.png) | ![grids solve rss](latest/figures/combined_rss_solve_grids.png) |
| ![ipm peak rss](latest/figures/combined_rss_peak_ipm.png) | ![ipm solve rss](latest/figures/combined_rss_solve_ipm.png) |
| ![ss peak rss](latest/figures/combined_rss_peak_suitesparse.png) | ![ss solve rss](latest/figures/combined_rss_solve_suitesparse.png) |

### GPU memory — peak VRAM (t16)

The device-memory analog of peak RSS, for **GPU solvers only**: one **heatmap** per family
(matrix × method), colour = peak device VRAM relative to the **least** in that matrix
(per column, green = lightest, **log** scale), annotated in GB over the ratio. Peak VRAM is
the high-water mark of the solver's *own* device allocations over the whole run, sampled
per-process (`nvidia-smi --query-compute-apps`), so it excludes the desktop/compositor and
is comparable across solvers. `OOM GPU` marks a solver that exceeded the 16 GB device (e.g.
BoomerAMG on as-Skitter/com-LiveJournal), `FAIL` an error (BoomerAMG/ParAC on com-Orkut).
Note apxchol is the only solver that fits **every** giant on the device — com-Orkut at
~14.3 GB — where the AMG/ParAC competitors OOM or fail. (Per-process sampling can't isolate
the solve *phase* alone, so unlike host RSS there is no separate solve-held VRAM panel.)

![grids peak vram](latest/figures/combined_vram_peak_grids.png)

![ipm peak vram](latest/figures/combined_vram_peak_ipm.png)

![ss peak vram](latest/figures/combined_vram_peak_suitesparse.png)

### PCG iterations (preconditioner quality, threads/device-independent)

**Heatmaps** (matrix × method, every measured grid): colour = iterations relative to
the fewest in that matrix (per column, green = fewest, log scale), each cell annotated
with the absolute iteration count over the ratio. BoomerAMG is the green floor
(near-constant ~7–8); the Cholesky-type solvers sit several × above.

<details open><summary><b>Combined — CPU vs GPU</b></summary>

![Grids combined iters](latest/figures/combined_iters_grids.png)
![IPM combined iters](latest/figures/combined_iters_ipm.png)
![SuiteSparse combined iters](latest/figures/combined_iters_suitesparse.png)
</details>

<details><summary><b>CPU only</b></summary>

![Grids CPU iters](latest/figures/combined_iters_cpu_grids.png)
![IPM CPU iters](latest/figures/combined_iters_cpu_ipm.png)
![SuiteSparse CPU iters](latest/figures/combined_iters_cpu_suitesparse.png)
</details>

<details><summary><b>GPU only</b></summary>

![Grids GPU iters](latest/figures/combined_iters_gpu_grids.png)
![IPM GPU iters](latest/figures/combined_iters_gpu_ipm.png)
![SuiteSparse GPU iters](latest/figures/combined_iters_gpu_suitesparse.png)
</details>

### Solution accuracy (final ‖b−Ax‖/‖b‖; bars above the 1e-8 line did not converge)

<details open><summary><b>CPU</b></summary>

![Grids 2D accuracy](latest/figures/accuracy_grids_2d.png)
![Grids 3D accuracy](latest/figures/accuracy_grids_3d.png)
![IPM accuracy](latest/figures/accuracy_ipm.png)
![SuiteSparse small accuracy](latest/figures/accuracy_suitesparse_small.png)
![SuiteSparse giants accuracy](latest/figures/accuracy_suitesparse_giants.png)
</details>

<details><summary><b>GPU</b></summary>

![Grids GPU accuracy](latest/figures/accuracy_gpu_grids.png)
![IPM GPU accuracy](latest/figures/accuracy_gpu_ipm.png)
![SuiteSparse GPU accuracy](latest/figures/accuracy_gpu_suitesparse.png)
</details>

### Thread scaling (1/2/4/8/16 threads) — setup and solve reported separately

Setup and solve scale very differently (setup parallelizes; the SpTRSV-bound solve
barely does), so they are charted apart. Speedup = t1/tN; efficiency = speedup/N.

**Caveat:** these two charts predate the pin protocol — they were measured on the
`ε·I`-regularized operator (`thread_scaling.py` now runs the singular `L` unshifted, so a
re-run replaces them). Scaling ratios are unaffected to first order, absolute times are not.

![Setup speedup](latest/figures/threads_setup_speedup.png)
![Solve speedup](latest/figures/threads_solve_speedup.png)

### Scaling (total time vs nnz, log-log — grid size ladder only)

| CPU | GPU |
|---|---|
| ![Grids scaling CPU](latest/figures/scaling_grids.png) | ![Grids scaling GPU](latest/figures/scaling_gpu_grids.png) |

### Fill (factor density — Cholesky family only)

Fill = `2·offdiag(L) / offdiag(A)` (factor off-diagonals per input off-diagonal).
This is the **fill-vs-iterations tradeoff** axis: a denser factor is a stronger
preconditioner (fewer PCG iterations) but costs more memory and SpTRSV work. One
consistent definition across the Cholesky-type solvers — AMG (BoomerAMG/AMGCL) has
no triangular factor, so no comparable metric and they are absent here. Series:
**apxchol per IS selector** (bg/greedy/bk — the order changes the factor density),
the **AC/AC2** Kyng16 reference, **RCHOL/pRCHOL**, and **ParAC** split into
**graph/physics × CPU/GPU** (the CPU AMD-reordered and GPU random-nnz-sort
implementations factor differently, so their fill genuinely differs). apxchol and AC
produce the **sparsest** factors (~1.5–2.2×), ParAC a bit denser (~2.5×), RCHOL/pRCHOL
the densest (up to ~17× on 3D grids). AC2's `(split=2, merge=2)` oversampling makes its
factor visibly denser than AC. (AC/AC2 cells missing from these heatmaps were not swept for fill.)

Each family is a **matrix × method heatmap** (every measured grid; colour = fill
relative to the sparsest factor in that matrix, green = sparsest, log scale; cell =
absolute fill over the ×sparsest ratio).

![Grids fill heatmap](latest/figures/fill_heatmap_grids.png)
![IPM fill heatmap](latest/figures/fill_heatmap_ipm.png)
![SuiteSparse fill heatmap](latest/figures/fill_heatmap_suitesparse.png)

### apxchol internal ablation: IS selector × storage backend

apxchol has two orthogonal design axes — the independent-set **selector** (bg =
block-greedy, greedy = fixed-priority greedy, bk = Baumann-Kyng) and the incidence
**storage backend** (`fwd_star` → `vec` → `bstr`
(bit-string) → indexed `vec_pool`). This historical full selector×storage sweep is shown as
small-multiple heatmaps (total / setup / solve / iterations), each cell medianed
over the family's matrices (a matrix set common to every config, for fairness),
colour = ×best CPU cell (green = fastest). A trailing `vec_pool (GPU)` column shows
that indexed ablation's CPU→GPU shift; the method headline now uses
`vec_pool_aos` on both CPU and GPU.

**Takeaway (read across the families):** indexed vec_pool is robust and is the
only legacy backend that does not collapse on the **high-degree IPM**
matrices — there `fwd_star`'s per-edge linked-list pointer chase blows setup up to
~3× (median total ≈3.1–3.7 s across the four selectors vs vec_pool's ≈1.4–1.9 s). On the
**low-degree grids** `fwd_star` is competitive (and marginally faster for `greedy` and `bk`,
though not for `bg`/`root`), confirming the pointer chase only bites at high degree.
The high-level default is now `vec_pool_aos`, whose inline records remove the
indexed pool lookup; its separate 77-cell GH200 gate is documented in the root
`AGENTS.md`. The `vec` (dense array) and `bstr`
(bit-string) backends are middling — never the fastest. Among selectors `bg` / `greedy`
/ `root` are competitive and `bk` is consistently slowest. The GPU vec_pool column is
fastest overall on grids (its solve is ≈4× the CPU's) and on most SuiteSparse, mixed
on IPM.

![apxchol ablation grids](latest/figures/ablation_grids.png)
![apxchol ablation IPM](latest/figures/ablation_ipm.png)
![apxchol ablation SuiteSparse](latest/figures/ablation_suitesparse.png)

### apxchol IS selector × graph type

A cross-family view of the same selector question: which IS selector wins on which
**graph type**, at indexed `vec_pool` storage (t16). Rows = the three selectors,
columns span the structured→irregular axis (2D/3D grids → FEM/planar → IPM →
social/scale-free); colour normalises *per column* so green = the best selector for
that graph and **bold** = the per-graph winner.

**Takeaway:** on structured grids and FEM, `bg` is best and the spread is small;
on the **dense social/citation graphs the selector matters a lot**. The retired
`root+tree` path produced extremely deep elimination trees, while the same selected
set emitted in candidate order by today's `greedy+tree` remains robust there.
`greedy+tree` is the most uniformly good non-default choice; `bk` is rarely the winner. (The iteration-count
variant is more uniform — the dramatic differences are in the SpTRSV solve, not the
PCG iteration count.)

![apxchol selector × graph, total time](latest/figures/selector_graph_total.png)
![apxchol selector × graph, PCG iterations](latest/figures/selector_graph_iters.png)

Per-family selector × matrix panels (all matrices of each family, total / setup /
solve / iterations small-multiples; per-column normalised), on both axes:

![selector × matrix, grids CPU](latest/figures/selector_grids_cpu.png)
![selector × matrix, grids GPU](latest/figures/selector_grids_gpu.png)
![selector × matrix, IPM CPU](latest/figures/selector_ipm_cpu.png)
![selector × matrix, IPM GPU](latest/figures/selector_ipm_gpu.png)
![selector × matrix, SuiteSparse CPU](latest/figures/selector_suitesparse_cpu.png)
![selector × matrix, SuiteSparse GPU](latest/figures/selector_suitesparse_gpu.png)

**Why the selector changes the solve so much — SpTRSV level count.** The triangular
solve is scheduled in level sets (each level is a batch of columns with no remaining
dependencies), so the back-solve *level count* is the length of the critical path —
fewer levels = a shallower factor = a faster SpTRSV. Dumping the per-`(selector,
matrix)` level count (`APXCHOL_LEVEL_DUMP`, vec_pool, t16) shows exactly the structure
behind the historical timing heatmap: the retired `root+tree` built a near-degenerate, very deep tree on the
dense social/citation graphs (coPapersDBLP **432,943** levels vs `bg`'s 4,956 — an 87×
deeper critical path; as-Skitter 21,530 vs 7,214; grid3d 14,664 vs 1,368), which is why
its SpTRSV explodes there. Conversely `bk+tree` deepens the structured grids and IPM
ladders (iter0010 1,834 vs `bg`'s 156, a 12× blow-up). `bg`/`greedy` stay shallow on every
family — the level count is the single best predictor of the selector's solve cost.

![apxchol selector × matrix, SpTRSV back-solve level count](latest/figures/selector_levels.png)

**De-singularization mode (per solver):** see the [Protocol](#protocol). The original
singular `L` is benchmarked; apxchol / BoomerAMG / AMGCL / RCHOL ground its null space
without changing the *scored* operator (mean-centring / Dirichlet pin / Dirichlet pin /
εI-preconditioner-only) and are scored on the true `‖b − L x‖`. ParAC is pin-grounded
(no longer `εI`) and — via the patched driver's per-component-consistent, pin-zeroed RHS —
is now **scored on the true `‖b − L x‖` against the original `L`** too (Protocol); only CMG
still keeps `L + εI`, `ε = 1e-6·mean|diag|`.

## Poster figures (CPU)

Two side-by-side heatmaps over **one shared, representative column set** — `grid 2D
(25M)`, `grid 3D (15.6M)` (the largest grids tested), IPM (geomean over the iter
ladder), and a popular SuiteSparse spread ordered structured→irregular that includes
both where apxchol **loses** (`G3_circuit`, `thermal2`, the `kron_g500` scale-free
graph, `coPapersDBLP`) and where it **wins** (`coAuthorsDBLP` + the social giants).
CPU, t16, **total time**, tol 1e-8. Colour normalises *per column* on a **log scale**
(green = fastest for that graph; `1×…16×` colorbar), each cell shows the absolute
seconds over its `×-best` ratio, **bold** = per-column winner. A timed-out cell with
schema-2 provenance is drawn at its exact persisted **`≥cap`** lower bound with a
black border; an old timeout without that field is labelled `Timeout, cap unknown`.

The first answers *which IS selector* to use; the second compares the declared
`apxchol/bg` default with the multigrid / Cholesky field. Same columns, so they read
together.

![apxchol IS-selector × graph, CPU total time](latest/figures/poster_selectors_cpu.png)
![apxchol vs multigrid/Cholesky, CPU total time](latest/figures/poster_comparison_cpu.png)

**Headline:** the algebraic multigrids (AMGCL, BoomerAMG) win the structured grids, FEM
and IPM ladders, while **apxchol wins the irregular social/scale-free giants**, where
classical-AMG hub coarsening is expensive (BoomerAMG 212 s on com-Youtube, 440 s on
com-LiveJournal vs apxchol's 1.6 s / 32 s; see the tables). On CPU the IS-selector choice is a near-wash
(robust); the dramatic selector spread is on the GPU SpTRSV and in the level-count chart
above. `RCHOL` times out (`≥`) on the grids and the giants.

**One row carries a wall-clock caveat** (read its *iteration count*, not its bar):
**CMG (†)** is **serial + MATLAB-runtime** (no GPU/threads — see the methods table), so
its wall-clock isn't comparable to the 16-thread C++ solvers even though its iteration
quality is excellent (25-28 on grids vs apxchol's ~45). **ParAC** is now scored against the
original `L` like every in-house solver (per-component-consistent RHS, Protocol), so its
bars *are* comparable — its physics driver is the per-component split on the pure Laplacian.

## Headline findings

- **On grids, multigrid wins.** Fastest on **total time** is **AMGCL with the Dirichlet
  pin** (e.g. grid_3000 ≈4.3 s vs BoomerAMG ≈9.3 s, apxchol ≈10.8 s); **BoomerAMG** has
  the **fewest iterations** (7–11, near-constant in size) but a heavier setup. The
  Cholesky-type solvers cluster behind — **apxchol strongest of them** (at `apxchol/bg`,
  its declared default; the other selectors are their own rows).
  PCG iters on grids: BoomerAMG 7–11 ≪ AMGCL 12–15 / CMG 25–28 ≪ RCHOL 22–40 (only the
  three smallest grids completed) ≈ apxchol 28–51 < ParAC 34–74.
- **On IPM (SDDM), BoomerAMG leads** (10 iterations on every rung, to true 1e-8);
  apxchol tracks it within ~25% on total time (and wins `iter0010`); the other
  Cholesky-type solvers trail it by 1.5–9× (ParAC 1.5–3.5×, pRCHOL 2–3×, RCHOL 5–9×).
  Among the historical apxchol configs the IPM lead was
  **`bg`/`greedy`** (median total 1.42 s each vs retired `root` 1.47 s, `bk` 1.86 s); on the GPU
  axis `greedy` takes IPM (1.16 s vs `bg` 1.31). No selector generalizes: on the CPU
  medians over the matrices every selector completed, `bg` leads grids (7.78 s) and
  SuiteSparse (1.15 s) with `root` third there (8.50 s / 1.25 s) and `bk` last
  everywhere. Pick per graph type — see the selector heatmaps above.
- **On the scale-free / social graphs (com-Amazon, coAuthorsDBLP), apxchol is fastest** —
  the regime where the Cholesky-type structure pays off (com-Amazon: apxchol ≈0.40 s vs
  AMGCL ≈0.94, BoomerAMG ≈1.12; coAuthorsDBLP: 0.40 vs 1.99 / 2.49).
- **Nobody scales well with threads** (see the thread charts): at 16 cores the best
  is BoomerAMG/apxchol at ~2–3×; RCHOL/ParAC are flat and pRCHOL can go *negative*.
  The work is bandwidth-bound with serial bookends (AMD reorder, deep-factor SpTRSV).
- **pRCHOL** (parallel RCHOL) beats serial RCHOL on **every matrix where both
  completed** — 1.6–5.8× on total time (grid_500 5.8×, com-Amazon 5.0×,
  coAuthorsDBLP 4.8×, the IPM ladder 2.4–2.8×, grid3d_100 1.6×). The win is in the
  **solve**, not the factorization: its ND ordering usually costs a few extra PCG
  iterations (e.g. grid_1000 57 vs 40) but gives a parallel triangular solve
  (grid_1000 2.0 s vs 5.7 s). It is not a free win at scale — pRCHOL still times out
  wherever serial RCHOL does (the large grids and most giants), and it is the only
  one of the two that finishes `coPapersDBLP`.
- **ParAC** factorizes fast (parallel dependency-DAG over the AMD order) but its
  triangular solve dominates; AMD reordering (author-recommended) is required for
  competitive performance and is counted in its setup time.
- **The GPU-resident PCG is apxchol's fastest solve.** Keeping all PCG vectors on
  device (measured then with cuBLAS + cuSPARSE; the loop is our own kernels since 2026-08-18) is **2-4.5× faster per iteration**
  than the host-PCG hybrid — ~13 ms/it vs ~59 ms/it on `grid_2000`, ~6.5 vs ~23 on
  `G3_circuit` (RTX 4090 Laptop, locked clocks). The host path does its SpMV on the
  contended CPU cores, so it is both slower and far noisier; GPU-PCG is now the only
  charted `apxchol (GPU)` config.

The breakdown charts make the cost split explicit: ParAC's and RCHOL's *solve*
dominates (many iterations × deep triangular solve), while multigrid has a tiny solve.

## Open / not yet covered

- The rectangular grid; boost-off (locked-frequency) "published" tier.
- VRAM heatmaps (`vram_peak`/`vram_solve`, GPU axis) need a re-sweep with the
  instrumented CUDA build to populate; CPU cells stay unmeasured by design.
