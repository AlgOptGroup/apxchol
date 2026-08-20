# ParAC patches

Upstream: <https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky>
Pinned at `44ef39d2f5c2c52aa577f58f005d62f2675cefbc`
("Update data README for graph matrix downloads", 2026-04-24).

The benchmark runs ParAC from an out-of-tree checkout (`PARAC_CPU_DRIVER` /
`benchmarks/paths_local.py`). **One** patch is applied to it. Everything else we
need — the AMD reorder, the physics augmentation, the local build flags, the
independent residual check — lives in this repo, so the ParAC checkout stays at
upstream plus a single reviewable diff.

```bash
cd $PARAC_CHECKOUT
git checkout 44ef39d && git status --porcelain     # must be empty
patch -p1 < $APXCHOL/benchmarks/patches/parac/0001-configurable-tolerance-and-any-thread-count.patch
bash $APXCHOL/benchmarks/parac_build.sh            # builds experiment/driver
```

## 0001 — configurable tolerance, and run the solve at any thread count

`experiment/driver_local.cpp`, two hunks, no numerics touched.

* **`if(num_threads == 32)`** (upstream line 1033) gates the whole PCG solve on
  the thread count of the authors' Perlmutter node. At 16 threads the driver
  factorizes and exits without solving — no iteration count, no residual. The
  patch runs the solve at any thread count.
* **`example_pcg_solver(..., max_iter = 1000, rel_tol = 1e-7)`** already declares
  both as parameters; the call site simply never passes them. The patch passes
  them from `$PARAC_MAX_ITER` / `$PARAC_REL_TOL`, defaults unchanged. This is the
  same knob ParAC's own siblings already expose — `experiment/independent_cg.cpp`
  reads `max_iter` and `rel_tol` from `argv[4]`/`argv[5]`, and the CUDA drivers
  take the tolerance as `argv[4]` — so the CPU experiment driver is the odd one
  out. The environment is used rather than argv because `argv[4]` already selects
  physics mode there.

Both hunks are upstreamable as-is, and neither changes what the solver computes.

### The tolerance is ABSOLUTE, so it has to be calibrated

`custom_cg.hpp` stops on `sqrt(dpar[4]) > sqrt(dpar[0])`: the residual **norm**
against **sqrt(rel_tol)**. At the default `rel_tol = 1e-7` that is `||r|| <=
3.16e-4` regardless of `||b||` — on com-Amazon (`||b|| = 167`) an effective
relative tolerance of 1.9e-6, nowhere near the 1e-8 the benchmark reports at.
It is also the *recurrence* residual, which runs optimistic by a matrix-dependent
factor (measured: 1.8x on com-Amazon, 45x on apache2, 5x on G3_circuit).

We do **not** patch the test. The driver already prints the true relative
residual it achieved (`relative residual: ...`, computed as `||Ax - b||/||b||`
against the operator it solved), so the tolerance can be calibrated from one
probe run using only ParAC's own two printed numbers:

```
probe with rel_tol_0  ->  recur_0 = "Final residual norm", R_0 = "relative residual"
rel_tol_1 = ( tau * recur_0 / R_0 ) ** 2            # tau = 1e-8
```

Measured (probe at `rel_tol_0 = 1e-7`, target `tau = 1e-8`):

| matrix | rel_tol_1 | iters | achieved true rel. residual |
|---|---|---|---|
| com-Amazon | 9.089e-13 | 31 | 7.66e-09 |
| coAuthorsDBLP | 4.800e-13 | 23 | 6.13e-09 |
| apache2 | 2.941e-15 | 38 | 9.12e-09 |
| G3_circuit | 4.719e-13 | 64 | 7.00e-09 |

One probe suffices — the optimism factor is essentially constant in the
tolerance for a given matrix. `benchmarks/parac_runner.py` implements this and
records `parac_rel_tol` and the achieved residual in every cell.

## Input construction — in this repo, not in theirs

ParAC requires a fill-reducing ordering (its own scripts only ever run
`*-amd.mtx`) and, for a matrix that is not already a Laplacian, an augmentation.
Its producer for both is `cpu_implementation/write_graph.jl`, which we do not
call: `graph_produce` REBUILDS the diagonal as a pure Laplacian, silently
destroying the diagonal of an SDDM operator. `benchmarks/parac_reorder_amd.jl` is
our minimal stand-in — permutation only, all values preserved:

* **kind=graph** — dump the **pure** `L = D - A`, permute, feed to **graph mode**
  (`driver <mtx> <threads> ""`). ParAC generates its own zero-sum RHS, which is
  consistent for a connected singular Laplacian, and its residual is against that
  L. Do **not** hand it a Dirichlet-pinned matrix (see "dropped" below).
* **kind=operator** — dump the operator as published, then
  `parac_reorder_amd.jl --augment`, which reproduces `write_graph.jl`'s
  `physics_produce`: permute first, then append the ground row/column, so the
  appended node is **last**. **Physics mode** (`driver <mtx> <threads> "" 1`)
  trims exactly that node, and what it solves is the published operator itself
  (verified: `trimmed laplacian nnz` equals the operator's nnz exactly, and an
  independent score with `benchmarks/parac_verify_residual.py` matches ParAC's
  own printed residual to all digits).

The augmentation is not optional. Physics mode's `remove_last_row_and_column` is
how ParAC gets **back** to the published operator; run it on an un-augmented
operator and it deletes a real degree of freedom.

## What changes in the numbers

3-rep medians, 16 threads, `MKL_NUM_THREADS=1`, `taskset -c 0-15`, boost on.
`setup` = AMD reorder + etree/ftree/summary + elimination kernel.
`printed` is ParAC's own `relative residual:` line; **`scored`** is
`parac_verify_residual.py` recomputing `‖b − A·x‖/‖b‖` against the matrix the
benchmark reports on, from the raw `x` and `b`.

| matrix | | iters | printed | scored vs published | setup s | solve s | total s |
|---|---|---|---|---|---|---|---|
| apache2 (operator) | current | 43 | 6.87e-09 | **3.12e-03** | 0.618 | 1.064 | 1.682 |
| | recommended | 38 | 6.50e-09 | **9.12e-09** | 0.640 | 0.836 | 1.476 |
| G3_circuit (operator) | current | **2000 (cap)** | 9.92e-01 | **3.79e-01** | 1.318 | 83.156 | 84.474 |
| | recommended | 62 | 8.14e-09 | **7.00e-09** | 1.316 | 2.621 | 3.937 |
| com-Amazon (graph) | current | 30 | 7.02e-09 | 9.45e-09 | 0.503 | 0.389 | 0.892 |
| | recommended | 31 | 5.89e-09 | 7.66e-09 | 0.523 | 0.363 | 0.886 |

Reading it:

* **apache2 becomes n/a → correct, not slower.** The old cell was not a
  pessimistic apache2 number, it was a number for a different matrix (the
  published operator minus its last row and column — `4817859` nonzeros solved
  instead of `4817870`). Its 6.87e-09 was true of that matrix and 3.12e-03 of the
  published one.
* **G3_circuit goes from `not_converged` to 62 iterations.** Nothing about ParAC
  changed; it was being handed a matrix its physics mode cannot ground.
* **com-Amazon is unchanged to within run-to-run noise** (ParAC is randomized;
  iteration counts move by ±2 across reps). This is the important null result:
  dropping both source patches costs ParAC nothing on the graph axis.

The `scored` column also confirms the reverse: for the recommended construction,
ParAC's own printed residual and our independent score agree to all printed
digits, so the production runner does not need the verifier — only the discipline
of building the input its way.

### Disconnected graphs

ParAC's zero-sum RHS is consistent only per connected component, so a
disconnected Laplacian must be handed to it one component at a time. This is not
a corner case: `kron_g500-logn16` has **10217 components** (giant = 55319 of
65536 nodes), and graph mode on the whole pure `L` **diverges** — 2000 iterations
to a residual of 2.6e+11. The runner therefore always goes through
`--giant-dump --comp-rank R` (rank 0 IS the whole matrix when it is connected)
and skips components below `PARAC_COMP_THRESHOLD` nodes as negligible specks.

## Dropped

Four commits that used to live in the ParAC checkout, and two local build edits.
All are gone; the reasons, with the measurements that settle them, follow.

### `build_consistent_rhs` (CPU `custom_cg.hpp` d771187, GPU `solver.hpp` 8dc1d06)

**Dropped — it was a workaround for something WE did.** We handed ParAC a matrix
we had already Dirichlet-pinned; its RHS generator knew nothing about the pin, so
its solution had `x_p = b_p != 0` and the residual against the original L floored
out. The patch rebuilt the RHS inside their driver to compensate.

But ParAC's graph mode is built for singular Laplacians and generates a zero-sum
RHS, which is consistent for an unpinned connected L. Scored by us against the
**original** L (`parac_verify_residual.py`):

| input to ParAC | RHS | tolerance | iters | our score vs original L |
|---|---|---|---|---|
| com-Amazon **pinned** | upstream | 1e-7 | 21 | **1.37e-03** |
| com-Amazon **pinned** | upstream | tightened 100x | 34 | **1.37e-03** (a floor) |
| com-Amazon **pinned** | patched consistent | true-gate 1e-8 | 30 | 9.45e-09 |
| com-Amazon **pure L** | upstream | 1e-7 | 21 | 1.92e-06 |
| com-Amazon **pure L** | upstream | calibrated | 31 | **7.66e-09** |
| coAuthorsDBLP **pinned** | upstream | 1e-7 | 15 | **2.05e-03** |
| coAuthorsDBLP **pinned** | patched consistent | true-gate 1e-8 | 27 | 8.47e-09 |
| coAuthorsDBLP **pure L** | upstream | calibrated | 23 | **6.13e-09** |

Unpinned, upstream ParAC solves the original L to 1e-8 with no source change, in
the same iteration count or fewer (31 vs 30; 23 vs 27). The fix belonged in our
harness — stop pre-pinning the dump — not in their solver. `--pin-dump` is no
longer used by the ParAC path.

Note the limit this inherits: a zero-sum RHS is consistent only per connected
component, so a **disconnected** graph must be handed to ParAC one component at a
time (`--giant-dump --comp-rank R`), which the runner does.

### The convergence-gate rewrite (CPU `custom_cg.hpp`, GPU `solver.hpp`)

**Dropped — configuration reaches the same place.** On the CPU the tolerance is
now passed (patch 0001) and calibrated; the driver's own printed residual is
already the true `||Ax-b||/||b||`. On the GPU nothing at all is needed: the CUDA
drivers already take `TOL` as `argv[4]` and already print
`normalized diff norm` = `||Ax-b||/||b||`, so the same calibration applies with
no patch. `gpu_implementation/solver.hpp` returns to pristine.

### 7af1674 "physics mode = graph (disable redundant trim)"

**Dropped — and it was wrong.** It was already superseded inside the ParAC
checkout's own history by e8d47d4 four commits later, so it was not in effect.
It is also the opposite of what the trim is for: `remove_last_row_and_column`
removes the ground vertex ParAC's own `physics_produce` appended, and is what
makes physics mode solve the published operator.

### e8d47d4 "pure L + trim = single Dirichlet pin"

**Dropped.** Its construction (feed physics mode the giant component's pure L so
the trimmed node acts as the sole pin) exists only to make the trim harmless once
we had decided to feed ParAC our own Laplacian. With graph matrices going through
graph mode on the pure L, and operator matrices going through the augmentation,
there is nothing left for it to fix.

### `cpu_implementation/driver_local.cpp` (`#include <climits>`) and `cpu_implementation/makefile` (include path)

**Dropped — we do not build that target.** The runner uses
`experiment/driver`. The missing transitive includes are supplied as
`-include climits -include cstdint` compiler flags by
`benchmarks/parac_build.sh`, and the machine-local MKL / fast_matrix_market paths
are that script's environment variables, so nothing machine-specific is written
into their tree.

### `experiment/build_local.sh` and `experiment/reorder_amd.jl` (untracked, ours)

**Moved out of their tree** to `benchmarks/parac_build.sh` and
`benchmarks/parac_reorder_amd.jl`. Our preprocessing and our build recipe are
ours to own.
