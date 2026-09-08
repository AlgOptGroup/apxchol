# ParAC benchmark adapter

Upstream [Parallel-Randomized-Cholesky](https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky),
pinned `44ef39d2f5c2c52aa577f58f005d62f2675cefbc`. The benchmark owns preparation,
configuration and grading; five patches expose existing controls, complete timing
boundaries and one producer reduction fix.

## Patch stack

| Patch | Purpose |
|---|---|
| [0001](0001-configurable-tolerance-and-any-thread-count.patch) | Remove the CPU solve’s 32-thread gate; expose `PARAC_REL_TOL`/`PARAC_MAX_ITER`; print RHS norm for component-weighted residuals. Defaults and stopping law remain upstream. |
| [0002](0002-report-complete-setup-boundaries.patch) | CPU post-parse adapter and complete factor setup intervals, including scheduling/materialization omitted by narrower upstream timers. |
| [0003](0003-report-complete-gpu-boundaries.patch) | Complete CUDA host preparation, allocation, transfers, factorization and solve intervals; honor calibrated tolerances below 1e-8. |
| [0004](0004-report-cuda-init-separately.patch) | Prewarm and report process-wide CUDA initialization separately. Solver-specific allocations, module loading and transfers remain charged. |
| [0005](0005-compensate-physics-global-sum.patch) | Neumaier compensation for the Physics global-sum branch decision. Preserve both ±1e-9 thresholds, ordering, per-column sums and sampling. |

0005 corrects cancellation that falsely rejected large grids. It does not prove
row-wise diagonal dominance or repair arbitrary operators. Do not reassociate
its error-recovery expressions with fast-math. Historical caches must not be
retroactively labelled as produced by the patch.

## Build

Fresh benchmark FetchContent checkouts apply the stack automatically. For an
external checkout, start at the pinned upstream revision in a fresh directory;
set absolute `APXCHOL` and `PARAC_CHECKOUT` paths:

```sh
cd "$PARAC_CHECKOUT"
for patchfile in "$APXCHOL"/benchmarks/patches/parac/000*.patch; do
  patch -p1 < "$patchfile"
done
bash "$APXCHOL/benchmarks/parac_build.sh"
julia --project="$APXCHOL/benchmarks/julia" -e 'using Pkg; Pkg.instantiate()'
```

The manual build checks for the compensated producer call. Driver/cache paths
are configured through `PARAC_CPU_DRIVER` and ignored `benchmarks/paths_local.py`;
build dependencies/flags live in [parac_build.sh](../../parac_build.sh), not
machine-specific upstream edits. The bundled CPU driver requires MKL. Daint’s
separately pinned `portable-cpp` comparison has a serial solve; requested threads
apply to factorization. Do not label that curve parallel PCG.

## Inputs and preparation

[parac_runner.py](../../parac_runner.py) calls upstream `write_graph.jl` through
[parac_produce_upstream.jl](../../parac_produce_upstream.jl), writing to our cache.
The recorded charge after reusable Julia load/JIT includes ordering and required
transformations **plus avoidable ASCII interchange and verification-only work**;
it is charged at every T, even when cached. It is not all AMD or algorithm-essential
preparation. Logged AMD is 0.721/0.356/32.363s on grid/IPM/Skitter, versus charged
24.837/10.452/58.097s. The difference is not an isolated I/O measurement.

Algorithm-performance comparisons are provisional pending adapter repair and fresh
measurement. Preserve current numbers; do not subtract the whole difference.
The repair must retain required transformations, native factor/workspace setup,
exact operator/permutation checks and every residual grade while separately
reporting interchange and audit costs. Input/timing-schema sidecars bind caches.

- **Graph:** `graph_produce(prefix,"amd")` rebuilds `L=D-A`; it is appropriate
  for graph adjacency or a pure Laplacian, not an arbitrary SDDM diagonal.
  Do not pre-pin it. Connected native graph files avoid redundant serialization.
- **Physics:** `physics_produce(prefix,"amd")` permutes then appends a ground
  node; Physics mode trims that last node. Passing an unaugmented operator would
  delete a real degree of freedom. The producer’s global sum check is not a
  per-row dominance test; clipping ground-edge weights is preconditioner
  handling, not proof of original-operator identity.
- **Eligibility:** positive stored Physics off-diagonals are unsupported before
  cache lookup or fallback because `-abs(weight)` changes the operator. Keep
  unsupported distinct from malformed-input failure. Any fallback preparation
  must retain its explicit source/reason label.
- **Disconnected graphs:** CPU runs each non-singleton component; combine using
  `sqrt(sum ||r_c||²)/sqrt(sum ||b_c||²)`. Singleton null blocks need no solve.
  The CUDA graph route without component handling is unsupported, not converged.

## Calibration, grading and timing

The upstream CPU stop compares recurrence residual norm with `sqrt(rel_tol)`,
not a true relative tolerance. From a valid probe with recurrence norm `r0`
and true relative residual `R0`, set `rel_tol=(tau*r0/R0)^2`, `tau=1e-8`.
This estimates a tolerance; it does not guarantee convergence. Reject failed,
nonfinite or iteration-capped probes without launching retained runs at a fallback
tolerance. A below-cap GPU probe is not proof of recurrence convergence.

Grade every retained solution against the **original A,b**. Selecting one
median-total repetition selects timings only; all retained residuals must pass.
Preserve calibration, warmup and retained denominators separately. Standalone
verification uses [parac_verify_residual.py](../../parac_verify_residual.py)
with the returned solution/RHS and recorded permutation; an upstream residual
against a transformed system alone is insufficient.

Recorded setup = charged producer + complete adapter + factor/workspace setup. Solve
includes required output transfers. Common input reading, independent grading
and process CUDA initialization are separate. Never poll peak VRAM inside timing.
A deadline covering calibration and all repeats is not a per-solve lower bound.

[Historical experiments and rejected adaptations](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/benchmarks/patches/parac/README.md)
retain the older evidence and formulas; they are not current verification.
