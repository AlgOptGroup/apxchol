# ParAC benchmark adapter

Upstream [Parallel-Randomized-Cholesky](https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky),
pinned `44ef39d2f5c2c52aa577f58f005d62f2675cefbc`. The benchmark owns preparation,
configuration and grading; six patches expose existing controls, complete timing
boundaries, one producer reduction fix and Graph preparation accounting.

## Patch stack

| Patch | Purpose |
|---|---|
| [0001](0001-configurable-tolerance-and-any-thread-count.patch) | Remove the CPU solve’s 32-thread gate; expose `PARAC_REL_TOL`/`PARAC_MAX_ITER`; print RHS norm for component-weighted residuals. Defaults and stopping law remain upstream. |
| [0002](0002-report-complete-setup-boundaries.patch) | CPU post-parse adapter and complete factor setup intervals, including scheduling/materialization omitted by narrower upstream timers. |
| [0003](0003-report-complete-gpu-boundaries.patch) | Complete CUDA host preparation, allocation, transfers, factorization and solve intervals; honor calibrated tolerances below 1e-8. |
| [0004](0004-report-cuda-init-separately.patch) | Prewarm and report process-wide CUDA initialization separately. Solver-specific allocations, module loading and transfers remain charged. |
| [0005](0005-compensate-physics-global-sum.patch) | Neumaier compensation for the Physics global-sum branch decision. Preserve both ±1e-9 thresholds, ordering, per-column sums and sampling. |
| [0006](0006-separate-graph-algorithm-preparation.patch) | Share the existing Graph producer with an in-memory entry; return disjoint transform, ordering/permutation, serialization and cleanup intervals. |

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

The manual build checks for both the compensated Physics sum and the in-memory
Graph producer. Driver/cache paths
are configured through `PARAC_CPU_DRIVER` and ignored `benchmarks/paths_local.py`;
build dependencies/flags live in [parac_build.sh](../../parac_build.sh), not
machine-specific upstream edits. The bundled CPU driver requires MKL. Daint’s
separately pinned `portable-cpp` comparison has a serial solve; requested threads
apply to factorization. Do not label that curve parallel PCG.

## Inputs and preparation

[parac_runner.py](../../parac_runner.py) calls upstream `write_graph.jl` through
[parac_produce_upstream.jl](../../parac_produce_upstream.jl), writing to our cache.
CPU Graph/AMD loads the common input before the preparation timer, then calls
patch0006's in-memory entry. Charged algorithm preparation includes the unchanged
graph transform, AMD, materializing `G[p,p]`, explicit final GC and remaining
producer work. Final MatrixMarket serialization has its own measured interval and
is excluded from the algorithm charge. Automatic GC follows the interval where it
occurs. This is fresh interval accounting, not subtraction from old measurements.

The versioned receipt reconciles complete preparation as algorithm + audit +
serialization; the runner defines and validates its fields. This entry performs
no separate original-system audit or permutation. Graph permutation is charged
inside ordering; adapters that also compute `A[p,p]` and `b[p]` must charge that
work separately.

Source/input/output identities and the accounting schema invalidate old Graph/AMD
caches. Cached preparation reuses its measured algorithm charge at every T;
complete/inclusive preparation and setup remain diagnostics. Missing, malformed
or inconsistent new receipts cannot become corrected records. True upstream
refusal may retain the explicitly labelled complete-timing fallback. Physics and
GPU `nnz-sort` retain their complete producer accounting; component extraction/dump
charges remain unchanged because their required computation and interchange have
not been separated here. Native adapter, full factor and workspace setup remain
charged. Eligibility, tolerance calibration and residual checks are unchanged.

The maintained Graph/AMD path passed a pinned Daint Julia smoke covering producer
semantics, preparation/cache accounting, calibration and original-system residuals
with the unchanged portable CPU driver. This does not validate the stock MKL,
Physics or GPU routes or establish a performance result. Published scaling cells
retain their private adapter provenance; they are not measurements of this port.
Upstream isolate-dimension behavior remains unchanged, and runtime/RSS identity
with the private original-system adapter is not claimed.

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
