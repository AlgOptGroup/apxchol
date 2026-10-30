# Changelog

## Unreleased (since v0.2.2)

### Solver and factorization
- New trace-cycle clique sampler (`--sampler trace_cycle`) on CPU and GPU; GKS stays the default.
- Faster setup: one AoS graph throughout factorization, incremental degrees, and parallel residual sparsification.
- Inputs are validated by operator class; nearly-SDDM inputs are repaired by M-matrix lumping.
- Batched solves take any number of right-hand sides on CPU, CUDA and Metal, processed in panels of up to eight columns on CPU and four on GPU.
- `apxchol::solver` owns a reusable factorization for repeated solves and can be saved to and loaded from portable archives.

### GPU backends
- CUDA setup, factorization and PCG run entirely on the device, with a sync-free dataflow triangular solve and optional fp16 factor storage.
- New Apple Metal backend: setup, factorization and PCG run on the device, for single and batched solves.
- `solve_options.backend` (`--backend auto|cpu|gpu`) selects a complete CPU or GPU route and never falls back silently.

### API and CLI changes
- Breaking: runtime environment variables are replaced by explicit solver options.
- Breaking: the CUDA build no longer uses cuSPARSE or cuBLAS, and the level-set GPU triangular solve is removed.
- Breaking: forward-star and indexed pooled graph storage and the standalone analysis/setup tools are retired.
- Breaking: `APXCHOL_INDEX_WIDTH=32|64_EDGES|64` replaces the `APXCHOL_64BIT_*_INDICES` build options.
- Breaking: `apxchol::cpu_solver` is removed; use `apxchol::solver` with `solve_backend::cpu`.
- Breaking: `save_solver`/`load_solver` are replaced by `solver::save` and `solver::load`.
- Breaking: `apxchol::apx_cholesky` is removed from the public API; use `apxchol::solver` (`apply` inside an external Krylov solver).
- Breaking: `apxchol::metal_solver` is removed from the public API; use `apxchol::solver` with `solve_backend::gpu`, and `backend_available(solve_backend::gpu)` instead of `metal_solver::available()`.
- Breaking: the triangular solves, CUDA PCG, allocator, graph storage pools, elimination workspace, built-in partitioners and their helpers, the partitioner name list and operator validation (`operator_scan`, `require_operator`, `operator_view`, ...) move to `apxchol::detail`.
- Breaking: the `factorize(graph, ...)` overloads are removed; pass the operator matrix to `factorize`.
- Breaking: the partitioner interface no longer exposes `selection::reset`/`finalize`, `partition_result` or `partition_context::cp`.
- Breaking: `factorization` drops `peak_graph_bytes` and the diagnostic `round_stats` fields `avg_deg`, `nnz_added` and `nnz_total`; its CUDA device factor is private.
- Breaking: `solve_result::lumped_offdiag` is removed; read `solver.factor().lumped_offdiag`.
- Breaking: `solve_options::keep_factor_values` is renamed `keep_factor`, matching Python and MATLAB/Octave.
- Breaking: the CLI and bindings no longer accept an adjacency matrix as if it were the operator. Previously both solved adjacency input as the operator and failed without a clear error; now validation rejects it and the error suggests the conversion.
- New `apxchol.as_operator(M)`, `apxchol_as_operator(M)` and CLI `--convert-input`: a valid operator is used unchanged, a matrix with no negative off-diagonal entry becomes its graph Laplacian `L = D - M` (self-loops cancel), and anything else is an error.
- The CLI solves an `n` by `k` `--rhs` file as one batch, writes an `n` by `k` solution with `-o`, and logs iterations and residuals per column.
- The v1 API compatibility scope is documented.
- The partitioner and eliminator extension interfaces are experimental in 1.x and may change in minor releases.
- Python wheels for CPython 3.11+ on Linux and macOS arm64 (GPU wheels select the backend at runtime), standalone CLI archives, and MATLAB/Octave batched solves with save/load.
- Wheels declare `License-Expression: BSD-3-Clause` and take the root LICENSE at build time; minimum scikit-build-core is 0.11.
- Relicensed from BSD-4-Clause to BSD-3-Clause.
- Breaking: `solver::solve()` and `solver::apply()` are non-const because they use the solver's workspace; concurrent non-const calls on one solver are not allowed.

### Benchmarks
- The benchmark harness checks every result against the original system's residual and reports setup, solve and memory separately.
- Published results now come from CSCS Daint; laptop results are archived.

### Fixes
- Reported residuals are now correct on early solve exits.
- Fixed SDDM misclassification of exact Laplacians.
- CPU solves rescale right-hand sides of extreme magnitude; operators whose scale is not representable in FP32 factor storage are still rejected.
- Parallel factorization handles a shrinking OpenMP team, and parallel vertex selection on the same graph is reproducible at a fixed thread count.
- Concurrent calls on one solver no longer corrupt each other's results: a call made while another thread uses the same `solver` throws. CPU solvers can be copied cheaply, sharing the factor, for one solver per thread (`copy.copy` in Python), and Python releases the GIL during factorization, `solve` and `apply`.

Full history: https://github.com/AlgOptGroup/apxchol/compare/v0.2.2...main
