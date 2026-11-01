# Changelog

## Unreleased (since v0.2.2)

### Solver and factorization
- New trace-cycle clique sampler (`--sampler trace_cycle`) on CPU and GPU; GKS stays the default.
- Faster setup: one AoS graph throughout factorization, incremental degrees, and parallel residual sparsification.
- Inputs are validated by operator class; nearly-SDDM inputs are repaired by M-matrix lumping.
- Batched solves take any number of right-hand sides on CPU, CUDA and Metal, processed in panels of up to eight columns on CPU and four on GPU.
- `apxchol::solver` owns a reusable factorization for repeated solves and can be saved to and loaded from portable archives.
- Any 1.x release reads the solver archives written by earlier 1.x releases. A minor release may add a format version and keeps reading older ones; an older apxchol given a newer archive fails with an upgrade message.

### GPU backends
- CUDA setup, factorization and PCG run entirely on the device, with a sync-free dataflow triangular solve and optional fp16 factor storage.
- New Apple Metal backend: setup, factorization and PCG run on the device, for single and batched solves.
- Metal backend and macOS support contributed by Samuel Talkington (@samtalki).
- GPU PCG waits for the device once every four iterations on CUDA (instead of every iteration) and every eight on Metal (instead of four), and skips the reductions of iterations after convergence; solutions are unchanged, and small solves are up to 15% faster on CUDA and 7% on Metal.
- `solve_options.backend` (`--backend auto|cpu|gpu`) selects a complete CPU or GPU route and never falls back silently.
- CUDA and Metal solvers can be copied: copies share the device factor and operator, and solve concurrently on their own streams or command queues and workspace.

### API and CLI changes
- Breaking: the C++ one-shot `solve(A, b, tol, max_iter, x0, opts)` takes an optional initial guess before the options, like the reusable solver, Python and MATLAB/Octave; the CLI reads one with `--x0 FILE`, shaped like the right-hand side.
- Breaking: runtime environment variables are replaced by explicit solver options.
- Breaking: the CUDA build no longer uses cuSPARSE or cuBLAS, and the level-set GPU triangular solve is removed.
- Breaking: forward-star and indexed pooled graph storage and the standalone analysis/setup tools are retired.
- Breaking: `APXCHOL_INDEX_WIDTH=32|64_EDGES|64` replaces the `APXCHOL_64BIT_*_INDICES` build options.
- Breaking: `APXCHOL_DEVICE=CPU|CUDA|METAL` replaces `APXCHOL_USE_CUDA`, which now stops configuration with a message. The runtime `factor_storage` option replaces `APXCHOL_SPTRSV_FP32`.
- Breaking: a parent project that adds apxchol with `add_subdirectory` builds only the portable library by default; the CLI, tests, examples and native tuning are opt-in there.
- Breaking: the Python default `maxiter` is now the native default, 200 (was 500).
- Breaking: Python `Solver.solve` and `solve` use `rtol` for the tolerance parameter; the `tol` alias is removed.
- Breaking: removed from Python: `apxchol.solver`, `Solver.aspreconditioner` (use `aslinearoperator`), the `storage` and `stagnation_window` options, and the `luby` and `rootset` partitioners.
- Breaking: Python 3.10 is no longer supported.
- Breaking: removed CLI flags `--stagnation-window`, `--graph-storage`, `--fs-compact` and `--fs-filter-append`.
- `apxchol::core` no longer passes its `-march` flag to consumers, and a `-march`/`-mcpu` already in `CMAKE_CXX_FLAGS` or `CMAKE_CXX_FLAGS_<CONFIG>` replaces apxchol's own. Consumers get `EIGEN_MAX_ALIGN_BYTES` and `EIGEN_MALLOC_ALREADY_ALIGNED=0` instead, so Eigen objects cross the library boundary safely whichever instruction set each side targets. Projects that embed apxchol by including `cmake/apxchol_core.cmake` now default to portable code, like `add_subdirectory` embedding; standalone root, Python and benchmark builds stay native.
- Breaking: `solve_result::solve_vram_mb` is removed. It reported device-wide memory in use, including other processes, so it did not measure the solve.
- Breaking: C++ `solve()` takes `std::optional` tolerance and iteration budget; `std::nullopt` selects the defaults, and a negative value now throws `std::invalid_argument` instead of selecting them.
- Breaking: `solve_options::tol` and `solve_options::max_iter` are removed; the tolerance and iteration budget are solve arguments in every interface, and an omitted one uses `default_tol` or `default_max_iter`. The one-shot C++ call becomes `solve(A, b, tol, max_iter, x0, opts)`.
- Breaking: the iteration count is `iterations` in every interface; Python `SolveResult.iters` and the MATLAB/Octave result field `iters` are renamed.
- Breaking: the Python option and property `factor_storage_dtype` are renamed `factor_storage`; the option accepts `"auto"` (the default), `"float16"`, `"float32"` or a NumPy dtype.
- Breaking: MATLAB/Octave `apxchol_solve(A, b, rtol, maxiter, x0, options)` takes the initial guess before the options, in the order of `s.solve`.
- Breaking: errors use two classes in every interface: `std::invalid_argument` (Python `ValueError`, MATLAB/Octave `apxchol:badInput`) for invalid input, options and unsupported requests, and `std::runtime_error` or a subclass (`RuntimeError`, `apxchol:error`) for execution failures. Metal input checks now throw `std::invalid_argument` instead of `std::domain_error` or `std::length_error`, an edge-index overflow throws `std::overflow_error` instead of `std::length_error`, Python no longer raises `OverflowError` or `ValueError` for execution failures, and MATLAB/Octave `apxchol:badMatrix`, `apxchol:badVector`, `apxchol:tooLarge` and `apxchol:badHandle` become `apxchol:badInput`.
- Breaking: `apxchol::cpu_solver` is removed; use `apxchol::solver` with `solve_backend::cpu`.
- Breaking: `save_solver`/`load_solver` are replaced by `solver::save` and `solver::load`.
- Breaking: `apxchol::apx_cholesky` is removed from the public API; use `apxchol::solver` (`apply` inside an external Krylov solver).
- Breaking: `apxchol::metal_solver` is removed from the public API; use `apxchol::solver` with `solve_backend::gpu`, and `backend_available(solve_backend::gpu)` instead of `metal_solver::available()`.
- Breaking: the triangular solves, CUDA PCG, allocator, graph storage pools, elimination workspace, built-in partitioners and their helpers, the partitioner name list and operator validation (`operator_scan`, `require_operator`, `operator_view`, ...) move to `apxchol::detail`.
- Breaking: the `factorize(graph, ...)` overloads are removed; pass the operator matrix to `factorize`.
- Breaking: the partitioner interface no longer exposes `selection::reset`/`finalize`, `partition_result` or `partition_context::cp`.
- Breaking: `factorization` drops `peak_graph_bytes` and the diagnostic `round_stats` fields `avg_deg`, `nnz_added` and `nnz_total`; its CUDA device factor is private.
- Breaking: `solve_result::lumped_offdiag` is removed; read `solver.factor().lumped_offdiag`.
- Breaking: `solve_options::keep_factor` and the Python and MATLAB/Octave `keep_factor` option are removed; `solver::factor()`, Python `chol()`/`P`/`factor_nnz` and `save()` use the installed factor, the preconditioner actually used after dropping and storage rounding.
- Breaking: `solver::factor()` returns a `factorization` by value, with new `storage` and `scale` members describing FP32 or scaled FP16 storage; adopting such a factor reinstalls it exactly.
- Breaking: Python `chol()` returns a float64 matrix; for a Laplacian the last row and column are empty.
- Breaking: solver archive version 1 stores the installed factor and its storage precision; `load()` takes storage and dropping from the file and reproduces the saved solver's solves bit for bit on the same backend and thread count.
- Breaking: the `APXCHOL_POOL_FP32` build option is removed; graph weights are always stored in FP32, as in default builds.
- Breaking: `apxchol::generate_test_rhs` is removed; build the right-hand side in the caller.
- Breaking: the per-stage timing tree leaves the public API: `checkpoint`, `solve_options::collect_timings`, the `timings` members of `solve_result` and `batch_solve_result`, the `checkpoint*` parameters of the `solver` constructors and `factorize`, and the `solver::solve(b, solve_result&, ...)` overload are removed.
- Breaking: the CLI prints the timing tree only with `-v` and drops `--no-timings`.
- `solver::setup_seconds()`, `solve_result::solve_seconds` and `batch_solve_result::solve_seconds` report wall-clock setup and solve time, also as `Solver.setup_seconds` and `SolveResult.solve_seconds` in Python and as `s.setup_seconds` and `res.solve_seconds` in MATLAB/Octave.
- Breaking: the `APXCHOL_SPTRSV_BACKEND` macro is removed from `apxchol/version.h`; CLI `--version` prints the device (`cpu`, `cuda` or `metal`) instead.
- Breaking: CLI `--random-rhs` uses a new generator, so its values change; they are now the same on every platform for a given `--seed`.
- Breaking: the CLI and bindings no longer accept an adjacency matrix as if it were the operator. Previously both solved adjacency input as the operator and failed without a clear error; now validation rejects it and the error suggests the conversion.
- Breaking: the factor type `sparse_csc` is a read-only view with `col_ptrs()`, `row_indices()`, `values()` and `has_values()`; its storage members, raw-pointer accessors and build mutators are no longer public.
- Breaking: the CMake helpers `apxchol_openmp_runtime()`, `apxchol_configure_core()`, `apxchol_install_core()` and the `APXCHOL_ARCH_FLAG` variable are internal (`_apxchol_*` and `_APXCHOL_ARCH_FLAG`); `find_package(apxchol)` and `apxchol_add_core()` are the supported entry points.
- Breaking: Python `Solver.L`, `Solver.D` and `Solver.fill_ratio` are removed (use `chol()`, `P` and `factor_nnz`), `Solver.lumped` is renamed `lumped_offdiag`, and `apxchol.DEFAULT_TOL`/`DEFAULT_MAXITER` are no longer public.
- Breaking: the raw MATLAB/Octave `apxchol_mex` entry point moves to `octave/private/`; use the `apxchol_*` wrappers.
- Breaking: custom partitioners receive a read-only `apxchol::partition_graph` (`n`, `is_active`, `for_each_neighbor`, `degree`) instead of the mutable graph; `graph`, `make_graph` and `laplacian` move to `apxchol::detail`.
- Breaking: partitioners no longer declare a `name`, and `partitioner_list.h` with `dispatch_partitioner` is removed; pass a custom partitioner as an instance to `factorize`.
- Breaking: `factorize(A, eliminator, opts)` no longer takes a partitioner template argument, and now uses the partitioner `opts.partitioner` names instead of always `block_greedy`; pass an instance to use another partitioner.
- Breaking: `tree_elimination` moves to `apxchol::detail`; configure the built-in rule through `factor_options`.
- Breaking: `sparse_csc::kEdgeMax` is removed; use `std::numeric_limits<apxchol::edge_index>::max()`.
- New `apxchol.as_operator(M)`, `apxchol_as_operator(M)` and CLI `--convert-input`: a finite symmetric matrix with at least one off-diagonal entry, all of them positive, becomes its graph Laplacian `L = D - M` (self-loops cancel), any other finite symmetric matrix is used as given and checked by the solver, and asymmetric or non-finite input is an error.
- The CLI solves an `n` by `k` `--rhs` file as one batch, writes an `n` by `k` solution with `-o`, and logs iterations and residuals per column.
- The v1 API compatibility scope is documented.
- Breaking: from 1.0, the installed CMake package accepts any version request with the same major version (0.x stays minor-scoped), and the shared core library's soname now carries `major.minor` (`libapxchol_core.so.0.3` for this version).
- The partitioner and eliminator extension interfaces are experimental in 1.x and may change in minor releases.
- Breaking: research controls (`omp_threshold`, `min_is_fraction`, `parallel_residual_threshold`, `exact_clique_max_degree`, `exact_core_max_h`, `double_cycle_min_h`, `degree_tiebreak`, `cpu_schedule`) move to `factor_options::experimental`, outside the 1.x compatibility promise, and leave the CLI, Python and MATLAB/Octave.
- Breaking: the `residual_peel` option and its `min_degree` and `bk_serial` strategies are removed; the serial tail always uses the natural order.
- Breaking: `factor_options::is_select` (a string) is replaced by `partitioner`, a `partitioner_kind` enum that option validation checks.
- Breaking: unset `degree_quantile`, `parallel_residual_threshold` and `exact_core_max_h` are `std::nullopt` instead of -1 or `SIZE_MAX`, and the sentinel constants and resolvers such as `exact_core_by_route` and `degree_quantile_or_host_default` are removed.
- Breaking: CLI `--is`, `--factor-drop` and `--maxiter` are renamed `--partitioner`, `--drop-tolerance` and `--max-iter`; `--sampler`, `--factor-storage` and `--partitioner` accept names only.
- Python and MATLAB/Octave accept the same stable setup options, adding `sampler` to Python and `partitioner`, `sampler`, `degree_quantile` and `degree_multiplier` to MATLAB/Octave.
- Python wheels for CPython 3.11+ on Linux and macOS arm64 (GPU wheels select the backend at runtime), standalone CLI archives, and MATLAB/Octave batched solves with save/load.
- Wheels declare a `License-Expression` that matches their contents (BSD-3-Clause, MPL-2.0, Apache-2.0 WITH LLVM-exception and Zlib; Linux CUDA wheels add the NVIDIA CUDA EULA and ship its text) and take the root LICENSE at build time; minimum scikit-build-core is 0.12.
- Relicensed from BSD-4-Clause to BSD-3-Clause.
- Breaking: `solver::solve()` and `solver::apply()` are non-const because they use the solver's workspace; concurrent non-const calls on one solver are not allowed.

### Benchmarks
- The benchmark harness checks every result against the original system's residual and reports setup, solve and memory separately.
- Published results now come from CSCS Daint; laptop results are archived.
- GPU memory is now this process's NVML usage above its post-context baseline, measured the same way for apxchol, AMGCL and Hypre, with a new `setup_vram_mb` column; other compute processes are reported separately and flagged as shared.
- The benchmark builds on macOS with a Metal route for apxchol (`--device metal`), with Metal warm-up excluded from setup and the factor-drop statistics in `--report-fill`.

### Documentation
- The website separates user and developer guides, adds installation, quick start, examples, changelog and citing pages, and marks development pages with a banner.
- The Performance page has a page per platform: Euler (x86_64, CUDA), Daint (ARM64, CUDA) and Apple M1 Max (Metal).
- The precision guide states when results are reproducible and that a release may change results, within the documented precision, for measured speed or memory gains.

### Fixes
- The CLI reports "matrix is empty" and exits 2 for a dense (array) input with no rows or columns, instead of crashing with SIGFPE while reading it.
- Reported residuals are now correct on early solve exits.
- Fixed SDDM misclassification of exact Laplacians.
- CPU solves rescale right-hand sides of extreme magnitude; operators whose scale is not representable in FP32 factor storage are still rejected.
- Parallel factorization handles a shrinking OpenMP team, and parallel vertex selection on the same graph is reproducible at a fixed thread count.
- Concurrent calls on one solver no longer corrupt each other's results: a call made while another thread uses the same `solver` throws. CPU solvers can be copied cheaply, sharing the factor, for one solver per thread (`copy.copy` in Python), and Python releases the GIL during factorization, `solve` and `apply`.
- The CLI rejects an empty matrix and RHS files with no rows or columns with an error (exit 2); it used to crash.
- Breaking: saving a solver replaces an existing file instead of refusing. It writes `<path>.tmp`, flushes it, renames it into place and syncs the directory, so readers never see a partial archive and the new file survives a power loss, and the new file gets the usual umask permissions (for example 0644) instead of always 0600.

Full history: <https://github.com/AlgOptGroup/apxchol/compare/v0.2.2...main>

0.1.0–0.2.0 were published before the public history was squashed and have no source tag.
