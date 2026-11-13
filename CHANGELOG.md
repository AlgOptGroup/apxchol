# Changelog

## Unreleased (since v0.2.2)

### Solver and factorization
- New experimental trace-cycle clique sampler (`factor_options::experimental.sampler`) on CPU and CUDA. GKS stays the default.
- Faster setup: one AoS graph throughout factorization, incremental degrees, and parallel residual sparsification.
- Inputs are validated by operator class; nearly-SDDM inputs are repaired by M-matrix lumping.
- Batched solves take any number of right-hand sides on CPU, CUDA and Metal, processed in panels of up to eight columns on CPU, 32 on CUDA and four on Metal.
- CPU batched solves place the solution and direction panels on the memory of the threads that update them. On a 128-core EPYC 7742 node, solving 32 right-hand sides on 5- and 7-point grids of 3 to 4 million rows takes 21 to 24% less time with 16 threads on four memory domains and 38 to 50% less with all cores. Runs confined to one memory domain are unchanged.
- CPU batched solves are faster on operators whose factor has a dense tail: the forward triangular solve sums the already-available part of each tail row on all threads and no longer leaves the heaviest rows to a single thread. On a 128-core EPYC 7742 node, solving 32 right-hand sides of an operator with 271 entries per row takes 34% less time with 16 threads and 30 to 50% less with 64 or more. Grid solves take the same time or up to 4% less. Panels of 2 or 4 columns can differ from earlier versions in the last digits.
- CPU batched solves carry the triangular sweeps' results in FP32, which halves the bytes each gathered row moves, and scale every column by a power of two relative to the factor's diagonal, as CUDA does. On a 128-core EPYC 7742 node, 32 right-hand sides at 16 threads take 16 to 22% less time on a 5-point grid of 4.0M rows, 20 to 30% less on a 7-point grid of 3.4M rows and 26 to 30% less on an operator with 271 entries per row. At 64 or 128 threads they take 6 to 19% less. Solutions differ from earlier versions in the last digits and iteration totals can rise by about 1%.
- `apxchol::solver` owns a reusable factorization for repeated solves and can be saved to and loaded from portable archives.
- Any 1.x release reads the solver archives written by earlier 1.x releases. A minor release may add a format version and keeps reading older ones; an older apxchol given a newer archive fails with an upgrade message.

### GPU backends
- CUDA solves converge when the operator and the right-hand side have very different magnitudes. Every column is scaled by a power of two relative to the diagonal of the factor, or of the operator when the factor is built on the device. On a 64 by 64 grid, an operator scaled by 1e30 with a right-hand side near 1e-8 stalled at a residual of 2.3e-7 after 200 iterations and now converges, and an operator scaled by 1e-30 with a right-hand side near 1e8 returned a zero solution and now converges. Small right-hand sides on a factor whose diagonal spans 60 decades converge as well. Other inputs keep their results.
- CUDA batched solves move less memory per iteration: 32-column panels are multiplied eight columns at a time, the preconditioned residual stays in the FP32 the triangular solve produces, and the forward triangular sweep writes its result once. A 32-column panel of a dense operator (271 entries per row) takes 30% less time on an RTX 4090, and panels of 5- and 7-point grids take up to 5% less on an RTX 4090 and up to 7% less on an RTX PRO 6000. Results are unchanged.
- CUDA batched solves copy right-hand sides and solutions between the caller and the device panel in parallel on the host when the panel is large enough. On grids of 3 to 4 million rows a 32-column panel takes about 19% less time on an RTX 4090 and 25% less on an A100.
- CUDA batched solves are faster on sparse operators: the vector updates of a panel read memory contiguously at every panel width. On 5- and 7-point grids of 3 to 4 million rows, a 32-column panel takes 15 to 21% less time on an RTX 4090 and 26 to 36% less on an A100. Single right-hand-side solves take the same time as before. CUDA sums are added in a different order, so solutions can differ from earlier versions in the last digits and an iteration count can change by one.
- Metal PCG uses full 32-thread SIMD groups for reductions and long-row multiplication.
- CUDA setup, factorization and PCG run entirely on the device, with a sync-free dataflow triangular solve and optional fp16 factor storage.
- New Apple Metal backend: setup, factorization and PCG run on the device, for single and batched solves.
- Metal backend and macOS support contributed by Samuel Talkington (@samtalki).
- GPU PCG waits for the device once every four iterations on CUDA (instead of every iteration) and every eight on Metal (instead of four), and skips the reductions of iterations after convergence; solutions are unchanged, and small solves are up to 15% faster on CUDA and 7% on Metal.
- `solve_options.backend` (`--backend auto|cpu|gpu`) selects a complete CPU or GPU route and never falls back silently.
- CUDA and Metal solvers can be copied: copies share the device factor and operator, and solve concurrently on their own streams or command queues and workspace.

### API and CLI changes
- Custom partitioners can query any graph vertex concurrently and from inside neighbor callbacks. The view remains valid only during the partitioner call.
- The built-in partitioners select through the same `partition_graph` view that custom partitioners receive. `baumann_kyng` as the main partitioner now samples every vertex in its first round, which changes its factors on graphs with average degree above 10. The Baumann–Kyng continuation of the default and `priority_greedy` partitioners starts the same way, which changes their factors on graphs where that continuation runs.
- Setup profiles include unlabelled work and agree with `solver::setup_seconds()`. Normal benchmark runs omit optional stage profiling; `--profile` retains the detailed breakdown.
- Breaking: Eigen types leave the C++ API. `solver`, `factorize` and the one-shot `solve` take a borrowed CSC matrix, `csc_view` (32-bit indices) or `csc_view64` (64-bit), from the new header `apxchol/views.h`; rows must be sorted and unique within each column, and an invalid view throws `std::invalid_argument`. Vectors are `std::span`s and matrices `const_matrix_view` / `matrix_view` (column-major).
- apxchol no longer depends on Eigen. The installed CMake package, SDK archives, Python wheels, CLI archives and the MATLAB/Octave build neither bundle nor require it, and their notices drop Eigen's MPL-2.0 terms; only the benchmark harness still uses Eigen, for its competitors and input reading.
- Breaking: the initial guess is a `std::span` that is empty by default instead of a `const Eigen::VectorXd*`, and the batch initial guess is a `const_matrix_view` whose null `data` means none.
- Breaking: `solve_result::x` and the vector `solver::apply` return `std::vector<double>`; `solve_result::iterations`, `factorization::lumped_offdiag`, `solver::rows()` and `solver::operator_nnz()` are `std::int64_t`.
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
- `apxchol::core` no longer passes its `-march` flag to consumers, and a `-march`/`-mcpu` already in `CMAKE_CXX_FLAGS` or `CMAKE_CXX_FLAGS_<CONFIG>` replaces apxchol's own. Consumers can target any instruction set: no public header depends on it. Projects that embed apxchol by including `cmake/apxchol_core.cmake` now default to portable code, like `add_subdirectory` embedding; standalone root, Python and benchmark builds stay native.
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
- Breaking: custom partitioners receive a read-only `apxchol::partition_graph` (`n`, `is_active`, `for_each_neighbor`, `any_neighbor`, `degree`) instead of the mutable graph; `graph`, `make_graph` and `laplacian` move to `apxchol::detail`.
- Breaking: partitioners no longer declare a `name`, and `partitioner_list.h` with `dispatch_partitioner` is removed; pass a custom partitioner as an instance to `factorize`.
- Breaking: `factorize(A, eliminator, opts)` no longer takes a partitioner template argument, and now uses the partitioner `opts.experimental.partitioner` names instead of always `block_greedy`; pass an instance to use another partitioner.
- Breaking: `tree_elimination` moves to `apxchol::detail`; configure the built-in rule through `factor_options`.
- Breaking: `sparse_csc::kEdgeMax` is removed; use `std::numeric_limits<apxchol::edge_index>::max()`.
- New `apxchol.as_operator(M)`, `apxchol_as_operator(M)` and CLI `--convert-input`: a finite symmetric matrix with at least one off-diagonal entry, all of them positive, becomes its graph Laplacian `L = D - M` (self-loops cancel), any other finite symmetric matrix is used as given and checked by the solver, and asymmetric or non-finite input is an error.
- The CLI solves an `n` by `k` `--rhs` file as one batch, writes an `n` by `k` solution with `-o`, and logs iterations and residuals per column.
- The v1 API compatibility scope is documented.
- Breaking: from 1.0, the installed CMake package accepts any version request with the same major version (0.x stays minor-scoped), and the shared core library's soname now carries `major.minor` (`libapxchol_core.so.0.3` for this version).
- The partitioner and eliminator extension interfaces are experimental in 1.x and may change in minor releases.
- Breaking: research controls (`omp_threshold`, `min_is_fraction`, `parallel_residual_threshold`, `exact_core_max_h`, `double_cycle_min_h`, `cpu_schedule`) move to `factor_options::experimental`, outside the 1.x compatibility promise, and leave the CLI, Python and MATLAB/Octave.
- Breaking: `partitioner`, `partition` (`degree_quantile`, `degree_multiplier`) and `sampler` move to `factor_options::experimental`, outside the 1.x compatibility promise. The stable factorization options are now `seed`, `factor_storage`, `drop_tolerance`, `positive_offdiag_tolerance` and `verbose`.
- Breaking: the `exact_clique_max_degree`, `degree_tiebreak` and `regularization` factor options are removed. The built-in partitioners always prefer lower-degree vertices in conflicts.
- Breaking: CLI `--partitioner`, `--sampler`, `--regularization`, `--degree-quantile` and `--degree-multiplier` are removed. `-v` still prints setup diagnostics and the timing tree.
- Breaking: Python `factorize` and `solve` drop the `partitioner`, `sampler`, `degree_quantile`, `degree_multiplier` and `regularization` keywords, and MATLAB/Octave drops the same option fields. Passing one raises the unknown-option error.
- Breaking: the `residual_peel` option and its `min_degree` and `bk_serial` strategies are removed; the serial tail always uses the natural order.
- Breaking: `factor_options::is_select` (a string) is replaced by `experimental.partitioner`, a `partitioner_kind` enum that option validation checks.
- Breaking: unset `degree_quantile`, `parallel_residual_threshold` and `exact_core_max_h` are `std::nullopt` instead of -1 or `SIZE_MAX`, and the sentinel constants and resolvers such as `exact_core_by_route` and `degree_quantile_or_host_default` are removed.
- Breaking: CLI `--is` is removed, `--factor-drop` and `--maxiter` are renamed `--drop-tolerance` and `--max-iter`, and `--factor-storage` accepts names only.
- Python and MATLAB/Octave accept the same stable setup options.
- Python wheels for CPython 3.11+ on Linux and macOS arm64 (GPU wheels select the backend at runtime), standalone CLI archives, and MATLAB/Octave batched solves with save/load.
- Wheels declare a `License-Expression` that matches their contents (BSD-3-Clause and Apache-2.0 WITH LLVM-exception; Linux CUDA wheels add the NVIDIA CUDA EULA and ship its text) and take the root LICENSE at build time; minimum scikit-build-core is 0.12.
- Relicensed from BSD-4-Clause to BSD-3-Clause.
- Breaking: `solver::solve()` and `solver::apply()` are non-const because they use the solver's workspace; concurrent non-const calls on one solver are not allowed.

### Benchmarks
- The benchmark harness checks every result against the original system's residual and reports setup, solve and memory separately.
- Published results now come from CSCS Daint; laptop results are archived.
- GPU memory is now this process's NVML usage above its post-context baseline, measured the same way for apxchol, AMGCL and Hypre, with a new `setup_vram_mb` column; other compute processes are reported separately and flagged as shared.
- The benchmark builds on macOS with a Metal route for apxchol (`--device metal`), with Metal warm-up excluded from setup and the factor-drop statistics in `--report-fill`.

### Documentation
- CLI help groups input, solve and setup options. The website links to the repository and shows a version selector only when release snapshots exist.
- The website separates user and developer guides, adds installation, quick start, examples, changelog and citing pages, and marks development pages with a banner.
- The precision guide states when results are reproducible and that a release may change results, within the documented precision, for measured speed or memory gains.
- Clearer messages for a solver used by 2 threads at once, an archive from a newer version, and positive off-diagonal input entries.
- Errors for an invalid tolerance, iteration budget, thread count or backend, an unsupported GPU request and a missing Metal device start with `apxchol:` and name the next step.
- A new Input formats page shows how to pass matrices from Eigen, SuiteSparse/CHOLMOD, SciPy, MATLAB/Octave, raw CSR/CSC arrays and Matrix Market files; CI builds and runs its examples.

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
