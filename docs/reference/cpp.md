# C++ reference

Include `apxchol.h` and link `apxchol::core`. Start with the
[usage guide](../usage.md) or [CMake integration](../cpp.md).
This page lists the existing public controls; the [API policy](../api.md)
defines their compatibility guarantees.

## Input views

Every C++ call borrows the caller's arrays through these views; nothing is
copied when a view is made. A `csc_view` holds 32-bit indices and a
`csc_view64` 64-bit indices; both are checked at the call and an invalid view
throws `std::invalid_argument`. Dense views are column-major.

```{doxygenstruct} apxchol::basic_csc_view
:members: n, col_ptr, row_idx, values
:undoc-members:
```

```{doxygentypedef} apxchol::csc_view
```

```{doxygentypedef} apxchol::csc_view64
```

```{doxygenstruct} apxchol::matrix_view
:members: data, rows, cols, col_stride
:undoc-members:
```

```{doxygenstruct} apxchol::const_matrix_view
:members: data, rows, cols, col_stride
:undoc-members:
```

## Reusable solver

Use `solver` for CPU, CUDA and Metal operation; there is no separate class per
backend. Its backend is selected during construction and stays fixed. To force
CPU use, set `opts.backend` to `solve_backend::cpu`.

`solve` checks the returned solution against the original matrix and RHS.
Inspect `converged`; reaching the iteration limit is not an exception.
A warm start is an initial solution estimate supplied through `x0`.

```{doxygenclass} apxchol::solver
:members: solver, solve, apply, factor, rows, operator_nnz, backend, factor_storage, setup_seconds, save, load
:undoc-members:
```

### Solve defaults

These values come directly from the native headers. Every interface uses them
when a solve omits the tolerance or iteration budget; an explicit invalid
value throws `std::invalid_argument`.

```{doxygenvariable} apxchol::default_tol
```

```{doxygenvariable} apxchol::default_max_iter
```

### Solve options

These options fix the setup and backend; the tolerance and iteration budget
are arguments of each solve.

```{doxygenstruct} apxchol::solve_options
:members: factor_opts, backend, num_threads
:undoc-members:
```

```{doxygenenum} apxchol::solve_backend
```

### Results

Scalar solves return the solution in `x`, unless the caller supplied output
storage. Matrix solves write into the caller's output matrix and return one
status record per RHS column. `solve_seconds` is the wall-clock time of the
call; compare it with `solver::setup_seconds()` to judge whether reusing the
factor pays off ([usage guide](../usage.md#judging-whether-reuse-pays-off)).

```{doxygenstruct} apxchol::solve_result
:members: x, iterations, residual, converged, solve_seconds, backend
:undoc-members:
```

```{doxygenstruct} apxchol::batch_solve_result
:members: columns, solve_seconds
:undoc-members:
```

## Factor construction and storage

The controls below belong to `solve_options::factor_opts` or the `opts`
argument of `factorize`. Storage precision is separate from construction
precision and exported values; see the [precision guide](../precision.md).

```{doxygenstruct} apxchol::factor_options
:members: seed, regularization, lump_max_mass, factor_storage, drop_tolerance, partition, partitioner, sampler, verbose, experimental
:undoc-members:
```

```{doxygenenum} apxchol::factor_storage_type
```

```{doxygenenum} apxchol::clique_sampler
```

```{doxygenenum} apxchol::partitioner_kind
```

### Candidate selection

The partitioner chooses independent vertices for elimination. Its degree
controls live in `factor_options::partition`.

```{doxygenstruct} apxchol::partition_options
:members: degree_quantile, degree_multiplier
```

### Experimental controls

`factor_options::experimental` holds research controls for scheduling, how
factorization finishes the remaining graph, and sampling rules. They are
outside the [1.x compatibility promise](../api.md) and may change in any
release. Keep their defaults unless comparing a specific alternative.
GPU routes require the defaults of `cpu_schedule`, `exact_clique_max_degree`,
`exact_core_max_h` and `double_cycle_min_h`; Metal also requires
`degree_tiebreak`. See [backend selection](../usage.md).

```{doxygenstruct} apxchol::experimental_options
:members: cpu_schedule, degree_tiebreak, min_is_fraction, omp_threshold, parallel_residual_threshold, exact_clique_max_degree, exact_core_max_h, double_cycle_min_h
```

```{doxygenenum} apxchol::cpu_sptrsv_schedule
```

## Constructing and reusing a factor

`factorize(A, opts)` constructs a factor on CPU and retains its host arrays.
Custom partitioner and eliminator overloads are described in the
[extension guide](../extending.md).

```{doxygenfunction} apxchol::factorize(basic_csc_view<Index> L, const factor_options &opts = {})
```

```{doxygenstruct} apxchol::factorization
:members: L, perm, sddm, rounds, storage, scale
:undoc-members:
```

`L` stores lower-triangular columns in compressed sparse column (CSC) form,
with each diagonal first. `perm[original_vertex]` gives its elimination
position. With S = diag(`scale`), or S = I when `scale` is empty, the
preconditioner is M = Pᵀ L S² Lᵀ P. Keep `L`, the permutation, operator
kind, round metadata, `storage` and `scale` together when handing a factor to
another solver.

```{doxygenclass} apxchol::sparse_csc
:members: rows, cols, nonZeros, has_values, col_ptrs, row_indices, values
```

`sparse_csc` is a read-only view: `col_ptrs()`, `row_indices()` and `values()`
return spans over the factor's own arrays.

`storage` distinguishes two forms. `factorize()` returns a construction
factor (`automatic`): a solver that adopts it applies its own
`drop_tolerance` and `factor_storage`. `solver.factor()` returns the installed
factor (`fp32` or `fp16`), after dropping and storage rounding; for a
Laplacian its last row and column are empty. A solver adopts an installed
factor exactly and ignores those two options. With `fp16`, every
off-diagonal of `L` is a binary16 value, each diagonal is FP32 and `scale`
holds the FP32 column scales.

Any solver, including one built on CUDA or Metal, exports its installed
factor through `solver.factor()`. Pass it to `solver(A, F, opts)` to reuse
it. Automatic adoption uses CPU; set `opts.backend = solve_backend::gpu` to
upload it to a GPU.
The original matrix `A` is still needed for PCG and residual checks.
For cross-process reuse, `solver.save(path)` and `solver::load(path, opts)`
[store the operator and factor together](../serialization.md).

## One-shot solve and availability

The one-shot solve takes a `csc_view` or a `csc_view64`.

```{doxygenfunction} apxchol::solve(csc_view A, std::span<const double> b, std::optional<double> tol = std::nullopt, std::optional<int> max_iter = std::nullopt, std::span<const double> x0 = {}, const solve_options &opts = {})
```

```{doxygenfunction} apxchol::backend_available
```
