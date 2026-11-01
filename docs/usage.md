# Using the solver

## The matrix you supply

Pass the sparse matrix `A` in the equation `A x = b`.
apxchol is designed for graph Laplacians and SDDM matrices: symmetric,
diagonally dominant matrices with nonpositive off-diagonal entries.

For a graph with nonnegative edge weights, the adjacency matrix stores those
weights. Its Laplacian is
`L = D - Adj`, where `D` contains the sums of incident weights.
A Laplacian RHS must sum to zero on each connected component; the solution is
defined up to a constant on each component.

### Converting input to an operator

The solver always treats its input as `A` and rejects anything that is not a
valid operator. To start from a graph instead, convert the input first:

| CLI | Python | MATLAB/Octave |
|---|---|---|
| `--convert-input` | `A = apxchol.as_operator(M)` | `A = apxchol_as_operator(M)` |

All three apply the same rule, like NumPy's `asarray`. `M` must be square,
finite and symmetric; otherwise the conversion fails and says why.

- If `M` has off-diagonal entries and none of them is negative, it is a graph
  adjacency matrix, and its Laplacian `L = D - M` is used, where `D` holds the
  row sums of `M`. A diagonal entry (self-loop) adds to `D` and is subtracted
  again, so it cancels. Pattern input (a Matrix Market `pattern` file, a
  MATLAB logical matrix, a boolean SciPy matrix) has unit weights.
- Otherwise `M` is used unchanged, and the solver checks it as an operator.
  This covers Laplacian, SDDM and nearly SDDM matrices and diagonal matrices.

Without conversion, when a matrix fails as an operator but has no negative
off-diagonal entry, the error suggests the conversion for each interface.

Python accepts SciPy sparse matrices and arrays. It converts them to CSC
(compressed sparse column) format, sorts row indices and combines duplicate
entries. Supply the full symmetric matrix, not just one triangle. Format conversion
does not invent missing symmetric entries.

C++ borrows the matrix as a `csc_view` (32-bit indices) or `csc_view64`
(64-bit indices) from `apxchol/views.h`: the dimension `n`, `col_ptr` with
`n + 1` offsets starting at 0, and `row_idx` and `values` with `col_ptr[n]`
entries each. Rows must be sorted and unique within each column; an invalid
view throws `std::invalid_argument`. Because `A` is symmetric, CSR arrays can
be passed unchanged. Index arrays as wide as the build's indices are used in
place; the others are converted once during setup. The solver copies what it
needs, so the arrays only have to outlive the call that receives the view.
[Input formats](input-formats.md) shows conversions from Eigen, CHOLMOD,
SciPy, MATLAB/Octave and Matrix Market.

```cpp
std::vector<std::int32_t> col_ptr, row_idx;   // filled by the caller
std::vector<double> values;
const apxchol::csc_view A{n, col_ptr, row_idx, values};
```

## Choosing CPU or GPU

The `CPU` build contains the CPU solver. `CUDA` and `METAL` builds add their
GPU backend and retain CPU support. Python wheels include CPU plus the GPU
backend for their platform; they work without a GPU.

| Request | Behavior |
|---|---|
| `auto` | Select a compatible available backend before setup; may prefer CPU for small inputs |
| `cpu` | Run setup and solves on CPU |
| `gpu` | Run setup and solves on the compiled GPU backend; fail if the device, input or options are unsupported |

Automatic selection is not a guarantee of the fastest backend for every
matrix. Once selected, the backend stays fixed. An execution error is
reported rather than silently retrying on another backend.

Use Python's `backend` argument, C++ `solve_options.backend`, or CLI
`--backend`. Python's `solver.backend` and native solve results report the
selected backend. The MATLAB/Octave binding currently uses CPU only.

### Native GPU input and options

For direct C++ input, GPU setup needs every stored off-diagonal `(i, j)` to
have its matching `(j, i)`. GPU setup also requires 32-bit vertex indices.

CPU setup supports custom partitioners and eliminators. CUDA setup supports
`block_greedy` and GKS or trace-cycle sampling; the experimental exact-clique,
exact-core and double-cycle rules require CPU construction.
Metal setup supports the default `block_greedy`/GKS options and FP32 graph
weights. See [precision limits](precision.md#apple-metal-pcg) for Metal ranges.

These construction restrictions do not prevent using an explicitly supplied
host factor with a GPU solver. See [factor handoff](extending.md#reusing-a-custom-factor).

## Option names

Each interface spells the stable options as follows. Setup options are
`solve_options` fields in C++, keywords in Python and struct fields in
MATLAB/Octave. The tolerance, iteration budget and initial guess are arguments
of each solve call in every library interface; omitted, the initial guess is
zero and the others take the
[native defaults](reference/cpp.md#solve-defaults).

| C++ | CLI | Python | MATLAB/Octave |
|---|---|---|---|
| `solve_options::backend` | `--backend` | `backend` | CPU only |
| `solve_options::num_threads` | `--threads` | `num_threads` | `num_threads` |
| `tol` (solve argument) | `--tol` | `rtol` (solve argument) | `rtol` (solve argument) |
| `max_iter` (solve argument) | `--max-iter` | `maxiter` (solve argument) | `maxiter` (solve argument) |
| `x0`, initial guess, default zero (solve argument) | `--x0` | `x0` (solve argument) | `x0` (solve argument) |
| `factor_options::seed` | `--seed` | `seed` | `seed` |
| `factor_options::partitioner` | `--partitioner` | `partitioner` | `partitioner` |
| `factor_options::sampler` | `--sampler` | `sampler` | `sampler` |
| `factor_options::factor_storage` | `--factor-storage` | `factor_storage` | `factor_storage` |
| `factor_options::drop_tolerance` | `--drop-tolerance` | `drop_tolerance` | `drop_tolerance` |
| `factor_options::regularization` | `--regularization` | `regularization` | `regularization` |
| `factor_options::lump_max_mass` | `--lump-max-mass` | `lump_max_mass` | `lump_max_mass` |
| `partition_options::degree_quantile` | `--degree-quantile` | `degree_quantile` | `degree_quantile` |
| `partition_options::degree_multiplier` | `--degree-multiplier` | `degree_multiplier` | `degree_multiplier` |
| `factor_options::verbose` | `--verbose` | `verbose` | `verbose` |

The partitioner (`block_greedy`, `priority_greedy` or `baumann_kyng`) and the
sampler (`gks` or `trace_cycle`) take the same names everywhere. Factor storage
is `auto`, `float16` or `float32` in the CLI, Python and MATLAB/Octave (Python
also accepts a NumPy dtype), and `factor_storage_type::automatic`, `fp16` or
`fp32` in C++. Python's `rtol` and `maxiter` follow SciPy's `cg`, and the CLI's
`--threads` is the usual command-line spelling. The research controls in `factor_options::experimental` are
available only from C++; see the [API policy](api.md).

## Reuse, batches and initial guesses

Construct a solver once, then call `solve` for each new RHS. A matrix RHS
represents several independent systems with the same `A`; pass the columns
together to use native batching. This reuses the factor and shares work across
columns.

A **warm start** is an initial guess `x0`, often the solution of a related
system. Without one, the solver starts from zero. A useful guess can reduce
iterations; a poor guess need not help. C++, Python and MATLAB/Octave support
vector and matrix initial guesses, and the CLI reads one with `--x0`. See the [Python](../python/README.md) and
[MATLAB/Octave](../octave/README.md) examples.

For C++:

```cpp
#include "apxchol.h"
apxchol::solver solver(A);                   // A: csc_view
auto result = solver.solve(b, 1e-8);         // b: std::vector<double> or span
// std::nullopt selects the default tolerance or iteration budget.
auto warm = solver.solve(b2, std::nullopt, std::nullopt, result.x);
// k right-hand sides, column-major: column j starts at B.data() + j * n.
std::vector<double> X(n * k);
auto batch = solver.solve(apxchol::const_matrix_view{B.data(), n, k, n},
                          apxchol::matrix_view{X.data(), n, k, n});
```

Vectors are `std::span<double>` arguments, so a `std::vector`, an array or a
pointer and length all work; `result.x` is a `std::vector<double>`. Matrix
views are column-major with column stride `col_stride >= rows`. An empty `x0`
(the default) or a `const_matrix_view` with a null `data` starts from zero.

One-shot `apxchol::solve(A, b, tol, max_iter, x0, opts)` uses the same backend
selection. For explicit CPU execution, set
`solve_options.backend = solve_backend::cpu`.

### Judging whether reuse pays off

Setup builds the factor and prepares the backend once; each solve then pays
only for its own solve work. Every solver reports both parts as wall-clock
seconds:

| | C++ | Python | MATLAB/Octave |
|---|---|---|---|
| Setup | `solver.setup_seconds()` | `solver.setup_seconds` | `s.setup_seconds` |
| One solve call | `result.solve_seconds` | `result.solve_seconds` | `res.solve_seconds` |

Over `k` separate one-RHS calls, each costs `setup_seconds / k + solve_seconds`
on average. A matrix RHS reports one `solve_seconds` for all its columns, so
one batched call of `k` columns costs `(setup_seconds + solve_seconds) / k` per
RHS. Reuse pays off once that is below the time of solving each system
another way, for example with a one-shot solve or another preconditioner. After
`load`, `setup_seconds` includes reading the file, so it also shows what a
saved solver saves over factorizing again.

```cpp
apxchol::solver solver(A);
auto result = solver.solve(b);
std::printf("setup %.3f s, solve %.3f s\n", solver.setup_seconds(), result.solve_seconds);
```

## Binding capabilities

The bindings reuse the native solver, convergence checks and option defaults.
Their language-specific helpers differ:

| Capability | Python | MATLAB/Octave |
|---|---|---|
| Reusable factor, scalar and batched solves/apply | Yes | Yes |
| Initial guess, per-call tolerance and iteration budget | Yes | Yes |
| GPU setup and solve | CUDA or Metal when built and available | CPU only |
| Factor export | Installed factor, float64 copies | Not exposed |
| External iterative solver | SciPy `LinearOperator` | Function handle for `pcg` |

## Results and errors

Every solve reports the same fields: the solution `x`, `iterations`,
`residual` and `converged`, plus `solve_seconds`. For a matrix RHS, C++ returns
one record per column, and Python and MATLAB/Octave return one entry per column
in each field.

Errors fall into two classes in every interface:

| Cause | C++ | Python | MATLAB/Octave | CLI |
|---|---|---|---|---|
| Invalid input or options, or a request the build or backend does not support | `std::invalid_argument` | `ValueError` | `apxchol:badInput` | exit code 2 |
| Execution failure, such as a solution outside the floating-point range, an unavailable device or a file error | `std::runtime_error` or a type derived from it | `RuntimeError` | `apxchol:error` | exit code 2 |

Not converging within the iteration budget is not an error: check
`converged` (the CLI exits with code 1).

## Convergence and cost

Built-in solves verify the returned solution against the original matrix and
RHS. `result.residual` reports that relative residual, and `result.converged`
tells whether it meets the requested tolerance. Inspect this flag even when
the call returns without an exception.

This check is always part of the solve. It usually costs one additional
sparse matrix–vector product and norm reductions per RHS; trivial exact exits
can avoid that work. Its cost is included in `solve_seconds`.

A failed final check does not trigger more iterations or a different backend.
The [precision contract](precision.md) explains where each backend checks the
residual and how rounding affects it.

## Using another iterative solver

To use apxchol as the preconditioner of another Krylov solver (Eigen, SciPy,
PETSc, MKL's reverse-communication interface), apply it with C++
`solver.apply(r)` for vectors or `solver.apply(R, Z)` for matrices, using
`const_matrix_view` and `matrix_view`, or Python `solver.aslinearoperator()`.
Each application computes `z = M^{-1} r` with the prepared factor; the other
solver controls the iteration and its stopping test.

For Eigen's iterative solvers, a small adapter is enough. It is an example
outside apxchol's stable API; copy it into your code. The solver borrows the
compressed `Eigen::SparseMatrix<double>` while it is built:

```{literalinclude} ../examples/input_formats/eigen_cg.cpp
:language: cpp
```

A GPU-backed SciPy operator exchanges host vectors with the device on each
application. Native `solve` keeps iteration vectors on the GPU, so prefer it
for GPU workloads unless you need another outer solver.

## Exporting a factor

`solver.factor()` in C++ and `chol()` and `P` in Python return the
installed factor: the preconditioner the solves actually use, after dropping
and in the solver's [storage precision](precision.md). The solver keeps no
separate copy of the factor; export reads its triangular-solve arrays, and a
GPU solver downloads them. The [factor reference](reference/cpp.md#constructing-and-reusing-a-factor)
defines the exported form.

CPU-, CUDA- and Metal-built factors can be reused through
[C++ factor handoff](extending.md#reusing-a-custom-factor). Automatic adoption
uses CPU; request `solve_backend::gpu` to upload the factor to a GPU.

Portable solver archives [save the original matrix and installed factor
together](serialization.md), so loading can prepare the solver without
repeating factorization.

## CPU threads

Set Python `num_threads`, C++ `solve_options.num_threads`, or CLI `--threads`
to cap CPU parallelism. Zero inherits the caller's OpenMP setting; a positive
value overrides it for the operation and is restored afterward. Small
workloads may use fewer threads.

The setting also covers CPU work in a GPU build; it does not control GPU
threadgroups. More CPU threads need not be faster. To choose which cores run those CPU workers,
see [thread placement](../benchmarks/README.md#thread-placement).

## Thread safety

Use one solver per thread. A solver reuses its workspace across `solve`,
`apply` and batch solves. In C++ these methods are non-const, and concurrent
non-const calls on one object are not allowed: use a copy, or batch the
right-hand sides. As a safety net, a call made while another thread is using
the same solver throws (`RuntimeError` in Python) instead of waiting.

Copies are cheap: a copy shares the factor, the triangular-solve structures
and the operator, which are read-only after setup, and allocates only its
own workspace. Copy a solver once per thread, with `solver copy = original;`
in C++ or `copy.copy(solver)` in Python, and the copies solve in parallel
with the same results as serial calls. Copies stay valid after the original
is destroyed. On a GPU, copies share the factor and operator in device memory
and each copy gets its own device workspace and stream (CUDA) or command queue
(Metal), so copies on one GPU solve concurrently without storing the factor
again; they share that GPU's compute and memory bandwidth.

To solve many right-hand sides from one thread, pass them to `solve(B)` as the
columns of one matrix; the batched solver shares work between columns.

Python releases the GIL during factorization, `solve` and `apply`, but holds it
while reading the input matrix.
