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

For direct C++ input, GPU setup needs compressed CSC storage: column offsets
must cover only stored entries, row indices within each column must be sorted
and unique, and every stored off-diagonal `(i, j)` needs its matching `(j, i)`.
GPU setup also requires 32-bit vertex indices.

CPU setup supports custom partitioners and eliminators. CUDA setup supports
`block_greedy` and GKS or trace-cycle sampling; the experimental exact-clique,
exact-core and double-cycle rules require CPU construction.
Metal setup supports the default `block_greedy`/GKS options and FP32 graph
weights. See [precision limits](precision.md#apple-metal-pcg) for Metal ranges.

These construction restrictions do not prevent using an explicitly supplied
host factor with a GPU solver. See [factor handoff](extending.md#reusing-a-custom-factor).

## Option names

Each interface spells the stable options as follows. Python and MATLAB/Octave
take setup options as keywords or struct fields, and the solve tolerance and
iteration budget as solve arguments.

| C++ | CLI | Python | MATLAB/Octave |
|---|---|---|---|
| `solve_options::backend` | `--backend` | `backend` | CPU only |
| `solve_options::num_threads` | `--threads` | `num_threads` | `num_threads` |
| `solve_options::keep_factor_values` | | `keep_factor` | `keep_factor` |
| `solve_options::tol` | `--tol` | `rtol` (solve argument) | `rtol` (solve argument) |
| `solve_options::max_iter` | `--max-iter` | `maxiter` (solve argument) | `maxiter` (solve argument) |
| `factor_options::seed` | `--seed` | `seed` | `seed` |
| `factor_options::partitioner` | `--partitioner` | `partitioner` | `partitioner` |
| `factor_options::sampler` | `--sampler` | `sampler` | `sampler` |
| `factor_options::factor_storage` | `--factor-storage` | `factor_storage_dtype` | `factor_storage` |
| `factor_options::drop_tolerance` | `--drop-tolerance` | `drop_tolerance` | `drop_tolerance` |
| `factor_options::regularization` | `--regularization` | `regularization` | `regularization` |
| `factor_options::lump_max_mass` | `--lump-max-mass` | `lump_max_mass` | `lump_max_mass` |
| `partition_options::degree_quantile` | `--degree-quantile` | `degree_quantile` | `degree_quantile` |
| `partition_options::degree_multiplier` | `--degree-multiplier` | `degree_multiplier` | `degree_multiplier` |
| `factor_options::verbose` | `--verbose` | `verbose` | `verbose` |

The partitioner (`block_greedy`, `priority_greedy` or `baumann_kyng`) and the
sampler (`gks` or `trace_cycle`) take the same names everywhere. Factor storage
is `auto`, `float16` or `float32` in the CLI and MATLAB/Octave, a NumPy dtype
in Python's `factor_storage_dtype`, and `factor_storage_type::automatic`,
`fp16` or `fp32` in C++. The research controls in `factor_options::experimental` are
available only from C++; see the [API policy](api.md).

## Reuse, batches and initial guesses

Construct a solver once, then call `solve` for each new RHS. A matrix RHS
represents several independent systems with the same `A`; pass the columns
together to use native batching. This reuses the factor and shares work across
columns.

A **warm start** is an initial guess `x0`, often the solution of a related
system. Without one, the solver starts from zero. A useful guess can reduce
iterations; a poor guess need not help. C++, Python and MATLAB/Octave support
vector and matrix initial guesses. See the [Python](../python/README.md) and
[MATLAB/Octave](../octave/README.md) examples.

For C++:

```cpp
#include "apxchol.h"
apxchol::solver solver(A);
auto result = solver.solve(b, 1e-8);
// Negative tolerance/budget arguments keep this solver's defaults.
auto warm = solver.solve(b2, -1, -1, &result.x);
Eigen::MatrixXd X(B.rows(), B.cols());
auto batch = solver.solve(B, X);
```

One-shot `apxchol::solve(A, b)` uses the same backend selection. For explicit
CPU execution, set `solve_options.backend = solve_backend::cpu`.

## Binding capabilities

The bindings reuse the native solver, convergence checks and option defaults.
Their language-specific helpers differ:

| Capability | Python | MATLAB/Octave |
|---|---|---|
| Reusable factor, scalar and batched solves/apply | Yes | Yes |
| Initial guess, per-call tolerance and iteration budget | Yes | Yes |
| GPU setup and solve | CUDA or Metal when built and available | CPU only |
| Factor export | FP32 values when retained | Not exposed |
| External iterative solver | SciPy `LinearOperator` | Function handle for `pcg` |

## Convergence and cost

Built-in solves verify the returned solution against the original matrix and
RHS. `result.residual` reports that relative residual, and `result.converged`
tells whether it meets the requested tolerance. Inspect this flag even when
the call returns without an exception.

This check is always part of the solve, including when timing collection is
disabled. It usually costs one additional sparse matrix–vector product and
norm reductions per RHS; trivial exact exits can avoid that work. Its cost is
included in reported solve time.

A failed final check does not trigger more iterations or a different backend.
The [precision contract](precision.md) explains where each backend checks the
residual and how rounding affects it.

## Using another iterative solver

apxchol's SciPy operator (`solver.aslinearoperator()`) and Eigen-compatible
preconditioner (`apx_cholesky`, CPU only) let another solver drive the
iteration. That solver controls its stopping test.

A GPU-backed SciPy operator exchanges host vectors with the device on each
application. Native `solve` keeps iteration vectors on the GPU, so prefer it
for GPU workloads unless you need another outer solver.

## Keeping and exporting a factor

Exported factor values are FP32. Internal triangular-solve storage can be
FP16 or FP32 independently; see [storage choices](precision.md).

C++ defaults to releasing factor values after preparation. Set
`keep_factor_values=true` when you need them later through `solver.factor()`.
Python's reusable solver instead defaults to `keep_factor=True`, enabling
`chol()`, `L` and `D` exports. Set it to `False` to save memory; the permutation
`P` remains available.

CPU-, CUDA- and Metal-built factors can be reused through
[C++ factor handoff](extending.md#reusing-a-custom-factor). Automatic adoption
uses CPU; request `solve_backend::gpu` to upload the factor to a GPU.

Portable solver archives [save the original matrix and retained factor
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
is destroyed. This covers `solver` and `apx_cholesky`; a new
factor installed in one copy of an `apx_cholesky` leaves the others unchanged.
GPU solvers cannot be copied: construct one per thread, or batch the
right-hand sides. A `metal_solver` used directly must not be called
concurrently.

To solve many right-hand sides from one thread, pass them to `solve(B)` as the
columns of one matrix; the batched solver shares work between columns.

Python releases the GIL during factorization, `solve` and `apply`, but holds it
while reading the input matrix.
