# apxchol for Python

Solve sparse Laplacian and SDDM systems with reusable approximate-Cholesky
factors. NumPy is required. Install the SciPy extra for sparse-matrix helpers.

## Install and solve

```bash
python -m pip install "apxchol[scipy]"
```

For an unreleased checkout, use the [source installation](#build-from-source)
below; pip installs the published version.

Wheels work on CPU without a GPU. Compatible hardware enables the included
GPU backend; no CUDA Toolkit or Homebrew installation is needed.

| Platform | CPU requirement | Optional GPU backend |
|---|---|---|
| Linux | x86-64-v2 or newer | NVIDIA CUDA |
| macOS 15+ | Apple silicon (arm64) | Apple Metal |

Wheels support CPython 3.11–3.15 and PyPy 7.3.23 (Python 3.11, `pp73`).
On PyPy, install `apxchol` and use the [NumPy-only interface](#numpy-only-input).

```python
import apxchol
import numpy as np
from scipy.sparse import diags

n = 1000
# Tridiagonal matrix: 3 on the diagonal, -1 on each neighboring diagonal.
A = diags([-1.0, 3.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")
b = np.ones(n)

solver = apxchol.factorize(A)
result = solver.solve(b, rtol=1e-8)
assert result.converged
x = result.x
```

For one solve, use `apxchol.solve(A, b)`. Results include `x`, `iters`,
`residual` and `converged`. Convergence is always checked against the original
`A` and `b`; inspect `converged` for success.

See the shared [binding capabilities](https://algoptgroup.github.io/apxchol/docs/usage.html#binding-capabilities)
for comparison with MATLAB/Octave.

## Matrix inputs

Pass the matrix `A` in `A @ x = b`. For a graph's adjacency matrix, first use
`A = apxchol.as_operator(Adj)`: it returns a valid operator unchanged and
otherwise forms the graph Laplacian. See [supported matrices, input
conversion and RHS requirements](https://algoptgroup.github.io/apxchol/docs/usage.html).

Any SciPy sparse format is converted to sorted, duplicate-free CSC without
changing the input. Supply the full symmetric matrix, including both
entries of each off-diagonal pair. Complex inputs are rejected, even with
zero imaginary parts.

### NumPy-only input

`python -m pip install apxchol` installs the solver without SciPy. Supply the
three zero-based CSC arrays directly; all usual solver options still apply:

```python
solver = apxchol.Solver.from_csc(data, indices, indptr, backend="auto")
result = solver.solve(b)
```

The matrix is square with `len(indptr) - 1` rows. Integer indices may be 32- or
64-bit; the native importer sorts columns and sums duplicate entries. SciPy
is needed for `factorize(A)`, `laplacian`, `aslinearoperator`, and sparse factor
exports. NumPy-only solves and preconditioner application do not import it.

## Reuse, batches and initial guesses

Subsequent solves reuse the factor and workspace:

```python
first = solver.solve(b)
b2 = np.arange(n, dtype=float)
B = np.column_stack((b, b2))
batch = solver.solve(B)            # native batched solve, one RHS per column
assert batch.converged.all()
```

Vector inputs have shape `(n,)`; batches have shape `(n, nrhs)`. `result.x`
and `apply` outputs preserve that shape and use float64. Batch convergence
fields contain one value per column.

A *warm start* supplies an initial solution estimate:

```python
result = solver.solve(b2, x0=first.x)
```

| Solve argument | Meaning |
|---|---|
| `rtol` | Positive, finite relative residual tolerance; omitted uses the native default |
| `maxiter` | Nonnegative iteration budget; omitted uses the native default |
| `x0` | Initial guess with the same shape as the RHS; omitted starts from zero |
| `out` | Writable, C- or F-contiguous float64 array with the RHS shape; returned as `result.x` |

The [native reference](https://algoptgroup.github.io/apxchol/docs/reference/cpp.html#solve-defaults) shows the
compiled tolerance and iteration defaults.

Overlapping inputs are copied before writing `out`. Factorization, `solve`
and `apply` release the GIL while they run native code; reading the input
matrix holds it. Calls on one solver run one at a time, so use one solver per
thread to solve in parallel.

## Save and reload

Save a reusable solver's operator and factor, then choose the solve backend
when loading. These methods also work without SciPy:

```python
solver.save("system.apxchol")
restored = apxchol.Solver.load("system.apxchol", backend="cpu")
result = restored.solve(b)
```

Saving requires `keep_factor=True`, the reusable solver's default. It creates
a new file and refuses to replace an existing one. The archive stores the
original operator and FP32 factor; RHSs, initial guesses and device workspace
are not saved. Loading adopts the factor without repeating factorization.

`backend="auto"` adopts on CPU; `backend="gpu"` uploads to a compatible GPU.
Storage and preparation options use the same defaults as construction. Pass
`lump_max_mass` again if the saved operator requires positive-entry lumping.

## Setup options

Pass these keywords to `factorize` or the one-shot `solve`:

| Keyword | Meaning |
|---|---|
| `backend` | `"auto"` chooses a compatible CPU/GPU route; `"cpu"` or `"gpu"` requests one explicitly |
| `num_threads` | CPU thread ceiling; `0` inherits the caller's OpenMP setting |
| `seed` | Sampling seed; `None` uses the native default |
| `partitioner` | `"block_greedy"`, `"priority_greedy"` or `"baumann_kyng"`; `None` uses the native default |
| `keep_factor` | Retain exportable arrays; defaults to `True` for `factorize`, `False` for one-shot `solve` |
| `factor_storage_dtype` | `np.float16`, `np.float32`, or `None` for the backend default; dtype names also work |

Read the chosen route from `solver.backend` and storage type from
`solver.factor_storage_dtype`. See [backend and thread selection](https://algoptgroup.github.io/apxchol/docs/usage.html)
for automatic selection and explicit-request behavior.

Storage defaults to FP32 on CPU/Metal and FP16 on CUDA. This affects
triangular-solve storage, not export or outer solve precision. Unsupported
FP16 requests fail; FP64 storage is unavailable. See [precision](https://algoptgroup.github.io/apxchol/docs/precision.html).

### Advanced options

These share the native defaults and can usually be left unset. GPU routes
support fewer construction choices.

| Keyword | Meaning |
|---|---|
| `degree_quantile` | Degree cutoff quantile; `0` selects the multiplier-based cutoff |
| `degree_multiplier` | Positive degree multiplier, used only when `degree_quantile=0` |
| `degree_tiebreak` | Prefer lower-degree vertices in selection conflicts |
| `exact_clique_max_degree` | Maximum degree for exact clique elimination; `0` disables it |
| `residual_peel` | Residual pivot order: `"natural"`, `"min_degree"` or `"bk_serial"` |
| `cpu_schedule` | CPU triangular-solve scheduling: `"auto"` or `"levels"` |
| `drop_tolerance` | Relative factor-entry threshold in `[0, 1]`; `0` disables dropping |
| `regularization` | Nonnegative diagonal fraction added during factor construction; `0` disables it |
| `lump_max_mass` | Allowed positive off-diagonal mass fraction in `[0, 1]`; `0` rejects such entries |
| `verbose` | Print setup diagnostics; off by default |

Unknown keywords raise `ValueError`. The [Python reference](https://algoptgroup.github.io/apxchol/docs/reference/python.html)
and [native option reference](https://algoptgroup.github.io/apxchol/docs/reference/cpp.html) describe the controls.

## Use with a SciPy solver

Use the built-in `solver.solve` for the standard PCG path. To use apxchol as a
preconditioner for another SciPy iterative method:

```python
from scipy.sparse.linalg import cg

M = solver.aslinearoperator()
x, info = cg(A, b, M=M, rtol=1e-8)
```

SciPy controls convergence. Use this when you need its algorithm or interface.
With a GPU factor, each application exchanges vectors with the host; the
built-in GPU solve keeps the iteration on the device.

## Export the factor

`factorize` retains exportable arrays by default. Exported factor values are
FP32, regardless of backend or internal storage:

```python
P = solver.P                          # P[original vertex] = elimination position
G = solver.chol()                     # lower CSC factor, including its diagonal
L, D = solver.L, solver.D             # unit lower CSC factor and diagonal
p = P.argsort()
# A[p][:, p] ≈ G @ G.T ≈ L @ diags(D) @ L.T
```

`chol()`, `L` and `D` return independent copies. For a Laplacian, they represent
the nonsingular subspace; see the [factor reference](https://algoptgroup.github.io/apxchol/docs/reference/python.html).

Use `keep_factor=False` to save the extra host factor copy when export is
unnecessary. Repeated solves, `P`, `factor_nnz` and `fill_ratio` remain available.

Matrix import is limited to signed 32-bit dimensions and stored-entry counts.
The native library's wide-index build options do not enlarge this binding.

## Build from source

From a repository checkout, install a development build and run its tests:

```bash
python -m pip install -e './python[test,scipy]'
python -m pytest python/tests -v
```

The `-e` option links the installed package to the local Python source.
Re-run the install command after changing C++ code. Local builds tune for the
build machine; see [build instructions](https://algoptgroup.github.io/apxchol/docs/building.html).

Set `CMAKE_ARGS="-DAPXCHOL_DEVICE=CUDA"` or `CMAKE_ARGS="-DAPXCHOL_DEVICE=METAL"`
for a GPU source build, with its native dependencies installed. Without build
isolation, install `pybind11 scikit-build-core` first. [License](https://github.com/AlgOptGroup/apxchol/blob/main/LICENSE).
