# apxchol for Python

CPU approximate-Cholesky preconditioning and PCG for sparse Laplacian/SDDM
operators.

```bash
pip install apxchol
```

NumPy and SciPy are required. Pass a SciPy sparse matrix or sparse array as the operator.

Wheels: Linux x86_64 (manylinux, x86-64-v2 CPU or newer) and macOS arm64
(macOS 15 or newer), CPython 3.10–3.14. Both platforms use Clang and bundle LLVM's OpenMP runtime
(license in `LICENSE.libomp.txt`); Homebrew is not needed to use the macOS
wheels. Importing packages that bundle another OpenMP runtime in the same
process can cause runtime conflicts.

```python
import apxchol
solver = apxchol.factorize(A)           # assembled scipy sparse operator
res = solver.solve(b, rtol=1e-8, maxiter=500)
res.x, res.iters, res.residual, res.converged
z = solver.apply(r)
M = solver.aspreconditioner()           # scipy.sparse.linalg.cg(..., M=M)
res = apxchol.solve(A, b)               # one-shot
```

For adjacency input, explicitly call `apxchol.laplacian(Adj)` to form `D - Adj`
with self-loops removed. Operator validation is shared with the C++ library.
Positive off-diagonal pairs are lumped onto the diagonal for preconditioning;
PCG and residual checks still use the original operator. `solver.lumped`
counts stored positive off-diagonal entries (two per symmetric pair). Laplacian nullspaces are handled componentwise.

Operators, right-hand sides, initial guesses, and preconditioner inputs must
be real. Complex dtypes raise `ValueError`, even when their imaginary parts
are zero.

## Options and reuse

`factorize` (alias `solver`) builds once; subsequent solves reuse the factor
and workspace. `OMP_NUM_THREADS` controls OpenMP parallelism. A solver cannot
serve concurrent `solve`/`apply` calls; use separate instances or serialize.

`solve` accepts `rtol` (default `1e-8`, overriding alias `tol`), `maxiter`,
`x0`, and `out`. The latter must be writable, C-contiguous float64 of length
`n`; it is returned as `res.x`. An already-converged guess takes zero iterations.

```python
solver = apxchol.factorize(A, seed=42, partitioner="block_greedy",
                           storage="vec_pool_aos", keep_factor=False)
```

Partitioners: `block_greedy`, `priority_greedy`, `baumann_kyng`.
Graph storage: `vec_pool_aos`, `vec`, `bstr`.
`keep_factor=True` is the reusable API default and retains an extra factor
copy (~8 bytes/nonzero in default builds). `False` disables factor export,
but preserves `P`, `factor_nnz`, `fill_ratio`; one-shot `solve` uses `False`.

`factor_storage_dtype=np.float16` or `np.float32` selects triangular-solve storage per solver; dtype objects and the strings `"float16"`/`"float32"` also work. Read the resolved dtype through `solver.factor_storage_dtype`. The default, `None`, respects `APXCHOL_SPTRSV_STORAGE=auto|float16|float32`, where `auto` means FP32 on the CPU. The legacy `APXCHOL_SPTRSV_FP16=0|1` is accepted when the named setting is unset or empty. Invalid settings raise an error; an explicit dtype overrides the environment without changing it.

FP16 stores scaled off-diagonals with FP32 diagonals and scales. It requires an x86 CPU with AVX/F16C; an unsupported request raises an error. CPU arithmetic remains FP64; constructed factors and exports remain FP32. FP64 factor storage is not supported. See [precision and storage](../docs/precision.md).

Advanced keywords: `degree_quantile`, `degree_multiplier`, `degree_tiebreak`,
`exact_clique_max_degree`, `residual_peel`, `stagnation_window`.
`degree_multiplier` applies only when `degree_quantile=0`; unknown keywords
raise `ValueError`. See [core defaults](../include/apxchol/solver/factor_options.h).

## Factor export

```python
solver = apxchol.factorize(A, keep_factor=True)
P = solver.P                           # P[original vertex] = elimination position
G = solver.chol()                      # lower CSC, including sqrt-diagonal
L, D = solver.L, solver.D              # unit lower CSC and diagonal
p = P.argsort()
# A[p][:, p] ≈ G @ G.T ≈ L @ scipy.sparse.diags(D) @ L.T
```

For a Laplacian with `k` components, this represents the rank-`n-k` subspace;
each component's last column contains a placeholder diagonal.
`fill_ratio = (2 * factor_nnz - n) / nnz(A)` counts the symmetric factor pattern.
`chol()`, `L` and `D` return float32 values, matching the C++ factor. `solve` and `apply` accept float32 inputs, but use FP64 CPU arithmetic and return float64 vectors. Use `.astype(np.float64)` if a consumer requires double-precision factor arrays.

Matrix import uses signed 32-bit dimensions/nonzero counts; default factor
offsets are unsigned 32-bit. The core's wide-index options are separate.

## Source build

```bash
pip install -e 'python[test]'
pytest python/tests -v
```

The extension directly compiles `factorization.cpp`, `operator_class.cpp`
and `solve.cpp`. Local builds tune for the build machine; distributed wheels
use the CPU baselines above. On macOS, install the [build dependencies](../README.md#build-and-run)
first, then use the same pip command above.
Without build isolation, install `pybind11 scikit-build-core` first. [License](../LICENSE).
