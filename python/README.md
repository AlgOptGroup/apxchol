# apxchol for Python

CPU approximate-Cholesky preconditioning and PCG for sparse Laplacian/SDDM
operators.

```bash
pip install apxchol
```

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
Storage: `vec_pool_aos`, `vec_pool`, `forward_star`, `vec`, `bstr`.
`keep_factor=True` is the reusable API default and retains an extra factor
copy (~8 bytes/nonzero in default builds). `False` disables factor export,
but preserves `P`, `factor_nnz`, `fill_ratio`; one-shot `solve` uses `False`.

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
Exports are float64 arrays carrying the default factor's fp32 precision.

The SciPy import uses signed 32-bit dimensions/nonzero counts; default factor
offsets are unsigned 32-bit. The core's wide-index options are separate.

## Source build

```bash
pip install -e python
pytest python/tests -v
```

The extension directly compiles `factorization.cpp`, `operator_class.cpp`
and `solve.cpp`. Local builds use `-O3 -march=native`; distributed wheels
are portable. Without build isolation, install `pybind11 scikit-build-core`
first. [License](../LICENSE).
