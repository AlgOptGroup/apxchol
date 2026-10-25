# Python reference

See the [Python guide](../../python/README.md) for installation, examples and backend support.

```{eval-rst}
.. autofunction:: apxchol.factorize

.. autofunction:: apxchol.solve

.. autofunction:: apxchol.laplacian

.. autoclass:: apxchol.Solver
   :members: solve, apply, aslinearoperator, shape, sddm, backend, factor_storage_dtype, factor_nnz, fill_ratio, lumped, P, chol, L, D

.. autoclass:: apxchol.SolveResult
   :members: x, iters, residual, converged
   :undoc-members:
```
