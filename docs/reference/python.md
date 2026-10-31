# Python reference

See the [Python guide](../../python/README.md) for installation, examples and backend support.

```{eval-rst}
.. autofunction:: apxchol.factorize

.. autofunction:: apxchol.solve

.. autofunction:: apxchol.as_operator

.. autoclass:: apxchol.Solver
   :members: from_csc, save, load, solve, apply, aslinearoperator, shape, sddm, backend, factor_storage_dtype, factor_nnz, lumped_offdiag, P, chol

.. autoclass:: apxchol.SolveResult
   :members: x, iters, residual, converged
   :undoc-members:
```
