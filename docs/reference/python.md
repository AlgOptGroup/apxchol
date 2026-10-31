# Python reference

See the [Python guide](../../python/README.md) for installation, examples and backend support.

```{eval-rst}
.. autofunction:: apxchol.factorize

.. autofunction:: apxchol.solve

.. autofunction:: apxchol.as_operator

.. autoclass:: apxchol.Solver
   :members: from_csc, save, load, solve, apply, aslinearoperator, shape, sddm, backend, setup_seconds, factor_storage, factor_nnz, lumped_offdiag, P, chol

.. autoclass:: apxchol.SolveResult
   :members: x, iterations, residual, converged, solve_seconds
   :undoc-members:
```
