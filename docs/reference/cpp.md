# C++ reference

Include `apxchol.h` and link `apxchol::core`. The [overview](../index.md)
shows a complete build and solve; [extension contracts](../extending.md)
cover custom factor construction. This page documents the reusable solver
and its main controls within the [public API scope](../api.md).

## Reusable solver

```{doxygenclass} apxchol::solver
:members: solver, solve, apply, factor, rows, backend, factor_storage
:undoc-members:
```

## Options

```{doxygenstruct} apxchol::solve_options
:members: tol, max_iter, keep_factor_values, factor_opts, backend, num_threads, collect_timings
:undoc-members:
```

```{doxygenstruct} apxchol::factor_options
:members: seed, regularization, lump_max_mass, factor_storage, cpu_schedule, drop_tolerance, partition, sampler, is_select
:undoc-members:
```

```{doxygenstruct} apxchol::partition_options
:members: degree_quantile, degree_multiplier, degree_tiebreak
```

```{doxygenenum} apxchol::solve_backend
```

```{doxygenenum} apxchol::factor_storage_type
```

```{doxygenenum} apxchol::cpu_sptrsv_schedule
```

```{doxygenenum} apxchol::clique_sampler
```

## Results

```{doxygenstruct} apxchol::solve_result
:members: x, iterations, residual, converged, lumped_offdiag, timings, backend
:undoc-members:
```

```{doxygenstruct} apxchol::batch_solve_result
:members: columns, timings
:undoc-members:
```

## One-shot solve and availability

```{doxygenfunction} apxchol::solve
```

```{doxygenfunction} apxchol::backend_available
```
