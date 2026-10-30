# Quick start

apxchol solves `A x = b` for a sparse graph Laplacian or SDDM matrix `A`.
It factors `A` once and reuses that factor for every right-hand side.

## Python

```{include} ../_readme.md
:start-after: <!-- quickstart -->
:end-before: <!-- /quickstart -->
```

Every result reports `converged`, checked against the original `A` and `b`.
For a graph adjacency matrix, convert it first with `apxchol.as_operator(Adj)`;
see [converting input](usage.md#converting-input-to-an-operator).

## Command line

```bash
apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
apxchol graph.mtx --convert-input --random-rhs --tol 1e-8
```

Matrices and right-hand sides are Matrix Market files. See the
[command-line guide](cli.md).

## C++

The [basic example](examples.md#basic-solves) shows one-shot, reusable and
batched solves. [C++ and CMake](cpp.md) explains how to link the library.

## Next steps

- [Using the solver](usage.md): supported matrices, CPU or GPU, convergence.
- [Batching and factor reuse](batching.md): many right-hand sides.
- [MATLAB and Octave](../octave/README.md).
