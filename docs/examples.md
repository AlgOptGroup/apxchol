# Examples

These C++ programs are in the repository's
[`examples/`](https://github.com/AlgOptGroup/apxchol/tree/main/examples)
directory. A source build compiles them, and `ctest` runs them; each exits
with a nonzero status if its solve misses the residual it checks.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
./build/examples/example_basic_solve
```

All three use a small grid Laplacian from
[`example_grid.h`](https://github.com/AlgOptGroup/apxchol/blob/main/examples/example_grid.h).
For Python and MATLAB/Octave, see the [Python guide](../python/README.md) and
the [MATLAB/Octave guide](../octave/README.md).

## Basic solves

One-shot, reusable and batched solves, and one preconditioner application.

```{literalinclude} ../examples/basic_solve.cpp
:language: cpp
```

## Custom elimination rule

A user-defined eliminator that adds the exact Schur-complement clique instead
of sampling it. See [custom clique sampling](extending.md#custom-clique-sampling).

```{literalinclude} ../examples/custom_eliminator.cpp
:language: cpp
```

## Custom elimination order

A partitioner that eliminates vertices in a user-supplied priority order. See
[custom selection and ordering](extending.md#custom-selection-and-ordering).

```{literalinclude} ../examples/custom_order.cpp
:language: cpp
```
