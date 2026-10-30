# apxchol for MATLAB and Octave

Both interfaces call the same compiled C++ CPU solver. Build the factor once
and reuse it for several right-hand sides:

```matlab
addpath('/path/to/apxchol/octave');
s = apxchol_solver(A);
res = s.solve(b);
assert(res.converged);
x = res.x;

res2 = s.solve(b2, 1e-10, 1000);   % tolerance and iteration budget for this call
```

For one solve, use `apxchol_solve(A, b)`. Results contain `x`, `iters`,
`residual` and `converged`. The built-in solver always checks the returned
solution against the original matrix and RHS.

Pass an `n`-by-`k` matrix to solve or apply the preconditioner to `k` RHSs
through the native batched path. Results have `k` columns; `iters`, `residual`
and `converged` are `1`-by-`k` arrays. A row-vector RHS is still accepted and
returns a column, except that a one-row system treats each column as a RHS.

GPU routes, factor export and wide indices are not exposed. See the shared
[binding capabilities](../docs/usage.md#binding-capabilities).

## Inputs and solve controls

Pass the sparse matrix `A` in `A * x = b`. For graph adjacency input, first
use `A = apxchol_as_operator(Adj)`: it returns a valid operator unchanged and
otherwise forms the graph Laplacian. The [input guide](../docs/usage.md)
explains supported matrices, input conversion and Laplacian RHS requirements.

`solve(b, rtol, maxiter, x0)` takes a positive, finite relative tolerance and a
nonnegative integer iteration budget. Omitted arguments or `[]` use native
defaults, shown in the [native reference](../docs/reference/cpp.md#solve-defaults):

```matlab
res = s.solve(b, [], 500);         % default tolerance, at most 500 iterations
warm = s.solve(B, 1e-8, 200, X0);  % optional initial guess, same shape as B
```

Omit `x0` or pass `[]` to start from zero. The one-shot form accepts it as
`apxchol_solve(A, B, rtol, maxiter, options, X0)`.

## Setup options

Pass an optional scalar struct to the constructor:

```matlab
s = apxchol_solver(A, struct('num_threads', 4, 'factor_storage', 'float32'));
```

Unspecified fields use the C++ defaults. Unknown fields are rejected.

| Field | Meaning |
|---|---|
| `num_threads` | CPU thread ceiling; `0` inherits the caller's OpenMP setting |
| `seed` | Sampling seed |
| `factor_storage` | `auto`, `float16` or `float32`; affects triangular-solve storage |
| `cpu_schedule` | CPU triangular-solve scheduling: `auto` or `levels` |
| `drop_tolerance` | Relative factor-entry threshold in `[0, 1]`; `0` disables dropping |
| `regularization` | Minimum diagonal excess of the preconditioner, as a fraction of each diagonal after lumping; excess below 1e-12 of the diagonal counts as zero first; `A` is not modified; `0` disables it |
| `lump_max_mass` | Allowed positive off-diagonal mass fraction in `[0, 1]`; `0` rejects such entries |
| `keep_factor` | Retain canonical FP32 factor values for saving; off by default |
| `verbose` | Logical flag for setup diagnostics; off by default |

See [precision](../docs/precision.md) and the [native option reference](../docs/reference/cpp.md)
for storage behavior and advanced controls.

## Save and load a solver

Enable factor retention when constructing a solver, then save its original
matrix and factor together:

```matlab
s = apxchol_solver(A, struct('keep_factor', true));
s.save('system.apxchol');
restored = apxchol_solver.load('system.apxchol');
res = restored.solve(B);  % scalar/matrix RHS and warm starts work as usual
```

The path must be a character row vector. Saving creates a new file and rejects
an existing destination. Loading accepts the same optional options struct as
the constructor, including `factor_storage` and `keep_factor`; enable retention
when loading if you want to save again. Setup defaults are unchanged.

Archives use the shared [portable file format](../docs/serialization.md).
Loading prepares the saved factor without factorizing the matrix again.

## Use another iterative solver

`apply(r)` computes one preconditioner application. Use it when integrating
with another iterative solver, such as MATLAB/Octave's `pcg`:

```matlab
z = s.apply(r);
[x, flag] = pcg(A, b, 1e-8, 500, @(r) s.apply(r));
```

That outer solver controls convergence and reports its own result. Use
`s.solve` for the built-in PCG path.

## Build for Octave

Requires CMake 3.20+, `mkoctfile` (Octave development tools), zlib and an
OpenMP-capable C++23 compiler (GCC 14 or equivalent). Use matching C/C++ compiler families
and one OpenMP runtime.

```bash
cd octave
./build.sh                       # optionally: CC=gcc-14 CXX=g++-14 ./build.sh
octave --no-gui --eval "addpath(pwd); run('tests/test_apxchol.m')"
```

The extension links the shared native CMake target. Eigen3 is found or fetched
by the native build helpers; zlib must be installed. A root build is not required.
`CMAKE_BUILD_PARALLEL_LEVEL` sets the build worker count (default: 2).

## Build for MATLAB

MATLAB and Octave share the wrapper source and tests, but their MEX binaries
are not interchangeable. Linux x86-64 builds are tested with MATLAB R2026b
and GCC 13. Install CMake, OpenMP and zlib, then run:

```bash
matlab -batch "run('/path/to/apxchol/octave/build_and_test.m')"
```

Use a C++ runtime compatible with MATLAB's selected compiler. [License](../LICENSE).
