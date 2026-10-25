# apxchol for Octave / MATLAB

CPU approximate-Cholesky preconditioning and PCG. Factor once, solve many:

```matlab
addpath('/path/to/apxchol/octave');
s = apxchol_solver(A);
% Optional setup controls; num_threads is a ceiling.
s = apxchol_solver(A, struct('num_threads', 4, 'factor_storage', 'float32'));
res = s.solve(b);                 % x, iters, residual, converged
res = s.solve(b2, 1e-10, 1000);
z = s.apply(r);
x = pcg(A, b, 1e-8, 500, @(r) s.apply(r));
res = apxchol_solve(A, b);        % one-shot
```

Supply an assembled operator. For adjacency input, explicitly use
`apxchol_laplacian(Adj)` to form `D - Adj` with self-loops removed.
Validation is shared with C++; positive off-diagonal pairs are lumped onto
the diagonal for preconditioning, while PCG and residual checks retain the
original operator. A Laplacian RHS must sum to zero within each connected component.
Omitted solve arguments use the C++ defaults (relative tolerance 1e-8 and
200 iterations). Tolerance must be finite and positive; the iteration limit
must be a nonnegative integer. `[]` selects either default.

The optional setup struct accepts `num_threads`, `seed`, `factor_storage`
(`auto`, `float16`, `float32`), `cpu_schedule` (`auto`, `levels`),
`drop_tolerance` (nonnegative; 0 disables), `regularization` (nonnegative
diagonal fraction; 0 disables), `lump_max_mass` (fraction in [0, 1]; 0
rejects positive off-diagonals), and
`verbose` (logical). Unspecified fields use C++ defaults. GPU and wide
indices are unavailable in this binding. Solve and apply accept one RHS vector
at a time; warm starts and factor export are not exposed.

## Octave build

Requires `mkoctfile` (Octave development tools), Eigen3 and an OpenMP-capable
C++23 compiler (GCC ≥14 or equivalent). Use matching C/C++ compiler families
and one OpenMP runtime.

```bash
cd octave
./build.sh                       # optionally: CC=gcc-14 CXX=g++-14 ./build.sh
octave --no-gui --eval "addpath(pwd); run('tests/test_apxchol.m')"
```

The extension directly compiles the three core CPU sources; it does not
require a root CMake build.

## MATLAB build

The wrappers/tests are shared, but MATLAB and Octave MEX binaries are
ABI-incompatible. Linux x86-64 builds are tested with MATLAB R2026b and GCC 13.
Install OpenMP and Eigen, then run:

```bash
matlab -batch "run('/path/to/apxchol/octave/build_and_test.m')"
```

MATLAB's selected compiler and C++ runtime must support the required language
features. See [historical Linux troubleshooting](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/octave/README.md#troubleshooting-on-recent-linux-distributions)
for previously encountered runtime issues. [License](../LICENSE).
