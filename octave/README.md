# apxchol for Octave / MATLAB

CPU approximate-Cholesky preconditioning and PCG. Factor once, solve many:

```matlab
addpath('/path/to/apxchol/octave');
s = apxchol_solver(A);
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
original operator. Laplacian nullspaces are handled componentwise.
The binding exposes default settings; GPU and wide indices are unavailable.

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
ABI-incompatible. From `octave/`, this GCC/OpenMP command requires GCC ≥14:

```matlab
mex -R2018a apxchol_mex.cpp ../src/factorization.cpp ...
    ../src/operator_class.cpp ../src/solve.cpp ...
    -I../include -I../src -I/usr/include/eigen3 ...
    -DAPXCHOL_POOL_FP32 ...
    CXXFLAGS='$CXXFLAGS -std=c++23 -fopenmp -O3 -fPIC' ...
    LDFLAGS='$LDFLAGS -fopenmp' -lgomp
addpath(pwd); run('tests/test_apxchol.m')
```

MATLAB's selected compiler and C++ runtime must support the required language
features. See [historical Linux troubleshooting](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/octave/README.md#troubleshooting-on-recent-linux-distributions)
for previously encountered runtime issues. [License](../LICENSE).
