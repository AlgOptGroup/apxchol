# apxchol (Octave, CPU)

Approximate-Cholesky preconditioner for graph-Laplacian / SDDM linear systems,
as an Octave MEX extension. Same API concept as the Python package: build the
factor once, solve many right-hand sides.

```matlab
addpath('/path/to/apxchol/octave');

s   = apxchol_solver(A);          % A: sparse SPD Laplacian or SDDM; factor built once
res = s.solve(b);                 % res.x, res.iters, res.residual, res.converged
res = s.solve(b2, 1e-10, 1000);   % per-solve tol / maxiter

z = s.apply(r);                   % one preconditioner application, z = M\r
x = pcg(A, b, 1e-8, 500, @(r) s.apply(r));   % drop into Octave's pcg as M

res = apxchol_solve(A, b);        % one-shot convenience
```

Laplacian vs SDDM is auto-detected (singular Laplacians get the rank-(n−1)
factor with null-space centering). Defaults only — no tuning knobs exposed.

## Build

Needs `mkoctfile` (Octave dev tools), Eigen3 headers, and an OpenMP-capable
**g++ >= 14** (the core uses C++23 "deducing this"). The MEX compiles the two
core library TUs directly — no dependency on the root CMake build:

```bash
cd octave && ./build.sh
```

If the default compiler is older, point `mkoctfile` at a newer one via `CXX`
(`build.sh` sets `CXXFLAGS` but leaves the compiler to `mkoctfile`):

```bash
cd octave && CXX=g++-14 ./build.sh
```

Test: `octave --no-gui --eval "addpath(pwd); run('tests/test_apxchol.m')"`

## MATLAB

The `.m` wrappers and `tests/test_apxchol.m` run unchanged on MATLAB; the MEX
source is plain `mex.h` API. Only the binary differs (`.mexa64` for MATLAB vs
`.mex` for Octave — ABI-incompatible), so rebuild with MATLAB's `mex`. The core
needs **C++23** ("deducing this", `std::ranges::iota`), so GCC ≥ 14 is required —
newer than MATLAB's officially "supported" GCC 12, which it accepts with a warning:

```matlab
mex -R2018a apxchol_mex.cpp ../src/factorization.cpp ../src/solve.cpp ...
    -I../include -I/usr/include/eigen3 ...
    -DAPXCHOL_POOL_FP32 ...
    CXXFLAGS='$CXXFLAGS -std=c++23 -fopenmp -O3 -fPIC' ...
    LDFLAGS='$LDFLAGS -fopenmp' -lgomp
addpath(pwd); run('tests/test_apxchol.m')
```

Validated on **MATLAB R2026a**, GCC 14, 6/6 tests passing.

### Troubleshooting on recent Linux distributions

1. **MATLAB won't launch**: its FlexLM licensing can segfault (`lc_new_job`)
   against a very new system glibc. Run MATLAB inside MathWorks' dependency
   container (`mathworks/matlab-deps:<release>`, Ubuntu-based, older glibc),
   mounting the MATLAB install read-only. Use `--network=host` so a node-locked
   license still sees the host's MAC address.

2. **`GLIBCXX` version error when loading the MEX**: a GCC-14 build needs a newer
   `libstdc++` than the one MATLAB bundles. Preload the system library (it is
   ABI-backward-compatible, so MATLAB still works):

   ```bash
   LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)" matlab -batch "..."
   ```

3. **`mkoctfile` not found**: it ships in Octave's development package
   (`octave-dev` on Debian/Ubuntu, `octave-devel` elsewhere), not in the base
   Octave package.

CPU only; GPU and 64-bit indices are not exposed in this version.

## License

Same terms as the rest of the repository — see the root
[LICENSE](../LICENSE).
