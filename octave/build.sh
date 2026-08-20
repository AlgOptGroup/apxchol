#!/usr/bin/env bash
# Build the apxchol Octave MEX extension (CPU only).
#
# Compiles the core library TUs straight into the MEX (same approach as the
# Python package: no dependency on the root CMake build). operator_class.cpp
# supplies the operator-contract assertion and the M-matrix lumping, shared
# verbatim with the CLI and the Python binding. Needs: mkoctfile
# (octave-devel), Eigen3 headers, OpenMP-capable g++.
set -euo pipefail
cd "$(dirname "$0")"

EIGEN_FLAGS=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

# CXXFLAGS overrides mkoctfile's defaults entirely -> restate optimization.
# fp32 pool defines match the library defaults (root CMakeLists defaults).
CXXFLAGS="-O3 -march=native -std=gnu++23 -fopenmp -fPIC -Wall" \
mkoctfile --mex \
    apxchol_mex.cpp ../src/factorization.cpp ../src/operator_class.cpp ../src/solve.cpp \
    -I../include -I../src ${EIGEN_FLAGS} \
    -DAPXCHOL_POOL_FP32 \
    -lgomp \
    -o apxchol_mex

echo "built: $(ls -la apxchol_mex.mex | awk '{print $9, "("$5" bytes)"}')"
echo "use:   addpath $(pwd)   then   s = apxchol_solver(A); res = s.solve(b);"
