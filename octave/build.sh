#!/usr/bin/env bash
# Build the apxchol Octave MEX extension (CPU only).
#
# Compiles the core library TUs straight into the MEX (same approach as the
# Python package: no dependency on the root CMake build). operator_class.cpp
# supplies the operator-contract assertion and the M-matrix lumping, shared
# verbatim with the CLI and the Python binding. Needs: mkoctfile
# (octave-devel), Eigen3 headers, and an OpenMP-capable g++ or clang++.
set -euo pipefail
cd "$(dirname "$0")"

# mkoctfile builds our TUs with $CXX, but it also compiles its own little
# SOVERSION stub with $CC, under -flto=auto. A gcc-compiled LTO object fed into
# a clang link silently drops __octave_mex_soversion__ and the resulting .mex
# then fails to load — so $CC has to come from the same family as $CXX.
# The OpenMP runtime follows the same rule from the other side: libomp and
# libgomp loaded into one Octave process SEGFAULT it, so link exactly one, the
# one that belongs to $CXX.
compiler_family() {
    if "$1" --version 2>/dev/null | head -n1 | grep -qi clang; then
        echo clang
    else
        echo gcc
    fi
}

CXX="${CXX:-g++}"
CXX_FAMILY=$(compiler_family "$CXX")
if [ "$CXX_FAMILY" = clang ]; then
    CC="${CC:-clang}"
    OMP_LIB="-lomp"
else
    CC="${CC:-gcc}"
    OMP_LIB="-lgomp"
fi
CC_FAMILY=$(compiler_family "$CC")
if [ "$CC_FAMILY" != "$CXX_FAMILY" ]; then
    echo "error: CC=$CC ($CC_FAMILY) and CXX=$CXX ($CXX_FAMILY) are different compilers." >&2
    echo "       mkoctfile compiles its SOVERSION stub with \$CC under -flto=auto; a" >&2
    echo "       cross-family object drops __octave_mex_soversion__ and the .mex will" >&2
    echo "       not load. Set both, or neither." >&2
    exit 1
fi
export CC CXX

# -march=native is the x86 spelling. On aarch64 clang accepts it and silently
# targets generic, so ask for the ISA's own spelling first and fall back; a
# toolchain that takes neither gets no tuning flag rather than a hard failure.
# Same rule as cmake/apxchol_native_arch.cmake.
case "$(uname -m)" in
    aarch64|arm64) ARCH_CANDIDATES="-mcpu=native -march=native" ;;
    *)             ARCH_CANDIDATES="-march=native -mcpu=native" ;;
esac
ARCH_FLAG=""
for _flag in $ARCH_CANDIDATES; do
    if printf 'int main(){return 0;}\n' | "$CXX" -x c++ "$_flag" -c -o /dev/null - 2>/dev/null; then
        ARCH_FLAG="$_flag"
        break
    fi
done
echo "toolchain: CXX=$CXX CC=$CC ($CXX_FAMILY), OpenMP=$OMP_LIB, tuning=${ARCH_FLAG:-none}"

EIGEN_FLAGS=$(pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")

# CXXFLAGS overrides mkoctfile's defaults entirely -> restate optimization.
# fp32 pool defines match the library defaults (root CMakeLists defaults).
CXXFLAGS="-O3 ${ARCH_FLAG} -std=gnu++23 -fopenmp -fPIC -Wall" \
mkoctfile --mex \
    apxchol_mex.cpp ../src/factorization.cpp ../src/operator_class.cpp ../src/solve.cpp \
    -I../include -I../src ${EIGEN_FLAGS} \
    -DAPXCHOL_POOL_FP32 \
    ${OMP_LIB} \
    -o apxchol_mex

echo "built: $(ls -la apxchol_mex.mex | awk '{print $9, "("$5" bytes)"}')"
echo "use:   addpath $(pwd)   then   s = apxchol_solver(A); res = s.solve(b);"
